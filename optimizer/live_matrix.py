"""RAM-safe expansion of the full local runtime configuration matrix."""
from __future__ import annotations

import json
import random
from typing import Any

import psutil

from core.settings import ROOT, settings
from optimizer.configuration_catalog import benchmark_configurations

SAFETY_RESERVE_MB = float(getattr(settings, "safety_reserve_mb", 512.0))


def _reclaimable_managed_runtime_mb() -> float:
    """The recorded Orbit server is stopped before sequential model switching."""
    state_path = ROOT / "runtime" / "active_runtime.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        process = psutil.Process(int(state["pid"]))
        if "llama-server" in process.name().lower():
            return process.memory_info().rss / 1024**2
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError, psutil.Error):
        pass
    return 0.0


def safe_available_memory_mb() -> float:
    """Available RAM plus only Orbit's safely reclaimable managed server RSS."""
    return psutil.virtual_memory().available / 1024**2 + _reclaimable_managed_runtime_mb()


def required_memory_mb(model: dict[str, Any], configuration: dict[str, Any]) -> float:
    """Conservative pre-launch admission estimate; benchmark RSS later replaces it."""
    model_size_mb = (ROOT / model["path"]).stat().st_size / 1024**2
    return model_size_mb * 1.3 + 300 + float(configuration["context_size"]) * 0.15


def configuration_is_safe(
    model: dict[str, Any], configuration: dict[str, Any], available_mb: float | None = None,
    safety_reserve_mb: float | None = None,
) -> tuple[bool, str, float]:
    """Check one explicit configuration before a runtime switch."""
    available = safe_available_memory_mb() if available_mb is None else available_mb
    reserve = float(getattr(settings, "safety_reserve_mb", SAFETY_RESERVE_MB)) if safety_reserve_mb is None else safety_reserve_mb
    required = required_memory_mb(model, configuration)
    safe = required + reserve <= available
    return safe, (
        f"estimated {required:.0f} MiB plus {reserve:.0f} MiB reserve; available {available:.0f} MiB"
    ), required


def full_live_candidates(
    models: list[dict[str, Any]],
    available_mb: float | None = None,
    safety_reserve_mb: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Expand all local configurations, retaining only candidates safe in isolation."""
    reserve = safety_reserve_mb if safety_reserve_mb is not None else float(getattr(settings, "safety_reserve_mb", SAFETY_RESERVE_MB))
    available = (
        available_mb if available_mb is not None
        else safe_available_memory_mb()
    )
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for model in models:
        if not model.get("available_locally") or "llama.cpp" not in model.get("backends", []):
            continue
        for configuration in benchmark_configurations(model):
            required = required_memory_mb(model, configuration)
            entry = {"model": model, "configuration": configuration, "required_mb": required}
            if required + reserve <= available:
                candidates.append(entry)
            else:
                skipped.append(entry | {"reason": f"RAM safety skip: estimated {required:.0f} MiB plus {reserve:.0f} MiB reserve; available {available:.0f} MiB"})
    return candidates, skipped


def stratified_sample(candidates: list[dict[str, Any]], n: int, seed: int = 42) -> list[dict[str, Any]]:
    """Return a deterministic sample that covers models before extra variants.

    Candidate coverage is the first stratum: a bounded campaign must not use
    all of its slots on configuration variants of the first model. Within each
    model, cache/compression/retrieval groups are then round-robined. If
    ``n`` is at least the candidate count, the result is a reproducibly
    shuffled copy of every candidate.
    """
    rng = random.Random(seed)
    if n >= len(candidates):
        shuffled = list(candidates)
        rng.shuffle(shuffled)
        return shuffled

    # Group by model first, then by configuration properties.  This preserves
    # configuration diversity while making a small subset representative of
    # the available local model inventory.
    model_groups: dict[str, dict[tuple[str, str, bool], list[dict[str, Any]]]] = {}
    for entry in candidates:
        cfg = entry["configuration"]
        model_id = str(entry["model"]["model_id"])
        key = (
            cfg.get("cache", ""),
            cfg.get("prompt_compression", cfg.get("compression", "")),
            bool(cfg.get("retrieval", {}).get("enabled", False)),
        )
        model_groups.setdefault(model_id, {}).setdefault(key, []).append(entry)

    # Each model keeps its own shuffled configuration groups.  Model ids are
    # sorted for reproducibility; randomisation happens inside each group.
    per_model_groups: list[list[list[dict[str, Any]]]] = []
    for model_id in sorted(model_groups):
        groups: list[list[dict[str, Any]]] = []
        for key in sorted(model_groups[model_id]):
            bucket = list(model_groups[model_id][key])
            rng.shuffle(bucket)
            groups.append(bucket)
        rng.shuffle(groups)
        per_model_groups.append(groups)

    # Round-robin models, then rotate through their configuration groups.
    result: list[dict[str, Any]] = []
    offsets = [[0] * len(groups) for groups in per_model_groups]
    group_offsets = [0] * len(per_model_groups)
    while len(result) < n:
        # Advance to the next group that still has candidates.
        made_progress = False
        for model_index, groups in enumerate(per_model_groups):
            if len(result) >= n:
                break
            for _ in range(len(groups)):
                group_index = group_offsets[model_index] % len(groups)
                group_offsets[model_index] += 1
                if offsets[model_index][group_index] < len(groups[group_index]):
                    result.append(groups[group_index][offsets[model_index][group_index]])
                    offsets[model_index][group_index] += 1
                    made_progress = True
                    break
        if not made_progress:
            break
    return result
