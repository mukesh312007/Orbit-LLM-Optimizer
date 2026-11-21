"""Evidence-labelled routing for one local request at a time."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import psutil

from core.settings import ROOT, settings
from database.telemetry import mirror_live_selection
from optimizer.profile_store import historical_profiles

SELECTION_HISTORY = ROOT / "logs" / "runtime_selection_history.jsonl"
CALIBRATION_HISTORY = ROOT / "logs" / "model_calibration.jsonl"


def classify_task(prompt: str) -> str:
    text = prompt.lower()
    if re.search(r"\b(sql|text[- ]to[- ]sql|select query)\b", text) or re.search(r"\b(?:write|generate|formulate)\b.{0,40}\b(?:query|database schema)\b", text) or re.search(r"\b(?:database|db)\b.{0,30}\b(?:query|select|table|row|record|list|find|count)\b", text):
        return "spider_text_to_sql"
    if re.search(r"\b(json|yaml|xml|structured output|schema|table|csv)\b", text):
        return "structured"
    if re.search(r"\b(arithmetic|calculate|calculation|what is)\b", text) and re.search(r"\d\s*[+\-*/%]\s*\d", text):
        return "arithmetic"
    if re.search(r"\b(code|python|function|bug|algorithm|script|program)\b", text):
        return "coding"
    if re.search(r"\b(calculate|solve|math|equation|reason|proof|why)\b", text):
        return "reasoning"
    if re.search(r"\b(summarize|summary|shorten)\b", text):
        return "summarization"
    return "general_knowledge"


def _measured_memory(model: dict[str, Any]) -> float | None:
    path = model.get("path")
    if not path:
        return None
    historical_name = path.split("/")[-1]
    values = [profile["ram_mb"] for profile in historical_profiles() if profile["model_id"] == historical_name and profile.get("ram_mb")]
    return min(values) if values else None


def _estimated_memory(model: dict[str, Any]) -> float:
    """Conservative feasibility estimate from the on-disk GGUF size plus 300 MiB runtime headroom."""
    size_mb = (ROOT / model["path"]).stat().st_size / 1024**2
    return size_mb * 1.3 + 300


def _active_orbit_runtime() -> tuple[str, float] | None:
    """Return the registered, currently running Orbit model and its RSS."""
    state_path = ROOT / "runtime" / "active_runtime.json"
    try:
        # Start-Agent.ps1 can create a UTF-8 BOM on Windows PowerShell.
        # ``utf-8-sig`` accepts that form as well as normal UTF-8 state files.
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        pid = state.get("pid")
        model_id = state.get("model_id")
        if not isinstance(pid, int) or not isinstance(model_id, str):
            return None
        process = psutil.Process(pid)
        if "llama-server" not in process.name().lower():
            return None
        return model_id, process.memory_info().rss / 1024**2
    except (FileNotFoundError, json.JSONDecodeError, psutil.Error):
        return None


def _latest_calibrations() -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    try:
        with CALIBRATION_HISTORY.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    record = json.loads(line)
                    latest[record["model_id"]] = record
    except FileNotFoundError:
        pass
    return latest


def recommend_runtime(
    prompt: str,
    models: list[dict[str, Any]],
    safety_reserve_mb: float | None = None,
) -> dict[str, Any]:
    task_type = classify_task(prompt)
    physical_available_mb = psutil.virtual_memory().available / 1024**2
    reserve_mb = (
        safety_reserve_mb if safety_reserve_mb is not None
        else float(getattr(settings, "safety_reserve_mb", 512.0))
    )
    # Switching models terminates Orbit's recorded llama-server first. Its RSS is
    # therefore reclaimable, not concurrent memory required by the next candidate.
    active_runtime = _active_orbit_runtime()
    reclaimable_runtime_mb = active_runtime[1] if active_runtime else 0.0
    available_mb = physical_available_mb + reclaimable_runtime_mb
    calibrations = _latest_calibrations()
    candidates: list[dict[str, Any]] = []
    for model in models:
        if not model.get("available_locally") or "llama.cpp" not in model.get("backends", []):
            continue
        measured_memory = _measured_memory(model)
        required_mb = measured_memory or _estimated_memory(model)
        candidates.append({
            "model": model,
            "required_mb": required_mb,
            "memory_source": "measured historical peak RSS" if measured_memory else "estimated from GGUF file size",
            "calibration": calibrations.get(model["model_id"]),
            "feasible": required_mb + reserve_mb <= available_mb,
        })
    feasible = [item for item in candidates if item["feasible"]]
    if not feasible:
        fallback = min(candidates, key=lambda item: item["required_mb"], default=None)
        resident = next(
            (
                item for item in candidates
                if active_runtime and item["model"]["model_id"] == active_runtime[0]
            ),
            None,
        )
        if resident:
            model = resident["model"]
            configuration = {
                "configuration_id": f"{model['model_id']}:t4:c1024:cpu",
                "backend": "llama.cpp", "threads": 4, "context_size": 1024,
                "gpu_layers": 0, "cache": "server-default",
            }
            return {
                "selected": resident, "configuration": configuration,
                "task_type": task_type, "available_mb": available_mb,
                "physical_available_mb": physical_available_mb,
                "reclaimable_runtime_mb": reclaimable_runtime_mb,
                "candidates": candidates, "fallback": fallback,
                "resident_fallback": True,
                "reason": (
                    "Memory is constrained, so Orbit is using the already-loaded local "
                    "model instead of unloading it to compare additional models."
                ),
            }
        reserve_str = f"{reserve_mb / 1024:.1f} GiB" if reserve_mb >= 1024 else f"{reserve_mb:.0f} MiB"
        return {
            "selected": None,
            "task_type": task_type,
            "available_mb": available_mb,
            "physical_available_mb": physical_available_mb,
            "reclaimable_runtime_mb": reclaimable_runtime_mb,
            "candidates": candidates,
            "reason": f"No registered model fits while retaining the {reserve_str} RAM safety reserve.",
            "fallback": fallback,
        }
    calibrated = [item for item in feasible if item["calibration"]]
    if calibrated:
        best_quality = max(item["calibration"]["quality_score"] for item in calibrated)
        close_quality = [
            item for item in calibrated
            if best_quality - item["calibration"]["quality_score"] <= 0.08
        ]
        if task_type in {"coding", "reasoning"} and close_quality:
            selected = max(close_quality, key=lambda item: item["model"].get("parameter_count_b", 0.0))
            reason = "For coding/reasoning, selected the largest model within 8 quality points of the measured calibration leader."
        else:
            selected = max(calibrated, key=lambda item: (item["calibration"]["quality_score"], -item["calibration"]["mean_latency_seconds"]))
            reason = "For general/summarization work, selected the calibration leader by measured quality, then mean latency."
    else:
        selected = max(feasible, key=lambda item: item["model"].get("parameter_count_b", 0.0))
        reason = "Largest feasible Qwen model selected for response quality; quality is a parameter-count proxy until task calibration is recorded."
    model = selected["model"]
    config = {
        "configuration_id": f"{model['model_id']}:t4:c1024:cpu",
        "backend": "llama.cpp",
        "threads": 4,
        "context_size": 1024,
        "gpu_layers": 0,
        "cache": "server-default",
    }
    return {
        "selected": selected,
        "configuration": config,
        "task_type": task_type,
        "available_mb": available_mb,
        "physical_available_mb": physical_available_mb,
        "reclaimable_runtime_mb": reclaimable_runtime_mb,
        "candidates": candidates,
        "reason": reason,
    }


def record_selection(decision: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, Any] | None:
    selected = decision.get("selected")
    if not selected:
        return None
    persisted_telemetry = dict(telemetry)
    if not settings.retain_live_response_text:
        persisted_telemetry["answer"] = None
        persisted_telemetry["response_text_retained"] = False
    else:
        persisted_telemetry["response_text_retained"] = True
    row = {
        "model_id": selected["model"]["model_id"],
        "configuration_id": decision["configuration"]["configuration_id"],
        "task_type": decision["task_type"],
        "configuration": decision["configuration"],
        "telemetry": persisted_telemetry,
    }
    with SELECTION_HISTORY.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
    return mirror_live_selection(decision, persisted_telemetry)


def selection_caption(decision: dict[str, Any]) -> str:
    selected = decision.get("selected")
    if not selected:
        return f"No safe model selected - available RAM {decision['available_mb']:.0f} MiB"
    model = selected["model"]
    config = decision["configuration"]
    if decision.get("resident_fallback"):
        return (
            f"Selected **{model['model_id']}** - llama.cpp - {config['threads']} CPU threads - "
            f"context {config['context_size']} - CPU-only. Low-memory resident mode: "
            f"{decision['reason']}"
        )
    if decision.get("live_comparison"):
        live = "; ".join(
            f"{entry['candidate']['model']['model_id']} draft: {entry['latency_seconds']:.2f}s"
            for entry in decision["live_comparison"]
        )
        return (
            f"Selected **{model['model_id']}** - llama.cpp - {config['threads']} CPU threads - "
            f"context {config['context_size']} - CPU-only. On-the-spot comparison: {live}. "
            f"{decision['reason']}"
        )
    comparison = []
    for candidate in decision.get("candidates", []):
        calibration = candidate.get("calibration")
        if calibration:
            comparison.append(
                f"{candidate['model']['model_id']}: quality {calibration['quality_score']:.3f}, "
                f"latency {calibration['mean_latency_seconds']:.2f}s"
            )
    return (
        f"Selected **{model['model_id']}** - llama.cpp - {config['threads']} CPU threads - "
        f"context {config['context_size']} - CPU-only - {selected['memory_source']}. "
        f"Selection basis: {decision.get('reason', 'estimated quality; no calibration record is available.')} "
        f"Compared: {'; '.join(comparison) if comparison else 'no calibrated candidates yet.'}"
    )
