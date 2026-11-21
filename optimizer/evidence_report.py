"""Local, read-only evidence views for the dashboard and API."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.dataset_loader import PROCESSED
from core.settings import ROOT, settings
from hardware.energy_capability import probe
from optimizer.calibrated_profiles import latest_profiles
from optimizer.pareto_optimizer import pareto_frontier

SELECTION_HISTORY = ROOT / "logs" / "runtime_selection_history.jsonl"
REQUIRED_GENERAL_DATASETS = ("gsm8k", "mmlu", "ifeval")


def dataset_readiness() -> list[dict[str, Any]]:
    """Report only local prepared datasets; never download or inspect remote data."""
    rows: list[dict[str, Any]] = []
    for dataset in REQUIRED_GENERAL_DATASETS:
        data_path = PROCESSED / f"{dataset}.jsonl"
        manifest_path = PROCESSED / f"{dataset}.manifest.json"
        samples = 0
        if data_path.is_file():
            with data_path.open(encoding="utf-8") as handle:
                samples = sum(1 for line in handle if line.strip())
        try:
            display_path = str(data_path.relative_to(ROOT))
        except ValueError:
            display_path = str(data_path)
        rows.append({
            "dataset": dataset,
            "prepared": data_path.is_file(),
            "samples": samples,
            "manifest_present": manifest_path.is_file(),
            "path": display_path,
        })
    return rows


def energy_readiness() -> dict[str, Any]:
    configured_path = settings.energy_meter_reading_path
    status = probe(configured_path)
    meter_exists = False
    if configured_path:
        p = Path(configured_path)
        meter_exists = p.is_file() or (ROOT / configured_path).is_file()
    return {
        **status,
        "configured_meter_path": configured_path,
        "meter_export_exists": meter_exists,
        "proxy_note": "CPU utilisation energy remains an estimate and is never presented as physical-meter energy.",
    }


def selection_history(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent private history without exposing stored response text."""
    if limit < 1:
        return []
    try:
        rows = [json.loads(line) for line in SELECTION_HISTORY.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    output: list[dict[str, Any]] = []
    for row in reversed(rows[-limit:]):
        telemetry = row.get("telemetry") or {}
        output.append({
            "model_id": row.get("model_id"),
            "configuration_id": row.get("configuration_id"),
            "task_type": row.get("task_type"),
            "latency_seconds": telemetry.get("end_to_end_seconds") or telemetry.get("latency_seconds"),
            "ttft_seconds": telemetry.get("ttft_seconds"),
            "peak_ram_mb": telemetry.get("peak_ram_mb"),
            "energy_joules": telemetry.get("energy_joules"),
            "energy_proxy_joules": telemetry.get("energy_proxy_joules"),
            "response_text_retained": bool(telemetry.get("response_text_retained")),
        })
    return output


def profile_pareto_view(task_type: str = "spider_text_to_sql") -> list[dict[str, Any]]:
    """Return non-dominated calibrated profiles with their measurement provenance."""
    rows = profile_comparison_view(task_type)
    return pareto_frontier(rows)


def profile_comparison_view(task_type: str = "spider_text_to_sql") -> list[dict[str, Any]]:
    """Return all selectable benchmark profiles for dashboard trade-off views."""
    rows = []
    for profile in latest_profiles():
        if profile.get("task_type") != task_type or profile.get("quality") is None:
            continue
        rows.append({
            "model_id": profile.get("model_id"),
            "configuration_id": profile.get("configuration_id"),
            "quality": profile.get("quality"),
            "quality_lower_confidence_bound": profile.get("quality_lower_confidence_bound"),
            "latency_seconds": profile.get("latency_seconds"),
            "ttft_seconds": profile.get("ttft_seconds"),
            "ram_mb": profile.get("peak_ram_mb", profile.get("ram_mb")),
            "energy_joules": profile.get("energy_joules"),
            "energy_proxy_joules": profile.get("energy_proxy_joules"),
            "sample_count": profile.get("sample_count"),
            "trial_count": profile.get("trial_count"),
            "unique_prompt_count": profile.get("unique_prompt_count"),
            "reproducibility_status": profile.get("reproducibility_status", "legacy/unverified"),
            "measurement_kind": profile.get("measurement_kind"),
            "dataset": profile.get("dataset"),
        })
    # Keep dominated profiles in the comparison view.  The dashboard needs to
    # disclose complete measured coverage (including Q5/Q6 profiles), while
    # ``profile_pareto_view`` is the explicit filtered view used for Pareto
    # presentation.
    return rows


def readiness_report() -> dict[str, Any]:
    return {
        "datasets": dataset_readiness(),
        "energy": energy_readiness(),
        "selection_history_count": len(selection_history(10_000)),
        "pareto_profile_count": len(profile_pareto_view()),
        "provenance_note": "Raw benchmark JSONL records retain model hash, runtime configuration, llama.cpp version, and local hardware snapshot.",
    }
