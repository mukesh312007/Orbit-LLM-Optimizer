"""Operational telemetry store: JSONL is the primary store; MySQL is an optional external mirror."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from core.ids import new_experiment_id
from core.provenance import hardware_snapshot
from core.settings import ROOT, settings
from database.repository import ExperimentRepository


def verify_jsonl_store() -> dict[str, Any]:
    """Inspect and verify the primary JSONL telemetry store and its record counts."""
    logs_dir = ROOT / "logs"
    categories = {
        "full_matrix_benchmarks": logs_dir / "full_matrix_benchmarks",
        "quantization_comparisons": logs_dir / "quantization_comparisons",
        "context_thread_calibrations": logs_dir / "context_thread_calibrations",
        "cache_prompt_experiments": logs_dir / "cache_prompt_experiments",
        "standard_quality_benchmarks": logs_dir / "standard_quality_benchmarks",
        "three_model_comparisons": logs_dir / "three_model_comparisons",
    }
    details: dict[str, Any] = {}
    total_records = 0
    total_bytes = 0
    invalid_records = 0
    provenance_gaps = 0

    for name, directory in categories.items():
        count = 0
        file_count = 0
        byte_count = 0
        if directory.is_dir():
            for jsonl_file in directory.glob("*.jsonl"):
                file_count += 1
                byte_count += jsonl_file.stat().st_size
                with jsonl_file.open("r", encoding="utf-8", errors="replace") as f:
                    for line_number, line in enumerate(f, start=1):
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            invalid_records += 1
                            continue
                        count += 1
                        if name == "full_matrix_benchmarks" and record.get("phase") in {"warmup", "measured"} and record.get("success") and not record.get("provenance"):
                            provenance_gaps += 1
        details[name] = {"files": file_count, "records": count, "bytes": byte_count}
        total_records += count
        total_bytes += byte_count

    history_file = logs_dir / "runtime_selection_history.jsonl"
    history_records = 0
    history_bytes = 0
    if history_file.is_file():
        history_bytes = history_file.stat().st_size
        with history_file.open("r", encoding="utf-8", errors="replace") as f:
            history_records = sum(1 for line in f if line.strip())
    details["runtime_selection_history"] = {"files": 1 if history_file.is_file() else 0, "records": history_records, "bytes": history_bytes}
    total_records += history_records
    total_bytes += history_bytes

    status = "operational" if not invalid_records and not provenance_gaps else "degraded"
    return {
        "primary_store": "jsonl",
        "status": status,
        "total_records": total_records,
        "total_bytes": total_bytes,
        "invalid_json_records": invalid_records,
        "provenance_gaps": provenance_gaps,
        "categories": details,
        "note": "JSONL is Orbit's authoritative local evidence store; MySQL functions as an optional mirror.",
    }


def mirror_live_selection(decision: dict[str, Any], telemetry: dict[str, Any]) -> dict[str, Any]:
    """Persist MySQL telemetry when configured, without affecting inference."""
    selected = decision.get("selected")
    url = settings.database_url
    if not selected or not url or not url.startswith("mysql+"):
        return {"status": "unavailable", "reason": "Local MySQL telemetry is not configured."}
    try:
        experiment_id = new_experiment_id("live_selection")
        ExperimentRepository(url).save_live_selection(
            experiment_id=experiment_id,
            model=selected["model"],
            configuration=decision["configuration"],
            task_type=decision["task_type"],
            telemetry=telemetry,
        )
        return {"status": "persisted", "experiment_id": experiment_id}
    except SQLAlchemyError as error:
        # A database outage must never prevent a local answer or JSONL evidence.
        return {"status": "unavailable", "reason": f"MySQL telemetry unavailable: {error.__class__.__name__}"}


def mirror_benchmark_trial(record: dict[str, Any]) -> dict[str, Any]:
    """Mirror a benchmark record without making MySQL a benchmark dependency."""
    url = settings.database_url
    if not url or not url.startswith("mysql+"):
        return {"status": "unavailable", "reason": "Local MySQL telemetry is not configured."}
    try:
        configuration_id = record["configuration"]["configuration_id"]
        suffix = hashlib.sha256(configuration_id.encode("utf-8")).hexdigest()[:12]
        database_record = record | {
            "model": {
                "model_id": record["model_id"], "family": "Qwen2.5", "quantization": record.get("quantization"),
                "path": record["model_file"], "format": "GGUF", "backends": ["llama.cpp"],
            },
            # The source schema uniquely identifies prompt/trial inside one
            # configuration experiment. Retain the original prompt in JSONL.
            "database_prompt_id": f"{configuration_id}|{record.get('prompt_id') or record.get('phase')}",
        }
        ExperimentRepository(url).save_benchmark_trial(
            experiment_id=f"{record['experiment_id'][:100]}_{suffix}",
            kind="full_matrix_spider", record=database_record,
        )
        return {"status": "persisted"}
    except (SQLAlchemyError, KeyError, TypeError) as error:
        return {"status": "unavailable", "reason": f"MySQL telemetry unavailable: {error.__class__.__name__}"}


def mirror_hardware_profile(profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Store a local hardware snapshot when MySQL telemetry is configured."""
    url = settings.database_url
    if not url or not url.startswith("mysql+"):
        return {"status": "unavailable", "reason": "Local MySQL telemetry is not configured."}
    try:
        ExperimentRepository(url).save_hardware_profile(profile or hardware_snapshot())
        return {"status": "persisted"}
    except SQLAlchemyError as error:
        return {"status": "unavailable", "reason": f"MySQL telemetry unavailable: {error.__class__.__name__}"}
