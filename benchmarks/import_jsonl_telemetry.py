"""Best-effort local import of benchmark JSONL evidence into MySQL telemetry."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from core.model_registry import get_model
from core.settings import ROOT
from database.telemetry import mirror_benchmark_trial


def canonicalize(record: dict[str, Any]) -> dict[str, Any] | None:
    """Map legacy/current benchmark rows to the common MySQL mirror shape."""
    if record.get("phase") not in {"warmup", "measured"} or not record.get("success"):
        return None
    configuration = record.get("configuration") or record.get("runtime_configuration")
    model_id = record.get("model_id")
    if not isinstance(configuration, dict) or not isinstance(model_id, str):
        return None
    model = get_model(model_id) or {}
    telemetry = dict(record.get("telemetry") or {})
    for key in ("end_to_end_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb", "generated_tokens", "energy_proxy_joules", "energy_joules", "energy_source", "energy_measurement_kind"):
        if key in record and key not in telemetry:
            telemetry[key] = record[key]
    quality = record.get("task_quality") or record.get("quality")
    if not quality and record.get("evaluation"):
        quality = {
            "metric": "local_retrieval_correctness", "score": record["evaluation"].get("correctness"),
            "details": record["evaluation"],
        }
    return {
        "experiment_id": record["experiment_id"], "measurement_kind": record.get("measurement_kind", "measured"),
        "trial_number": record.get("trial_number", record.get("trial")),
        "model_id": model_id, "model_file": record.get("model_file", model.get("path", "unknown")),
        "quantization": record.get("quantization", model.get("quantization")),
        "configuration": configuration, "prompt_id": record.get("prompt_id"), "response": record.get("response"),
        "task_quality": quality or {"metric": "unavailable", "score": None}, "telemetry": telemetry,
    }


def import_logs(dry_run: bool = False) -> dict[str, int]:
    counts = {"scanned": 0, "canonical": 0, "persisted": 0, "unavailable": 0}
    for path in ROOT.glob("logs/**/*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            counts["scanned"] += 1
            try:
                normalized = canonicalize(json.loads(line))
            except json.JSONDecodeError:
                continue
            if not normalized:
                continue
            counts["canonical"] += 1
            if dry_run:
                continue
            status = mirror_benchmark_trial(normalized).get("status")
            counts["persisted" if status == "persisted" else "unavailable"] += 1
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror local benchmark JSONL records into configured local MySQL.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(import_logs(args.dry_run), indent=2))
