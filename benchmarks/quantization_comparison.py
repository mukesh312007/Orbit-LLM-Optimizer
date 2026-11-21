"""Preflight and controlled comparison for locally available Qwen 1.5B GGUF quantizations.

This module intentionally never downloads, converts, moves, or registers a
model.  It becomes runnable only after the user places compatible GGUF files
in the documented local paths.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.spider_calibration import load_spider_dev, prompt_for
from core.ids import new_experiment_id
from core.settings import ROOT
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


VARIANTS: dict[str, str] = {
    "Q4_K_M": "models/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    "Q5_K_M": "models/quantizations/qwen2.5-1.5b-instruct-q5_k_m.gguf",
    "Q6_K": "models/quantizations/qwen2.5-1.5b-instruct-q6_k.gguf",
}


def _is_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def inventory(root: Path = ROOT) -> dict[str, Any]:
    """Report only local, verifiable artifacts and prerequisites."""
    drive = shutil.disk_usage(root.anchor or root)
    variants: list[dict[str, Any]] = []
    for quantization, relative_path in VARIANTS.items():
        path = root / relative_path
        exists = path.is_file()
        variants.append({
            "quantization": quantization, "path": relative_path, "exists": exists,
            "file_size_bytes": path.stat().st_size if exists else None,
            "file_size_gib": round(path.stat().st_size / 1024**3, 3) if exists else None,
            "gguf_header_valid": _is_gguf(path) if exists else False,
            "status": "ready" if exists and _is_gguf(path) else "missing" if not exists else "invalid_gguf_header",
        })
    return {
        "model_family": "Qwen2.5-1.5B-Instruct", "architecture_requirement": "same base model and compatible GGUF tokenizer",
        "disk_free_gib": round(drive.free / 1024**3, 2), "quantization_tool_available": shutil.which("llama-quantize") is not None,
        "variants": variants,
        "download_or_conversion_authorization_required": True,
        "source_status": "Official acquisition metadata is recorded in docs/MODEL_SOURCES.md; this preflight itself performs no network access.",
    }


def configuration(model_id: str) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu:quantization-study", "backend": "llama.cpp",
        "threads": 4, "context_size": 2048, "gpu_layers": 0, "cache": "server-default",
    }


def _model(quantization: str, relative_path: str) -> dict[str, Any]:
    return {
        "model_id": f"qwen2.5-1.5b-{quantization.lower()}", "family": "Qwen2.5", "parameter_count_b": 1.5,
        "quantization": quantization, "path": relative_path, "format": "GGUF", "backends": ["llama.cpp"], "available_locally": True,
    }


def select_diverse_samples(limit: int) -> list[dict[str, Any]]:
    """Choose reproducible development prompts covering distinct databases first."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    all_samples = load_spider_dev(10_000)
    selected: list[dict[str, Any]] = []
    seen_databases: set[str] = set()
    for sample in all_samples:
        if sample["db_id"] not in seen_databases:
            selected.append(sample)
            seen_databases.add(sample["db_id"])
        if len(selected) == limit:
            return selected
    for sample in all_samples:
        if sample not in selected:
            selected.append(sample)
        if len(selected) == limit:
            return selected
    return selected


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "stddev": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if record.get("phase") == "measured" and record.get("success")]
    result: dict[str, Any] = {}
    for quantization in sorted({record["quantization"] for record in measured}):
        group = [record for record in measured if record["quantization"] == quantization]
        result[quantization] = {
            "successful_requests": len(group), "task_quality": _stats([record["task_quality"]["score"] for record in group]),
            "metrics": {key: _stats([record["telemetry"].get(key) for record in group if record["telemetry"].get(key) is not None]) for key in ("end_to_end_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")},
        }
    return result


def run(limit: int = 5, trials: int = 3, max_output_tokens: int = 128) -> tuple[Path, Path]:
    """Run identical Spider workload after all three local variants are verified."""
    check = inventory()
    ready = {item["quantization"]: item for item in check["variants"] if item["status"] == "ready"}
    missing = [name for name in VARIANTS if name not in ready]
    if missing:
        raise FileNotFoundError(
            "Quantization study needs local, valid GGUF files for: " + ", ".join(missing) + ". "
            "Run the preflight first; this command will not download or convert them."
        )
    if limit < 1 or trials < 1:
        raise ValueError("limit and trials must both be at least one")
    experiment_id = new_experiment_id("quantization_spider")
    samples = select_diverse_samples(limit)
    records: list[dict[str, Any]] = []
    for quantization, relative_path in VARIANTS.items():
        model = _model(quantization, relative_path)
        config = configuration(model["model_id"])
        try:
            ensure_runtime(model, config, force_restart=True)
            warmup = generate(model, prompt_for(samples[0]), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
            records.append(_record(experiment_id, model, samples[0], 0, "warmup", warmup))
        except (InferenceError, RuntimeSwitchError) as error:
            records.append(_record(experiment_id, model, samples[0], 0, "warmup", error=error))
            continue
        for trial in range(1, trials + 1):
            for sample in samples:
                try:
                    telemetry = generate(model, prompt_for(sample), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(_record(experiment_id, model, sample, trial, "measured", telemetry))
                except (InferenceError, RuntimeSwitchError) as error:
                    records.append(_record(experiment_id, model, sample, trial, "measured", error=error))
    output_dir = ROOT / "logs" / "quantization_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": "spider_dev", "sample_limit": limit, "trials": trials,
        "sample_selection": "first development example from each database, then remaining examples in source order",
        "database_ids": [sample["db_id"] for sample in samples], "preflight": check, "summary": summarize(records),
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


def _record(experiment_id: str, model: dict[str, Any], sample: dict[str, Any], trial: int, phase: str, telemetry: dict[str, Any] | None = None, error: Exception | None = None) -> dict[str, Any]:
    record = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": "spider_dev", "phase": phase,
        "trial_number": trial, "model_id": model["model_id"], "model_file": model["path"], "quantization": model["quantization"],
        "configuration": configuration(model["model_id"]), "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}",
    }
    if error:
        return record | {"success": False, "return_code": 1, "error": str(error), "measurement_kind": "measured_failure"}
    assert telemetry is not None
    return record | {
        "success": True, "return_code": 0, "measurement_kind": "measured" if phase == "measured" else "warmup",
        "response": telemetry["answer"], "output_tokens": telemetry.get("generated_tokens"),
        "task_quality": evaluate_spider_sql(telemetry["answer"], sample["query"]),
        "telemetry": telemetry, "energy_note": "CPU-utilisation proxy only; not a hardware energy measurement",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preflight or run a local Qwen 1.5B quantization study.")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    if arguments.preflight:
        print(json.dumps(inventory(), indent=2))
    else:
        raw, summary = run(arguments.limit, arguments.trials, arguments.max_output_tokens)
        print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
