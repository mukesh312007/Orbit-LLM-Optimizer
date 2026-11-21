"""Reproducible warm-up and measured comparison for Orbit's three local Qwen models."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.spider_calibration import load_spider_dev, prompt_for
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.settings import ROOT
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate
from runtime.supervisor import RuntimeSwitchError, ensure_runtime

MODEL_IDS = ("qwen2.5-0.5b-q4_k_m", "qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m")


def configuration(model_id: str) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu",
        "backend": "llama.cpp", "threads": 4, "context_size": 2048,
        "gpu_layers": 0, "cache": "server-default",
    }


def _record(experiment_id: str, model: dict[str, Any], sample: dict[str, Any], trial: int, phase: str, response: dict[str, Any] | None = None, error: Exception | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(),
        "phase": phase, "trial_number": trial, "model_id": model["model_id"],
        "model_file": model["path"], "quantization": model.get("quantization"),
        "parameter_count_b": model.get("parameter_count_b"), "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}",
        "prompt_category": "spider_text_to_sql", "runtime_configuration": configuration(model["model_id"]),
    }
    if error:
        return record | {"success": False, "error": str(error), "return_code": 1, "measurement_kind": "measured_failure"}
    assert response is not None
    quality = evaluate_spider_sql(response["answer"], sample["query"])
    return record | {
        "success": True, "return_code": 0, "response": response["answer"],
        "input_tokens": None, "output_tokens": response.get("generated_tokens"),
        "wall_clock_seconds": response.get("end_to_end_seconds"), "ttft_seconds": response.get("ttft_seconds"),
        "generation_tokens_per_second": response.get("generation_tokens_per_second"),
        "peak_ram_mb": response.get("peak_ram_mb"), "task_quality": quality,
        "energy": {"joules": response.get("energy_proxy_joules"), "kind": "estimated_proxy"},
        "measurement_kind": "measured" if phase == "measured" else "warmup",
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if record["phase"] == "measured" and record["success"]]
    summary: dict[str, Any] = {}
    for model_id in sorted({record["model_id"] for record in records}):
        group = [record for record in measured if record["model_id"] == model_id]
        metrics = {}
        for key in ("wall_clock_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb"):
            values = [record[key] for record in group if record.get(key) is not None]
            metrics[key] = _stats(values)
        quality = [record["task_quality"]["score"] for record in group if record.get("task_quality", {}).get("score") is not None]
        summary[model_id] = {"measured_samples": len(group), "failures": sum(not record["success"] for record in records if record["model_id"] == model_id), "task_quality": _stats(quality), "metrics": metrics}
    return summary


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "stddev": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values)}


def run(limit: int = 5, trials: int = 3, max_output_tokens: int = 128) -> tuple[Path, Path]:
    models = {model["model_id"]: model for model in load_models() if model.get("available_locally")}
    missing = [model_id for model_id in MODEL_IDS if model_id not in models]
    if missing:
        raise FileNotFoundError(f"Required local models are unavailable: {', '.join(missing)}")
    samples = load_spider_dev(limit)
    experiment_id = new_experiment_id("three_model_spider")
    records: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        model = models[model_id]
        ensure_runtime(model, configuration(model_id))
        try:
            warmup = generate(model, prompt_for(samples[0]), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
            records.append(_record(experiment_id, model, samples[0], 0, "warmup", warmup))
        except (InferenceError, RuntimeSwitchError) as error:
            records.append(_record(experiment_id, model, samples[0], 0, "warmup", error=error))
        for trial in range(1, trials + 1):
            for sample in samples:
                try:
                    response = generate(model, prompt_for(sample), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(_record(experiment_id, model, sample, trial, "measured", response))
                except (InferenceError, RuntimeSwitchError) as error:
                    records.append(_record(experiment_id, model, sample, trial, "measured", error=error))
    output_dir = ROOT / "logs" / "three_model_comparisons"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({"experiment_id": experiment_id, "configuration": {"threads": 4, "context_size": 2048, "gpu_layers": 0}, "summary": summarize(records)}, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    args = parser.parse_args()
    raw, summary = run(args.limit, args.trials, args.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
