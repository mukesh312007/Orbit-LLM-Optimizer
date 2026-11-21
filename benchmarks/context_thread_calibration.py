"""Sequential context/thread calibration with explicit safety and failure records."""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.spider_calibration import load_spider_dev, prompt_for
from core.ids import new_experiment_id
from core.model_registry import get_model
from core.settings import ROOT
from optimizer.live_matrix import SAFETY_RESERVE_MB, required_memory_mb, safe_available_memory_mb
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate
from runtime.supervisor import RuntimeSwitchError, ensure_runtime

CONTEXT_SIZES = (1024, 2048, 4096)
THREAD_COUNTS = (2, 4, 6, 8)


def configuration(model_id: str, threads: int, context_size: int) -> dict[str, Any]:
    return {"configuration_id": f"{model_id}:t{threads}:c{context_size}:cpu", "backend": "llama.cpp", "threads": threads, "context_size": context_size, "gpu_layers": 0, "cache": "server-default"}


def safe_to_run(model: dict[str, Any], context_size: int, available_mb: float | None = None) -> tuple[bool, str]:
    available = available_mb if available_mb is not None else safe_available_memory_mb()
    estimated_mb = required_memory_mb(model, configuration(model["model_id"], 4, context_size))
    return (estimated_mb + SAFETY_RESERVE_MB <= available, f"estimated {estimated_mb:.0f} MiB plus {SAFETY_RESERVE_MB} MiB reserve; available {available:.0f} MiB")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record.get("phase") == "measured" and record.get("success")]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in completed:
        groups.setdefault(record["configuration"]["configuration_id"], []).append(record)
    configurations: list[dict[str, Any]] = []
    for configuration_id, group in groups.items():
        quality = [item["quality"]["score"] for item in group]
        latency = [item["telemetry"]["end_to_end_seconds"] for item in group]
        throughput = [item["telemetry"]["generation_tokens_per_second"] for item in group if item["telemetry"].get("generation_tokens_per_second")]
        configurations.append({"configuration": group[0]["configuration"], "samples": len(group), "quality_mean": statistics.mean(quality), "latency_mean": statistics.mean(latency), "throughput_mean": statistics.mean(throughput) if throughput else None, "latency_stddev": statistics.stdev(latency) if len(latency) > 1 else None})
    recommendation = max(configurations, key=lambda item: (item["quality_mean"], item["throughput_mean"] or 0, -item["latency_mean"]), default=None)
    return {"completed_configurations": configurations, "recommendation": recommendation, "skipped_or_failed": [record for record in records if not record.get("success", False)]}


def run(model_id: str, limit: int = 3, trials: int = 2, max_output_tokens: int = 128) -> tuple[Path, Path]:
    model = get_model(model_id)
    if not model or not model.get("available_locally"):
        raise FileNotFoundError(f"Local model is unavailable: {model_id}")
    samples = load_spider_dev(limit)
    experiment_id = new_experiment_id("context_thread")
    records: list[dict[str, Any]] = []
    for context_size in CONTEXT_SIZES:
        for threads in THREAD_COUNTS:
            config = configuration(model_id, threads, context_size)
            safe, reason = safe_to_run(model, context_size)
            if not safe:
                records.append({"experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "phase": "skipped", "success": False, "model_id": model_id, "configuration": config, "error": f"RAM safety skip: {reason}"})
                continue
            try:
                ensure_runtime(model, config)
                generate(model, prompt_for(samples[0]), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
            except (InferenceError, RuntimeSwitchError) as error:
                records.append({"experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "phase": "warmup", "success": False, "model_id": model_id, "configuration": config, "error": str(error)})
                continue
            for trial in range(1, trials + 1):
                for sample in samples:
                    try:
                        telemetry = generate(model, prompt_for(sample), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                        quality = evaluate_spider_sql(telemetry["answer"], sample["query"])
                        records.append({"experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "phase": "measured", "success": True, "model_id": model_id, "configuration": config, "prompt_id": sample["db_id"], "trial": trial, "telemetry": telemetry, "quality": quality})
                    except (InferenceError, RuntimeSwitchError) as error:
                        records.append({"experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "phase": "measured", "success": False, "model_id": model_id, "configuration": config, "prompt_id": sample["db_id"], "trial": trial, "error": str(error)})
    output = ROOT / "logs" / "context_thread_calibrations"
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path = output / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps(summarize(records), indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="qwen2.5-1.5b-q4_k_m")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    args = parser.parse_args()
    raw, summary = run(args.model_id, args.limit, args.trials, args.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
