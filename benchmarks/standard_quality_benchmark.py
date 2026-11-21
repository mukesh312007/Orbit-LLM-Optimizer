"""Run locally prepared, task-specific quality datasets with reproducible telemetry.

Datasets are never downloaded here.  Use ``prepare_datasets`` on a reviewed
official source first, then this runner records raw trial evidence in JSONL.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.dataset_loader import load_processed
from benchmarks.quality_metrics import evaluate
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.provenance import benchmark_provenance
from core.settings import ROOT
from database.telemetry import mirror_hardware_profile
from optimizer.configuration_catalog import baseline_configuration
from optimizer.live_matrix import required_memory_mb, SAFETY_RESERVE_MB
from runtime.clients import InferenceError, generate, tokenize
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None}
    return {
        "count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else None,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if record.get("phase") == "measured" and record.get("success")]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in measured:
        grouped.setdefault(record["model_id"], []).append(record)
    return {
        model_id: {
            "successful_trials": len(group),
            "quality": _stats([record["quality"]["score"] for record in group if record["quality"].get("score") is not None]),
            "unscored_trials": sum(record["quality"].get("score") is None for record in group),
            "metrics": {
                metric: _stats([record["telemetry"].get(metric) for record in group if record["telemetry"].get(metric) is not None])
                for metric in ("end_to_end_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb", "energy_joules", "energy_proxy_joules")
            },
        }
        for model_id, group in grouped.items()
    }


def run(
    dataset: str,
    model_ids: list[str] | None = None,
    limit: int = 50,
    trials: int = 3,
    max_output_tokens: int = 256,
) -> tuple[Path, Path]:
    if limit < 1 or trials < 1:
        raise ValueError("limit and trials must both be at least one")
    samples = load_processed(dataset)[:limit]
    hardware_mirror = mirror_hardware_profile()
    if not samples:
        raise ValueError(f"No prepared records were found for {dataset}")
    models = [model for model in load_models() if model.get("available_locally") and "llama.cpp" in model.get("backends", [])]
    selected = [model for model in models if not model_ids or model["model_id"] in model_ids]
    missing = sorted(set(model_ids or []) - {model["model_id"] for model in selected})
    if missing:
        raise FileNotFoundError(f"Requested local models are unavailable: {', '.join(missing)}")

    import psutil
    available_mb = psutil.virtual_memory().available / 1024**2
    experiment_id = new_experiment_id(f"quality_{dataset}")
    records: list[dict[str, Any]] = []
    for model in selected:
        configuration = baseline_configuration(model)
        required_mb = required_memory_mb(model, configuration)
        if required_mb + SAFETY_RESERVE_MB > available_mb:
            records.append({
                "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": dataset,
                "phase": "safety_skip", "success": False, "model_id": model["model_id"],
                "configuration": configuration, "provenance": benchmark_provenance(model, configuration),
                "error": f"RAM safety skip: estimated {required_mb:.0f} MiB plus {SAFETY_RESERVE_MB:.0f} MiB reserve; available {available_mb:.0f} MiB",
            })
            continue
        try:
            ensure_runtime(model, configuration, force_restart=True)
            generate(model, "Reply only with ready.", {"backend": "llama.cpp", "max_output_tokens": 8, "temperature": 0})
        except (InferenceError, RuntimeSwitchError) as error:
            records.append({
                "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": dataset,
                "phase": "warmup", "success": False, "model_id": model["model_id"], "configuration": configuration,
                "provenance": benchmark_provenance(model, configuration), "error": str(error),
            })
            continue
        for trial in range(1, trials + 1):
            for sample in samples:
                record = {
                    "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": dataset,
                    "phase": "measured", "trial_number": trial, "success": False,
                    "model_id": model["model_id"], "configuration": configuration,
                    "prompt_id": sample["sample_id"], "task_type": sample["task_type"],
                    "evaluation_type": sample["evaluation_type"], "provenance": benchmark_provenance(model, configuration),
                }
                try:
                    telemetry = generate(model, sample["prompt"], {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(record | {
                        "success": True, "response": telemetry["answer"], "input_tokens": tokenize(sample["prompt"]),
                        "output_tokens": telemetry.get("generated_tokens"), "telemetry": telemetry,
                        "quality": evaluate(sample, telemetry["answer"]),
                    })
                except (InferenceError, RuntimeSwitchError) as error:
                    records.append(record | {"error": str(error)})
    directory = ROOT / "logs" / "standard_quality_benchmarks"
    directory.mkdir(parents=True, exist_ok=True)
    raw = directory / f"{experiment_id}.jsonl"
    raw.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary = directory / f"{experiment_id}.summary.json"
    summary.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": dataset, "sample_count": len(samples), "trials": trials,
        "models": [model["model_id"] for model in selected], "summary": summarize(records),
        "hardware_profile_mirror": hardware_mirror,
        "quality_note": "Only task-specific evaluator results are quality evidence; unavailable scores do not satisfy a quality threshold.",
    }, indent=2), encoding="utf-8")
    return raw, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a locally prepared task-specific quality dataset.")
    parser.add_argument("dataset", choices=["mmlu", "mmlu_official", "gsm8k", "mbpp", "ifeval", "ifeval_official", "cnn_dailymail"])
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()
    raw_path, summary_path = run(args.dataset, args.model_ids, args.limit, args.trials, args.max_output_tokens)
    print(json.dumps({"raw_results": str(raw_path), "summary": str(summary_path)}, indent=2))
