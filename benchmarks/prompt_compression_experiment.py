"""Measure local prompt compression without conflating it with retrieval or cache reuse."""
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
from runtime.clients import InferenceError, generate, tokenize
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


def configuration(model_id: str, compression: str) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu:compression-{compression}",
        "backend": "llama.cpp", "threads": 4, "context_size": 2048, "gpu_layers": 0,
        "cache": "server-default", "prompt_compression": compression,
        "retrieval": {"enabled": False},
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in records if item.get("phase") == "measured" and item.get("success")]
    groups: dict[str, Any] = {}
    for model_id in sorted({item["model_id"] for item in measured}):
        base_quality: float | None = None
        for compression in ("none", "whitespace", "conservative"):
            group = [item for item in measured if item["model_id"] == model_id and item["compression"] == compression]
            quality = _stats([item["task_quality"]["score"] for item in group])
            if compression == "none":
                base_quality = quality["mean"] if isinstance(quality["mean"], (int, float)) else None
            metrics = {
                key: _stats([item["telemetry"].get(key) for item in group if item["telemetry"].get(key) is not None])
                for key in ("end_to_end_seconds", "ttft_seconds", "prompt_processing_seconds", "prompt_tokens_per_second", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")
            }
            groups[f"{model_id}:{compression}"] = {
                "successful_requests": len(group), "task_quality": quality, "metrics": metrics,
                "input_characters": _stats([item["preparation"]["compression"]["output_characters"] for item in group]),
                "exact_input_tokens": _stats([item["input_tokens"] for item in group if item.get("input_tokens") is not None]),
            }
        for compression in ("whitespace", "conservative"):
            compressed_quality = groups.get(f"{model_id}:{compression}", {}).get("task_quality", {}).get("mean")
            groups[f"{model_id}:{compression}"]["quality_delta_vs_none"] = (
                compressed_quality - base_quality
                if isinstance(compressed_quality, (int, float)) and base_quality is not None else None
            )
    return groups


def _record(
    experiment_id: str, model: dict[str, Any], sample: dict[str, Any], compression: str, trial: int,
    phase: str, prepared: str | None = None, preparation: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None, error: Exception | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": "spider_dev",
        "model_id": model["model_id"], "model_file": model["path"], "quantization": model.get("quantization"),
        "compression": compression, "trial_number": trial, "phase": phase,
        "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}",
        "configuration": configuration(model["model_id"], compression),
    }
    if error:
        return record | {"success": False, "error": str(error), "return_code": 1, "measurement_kind": "measured_failure"}
    assert prepared is not None and preparation is not None and telemetry is not None
    quality = evaluate_spider_sql(telemetry["answer"], sample["query"])
    watts = telemetry.get("energy_proxy_watts")
    return record | {
        "success": True, "return_code": 0, "measurement_kind": "measured" if phase == "measured" else "warmup",
        "prepared_prompt": prepared, "preparation": preparation, "input_tokens": tokenize(prepared),
        "output_tokens": telemetry.get("generated_tokens"), "response": telemetry["answer"], "task_quality": quality,
        "quality_per_estimated_watt": quality["score"] / watts if watts else None, "telemetry": telemetry,
        "energy_note": "CPU-utilisation proxy only; not a hardware energy measurement",
    }


def run(model_ids: list[str] | None = None, limit: int = 5, trials: int = 3, max_output_tokens: int = 128) -> tuple[Path, Path]:
    """Compare exact prompt text with whitespace-only compression.

    Retrieval is explicitly disabled.  The server restarts before each condition
    and a distinct warm-up is excluded from summary statistics, preventing a
    previous condition's cache from contaminating the measured result.
    """
    if limit < 1 or trials < 1:
        raise ValueError("limit and trials must both be at least one")
    available = [item for item in load_models() if item.get("available_locally") and "llama.cpp" in item.get("backends", [])]
    chosen_ids = model_ids or [item["model_id"] for item in available]
    models = [item for item in available if item["model_id"] in chosen_ids]
    missing = sorted(set(chosen_ids) - {item["model_id"] for item in models})
    if missing:
        raise FileNotFoundError(f"Requested local llama.cpp models are unavailable: {', '.join(missing)}")
    samples = load_spider_dev(limit)
    experiment_id = new_experiment_id("prompt_compression")
    records: list[dict[str, Any]] = []
    for model in models:
        for compression in ("none", "whitespace", "conservative"):
            config = configuration(model["model_id"], compression)
            try:
                ensure_runtime(model, config, force_restart=True)
                warmup_sample = samples[0]
                warmup_prompt, warmup_preparation = prepare_prompt("Reply with only the word ready.", config)
                warmup = generate(model, warmup_prompt, {"backend": "llama.cpp", "max_output_tokens": 8, "temperature": 0})
                records.append(_record(experiment_id, model, warmup_sample, compression, 0, "warmup", warmup_prompt, warmup_preparation, warmup))
            except (InferenceError, RuntimeSwitchError) as error:
                records.append(_record(experiment_id, model, samples[0], compression, 0, "warmup", error=error))
                continue
            for trial in range(1, trials + 1):
                for sample in samples:
                    prepared, preparation = prepare_prompt(prompt_for(sample), config)
                    try:
                        telemetry = generate(model, prepared, {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                        records.append(_record(experiment_id, model, sample, compression, trial, "measured", prepared, preparation, telemetry))
                    except (InferenceError, RuntimeSwitchError) as error:
                        records.append(_record(experiment_id, model, sample, compression, trial, "measured", error=error))
    output_dir = ROOT / "logs" / "prompt_compression_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": "spider_dev", "sample_limit": limit, "trials": trials,
        "models": [model["model_id"] for model in models], "retrieval": "disabled", "compression_modes": ["none", "whitespace", "conservative"],
        "token_count_note": "Exact local token counts are null when the active llama.cpp build does not expose /tokenize.",
        "summary": summarize(records),
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local prompt-compression study.")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    raw, summary = run(arguments.model_ids, arguments.limit, arguments.trials, arguments.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
