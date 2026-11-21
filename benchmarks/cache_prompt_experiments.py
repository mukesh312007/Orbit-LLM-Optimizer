"""Local, controlled prompt-cache benchmark for a llama.cpp server."""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.spider_calibration import load_spider_dev, prompt_for
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.settings import ROOT
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate, tokenize
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


def configuration(model_id: str) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu:cache-study",
        "backend": "llama.cpp", "threads": 4, "context_size": 2048,
        "gpu_layers": 0, "cache": "controlled-prompt-cache",
    }


def cache_controls_supported(help_text: str) -> bool:
    """True only when this llama.cpp binary can explicitly toggle prompt cache."""
    return "--cache-prompt" in help_text and "--no-cache-prompt" in help_text


def _server_help() -> str:
    executable = shutil.which("llama-server")
    if not executable:
        return ""
    try:
        completed = subprocess.run([executable, "--help"], capture_output=True, text=True, timeout=15, check=False)
        return completed.stdout + completed.stderr
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for cache_mode in ("disabled", "enabled"):
        for request_state in ("cold", "warm_repeat"):
            group = [
                item for item in records
                if item.get("success") and item.get("cache_mode") == cache_mode and item.get("request_state") == request_state
            ]
            telemetry_keys = ("end_to_end_seconds", "ttft_seconds", "prompt_processing_seconds", "prompt_tokens_per_second", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")
            summary[f"{cache_mode}:{request_state}"] = {
                "successful_requests": len(group),
                "task_quality": _stats([item["task_quality"]["score"] for item in group]),
                "metrics": {key: _stats([item["telemetry"].get(key) for item in group if item["telemetry"].get(key) is not None]) for key in telemetry_keys},
            }
    return summary


def _record(
    experiment_id: str, model: dict[str, Any], sample: dict[str, Any], cache_mode: str,
    request_state: str, trial: int, telemetry: dict[str, Any] | None = None, error: Exception | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(),
        "dataset": "spider_dev", "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}",
        "model_id": model["model_id"], "model_file": model["path"], "quantization": model.get("quantization"),
        "configuration": configuration(model["model_id"]), "cache_mode": cache_mode,
        "request_state": request_state, "trial_number": trial, "measurement_kind": "measured",
    }
    if error:
        return record | {"success": False, "error": str(error), "return_code": 1}
    assert telemetry is not None
    quality = evaluate_spider_sql(telemetry["answer"], sample["query"])
    watts = telemetry.get("energy_proxy_watts")
    return record | {
        "success": True, "return_code": 0, "response": telemetry["answer"], "input_tokens": tokenize(prompt_for(sample)),
        "output_tokens": telemetry.get("generated_tokens"), "task_quality": quality,
        "quality_per_estimated_watt": quality["score"] / watts if watts else None,
        "telemetry": telemetry,
        "energy_note": "CPU-utilisation proxy only; not a hardware energy measurement",
    }


def run(model_id: str = "qwen2.5-1.5b-q4_k_m", limit: int = 5, trials: int = 3, max_output_tokens: int = 128) -> tuple[Path, Path]:
    """Measure a fresh request and immediate repeat for both cache settings.

    Every trial force-restarts the managed local server before the first request.
    This makes the first request cold and the second request the only cache-warm
    request.  The experiment never stops a server it does not manage.
    """
    if limit < 1 or trials < 1:
        raise ValueError("limit and trials must both be at least one")
    experiment_id = new_experiment_id("cache_prompt")
    output_dir = ROOT / "logs" / "cache_prompt_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    summary_path = output_dir / f"{experiment_id}.summary.json"
    help_text = _server_help()
    if not cache_controls_supported(help_text):
        summary_path.write_text(json.dumps({"experiment_id": experiment_id, "status": "unavailable", "reason": "This local llama-server does not expose both prompt-cache control flags."}, indent=2), encoding="utf-8")
        raw_path.write_text("", encoding="utf-8")
        return raw_path, summary_path

    model = next((item for item in load_models() if item["model_id"] == model_id and item.get("available_locally")), None)
    if not model:
        raise FileNotFoundError(f"Requested local llama.cpp model is unavailable: {model_id}")
    samples = load_spider_dev(limit)
    records: list[dict[str, Any]] = []
    for cache_mode, switch in (("disabled", "--no-cache-prompt"), ("enabled", "--cache-prompt")):
        for trial in range(1, trials + 1):
            for sample in samples:
                try:
                    ensure_runtime(model, configuration(model_id), [switch], force_restart=True)
                    cold = generate(model, prompt_for(sample), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(_record(experiment_id, model, sample, cache_mode, "cold", trial, cold))
                    warm = generate(model, prompt_for(sample), {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(_record(experiment_id, model, sample, cache_mode, "warm_repeat", trial, warm))
                except (InferenceError, RuntimeSwitchError) as error:
                    records.append(_record(experiment_id, model, sample, cache_mode, "cold", trial, error=error))
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "model_id": model_id, "dataset": "spider_dev", "sample_limit": limit,
        "trials": trials, "cache_control_flags": ["--no-cache-prompt", "--cache-prompt"],
        "cold_definition": "first generation after a managed-server restart", "warm_definition": "immediate identical-prompt repeat", "summary": summarize(records),
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run local llama.cpp prompt-cache experiment.")
    parser.add_argument("--model-id", default="qwen2.5-1.5b-q4_k_m")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    raw, summary = run(arguments.model_id, arguments.limit, arguments.trials, arguments.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
