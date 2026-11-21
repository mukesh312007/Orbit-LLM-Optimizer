"""Reproducible all-model/all-configuration Spider benchmark runner.

The runner is intentionally sequential: a single llama.cpp server is active at
any instant.  A full matrix is expensive, so this module exposes it explicitly
instead of running it as a side effect of the chat dashboard.
"""
from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.local_retrieval_benchmark import build_schema_corpus
from benchmarks.spider_calibration import load_spider_dev, prompt_for
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.provenance import benchmark_provenance
from core.settings import ROOT
from database.telemetry import mirror_benchmark_trial, mirror_hardware_profile
from optimizer.configuration_catalog import baseline_configuration, cache_server_arguments
from optimizer.live_matrix import configuration_is_safe, full_live_candidates, stratified_sample
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate, tokenize
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, active_runtime_state, ensure_runtime


def select_diverse_samples(limit: int) -> list[dict[str, Any]]:
    """Choose one reproducible development sample per database before repeats."""
    if limit < 1:
        raise ValueError("limit must be at least one")
    all_samples = load_spider_dev(10_000)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sample in all_samples:
        if sample["db_id"] not in seen:
            selected.append(sample)
            seen.add(sample["db_id"])
        if len(selected) >= limit:
            return selected
    for sample in all_samples:
        if sample not in selected:
            selected.append(sample)
        if len(selected) >= limit:
            return selected
    return selected


def resident_configurations(model: dict[str, Any], state: dict[str, Any], corpus_dir: str) -> list[dict[str, Any]]:
    """Prompt-only variants that safely reuse an already-loaded Orbit model."""
    return [
        {
            "configuration_id": (
                f"{model['model_id']}:t{state['threads']}:c{state['context_size']}:cpu:"
                f"cache-server-default:compress-{compression}:retrieval-{'on' if retrieval_enabled else 'off'}:resident"
            ),
            "backend": "llama.cpp", "threads": int(state["threads"]),
            "context_size": int(state["context_size"]), "gpu_layers": int(state.get("gpu_layers", 0)),
            "cache": "server-default", "prompt_compression": compression,
            "retrieval": {"enabled": retrieval_enabled, "top_k": 3, "corpus_dir": corpus_dir},
            "resident_reuse": True,
        }
        for compression, retrieval_enabled in itertools.product(("none", "whitespace", "conservative"), (False, True))
    ]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "stddev": None, "min": None, "max": None}
    return {
        "count": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else None, "min": min(values), "max": max(values),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in records if item.get("phase") == "measured" and item.get("success")]
    configurations: dict[str, list[dict[str, Any]]] = {}
    for record in measured:
        configurations.setdefault(record["configuration"]["configuration_id"], []).append(record)
    summary: dict[str, Any] = {}
    for configuration_id, group in configurations.items():
        summary[configuration_id] = {
            "model_id": group[0]["model_id"], "quantization": group[0].get("quantization"),
            "configuration": group[0]["configuration"], "successful_trials": len(group),
            "task_quality": _stats([item["task_quality"]["score"] for item in group if item["task_quality"].get("score") is not None]),
            "metrics": {
                name: _stats([item["telemetry"].get(name) for item in group if item["telemetry"].get(name) is not None])
                for name in ("end_to_end_seconds", "ttft_seconds", "prompt_processing_seconds", "prompt_tokens_per_second", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")
            },
        }
    return {
        "configurations": summary,
        "success_count": len(measured),
        "failure_count": sum(item.get("phase") == "measured" and not item.get("success") for item in records),
        "safety_skip_count": sum(item.get("phase") == "safety_skip" for item in records),
        "deferred_count": sum(item.get("phase") == "deferred" for item in records),
    }


def _record(
    experiment_id: str,
    model: dict[str, Any],
    configuration: dict[str, Any],
    sample: dict[str, Any] | None,
    phase: str,
    trial: int | None,
    activation_seconds: float | None = None,
    prepared_prompt: str | None = None,
    preparation: dict[str, Any] | None = None,
    telemetry: dict[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": "spider_dev",
        "phase": phase, "trial_number": trial, "model_id": model["model_id"], "model_file": model["path"],
        "quantization": model.get("quantization"), "configuration": configuration,
        "provenance": benchmark_provenance(model, configuration),
        "runtime_activation_seconds": activation_seconds,
        "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}" if sample else None,
    }
    if error:
        return record | {"success": False, "return_code": 1, "error": error, "measurement_kind": "measured_failure"}
    assert sample is not None and prepared_prompt is not None and preparation is not None and telemetry is not None
    quality = evaluate_spider_sql(telemetry["answer"], sample["query"])
    watts = telemetry.get("energy_proxy_watts")
    return record | {
        "success": True, "return_code": 0,
        "measurement_kind": "warmup" if phase == "warmup" else "measured",
        "prepared_prompt": prepared_prompt, "preparation": preparation,
        "input_tokens": tokenize(prepared_prompt), "output_tokens": telemetry.get("generated_tokens"),
        "response": telemetry["answer"], "task_quality": quality,
        "quality_per_proxy_joule": quality["score"] / telemetry["energy_proxy_joules"] if quality.get("score") is not None and telemetry.get("energy_proxy_joules") and telemetry["energy_proxy_joules"] > 0 else None,
        "quality_per_estimated_watt": quality["score"] / watts if quality.get("score") is not None and watts else None,
        "quality_per_watt": quality["score"] / watts if quality.get("score") is not None and watts else None,
        "telemetry": telemetry,
        "energy_note": "CPU-utilisation proxy only; not a hardware energy measurement",
    }


def _persist(record: dict[str, Any]) -> dict[str, Any]:
    """Attach best-effort local MySQL status after JSONL-compatible record creation."""
    if record.get("phase") not in {"warmup", "measured"}:
        return record
    return record | {"mysql_telemetry": mirror_benchmark_trial(record)}


def run(
    limit: int = 10,
    trials: int = 3,
    max_output_tokens: int = 128,
    model_ids: list[str] | None = None,
    max_configurations: int | None = None,
    resident_only: bool = False,
    sampling_strategy: str = "stratified",
    baseline_only: bool = False,
) -> tuple[Path, Path]:
    """Run all safe candidates, or a complete baseline subset across models."""
    if trials < 1:
        raise ValueError("trials must be at least one")
    if max_configurations is not None and max_configurations < 1:
        raise ValueError("max_configurations must be at least one when provided")
    samples = select_diverse_samples(limit)
    hardware_mirror = mirror_hardware_profile()
    corpus_dir = build_schema_corpus(samples)
    all_models = [item for item in load_models() if item.get("available_locally") and "llama.cpp" in item.get("backends", [])]
    selected_models = [item for item in all_models if not model_ids or item["model_id"] in model_ids]
    missing = sorted(set(model_ids or []) - {item["model_id"] for item in selected_models})
    if missing:
        raise FileNotFoundError(f"Requested local models are unavailable: {', '.join(missing)}")
    if resident_only and baseline_only:
        raise ValueError("baseline_only and resident_only cannot be combined")
    if resident_only:
        state = active_runtime_state()
        if not state:
            raise RuntimeError("Resident-only mode needs an active Orbit-managed llama.cpp server.")
        active_model = next((model for model in selected_models if model["model_id"] == state.get("model_id")), None)
        if not active_model:
            raise RuntimeError("Resident-only mode may benchmark only the model already managed by Orbit.")
        candidates = [
            {"model": active_model, "configuration": configuration, "required_mb": 0.0}
            for configuration in resident_configurations(active_model, state, corpus_dir)
        ]
        safety_skips = []
    elif baseline_only:
        candidates = []
        safety_skips = []
        for model in selected_models:
            configuration = baseline_configuration(model)
            safe, reason, required = configuration_is_safe(model, configuration)
            candidate = {"model": model, "configuration": configuration, "required_mb": required}
            if safe:
                candidates.append(candidate)
            else:
                safety_skips.append(candidate | {"reason": f"RAM safety skip: {reason}"})
    else:
        candidates, safety_skips = full_live_candidates(selected_models)
    deferred: list[dict[str, Any]] = []
    if max_configurations is not None:
        if sampling_strategy == "stratified":
            sampled = stratified_sample(candidates, max_configurations)
            sampled_ids = {id(c) for c in sampled}
            deferred = [c for c in candidates if id(c) not in sampled_ids]
            candidates = sampled
        else:  # "prefix" – original behaviour
            deferred, candidates = candidates[max_configurations:], candidates[:max_configurations]
    experiment_id = new_experiment_id("full_matrix_spider")
    output_dir = ROOT / "logs" / "full_matrix_benchmarks"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    records: list[dict[str, Any]] = []

    def checkpoint() -> None:
        """Persist partial evidence so an interrupted local campaign is auditable."""
        raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")

    for item in safety_skips:
        records.append(_persist(_record(experiment_id, item["model"], item["configuration"], None, "safety_skip", None, error=item["reason"])))
    for item in deferred:
        records.append(_persist(_record(experiment_id, item["model"], item["configuration"], None, "deferred", None, error="Deferred by max_configurations for this partial run.")))
    checkpoint()
    for index, item in enumerate(candidates, start=1):
        model = item["model"]
        configuration = dict(item["configuration"])
        configuration["retrieval"] = {**configuration["retrieval"], "corpus_dir": corpus_dir}
        print(
            f"[{index}/{len(candidates)}] {model['model_id']} {configuration['configuration_id']}",
            flush=True,
        )
        warmup_sample = samples[0]
        try:
            if configuration.get("resident_reuse"):
                activation_seconds = 0.0
            else:
                activation_started = time.perf_counter()
                ensure_runtime(model, configuration, cache_server_arguments(configuration))
                activation_seconds = time.perf_counter() - activation_started
            warmup_prompt, warmup_preparation = prepare_prompt(prompt_for(warmup_sample), configuration)
            warmup = generate(model, warmup_prompt, {"backend": "llama.cpp", "max_output_tokens": min(32, max_output_tokens), "temperature": 0})
            records.append(_persist(_record(experiment_id, model, configuration, warmup_sample, "warmup", 0, activation_seconds, warmup_prompt, warmup_preparation, warmup)))
        except (InferenceError, RuntimeSwitchError) as error:
            records.append(_persist(_record(experiment_id, model, configuration, warmup_sample, "warmup", 0, error=str(error))))
            checkpoint()
            continue
        for trial in range(1, trials + 1):
            for sample in samples:
                prepared_prompt, preparation = prepare_prompt(prompt_for(sample), configuration)
                try:
                    telemetry = generate(model, prepared_prompt, {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                    records.append(_persist(_record(experiment_id, model, configuration, sample, "measured", trial, prepared_prompt=prepared_prompt, preparation=preparation, telemetry=telemetry)))
                except (InferenceError, RuntimeSwitchError) as error:
                    records.append(_persist(_record(experiment_id, model, configuration, sample, "measured", trial, error=str(error))))
        checkpoint()
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": "spider_dev", "sample_limit": limit, "trials": trials,
        "models": [item["model_id"] for item in selected_models], "candidate_count": len(candidates),
        "execution_mode": (
            "resident_prompt_variants" if resident_only
            else "baseline_per_local_model" if baseline_only
            else "full_safe_matrix"
        ),
        "matrix_dimensions": {"threads": [2, 4, 6, 8], "contexts": [1024, 2048, 4096], "cache": ["enabled", "disabled"], "compression": ["none", "whitespace", "conservative"], "retrieval": [False, True]},
        "sample_selection": "first development example from each database, then remaining source order",
        "database_ids": [sample["db_id"] for sample in samples], "retrieval_corpus": corpus_dir,
        "summary": summarize(records), "hardware_profile_mirror": hardware_mirror,
        "provenance_note": "Each trial stores model SHA-256, runtime configuration, llama.cpp version, and hardware snapshot in its raw JSONL record.",
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all safe local model/configuration Spider trials sequentially.")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--max-configurations", type=int)
    parser.add_argument("--sampling-strategy", choices=["stratified", "prefix"], default="stratified",
                        help="How to select configurations when --max-configurations is set (default: stratified).")
    parser.add_argument("--resident-only", action="store_true", help="Reuse the active Orbit model without a restart; evaluates prompt/retrieval variants only.")
    parser.add_argument("--baseline-only", action="store_true", help="Benchmark the baseline configuration for every safe local model.")
    arguments = parser.parse_args()
    raw, summary = run(arguments.limit, arguments.trials, arguments.max_output_tokens, arguments.model_ids, arguments.max_configurations, arguments.resident_only, arguments.sampling_strategy, arguments.baseline_only)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
