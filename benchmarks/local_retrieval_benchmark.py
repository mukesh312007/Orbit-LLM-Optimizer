"""Controlled, local-only schema retrieval study using Spider development data."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.spider_calibration import load_spider_dev, schema_for
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.settings import ROOT
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import InferenceError, generate, tokenize
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


def prompt_for_retrieval(sample: dict[str, Any]) -> str:
    return (
        "Write one SQL SELECT query and nothing else.\n\n"
        f"Database ID: {sample['db_id']}\n"
        f"Question: {sample['question']}"
    )


def build_schema_corpus(samples: list[dict[str, Any]]) -> str:
    """Make an experiment-local retrieval corpus from Spider schema metadata only.

    It contains no gold SQL or answers.  The stable digest lets later trials
    reuse the exact corpus and exposes its provenance in the result record.
    """
    database_ids = sorted({sample["db_id"] for sample in samples})
    digest = hashlib.sha256("\n".join(database_ids).encode()).hexdigest()[:12]
    relative = f"logs/retrieval_corpora/spider_schema_{digest}"
    root = ROOT / relative
    root.mkdir(parents=True, exist_ok=True)
    for database_id in database_ids:
        document = (
            "Spider database schema reference.\n"
            f"DATABASE_ID: {database_id}\n"
            "Use this source only to identify tables and columns; do not invent schema.\n\n"
            f"{schema_for(database_id, max_characters=12_000)}\n"
        )
        (root / f"{database_id}.schema.txt").write_text(document, encoding="utf-8")
    manifest = {"dataset": "spider_dev", "database_ids": database_ids, "document_count": len(database_ids), "contains_gold_sql": False}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return relative


def configuration(model_id: str, enabled: bool, corpus_dir: str) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu:retrieval-{'on' if enabled else 'off'}",
        "backend": "llama.cpp", "threads": 4, "context_size": 2048, "gpu_layers": 0,
        "cache": "server-default", "prompt_compression": "none",
        "retrieval": {"enabled": enabled, "top_k": 1, "corpus_dir": corpus_dir},
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [item for item in records if item.get("phase") == "measured" and item.get("success")]
    result: dict[str, Any] = {}
    for model_id in sorted({item["model_id"] for item in measured}):
        baseline: float | None = None
        for enabled in (False, True):
            group = [item for item in measured if item["model_id"] == model_id and item["retrieval_enabled"] == enabled]
            quality = _stats([item["task_quality"]["score"] for item in group])
            if not enabled:
                baseline = quality["mean"] if isinstance(quality["mean"], (int, float)) else None
            key = f"{model_id}:retrieval_{'on' if enabled else 'off'}"
            result[key] = {
                "successful_requests": len(group), "task_quality": quality,
                "metrics": {name: _stats([item["telemetry"].get(name) for item in group if item["telemetry"].get(name) is not None]) for name in ("end_to_end_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")},
                "exact_input_tokens": _stats([item["input_tokens"] for item in group if item.get("input_tokens") is not None]),
                "retrieval_hit_rate": (sum(item["retrieval_hit"] for item in group) / len(group)) if group else None,
            }
        retrieved_quality = result[f"{model_id}:retrieval_on"]["task_quality"]["mean"]
        result[f"{model_id}:retrieval_on"]["quality_delta_vs_off"] = retrieved_quality - baseline if isinstance(retrieved_quality, (int, float)) and baseline is not None else None
    return result


def _record(
    experiment_id: str, model: dict[str, Any], sample: dict[str, Any], enabled: bool, trial: int, phase: str,
    prepared: str | None = None, preparation: dict[str, Any] | None = None, telemetry: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": "spider_dev",
        "model_id": model["model_id"], "model_file": model["path"], "quantization": model.get("quantization"),
        "prompt_id": f"spider:{sample['db_id']}:{sample['question'][:40]}", "phase": phase, "trial_number": trial,
        "retrieval_enabled": enabled, "retrieval_target_db_id": sample["db_id"],
    }
    if error:
        return record | {"success": False, "return_code": 1, "error": str(error), "measurement_kind": "measured_failure"}
    assert prepared is not None and preparation is not None and telemetry is not None
    docs = preparation.get("retrieval_documents", [])
    retrieval_hit = any(Path(document["path"]).stem.split(".")[0] == sample["db_id"] for document in docs) if enabled else False
    quality = evaluate_spider_sql(telemetry["answer"], sample["query"])
    return record | {
        "success": True, "return_code": 0, "measurement_kind": "measured" if phase == "measured" else "warmup",
        "prepared_prompt": prepared, "preparation": preparation, "retrieval_hit": retrieval_hit, "input_tokens": tokenize(prepared),
        "output_tokens": telemetry.get("generated_tokens"), "response": telemetry["answer"], "task_quality": quality, "telemetry": telemetry,
        "energy_note": "CPU-utilisation proxy only; not a hardware energy measurement",
    }


def run(model_ids: list[str] | None = None, limit: int = 5, trials: int = 3, max_output_tokens: int = 128) -> tuple[Path, Path]:
    if limit < 1 or trials < 1:
        raise ValueError("limit and trials must both be at least one")
    samples = load_spider_dev(limit)
    corpus_dir = build_schema_corpus(samples)
    available = [item for item in load_models() if item.get("available_locally") and "llama.cpp" in item.get("backends", [])]
    chosen_ids = model_ids or [item["model_id"] for item in available]
    models = [item for item in available if item["model_id"] in chosen_ids]
    missing = sorted(set(chosen_ids) - {item["model_id"] for item in models})
    if missing:
        raise FileNotFoundError(f"Requested local llama.cpp models are unavailable: {', '.join(missing)}")
    experiment_id = new_experiment_id("local_retrieval")
    records: list[dict[str, Any]] = []
    for model in models:
        for enabled in (False, True):
            config = configuration(model["model_id"], enabled, corpus_dir)
            try:
                ensure_runtime(model, config, force_restart=True)
                warmup_prompt, warmup_preparation = prepare_prompt("Reply with only the word ready.", config)
                warmup = generate(model, warmup_prompt, {"backend": "llama.cpp", "max_output_tokens": 8, "temperature": 0})
                records.append(_record(experiment_id, model, samples[0], enabled, 0, "warmup", warmup_prompt, warmup_preparation, warmup))
            except (InferenceError, RuntimeSwitchError) as error:
                records.append(_record(experiment_id, model, samples[0], enabled, 0, "warmup", error=error))
                continue
            for trial in range(1, trials + 1):
                for sample in samples:
                    prepared, preparation = prepare_prompt(prompt_for_retrieval(sample), config)
                    try:
                        telemetry = generate(model, prepared, {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                        records.append(_record(experiment_id, model, sample, enabled, trial, "measured", prepared, preparation, telemetry))
                    except (InferenceError, RuntimeSwitchError) as error:
                        records.append(_record(experiment_id, model, sample, enabled, trial, "measured", error=error))
    output_dir = ROOT / "logs" / "local_retrieval_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": "spider_dev", "sample_limit": limit, "trials": trials,
        "models": [model["model_id"] for model in models], "corpus_dir": corpus_dir,
        "corpus_provenance": "Generated locally from Spider schema metadata only; it contains no gold SQL.",
        "retrieval_method": "local lexical-overlap top-1 retrieval", "token_count_note": "Exact local token counts are null if /tokenize is unavailable.",
        "summary": summarize(records),
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a local Spider schema-retrieval study.")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    raw, summary = run(arguments.model_ids, arguments.limit, arguments.trials, arguments.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
