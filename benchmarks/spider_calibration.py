"""Offline Spider calibration for quality-aware local runtime selection."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.model_registry import load_models
from core.settings import ROOT
from optimizer.configuration_catalog import baseline_configuration, benchmark_configurations
from optimizer.task_quality import evaluate_spider_sql
from runtime.clients import generate
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import ensure_runtime

SPIDER_ROOT = ROOT / "spider_data" / "spider_data"
CALIBRATION_PATH = ROOT / "logs" / "spider_calibration.jsonl"


def load_spider_dev(limit: int) -> list[dict[str, Any]]:
    records = json.loads((SPIDER_ROOT / "dev.json").read_text(encoding="utf-8"))
    return records[:limit]


def schema_for(database_id: str, max_characters: int = 2_400) -> str:
    """Compact schema context so calibration remains valid at a 1024-token context."""
    tables = next(item for item in json.loads((SPIDER_ROOT / "tables.json").read_text(encoding="utf-8")) if item["db_id"] == database_id)
    rows = [
        f"{table}({', '.join(column for (table_index, column) in tables['column_names_original'] if table_index == index)})"
        for index, table in enumerate(tables["table_names_original"])
    ]
    schema = "\n".join(rows)
    return schema[:max_characters]


def prompt_for(sample: dict[str, Any]) -> str:
    return (
        "Write one SQL SELECT query and nothing else.\n\n"
        f"Database schema:\n{schema_for(sample['db_id'])}\n\n"
        f"Question: {sample['question']}"
    )


def calibrate(limit: int = 50, max_output_tokens: int = 128, all_configurations: bool = False) -> list[dict[str, Any]]:
    if not (SPIDER_ROOT / "dev.json").exists():
        raise FileNotFoundError(f"Spider dev split not found: {SPIDER_ROOT / 'dev.json'}")
    samples = load_spider_dev(limit)
    records: list[dict[str, Any]] = []
    for model in (model for model in load_models() if model.get("available_locally") and "llama.cpp" in model.get("backends", [])):
        configurations = benchmark_configurations(model) if all_configurations else [baseline_configuration(model)]
        for configuration in configurations:
            ensure_runtime(model, configuration)
            outcomes: list[dict[str, Any]] = []
            for sample in samples:
                prepared_prompt, preparation = prepare_prompt(prompt_for(sample), configuration)
                response = generate(model, prepared_prompt, {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                quality = evaluate_spider_sql(response["answer"], sample["query"])
                outcomes.append({"quality": quality, "telemetry": response, "preparation": preparation})
            quality_values = [outcome["quality"]["score"] for outcome in outcomes]
            record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "dataset": "spider_dev", "sample_count": len(outcomes),
                "model_id": model["model_id"], "configuration": configuration,
                "task_quality": sum(quality_values) / len(quality_values) if quality_values else None,
                "mean_tokens_per_second": _mean(outcomes, "generation_tokens_per_second"),
                "mean_ttft_seconds": _mean(outcomes, "ttft_seconds"),
                "mean_peak_ram_mb": _mean(outcomes, "peak_ram_mb"),
                "mean_energy_proxy_joules": _mean(outcomes, "energy_proxy_joules"),
                "mean_quality_per_watt": _mean_quality_per_watt(outcomes),
                "prompt_preparation": outcomes[0]["preparation"] if outcomes else {},
                "measurement_kind": "measured_spider_exact_match_calibration",
            }
            records.append(record)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CALIBRATION_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


def _mean(outcomes: list[dict[str, Any]], key: str) -> float | None:
    values = [outcome["telemetry"].get(key) for outcome in outcomes if outcome["telemetry"].get(key) is not None]
    return sum(values) / len(values) if values else None


def _mean_quality_per_watt(outcomes: list[dict[str, Any]]) -> float | None:
    values = [
        outcome["quality"]["score"] / outcome["telemetry"]["energy_proxy_watts"]
        for outcome in outcomes
        if outcome["telemetry"].get("energy_proxy_watts")
    ]
    return sum(values) / len(values) if values else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--all-configurations", action="store_true")
    args = parser.parse_args()
    for result in calibrate(args.limit, args.max_output_tokens, args.all_configurations):
        print(json.dumps(result, indent=2))
