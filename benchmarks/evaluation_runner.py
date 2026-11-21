from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from benchmarks.quality_metrics import evaluate
from core.ids import new_experiment_id
from hardware.monitoring import sample_resources
from runtime.clients import generate


def run_evaluation(samples: list[dict[str, Any]], model: dict[str, Any], configuration: dict[str, Any], trial_number: int = 1) -> list[dict[str, Any]]:
    """Execute a supplied normalized dataset and retain automation/telemetry separately."""
    experiment_id = new_experiment_id("eval")
    results: list[dict[str, Any]] = []
    for sample in samples:
        before = sample_resources()
        response = generate(model, sample["prompt"], configuration)
        quality = evaluate(sample, response["answer"])
        after = sample_resources()
        results.append({"experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "model_id": model["model_id"], "configuration_id": configuration.get("configuration_id", "ad-hoc"), "dataset_id": sample["dataset"], "prompt_id": sample["sample_id"], "trial_number": trial_number, "answer": response["answer"], "quality": quality, "performance": response, "resources": {"before": before.__dict__, "after": after.__dict__}, "energy": None})
    return results
