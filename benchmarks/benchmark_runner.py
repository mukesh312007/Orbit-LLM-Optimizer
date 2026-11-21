"""New append-only benchmark runner using the unified telemetry schema."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from benchmarks.evaluation_runner import evaluate_response
from benchmarks.telemetry import EnergyMetrics, TelemetryRecord, derive_performance, host_metadata, snapshot_resources
from inference.backends import InferenceBackend
from runtime_optimizer.types import RuntimeConfiguration, new_experiment_id


def run_trial(backend: InferenceBackend, config: RuntimeConfiguration, sample: dict[str, Any], trial_number: int = 1, experiment_id: str | None = None, timeout: float = 120) -> TelemetryRecord:
    experiment_id = experiment_id or new_experiment_id()
    result = backend.generate(config.model_id, sample["prompt"], config.max_output_tokens, config.temperature, timeout)
    quality = evaluate_response(sample, result.text)
    performance = derive_performance(0.0, None, result.latency_seconds, result.prompt_tokens, result.generated_tokens)
    return TelemetryRecord(experiment_id=experiment_id, timestamp=datetime.now(timezone.utc).isoformat(), model_id=config.model_id, configuration_id=config.configuration_id, dataset_id=sample["dataset"], prompt_id=sample["sample_id"], trial_number=trial_number, backend=result.backend, performance=performance, resources=snapshot_resources(), energy=EnergyMetrics(), quality={"automated_score": quality.get("score")}, hardware=host_metadata())
