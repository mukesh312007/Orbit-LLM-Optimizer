from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Telemetry:
    experiment_id: str
    timestamp: str
    model_id: str
    configuration_id: str
    dataset_id: str | None = None
    prompt_id: str | None = None
    trial_number: int | None = None
    ttft_seconds: float | None = None
    tpot_ms: float | None = None
    end_to_end_seconds: float | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    generated_tokens: int | None = None
    process_memory_mb: float | None = None
    system_memory_percent: float | None = None
    cpu_percent: float | None = None
    gpu_utilization_percent: float | None = None
    energy_joules: float | None = None
    energy_source: str | None = None
    measurement_kind: str = "measured"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
