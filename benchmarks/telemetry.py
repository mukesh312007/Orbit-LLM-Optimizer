"""Unified telemetry contract and process/resource sampling.

Energy fields intentionally remain None unless a real external meter adapter is
provided. CPU/GPU utilization is never converted into energy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import platform
import time

import psutil


@dataclass
class PerformanceMetrics:
    ttft_seconds: float | None = None
    tpot_seconds: float | None = None
    end_to_end_seconds: float | None = None
    prompt_tokens: int | None = None
    generated_tokens: int | None = None
    prompt_tokens_per_second: float | None = None
    generation_tokens_per_second: float | None = None
    model_load_seconds: float | None = None


@dataclass
class ResourceMetrics:
    process_rss_mb: float | None = None
    process_tree_rss_mb: float | None = None
    system_ram_percent: float | None = None
    cpu_percent: float | None = None
    gpu_percent: float | None = None
    gpu_measurement_note: str | None = "GPU utilization unavailable"


@dataclass
class EnergyMetrics:
    joules: float | None = None
    joules_per_generated_token: float | None = None
    measurement_method: str = "unavailable"
    note: str = "No reliable energy counter or external power meter was configured."


@dataclass
class TelemetryRecord:
    experiment_id: str
    timestamp: str
    model_id: str
    configuration_id: str
    dataset_id: str | None
    prompt_id: str | None
    trial_number: int
    backend: str
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    energy: EnergyMetrics = field(default_factory=EnergyMetrics)
    quality: dict[str, float | None] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def host_metadata() -> dict[str, Any]:
    memory = psutil.virtual_memory()
    return {"os": platform.platform(), "cpu_logical_cores": psutil.cpu_count(), "ram_total_mb": round(memory.total / 1024**2, 2)}


def snapshot_resources(pid: int | None = None) -> ResourceMetrics:
    system = psutil.virtual_memory()
    result = ResourceMetrics(system_ram_percent=system.percent, cpu_percent=psutil.cpu_percent(interval=None))
    if pid is None:
        return result
    try:
        process = psutil.Process(pid)
        result.process_rss_mb = round(process.memory_info().rss / 1024**2, 2)
        descendants = process.children(recursive=True)
        result.process_tree_rss_mb = round((process.memory_info().rss + sum(p.memory_info().rss for p in descendants if p.is_running())) / 1024**2, 2)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return result


def derive_performance(start: float, first_token: float | None, end: float, prompt_tokens: int | None, generated_tokens: int | None) -> PerformanceMetrics:
    total = max(end - start, 0)
    ttft = first_token - start if first_token is not None else None
    generation_time = end - first_token if first_token is not None else None
    return PerformanceMetrics(
        ttft_seconds=ttft,
        tpot_seconds=(generation_time / generated_tokens if generation_time and generated_tokens else None),
        end_to_end_seconds=total,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prompt_tokens_per_second=(prompt_tokens / ttft if prompt_tokens and ttft else None),
        generation_tokens_per_second=(generated_tokens / generation_time if generated_tokens and generation_time else None),
    )
