from __future__ import annotations

import time
from dataclasses import dataclass

import psutil


@dataclass
class ResourceSnapshot:
    process_memory_mb: float | None
    process_tree_memory_mb: float | None
    system_memory_percent: float
    cpu_percent: float
    gpu_utilization_percent: float | None = None
    energy_joules: float | None = None
    energy_source: str | None = None


def sample_resources(pid: int | None = None) -> ResourceSnapshot:
    """Sample observable resources; energy remains null without a supported meter/counter."""
    process_memory = tree_memory = None
    if pid:
        try:
            process = psutil.Process(pid)
            children = process.children(recursive=True)
            values = [process.memory_info().rss, *(child.memory_info().rss for child in children)]
            tree_memory = sum(values) / 1024**2
            process_memory = values[0] / 1024**2
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return ResourceSnapshot(process_memory, tree_memory, psutil.virtual_memory().percent, psutil.cpu_percent(interval=0.1))


class ExternalEnergyMeter:
    """Extension point for a USB/wall-meter adapter; no energy values are fabricated."""
    def read_joules(self) -> float | None:
        return None
