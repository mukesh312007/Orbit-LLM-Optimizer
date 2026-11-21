from __future__ import annotations

from typing import Any


WEIGHTS = {
    "quality": {"quality": 0.70, "latency_seconds": -0.15, "ram_mb": -0.10, "energy_joules": -0.05},
    "latency": {"quality": 0.20, "latency_seconds": -0.60, "ram_mb": -0.15, "energy_joules": -0.05},
    "memory": {"quality": 0.20, "latency_seconds": -0.15, "ram_mb": -0.60, "energy_joules": -0.05},
    "energy": {"quality": 0.20, "latency_seconds": -0.15, "ram_mb": -0.10, "energy_joules": -0.55},
    "balanced": {"quality": 0.45, "latency_seconds": -0.25, "ram_mb": -0.20, "energy_joules": -0.10},
}


def score(candidate: dict[str, Any], preference: str) -> float:
    weights = WEIGHTS.get(preference, WEIGHTS["balanced"])
    quality = candidate.get("quality") or 0.0
    latency = candidate.get("latency_seconds") or float("inf")
    memory = candidate.get("ram_mb") or float("inf")
    energy = candidate.get("energy_joules")
    # Missing energy is excluded from its component rather than guessed.
    values = {"quality": quality, "latency_seconds": 1 / (1 + latency), "ram_mb": 1 / (1 + memory), "energy_joules": 0 if energy is None else 1 / (1 + energy)}
    return sum(weight * values[key] for key, weight in weights.items())
