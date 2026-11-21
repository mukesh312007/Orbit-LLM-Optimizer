"""Transparent, normalized multi-objective selection for local candidates."""
from __future__ import annotations

from typing import Any


DEFAULT_WEIGHTS = {
    "task_quality": 0.40,
    "generation_tokens_per_second": 0.15,
    "ttft_seconds": 0.10,
    "peak_ram_mb": 0.10,
    "energy_joules": 0.10,
}
BENEFIT_METRICS = {
    "task_quality",
    "generation_tokens_per_second",
    "quality_per_estimated_watt",
    "quality_per_proxy_joule",
    "quality_per_measured_watt",
    "quality_per_watt",
}
COST_METRICS = {"ttft_seconds", "peak_ram_mb", "energy_joules"}


def _normalise(values: list[float], benefit: bool) -> list[float]:
    low, high = min(values), max(values)
    if high == low:
        return [1.0] * len(values)
    return [(value - low) / (high - low) if benefit else (high - value) / (high - low) for value in values]


def score_candidates(candidates: list[dict[str, Any]], weights: dict[str, float] | None = None) -> dict[str, Any]:
    """Return metric-normalized scores; unavailable metrics are excluded, not guessed."""
    if weights:
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown:
            raise ValueError(f"Unsupported selection metrics: {', '.join(sorted(unknown))}")
    active_weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    normalized = [dict() for _ in candidates]
    available_weights: dict[str, float] = {}
    for metric, weight in active_weights.items():
        values = [candidate.get(metric) for candidate in candidates]
        if weight <= 0 or any(value is None for value in values):
            continue
        numeric = [float(value) for value in values if value is not None]
        for row, value in zip(normalized, _normalise(numeric, metric in BENEFIT_METRICS), strict=True):
            row[metric] = value
        available_weights[metric] = weight

    total_weight = sum(available_weights.values())
    for candidate, row in zip(candidates, normalized, strict=True):
        candidate["normalized_metrics"] = row
        candidate["weighted_score"] = (
            sum(row[metric] * available_weights[metric] for metric in row) / total_weight
            if total_weight else None
        )
    excluded_metrics = {
        metric: "unavailable for at least one candidate" if any(candidate.get(metric) is None for candidate in candidates) else "weight is zero"
        for metric, weight in active_weights.items()
        if weight > 0 and metric not in available_weights
    }
    return {"weights": active_weights, "active_weights": available_weights, "excluded_metrics": excluded_metrics}
