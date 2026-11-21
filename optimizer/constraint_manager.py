from __future__ import annotations

from typing import Any


def violations(candidate: dict[str, Any], request: Any) -> list[str]:
    failed: list[str] = []
    comparisons = (("quality", request.minimum_quality, ">="), ("latency_seconds", request.maximum_latency_seconds, "<="), ("ram_mb", request.maximum_ram_mb, "<="), ("energy_joules", request.maximum_energy_joules, "<="))
    for key, limit, operator in comparisons:
        value = candidate.get("quality_lower_confidence_bound") if key == "quality" and candidate.get("quality_lower_confidence_bound") is not None else candidate.get(key)
        if limit is None:
            continue
        if value is None:
            failed.append(f"{key} is unavailable")
        elif (operator == ">=" and value < limit) or (operator == "<=" and value > limit):
            failed.append(f"{key} {value} violates {operator} {limit}")
    return failed


def feasible(candidates: list[dict[str, Any]], request: Any) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if not violations(candidate, request)]
