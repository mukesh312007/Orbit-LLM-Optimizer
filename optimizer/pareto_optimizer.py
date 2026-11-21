from __future__ import annotations

from typing import Any


def pareto_frontier(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Higher quality, lower measured cost. Null energy does not dominate a measured energy value."""
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            if other is candidate:
                continue
            keys = [("quality", True), ("latency_seconds", False), ("ram_mb", False)]
            comparable = all(candidate.get(key) is not None and other.get(key) is not None for key, _ in keys)
            if comparable and all((other[key] >= candidate[key] if maximize else other[key] <= candidate[key]) for key, maximize in keys) and any((other[key] > candidate[key] if maximize else other[key] < candidate[key]) for key, maximize in keys):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier
