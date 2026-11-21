from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from optimizer.candidate_generator import candidates_from_profiles
from optimizer.calibrated_profiles import select_lowest_cost
from optimizer.constraint_manager import feasible, violations
from optimizer.pareto_optimizer import pareto_frontier
from optimizer.recommendation_engine import explain
from optimizer.scoring_engine import score


@dataclass(frozen=True)
class OptimizationRequest:
    task_type: str
    minimum_quality: float | None = None
    maximum_latency_seconds: float | None = None
    maximum_ram_mb: float | None = None
    maximum_energy_joules: float | None = None
    preference: str = "balanced"


class Optimizer:
    def recommend(self, profiles: list[dict[str, Any]], request: OptimizationRequest) -> dict[str, Any]:
        candidates = candidates_from_profiles(profiles, request.task_type)
        eligible = feasible(candidates, request)
        frontier = pareto_frontier(eligible or candidates)
        threshold_selection = None
        if request.minimum_quality is not None and eligible:
            threshold_selection = select_lowest_cost(eligible, request.minimum_quality, request.preference)
            selected = threshold_selection.get("selected")
            ranked = [selected] if selected else []
        else:
            ranked = sorted(eligible, key=lambda item: score(item, request.preference), reverse=True)
            selected = ranked[0] if ranked else None
        alternatives = [{"configuration_id": item.get("configuration_id"), "violations": violations(item, request)} for item in candidates if item is not selected]
        explanation = (
            threshold_selection["reason"] if threshold_selection and selected
            else explain(selected, request.preference)
        )
        selection_mode = "quality_constrained" if request.minimum_quality is not None else "performance_only"
        if request.minimum_quality is not None and not selected:
            explanation = "Quality-constrained selection unavailable: no profile satisfies the requested quality and performance constraints."
        return {"selected": selected, "feasible": bool(selected), "pareto_frontier": frontier, "alternatives": alternatives, "explanation": explanation, "request": asdict(request), "threshold_selection": threshold_selection, "selection_mode": selection_mode}
