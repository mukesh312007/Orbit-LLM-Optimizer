from optimizer.optimizer import OptimizationRequest, Optimizer


PROFILES = [
    {"configuration_id": "fast", "task_type": "reasoning", "quality": 0.70, "latency_seconds": 1.0, "ram_mb": 500, "energy_joules": None, "sample_count": 10},
    {"configuration_id": "accurate", "task_type": "reasoning", "quality": 0.90, "latency_seconds": 3.0, "ram_mb": 900, "energy_joules": None, "sample_count": 10},
]


def test_applies_hard_quality_constraint():
    result = Optimizer().recommend(PROFILES, OptimizationRequest(task_type="reasoning", minimum_quality=.8))
    assert result["selected"]["configuration_id"] == "accurate"
    assert result["threshold_selection"]["quality_threshold"] == 0.8


def test_reports_no_feasible_choice():
    result = Optimizer().recommend(PROFILES, OptimizationRequest(task_type="reasoning", maximum_latency_seconds=.1))
    assert not result["feasible"]
    assert result["alternatives"]
