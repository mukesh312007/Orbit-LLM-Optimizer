from optimizer import live_comparator
from optimizer.live_comparator import compare_on_spot


def test_live_quality_threshold_requires_task_valid_reference(monkeypatch):
    feasibility = {"task_type": "general", "available_mb": 10_000, "candidates": []}
    monkeypatch.setattr(live_comparator, "recommend_runtime", lambda *_, **__: feasibility)

    result = compare_on_spot("Explain databases", [], 32, 0, quality_threshold=0.8)

    assert result["selected"] is None
    assert "reference evidence" in result["reason"]


def test_sql_prompt_is_routed_to_spider_task_type():
    from optimizer.request_router import classify_task

    assert classify_task("Write a SQL SELECT query for this database") == "spider_text_to_sql"


def test_select_lowest_cost_threshold_acceptance_and_rejection():
    from optimizer.calibrated_profiles import select_lowest_cost

    mock_profiles = [
        {"configuration_id": "cfg_high_qual", "model_id": "m1", "quality": 0.95, "quality_lower_confidence_bound": 0.90, "latency_seconds": 2.0, "peak_ram_mb": 500, "generation_tokens_per_second": 20.0, "ttft_seconds": 0.5, "energy_proxy_joules": 10.0, "quality_per_estimated_watt": 0.5},
        {"configuration_id": "cfg_mid_qual", "model_id": "m2", "quality": 0.75, "quality_lower_confidence_bound": 0.70, "latency_seconds": 1.0, "peak_ram_mb": 300, "generation_tokens_per_second": 30.0, "ttft_seconds": 0.2, "energy_proxy_joules": 5.0, "quality_per_estimated_watt": 0.8},
        {"configuration_id": "cfg_low_qual", "model_id": "m3", "quality": 0.50, "quality_lower_confidence_bound": 0.45, "latency_seconds": 0.5, "peak_ram_mb": 200, "generation_tokens_per_second": 40.0, "ttft_seconds": 0.1, "energy_proxy_joules": 2.0, "quality_per_estimated_watt": 1.0},
    ]

    # At relaxed threshold 0.40: all 3 eligible, cfg_mid_qual or cfg_low_qual wins on cost
    outcome_low = select_lowest_cost(mock_profiles, 0.40)
    assert outcome_low["eligible_count"] == 3
    assert outcome_low["selected"] is not None

    # At strict threshold 0.80: only cfg_high_qual is eligible
    outcome_high = select_lowest_cost(mock_profiles, 0.80)
    assert outcome_high["eligible_count"] == 1
    assert outcome_high["selected"]["configuration_id"] == "cfg_high_qual"

    # At impossible threshold 0.98: all rejected
    outcome_impossible = select_lowest_cost(mock_profiles, 0.98)
    assert outcome_impossible["eligible_count"] == 0
    assert outcome_impossible["selected"] is None
    assert "No measured profile meets" in outcome_impossible["reason"]
