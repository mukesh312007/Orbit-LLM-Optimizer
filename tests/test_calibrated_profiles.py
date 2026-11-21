from optimizer.calibrated_profiles import _normalise_controlled_record, aggregate_records, aggregate_standard_quality_records, select_lowest_cost


def _record(configuration_id: str, quality: float, latency: float, ram: float, trial: int = 1) -> dict:
    return {
        "phase": "measured", "success": True, "model_id": configuration_id.split(":")[0], "quantization": "Q4",
        "configuration": {"configuration_id": configuration_id}, "task_quality": {"score": quality},
        "trial_number": trial, "prompt_id": f"prompt-{trial}",
        "quality_per_watt": quality / 20,
        "telemetry": {"end_to_end_seconds": latency, "ttft_seconds": latency / 2, "generation_tokens_per_second": 10 / latency, "peak_ram_mb": ram, "energy_proxy_joules": latency * 10},
    }


def test_profile_aggregation_requires_repeated_measured_trials():
    records = [_record("fast", 0.7, 1, 500), _record("fast", 0.7, 1, 500), _record("slow", 0.9, 2, 800)]
    profiles = aggregate_records(records, minimum_samples=2)
    assert [profile["configuration_id"] for profile in profiles] == ["fast"]
    assert profiles[0]["quality"] == 0.7


def test_quality_gate_selects_lowest_cost_eligible_profile():
    profiles = aggregate_records([
        _record("fast", 0.8, 1, 500, 1), _record("fast", 0.8, 1, 500, 2), _record("fast", 0.8, 1, 500, 3),
        _record("accurate", 0.9, 2, 800, 1), _record("accurate", 0.9, 2, 800, 2), _record("accurate", 0.9, 2, 800, 3),
    ], minimum_samples=3)
    selection = select_lowest_cost(profiles, 0.8)
    assert selection["selected"]["configuration_id"] == "fast"
    assert selection["eligible_count"] == 2


def test_standard_quality_records_become_task_specific_profiles():
    records = [
        {
            "phase": "measured", "success": True, "model_id": "qwen", "dataset": "gsm8k", "task_type": "reasoning",
            "configuration": {"configuration_id": "qwen:t4"}, "quality": {"score": 1.0},
            "telemetry": {"end_to_end_seconds": 1.0, "ttft_seconds": 0.2, "generation_tokens_per_second": 10.0, "peak_ram_mb": 500.0},
        }
        for _ in range(3)
    ]
    profiles = aggregate_standard_quality_records(records, minimum_samples=3)
    assert profiles[0]["task_type"] == "reasoning"
    assert profiles[0]["quality_lower_confidence_bound"] == 1.0


def test_three_model_rows_are_normalised_into_selector_profiles():
    raw = {
        "phase": "measured", "success": True, "model_id": "qwen-3b",
        "runtime_configuration": {"configuration_id": "qwen-3b:t4:c2048:cpu"},
        "task_quality": {"score": 1.0}, "wall_clock_seconds": 2.0,
        "ttft_seconds": 0.5, "generation_tokens_per_second": 10.0, "peak_ram_mb": 2000.0,
        "energy": {"joules": 20.0, "kind": "estimated_proxy"},
    }
    records = [_normalise_controlled_record(raw, "three_model") for _ in range(3)]
    profiles = aggregate_records(records, minimum_samples=3)
    assert profiles[0]["configuration_id"] == "qwen-3b:t4:c2048:cpu"
    assert profiles[0]["energy_joules"] is None
    assert profiles[0]["energy_proxy_joules"] == 20.0


def test_cache_modes_remain_distinct_selector_configurations():
    raw = _record("qwen:t4:c2048:cache-study", 0.8, 1.0, 500.0)
    raw["cache_mode"] = "enabled"
    normalised = _normalise_controlled_record(raw, "cache")
    assert normalised["configuration"]["configuration_id"].endswith(":cache-enabled")
