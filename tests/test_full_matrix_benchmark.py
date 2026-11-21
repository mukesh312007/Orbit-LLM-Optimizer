import pytest

from benchmarks.full_matrix_benchmark import resident_configurations, run, summarize


def test_full_matrix_summary_separates_measured_trials_from_warmups_and_skips():
    configuration = {"configuration_id": "qwen:t4:c1024"}
    records = [
        {"phase": "warmup", "success": True, "configuration": configuration, "model_id": "qwen"},
        {"phase": "safety_skip", "success": False, "configuration": configuration, "model_id": "qwen"},
        {"phase": "measured", "success": True, "configuration": configuration, "model_id": "qwen", "quantization": "Q4", "task_quality": {"score": 1.0}, "telemetry": {"end_to_end_seconds": 2.0, "ttft_seconds": 1.0, "generation_tokens_per_second": 8.0, "peak_ram_mb": 500.0, "energy_proxy_joules": 20.0}},
    ]
    result = summarize(records)
    assert result["success_count"] == 1
    assert result["safety_skip_count"] == 1
    assert result["deferred_count"] == 0
    assert result["configurations"]["qwen:t4:c1024"]["task_quality"]["mean"] == 1.0


def test_resident_configurations_keep_runtime_fixed_and_vary_prompt_features():
    configs = resident_configurations(
        {"model_id": "qwen"}, {"threads": 4, "context_size": 2048, "gpu_layers": 0}, "logs/corpus",
    )
    assert len(configs) == 6
    assert all(config["resident_reuse"] for config in configs)
    assert {config["threads"] for config in configs} == {4}
    assert {config["retrieval"]["enabled"] for config in configs} == {False, True}


def test_baseline_only_cannot_be_combined_with_resident_only():
    with pytest.raises(ValueError, match="cannot be combined"):
        run(limit=1, trials=1, resident_only=True, baseline_only=True)
