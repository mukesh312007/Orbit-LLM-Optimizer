from benchmarks.cache_prompt_experiments import cache_controls_supported, summarize


def test_cache_controls_require_an_explicit_enable_and_disable_flag():
    assert cache_controls_supported("--cache-prompt, --no-cache-prompt") is True
    assert cache_controls_supported("--cache-prompt") is False


def test_cache_summary_keeps_cold_and_warm_results_separate():
    records = [
        {"success": True, "cache_mode": "enabled", "request_state": "cold", "task_quality": {"score": 1}, "telemetry": {"end_to_end_seconds": 2, "ttft_seconds": 1, "generation_tokens_per_second": 4, "peak_ram_mb": 500, "energy_proxy_joules": 20}},
        {"success": True, "cache_mode": "enabled", "request_state": "warm_repeat", "task_quality": {"score": 1}, "telemetry": {"end_to_end_seconds": 1, "ttft_seconds": 0.2, "generation_tokens_per_second": 4, "peak_ram_mb": 500, "energy_proxy_joules": 10}},
    ]
    summary = summarize(records)
    assert summary["enabled:cold"]["metrics"]["ttft_seconds"]["mean"] == 1
    assert summary["enabled:warm_repeat"]["metrics"]["ttft_seconds"]["mean"] == 0.2
