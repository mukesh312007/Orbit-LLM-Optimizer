from benchmarks.three_model_comparison import summarize


def test_summary_excludes_warmups_and_reports_statistics():
    records = [
        {"phase": "warmup", "success": True, "model_id": "qwen", "wall_clock_seconds": 9},
        {"phase": "measured", "success": True, "model_id": "qwen", "wall_clock_seconds": 2, "ttft_seconds": 1, "generation_tokens_per_second": 4, "peak_ram_mb": 100, "task_quality": {"score": 1}},
        {"phase": "measured", "success": True, "model_id": "qwen", "wall_clock_seconds": 4, "ttft_seconds": 2, "generation_tokens_per_second": 2, "peak_ram_mb": 120, "task_quality": {"score": 0}},
    ]
    summary = summarize(records)["qwen"]
    assert summary["measured_samples"] == 2
    assert summary["metrics"]["wall_clock_seconds"]["mean"] == 3
    assert summary["task_quality"]["median"] == 0.5
