from benchmarks.standard_quality_benchmark import summarize


def test_standard_quality_summary_separates_scored_and_unscored_trials():
    records = [
        {"phase": "measured", "success": True, "model_id": "qwen", "quality": {"score": 1.0}, "telemetry": {"end_to_end_seconds": 1.0}},
        {"phase": "measured", "success": True, "model_id": "qwen", "quality": {"score": None}, "telemetry": {"end_to_end_seconds": 2.0}},
        {"phase": "warmup", "success": True, "model_id": "qwen", "quality": {"score": 1.0}, "telemetry": {}},
    ]

    result = summarize(records)["qwen"]

    assert result["successful_trials"] == 2
    assert result["quality"]["mean"] == 1.0
    assert result["unscored_trials"] == 1
