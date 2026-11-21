from benchmarks.context_thread_calibration import safe_to_run, summarize


def test_safety_check_skips_insufficient_memory(monkeypatch, tmp_path):
    from benchmarks import context_thread_calibration as calibration
    monkeypatch.setattr(calibration, "required_memory_mb", lambda *_: 100.0)
    safe, _ = safe_to_run({"model_id": "test", "path": "model.gguf"}, 4096, available_mb=1)
    assert safe is False


def test_summary_recommends_completed_best_quality_configuration():
    records = [
        {"phase": "measured", "success": True, "configuration": {"configuration_id": "a"}, "quality": {"score": 0}, "telemetry": {"end_to_end_seconds": 1, "generation_tokens_per_second": 9}},
        {"phase": "measured", "success": True, "configuration": {"configuration_id": "b"}, "quality": {"score": 1}, "telemetry": {"end_to_end_seconds": 2, "generation_tokens_per_second": 2}},
    ]
    assert summarize(records)["recommendation"]["configuration"]["configuration_id"] == "b"
