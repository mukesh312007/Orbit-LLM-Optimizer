from benchmarks.import_jsonl_telemetry import canonicalize


def test_importer_canonicalizes_legacy_context_benchmark_record():
    record = {
        "experiment_id": "exp", "phase": "measured", "success": True, "model_id": "qwen2.5-0.5b-q4_k_m",
        "configuration": {"configuration_id": "qwen:t4", "backend": "llama.cpp"}, "prompt_id": "db", "trial": 1,
        "quality": {"score": 1.0, "metric": "sql"}, "telemetry": {"end_to_end_seconds": 1.0},
    }
    result = canonicalize(record)
    assert result["trial_number"] == 1
    assert result["task_quality"]["score"] == 1.0


def test_importer_skips_nonmeasured_failed_records():
    assert canonicalize({"phase": "safety_skip", "success": False}) is None
