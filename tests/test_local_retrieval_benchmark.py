from benchmarks.local_retrieval_benchmark import prompt_for_retrieval, summarize


def _record(enabled: bool, score: float, hit: bool = False) -> dict:
    return {
        "phase": "measured", "success": True, "model_id": "qwen", "retrieval_enabled": enabled,
        "task_quality": {"score": score}, "retrieval_hit": hit, "input_tokens": 10,
        "telemetry": {"end_to_end_seconds": 1, "ttft_seconds": 0.2, "generation_tokens_per_second": 8, "peak_ram_mb": 100, "energy_proxy_joules": 5},
    }


def test_retrieval_prompt_identifies_its_database_without_gold_sql():
    prompt = prompt_for_retrieval({"db_id": "concert_singer", "question": "How many singers are there?"})
    assert "concert_singer" in prompt
    assert "SELECT COUNT" not in prompt


def test_retrieval_summary_reports_hit_rate_and_quality_delta():
    summary = summarize([_record(False, 0.0), _record(True, 1.0, True)])
    retrieval = summary["qwen:retrieval_on"]
    assert retrieval["retrieval_hit_rate"] == 1
    assert retrieval["quality_delta_vs_off"] == 1
