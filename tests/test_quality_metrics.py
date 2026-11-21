from benchmarks.quality_metrics.metrics import evaluate
from optimizer.task_quality import evaluate_spider_sql
from optimizer.weighted_selection import score_candidates
from runtime.prompt_pipeline import compress_prompt
from runtime.clients import _prompt_processing_metrics


def test_numeric_extracts_final_answer():
    result = evaluate({"evaluation_type": "numeric_exact", "reference_answer": "Work gives 42."}, "Therefore the answer is 42")
    assert result["score"] == 1


def test_rouge_is_not_fabricated():
    result = evaluate({"evaluation_type": "rouge", "reference_answer": "cats sit on mats"}, "cats sit")
    assert 0 < result["score"] < 1


def test_spider_sql_exact_match_ignores_case_and_spacing():
    result = evaluate_spider_sql("```sql\nSELECT *   FROM singer;\n```", "select * from singer")
    assert result["score"] == 1


def test_weighted_score_uses_quality_and_cost_metrics():
    candidates = [
        {"task_quality": 1.0, "generation_tokens_per_second": 10, "ttft_seconds": 2, "peak_ram_mb": 500, "energy_proxy_joules": 20, "quality_per_estimated_watt": 0.1},
        {"task_quality": 0.0, "generation_tokens_per_second": 20, "ttft_seconds": 1, "peak_ram_mb": 300, "energy_proxy_joules": 10, "quality_per_estimated_watt": 0.0},
    ]
    score_candidates(candidates)
    assert candidates[0]["weighted_score"] > candidates[1]["weighted_score"]


def test_proxy_joules_are_not_mislabeled_as_watts():
    candidates = [{"task_quality": 1.0, "energy_proxy_joules": 20, "quality_per_watt": 0.05}]
    report = score_candidates(candidates)
    assert "quality_per_estimated_watt" not in report["active_weights"]


def test_whitespace_prompt_compression_preserves_words():
    compressed, metadata = compress_prompt("  Find\n\n singers   now. ", "whitespace")
    assert compressed == "Find\nsingers now."
    assert metadata["output_characters"] < metadata["input_characters"]


def test_conservative_prompt_compression_preserves_code_indentation():
    compressed, _ = compress_prompt("\n\nUse SQL:\n\n\n    SELECT  *\n\n", "conservative")
    assert compressed == "Use SQL:\n\n    SELECT  *"


def test_prompt_timing_uses_server_measurement_when_available():
    metrics = _prompt_processing_metrics({"prompt_n": 100, "prompt_ms": 250, "prompt_per_second": 400}, {})
    assert metrics == {"prompt_tokens": 100, "prompt_processing_seconds": 0.25, "prompt_tokens_per_second": 400}
