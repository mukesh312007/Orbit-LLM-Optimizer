from optimizer.request_router import classify_task
from optimizer.task_quality import evaluate_task_quality
from optimizer.weighted_selection import score_candidates
from optimizer.live_comparator import compare_on_spot
import json


def test_general_knowledge_prompt_is_not_spider():
    assert classify_task("What players are there in Man City?") == "general_knowledge"


def test_arithmetic_is_independently_validated():
    assert classify_task("What is 12 + 7?") == "arithmetic"
    result = evaluate_task_quality("arithmetic", "The answer is 19", prompt="What is 12 + 7?", reference_answer="19")
    assert result["score"] == 1.0
    assert result["metric"] == "arithmetic_independent"
    assert result["factual_verification"] == "verified_by_deterministic_calculation"


def test_arithmetic_incorrect_answer_fails():
    result = evaluate_task_quality("arithmetic", "The answer is 20", prompt="What is 12 + 7?", reference_answer="19")
    assert result["score"] == 0.0


def test_spider_requires_reference_sql_but_general_knowledge_requires_reference_answer():
    sql = evaluate_task_quality("spider_text_to_sql", "SELECT 1")
    general = evaluate_task_quality("general_knowledge", "Manchester City has several players.")
    assert sql["score"] is None
    assert sql["status"] == "reference_sql_required"
    assert general["score"] is None
    assert general["status"] == "reference_answer_required"


def test_general_knowledge_uses_reference_evidence_and_discloses_limit():
    result = evaluate_task_quality(
        "general_knowledge",
        "Manchester City is a football club with players.",
        reference_answer="Manchester City is a football club with many players.",
    )
    assert 0 <= result["score"] <= 1
    assert result["metric"] == "semantic_reference_overlap"
    assert result["factual_verification"] == "unverified"


def test_unsupported_task_has_no_fabricated_quality():
    result = evaluate_task_quality("unsupported", "anything", reference_answer="anything")
    assert result["score"] is None
    assert result["status"] == "unsupported_task_type"


def test_missing_metrics_are_excluded_and_weights_renormalized():
    candidates = [
        {"task_quality": 0.9, "generation_tokens_per_second": 10, "ttft_seconds": None, "peak_ram_mb": 500, "energy_joules": None},
        {"task_quality": 0.8, "generation_tokens_per_second": 20, "ttft_seconds": None, "peak_ram_mb": 600, "energy_joules": None},
    ]
    report = score_candidates(candidates)
    assert "ttft_seconds" not in report["active_weights"]
    assert "energy_joules" not in report["active_weights"]
    assert set(report["active_weights"]) == {"task_quality", "generation_tokens_per_second", "peak_ram_mb"}
    assert all(candidate["weighted_score"] is not None for candidate in candidates)


def test_fast_inaccurate_candidate_cannot_beat_quality_eligible_candidate(monkeypatch):
    models = [
        {"model_id": "slow-accurate", "parameter_count_b": 1.5, "path": "models/a.gguf", "available_locally": True, "backends": ["llama.cpp"]},
        {"model_id": "fast-wrong", "parameter_count_b": 0.5, "path": "models/b.gguf", "available_locally": True, "backends": ["llama.cpp"]},
    ]
    candidates = [{"model": model, "required_mb": 100, "memory_source": "test", "feasible": True, "calibration": None} for model in models]
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {"task_type": "general_knowledge", "available_mb": 10000, "candidates": candidates})
    monkeypatch.setattr("optimizer.live_comparator.ensure_runtime", lambda *args, **kwargs: {"switched": False})
    monkeypatch.setattr("optimizer.live_comparator.generate", lambda model, *args, **kwargs: {
        "answer": "The answer is correct and verified." if model["model_id"] == "slow-accurate" else "I do not know.",
        "end_to_end_seconds": 3.0 if model["model_id"] == "slow-accurate" else 0.1,
        "ttft_seconds": 1.0, "generated_tokens": 10, "generation_tokens_per_second": 10.0,
        "peak_ram_mb": 500.0, "energy_proxy_watts": 20.0, "energy_proxy_joules": 20.0,
        "energy_measurement_kind": "estimated",
    })
    result = compare_on_spot(
        "What is the answer?", models, 32, 0, compare_all_models=True,
        task_type="general_knowledge", reference_answer="The answer is correct and verified.", quality_threshold=0.8,
    )
    assert result["selected"]["model"]["model_id"] == "slow-accurate"
    assert any(item["model_id"] == "fast-wrong" for item in result["quality_rejections"])
    rejected = next(item for item in result["live_comparison"] if item["candidate"]["model"]["model_id"] == "fast-wrong")
    assert rejected["weighted_score"] is None


def test_all_models_below_threshold_produce_no_suitable_configuration(monkeypatch):
    model = {"model_id": "wrong", "parameter_count_b": 0.5, "path": "models/wrong.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    candidate = {"model": model, "required_mb": 100, "memory_source": "test", "feasible": True, "calibration": None}
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {"task_type": "general_knowledge", "available_mb": 10000, "candidates": [candidate]})
    monkeypatch.setattr("optimizer.live_comparator.ensure_runtime", lambda *args, **kwargs: {"switched": False})
    monkeypatch.setattr("optimizer.live_comparator.generate", lambda *args, **kwargs: {
        "answer": "unrelated", "end_to_end_seconds": 0.1, "ttft_seconds": 0.01,
        "generated_tokens": 2, "generation_tokens_per_second": 20.0, "peak_ram_mb": 50.0,
    })
    result = compare_on_spot("Question", [model], 32, 0, compare_all_models=True, task_type="general_knowledge", reference_answer="verified answer", quality_threshold=0.8)
    assert result["selected"] is None
    assert "No suitable configuration found" in result["reason"]


def test_all_candidates_share_reference_and_are_stored_in_benchmark_history(monkeypatch, tmp_path):
    models = [
        {"model_id": "model-a", "parameter_count_b": 1.0, "path": "models/a.gguf", "available_locally": True, "backends": ["llama.cpp"]},
        {"model_id": "model-b", "parameter_count_b": 1.0, "path": "models/b.gguf", "available_locally": True, "backends": ["llama.cpp"]},
    ]
    candidates = [{"model": model, "required_mb": 100, "memory_source": "test", "feasible": True, "calibration": None} for model in models]
    monkeypatch.setattr("optimizer.live_comparator.LIVE_BENCHMARK_HISTORY", tmp_path / "history.jsonl")
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {"task_type": "general_knowledge", "available_mb": 10000, "candidates": candidates})
    monkeypatch.setattr("optimizer.live_comparator.ensure_runtime", lambda *args, **kwargs: {"switched": False})
    monkeypatch.setattr("optimizer.live_comparator.generate", lambda *args, **kwargs: {
        "answer": "The reference answer.", "end_to_end_seconds": 1.0, "ttft_seconds": 0.2,
        "generated_tokens": 4, "generation_tokens_per_second": 4.0, "peak_ram_mb": 100.0,
        "energy_measurement_kind": "unavailable",
    })
    compare_on_spot("Question", models, 32, 0, compare_all_models=True, task_type="general_knowledge", reference_answer="The reference answer.", quality_threshold=0.8)
    records = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {record["model_id"] for record in records} == {"model-a", "model-b"}
    assert {record["evaluation_method"] for record in records} == {"semantic_reference_overlap"}
    assert {record["reference_evidence"]["content"] for record in records} == {"The reference answer."}


def test_missing_quality_evidence_is_explicitly_performance_only(monkeypatch):
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {"task_type": "general_knowledge", "available_mb": 10000, "candidates": []})
    result = compare_on_spot("What is Orbit?", [], 32, 0)
    assert result["selection_mode"] == "performance_only"


def test_separate_local_reference_judge_is_used_for_all_candidates(monkeypatch, tmp_path):
    judge = {"model_id": "qwen2.5-1.5b-q4_k_m", "parameter_count_b": 1.5, "path": "models/judge.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    candidate = {"model_id": "qwen2.5-3b-q4_k_m", "parameter_count_b": 3.0, "path": "models/candidate.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    models = [judge, candidate]
    candidates = [{"model": model, "required_mb": 100, "memory_source": "test", "feasible": True, "calibration": None} for model in models]
    monkeypatch.setattr("optimizer.live_comparator.LIVE_BENCHMARK_HISTORY", tmp_path / "history.jsonl")
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {"task_type": "general_knowledge", "available_mb": 10000, "candidates": candidates})
    monkeypatch.setattr("optimizer.live_comparator.ensure_runtime", lambda *args, **kwargs: {"switched": False})
    seen = []

    def fake_generate(model, prompt, options):
        seen.append(model["model_id"])
        answer = "Paris is the capital of France." if model["model_id"] == judge["model_id"] else "Paris is the capital of France."
        return {"answer": answer, "end_to_end_seconds": 1.0, "ttft_seconds": 0.1, "generated_tokens": 5, "generation_tokens_per_second": 5.0, "peak_ram_mb": 100.0, "energy_measurement_kind": "unavailable"}

    monkeypatch.setattr("optimizer.live_comparator.generate", fake_generate)
    result = compare_on_spot(
        "What is the capital of France?", models, 32, 0, compare_all_models=True,
        task_type="general_knowledge", quality_threshold=0.8,
        use_model_reference=True, reference_model_id=judge["model_id"],
    )
    assert seen[0] == judge["model_id"]
    assert result["reference_evidence"]["source"] == "model_generated"
    assert result["reference_evidence"]["model_id"] == judge["model_id"]
    assert all(entry["task_quality"] == 1.0 for entry in result["live_comparison"])
    records = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {record["reference_evidence"]["content"] for record in records} == {"Paris is the capital of France."}
    assert {record["reference_evidence"]["model_id"] for record in records} == {judge["model_id"]}
