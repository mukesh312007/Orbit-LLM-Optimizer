from benchmarks.general_retrieval_evaluation import evaluate_answer, load_evaluation, summarize


def test_general_retrieval_suite_has_requested_local_scale():
    samples = load_evaluation()
    assert len(samples) == 10
    assert len({sample["source_document"] for sample in samples}) >= 3


def test_retrieval_answer_evaluation_requires_expected_source_and_terms():
    sample = {"reference_answer": "The dashboard uses 8502 and llama.cpp uses 9931.", "required_terms": ["8502", "9931"], "source_document": "datasets/retrieval/local_first.md"}
    result = evaluate_answer("The ports are 8502 and 9931.", sample, [{"path": "datasets/retrieval/local_first.md", "score": 3}])
    assert result["correctness"] == 1
    assert result["support"] == 1


def test_general_retrieval_summary_reports_delta():
    base = {"phase": "measured", "success": True, "model_id": "qwen", "retrieval_enabled": False, "input_tokens": 5, "evaluation": {"correctness": 0.0, "relevance": 0.0, "support": 0.0, "source_retrieved": False}, "telemetry": {"end_to_end_seconds": 1, "ttft_seconds": 0.2, "generation_tokens_per_second": 8, "peak_ram_mb": 500, "energy_proxy_joules": 2}}
    with_retrieval = base | {"retrieval_enabled": True, "evaluation": {"correctness": 1.0, "relevance": 1.0, "support": 1.0, "source_retrieved": True}}
    assert summarize([base, with_retrieval])["qwen:retrieval_on"]["correctness_delta_vs_off"] == 1.0
