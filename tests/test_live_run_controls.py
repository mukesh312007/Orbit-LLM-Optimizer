from optimizer.live_comparator import compare_on_spot


def test_full_live_dry_run_reports_limit_without_loading_models(monkeypatch):
    from optimizer import live_comparator
    model = {"model_id": "qwen", "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    feasibility = {"task_type": "general", "available_mb": 10000, "candidates": []}
    configurations = [{"model": model, "configuration": {"configuration_id": f"qwen:{index}"}, "required_mb": 1} for index in range(3)]
    monkeypatch.setattr(live_comparator, "recommend_runtime", lambda *_, **__: feasibility)
    monkeypatch.setattr(live_comparator, "full_live_candidates", lambda *_, **__: (configurations, []))
    result = compare_on_spot("hello", [model], 8, 0, full_configuration_matrix=True, max_live_candidates=2, dry_run=True)
    assert result["candidate_count"] == 3
    assert result["scheduled_candidate_count"] == 2
    assert result["deferred_candidate_count"] == 1


def test_matrix_candidate_metadata_does_not_require_duplicate_required_memory(monkeypatch):
    from optimizer import live_comparator

    model = {"model_id": "qwen", "path": "model.gguf"}
    candidate = {"model": model, "required_mb": 100, "memory_source": "matrix safety estimate"}
    feasibility = {"task_type": "general", "available_mb": 10_000, "candidates": [candidate]}
    run = {"model": model, "configuration": {"configuration_id": "qwen:t4:c2048"}, "candidate": candidate}
    monkeypatch.setattr(live_comparator, "recommend_runtime", lambda *_, **__: feasibility)
    monkeypatch.setattr(live_comparator, "full_live_candidates", lambda *_, **__: ([run], []))
    monkeypatch.setattr(live_comparator, "ensure_runtime", lambda *_: {"switched": False})
    monkeypatch.setattr(live_comparator, "prepare_prompt", lambda prompt, _: (prompt, {}))
    monkeypatch.setattr(live_comparator, "generate", lambda *_: {
        "answer": "hello", "end_to_end_seconds": 1.0, "ttft_seconds": 0.1,
        "generated_tokens": 1, "generation_tokens_per_second": 1.0, "peak_ram_mb": 100.0,
    })

    result = compare_on_spot("hello", [model], 8, 0, full_configuration_matrix=True)

    assert result["selected"] == candidate
