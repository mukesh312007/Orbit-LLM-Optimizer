from optimizer.live_comparator import _model_balanced_runs, compare_on_spot, parse_judgement


def test_parses_local_judge_choice_only_when_valid():
    assert parse_judgement("2", 2) == 1
    assert parse_judgement("WINNER: 1", 2) == 0
    assert parse_judgement("WINNER: 3", 2) is None
    assert parse_judgement("WINNER: TIE", 2) is None
    assert parse_judgement("I cannot choose", 2) is None


def test_matrix_sample_covers_each_model_before_repeating_a_model():
    runs = [
        {
            "model": {"model_id": model_id},
            "configuration": {"configuration_id": f"{model_id}:config-{configuration}"},
        }
        for model_id in ("a", "b", "c")
        for configuration in (1, 2)
    ]

    selected = _model_balanced_runs(runs, 3)

    assert [entry["model"]["model_id"] for entry in selected] == ["a", "b", "c"]


def test_compare_on_spot_separates_proxy_joules_and_estimated_watts(monkeypatch):
    model = {"model_id": "qwen2.5-0.5b-instruct-q4_k_m", "parameter_count_b": 0.5, "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    candidate = {"model": model, "required_mb": 500, "memory_source": "test", "feasible": True, "calibration": None}
    monkeypatch.setattr("optimizer.live_comparator.recommend_runtime", lambda *args, **kwargs: {
        "task_type": "spider_text_to_sql", "available_mb": 10000, "candidates": [candidate],
    })
    monkeypatch.setattr("optimizer.live_comparator.ensure_runtime", lambda *args, **kwargs: {"switched": False})
    monkeypatch.setattr("optimizer.live_comparator.generate", lambda *args, **kwargs: {
        "answer": "SELECT * FROM t;",
        "end_to_end_seconds": 1.0,
        "ttft_seconds": 0.2,
        "generated_tokens": 10,
        "generation_tokens_per_second": 10.0,
        "peak_ram_mb": 500.0,
        "energy_proxy_watts": 20.0,
        "energy_proxy_joules": 20.0,
        "energy_proxy_method": "CPU proxy",
        "energy_measurement_kind": "estimated",
    })
    monkeypatch.setattr("optimizer.live_comparator.evaluate_spider_sql", lambda *args, **kwargs: {"score": 1.0, "match": True})

    result = compare_on_spot("SELECT query", [model], 64, 0.0, reference_sql="SELECT * FROM t;")
    entry = result["live_comparison"][0]
    assert "quality_per_proxy_joule" in entry
    assert entry["quality_per_proxy_joule"] == 1.0 / 20.0
    assert "quality_per_estimated_watt" in entry
    assert entry["quality_per_estimated_watt"] == 1.0 / 20.0
    assert "energy_unit_note" in entry
