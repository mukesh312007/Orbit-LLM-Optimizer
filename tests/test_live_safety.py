from optimizer import live_comparator
from optimizer.live_comparator import compare_on_spot


def test_broad_live_comparison_keeps_only_safe_candidates(monkeypatch):
    safe_model = {"model_id": "safe", "path": "safe.gguf"}
    unsafe_model = {"model_id": "unsafe", "path": "unsafe.gguf"}
    feasibility = {
        "task_type": "general", "available_mb": 2000,
        "candidates": [
            {"model": safe_model, "feasible": True},
            {"model": unsafe_model, "feasible": False},
        ],
    }
    monkeypatch.setattr(live_comparator, "recommend_runtime", lambda *_, **__: feasibility)

    result = compare_on_spot("hello", [safe_model, unsafe_model], 8, 0, compare_all_models=True, dry_run=True)

    assert result["candidate_count"] == 1
