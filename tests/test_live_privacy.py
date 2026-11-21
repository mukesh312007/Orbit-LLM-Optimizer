from optimizer import request_router
from types import SimpleNamespace


def test_live_selection_redacts_response_text_by_default(monkeypatch, tmp_path):
    history = tmp_path / "history.jsonl"
    monkeypatch.setattr(request_router, "SELECTION_HISTORY", history)
    monkeypatch.setattr(request_router, "mirror_live_selection", lambda *_: {"status": "unavailable"})
    monkeypatch.setattr(request_router, "settings", SimpleNamespace(retain_live_response_text=False))

    request_router.record_selection(
        {"selected": {"model": {"model_id": "local"}}, "configuration": {"configuration_id": "cfg"}, "task_type": "general"},
        {"answer": "sensitive response", "end_to_end_seconds": 1.0},
    )

    assert "sensitive response" not in history.read_text(encoding="utf-8")
    assert '"response_text_retained": false' in history.read_text(encoding="utf-8")
