from api import app as api_module
from api.app import GenerateBody, OptimizeBody, generate_request, health, models, optimize, readiness, store_status


def test_health_and_models_endpoints():
    health_res = health()
    assert health_res["status"] == "ok"
    assert "backends" in health_res
    models_res = models()
    assert isinstance(models_res, list)


def test_optimize_endpoint_avoids_fake_quality():
    body = OptimizeBody(task_type="all", preference="latency")
    res = optimize(body)
    assert "selected" in res


def test_readiness_endpoint_reports_measurement_limits(monkeypatch):
    monkeypatch.setattr("api.app.full_live_candidates", lambda _: ([], [{"reason": "RAM safety skip"}]))
    res = readiness()
    assert res["local_only"] is True
    assert "Physical energy" in res["energy_measurement_note"]


def test_live_api_selection_is_persisted_with_private_telemetry(monkeypatch):
    decision = {
        "selected": {"model": {"model_id": "qwen"}}, "configuration": {"configuration_id": "qwen:t4"},
        "task_type": "general", "winner_index": 0, "live_comparison": [{"draft": "answer"}],
    }
    monkeypatch.setattr(api_module, "compare_on_spot", lambda *args, **kwargs: decision)
    monkeypatch.setattr(api_module, "record_selection", lambda selected, telemetry: {"status": "persisted", "answer": telemetry["answer"]})

    body = GenerateBody(prompt="hello", live_compare=True)
    res = generate_request(body)

    assert res["answer"] == "answer"
    assert res["telemetry_persistence"]["status"] == "persisted"


def test_generate_launches_the_measured_recommendation_before_inference(monkeypatch):
    model = {"model_id": "qwen-test", "path": "models/test.gguf", "backends": ["llama.cpp"]}
    profile = {"model_id": "qwen-test", "sample_count": 3, "quality": 0.8, "latency_seconds": 1.0, "ram_mb": 500.0, "configuration": {"configuration_id": "qwen-test:t2:c1024", "backend": "llama.cpp", "threads": 2, "context_size": 1024, "cache": "server-default", "retrieval": {"enabled": False}}}
    monkeypatch.setattr("api.app.historical_profiles", lambda: [profile])
    monkeypatch.setattr("api.app.get_model", lambda _: model)
    monkeypatch.setattr("api.app.configuration_is_safe", lambda *_: (True, "safe", 1.0))
    monkeypatch.setattr("api.app.ensure_runtime", lambda *args: {"switched": True})
    monkeypatch.setattr("api.app.prepare_prompt", lambda prompt, _: (f"prepared:{prompt}", {"retrieval_enabled": False}))
    monkeypatch.setattr("api.app.generate", lambda *args, **kwargs: {"answer": "ok", "measurement_kind": "measured"})
    monkeypatch.setattr("api.app.record_selection", lambda *_: {"status": "persisted"})
    body = GenerateBody(prompt="hello", task_type="all")
    res = generate_request(body)
    assert res["answer"] == "ok"
    assert res["configuration"]["threads"] == 2
    assert res["recommendation_is_historical"] is True


def test_store_status_endpoint():
    res = store_status()
    assert "primary_store" in res
    assert res["primary_store"]["primary_store"] == "jsonl"
    assert "optional_mysql_mirror" in res
