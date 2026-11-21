from types import SimpleNamespace
import json

import optimizer.request_router as request_router
from optimizer.request_router import classify_task, recommend_runtime, selection_caption


def test_classifies_common_task_types():
    assert classify_task("Write a Python function") == "coding"
    assert classify_task("Solve this math equation") == "reasoning"
    assert classify_task("Summarize this paragraph") == "summarization"


def test_selection_caption_discloses_estimated_quality():
    decision = {
        "selected": {"model": {"model_id": "qwen-test"}, "memory_source": "estimated from GGUF file size"},
        "configuration": {"threads": 4, "context_size": 2048},
    }
    assert "estimate" in selection_caption(decision).lower()


def test_uses_already_loaded_model_when_ram_reserve_excludes_all_candidates(monkeypatch):
    model = {
        "model_id": "qwen-resident", "path": "models/example.gguf",
        "available_locally": True, "backends": ["llama.cpp"],
    }
    monkeypatch.setattr(request_router.psutil, "virtual_memory", lambda: SimpleNamespace(available=256 * 1024**2))
    monkeypatch.setattr(request_router, "_active_orbit_runtime", lambda: ("qwen-resident", 512.0))
    monkeypatch.setattr(request_router, "_measured_memory", lambda _: 900.0)

    decision = recommend_runtime("Hi", [model])

    assert decision["selected"]["model"]["model_id"] == "qwen-resident"
    assert decision["resident_fallback"] is True


def test_configurable_safety_reserve_allows_model_admission(monkeypatch):
    model = {
        "model_id": "qwen-test", "path": "models/example.gguf",
        "available_locally": True, "backends": ["llama.cpp"],
    }
    monkeypatch.setattr(request_router.psutil, "virtual_memory", lambda: SimpleNamespace(available=1200 * 1024**2))
    monkeypatch.setattr(request_router, "_active_orbit_runtime", lambda: None)
    monkeypatch.setattr(request_router, "_measured_memory", lambda _: 900.0)

    decision_strict = recommend_runtime("Hi", [model], safety_reserve_mb=1536.0)
    assert decision_strict["selected"] is None
    assert "RAM safety reserve" in decision_strict["reason"]

    decision_lenient = recommend_runtime("Hi", [model], safety_reserve_mb=256.0)
    assert decision_lenient["selected"] is not None
    assert decision_lenient["selected"]["model"]["model_id"] == "qwen-test"


def test_reads_bom_marked_runtime_state(monkeypatch, tmp_path):
    runtime_directory = tmp_path / "runtime"
    runtime_directory.mkdir()
    state_path = runtime_directory / "active_runtime.json"
    state_path.write_bytes(b"\xef\xbb\xbf" + json.dumps({"pid": 42, "model_id": "qwen-resident"}).encode())

    class Process:
        def name(self):
            return "llama-server.exe"

        def memory_info(self):
            return SimpleNamespace(rss=512 * 1024**2)

    monkeypatch.setattr(request_router, "ROOT", tmp_path)
    monkeypatch.setattr(request_router.psutil, "Process", lambda _: Process())

    assert request_router._active_orbit_runtime() == ("qwen-resident", 512.0)
