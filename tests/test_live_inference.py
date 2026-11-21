"""Opt-in real local inference test; never runs in the default fast unit suite."""

import os

import pytest

from config.settings import settings
from inference.backends import OllamaBackend


@pytest.mark.skipif(os.getenv("RUN_LIVE_INFERENCE") != "1", reason="Set RUN_LIVE_INFERENCE=1 to run a local Ollama model.")
def test_ollama_local_inference():
    backend = OllamaBackend(settings.ollama_url)
    if not backend.health():
        pytest.skip("Ollama is not running locally.")
    response = backend.generate("qwen2.5:0.5b", "Reply with exactly: local inference works", 12, 0.0, 60)
    assert response.text.strip()
    assert response.backend == "ollama"
