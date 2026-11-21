"""Clients for loopback llama.cpp and Ollama servers."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import requests


@dataclass(frozen=True)
class InferenceResult:
    text: str
    prompt_tokens: int | None
    generated_tokens: int | None
    latency_seconds: float
    backend: str
    raw: dict[str, Any]


class InferenceBackend(Protocol):
    def generate(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> InferenceResult: ...


class LlamaCppServerBackend:
    name = "llama.cpp"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 3) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=timeout).ok
        except requests.RequestException:
            return False

    def generate(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> InferenceResult:
        start = time.perf_counter()
        response = requests.post(f"{self.base_url}/v1/chat/completions", json={"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature, "stream": False}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        return InferenceResult(payload["choices"][0]["message"]["content"], usage.get("prompt_tokens"), usage.get("completion_tokens"), time.perf_counter() - start, self.name, payload)


class OllamaBackend:
    name = "ollama"

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def health(self, timeout: float = 3) -> bool:
        try:
            return requests.get(f"{self.base_url}/api/tags", timeout=timeout).ok
        except requests.RequestException:
            return False

    def generate(self, model: str, prompt: str, max_tokens: int, temperature: float, timeout: float) -> InferenceResult:
        start = time.perf_counter()
        response = requests.post(f"{self.base_url}/api/generate", json={"model": model, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens, "temperature": temperature}}, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
        return InferenceResult(payload["response"], payload.get("prompt_eval_count"), payload.get("eval_count"), time.perf_counter() - start, self.name, payload)
