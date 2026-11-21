"""Versioned, locally reproducible candidate runtime configurations."""
from __future__ import annotations

import itertools
import json
from typing import Any

from core.settings import ROOT

MATRIX_PATH = ROOT / "config" / "benchmark_matrix.json"


def load_matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


def baseline_configuration(model: dict[str, Any]) -> dict[str, Any]:
    return {
        "configuration_id": f"{model['model_id']}:t4:c1024:cpu",
        "backend": "llama.cpp", "threads": 4, "context_size": 1024,
        "gpu_layers": 0, "cache": "server-default",
        "prompt_compression": "none", "retrieval": {"enabled": False},
    }


def cache_server_arguments(configuration: dict[str, Any]) -> list[str]:
    """Translate a recorded cache policy to supported llama.cpp server flags."""
    cache = configuration.get("cache", "server-default")
    if cache == "prompt-cache-enabled":
        return ["--cache-prompt"]
    if cache == "prompt-cache-disabled":
        return ["--no-cache-prompt"]
    return []


def benchmark_configurations(model: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = load_matrix()
    retrieval_defaults = matrix["retrieval"]
    configurations: list[dict[str, Any]] = []
    for threads, context_size, cache, compression, retrieval_enabled in itertools.product(
        matrix["threads"], matrix["context_sizes"], matrix["cache_modes"],
        matrix["prompt_compression_modes"], matrix["retrieval_modes"],
    ):
        retrieval = {**retrieval_defaults, "enabled": retrieval_enabled}
        configuration = {
            "configuration_id": (
                f"{model['model_id']}:t{threads}:c{context_size}:cpu:"
                f"cache-{cache}:compress-{compression}:retrieval-{'on' if retrieval_enabled else 'off'}"
            ),
            "backend": "llama.cpp", "threads": threads, "context_size": context_size,
            "gpu_layers": 0, "cache": cache,
            "prompt_compression": compression, "retrieval": retrieval,
        }
        configurations.append(configuration)
    return configurations
