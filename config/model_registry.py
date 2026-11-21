"""Versioned model metadata registry; it never probes or downloads models."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY = PROJECT_ROOT / "config" / "model_registry.json"


def load_model_registry(path: Path = DEFAULT_REGISTRY) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("Model registry must contain a 'models' array.")
    identifiers = [item.get("model_id") for item in models]
    if len(identifiers) != len(set(identifiers)) or not all(identifiers):
        raise ValueError("Each model registry entry needs a unique model_id.")
    return models


def model_by_id(model_id: str, path: Path = DEFAULT_REGISTRY) -> dict[str, Any] | None:
    return next((item for item in load_model_registry(path) if item["model_id"] == model_id), None)
