from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.settings import ROOT

REGISTRY_PATH = ROOT / "config" / "models.json"
QUANTIZATION_DIRECTORY = ROOT / "models" / "quantizations"
MODEL_DIRECTORY = ROOT / "models"
QWEN_GGUF_NAME = re.compile(r"^qwen2\.5-(?P<size>\d+(?:\.\d+)?)b-instruct-(?P<quantization>q\d(?:_k_[ms]|_k|_0)?)\.gguf$", re.IGNORECASE)
GENERIC_QUANTIZATION = re.compile(r"\b(q\d(?:_k_[ms]|_k|_\d)?|f16|bf16)\b", re.IGNORECASE)


def _valid_gguf(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def _model_from_file(path: Path) -> dict[str, Any] | None:
    if not _valid_gguf(path):
        return None
    relative_path = path.relative_to(ROOT).as_posix()
    match = QWEN_GGUF_NAME.fullmatch(path.name)
    if match:
        size = float(match["size"])
        quantization = match["quantization"].upper()
        return {
            "model_id": f"qwen2.5-{match['size']}b-{match['quantization'].lower()}",
            "family": "Qwen2.5", "parameter_count_b": size, "quantization": quantization,
            "path": relative_path, "format": "GGUF", "backends": ["llama.cpp"],
            "license": "Apache-2.0", "context_limit": 32768,
            "minimum_ram_gb": round(path.stat().st_size / 1024**3 * 1.3 + 0.3, 1),
            "discovered_locally": True,
        }
    slug = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")
    quantization = GENERIC_QUANTIZATION.search(path.stem)
    return {
        "model_id": f"local-gguf-{slug}", "family": "Unknown GGUF", "parameter_count_b": 0.0,
        "quantization": quantization.group(1).upper() if quantization else None,
        "path": relative_path, "format": "GGUF", "backends": ["llama.cpp"],
        "license": "unknown", "context_limit": None,
        "minimum_ram_gb": round(path.stat().st_size / 1024**3 * 1.3 + 0.3, 1),
        "discovered_locally": True,
        "metadata_note": "Generic GGUF discovered from a valid header; inspect metadata before comparing architecture-specific quality.",
    }


def _discovered_models() -> list[dict[str, Any]]:
    """Discover valid local GGUF files without requiring registry edits."""
    if not MODEL_DIRECTORY.exists():
        return []
    discovered: list[dict[str, Any]] = []
    for path in sorted(MODEL_DIRECTORY.rglob("*.gguf")):
        if ".cache" in path.parts:
            continue
        model = _model_from_file(path)
        if not model:
            continue
        discovered.append(model)
    return discovered


def load_models() -> list[dict[str, Any]]:
    with REGISTRY_PATH.open(encoding="utf-8") as handle:
        models = json.load(handle)
    known_paths = {model.get("path") for model in models}
    models.extend(model for model in _discovered_models() if model["path"] not in known_paths)
    for model in models:
        path = model.get("path")
        model["available_locally"] = bool(path and (ROOT / path).exists())
    return models


def get_model(model_id: str) -> dict[str, Any] | None:
    return next((item for item in load_models() if item["model_id"] == model_id), None)
