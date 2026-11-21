"""Local, privacy-preserving provenance captured with every benchmark run."""
from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import psutil

from core.settings import ROOT


@lru_cache(maxsize=64)
def _sha256_path(path: str) -> str | None:
    """Return a file hash without loading a model into memory."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def sha256_file(path: Path) -> str | None:
    """Hash each immutable local model once per benchmark process."""
    return _sha256_path(str(path.resolve()))


@lru_cache(maxsize=1)
def llama_cpp_version() -> str | None:
    executable = shutil.which("llama-server")
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, "--version"], capture_output=True, text=True,
            timeout=10, check=False,
        )
        text = (completed.stdout or completed.stderr).strip()
        return text[:500] or None
    except (OSError, subprocess.TimeoutExpired):
        return None


@lru_cache(maxsize=1)
def _hardware_snapshot() -> tuple[tuple[str, Any], ...]:
    memory = psutil.virtual_memory()
    return tuple({
        "operating_system": platform.platform(),
        "python_version": sys.version.split()[0],
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "total_ram_mb": round(memory.total / 1024**2, 1),
        "llama_cpp_version": llama_cpp_version(),
    }.items())


def hardware_snapshot() -> dict[str, Any]:
    """Return a copy so callers cannot mutate the cached machine identity."""
    return dict(_hardware_snapshot())


def benchmark_provenance(model: dict[str, Any], configuration: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / model["path"]
    return {
        "model_id": model["model_id"],
        "model_file": model["path"],
        "model_sha256": sha256_file(path),
        "model_bytes": path.stat().st_size if path.exists() else None,
        "quantization": model.get("quantization"),
        "runtime_configuration": configuration,
        "hardware": hardware_snapshot(),
    }
