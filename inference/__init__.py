"""Modular local-only inference backends."""

from .backends import InferenceResult, LlamaCppServerBackend, OllamaBackend

__all__ = ["InferenceResult", "LlamaCppServerBackend", "OllamaBackend"]
