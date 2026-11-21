"""Public optimizer API without eagerly importing command-line submodules.

Keeping these imports lazy prevents ``python -m optimizer.calibrated_profiles``
from loading that same module through the package initializer before runpy can
execute it.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from optimizer.optimizer import OptimizationRequest, Optimizer

__all__ = ["Optimizer", "OptimizationRequest"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from optimizer.optimizer import OptimizationRequest, Optimizer
        return {"Optimizer": Optimizer, "OptimizationRequest": OptimizationRequest}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
