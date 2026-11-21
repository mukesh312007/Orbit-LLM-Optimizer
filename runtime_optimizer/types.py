"""Stable domain objects shared by benchmark, optimizer, API and persistence layers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4


Preference = Literal["quality", "latency", "memory", "energy", "balanced"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_experiment_id() -> str:
    return str(uuid4())


@dataclass(frozen=True)
class RuntimeConfiguration:
    configuration_id: str
    model_id: str
    backend: Literal["llama.cpp", "ollama"]
    threads: int | None = None
    context_length: int | None = None
    gpu_layers: int = 0
    prompt_cache: bool = False
    temperature: float = 0.0
    max_output_tokens: int = 256

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateProfile:
    configuration: RuntimeConfiguration
    task_type: str
    quality: float | None
    latency_seconds: float | None
    ram_mb: float | None
    energy_joules: float | None
    sample_count: int
    source: Literal["measured", "estimated"] = "measured"
    quality_stddev: float | None = None
    latency_stddev: float | None = None
    notes: str | None = None


@dataclass(frozen=True)
class OptimizationRequest:
    prompt: str
    task_type: str = "general"
    minimum_quality: float | None = None
    maximum_latency_seconds: float | None = None
    maximum_ram_mb: float | None = None
    maximum_energy_joules: float | None = None
    preference: Preference = "balanced"


@dataclass
class Recommendation:
    selected: CandidateProfile | None
    feasible: list[CandidateProfile] = field(default_factory=list)
    pareto_frontier: list[CandidateProfile] = field(default_factory=list)
    violations: dict[str, list[str]] = field(default_factory=dict)
    alternatives: list[CandidateProfile] = field(default_factory=list)
    explanation: str = ""

    def as_dict(self) -> dict[str, Any]:
        def profile(p: CandidateProfile) -> dict[str, Any]:
            result = asdict(p)
            result["configuration"] = p.configuration.as_dict()
            return result
        return {
            "selected": profile(self.selected) if self.selected else None,
            "feasible": [profile(p) for p in self.feasible],
            "pareto_frontier": [profile(p) for p in self.pareto_frontier],
            "violations": self.violations,
            "alternatives": [profile(p) for p in self.alternatives],
            "explanation": self.explanation,
        }
