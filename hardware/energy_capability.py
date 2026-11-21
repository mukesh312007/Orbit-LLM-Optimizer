"""Truthful capability reporting for actual energy measurement.

The application has a CPU-utilisation energy proxy for exploratory comparison,
but that proxy is never exposed here as a physical energy reading.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


_NON_PHYSICAL_SOURCE_MARKERS = ("proxy", "simulat", "estimate", "tdp", "cpu_time", "cpu-time")


@dataclass(frozen=True)
class EnergyReading:
    source: str
    cumulative_joules: float | None
    measurement_kind: str = "measured"


def joule_delta(start: EnergyReading, end: EnergyReading) -> dict[str, Any]:
    """Calculate a valid cumulative-meter delta or label it unavailable."""
    if start.source != end.source:
        return {"energy_joules": None, "measurement_kind": "unavailable", "reason": "Meter source changed during measurement."}
    if start.cumulative_joules is None or end.cumulative_joules is None:
        return {"energy_joules": None, "measurement_kind": "unavailable", "reason": "No actual cumulative energy reading was supplied."}
    if end.cumulative_joules < start.cumulative_joules:
        return {"energy_joules": None, "measurement_kind": "unavailable", "reason": "Meter reading decreased or reset."}
    return {"energy_joules": end.cumulative_joules - start.cumulative_joules, "measurement_kind": "measured", "source": end.source}


def probe(path: str | Path | None = None) -> dict[str, Any]:
    """Describe physical-meter readiness without inventing a power reading."""
    reading = read_meter_export(path)
    if reading is not None:
        return {
            "actual_energy_available": True,
            "measurement_kind": "measured",
            "energy_joules": None,
            "source": reading.source,
            "supported_sources": ["external wall-power meter with cumulative energy export", "hardware counter adapter that supplies cumulative joules"],
            "reason": "A valid cumulative-joule reading is available; a second reading after inference is required for an energy delta.",
            "proxy_note": "CPU-utilisation energy proxy remains estimated and is not a replacement for this measurement.",
        }
    return {
        "actual_energy_available": False,
        "measurement_kind": "unavailable",
        "energy_joules": None,
        "supported_sources": ["external wall-power meter with cumulative energy export", "hardware counter adapter that supplies cumulative joules"],
        "reason": "No configured local adapter currently supplies actual cumulative energy readings.",
        "proxy_note": "CPU-utilisation energy proxy remains estimated and is not a replacement for this measurement.",
    }


def read_meter_export(path: str | Path | None) -> EnergyReading | None:
    """Read a local physical-meter export with a cumulative joule counter.

    Expected JSON: ``{"source": "wall power meter", "cumulative_joules": 123.4}``.
    The file must be updated by the physical meter or its local software; this
    code does not invent, estimate, or write any energy reading.
    """
    if not path:
        return None
    try:
        target = Path(path)
        if not target.is_absolute() and not target.is_file():
            try:
                from core.settings import ROOT
                target = ROOT / path
            except ImportError:
                pass
        payload = json.loads(target.read_text(encoding="utf-8"))
        source = payload.get("source")
        joules = payload.get("cumulative_joules")
        if not isinstance(source, str) or not source.strip() or not isinstance(joules, (int, float)) or joules < 0:
            return None
        # A cumulative number is not proof of physical energy.  In particular,
        # reject the bundled demonstration meter and any source that declares
        # itself an estimate/proxy before it can enter the measured-energy path.
        if any(marker in source.lower() for marker in _NON_PHYSICAL_SOURCE_MARKERS):
            return None
        declared_kind = payload.get("measurement_kind")
        if declared_kind is not None and declared_kind not in {"physical_meter", "hardware_counter"}:
            return None
        return EnergyReading(source=source, cumulative_joules=float(joules))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def example_reading(source: str, cumulative_joules: float | None) -> dict[str, Any]:
    """Small serialisable adapter boundary for a future physical-meter integration."""
    return asdict(EnergyReading(source=source, cumulative_joules=cumulative_joules))
