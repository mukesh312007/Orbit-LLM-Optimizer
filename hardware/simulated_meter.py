"""Simulated proxy physical energy meter for demonstration and testing.

Estimates cumulative energy (joules) using host CPU user seconds and an
assumed Thermal Design Power (TDP) in Watts.

WARNING:
This is a software CPU-time proxy demonstration.
It is NOT a physical power measurement from a hardware counter or wall meter.
For local demonstration and calibration delta validation only.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil


def measure_simulated_joules(tdp_watts: float = 15.0) -> dict[str, Any]:
    """Calculate simulated cumulative joules from system CPU user time."""
    cpu_times = psutil.cpu_times()
    cpu_user_seconds = float(cpu_times.user)
    cumulative_joules = round(cpu_user_seconds * tdp_watts, 4)
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "source": "cpu_time_proxy_simulated",
        "cumulative_joules": cumulative_joules,
        "cpu_user_seconds": cpu_user_seconds,
        "estimated_tdp_watts": tdp_watts,
        "timestamp": now_iso,
        "note": "Simulated meter using CPU time proxy. NOT a physical power measurement. For demonstration only.",
        "warning": "Simulated meter proxy: cpu_user_seconds * TDP. NOT a physical hardware meter.",
    }


def write_meter_export(
    output_path: str | Path = "hardware/meter_export.json",
    tdp_watts: float = 15.0,
) -> dict[str, Any]:
    """Capture current simulated reading and write to meter export file."""
    reading = measure_simulated_joules(tdp_watts=tdp_watts)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(reading, indent=2) + "\n", encoding="utf-8")
    return reading


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulated physical energy meter proxy for demonstration."
    )
    parser.add_argument(
        "--tdp",
        type=float,
        default=15.0,
        help="Estimated TDP in Watts (default: 15.0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="hardware/meter_export.json",
        help="Path to write meter JSON export (default: hardware/meter_export.json)",
    )
    args = parser.parse_args()

    result = write_meter_export(output_path=args.output, tdp_watts=args.tdp)
    print(f"[simulated_meter] Wrote simulated reading to {args.output}:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
