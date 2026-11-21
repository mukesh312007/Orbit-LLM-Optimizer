from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

from core.settings import ROOT
from optimizer.calibrated_profiles import latest_profiles


def historical_profiles() -> list[dict[str, Any]]:
    """Expose existing CSV data as measured profiles, carefully leaving absent metrics null."""
    paths = [ROOT / "logs" / "benchmark_results_v2.csv", ROOT / "logs" / "streaming_results_v2.csv"]
    rows: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists(): continue
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                model = row.get("model", "unknown")
                latency = row.get("total_request_seconds") or row.get("wall_time_seconds")
                rows.append({"configuration_id": f"historical:{model}:{row.get('threads', 'unknown')}:{row.get('cache_mode', 'unknown')}", "model_id": model, "latency_seconds": float(latency) if latency else None, "ram_mb": float(row["peak_process_ram_mb"]) if row.get("peak_process_ram_mb") else None})
    profiles: list[dict[str, Any]] = []
    for configuration_id in sorted({row["configuration_id"] for row in rows}):
        group = [row for row in rows if row["configuration_id"] == configuration_id]
        latency_values = [row["latency_seconds"] for row in group if row["latency_seconds"] is not None]
        memory_values = [row["ram_mb"] for row in group if row["ram_mb"] is not None]
        profiles.append({"configuration_id": configuration_id, "model_id": group[0]["model_id"], "task_type": "all", "quality": None, "latency_seconds": statistics.mean(latency_values) if latency_values else None, "latency_median_seconds": statistics.median(latency_values) if latency_values else None, "latency_stddev_seconds": statistics.stdev(latency_values) if len(latency_values) > 1 else None, "ram_mb": statistics.mean(memory_values) if memory_values else None, "energy_joules": None, "sample_count": len(group), "measurement_kind": "measured"})
    return profiles + latest_profiles()
