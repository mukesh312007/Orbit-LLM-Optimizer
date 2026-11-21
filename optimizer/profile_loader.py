"""Read-only aggregation of legacy CSV measurements into clearly labelled profiles."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path

from runtime_optimizer.types import CandidateProfile, RuntimeConfiguration

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def legacy_profiles(log_dir: Path = PROJECT_ROOT / "logs") -> list[CandidateProfile]:
    """Produce a general-task profile from existing experiments without modifying them."""
    rows = list(csv.DictReader((log_dir / "benchmark_results_v2.csv").open(encoding="utf-8")))
    if not rows:
        return []
    latency = [float(row["wall_time_seconds"]) for row in rows if row.get("wall_time_seconds")]
    ram = [float(row["peak_process_ram_mb"]) for row in rows if row.get("peak_process_ram_mb")]
    score_path = log_dir / "combined_quality_scores.csv"
    scores = [float(row["quality_percentage"]) / 100 for row in csv.DictReader(score_path.open(encoding="utf-8"))] if score_path.exists() else []
    row = rows[-1]
    config = RuntimeConfiguration(configuration_id=f"legacy-llamacpp-t{row.get('threads', '4')}-c{row.get('context_size', '2048')}", model_id="qwen2.5-1.5b-instruct-q4_k_m-gguf", backend="llama.cpp", threads=int(row.get("threads") or 4), context_length=int(row.get("context_size") or 2048), max_output_tokens=int(row.get("max_output_tokens") or 128))
    return [CandidateProfile(configuration=config, task_type="general", quality=statistics.mean(scores) if scores else None, latency_seconds=statistics.mean(latency), ram_mb=statistics.mean(ram), energy_joules=None, sample_count=min(len(rows), len(scores)) if scores else len(rows), quality_stddev=statistics.stdev(scores) if len(scores) > 1 else None, latency_stddev=statistics.stdev(latency) if len(latency) > 1 else None, notes="Aggregate of separate legacy benchmark CSVs; task-specific pairing is unavailable.")]
