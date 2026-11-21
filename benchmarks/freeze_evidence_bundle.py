"""Freeze a presentation-ready evidence bundle summarizing all benchmark data.

Collects evidence across all benchmark logs, calibrated profiles, hardware
snapshots, dataset readiness, and transparency disclosures without mutating data.
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.provenance import hardware_snapshot
from core.settings import ROOT
from hardware.energy_capability import probe
from optimizer.calibrated_profiles import latest_profiles
from optimizer.evidence_report import dataset_readiness, profile_pareto_view


def build_evidence_bundle() -> dict[str, Any]:
    logs_dir = ROOT / "logs"
    categories = {
        "full_matrix_benchmarks": logs_dir / "full_matrix_benchmarks",
        "quantization_comparisons": logs_dir / "quantization_comparisons",
        "context_thread_calibrations": logs_dir / "context_thread_calibrations",
        "cache_prompt_experiments": logs_dir / "cache_prompt_experiments",
        "standard_quality_benchmarks": logs_dir / "standard_quality_benchmarks",
        "three_model_comparisons": logs_dir / "three_model_comparisons",
    }

    total_records = 0
    measured_success_count = 0
    models_seen: set[str] = set()
    configurations_seen: set[str] = set()
    spider_databases_seen: set[str] = set()
    source_files_summary: list[dict[str, Any]] = []

    for category, directory in categories.items():
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.jsonl")):
            file_records = 0
            file_success = 0
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    file_records += 1
                    total_records += 1
                    model_id = record.get("model_id")
                    if model_id:
                        models_seen.add(model_id)
                    config = record.get("configuration")
                    if isinstance(config, dict) and config.get("configuration_id"):
                        configurations_seen.add(config["configuration_id"])
                    prompt_id = record.get("prompt_id")
                    if prompt_id and isinstance(prompt_id, str) and prompt_id.startswith("spider:"):
                        parts = prompt_id.split(":")
                        if len(parts) >= 2:
                            spider_databases_seen.add(parts[1])
                    if record.get("phase") == "measured" and record.get("success"):
                        file_success += 1
                        measured_success_count += 1
            source_files_summary.append({
                "category": category,
                "file": path.name,
                "total_records": file_records,
                "successful_measured": file_success,
                "bytes": path.stat().st_size,
            })

    history_path = logs_dir / "runtime_selection_history.jsonl"
    history_count = 0
    if history_path.is_file():
        with history_path.open("r", encoding="utf-8", errors="replace") as f:
            history_count = sum(1 for line in f if line.strip())

    profiles = latest_profiles()
    pareto_profiles = profile_pareto_view()
    datasets = dataset_readiness()
    hw = hardware_snapshot()
    energy = probe(ROOT / "hardware" / "meter_export.json")

    bundle: dict[str, Any] = {
        "bundle_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "summary": {
            "total_raw_telemetry_records": total_records,
            "successful_measured_trials": measured_success_count,
            "models_evaluated": sorted(models_seen),
            "model_count": len(models_seen),
            "unique_configurations_evaluated": len(configurations_seen),
            "spider_databases_covered": sorted(spider_databases_seen),
            "spider_database_count": len(spider_databases_seen),
            "calibrated_profiles_count": len(profiles),
            "pareto_optimal_profiles_count": len(pareto_profiles),
            "runtime_selection_history_records": history_count,
        },
        "hardware_environment": {
            "platform": hw.get("platform"),
            "os_release": hw.get("os_release"),
            "cpu_architecture": hw.get("machine"),
            "total_physical_ram_mb": hw.get("total_ram_mb"),
            "cpu_logical_cores": hw.get("cpu_logical_cores"),
            "cpu_physical_cores": hw.get("cpu_physical_cores"),
        },
        "energy_capability_status": {
            "status": "simulated_proxy" if energy.get("actual_energy_available") else "unavailable",
            "source": energy.get("source"),
            "actual_energy_available": energy.get("actual_energy_available"),
            "proxy_note": energy.get("proxy_note"),
            "supported_physical_sources": energy.get("supported_sources"),
        },
        "dataset_readiness": datasets,
        "calibrated_profiles": profiles,
        "pareto_frontier": pareto_profiles,
        "source_files": source_files_summary,
        "limitations_and_integrity_disclosures": {
            "single_server_architecture": "One llama.cpp process active at a time to respect RAM limits.",
            "ram_safety_reserve": "1.5 GiB headroom strictly enforced before loading any runtime candidate.",
            "energy_metric_distinction": "quality_per_proxy_joule uses CPU-time proxy; physical watts require an external hardware meter.",
            "general_datasets_status": "GSM8K, MMLU, and IFEval prepared locally without network downloads.",
            "storage_of_truth": "Append-only JSONL files are the primary truth; MySQL is an optional mirror.",
        },
    }
    return bundle


def freeze_bundle(output_path: Path | None = None) -> Path:
    bundle = build_evidence_bundle()
    demo_dir = ROOT / "logs" / "demonstrations"
    demo_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = output_path or (demo_dir / f"evidence_bundle_{timestamp}.json")
    target.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    latest = demo_dir / "latest_evidence_bundle.json"
    latest.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze demonstration evidence bundle.")
    parser.add_argument("--output", type=Path, default=None, help="Custom output path")
    args = parser.parse_args()
    path = freeze_bundle(args.output)
    bundle = json.loads(path.read_text(encoding="utf-8"))
    summary = bundle["summary"]
    print(f"=== ORBIT DEMONSTRATION EVIDENCE BUNDLE ===")
    print(f"Frozen at: {path}")
    print(f"Total raw telemetry records: {summary['total_raw_telemetry_records']}")
    print(f"Successful measured trials:  {summary['successful_measured_trials']}")
    print(f"Models evaluated:            {', '.join(summary['models_evaluated'])}")
    print(f"Unique configurations:       {summary['unique_configurations_evaluated']}")
    print(f"Spider databases covered:    {summary['spider_database_count']} ({', '.join(summary['spider_databases_covered'][:5])}...)")
    print(f"Calibrated profiles:         {summary['calibrated_profiles_count']}")
    print(f"Pareto-optimal profiles:     {summary['pareto_optimal_profiles_count']}")
    print(f"Energy readiness:            {bundle['energy_capability_status']['source']}")
    print(f"===========================================")


if __name__ == "__main__":
    main()
