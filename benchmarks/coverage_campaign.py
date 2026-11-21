"""Run or document the complete local evidence campaign without any downloads."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.cache_prompt_experiments import run as run_cache
from benchmarks.context_thread_calibration import run as run_context_thread
from benchmarks.full_matrix_benchmark import run as run_full_matrix
from benchmarks.standard_quality_benchmark import run as run_standard_quality
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.settings import ROOT
from optimizer.calibrated_profiles import create_profiles
from optimizer.evidence_report import readiness_report

DEFAULT_MODELS = ("qwen2.5-0.5b-q4_k_m", "qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m")
GENERAL_DATASETS = ("gsm8k", "mmlu", "ifeval")


def plan(spider_limit: int, trials: int, max_configurations: int) -> dict[str, Any]:
    available = {model["model_id"] for model in load_models() if model.get("available_locally")}
    readiness = readiness_report()
    return {
        "local_only": True,
        "models_requested": list(DEFAULT_MODELS),
        "models_available": [model_id for model_id in DEFAULT_MODELS if model_id in available],
        "spider": {
            "sample_limit": spider_limit,
            "trials": trials,
            "max_configurations_per_model": max_configurations,
            "estimated_measured_requests_per_model": spider_limit * trials * max_configurations,
        },
        "controlled_studies": {
            "thread_context_models": ["qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m"],
            "cache_models": ["qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m"],
            "general_datasets": readiness["datasets"],
        },
        "physical_energy": readiness["energy"],
        "provenance": readiness["provenance_note"],
        "execution_note": "All stages run sequentially on loopback. --execute can take a long time and checkpoints the Spider JSONL after every configuration.",
    }


def execute(spider_limit: int, trials: int, max_configurations: int) -> Path:
    campaign_id = new_experiment_id("local_evidence_campaign")
    artifact = ROOT / "logs" / "demonstrations" / f"{campaign_id}.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "campaign_id": campaign_id, "started_at": datetime.now(UTC).isoformat(),
        "plan": plan(spider_limit, trials, max_configurations), "stages": [],
    }

    def stage(name: str, callback: Any) -> None:
        try:
            result = callback()
            report["stages"].append({"name": name, "status": "completed", "result": [str(item) for item in result] if isinstance(result, tuple) else str(result)})
        except Exception as error:  # Preserve evidence of a local prerequisite failure and continue.
            report["stages"].append({"name": name, "status": "unavailable", "reason": f"{error.__class__.__name__}: {error}"})
        artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")

    available = set(report["plan"]["models_available"])
    for model_id in DEFAULT_MODELS:
        if model_id in available:
            stage(f"spider_matrix:{model_id}", lambda model_id=model_id: run_full_matrix(
                limit=spider_limit, trials=trials, model_ids=[model_id], max_configurations=max_configurations,
                sampling_strategy="stratified",
            ))
    for model_id in ("qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m"):
        if model_id in available:
            stage(f"thread_context:{model_id}", lambda model_id=model_id: run_context_thread(model_id, limit=spider_limit, trials=trials))
            stage(f"cache:{model_id}", lambda model_id=model_id: run_cache(model_id, limit=spider_limit, trials=trials))
    for dataset in GENERAL_DATASETS:
        if next((row for row in report["plan"]["controlled_studies"]["general_datasets"] if row["dataset"] == dataset and row["prepared"]), None):
            stage(f"quality:{dataset}", lambda dataset=dataset: run_standard_quality(dataset, limit=spider_limit, trials=trials))
        else:
            report["stages"].append({"name": f"quality:{dataset}", "status": "awaiting_local_dataset", "reason": "Prepare a reviewed official local source; Orbit never downloads benchmark datasets."})
            artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    stage("calibrated_profiles", lambda: create_profiles(minimum_samples=trials, include_standard=True))
    report["completed_at"] = datetime.now(UTC).isoformat()
    artifact.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plan or execute a reproducible local benchmark evidence campaign.")
    parser.add_argument("--execute", action="store_true", help="Run the long local campaign. Without it, only print the local readiness plan.")
    parser.add_argument("--spider-limit", type=int, default=25)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-configurations", type=int, default=6)
    args = parser.parse_args()
    if args.spider_limit < 1 or args.trials < 1 or args.max_configurations < 1:
        parser.error("spider-limit, trials, and max-configurations must be positive")
    if args.execute:
        print(execute(args.spider_limit, args.trials, args.max_configurations))
    else:
        print(json.dumps(plan(args.spider_limit, args.trials, args.max_configurations), indent=2))
