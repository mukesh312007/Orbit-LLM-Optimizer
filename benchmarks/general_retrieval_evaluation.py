"""Controlled local retrieval evaluation over inspectable project documents."""
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks.quality_metrics.metrics import rouge1
from core.ids import new_experiment_id
from core.model_registry import load_models
from core.settings import ROOT
from runtime.clients import InferenceError, generate, tokenize
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, ensure_runtime

EVALUATION_PATH = ROOT / "datasets" / "retrieval_eval.json"
CORPUS_DIR = "datasets/retrieval"


def load_evaluation() -> list[dict[str, Any]]:
    samples = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
    if len(samples) < 10:
        raise ValueError("General retrieval evaluation requires at least ten labelled questions.")
    return samples


def configuration(model_id: str, retrieval_enabled: bool) -> dict[str, Any]:
    return {
        "configuration_id": f"{model_id}:t4:c2048:cpu:project-retrieval-{'on' if retrieval_enabled else 'off'}",
        "backend": "llama.cpp", "threads": 4, "context_size": 2048, "gpu_layers": 0,
        "cache": "server-default", "prompt_compression": "none",
        "retrieval": {"enabled": retrieval_enabled, "top_k": 2, "corpus_dir": CORPUS_DIR},
    }


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_.]+", text.lower()))


def evaluate_answer(answer: str, sample: dict[str, Any], retrieval_documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Transparent lexical scoring; it is not presented as semantic judgment."""
    answer_tokens = _tokens(answer)
    required_terms = [term.lower() for term in sample["required_terms"]]
    term_coverage = sum(any(term in token for token in answer_tokens) for term in required_terms) / len(required_terms)
    relevance = rouge1(answer, sample["reference_answer"])
    expected_source = Path(sample["source_document"]).as_posix()
    retrieved_paths = [Path(item["path"]).as_posix() for item in retrieval_documents]
    source_retrieved = expected_source in retrieved_paths
    source_text = " ".join((ROOT / path).read_text(encoding="utf-8", errors="replace") for path in retrieved_paths if (ROOT / path).exists())
    source_supports_terms = all(term in _tokens(source_text) for term in required_terms)
    supported = bool(source_retrieved and source_supports_terms and term_coverage == 1.0)
    return {
        "correctness": term_coverage,
        "relevance": relevance["score"],
        "support": float(supported),
        "evaluation_method": "local_required_term_coverage_and_rouge1",
        "expected_source": expected_source,
        "retrieved_paths": retrieved_paths,
        "source_retrieved": source_retrieved,
        "required_terms": required_terms,
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"count": len(values), "mean": statistics.mean(values), "median": statistics.median(values), "min": min(values), "max": max(values)}


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [record for record in records if record.get("phase") == "measured" and record.get("success")]
    summary: dict[str, Any] = {}
    for model_id in sorted({record["model_id"] for record in measured}):
        baseline_correctness: float | None = None
        for enabled in (False, True):
            group = [record for record in measured if record["model_id"] == model_id and record["retrieval_enabled"] == enabled]
            correctness = _stats([record["evaluation"]["correctness"] for record in group])
            if not enabled:
                baseline_correctness = correctness["mean"] if isinstance(correctness["mean"], (int, float)) else None
            key = f"{model_id}:retrieval_{'on' if enabled else 'off'}"
            summary[key] = {
                "successful_requests": len(group), "correctness": correctness,
                "relevance": _stats([record["evaluation"]["relevance"] for record in group]),
                "support": _stats([record["evaluation"]["support"] for record in group]),
                "source_retrieval_rate": _stats([float(record["evaluation"]["source_retrieved"]) for record in group]),
                "metrics": {name: _stats([record["telemetry"].get(name) for record in group if record["telemetry"].get(name) is not None]) for name in ("end_to_end_seconds", "ttft_seconds", "generation_tokens_per_second", "peak_ram_mb", "energy_proxy_joules")},
                "input_tokens": _stats([record["input_tokens"] for record in group if record.get("input_tokens") is not None]),
            }
        with_retrieval = summary[f"{model_id}:retrieval_on"]["correctness"]["mean"]
        summary[f"{model_id}:retrieval_on"]["correctness_delta_vs_off"] = with_retrieval - baseline_correctness if isinstance(with_retrieval, (int, float)) and baseline_correctness is not None else None
    return summary


def run(model_ids: list[str] | None = None, trials: int = 2, max_output_tokens: int = 128) -> tuple[Path, Path]:
    if trials < 1:
        raise ValueError("trials must be at least one")
    samples = load_evaluation()
    available = [model for model in load_models() if model.get("available_locally") and "llama.cpp" in model.get("backends", [])]
    selected = [model for model in available if not model_ids or model["model_id"] in model_ids]
    missing = sorted(set(model_ids or []) - {model["model_id"] for model in selected})
    if missing:
        raise FileNotFoundError(f"Requested local models are unavailable: {', '.join(missing)}")
    experiment_id = new_experiment_id("general_retrieval")
    records: list[dict[str, Any]] = []
    for model in selected:
        for enabled in (False, True):
            config = configuration(model["model_id"], enabled)
            try:
                ensure_runtime(model, config, force_restart=True)
                warmup_prompt, warmup_preparation = prepare_prompt("Reply only with ready.", config)
                generate(model, warmup_prompt, {"backend": "llama.cpp", "max_output_tokens": 8, "temperature": 0})
            except (InferenceError, RuntimeSwitchError) as error:
                records.append({"experiment_id": experiment_id, "phase": "warmup", "success": False, "model_id": model["model_id"], "retrieval_enabled": enabled, "error": str(error)})
                continue
            for trial in range(1, trials + 1):
                for sample in samples:
                    prepared, preparation = prepare_prompt(sample["question"], config)
                    record: dict[str, Any] = {
                        "experiment_id": experiment_id, "timestamp": datetime.now(UTC).isoformat(), "dataset": "orbit_project_retrieval",
                        "phase": "measured", "trial_number": trial, "model_id": model["model_id"], "model_file": model["path"],
                        "quantization": model.get("quantization"), "configuration": config, "prompt_id": sample["id"],
                        "query": sample["question"], "retrieval_enabled": enabled, "preparation": preparation, "input_tokens": tokenize(prepared),
                    }
                    try:
                        telemetry = generate(model, prepared, {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
                        evaluation = evaluate_answer(telemetry["answer"], sample, preparation.get("retrieval_documents", []))
                        records.append(record | {"success": True, "return_code": 0, "response": telemetry["answer"], "output_tokens": telemetry.get("generated_tokens"), "telemetry": telemetry, "evaluation": evaluation, "measurement_kind": "measured"})
                    except (InferenceError, RuntimeSwitchError) as error:
                        records.append(record | {"success": False, "return_code": 1, "error": str(error), "measurement_kind": "measured_failure"})
    output_dir = ROOT / "logs" / "general_retrieval_evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{experiment_id}.jsonl"
    raw_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    summary_path = output_dir / f"{experiment_id}.summary.json"
    summary_path.write_text(json.dumps({
        "experiment_id": experiment_id, "dataset": "orbit_project_retrieval", "question_count": len(samples), "trials": trials,
        "corpus_dir": CORPUS_DIR, "document_count": len(list((ROOT / CORPUS_DIR).glob("*.md"))),
        "evaluation_note": "Correctness, relevance, and support are deterministic local lexical measures, not a semantic judge.",
        "summary": summarize(records),
    }, indent=2), encoding="utf-8")
    return raw_path, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the local project-document retrieval evaluation.")
    parser.add_argument("--model-id", action="append", dest="model_ids")
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    arguments = parser.parse_args()
    raw, summary = run(arguments.model_ids, arguments.trials, arguments.max_output_tokens)
    print(json.dumps({"raw_results": str(raw), "summary": str(summary)}, indent=2))
