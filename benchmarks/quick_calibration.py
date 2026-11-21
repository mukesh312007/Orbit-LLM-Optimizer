"""Small, repeatable local calibration for Qwen routing; not a replacement for official benchmark suites."""
from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.model_registry import load_models
from core.settings import ROOT
from optimizer.request_router import recommend_runtime
from runtime.clients import generate
from runtime.supervisor import ensure_runtime

CALIBRATION_PATH = ROOT / "logs" / "model_calibration.jsonl"
QUALITY_DATASET = ROOT / "datasets" / "quality_eval.json"


def _score(item: dict[str, Any], answer: str) -> float:
    task = item["task_type"]
    normalized = answer.lower()
    if task == "reasoning":
        return float(bool(re.search(r"\b80\b", normalized)))
    if task == "coding":
        try:
            code = re.sub(r"```(?:python)?|```", "", answer, flags=re.I).strip()
            compile(code, "<candidate>", "exec")
            return float("def " in code and "return" in code)
        except SyntaxError:
            return 0.0
    if task == "summarization":
        reference_words = set(re.findall(r"\w+", item["reference_answer"].lower()))
        answer_words = set(re.findall(r"\w+", normalized))
        return len(reference_words & answer_words) / len(reference_words) if reference_words else 0.0
    points = item.get("required_points", [])
    point_scores = []
    for point in points:
        keywords = [word for word in re.findall(r"[a-z]{4,}", point.lower()) if word not in {"used", "used", "data", "that", "with", "from"}]
        point_scores.append(float(bool(keywords) and any(word in normalized for word in keywords)))
    return sum(point_scores) / len(point_scores) if point_scores else 0.0


def calibrate(max_output_tokens: int = 96) -> list[dict[str, Any]]:
    """Run the bundled five-prompt smoke set sequentially and append measured evidence."""
    dataset = json.loads(QUALITY_DATASET.read_text(encoding="utf-8"))
    models = load_models()
    base_decision = recommend_runtime("solve a reasoning problem", models)
    eligible = [item["model"] for item in base_decision["candidates"] if item["feasible"]]
    records: list[dict[str, Any]] = []
    for model in eligible:
        configuration = {
            "configuration_id": f"{model['model_id']}:t4:c2048:cpu",
            "backend": "llama.cpp", "threads": 4, "context_size": 2048,
            "gpu_layers": 0, "cache": "server-default",
        }
        ensure_runtime(model, configuration)
        scores, latencies = [], []
        for item in dataset:
            response = generate(model, item["prompt"], {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0})
            scores.append(_score(item, response["answer"]))
            latencies.append(response["end_to_end_seconds"])
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "model_id": model["model_id"], "configuration": configuration,
            "sample_count": len(dataset), "quality_score": sum(scores) / len(scores),
            "mean_latency_seconds": sum(latencies) / len(latencies),
            "measurement_kind": "measured_smoke_calibration",
            "limitation": "Five bundled prompts with lexical/syntax checks; not an official benchmark or a quality guarantee.",
        }
        records.append(record)
    with CALIBRATION_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return records


if __name__ == "__main__":
    for result in calibrate():
        print(json.dumps(result, indent=2))
