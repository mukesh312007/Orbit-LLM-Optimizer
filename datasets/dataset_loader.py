"""Reproducible, opt-in preparation of small official benchmark subsets.

No dataset is downloaded during application startup. Run this module explicitly
after reviewing each upstream dataset's current terms of use.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

from datasets.dataset_validator import validate_samples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = PROJECT_ROOT / "datasets" / "manifests" / "official_datasets.json"
PROCESSED_DIR = PROJECT_ROOT / "datasets" / "processed"


def load_manifest(path: Path = MANIFEST) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))["datasets"]


def _mmlu(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    choices = item["choices"]
    labels = ["A", "B", "C", "D"][:len(choices)]
    prompt = item["question"] + "\n" + "\n".join(f"{label}. {choice}" for label, choice in zip(labels, choices)) + "\nAnswer with one letter."
    answer = labels[int(item["answer"])]
    return _base(item, dataset, "multiple_choice", prompt, answer, "exact_match")


def _gsm8k(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    return _base(item, dataset, "reasoning", item["question"], item["answer"], "numeric_exact_match")


def _mbpp(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    result = _base(item, dataset, "code", item["text"], item["code"], "code_tests")
    result["metadata"]["test_list"] = item.get("test_list", [])
    return result


def _ifeval(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    result = _base(item, dataset, "instruction_following", item["prompt"], "", "ifeval")
    result["metadata"]["instruction_id_list"] = item.get("instruction_id_list", [])
    result["metadata"]["kwargs"] = item.get("kwargs", [])
    return result


def _cnn(item: dict[str, Any], dataset: str) -> dict[str, Any]:
    return _base(item, dataset, "summarization", "Summarize the following article:\n\n" + item["article"], item["highlights"], "rouge")


def _base(item: dict[str, Any], dataset: str, task_type: str, prompt: str, answer: str, evaluation: str) -> dict[str, Any]:
    source_id = item.get("id") or item.get("idx") or item.get("source") or "unknown"
    return {"sample_id": f"{dataset}:{source_id}", "dataset": dataset, "task_type": task_type,
            "prompt": prompt, "reference_answer": answer, "evaluation_type": evaluation,
            "metadata": {"source_identifier": source_id}}


NORMALIZERS = {"mmlu": _mmlu, "gsm8k": _gsm8k, "mbpp": _mbpp, "ifeval": _ifeval, "cnn_dailymail": _cnn}


def prepare_dataset(dataset_name: str, limit: int = 50) -> Path:
    entry = next((x for x in load_manifest() if x["name"] == dataset_name), None)
    if entry is None:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError("Install requirements.txt to prepare benchmark datasets.") from error
    kwargs = {"split": entry["split"]}
    if entry.get("config"):
        raw = load_dataset(entry["hf_path"], entry["config"], **kwargs)
    else:
        raw = load_dataset(entry["hf_path"], **kwargs)
    samples = [NORMALIZERS[dataset_name](item, dataset_name) for item in raw.select(range(min(limit, len(raw))))]
    errors = validate_samples(samples)
    if errors:
        raise ValueError("Normalization failed: " + "; ".join(errors))
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output = PROCESSED_DIR / f"{dataset_name}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    provenance = {**entry, "download_date": date.today().isoformat(), "sample_count": len(samples), "output": str(output.relative_to(PROJECT_ROOT))}
    (PROJECT_ROOT / "datasets" / "manifests" / f"{dataset_name}.provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return output


def load_processed(dataset_name: str) -> list[dict[str, Any]]:
    path = PROCESSED_DIR / f"{dataset_name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and normalize a reviewed official benchmark subset.")
    parser.add_argument("dataset", choices=sorted(NORMALIZERS))
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print(prepare_dataset(args.dataset, args.limit))
