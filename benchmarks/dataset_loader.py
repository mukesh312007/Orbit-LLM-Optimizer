from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from benchmarks.dataset_validator import validate_dataset
from core.settings import ROOT

PROCESSED = ROOT / "datasets" / "processed"


def load_processed(dataset: str) -> list[dict[str, Any]]:
    path = PROCESSED / f"{dataset}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found: {path}. Run dataset preparation first.")
    samples = list(_read_jsonl(path))
    errors = validate_dataset(samples)
    if errors:
        raise ValueError("Invalid dataset: " + "; ".join(errors))
    return samples


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def normalize_record(dataset: str, record: dict[str, Any], index: int) -> dict[str, Any]:
    """Normalize a locally supplied official record without changing labels."""
    mapping = {
        "mmlu": ("question", "answer", "reasoning", "exact_match"),
        "gsm8k": ("question", "answer", "reasoning", "numeric_exact"),
        "mbpp": ("text", "code", "code", "code_execution"),
        "ifeval": ("prompt", "kwargs", "instruction_following", "instruction_following"),
        "cnn_dailymail": ("article", "highlights", "summarization", "rouge"),
    }
    if dataset not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset}")
    prompt_key, answer_key, task_type, evaluation_type = mapping[dataset]
    prompt = record.get(prompt_key)
    answer = record.get(answer_key)
    if dataset == "mmlu":
        answer = record.get("answer", record.get("answer_index", record.get("label", record.get("target", answer))))
        choices = record.get("choices") or record.get("options") or []
        if isinstance(choices, dict):
            choices = choices.get("text", [])
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError("MMLU record needs a choices/options list with at least two answers")
        prompt = (
            f"{prompt}\n" + "\n".join(f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices))
            + "\n\nReply with only the final option letter."
        )
        record = {**record, "choices": choices}
    if dataset == "gsm8k":
        prompt = f"{prompt}\n\nGive the final numeric answer clearly at the end."
    if dataset == "ifeval" and not isinstance(record.get("kwargs"), dict):
        raise ValueError("IFEval record needs its official instruction metadata in kwargs")
    return {"sample_id": f"{dataset}:{record.get('id', index)}", "dataset": dataset, "task_type": task_type,
            "prompt": prompt, "reference_answer": answer, "evaluation_type": evaluation_type,
            "metadata": {key: value for key, value in record.items() if key not in {prompt_key, answer_key}}}
