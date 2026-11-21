from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = {"sample_id", "dataset", "task_type", "prompt", "reference_answer", "evaluation_type", "metadata"}
VALID_EVALUATIONS = {"exact_match", "numeric_exact", "code_execution", "instruction_following", "rouge"}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - sample.keys()
    if missing:
        errors.append(f"missing fields: {', '.join(sorted(missing))}")
    if not isinstance(sample.get("sample_id"), str) or not sample.get("sample_id"):
        errors.append("sample_id must be a non-empty string")
    if not isinstance(sample.get("prompt"), str) or not sample.get("prompt").strip():
        errors.append("prompt must be non-empty")
    if sample.get("evaluation_type") not in VALID_EVALUATIONS:
        errors.append("unsupported evaluation_type")
    if not isinstance(sample.get("metadata"), dict):
        errors.append("metadata must be an object")
    return errors


def validate_dataset(samples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    for index, sample in enumerate(samples):
        errors.extend(f"sample {index}: {error}" for error in validate_sample(sample))
        sample_id = sample.get("sample_id")
        if sample_id in ids:
            errors.append(f"sample {index}: duplicate sample_id {sample_id}")
        ids.add(sample_id)
    return errors
