"""Validation for the portable JSONL dataset contract."""
from __future__ import annotations

from typing import Any

REQUIRED_FIELDS = {"sample_id", "dataset", "task_type", "prompt", "reference_answer", "evaluation_type", "metadata"}


def validate_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_FIELDS - sample.keys()
    if missing:
        errors.append(f"missing fields: {', '.join(sorted(missing))}")
        return errors
    for field in ("sample_id", "dataset", "task_type", "prompt", "evaluation_type"):
        if not isinstance(sample[field], str) or not sample[field].strip():
            errors.append(f"{field} must be a non-empty string")
    if not isinstance(sample["metadata"], dict):
        errors.append("metadata must be an object")
    return errors


def validate_samples(samples: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, sample in enumerate(samples):
        for error in validate_sample(sample):
            errors.append(f"sample {index}: {error}")
        identifier = sample.get("sample_id")
        if identifier in seen:
            errors.append(f"sample {index}: duplicate sample_id {identifier}")
        seen.add(identifier)
    return errors
