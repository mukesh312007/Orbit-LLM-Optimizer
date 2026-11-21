from __future__ import annotations

import re
import os
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Any


def _normalise(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s.-]", " ", text.lower()).split())


def _last_number(text: str) -> str | None:
    values = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", text.replace(",", ""))
    return values[-1] if values else None


def exact_match(answer: str, reference: str) -> dict[str, Any]:
    score = float(_normalise(answer) == _normalise(reference))
    return {"score": score, "metric": "exact_match"}


def mmlu_exact(answer: str, reference: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Accept official index/letter keys while requiring a distinct final option response."""
    expected = str(reference).strip().upper()
    if expected.isdigit():
        expected = chr(65 + int(expected))
    choices = metadata.get("choices", [])
    if expected in choices:
        expected = chr(65 + choices.index(expected))
    letters = re.findall(r"\b([A-D])\b", answer.upper())
    predicted = letters[-1] if letters else None
    return {"score": float(predicted == expected), "metric": "mmlu_exact_match", "predicted": predicted, "expected": expected}


def numeric_exact(answer: str, reference: str) -> dict[str, Any]:
    predicted, expected = _last_number(answer), _last_number(reference)
    return {"score": float(predicted is not None and predicted == expected), "metric": "numeric_exact", "predicted": predicted, "expected": expected}


def rouge1(answer: str, reference: str) -> dict[str, Any]:
    a, b = _normalise(answer).split(), _normalise(reference).split()
    overlap = sum((Counter(a) & Counter(b)).values())
    precision, recall = overlap / len(a) if a else 0.0, overlap / len(b) if b else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"score": f1, "metric": "rouge1_f1", "precision": precision, "recall": recall}


def code_execution(answer: str, tests: list[str], timeout_seconds: int = 3) -> dict[str, Any]:
    """Run only supplied benchmark tests in an isolated temp cwd with timeout; not a security boundary."""
    if os.getenv("ALLOW_UNSAFE_CODE_EVAL") != "1":
        return {"score": None, "metric": "code_execution", "status": "disabled_without_isolated_runner"}
    code = re.sub(r"^```(?:python)?|```$", "", answer.strip(), flags=re.MULTILINE).strip()
    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/candidate.py"
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(code + "\n" + "\n".join(tests))
        try:
            result = subprocess.run([sys.executable, "-I", path], cwd=directory, capture_output=True, text=True, timeout=timeout_seconds)
            return {"score": float(result.returncode == 0), "metric": "code_execution", "stderr": result.stderr[-1000:]}
        except subprocess.TimeoutExpired:
            return {"score": 0.0, "metric": "code_execution", "error": "timeout"}


def instruction_following(answer: str, metadata: dict[str, Any]) -> dict[str, Any]:
    """Do not present heuristic IFEval checks as benchmark-quality evidence.

    The preparation pipeline preserves official instruction metadata, but the
    official checker is not vendored in this prototype.  Returning an
    unavailable score is safer than allowing substring checks to pass a model
    through a quality gate.
    """
    return {
        "score": None,
        "metric": "ifeval_official_required",
        "status": "unavailable_without_official_ifeval_checker",
        "instruction_ids": metadata.get("instruction_id_list", []),
    }


def evaluate(sample: dict[str, Any], answer: str) -> dict[str, Any]:
    kind = sample["evaluation_type"]
    if kind == "exact_match":
        if str(sample.get("dataset", "")).startswith("mmlu"): return mmlu_exact(answer, str(sample["reference_answer"]), sample.get("metadata", {}))
        return exact_match(answer, str(sample["reference_answer"]))
    if kind == "numeric_exact": return numeric_exact(answer, str(sample["reference_answer"]))
    if kind == "rouge": return rouge1(answer, str(sample["reference_answer"]))
    if kind == "code_execution": return code_execution(answer, sample.get("metadata", {}).get("test_list", []))
    if kind == "instruction_following": return instruction_following(answer, sample.get("metadata", {}))
    raise ValueError(f"Unsupported evaluation type: {kind}")
