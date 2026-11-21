"""Reference-based task-quality checks used by local model comparison."""
from __future__ import annotations

import re
import ast
import math
from collections import Counter
from typing import Any

from benchmarks.quality_metrics import evaluate as evaluate_benchmark_sample


def _sql_from_response(text: str) -> str:
    """Extract a SQL statement from a model response without sending it elsewhere."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    statement = re.search(r"\b(?:select|with|insert|update|delete)\b.*", candidate, flags=re.IGNORECASE | re.DOTALL)
    return statement.group(0) if statement else candidate


def _normalise_sql(text: str) -> str:
    text = _sql_from_response(text).strip().rstrip(";")
    return re.sub(r"\s+", " ", text).lower()


def evaluate_spider_sql(answer: str, reference_sql: str | None) -> dict[str, Any]:
    """Score one Spider response against a user-supplied gold SQL query.

    This is exact normalized SQL matching. Equivalent SQL written differently is
    deliberately reported as a non-match.
    """
    if not reference_sql or not reference_sql.strip():
        return {
            "score": None,
            "metric": "spider_sql_exact_match",
            "status": "reference_sql_required",
        }
    expected = _normalise_sql(reference_sql)
    predicted = _normalise_sql(answer)
    if predicted == expected:
        return {
            "score": 1.0,
            "metric": "spider_sql_exact_match",
            "status": "measured_exact_match",
            "predicted_sql": predicted,
            "reference_sql": expected,
        }
    return {
        "score": 0.0,
        "metric": "spider_sql_exact_match",
        "status": "nonmatching_exact_sql",
        "predicted_sql": predicted,
        "reference_sql": expected,
    }


def _reference_overlap(answer: str, reference_answer: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a bounded reference-answer similarity score.

    This is evidence of semantic overlap only.  It does not verify claims against
    the world, so callers must keep ``factual_verification`` separate from score.
    """
    metadata = metadata or {}
    answer_tokens = re.findall(r"[\w']+", answer.lower())
    reference_tokens = re.findall(r"[\w']+", reference_answer.lower())
    answer_counts, reference_counts = Counter(answer_tokens), Counter(reference_tokens)
    overlap = sum((answer_counts & reference_counts).values())
    precision = overlap / len(answer_tokens) if answer_tokens else 0.0
    recall = overlap / len(reference_tokens) if reference_tokens else 0.0
    similarity = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    required_terms = [str(term).lower() for term in metadata.get("required_terms", [])]
    missing_terms = [term for term in required_terms if term not in answer.lower()]
    if required_terms:
        similarity = min(similarity, (len(required_terms) - len(missing_terms)) / len(required_terms))
    return {
        "score": max(0.0, min(1.0, similarity)),
        "metric": "semantic_reference_overlap",
        "status": "measured_reference_similarity",
        "quality_confidence": "limited",
        "factual_verification": "unverified",
        "semantic_similarity_note": "Token overlap measures similarity to the supplied reference; it does not establish factual correctness.",
        "required_terms": required_terms,
        "missing_required_terms": missing_terms,
    }


def _safe_arithmetic(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_arithmetic(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_arithmetic(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)):
        left, right = _safe_arithmetic(node.left), _safe_arithmetic(node.right)
        operations = {
            ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right,
            ast.Pow: lambda: left ** right,
        }
        return operations[type(node.op)]()
    raise ValueError("expression contains unsupported arithmetic syntax")


def evaluate_arithmetic(answer: str, prompt: str, reference_answer: str | None = None) -> dict[str, Any]:
    """Independently calculate a numeric expression from the task prompt."""
    expression_match = re.search(r"(?<!\w)(?:\d[\d\s()+\-*/%.]*\d)(?!\w)", prompt)
    if not expression_match:
        return {"score": None, "metric": "arithmetic_independent", "status": "expression_not_found", "quality_confidence": "unavailable"}
    expression = expression_match.group(0).strip()
    try:
        expected = _safe_arithmetic(ast.parse(expression, mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return {"score": None, "metric": "arithmetic_independent", "status": "expression_not_evaluable", "quality_confidence": "unavailable"}
    numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", answer.replace(",", ""))
    predicted = float(numbers[-1]) if numbers else None
    reference_numbers = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", (reference_answer or "").replace(",", ""))
    reference_value = float(reference_numbers[-1]) if reference_numbers else None
    if reference_value is not None and not math.isclose(reference_value, expected, rel_tol=1e-9, abs_tol=1e-9):
        return {"score": None, "metric": "arithmetic_independent", "status": "reference_conflicts_with_calculation", "expected": expected, "reference": reference_value, "quality_confidence": "unavailable"}
    matched = predicted is not None and math.isclose(predicted, expected, rel_tol=1e-9, abs_tol=1e-9)
    return {
        "score": float(matched), "metric": "arithmetic_independent",
        "status": "measured_deterministic_match" if matched else "arithmetic_mismatch",
        "expected": expected, "predicted": predicted,
        "quality_confidence": "high", "factual_verification": "verified_by_deterministic_calculation",
    }


def evaluate_task_quality(
    task_type: str,
    answer: str,
    *,
    reference_answer: str | None = None,
    reference_sql: str | None = None,
    evaluation_type: str | None = None,
    metadata: dict[str, Any] | None = None,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Route one answer to the evaluator supported by its task and evidence."""
    category = {"general": "general_knowledge", "text_to_sql": "spider_text_to_sql"}.get(task_type, task_type)
    metadata = metadata or {}
    if category == "arithmetic":
        return evaluate_arithmetic(answer, prompt or "", reference_answer)
    if category == "spider_text_to_sql":
        return evaluate_spider_sql(answer, reference_sql)
    if category == "general_knowledge":
        if not reference_answer or not reference_answer.strip():
            return {
                "score": None, "metric": "semantic_reference_overlap",
                "status": "reference_answer_required", "quality_confidence": "unavailable",
                "factual_verification": "unverified",
            }
        return _reference_overlap(answer, reference_answer, metadata)
    if category in {"reasoning", "structured", "coding", "summarization"}:
        if not reference_answer or not reference_answer.strip():
            return {
                "score": None, "metric": "task_specific_evidence_required",
                "status": "reference_evidence_required", "quality_confidence": "unavailable",
            }
        kind = evaluation_type or metadata.get("evaluation_type")
        if kind in {"numeric_exact_match", "numeric_exact"}:
            kind = "numeric_exact"
        if kind in {"exact_match", "numeric_exact", "rouge", "code_execution", "instruction_following"}:
            sample = {"evaluation_type": kind, "reference_answer": reference_answer, "metadata": metadata}
            return evaluate_benchmark_sample(sample, answer)
        return _reference_overlap(answer, reference_answer, metadata)
    return {
        "score": None, "metric": "unavailable", "status": "unsupported_task_type",
        "quality_confidence": "unavailable",
    }
