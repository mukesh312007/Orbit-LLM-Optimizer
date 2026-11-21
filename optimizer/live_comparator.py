"""Prompt-specific, sequential local-model comparison before final generation."""
from __future__ import annotations

import os
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from optimizer.configuration_catalog import baseline_configuration, cache_server_arguments
from optimizer.live_matrix import full_live_candidates
from optimizer.request_router import classify_task, recommend_runtime
from optimizer.task_quality import evaluate_spider_sql, evaluate_task_quality  # compatibility hook for existing integrations
from optimizer.weighted_selection import DEFAULT_WEIGHTS, score_candidates
from runtime.clients import InferenceError, generate
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, ensure_runtime


LIVE_BENCHMARK_HISTORY = Path(__file__).resolve().parents[1] / "logs" / "live_benchmark_history.jsonl"


def _configuration(model: dict[str, Any]) -> dict[str, Any]:
    return baseline_configuration(model)


def _reference_evidence(
    reference_answer: str | None,
    reference_sql: str | None,
    *,
    source: str = "user_provided",
    model_id: str | None = None,
) -> dict[str, Any]:
    if reference_sql and reference_sql.strip():
        return {"kind": "reference_sql", "provided": True, "content": reference_sql, "source": source, "model_id": model_id}
    if reference_answer and reference_answer.strip():
        return {"kind": "reference_answer", "provided": True, "content": reference_answer, "source": source, "model_id": model_id}
    return {"kind": "none", "provided": False, "content": None, "source": source, "model_id": model_id}


def _store_live_benchmark_history(
    comparisons: list[dict[str, Any]],
    *,
    task_type: str,
    quality_threshold: float | None,
    reference_answer: str | None,
    reference_sql: str | None,
    reference_source: str = "user_provided",
    reference_model_id: str | None = None,
) -> None:
    """Persist one local evidence record per completed candidate evaluation."""
    if not comparisons:
        return
    evidence = _reference_evidence(reference_answer, reference_sql, source=reference_source, model_id=reference_model_id)
    LIVE_BENCHMARK_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with LIVE_BENCHMARK_HISTORY.open("a", encoding="utf-8") as handle:
        for entry in comparisons:
            detail = entry.get("task_quality_detail") or {}
            handle.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "live_comparison",
                "task_type": task_type,
                "evaluation_method": detail.get("metric"),
                "reference_evidence": evidence,
                "minimum_quality": quality_threshold,
                "quality_score": entry.get("task_quality"),
                "quality_confidence": entry.get("quality_confidence"),
                "quality_detail": detail,
                "model_id": entry["candidate"]["model"].get("model_id"),
                "configuration": entry.get("configuration"),
                "telemetry": {key: entry.get(key) for key in (
                    "latency_seconds", "ttft_seconds", "generated_tokens",
                    "generation_tokens_per_second", "peak_ram_mb", "energy_joules",
                    "energy_measurement_kind",
                )},
            }, ensure_ascii=False) + "\n")


def parse_judgement(text: str, candidate_count: int) -> int | None:
    """Return a valid, explicitly labelled judge choice, if one was supplied."""
    lowered = text.lower()
    if re.search(r"\b(?:winner\s*:\s*)?tie\b", lowered):
        return None
    match = re.search(r"\bwinner\s*:\s*(\d+)\b", lowered)
    if not match:
        match = re.fullmatch(r"\s*(?:candidate\s*)?(\d+)\s*[.!]?\s*", lowered)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return index if 0 <= index < candidate_count else None


def _model_balanced_runs(runs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Select a bounded matrix sample without repeatedly choosing one model first.

    A configuration-matrix run can contain dozens of settings per model.  Taking
    a prefix therefore makes a small smoke run look like an all-model comparison
    while it really measures only the first model.  Pick one deterministic
    configuration per model before selecting a second configuration for any
    model.  This is a live exploration aid, not a substitute for the complete
    benchmark campaign.
    """
    if limit >= len(runs):
        return list(runs)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        model_id = str(run["model"]["model_id"])
        grouped.setdefault(model_id, []).append(run)
    groups = [
        sorted(entries, key=lambda entry: str(entry["configuration"]["configuration_id"]))
        for _, entries in sorted(grouped.items())
    ]
    selected: list[dict[str, Any]] = []
    offsets = [0] * len(groups)
    while len(selected) < limit:
        made_progress = False
        for index, entries in enumerate(groups):
            if len(selected) >= limit:
                break
            if offsets[index] < len(entries):
                selected.append(entries[offsets[index]])
                offsets[index] += 1
                made_progress = True
        if not made_progress:
            break
    return selected


def compare_on_spot(
    prompt: str,
    models: list[dict[str, Any]],
    max_output_tokens: int,
    temperature: float,
    compare_all_models: bool = False,
    full_configuration_matrix: bool = False,
    reference_sql: str | None = None,
    reference_answer: str | None = None,
    evaluation_type: str | None = None,
    evaluation_metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    quality_threshold: float | None = None,
    weights: dict[str, float] | None = None,
    max_live_candidates: int | None = None,
    dry_run: bool = False,
    use_model_reference: bool = False,
    reference_model_id: str | None = None,
) -> dict[str, Any]:
    """Generate, measure, score, and select local candidate responses.

    This deliberately runs one server/model at a time to stay within laptop RAM limits.
    The RAM safety reserve is read from ORBIT_SAFETY_RESERVE_MB env var (default 512 MiB).
    """
    if quality_threshold is not None and not 0 <= quality_threshold <= 1:
        raise ValueError("quality_threshold must be between zero and one")
    if max_live_candidates is not None and max_live_candidates < 1:
        raise ValueError("max_live_candidates must be at least one when provided")
    if weights and any(value < 0 for value in weights.values()):
        raise ValueError("selection weights cannot be negative")
    # Read the safety reserve at call time so the dashboard slider (which writes
    # ORBIT_SAFETY_RESERVE_MB to os.environ) takes effect without a restart.
    reserve_mb = float(os.environ.get("ORBIT_SAFETY_RESERVE_MB", "512.0"))
    feasibility = recommend_runtime(prompt, models, safety_reserve_mb=reserve_mb)
    # A threshold is a hard claim about task quality.  Do not silently reuse a
    # historical Spider score for an unrelated live chat prompt: that would
    # make a quality-gated selection look measured when it is not.
    resolved_task_type = task_type or feasibility["task_type"]
    reference_source = "user_provided"
    reference_model: dict[str, Any] | None = None
    if (
        quality_threshold is not None
        and resolved_task_type == "general_knowledge"
        and not (reference_answer and reference_answer.strip())
        and use_model_reference
    ):
        requested_judge = reference_model_id or "qwen2.5-1.5b-q4_k_m"
        reference_model = next((model for model in models if model.get("model_id") == requested_judge and model.get("available_locally")), None)
        if reference_model is None:
            return {
                "selected": None, "task_type": resolved_task_type,
                "available_mb": feasibility["available_mb"], "candidates": feasibility["candidates"],
                "quality_threshold": quality_threshold,
                "reason": f"Quality-constrained selection is unavailable: judge model {requested_judge} is not available locally.",
                "selection_mode": "quality_constrained_unavailable",
                "reference_evidence": _reference_evidence(None, None, source="model_generated", model_id=requested_judge),
            }
    has_reference_evidence = bool(reference_sql and reference_sql.strip()) if resolved_task_type == "spider_text_to_sql" else bool(reference_answer and reference_answer.strip()) or reference_model is not None
    if quality_threshold is not None and not has_reference_evidence:
        return {
            "selected": None,
            "task_type": resolved_task_type,
            "available_mb": feasibility["available_mb"],
            "candidates": feasibility["candidates"],
            "quality_threshold": quality_threshold,
            "reason": (
                f"Quality-constrained selection is unavailable for {resolved_task_type}: "
                "valid task-specific reference evidence is required."
            ),
            "selection_mode": "quality_constrained_unavailable",
        }
    matrix_skips: list[dict[str, Any]] = []
    if full_configuration_matrix:
        runs, matrix_skips = full_live_candidates(models, safety_reserve_mb=reserve_mb)
        candidates = []
        resident_fallback = False
    elif compare_all_models:
        # Compare every installed local candidate one at a time. Only one
        # llama.cpp server is live, so model RAM is released before the next
        # candidate is loaded. Never bypass the admission gate merely because
        # the caller requested a broad comparison.
        candidates = [item for item in feasibility["candidates"] if item["feasible"]]
        if not candidates and feasibility.get("resident_fallback"):
            candidates = [feasibility["selected"]]
        runs = [{"model": item["model"], "configuration": _configuration(item["model"]), "candidate": item} for item in candidates]
        resident_fallback = feasibility.get("resident_fallback", False)
    else:
        candidates = [item for item in feasibility["candidates"] if item["feasible"]]
        resident_fallback = feasibility.get("resident_fallback", False)
        if not candidates and resident_fallback:
            # The model is already loaded and reusing it needs no additional RAM.
            candidates = [feasibility["selected"]]
        runs = [{"model": item["model"], "configuration": _configuration(item["model"]), "candidate": item} for item in candidates]
    if not runs:
        return {"selected": None, "task_type": resolved_task_type, "available_mb": feasibility["available_mb"], "candidates": feasibility["candidates"], "reason": feasibility.get("reason", "No candidate configuration is available for comparison."), "selection_mode": "quality_constrained" if quality_threshold is not None else "performance_only"}

    total_candidate_count = len(runs)
    candidate_model_ids = sorted({run["model"]["model_id"] for run in runs})
    deferred_runs: list[dict[str, Any]] = []
    if max_live_candidates is not None and len(runs) > max_live_candidates:
        scheduled_runs = _model_balanced_runs(runs, max_live_candidates)
        scheduled_keys = {
            (run["model"]["model_id"], run["configuration"]["configuration_id"])
            for run in scheduled_runs
        }
        deferred_runs = [
            run for run in runs
            if (run["model"]["model_id"], run["configuration"]["configuration_id"])
            not in scheduled_keys
        ]
        runs = scheduled_runs
    scheduled_model_ids = sorted({run["model"]["model_id"] for run in runs})
    deferred_model_ids = sorted({run["model"]["model_id"] for run in deferred_runs})
    if dry_run:
        return {
            "selected": None, "task_type": resolved_task_type, "available_mb": feasibility["available_mb"],
            "candidate_count": total_candidate_count, "scheduled_candidate_count": len(runs),
            "deferred_candidate_count": len(deferred_runs), "matrix_safety_skips": matrix_skips,
            "candidate_model_ids": candidate_model_ids, "scheduled_model_ids": scheduled_model_ids,
            "deferred_model_ids": deferred_model_ids,
            "quality_threshold": quality_threshold,
            "selection_mode": "quality_constrained" if quality_threshold is not None else "performance_only",
            "reason": "Dry run only: no model was loaded or response generated.",
        }

    if reference_model is not None:
        try:
            judge_configuration = _configuration(reference_model)
            ensure_runtime(reference_model, judge_configuration, cache_server_arguments(judge_configuration))
            judge_prompt = (
                "Answer the user's question with a concise factual reference answer. "
                "Return only the answer, with no discussion of being a judge or model.\n\n"
                f"User question:\n{prompt}"
            )
            prepared_judge_prompt, _ = prepare_prompt(prompt=judge_prompt, configuration=judge_configuration)
            judge_response = generate(
                reference_model, prepared_judge_prompt,
                {"backend": "llama.cpp", "max_output_tokens": max_output_tokens, "temperature": 0},
            )
            reference_answer = judge_response.get("answer", "").strip() or None
            if reference_answer:
                reference_source = "model_generated"
            else:
                reference_model = None
        except (InferenceError, RuntimeSwitchError):
            reference_model = None
        if reference_model is None or not reference_answer:
            return {
                "selected": None, "task_type": resolved_task_type,
                "available_mb": feasibility["available_mb"], "candidates": feasibility["candidates"],
                "quality_threshold": quality_threshold,
                "reason": "Quality-constrained selection is unavailable: the configured judge did not produce reference evidence.",
                "selection_mode": "quality_constrained_unavailable",
                "reference_evidence": _reference_evidence(None, None, source="model_generated", model_id=reference_model_id or "qwen2.5-1.5b-q4_k_m"),
            }

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in runs:
        model = run["model"]
        configuration = run["configuration"]
        # Do not use ``dict.get(key, run[...])`` here: Python evaluates the
        # fallback eagerly, which previously raised KeyError for matrix runs
        # that already supplied their candidate metadata.
        item = run.get("candidate")
        if item is None:
            item = {
                "model": model,
                "required_mb": run.get("required_mb"),
                "memory_source": "configuration safety estimate",
            }
        try:
            ensure_runtime(model, configuration, cache_server_arguments(configuration))
            candidate_prompt = prompt
            if resolved_task_type == "spider_text_to_sql":
                candidate_prompt = (
                    "Convert the user's database question into one read-only SQL query. "
                    "Return only the SQL query, with no explanation or markdown.\n\n"
                    f"User question:\n{prompt}"
                )
            prepared_prompt, preparation = prepare_prompt(candidate_prompt, configuration)
            response = generate(
                model,
                prepared_prompt,
                {
                    "backend": "llama.cpp",
                    # These are complete candidate answers, not short routing probes.
                    "max_output_tokens": max_output_tokens,
                    "temperature": temperature,
                },
            )
        except (InferenceError, RuntimeSwitchError) as error:
            failures.append({"candidate": item, "configuration": configuration, "error": str(error)})
            continue
        quality = evaluate_task_quality(
            resolved_task_type, response["answer"], reference_answer=reference_answer,
            reference_sql=reference_sql,
            evaluation_type=evaluation_type, metadata=evaluation_metadata, prompt=prompt,
        )
        watts = response.get("energy_proxy_watts")
        joules = response.get("energy_proxy_joules")
        task_quality = quality.get("score")
        comparisons.append({
            "candidate": item, "configuration": configuration, "draft": response["answer"],
            "latency_seconds": response["end_to_end_seconds"],
            "ttft_seconds": response.get("ttft_seconds"),
            "generated_tokens": response.get("generated_tokens"),
            "generation_tokens_per_second": response.get("generation_tokens_per_second"),
            "peak_ram_mb": response.get("peak_ram_mb"),
            "energy_joules": response.get("energy_joules"),
            "energy_source": response.get("energy_source"),
            "energy_proxy_watts": watts,
            "energy_proxy_joules": joules,
            "energy_proxy_method": response.get("energy_proxy_method"),
            "energy_measurement_kind": response.get("energy_measurement_kind"),
            "task_quality": task_quality,
            "quality_confidence": quality.get("quality_confidence"),
            "task_quality_detail": quality,
            "quality_per_proxy_joule": task_quality / joules if task_quality is not None and joules and joules > 0 else None,
            "quality_per_estimated_watt": task_quality / watts if task_quality is not None and watts else None,
            "quality_per_measured_joule": task_quality / response["energy_joules"] if task_quality is not None and response.get("energy_joules") and response["energy_joules"] > 0 else None,
            "energy_unit_note": "Proxy energy is displayed for exploration only and is excluded from selection; physical joules are used only when a validated meter is available.",
            "prompt_preparation": preparation,
        })

    _store_live_benchmark_history(
        comparisons,
        task_type=resolved_task_type,
        quality_threshold=quality_threshold,
        reference_answer=reference_answer,
        reference_sql=reference_sql,
        reference_source=reference_source,
        reference_model_id=reference_model.get("model_id") if reference_model else None,
    )

    if not comparisons:
        return {
            "selected": None, "task_type": feasibility["task_type"],
            "available_mb": feasibility["available_mb"], "candidates": feasibility["candidates"],
            "comparison_failures": failures,
            "reason": "No selected local model completed the response comparison.",
            "selection_mode": "quality_constrained" if quality_threshold is not None else "performance_only",
        }

    eligible_indices = [
        index for index, entry in enumerate(comparisons)
        if quality_threshold is None or (entry["task_quality"] is not None and entry["task_quality"] >= quality_threshold)
    ]
    for entry in comparisons:
        score_value = entry.get("task_quality")
        if quality_threshold is None:
            entry["eligibility"] = "NOT EVALUATED"
            entry["eligibility_reason"] = "No quality threshold was requested; this is an explicit performance-only comparison."
        elif score_value is None:
            entry["eligibility"] = "NOT EVALUATED"
            entry["eligibility_reason"] = "Quality evidence is unavailable."
        elif score_value >= quality_threshold:
            entry["eligibility"] = "PASS"
            entry["eligibility_reason"] = f"Quality {score_value:.3f} meets threshold {quality_threshold:.3f}."
        else:
            entry["eligibility"] = "FAIL"
            entry["eligibility_reason"] = f"Quality {score_value:.3f} is below threshold {quality_threshold:.3f}."
    quality_rejections = [
        {
            "model_id": entry["candidate"]["model"]["model_id"],
            "configuration_id": entry["configuration"]["configuration_id"],
            "reason": (
                "Task-valid quality is unavailable"
                if entry["task_quality"] is None
                else f"Task quality {entry['task_quality']:.3f} is below threshold {quality_threshold:.3f}"
            ),
        }
        for entry in comparisons
        if quality_threshold is not None
        and (entry["task_quality"] is None or entry["task_quality"] < quality_threshold)
    ]
    scoring_weights = dict(weights or {})
    if quality_threshold is not None:
        # Quality is a hard admission requirement. Once eligible, choose the
        # lowest-cost configuration instead of rewarding surplus quality twice.
        scoring_weights["task_quality"] = 0.0
    eligible_entries = [comparisons[index] for index in eligible_indices]
    scoring = score_candidates(eligible_entries, scoring_weights) if eligible_entries else {
        "weights": {**DEFAULT_WEIGHTS, **scoring_weights},
        "active_weights": {}, "excluded_metrics": {"all": "no quality-eligible candidates"},
    }
    eligible_ids = {id(entry) for entry in eligible_entries}
    for entry in comparisons:
        if id(entry) not in eligible_ids:
            entry["weighted_score"] = None
            entry["normalized_metrics"] = {}
    scored_indices = [index for index in eligible_indices if comparisons[index]["weighted_score"] is not None]
    if scored_indices:
        winner_index = max(scored_indices, key=lambda index: comparisons[index]["weighted_score"])
        basis = (
            f"Met quality threshold {quality_threshold:.3f}; selected by normalized weighted cost/efficiency score."
            if quality_threshold is not None else "Selected by normalized weighted score across the measured available metrics."
        )
    elif quality_threshold is not None:
        return {
            "selected": None, "task_type": resolved_task_type, "available_mb": feasibility["available_mb"],
            "candidates": feasibility["candidates"], "live_comparison": comparisons, "scoring": scoring,
            "comparison_failures": failures, "matrix_safety_skips": matrix_skips,
            "quality_threshold": quality_threshold,
            "quality_rejections": quality_rejections,
            "reason": "No suitable configuration found. No candidate satisfies the requested task-quality threshold. See quality_rejections for each candidate.",
            "selection_mode": "quality_constrained",
        }
    else:
        winner_index = min(range(len(comparisons)), key=lambda index: comparisons[index]["latency_seconds"])
        basis = "No weighted metrics were available; selected the lowest-latency completed response."
    winner = comparisons[winner_index]
    runtime_state = ensure_runtime(winner["candidate"]["model"], winner["configuration"], cache_server_arguments(winner["configuration"]))
    return {
        "selected": winner["candidate"], "configuration": winner["configuration"],
        "task_type": resolved_task_type, "available_mb": feasibility["available_mb"],
        "candidates": feasibility["candidates"], "reason": basis,
        "live_comparison": comparisons, "scoring": scoring,
        "comparison_failures": failures, "matrix_safety_skips": matrix_skips,
        "deferred_candidate_count": len(deferred_runs), "candidate_count": total_candidate_count,
        "candidate_model_ids": candidate_model_ids, "scheduled_model_ids": scheduled_model_ids,
        "deferred_model_ids": deferred_model_ids,
        "quality_threshold": quality_threshold,
        "quality_rejections": quality_rejections,
        "reference_evidence": _reference_evidence(
            reference_answer, reference_sql, source=reference_source,
            model_id=reference_model.get("model_id") if reference_model else None,
        ),
        "winner_index": winner_index, "resident_fallback": resident_fallback,
        "runtime_switched": runtime_state["switched"],
        "selection_mode": "quality_constrained" if quality_threshold is not None else "performance_only",
    }
