from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from core.model_registry import get_model, load_models
from optimizer.optimizer import OptimizationRequest, Optimizer
from optimizer.live_comparator import compare_on_spot
from optimizer.request_router import record_selection
from optimizer.profile_store import historical_profiles
from optimizer.calibrated_profiles import latest_profiles
from optimizer.evidence_report import profile_pareto_view, readiness_report, selection_history
from optimizer.live_matrix import configuration_is_safe, full_live_candidates
from optimizer.configuration_catalog import baseline_configuration, cache_server_arguments
from runtime.clients import InferenceError, backend_health, generate
from runtime.prompt_pipeline import prepare_prompt
from runtime.supervisor import RuntimeSwitchError, ensure_runtime
from database.telemetry import verify_jsonl_store
from database.verify_telemetry import verify as verify_mysql

app = FastAPI(title="Local LLM Runtime Optimizer", version="1.0.0")


class OptimizeBody(BaseModel):
    task_type: str = "all"
    minimum_quality: float | None = Field(None, ge=0, le=1)
    maximum_latency_seconds: float | None = Field(None, gt=0)
    maximum_ram_mb: float | None = Field(None, gt=0)
    maximum_energy_joules: float | None = Field(None, gt=0)
    preference: str = Field("balanced", pattern="^(quality|latency|memory|energy|balanced)$")


class GenerateBody(OptimizeBody):
    prompt: str = Field(min_length=1, max_length=20000)
    model_id: str | None = None
    max_output_tokens: int = Field(256, ge=1, le=2048)
    live_compare: bool = False
    full_configuration_matrix: bool = False
    reference_sql: str | None = None
    reference_answer: str | None = None
    use_model_reference: bool = False
    reference_model_id: str | None = None
    evaluation_type: str | None = None
    evaluation_metadata: dict[str, Any] | None = None
    selection_weights: dict[str, float] | None = None
    max_live_candidates: int | None = Field(None, ge=1)
    dry_run: bool = False


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "backends": {"llama.cpp": backend_health("llama.cpp"), "ollama": backend_health("ollama")}}


@app.get("/api/v1/models")
def models() -> list[dict[str, Any]]:
    return load_models()


@app.get("/api/v1/readiness")
def readiness() -> dict[str, Any]:
    """Expose local evidence readiness without starting a model or benchmark."""
    models = load_models()
    local_models = [model for model in models if model.get("available_locally") and "llama.cpp" in model.get("backends", [])]
    safe, skipped = full_live_candidates(local_models)
    profiles = latest_profiles()
    return {
        "local_only": True,
        "llama_cpp_available": backend_health("llama.cpp"),
        "available_models": [model["model_id"] for model in local_models],
        "safe_full_matrix_candidates": len(safe),
        "ram_safety_skips": len(skipped),
        "calibrated_profile_count": len(profiles),
        "quality_threshold_ready": bool(profiles),
        "energy_measurement_note": "Physical energy is unavailable unless a cumulative-joule meter export is configured.",
        "evidence": readiness_report(),
    }


@app.get("/api/v1/store-status")
def store_status() -> dict[str, Any]:
    """Report status of the operational primary JSONL store and optional MySQL mirror."""
    return {
        "primary_store": verify_jsonl_store(),
        "optional_mysql_mirror": verify_mysql(),
    }


@app.post("/api/v1/optimize")
def optimize(body: OptimizeBody) -> dict[str, Any]:
    return Optimizer().recommend(historical_profiles(), OptimizationRequest(**body.model_dump()))


def _profile_configuration(profile: dict[str, Any] | None, model: dict[str, Any]) -> dict[str, Any]:
    """Use the configuration that was measured, never a filename-only fallback."""
    configuration = dict((profile or {}).get("configuration") or baseline_configuration(model))
    required = {"configuration_id", "threads", "context_size"}
    if not required.issubset(configuration):
        configuration = baseline_configuration(model)
    configuration.setdefault("backend", "llama.cpp")
    configuration.setdefault("gpu_layers", 0)
    configuration.setdefault("cache", "server-default")
    configuration.setdefault("prompt_compression", "none")
    configuration.setdefault("retrieval", {"enabled": False})
    return configuration


@app.post("/api/v1/generate")
def generate_request(body: GenerateBody) -> dict[str, Any]:
    if body.live_compare:
        try:
            decision = compare_on_spot(
                body.prompt, load_models(), body.max_output_tokens, 0,
                compare_all_models=True, full_configuration_matrix=body.full_configuration_matrix,
                reference_sql=body.reference_sql,
                reference_answer=body.reference_answer, evaluation_type=body.evaluation_type,
                evaluation_metadata=body.evaluation_metadata, task_type=None if body.task_type == "all" else body.task_type,
                quality_threshold=body.minimum_quality,
                use_model_reference=body.use_model_reference, reference_model_id=body.reference_model_id,
                weights=body.selection_weights, max_live_candidates=body.max_live_candidates, dry_run=body.dry_run,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if body.dry_run:
            return {"answer": None, "selected_model": None, "configuration": None, "telemetry": None, "recommendation": decision, "recommendation_is_historical": False}
        if not decision.get("selected"):
            raise HTTPException(422, decision["reason"])
        winner = decision["live_comparison"][decision["winner_index"]]
        persistence = record_selection(decision, winner | {"answer": winner["draft"]})
        return {"answer": winner["draft"], "selected_model": decision["selected"]["model"]["model_id"], "configuration": decision["configuration"], "telemetry": winner, "recommendation": decision, "telemetry_persistence": persistence, "recommendation_is_historical": False}
    recommendation = Optimizer().recommend(historical_profiles(), OptimizationRequest(**body.model_dump(exclude={
        "prompt", "model_id", "max_output_tokens", "live_compare", "full_configuration_matrix", "reference_sql", "reference_answer", "use_model_reference", "reference_model_id", "evaluation_type", "evaluation_metadata", "selection_weights", "max_live_candidates", "dry_run",
    })))
    selected_profile = recommendation.get("selected")
    if body.minimum_quality is not None and not selected_profile:
        raise HTTPException(422, recommendation.get("explanation", "No measured profile satisfies the requested constraints."))
    selected_id = body.model_id or (selected_profile or {}).get("model_id")
    model = get_model(selected_id) if selected_id else None
    if model is None and body.model_id is None:
        raise HTTPException(503, "No available registered model matches the measured recommendation.")
    if model is None:
        raise HTTPException(422, "The requested model is not registered and available locally.")
    if not model:
        raise HTTPException(503, "No requested local model is registered and available.")
    configuration = _profile_configuration(None if body.model_id else selected_profile, model)
    safe, safety_reason, _ = configuration_is_safe(model, configuration)
    if not safe:
        raise HTTPException(422, f"The selected configuration is unsafe to start: {safety_reason}")
    try:
        runtime_state = ensure_runtime(model, configuration, cache_server_arguments(configuration))
        prepared_prompt, preparation = prepare_prompt(body.prompt, configuration)
        inference = generate(model, prepared_prompt, {
            "backend": configuration["backend"], "max_output_tokens": body.max_output_tokens,
            "temperature": 0,
        })
    except (InferenceError, RuntimeSwitchError) as error:
        raise HTTPException(503, str(error)) from error
    execution_decision = {
        "selected": {"model": model}, "configuration": configuration,
        "task_type": body.task_type, "runtime_switched": runtime_state["switched"],
    }
    persistence = record_selection(execution_decision, inference | {"runtime_switched": runtime_state["switched"]})
    return {
        "answer": inference["answer"], "selected_model": model["model_id"], "configuration": configuration,
        "telemetry": inference, "prompt_preparation": preparation, "recommendation": recommendation,
        "telemetry_persistence": persistence,
        "recommendation_is_historical": bool(selected_profile and body.model_id is None),
        "user_override": bool(body.model_id and (not selected_profile or body.model_id != selected_profile.get("model_id"))),
    }


@app.get("/api/v1/experiments")
def experiments() -> list[dict[str, Any]]:
    return historical_profiles()


@app.get("/api/v1/history")
def history() -> list[dict[str, Any]]:
    """Recent private selection metadata; response text is never returned."""
    return selection_history()


@app.get("/api/v1/pareto")
def pareto() -> list[dict[str, Any]]:
    """Measured, non-dominated Spider profiles only."""
    return profile_pareto_view()


@app.get("/api/v1/recommendations")
def recommendations() -> dict[str, Any]:
    return Optimizer().recommend(historical_profiles(), OptimizationRequest(task_type="all"))
