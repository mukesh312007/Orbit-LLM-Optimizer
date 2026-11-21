"""Orbit: a focused, local-first chat workspace."""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import streamlit as st

from core.model_registry import load_models
from core.settings import settings
from optimizer.live_comparator import compare_on_spot
from optimizer.evidence_report import dataset_readiness, energy_readiness, profile_comparison_view, profile_pareto_view, selection_history
from optimizer.request_router import record_selection, selection_caption
from runtime.clients import InferenceError, backend_health
from runtime.supervisor import RuntimeSwitchError, ensure_runtime

st.set_page_config(
    page_title="Orbit",
    page_icon=":material/smart_toy:",
    layout="centered",
    initial_sidebar_state="expanded",
)

SUGGESTIONS = {
    ":material/lightbulb: Brainstorm an idea": "Help me brainstorm a practical project idea.",
    ":material/edit_note: Improve my writing": "Help me improve this writing: ",
    ":material/code: Solve a coding problem": "Help me solve this programming problem: ",
}


@st.cache_data(ttl=2)
def model_is_ready() -> bool:
    return backend_health("llama.cpp")


def local_models() -> list[dict[str, Any]]:
    return [model for model in load_models() if model["available_locally"]]


def clear_conversation() -> None:
    st.session_state.messages = []
    st.session_state.last_telemetry = None


if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_telemetry" not in st.session_state:
    st.session_state.last_telemetry = None

models = local_models()
installed_model_ids = sorted(model["model_id"] for model in models)
runtime_model = models[0] if models else None
ready = model_is_ready()

with st.sidebar:
    st.title("Orbit", icon=":material/smart_toy:")
    st.caption("Your local AI workspace")
    if ready:
        st.badge("Local model online", icon=":material/check_circle:", color="green")
    else:
        st.badge("Local model offline", icon=":material/pending:", color="orange")
    st.space("medium")
    if st.button("New chat", icon=":material/add:", width="stretch", key="new_chat"):
        clear_conversation()
        st.rerun()
    st.space("small")
    temperature = st.slider("Creative latitude", 0.0, 1.0, 0.2, 0.1, key="temperature")
    max_tokens = st.slider("Response length", 64, 512, 256, 32, key="max_tokens")
    with st.expander("Optimization settings", expanded=False):
        task_type = st.selectbox(
            "Task category", ["auto", "general_knowledge", "spider_text_to_sql", "reasoning", "structured", "coding", "summarization"],
            key="task_type", help="Auto routes SQL, general knowledge, reasoning, and structured prompts to their supported evaluator.",
        )
        quality_gate = st.toggle(
            "Require quality-gated selection",
            value=True,
            key="quality_gate_required",
            help="Orbit will not select a model without valid task-specific quality evidence. Disable only for an explicitly labelled performance-only benchmark.",
        )
        quality_threshold = (
            st.slider("Minimum task quality", 0.0, 1.0, 0.8, 0.05, key="quality_threshold")
            if quality_gate else None
        )
        reference_sql = st.text_area("Gold SQL reference", key="reference_sql", help="Required only for SQL/database tasks.") if quality_gate and task_type == "spider_text_to_sql" else None
        reference_answer = st.text_area("Reference answer", key="reference_answer", help="Use a verified answer when available. Semantic overlap does not prove factual correctness.") if quality_gate and task_type != "spider_text_to_sql" else None
        judge_options = [model_id for model_id in installed_model_ids if model_id != "qwen2.5-3b-q4_k_m"]
        default_judge = "qwen2.5-1.5b-q4_k_m" if "qwen2.5-1.5b-q4_k_m" in judge_options else (judge_options[0] if judge_options else None)
        use_model_reference = False
        reference_model_id = None
        if quality_gate and task_type == "general_knowledge" and not (reference_answer or "").strip() and judge_options:
            use_model_reference = st.checkbox(
                "Use a local Qwen model as a provisional reference judge",
                value=True,
                key="use_model_reference",
                help="Generates one reference answer with a separate model. This is model-generated evidence, not fact verification.",
            )
            reference_model_id = st.selectbox(
                "Reference judge model",
                judge_options,
                index=judge_options.index(default_judge) if default_judge in judge_options else 0,
                key="reference_model_id",
            ) if use_model_reference else None
        evaluation_type = st.selectbox("Evaluation method", ["semantic_reference_overlap", "exact_match", "numeric_exact", "rouge", "code_execution"], key="evaluation_type") if quality_gate and task_type in {"reasoning", "structured", "coding", "summarization"} else None
        full_configuration_matrix = st.toggle("Compare safe runtime configurations", key="full_configuration_matrix")
        max_live_candidates = (
            int(st.number_input(
                "Maximum live configurations",
                min_value=1,
                max_value=720,
                value=max(1, min(len(installed_model_ids), 720)),
                step=1,
                key="max_live_candidates_model_balanced",
                help=(
                    "A limited matrix run schedules one configuration per installed model before it schedules "
                    "additional variants. Use at least the installed-model count to cover every model."
                ),
            ))
            if full_configuration_matrix else None
        )
        safety_reserve_mb = float(
            st.slider(
                "RAM safety reserve (MiB)",
                min_value=256,
                max_value=2048,
                value=int(getattr(settings, "safety_reserve_mb", 512.0)),
                step=128,
                key="safety_reserve_mb",
                help="Headroom kept free for the operating system, Streamlit, and background processes.",
            )
        )
        # Write to env so compare_on_spot and recommend_runtime pick it up at call time.
        os.environ["ORBIT_SAFETY_RESERVE_MB"] = str(int(safety_reserve_mb))
        st.caption(
            f"{'Quality-gated selection requires valid task evidence.' if quality_gate else 'Current mode: performance-only comparison; no quality claim is made.'} "
            f"Full comparison runs sequentially and preserves the {safety_reserve_mb:.0f} MiB RAM reserve."
        )
    st.caption(
        f"Matrix exploration covers distinct models first; the current cap is {max_live_candidates} configuration(s) across {len(installed_model_ids)} installed model(s)."
        if full_configuration_matrix else f"Each message compares all {len(models)} installed local models sequentially."
    )
    st.caption("Installed models: " + (", ".join(installed_model_ids) if installed_model_ids else "none"))
    st.space("medium")
    st.caption("Private by design. Requests stay on this machine.")

if not st.session_state.messages:
    st.space("large")
    st.title("What can we figure out?", icon=":material/auto_awesome:")
    st.caption("Orbit is ready to help you reason, write, code, and explore.")
    st.space("medium")
    selected_suggestion = st.pills(
        "Start a conversation",
        list(SUGGESTIONS),
        selection_mode="single",
        label_visibility="collapsed",
        key="suggestions",
    )
else:
    st.title("Orbit", icon=":material/smart_toy:")
    st.caption("A private conversation with your local model")
    selected_suggestion = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("runtime"):
            st.caption(message["runtime"])

prompt = st.chat_input("Message Orbit", submit_mode="disable", key="chat_input")
if selected_suggestion:
    prompt = SUGGESTIONS[selected_suggestion]

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar=":material/smart_toy:"):
        selected = None
        decision: dict[str, Any] | None = None
        runtime_note: str | None = None
        if runtime_model is None:
            response_text = "No local GGUF model is registered. Add a model to the models folder, then restart Orbit."
            st.error(response_text, icon=":material/error:")
        else:
            started = time.perf_counter()
            completed = False
            with st.status(":shimmer[Generating and measuring every local Qwen response]", type="compact") as status:
                try:
                    decision = compare_on_spot(
                        prompt,
                        models,
                        max_tokens,
                        temperature,
                        compare_all_models=True,
                        full_configuration_matrix=full_configuration_matrix,
                        reference_sql=reference_sql,
                        reference_answer=reference_answer,
                        use_model_reference=use_model_reference,
                        reference_model_id=reference_model_id,
                        evaluation_type=evaluation_type,
                        task_type=None if task_type == "auto" else task_type,
                        quality_threshold=quality_threshold,
                        max_live_candidates=max_live_candidates,
                    )
                    selected = decision.get("selected")
                    if selected is None:
                        response_text = decision["reason"]
                        status.update(label="No suitable configuration found", state="error")
                        st.warning(response_text, icon=":material/verified:")
                        # Preserve the full live comparison so failed quality gates
                        # remain diagnosable instead of collapsing to one sentence.
                        st.session_state.last_telemetry = {
                            "final_generation": None,
                            "runtime_state": None,
                            "decision": decision,
                        }
                    else:
                        selected_model = selected["model"]
                        runtime_state = ensure_runtime(selected_model, decision["configuration"])
                        winner = decision["live_comparison"][decision["winner_index"]]
                        # Use the exact response that won the live comparison. Regenerating here
                        # would sample a different response after the model decision was made.
                        response_text = winner["draft"]
                        result = {
                            "answer": response_text,
                            "backend": "llama.cpp",
                            "end_to_end_seconds": winner["latency_seconds"],
                            "ttft_seconds": winner.get("ttft_seconds"),
                            "generated_tokens": winner.get("generated_tokens"),
                            "generation_tokens_per_second": winner.get("generation_tokens_per_second"),
                            "peak_ram_mb": winner.get("peak_ram_mb"),
                            "energy_proxy_watts": winner.get("energy_proxy_watts"),
                            "energy_proxy_joules": winner.get("energy_proxy_joules"),
                            "energy_measurement_kind": winner.get("energy_measurement_kind"),
                            "task_quality": winner.get("task_quality"),
                            "task_quality_detail": winner.get("task_quality_detail"),
                            "quality_confidence": winner.get("quality_confidence"),
                            "quality_per_watt": winner.get("quality_per_watt"),
                            "weighted_score": winner.get("weighted_score"),
                            "measurement_kind": "selected_candidate_response",
                        }
                        elapsed = round(time.perf_counter() - started, 1)
                        status.update(label=f"Selected response ready · {elapsed}s", state="complete")
                        st.session_state.last_telemetry = {
                            "final_generation": result,
                            "runtime_state": runtime_state,
                            "decision": decision,
                        }
                        record_selection(decision, result | {"runtime_switched": runtime_state["switched"]})
                        runtime_note = selection_caption(decision)
                        completed = True
                except (InferenceError, RuntimeSwitchError) as error:
                    response_text = f"Orbit could not complete that request: {error}"
                    status.update(label="Response unavailable", state="error")
                    st.error(response_text, icon=":material/error:")
            if completed:
                st.markdown(response_text)
                st.caption(runtime_note)
    st.session_state.messages.append({"role": "assistant", "content": response_text, "runtime": runtime_note})

if st.session_state.last_telemetry:
    with st.expander("Response details", icon=":material/info:"):
        details = st.session_state.last_telemetry
        raw_decision = details.get("decision")
        decision = raw_decision if isinstance(raw_decision, dict) else {}
        comparison = decision.get("live_comparison", [])
        failures = decision.get("comparison_failures", [])
        if comparison:
            st.subheader("On-the-spot model comparison")
            st.caption(f"Live benchmark evidence: candidates were generated sequentially under the same requested conditions. Selection mode: {decision.get('selection_mode', 'unavailable')}. Historical benchmark evidence is shown separately below.")
            rows = []
            selected_record = decision.get("selected") or {}
            selected_id = (selected_record.get("model") or {}).get("model_id")
            benchmark_profiles = {
                row["configuration_id"]: row
                for row in profile_comparison_view()
                if row.get("configuration_id") and row.get("quality") is not None
            }
            for index, entry in enumerate(comparison, start=1):
                configuration = entry["configuration"]
                model_id = entry["candidate"]["model"]["model_id"]
                benchmark = benchmark_profiles.get(configuration["configuration_id"])
                rows.append({
                    "Candidate": index,
                    "Model": model_id,
                    "Selected": model_id == selected_id,
                    "Backend": configuration["backend"],
                    "Threads": configuration["threads"],
                    "Context": configuration["context_size"],
                    "Cache": configuration["cache"],
                    "Outcome": entry.get("eligibility", "NOT EVALUATED"),
                    "Detail": entry.get("eligibility_reason", "Response generated"),
                    "Response latency (s)": round(entry["latency_seconds"], 3),
                    "TTFT (s)": round(entry["ttft_seconds"], 3) if entry.get("ttft_seconds") is not None else None,
                    "Generated tokens": entry.get("generated_tokens"),
                    "Tokens / second": round(entry["generation_tokens_per_second"], 2) if entry.get("generation_tokens_per_second") else None,
                    "Peak RAM (MiB)": round(entry["peak_ram_mb"], 1) if entry.get("peak_ram_mb") is not None else None,
                    "Energy proxy (J)": round(entry["energy_proxy_joules"], 2) if entry.get("energy_proxy_joules") is not None else None,
                    "Estimated power (W)": round(entry["energy_proxy_watts"], 2) if entry.get("energy_proxy_watts") is not None else None,
                    "Live task quality": entry.get("task_quality"),
                    "Evaluation method": (entry.get("task_quality_detail") or {}).get("metric"),
                    "Quality confidence": entry.get("quality_confidence"),
                    "Benchmark quality (Spider)": benchmark.get("quality") if benchmark else None,
                    "Benchmark quality LCB": benchmark.get("quality_lower_confidence_bound") if benchmark else None,
                    "Benchmark trials": benchmark.get("trial_count") if benchmark else None,
                    "Benchmark evidence": "Exact configuration" if benchmark else "No exact configuration benchmark",
                    "Quality / proxy J": round(entry["quality_per_proxy_joule"], 4) if entry.get("quality_per_proxy_joule") is not None else None,
                    "Quality / estimated W": round(entry["quality_per_estimated_watt"], 4) if entry.get("quality_per_estimated_watt") is not None else None,
                    "Energy status": entry.get("energy_measurement_kind") or "proxy only",
                    "Weighted score": round(entry["weighted_score"], 4) if entry.get("weighted_score") is not None else None,
                })
            for entry in failures:
                configuration = entry["configuration"]
                rows.append({
                    "Candidate": "—",
                    "Model": entry["candidate"]["model"]["model_id"],
                    "Selected": False,
                    "Backend": configuration["backend"],
                    "Threads": configuration["threads"],
                    "Context": configuration["context_size"],
                    "Cache": configuration["cache"],
                    "Outcome": "NOT EVALUATED",
                    "Detail": entry["error"],
                    "Response latency (s)": None,
                    "TTFT (s)": None,
                    "Generated tokens": None,
                    "Tokens / second": None,
                    "Peak RAM (MiB)": None,
                    "Energy proxy (J)": None,
                    "Estimated power (W)": None,
                    "Live task quality": None,
                    "Benchmark quality (Spider)": None,
                    "Benchmark quality LCB": None,
                    "Benchmark trials": None,
                    "Benchmark evidence": "No generated response",
                    "Quality / proxy J": None,
                    "Quality / estimated W": None,
                    "Energy status": "unavailable",
                    "Weighted score": None,
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, alt="Per-model local quality, performance, memory, and energy-proxy comparison")
            st.caption(decision.get("reason", "No decision reason recorded."))
            if decision.get("quality_threshold") is not None:
                st.caption(f"Minimum quality: {decision['quality_threshold']:.2f}. Candidates without valid task-quality evidence are rejected.")
            st.subheader("Task-aware evaluation")
            st.json({
                "task_category": decision.get("task_type"),
                "selection_mode": decision.get("selection_mode"),
                "quality_threshold": decision.get("quality_threshold"),
                "reference_evidence": decision.get("reference_evidence", "see candidate details"),
            })
            if decision.get("deferred_candidate_count"):
                st.caption(f"{decision['deferred_candidate_count']} safe configurations were deferred by the live-run limit and were not compared in this response.")
            candidate_models = decision.get("candidate_model_ids", [])
            scheduled_models = decision.get("scheduled_model_ids", [])
            if candidate_models:
                coverage_rows = [
                    {
                        "Model": model_id,
                        "Status": "Compared" if model_id in scheduled_models else "Not scheduled",
                        "Reason": "Live response generated or reported as unavailable" if model_id in scheduled_models else "Configuration cap reached before this model was scheduled",
                    }
                    for model_id in candidate_models
                ]
                st.subheader("Candidate coverage")
                st.dataframe(pd.DataFrame(coverage_rows), hide_index=True, alt="Installed-model coverage for this live comparison")
            st.subheader("Weighted selection")
            st.json(decision.get("scoring", {}))
            st.subheader("Complete candidate responses")
            for index, entry in enumerate(comparison, start=1):
                st.markdown(f"**Candidate {index} — {entry['candidate']['model']['model_id']}**")
                st.code(entry["draft"], language=None)
        st.subheader("Selected response telemetry")
        st.json({
                        "selected_model": ((decision.get("selected") or {}).get("model") or {}).get("model_id"),
                        "selection_mode": decision.get("selection_mode"),
            "configuration": decision.get("configuration"),
                        "runtime_switched": (details.get("runtime_state") or {}).get("switched"),
            "telemetry": details.get("final_generation"),
        })
        with st.expander("Evidence, history, and provenance", expanded=False, icon=":material/analytics:"):
            st.caption("Local evidence only. Response text is not retained in history unless explicitly enabled in local settings.")
            rejection_rows = []
            for entry in decision.get("quality_rejections", []):
                rejection_rows.append({"Type": "Quality gate", "Model": entry.get("model_id"), "Configuration": entry.get("configuration_id"), "Reason": entry.get("reason")})
            for entry in decision.get("matrix_safety_skips", []):
                model = entry.get("model") or {}
                configuration = entry.get("configuration") or {}
                rejection_rows.append({"Type": "RAM safety", "Model": model.get("model_id"), "Configuration": configuration.get("configuration_id"), "Reason": entry.get("reason")})
            if rejection_rows:
                st.subheader("Rejection reasons")
                st.dataframe(pd.DataFrame(rejection_rows), hide_index=True, alt="Quality-gate and memory-safety rejection reasons")

            pareto_rows = profile_pareto_view()
            if pareto_rows:
                evidence_models = {row.get("model_id") for row in pareto_rows}
                missing_evidence = [model_id for model_id in installed_model_ids if model_id not in evidence_models]
                st.subheader("Historical measured Pareto frontier")
                st.caption("This is stored Spider benchmark evidence, not the current live comparison. Higher quality, lower latency, and lower RAM are preferred; proxy energy remains an estimate and is excluded from automated selection.")
                if missing_evidence:
                    st.warning(
                        "Historical benchmark coverage is incomplete for: " + ", ".join(missing_evidence) + ". "
                        "This affects only the historical evidence display; it does not reject or qualify live comparison candidates.",
                        icon=":material/info:",
                    )
                st.dataframe(pd.DataFrame(pareto_rows), hide_index=True, alt="Measured non-dominated Spider benchmark profiles")

            comparison_rows = profile_comparison_view()
            if comparison_rows:
                comparison_frame = pd.DataFrame(comparison_rows).rename(columns={
                    "quality": "Quality", "latency_seconds": "Latency (s)", "ram_mb": "Peak RAM (MiB)",
                    "model_id": "Model", "sample_count": "Samples", "trial_count": "Trials",
                })
                st.subheader("Measured trade-offs")
                st.caption("Each point is a recorded Spider configuration. Profiles with fewer than three recorded trials are shown for transparency but are not eligible for confidence-aware quality selection.")
                chart_columns = st.columns(2)
                with chart_columns[0]:
                    st.scatter_chart(comparison_frame, x="Latency (s)", y="Quality", color="Model", alt="Measured Spider quality versus end-to-end latency by configuration")
                with chart_columns[1]:
                    st.scatter_chart(comparison_frame, x="Peak RAM (MiB)", y="Quality", color="Model", alt="Measured Spider quality versus peak RAM by configuration")
                st.dataframe(
                    comparison_frame[["Model", "Quality", "Latency (s)", "Peak RAM (MiB)", "Samples", "Trials", "reproducibility_status", "energy_joules", "energy_proxy_joules"]],
                    hide_index=True,
                    alt="Measured configuration coverage, confidence, actual energy, and proxy energy",
                )

            history_rows = selection_history()
            if history_rows:
                st.subheader("Recent local selections")
                st.dataframe(pd.DataFrame(history_rows), hide_index=True, alt="Recent private local model selections without response text")

            st.subheader("Dataset and energy readiness")
            st.dataframe(pd.DataFrame(dataset_readiness()), hide_index=True, alt="Prepared local general-quality dataset readiness")
            meter = energy_readiness()
            st.json({
                "physical_energy_available": meter.get("actual_energy_available"),
                "meter_export_exists": meter.get("meter_export_exists"),
                "configured_meter_path": meter.get("configured_meter_path"),
                "note": meter.get("proxy_note"),
            })
            st.caption("Each benchmark JSONL record includes model hash, runtime configuration, llama.cpp version, and local hardware snapshot.")
