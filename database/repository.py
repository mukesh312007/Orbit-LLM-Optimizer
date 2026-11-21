from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from database.schema import (
    BenchmarkExperiment,
    BenchmarkResult,
    EnergyMetric,
    Model,
    OptimizationRecommendation,
    PerformanceMetric,
    QualityResult,
    ResourceMetric,
    RuntimeConfiguration,
    create_schema,
    make_engine,
)


class ExperimentRepository:
    """Small transactional repository. Production URL must point to MySQL."""
    def __init__(self, url: str, create: bool = False):
        self.engine = make_engine(url)
        if create:
            create_schema(url)
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    def upsert_model(self, model: dict[str, Any]) -> None:
        with self.sessions.begin() as session:
            existing = session.get(Model, model["model_id"])
            if existing:
                existing.family, existing.quantization, existing.metadata_json = model["family"], model.get("quantization"), model
            else:
                session.add(Model(model_id=model["model_id"], family=model["family"], quantization=model.get("quantization"), metadata_json=model))

    def save_experiment(self, experiment_id: str, kind: str, model_id: str | None = None, configuration_id: str | None = None, dataset_id: str | None = None, metadata: dict[str, Any] | None = None) -> None:
        with self.sessions.begin() as session:
            if not session.get(BenchmarkExperiment, experiment_id):
                session.add(BenchmarkExperiment(experiment_id=experiment_id, kind=kind, model_id=model_id, configuration_id=configuration_id, dataset_id=dataset_id, metadata_json=metadata or {}))

    def save_hardware_profile(self, profile: dict[str, Any]) -> None:
        """Preserve an immutable hardware snapshot alongside experiments."""
        from database.schema import HardwareProfile
        with self.sessions.begin() as session:
            session.add(HardwareProfile(profile_json=profile))

    def save_live_selection(
        self,
        experiment_id: str,
        model: dict[str, Any],
        configuration: dict[str, Any],
        task_type: str,
        telemetry: dict[str, Any],
    ) -> None:
        """Persist one already-completed local selection in a single transaction.

        Energy proxies are retained as experiment metadata only.  The energy
        table is deliberately left null until an actual cumulative-joule meter
        is connected.
        """
        with self.sessions.begin() as session:
            stored_model = session.get(Model, model["model_id"])
            if stored_model:
                stored_model.family = model["family"]
                stored_model.quantization = model.get("quantization")
                stored_model.metadata_json = model
            else:
                session.add(Model(
                    model_id=model["model_id"], family=model["family"],
                    quantization=model.get("quantization"), metadata_json=model,
                ))
            configuration_id = configuration["configuration_id"]
            stored_configuration = session.get(RuntimeConfiguration, configuration_id)
            if stored_configuration:
                stored_configuration.model_id = model["model_id"]
                stored_configuration.backend = configuration.get("backend", "llama.cpp")
                stored_configuration.settings_json = configuration
            else:
                session.add(RuntimeConfiguration(
                    configuration_id=configuration_id, model_id=model["model_id"],
                    backend=configuration.get("backend", "llama.cpp"), settings_json=configuration,
                ))
            session.flush()
            session.add(BenchmarkExperiment(
                experiment_id=experiment_id, kind="live_selection", model_id=model["model_id"],
                configuration_id=configuration_id,
                metadata_json={
                    "task_type": task_type,
                    "runtime_switched": telemetry.get("runtime_switched"),
                    "energy_proxy_joules": telemetry.get("energy_proxy_joules"),
                    "energy_proxy_method": telemetry.get("energy_proxy_method"),
                },
            ))
            result = BenchmarkResult(
                experiment_id=experiment_id, prompt_id=None, trial_number=1,
                response_text=telemetry.get("answer"),
            )
            session.add(result)
            session.flush()
            session.add(PerformanceMetric(
                result_id=result.result_id, ttft_seconds=telemetry.get("ttft_seconds"),
                tpot_ms=None, end_to_end_seconds=telemetry.get("end_to_end_seconds"),
                generation_tokens_per_second=telemetry.get("generation_tokens_per_second"),
                generated_tokens=telemetry.get("generated_tokens"),
            ))
            session.add(ResourceMetric(
                result_id=result.result_id, process_memory_mb=telemetry.get("peak_ram_mb"),
                system_memory_percent=None, cpu_percent=telemetry.get("average_system_cpu_percent"),
                gpu_utilization_percent=None,
            ))
            session.add(EnergyMetric(
                result_id=result.result_id, energy_joules=telemetry.get("energy_joules"),
                energy_proxy_joules=telemetry.get("energy_proxy_joules"),
                source=telemetry.get("energy_source"), measured=telemetry.get("energy_measurement_kind") == "measured",
                measurement_kind=telemetry.get("energy_measurement_kind") or "unavailable",
                proxy_method=telemetry.get("energy_proxy_method"),
            ))

    def save_benchmark_trial(self, experiment_id: str, kind: str, record: dict[str, Any]) -> None:
        """Store one benchmark record and its labelled metrics transactionally."""
        model = record["model"]
        configuration = record["configuration"]
        telemetry = record.get("telemetry") or {}
        quality = record.get("task_quality") or {}
        prompt_id = record.get("database_prompt_id")
        with self.sessions.begin() as session:
            stored_model = session.get(Model, model["model_id"])
            if stored_model:
                stored_model.family = model.get("family", "Unknown local model")
                stored_model.quantization = model.get("quantization")
                stored_model.metadata_json = model
            else:
                session.add(Model(model_id=model["model_id"], family=model.get("family", "Unknown local model"), quantization=model.get("quantization"), metadata_json=model))
            configuration_id = configuration["configuration_id"]
            stored_configuration = session.get(RuntimeConfiguration, configuration_id)
            if stored_configuration:
                stored_configuration.model_id = model["model_id"]
                stored_configuration.backend = configuration.get("backend", "llama.cpp")
                stored_configuration.settings_json = configuration
            else:
                session.add(RuntimeConfiguration(configuration_id=configuration_id, model_id=model["model_id"], backend=configuration.get("backend", "llama.cpp"), settings_json=configuration))
            session.flush()
            if not session.get(BenchmarkExperiment, experiment_id):
                session.add(BenchmarkExperiment(
                    experiment_id=experiment_id, kind=kind, model_id=model["model_id"], configuration_id=configuration_id,
                    metadata_json={
                        "parent_experiment_id": record["experiment_id"],
                        "measurement_kind": record.get("measurement_kind"),
                        "provenance": record.get("provenance"),
                    },
                ))
            existing = session.scalar(select(BenchmarkResult).where(
                BenchmarkResult.experiment_id == experiment_id,
                BenchmarkResult.prompt_id == prompt_id,
                BenchmarkResult.trial_number == record.get("trial_number"),
            ))
            if existing:
                return
            result = BenchmarkResult(experiment_id=experiment_id, prompt_id=prompt_id, trial_number=record.get("trial_number"), response_text=record.get("response"))
            session.add(result)
            session.flush()
            session.add(QualityResult(result_id=result.result_id, metric=quality.get("metric", "unavailable"), score=quality.get("score"), source="automated", details_json=quality))
            session.add(PerformanceMetric(result_id=result.result_id, ttft_seconds=telemetry.get("ttft_seconds"), tpot_ms=None, end_to_end_seconds=telemetry.get("end_to_end_seconds"), generation_tokens_per_second=telemetry.get("generation_tokens_per_second"), generated_tokens=telemetry.get("generated_tokens")))
            session.add(ResourceMetric(result_id=result.result_id, process_memory_mb=telemetry.get("peak_ram_mb"), system_memory_percent=None, cpu_percent=telemetry.get("average_system_cpu_percent"), gpu_utilization_percent=None))
            session.add(EnergyMetric(result_id=result.result_id,
                energy_joules=telemetry.get("energy_joules"),
                energy_proxy_joules=telemetry.get("energy_proxy_joules"),
                source=telemetry.get("energy_source"),
                measured=telemetry.get("energy_measurement_kind") == "measured",
                measurement_kind=telemetry.get("energy_measurement_kind") or "unavailable",
                proxy_method=telemetry.get("energy_proxy_method")))

    def save_recommendation(self, request: dict[str, Any], selected: str | None, explanation: str, predicted: bool = True) -> None:
        with self.sessions.begin() as session:
            session.add(OptimizationRecommendation(request_json=request, selected_configuration_id=selected, explanation=explanation, predicted=predicted))

    def list_experiments(self, limit: int = 100) -> list[BenchmarkExperiment]:
        with self.sessions() as session:
            return list(session.scalars(select(BenchmarkExperiment).order_by(BenchmarkExperiment.started_at.desc()).limit(limit)))
