from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    return datetime.now(UTC)


class Model(Base):
    __tablename__ = "models"
    model_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    family: Mapped[str] = mapped_column(String(128))
    quantization: Mapped[str | None] = mapped_column(String(64))
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class HardwareProfile(Base):
    __tablename__ = "hardware_profiles"
    hardware_profile_id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    profile_json: Mapped[dict] = mapped_column(JSON)


class Dataset(Base):
    __tablename__ = "datasets"
    dataset_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str | None] = mapped_column(Text)
    manifest_json: Mapped[dict] = mapped_column(JSON, default=dict)


class RuntimeConfiguration(Base):
    __tablename__ = "runtime_configurations"
    configuration_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_id: Mapped[str] = mapped_column(ForeignKey("models.model_id"))
    backend: Mapped[str] = mapped_column(String(32))
    settings_json: Mapped[dict] = mapped_column(JSON)


class BenchmarkExperiment(Base):
    __tablename__ = "benchmark_experiments"
    experiment_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_id: Mapped[str | None] = mapped_column(ForeignKey("models.model_id"))
    configuration_id: Mapped[str | None] = mapped_column(ForeignKey("runtime_configurations.configuration_id"))
    dataset_id: Mapped[str | None] = mapped_column(ForeignKey("datasets.dataset_id"))
    kind: Mapped[str] = mapped_column(String(64), default="benchmark")
    metadata_json: Mapped[dict] = mapped_column(JSON, default=dict)


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    result_id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(ForeignKey("benchmark_experiments.experiment_id"))
    prompt_id: Mapped[str | None] = mapped_column(String(128))
    trial_number: Mapped[int | None] = mapped_column(Integer)
    response_text: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (UniqueConstraint("experiment_id", "prompt_id", "trial_number", name="uq_benchmark_trial"),)


class QualityResult(Base):
    __tablename__ = "quality_results"
    quality_result_id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.result_id"))
    metric: Mapped[str] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), default="automated")
    details_json: Mapped[dict] = mapped_column(JSON, default=dict)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    metric_id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.result_id"))
    ttft_seconds: Mapped[float | None] = mapped_column(Float)
    tpot_ms: Mapped[float | None] = mapped_column(Float)
    end_to_end_seconds: Mapped[float | None] = mapped_column(Float)
    generation_tokens_per_second: Mapped[float | None] = mapped_column(Float)
    generated_tokens: Mapped[int | None] = mapped_column(Integer)


class ResourceMetric(Base):
    __tablename__ = "resource_metrics"
    metric_id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.result_id"))
    process_memory_mb: Mapped[float | None] = mapped_column(Float)
    system_memory_percent: Mapped[float | None] = mapped_column(Float)
    cpu_percent: Mapped[float | None] = mapped_column(Float)
    gpu_utilization_percent: Mapped[float | None] = mapped_column(Float)


class EnergyMetric(Base):
    __tablename__ = "energy_metrics"
    metric_id: Mapped[int] = mapped_column(primary_key=True)
    result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.result_id"))
    energy_joules: Mapped[float | None] = mapped_column(Float)
    energy_proxy_joules: Mapped[float | None] = mapped_column(Float)
    source: Mapped[str | None] = mapped_column(String(128))
    measured: Mapped[bool] = mapped_column(default=False)
    measurement_kind: Mapped[str] = mapped_column(String(32), default="unavailable")
    proxy_method: Mapped[str | None] = mapped_column(Text)


class OptimizationRecommendation(Base):
    __tablename__ = "optimization_recommendations"
    recommendation_id: Mapped[int] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    request_json: Mapped[dict] = mapped_column(JSON)
    selected_configuration_id: Mapped[str | None] = mapped_column(String(128))
    explanation: Mapped[str] = mapped_column(Text)
    predicted: Mapped[bool] = mapped_column(default=True)


def make_engine(url: str):
    connect_args = {}
    if "pymysql" in url:
        connect_args["connect_timeout"] = 2
    return create_engine(url, future=True, pool_pre_ping=True, connect_args=connect_args)


def create_schema(url: str) -> None:
    engine = make_engine(url)
    Base.metadata.create_all(engine)
    # ``create_all`` does not evolve an existing local MySQL installation.
    # Keep this deliberately small and additive; versioned SQL remains the
    # reviewable deployment path for clean installations.
    from sqlalchemy import inspect, text
    inspector = inspect(engine)
    if "energy_metrics" not in inspector.get_table_names():
        return
    existing = {column["name"] for column in inspector.get_columns("energy_metrics")}
    additions = {
        "energy_proxy_joules": "DOUBLE NULL",
        "measurement_kind": "VARCHAR(32) NOT NULL DEFAULT 'unavailable'",
        "proxy_method": "TEXT NULL",
    }
    with engine.begin() as connection:
        for name, declaration in additions.items():
            if name not in existing:
                connection.execute(text(f"ALTER TABLE energy_metrics ADD COLUMN {name} {declaration}"))
