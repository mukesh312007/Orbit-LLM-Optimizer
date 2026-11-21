"""SQLAlchemy relational schema for reproducible experiment history."""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Model(Base):
    __tablename__ = "models"
    id: Mapped[int] = mapped_column(primary_key=True)
    model_id: Mapped[str] = mapped_column(String(160), unique=True)
    metadata_json: Mapped[dict] = mapped_column(JSON)


class HardwareProfile(Base):
    __tablename__ = "hardware_profiles"
    id: Mapped[int] = mapped_column(primary_key=True)
    fingerprint: Mapped[str] = mapped_column(String(128), unique=True)
    profile_json: Mapped[dict] = mapped_column(JSON)


class Dataset(Base):
    __tablename__ = "datasets"
    id: Mapped[int] = mapped_column(primary_key=True)
    dataset_id: Mapped[str] = mapped_column(String(160), unique=True)
    manifest_json: Mapped[dict] = mapped_column(JSON)


class RuntimeConfigurationRow(Base):
    __tablename__ = "runtime_configurations"
    id: Mapped[int] = mapped_column(primary_key=True)
    configuration_id: Mapped[str] = mapped_column(String(160), unique=True)
    configuration_json: Mapped[dict] = mapped_column(JSON)


class BenchmarkExperiment(Base):
    __tablename__ = "benchmark_experiments"
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    model_id: Mapped[str] = mapped_column(String(160))
    configuration_id: Mapped[str] = mapped_column(String(160))
    dataset_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    backend: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32), default="completed")


class BenchmarkResult(Base):
    __tablename__ = "benchmark_results"
    __table_args__ = (UniqueConstraint("experiment_id", "prompt_id", "trial_number", name="uq_result_trial"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    experiment_id: Mapped[str] = mapped_column(String(64), ForeignKey("benchmark_experiments.experiment_id"))
    prompt_id: Mapped[str] = mapped_column(String(255))
    trial_number: Mapped[int] = mapped_column(Integer)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)


class QualityResult(Base):
    __tablename__ = "quality_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.id"))
    evaluation_type: Mapped[str] = mapped_column(String(64))
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str] = mapped_column(String(32), default="automated")
    details_json: Mapped[dict] = mapped_column(JSON)


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.id"))
    metrics_json: Mapped[dict] = mapped_column(JSON)


class ResourceMetric(Base):
    __tablename__ = "resource_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.id"))
    metrics_json: Mapped[dict] = mapped_column(JSON)


class EnergyMetric(Base):
    __tablename__ = "energy_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    benchmark_result_id: Mapped[int] = mapped_column(ForeignKey("benchmark_results.id"))
    joules: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics_json: Mapped[dict] = mapped_column(JSON)


class OptimizationRecommendationRow(Base):
    __tablename__ = "optimization_recommendations"
    id: Mapped[int] = mapped_column(primary_key=True)
    request_json: Mapped[dict] = mapped_column(JSON)
    recommendation_json: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
