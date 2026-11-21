from database.repository import ExperimentRepository


def test_live_selection_repository_stores_performance_without_claiming_actual_energy(tmp_path):
    repository = ExperimentRepository(f"sqlite:///{tmp_path / 'telemetry.db'}", create=True)
    model = {"model_id": "qwen-test", "family": "Qwen", "quantization": "Q4_K_M"}
    configuration = {"configuration_id": "qwen-test:t4", "backend": "llama.cpp", "threads": 4}
    repository.save_live_selection(
        "selection-1", model, configuration, "general",
        {"answer": "hello", "ttft_seconds": 0.1, "end_to_end_seconds": 1.0, "generation_tokens_per_second": 10.0, "generated_tokens": 9, "peak_ram_mb": 500.0, "average_system_cpu_percent": 25.0, "energy_proxy_joules": 11.0},
    )
    with repository.sessions() as session:
        from database.schema import EnergyMetric, PerformanceMetric
        assert session.query(PerformanceMetric).count() == 1
        energy = session.query(EnergyMetric).one()
        assert energy.energy_joules is None
        assert energy.measured is False


def test_benchmark_repository_marks_energy_proxy_as_estimated(tmp_path):
    repository = ExperimentRepository(f"sqlite:///{tmp_path / 'benchmark.db'}", create=True)
    record = {
        "experiment_id": "parent", "measurement_kind": "measured", "trial_number": 1,
        "model": {"model_id": "qwen-test", "family": "Qwen", "quantization": "Q4"},
        "configuration": {"configuration_id": "qwen-test:t4", "backend": "llama.cpp"},
        "database_prompt_id": "qwen-test:t4|prompt-1", "response": "SELECT 1",
        "task_quality": {"metric": "sql", "score": 1.0},
        "telemetry": {"energy_proxy_joules": 2.0},
    }
    repository.save_benchmark_trial("benchmark-1", "full_matrix_spider", record)
    with repository.sessions() as session:
        from database.schema import EnergyMetric, QualityResult
        assert session.query(QualityResult).one().score == 1.0
        energy = session.query(EnergyMetric).one()
        assert energy.energy_joules is None
        assert energy.energy_proxy_joules == 2.0
        assert energy.measured is False
