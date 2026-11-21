from core.model_registry import load_models
from database.repository import ExperimentRepository
from database.schema import HardwareProfile


def test_registry_only_marks_existing_files_available():
    models = load_models()
    assert any(model["available_locally"] for model in models)


def test_sqlite_repository_for_isolated_test(tmp_path):
    repository = ExperimentRepository(f"sqlite:///{tmp_path / 'test.db'}", create=True)
    repository.upsert_model({"model_id": "test", "family": "test", "quantization": "q4"})
    repository.save_experiment("exp-1", "test", model_id="test")
    assert repository.list_experiments()[0].experiment_id == "exp-1"


def test_repository_can_preserve_hardware_snapshot(tmp_path):
    repository = ExperimentRepository(f"sqlite:///{tmp_path / 'telemetry.db'}", create=True)
    repository.save_hardware_profile({"cpu_logical_cores": 8})
    with repository.sessions() as session:
        assert session.query(HardwareProfile).count() == 1
