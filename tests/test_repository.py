from database.repository import ExperimentRepository


def test_sqlite_file_repository_supports_isolated_tests(tmp_path):
    repo = ExperimentRepository(f"sqlite:///{tmp_path / 'test.db'}", create=True)
    repo.upsert_model({"model_id": "model", "family": "test"})
    repo.save_experiment("run-1", "benchmark", model_id="model")
    repo.save_experiment("run-1", "benchmark", model_id="model")

    assert [experiment.experiment_id for experiment in repo.list_experiments()] == ["run-1"]
