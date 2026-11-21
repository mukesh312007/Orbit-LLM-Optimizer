import json

from optimizer import evidence_report


def test_dataset_readiness_reports_missing_local_data_without_downloading(monkeypatch, tmp_path):
    monkeypatch.setattr(evidence_report, "PROCESSED", tmp_path)
    rows = evidence_report.dataset_readiness()
    assert [row["dataset"] for row in rows] == ["gsm8k", "mmlu", "ifeval"]
    assert all(not row["prepared"] for row in rows)


def test_profile_comparison_view_keeps_dominated_profiles(monkeypatch):
    profiles = [
        {
            "model_id": "fast", "configuration_id": "fast:t4", "task_type": "spider_text_to_sql",
            "quality": 0.5, "latency_seconds": 1.0, "peak_ram_mb": 100.0,
        },
        {
            "model_id": "slow", "configuration_id": "slow:t4", "task_type": "spider_text_to_sql",
            "quality": 0.4, "latency_seconds": 2.0, "peak_ram_mb": 200.0,
        },
    ]
    monkeypatch.setattr(evidence_report, "latest_profiles", lambda: profiles)

    rows = evidence_report.profile_comparison_view()

    assert {row["model_id"] for row in rows} == {"fast", "slow"}
    assert [row["model_id"] for row in evidence_report.profile_pareto_view()] == ["fast"]


def test_selection_history_removes_answer_text(monkeypatch, tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps({
        "model_id": "qwen", "configuration_id": "qwen:t4", "task_type": "general",
        "telemetry": {"answer": "private", "end_to_end_seconds": 1.2, "response_text_retained": False},
    }) + "\n", encoding="utf-8")
    monkeypatch.setattr(evidence_report, "SELECTION_HISTORY", history)
    row = evidence_report.selection_history()[0]
    assert row["model_id"] == "qwen"
    assert "answer" not in row


def test_pareto_view_keeps_only_non_dominated_measured_profiles(monkeypatch):
    monkeypatch.setattr(evidence_report, "latest_profiles", lambda: [
        {"task_type": "spider_text_to_sql", "model_id": "fast", "configuration_id": "fast", "quality": 0.9, "latency_seconds": 1.0, "peak_ram_mb": 500, "sample_count": 3},
        {"task_type": "spider_text_to_sql", "model_id": "slow", "configuration_id": "slow", "quality": 0.8, "latency_seconds": 2.0, "peak_ram_mb": 800, "sample_count": 3},
    ])
    assert [row["model_id"] for row in evidence_report.profile_pareto_view()] == ["fast"]
