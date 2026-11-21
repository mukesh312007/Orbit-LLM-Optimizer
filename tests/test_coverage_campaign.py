from benchmarks.coverage_campaign import execute, plan


def test_campaign_plan_is_local_and_describes_larger_spider_coverage():
    report = plan(spider_limit=25, trials=3, max_configurations=6)
    assert report["local_only"] is True
    assert report["spider"]["estimated_measured_requests_per_model"] == 450
    assert report["controlled_studies"]["thread_context_models"] == ["qwen2.5-1.5b-q4_k_m", "qwen2.5-3b-q4_k_m"]


def test_campaign_execute_records_stages_and_preserves_manifest(monkeypatch, tmp_path):
    import json
    from benchmarks import coverage_campaign

    fake_stages = []

    def fake_full_matrix(**kwargs):
        fake_stages.append("spider")
        return (tmp_path / "raw.jsonl", tmp_path / "summary.json")

    def fake_context_thread(*args, **kwargs):
        fake_stages.append("context_thread")
        return (tmp_path / "ct.jsonl", tmp_path / "ct.summary.json")

    def fake_cache(*args, **kwargs):
        fake_stages.append("cache")
        return (tmp_path / "cache.jsonl", tmp_path / "cache.summary.json")

    def fake_quality(*args, **kwargs):
        fake_stages.append("quality")
        return (tmp_path / "q.jsonl", tmp_path / "q.summary.json")

    def fake_profiles(*args, **kwargs):
        fake_stages.append("profiles")
        return tmp_path / "profiles.json"

    monkeypatch.setattr(coverage_campaign, "run_full_matrix", fake_full_matrix)
    monkeypatch.setattr(coverage_campaign, "run_context_thread", fake_context_thread)
    monkeypatch.setattr(coverage_campaign, "run_cache", fake_cache)
    monkeypatch.setattr(coverage_campaign, "run_standard_quality", fake_quality)
    monkeypatch.setattr(coverage_campaign, "create_profiles", fake_profiles)
    monkeypatch.setattr(coverage_campaign, "ROOT", tmp_path)

    manifest_path = execute(spider_limit=2, trials=1, max_configurations=1)
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["campaign_id"].startswith("local_evidence_campaign")
    assert "completed_at" in manifest
    assert len(manifest["stages"]) > 0
    stage_names = [s["name"] for s in manifest["stages"]]
    assert any("spider_matrix" in name for name in stage_names)
    assert any("calibrated_profiles" in name for name in stage_names)
