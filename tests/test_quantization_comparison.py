from benchmarks.quantization_comparison import VARIANTS, inventory, select_diverse_samples


def test_quantization_inventory_reports_missing_variants_without_downloading(tmp_path):
    report = inventory(tmp_path)
    assert report["download_or_conversion_authorization_required"] is True
    assert {item["quantization"] for item in report["variants"]} == set(VARIANTS)
    assert all(item["status"] == "missing" for item in report["variants"])


def test_quantization_selection_covers_distinct_spider_databases_first(monkeypatch):
    from benchmarks import quantization_comparison as comparison
    samples = [{"db_id": "a"}, {"db_id": "a"}, {"db_id": "b"}, {"db_id": "c"}]
    monkeypatch.setattr(comparison, "load_spider_dev", lambda _: samples)
    assert [item["db_id"] for item in select_diverse_samples(3)] == ["a", "b", "c"]
