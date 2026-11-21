from core.provenance import benchmark_provenance, sha256_file


def test_file_hash_and_benchmark_provenance_are_local_and_repeatable(tmp_path, monkeypatch):
    from core import provenance

    model_path = tmp_path / "models" / "test.gguf"
    model_path.parent.mkdir()
    model_path.write_bytes(b"GGUF test")
    monkeypatch.setattr(provenance, "ROOT", tmp_path)
    monkeypatch.setattr(provenance, "hardware_snapshot", lambda: {"cpu_logical_cores": 4})

    digest = sha256_file(model_path)
    record = benchmark_provenance(
        {"model_id": "test", "path": "models/test.gguf", "quantization": "Q4"},
        {"configuration_id": "test:t4"},
    )

    assert digest == sha256_file(model_path)
    assert record["model_sha256"] == digest
    assert record["hardware"]["cpu_logical_cores"] == 4
