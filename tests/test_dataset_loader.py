import json

from benchmarks.dataset_loader import normalize_record
from benchmarks.prepare_datasets import prepare


def test_mmlu_normalization_requires_a_final_option_letter_and_preserves_choices():
    sample = normalize_record("mmlu", {"question": "2 + 2?", "choices": ["1", "4", "5", "7"], "answer": 1}, 0)
    assert "Reply with only the final option letter" in sample["prompt"]
    assert sample["reference_answer"] == 1
    assert sample["metadata"]["choices"] == ["1", "4", "5", "7"]


def test_prepared_manifest_hashes_the_user_supplied_source(monkeypatch, tmp_path):
    from benchmarks import prepare_datasets
    source = tmp_path / "gsm8k.jsonl"
    source.write_text(json.dumps({"question": "1 + 1", "answer": "2"}) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare_datasets, "PROCESSED", tmp_path / "processed")
    output = prepare("gsm8k", source, limit=1)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"]
    assert manifest["normalization"] == "orbit_local_dataset_v2"
