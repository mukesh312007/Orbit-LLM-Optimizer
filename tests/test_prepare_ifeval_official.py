import json

from benchmarks import prepare_ifeval_official


def test_prepare_official_ifeval_keeps_checker_metadata_and_uses_separate_dataset(monkeypatch, tmp_path):
    source = tmp_path / "input_data.jsonl"
    source.write_text(json.dumps({"key": 1000, "prompt": "Use no commas.", "instruction_id_list": ["punctuation:no_comma"], "kwargs": [{}]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(prepare_ifeval_official, "PROCESSED", tmp_path / "processed")
    output = prepare_ifeval_official.prepare(source)
    sample = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert sample["dataset"] == "ifeval_official"
    assert sample["metadata"]["official_checker_required"] is True
