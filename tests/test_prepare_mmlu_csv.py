import json

from benchmarks import prepare_mmlu_csv


def test_prepare_mmlu_csv_combines_official_subject_files_without_overwriting_demo(monkeypatch, tmp_path):
    source = tmp_path / "test"
    source.mkdir()
    (source / "abstract_algebra_test.csv").write_text('What is two plus two?,3,4,5,6,B\n', encoding="utf-8")
    monkeypatch.setattr(prepare_mmlu_csv, "PROCESSED", tmp_path / "processed")
    output = prepare_mmlu_csv.prepare(source)
    record = json.loads(output.read_text(encoding="utf-8").splitlines()[0])
    assert output.name == "mmlu_official.jsonl"
    assert record["dataset"] == "mmlu_official"
    assert record["reference_answer"] == "B"
    assert record["metadata"]["subject"] == "abstract_algebra"
