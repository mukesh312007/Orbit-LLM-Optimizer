from benchmarks.dataset_loader import normalize_record
from benchmarks.dataset_validator import validate_dataset


def test_normalizes_and_validates_gsm8k():
    sample = normalize_record("gsm8k", {"question": "1 + 1?", "answer": "2", "id": "x"}, 0)
    assert sample["sample_id"] == "gsm8k:x"
    assert sample["evaluation_type"] == "numeric_exact"
    assert not validate_dataset([sample])


def test_rejects_missing_fields():
    assert validate_dataset([{"sample_id": "a"}])
