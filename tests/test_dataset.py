from datasets.dataset_validator import validate_sample, validate_samples


def sample():
    return {"sample_id": "x:1", "dataset": "x", "task_type": "reasoning", "prompt": "2+2", "reference_answer": "4", "evaluation_type": "numeric_exact_match", "metadata": {}}


def test_valid_sample():
    assert validate_sample(sample()) == []


def test_duplicate_identifier_is_rejected():
    assert "duplicate sample_id x:1" in validate_samples([sample(), sample()])[0]
