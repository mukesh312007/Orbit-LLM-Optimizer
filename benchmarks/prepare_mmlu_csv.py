"""Prepare the official Hendrycks MMLU per-subject CSV test split locally."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.dataset_loader import PROCESSED, normalize_record
from benchmarks.dataset_validator import validate_dataset


def _tree_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare(source_directory: Path, limit: int = 0) -> Path:
    """Convert headerless MMLU subject CSVs without changing the source files."""
    paths = sorted(source_directory.glob("*_test.csv"))
    if not paths:
        raise FileNotFoundError(f"No *_test.csv files found in {source_directory}")
    rows: list[dict] = []
    for path in paths:
        subject = path.name.removesuffix("_test.csv")
        with path.open(encoding="utf-8", newline="") as handle:
            for row_index, row in enumerate(csv.reader(handle)):
                if len(row) != 6:
                    raise ValueError(f"{path.name} row {row_index + 1} has {len(row)} columns; expected question, A, B, C, D, answer")
                question, *choices, answer = row
                if answer not in {"A", "B", "C", "D"}:
                    raise ValueError(f"{path.name} row {row_index + 1} has invalid answer {answer!r}")
                rows.append({"id": f"{subject}:{row_index}", "question": question, "choices": choices, "answer": answer, "subject": subject})
    if limit > 0:
        rows = rows[:limit]
    samples = [normalize_record("mmlu", row, index) for index, row in enumerate(rows)]
    for sample in samples:
        sample["sample_id"] = sample["sample_id"].replace("mmlu:", "mmlu_official:", 1)
        sample["dataset"] = "mmlu_official"
    errors = validate_dataset(samples)
    if errors:
        raise ValueError("; ".join(errors))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "mmlu_official.jsonl"
    output.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "dataset": "mmlu_official", "source_directory": str(source_directory),
        "split": "test", "subject_file_count": len(paths), "sample_count": len(samples),
        "source_tree_sha256": _tree_hash(paths), "prepared_at": datetime.now(UTC).isoformat(),
        "format": "official_headerless_question_A_B_C_D_answer_csv",
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare official MMLU *_test.csv files locally.")
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="Use zero for every row; otherwise take a deterministic prefix.")
    arguments = parser.parse_args()
    print(prepare(arguments.source_directory, arguments.limit))
