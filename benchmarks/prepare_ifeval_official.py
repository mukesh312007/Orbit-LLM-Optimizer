"""Prepare Google's official IFEval input JSONL without overwriting demo data."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmarks.dataset_loader import PROCESSED
from benchmarks.dataset_validator import validate_dataset


def prepare(source: Path, limit: int = 0) -> Path:
    lines = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit > 0:
        lines = lines[:limit]
    samples = []
    for record in lines:
        if not isinstance(record.get("key"), int) or not isinstance(record.get("prompt"), str):
            raise ValueError("Official IFEval records need integer key and prompt fields")
        if not isinstance(record.get("instruction_id_list"), list) or not isinstance(record.get("kwargs"), list):
            raise ValueError("Official IFEval records need instruction_id_list and kwargs lists")
        samples.append({
            "sample_id": f"ifeval_official:{record['key']}", "dataset": "ifeval_official",
            "task_type": "instruction_following", "prompt": record["prompt"],
            "reference_answer": None, "evaluation_type": "instruction_following",
            "metadata": {"instruction_id_list": record["instruction_id_list"], "kwargs": record["kwargs"], "official_checker_required": True},
        })
    errors = validate_dataset(samples)
    if errors:
        raise ValueError("; ".join(errors))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / "ifeval_official.jsonl"
    output.write_text("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples), encoding="utf-8")
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "dataset": "ifeval_official", "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size, "sample_count": len(samples), "prepared_at": datetime.now(UTC).isoformat(),
        "evaluation": "Google IFEval official strict/loose checker required; local generic scorer intentionally leaves these unscored.",
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare official IFEval input_data.jsonl locally.")
    parser.add_argument("source", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    arguments = parser.parse_args()
    print(prepare(arguments.source, arguments.limit))
