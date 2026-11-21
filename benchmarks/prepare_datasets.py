from __future__ import annotations

"""Convert already-downloaded official JSON/JSONL samples; intentionally does not auto-download datasets."""
import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.dataset_loader import PROCESSED, normalize_record
from benchmarks.dataset_validator import validate_dataset
from core.settings import ROOT


def read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    return json.loads(text) if path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]


def prepare(dataset: str, source: Path, limit: int = 50) -> Path:
    records = read_records(source)[:limit]
    samples = [normalize_record(dataset, record, i) for i, record in enumerate(records)]
    errors = validate_dataset(samples)
    if errors:
        raise ValueError("\n".join(errors))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    output = PROCESSED / f"{dataset}.jsonl"
    with output.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    provenance = output.with_suffix(".manifest.json")
    provenance.write_text(json.dumps({
        "dataset": dataset, "source_file": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_bytes": source.stat().st_size, "prepared_at": datetime.now(UTC).isoformat(),
        "sample_count": len(samples), "normalization": "orbit_local_dataset_v2",
    }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["mmlu", "gsm8k", "mbpp", "ifeval", "cnn_dailymail"])
    parser.add_argument("source", type=Path, help="Official dataset file downloaded by the user")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    print(prepare(args.dataset, args.source, args.limit))
