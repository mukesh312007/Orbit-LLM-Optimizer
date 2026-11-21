"""
create_local_datasets.py
========================
Constructs minimal but valid local evaluation datasets (25 samples each) for
GSM8K, MMLU, and IFEval from scratch — no internet download required.

These are LOCAL DEMONSTRATION SAMPLES ONLY.  They are hand-crafted and are NOT
official benchmark splits.  Each manifest is clearly labelled accordingly.

Usage
-----
    python benchmarks/create_local_datasets.py          # write files
    python benchmarks/create_local_datasets.py --dry-run  # print only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))

from core.settings import ROOT  # noqa: E402  (after path fixup)

PROCESSED = ROOT / "datasets" / "processed"
NOTE = "Hand-crafted demonstration samples for local evaluation. Not an official benchmark split."

# ---------------------------------------------------------------------------
# GSM8K  — 25 grade-school arithmetic word problems
# Each answer is an integer; stored as "#### N" to match canonical GSM8K fmt.
# ---------------------------------------------------------------------------
GSM8K_RAW: list[dict[str, Any]] = [
    {"question": "Sally has 5 apples. She buys 3 more. How many apples does she have now?", "answer": "#### 8"},
    {"question": "A box holds 12 eggs. If 4 eggs are broken, how many are left?", "answer": "#### 8"},
    {"question": "Tom scored 7 goals on Monday and 9 on Tuesday. How many goals did he score in total?", "answer": "#### 16"},
    {"question": "A baker made 30 cookies and sold 18. How many cookies remain?", "answer": "#### 12"},
    {"question": "A train travels 60 km/h for 3 hours. How far does it travel?", "answer": "#### 180"},
    {"question": "Maria has $45. She spends $17 on a book. How much money does she have left?", "answer": "#### 28"},
    {"question": "There are 8 rows of chairs with 6 chairs in each row. How many chairs are there in total?", "answer": "#### 48"},
    {"question": "A rectangle is 9 m long and 4 m wide. What is its perimeter?", "answer": "#### 26"},
    {"question": "Jake collects 15 stamps each month. How many stamps will he have after 4 months?", "answer": "#### 60"},
    {"question": "A school has 240 students split equally into 6 classes. How many students are in each class?", "answer": "#### 40"},
    {"question": "Emma read 12 pages a day for 7 days. How many pages did she read in total?", "answer": "#### 84"},
    {"question": "A fruit stall has 50 oranges and sells 23. How many oranges are left?", "answer": "#### 27"},
    {"question": "Carlos earns $12 per hour and works 8 hours. How much does he earn?", "answer": "#### 96"},
    {"question": "A container holds 100 litres. If 37 litres are poured out, how many litres remain?", "answer": "#### 63"},
    {"question": "There are 5 bags with 9 marbles each. How many marbles are there altogether?", "answer": "#### 45"},
    {"question": "A plant grows 3 cm per week. How tall will it be after 11 weeks?", "answer": "#### 33"},
    {"question": "A pizza is cut into 8 slices. If 3 friends each eat 2 slices, how many slices are left?", "answer": "#### 2"},
    {"question": "Ahmed saves $6 every day. How much will he save in 15 days?", "answer": "#### 90"},
    {"question": "A warehouse stores 500 boxes. Workers remove 125 boxes. How many boxes remain?", "answer": "#### 375"},
    {"question": "A car uses 7 litres of fuel per 100 km. How many litres are needed for 300 km?", "answer": "#### 21"},
    {"question": "Lucy has 3 packs of stickers with 12 stickers each. She gives away 10 stickers. How many does she have?", "answer": "#### 26"},
    {"question": "A field is 25 m long and 10 m wide. What is its area?", "answer": "#### 250"},
    {"question": "A bus can carry 45 passengers. How many buses are needed for 180 passengers?", "answer": "#### 4"},
    {"question": "Kevin earned $200 and spent $55 on shoes and $30 on food. How much does he have left?", "answer": "#### 115"},
    {"question": "A pool is filled at a rate of 50 litres per minute. How many litres are added in 8 minutes?", "answer": "#### 400"},
]

# ---------------------------------------------------------------------------
# MMLU  — 25 multiple-choice questions, varied subjects
# answer is the 0-based integer index of the correct choice.
# ---------------------------------------------------------------------------
MMLU_RAW: list[dict[str, Any]] = [
    # Math
    {"question": "What is the square root of 144?", "choices": ["10", "11", "12", "13"], "answer": 2},
    {"question": "What is 15% of 200?", "choices": ["20", "25", "30", "35"], "answer": 2},
    {"question": "Which of the following is a prime number?", "choices": ["9", "15", "17", "21"], "answer": 2},
    {"question": "What is the value of pi to two decimal places?", "choices": ["3.12", "3.14", "3.16", "3.18"], "answer": 1},
    {"question": "How many sides does a hexagon have?", "choices": ["5", "6", "7", "8"], "answer": 1},
    # Science
    {"question": "What is the chemical symbol for water?", "choices": ["WO", "HO", "H2O", "OH2"], "answer": 2},
    {"question": "Which planet is closest to the Sun?", "choices": ["Venus", "Earth", "Mercury", "Mars"], "answer": 2},
    {"question": "What gas do plants absorb from the atmosphere during photosynthesis?", "choices": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"], "answer": 2},
    {"question": "What is the powerhouse of the cell?", "choices": ["Nucleus", "Ribosome", "Mitochondria", "Chloroplast"], "answer": 2},
    {"question": "What is the atomic number of carbon?", "choices": ["4", "6", "8", "12"], "answer": 1},
    # History
    {"question": "In which year did World War II end?", "choices": ["1943", "1944", "1945", "1946"], "answer": 2},
    {"question": "Who was the first President of the United States?", "choices": ["John Adams", "Thomas Jefferson", "Benjamin Franklin", "George Washington"], "answer": 3},
    {"question": "In which year did the French Revolution begin?", "choices": ["1776", "1789", "1799", "1812"], "answer": 1},
    {"question": "Who wrote the Declaration of Independence?", "choices": ["Benjamin Franklin", "John Adams", "Thomas Jefferson", "James Madison"], "answer": 2},
    {"question": "The ancient Olympic Games were first held in which country?", "choices": ["Rome", "Egypt", "Greece", "Persia"], "answer": 2},
    # Geography
    {"question": "What is the capital city of France?", "choices": ["Berlin", "Madrid", "Rome", "Paris"], "answer": 3},
    {"question": "Which is the longest river in the world?", "choices": ["Amazon", "Nile", "Yangtze", "Mississippi"], "answer": 1},
    {"question": "On which continent is the Sahara Desert located?", "choices": ["Asia", "Australia", "Africa", "South America"], "answer": 2},
    {"question": "Which country has the largest population in the world?", "choices": ["USA", "India", "China", "Russia"], "answer": 2},
    {"question": "What is the smallest country in the world by area?", "choices": ["Monaco", "San Marino", "Liechtenstein", "Vatican City"], "answer": 3},
    # General Knowledge
    {"question": "How many colors are in a rainbow?", "choices": ["5", "6", "7", "8"], "answer": 2},
    {"question": "Which element has the chemical symbol 'O'?", "choices": ["Gold", "Osmium", "Oxygen", "Ozone"], "answer": 2},
    {"question": "How many days are in a leap year?", "choices": ["364", "365", "366", "367"], "answer": 2},
    {"question": "What is the speed of light in a vacuum (approximately)?", "choices": ["300 km/s", "3,000 km/s", "30,000 km/s", "300,000 km/s"], "answer": 3},
    {"question": "Which instrument measures atmospheric pressure?", "choices": ["Thermometer", "Barometer", "Hygrometer", "Anemometer"], "answer": 1},
]

# ---------------------------------------------------------------------------
# IFEval  — 25 instruction-following prompts with explicit keyword checks
# checks lists words that MUST appear in a correct response.
# ---------------------------------------------------------------------------
IFEVAL_RAW: list[dict[str, Any]] = [
    {"prompt": "List three common fruits. Your answer must include: apple, banana, cherry.", "kwargs": {"checks": ["apple", "banana", "cherry"]}},
    {"prompt": "Name three primary colors. Include the words red, blue, and yellow.", "kwargs": {"checks": ["red", "blue", "yellow"]}},
    {"prompt": "List the first three planets from the Sun. Include: Mercury, Venus, Earth.", "kwargs": {"checks": ["mercury", "venus", "earth"]}},
    {"prompt": "Write one sentence about photosynthesis that includes the words: sunlight, chlorophyll, glucose.", "kwargs": {"checks": ["sunlight", "chlorophyll", "glucose"]}},
    {"prompt": "Describe water in one sentence. Include: liquid, hydrogen, oxygen.", "kwargs": {"checks": ["liquid", "hydrogen", "oxygen"]}},
    {"prompt": "Name three programming languages. Include: Python, Java, JavaScript.", "kwargs": {"checks": ["python", "java", "javascript"]}},
    {"prompt": "List three continents. Include: Asia, Africa, Europe.", "kwargs": {"checks": ["asia", "africa", "europe"]}},
    {"prompt": "List three animals that are mammals. Include: whale, dolphin, bat.", "kwargs": {"checks": ["whale", "dolphin", "bat"]}},
    {"prompt": "Describe the water cycle briefly. Include: evaporation, condensation, precipitation.", "kwargs": {"checks": ["evaporation", "condensation", "precipitation"]}},
    {"prompt": "Name three bones in the human body. Include: femur, tibia, humerus.", "kwargs": {"checks": ["femur", "tibia", "humerus"]}},
    {"prompt": "List three countries in Europe. Include: France, Germany, Italy.", "kwargs": {"checks": ["france", "germany", "italy"]}},
    {"prompt": "List three sports played with a ball. Include: soccer, basketball, tennis.", "kwargs": {"checks": ["soccer", "basketball", "tennis"]}},
    {"prompt": "Name three types of clouds. Include: cumulus, stratus, cirrus.", "kwargs": {"checks": ["cumulus", "stratus", "cirrus"]}},
    {"prompt": "Describe DNA briefly. Include: deoxyribonucleic, double, helix.", "kwargs": {"checks": ["deoxyribonucleic", "double", "helix"]}},
    {"prompt": "List three musical instruments. Include: guitar, piano, violin.", "kwargs": {"checks": ["guitar", "piano", "violin"]}},
    {"prompt": "Name three planets with rings. Include: Saturn, Uranus, Neptune.", "kwargs": {"checks": ["saturn", "uranus", "neptune"]}},
    {"prompt": "List three chemical elements. Include: carbon, nitrogen, oxygen.", "kwargs": {"checks": ["carbon", "nitrogen", "oxygen"]}},
    {"prompt": "Name three famous scientists. Include: Einstein, Newton, Curie.", "kwargs": {"checks": ["einstein", "newton", "curie"]}},
    {"prompt": "List three types of triangles. Include: equilateral, isosceles, scalene.", "kwargs": {"checks": ["equilateral", "isosceles", "scalene"]}},
    {"prompt": "Name three major world religions. Include: Christianity, Islam, Buddhism.", "kwargs": {"checks": ["christianity", "islam", "buddhism"]}},
    {"prompt": "List three renewable energy sources. Include: solar, wind, hydroelectric.", "kwargs": {"checks": ["solar", "wind", "hydroelectric"]}},
    {"prompt": "Name three programming paradigms. Include: procedural, object-oriented, functional.", "kwargs": {"checks": ["procedural", "object-oriented", "functional"]}},
    {"prompt": "List three layers of Earth. Include: crust, mantle, core.", "kwargs": {"checks": ["crust", "mantle", "core"]}},
    {"prompt": "Name three Greek gods. Include: Zeus, Apollo, Athena.", "kwargs": {"checks": ["zeus", "apollo", "athena"]}},
    {"prompt": "List three types of chemical bonds. Include: covalent, ionic, metallic.", "kwargs": {"checks": ["covalent", "ionic", "metallic"]}},
]

# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def _gsm8k_to_normalized(record: dict[str, Any], index: int) -> dict[str, Any]:
    question = record["question"]
    answer = record["answer"]  # e.g. "#### 42"
    return {
        "sample_id": f"gsm8k:{index}",
        "dataset": "gsm8k",
        "task_type": "reasoning",
        "prompt": question,
        "reference_answer": answer,
        "evaluation_type": "numeric_exact",
        "metadata": {},
    }


def _mmlu_to_normalized(record: dict[str, Any], index: int) -> dict[str, Any]:
    question = record["question"]
    choices = record["choices"]
    answer = record["answer"]  # 0-based int index
    # Build the full prompt the way normalize_record() does
    prompt = question + "\n" + "\n".join(f"{chr(65+i)}. {c}" for i, c in enumerate(choices))
    return {
        "sample_id": f"mmlu:{index}",
        "dataset": "mmlu",
        "task_type": "reasoning",
        "prompt": prompt,
        "reference_answer": str(answer),
        "evaluation_type": "exact_match",
        "metadata": {"choices": choices},
    }


def _ifeval_to_normalized(record: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "sample_id": f"ifeval:{index}",
        "dataset": "ifeval",
        "task_type": "instruction_following",
        "prompt": record["prompt"],
        "reference_answer": record["prompt"],   # prompt IS the reference for IFEval
        "evaluation_type": "instruction_following",
        "metadata": {"checks": record["kwargs"]["checks"]},
    }


# ---------------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------------

DATASETS: dict[str, list[dict[str, Any]]] = {
    "gsm8k": [_gsm8k_to_normalized(r, i) for i, r in enumerate(GSM8K_RAW)],
    "mmlu":  [_mmlu_to_normalized(r, i)  for i, r in enumerate(MMLU_RAW)],
    "ifeval": [_ifeval_to_normalized(r, i) for i, r in enumerate(IFEVAL_RAW)],
}

# ---------------------------------------------------------------------------
# Validation (inline — avoids import-path issues)
# ---------------------------------------------------------------------------

def _validate(samples: list[dict[str, Any]]) -> list[str]:
    from benchmarks.dataset_validator import validate_dataset
    return validate_dataset(samples)


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def _write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for sample in samples:
            fh.write(json.dumps(sample, ensure_ascii=False) + "\n")


def _write_manifest(path: Path, dataset: str, count: int) -> None:
    manifest = {
        "dataset": dataset,
        "source_file": "local_demonstration_samples",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": count,
        "note": NOTE,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False) -> int:
    all_ok = True

    for dataset, samples in DATASETS.items():
        data_path = PROCESSED / f"{dataset}.jsonl"
        manifest_path = PROCESSED / f"{dataset}.manifest.json"

        # Validate before writing
        errors = _validate(samples)
        if errors:
            print(f"[ERROR] {dataset}: validation failed before write:")
            for e in errors:
                print(f"  {e}")
            all_ok = False
            continue

        if dry_run:
            print(f"[DRY-RUN] Would write {len(samples)} {dataset} samples → {data_path}")
            print(f"[DRY-RUN] Would write manifest → {manifest_path}")
            # Show first sample as preview
            print(f"[DRY-RUN] First sample: {json.dumps(samples[0], ensure_ascii=False)}")
        else:
            _write_jsonl(data_path, samples)
            _write_manifest(manifest_path, dataset, len(samples))
            print(f"[OK] {dataset}: wrote {len(samples)} samples → {data_path}")
            print(f"[OK] {dataset}: manifest → {manifest_path}")

    if not dry_run:
        # Post-write validation round-trip
        print("\n--- Post-write validation ---")
        for dataset in DATASETS:
            data_path = PROCESSED / f"{dataset}.jsonl"
            loaded = []
            with data_path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        loaded.append(json.loads(line))
            errors = _validate(loaded)
            if errors:
                print(f"[FAIL] {dataset}: {errors}")
                all_ok = False
            else:
                print(f"[PASS] {dataset}: {len(loaded)} samples validated OK")

    return 0 if all_ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create local demonstration datasets (no internet required).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created without writing files.")
    args = parser.parse_args()
    sys.exit(main(dry_run=args.dry_run))
