
import csv
from pathlib import Path
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "logs" / "combined_results.csv"

OUTPUT_PATH = PROJECT_ROOT / "logs" / "combined_quality_scores.csv"

DIMENSIONS = [
    "correctness",
    "completeness",
    "relevance",
    "clarity",
    "instruction_following"
]


# ==========================================
# LOAD COMBINED EXPERIMENT
# ==========================================

def load_results():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    with open(INPUT_PATH, "r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError("No combined experiment records found.")

    return rows


# ==========================================
# COLLECT HUMAN SCORE
# ==========================================

def get_score(dimension):

    while True:

        try:
            score = int(input(
                f"Score {dimension} (0, 1, or 2): "
            ))

            if score in (0, 1, 2):
                return score

            print("Enter 0, 1, or 2.")

        except ValueError:
            print("Please enter a valid integer.")


# ==========================================
# SCORE ONE RESPONSE
# ==========================================

def score_response(row):

    print("\n" + "=" * 65)

    print(f"Experiment: {row['experiment_id']}")
    print(f"Question: {row['question_id']}")
    print(f"Task: {row['task_type']}")

    print("\nQuestion:")
    print(row["prompt"])

    print("\nReference answer:")
    print(row["reference_answer"])

    print("\nModel answer:")
    print(row["model_answer"])

    scores = {}

    print("\nEnter your evaluation scores:")

    for dimension in DIMENSIONS:
        scores[dimension] = get_score(dimension)

    total = sum(scores.values())

    return {
        "experiment_id": row["experiment_id"],
        "model": row["model"],
        "cache_mode": row["cache_mode"],
        "question_id": row["question_id"],
        "task_type": row["task_type"],
        **scores,
        "total_score": total,
        "quality_percentage": round(total * 10, 2),
        "reviewed_at": datetime.now().isoformat(
            timespec="seconds"
        )
    }


# ==========================================
# SAVE SCORES
# ==========================================

def save_scores(results):

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment_id",
        "model",
        "cache_mode",
        "question_id",
        "task_type",
        *DIMENSIONS,
        "total_score",
        "quality_percentage",
        "reviewed_at"
    ]

    file_exists = (
        OUTPUT_PATH.exists()
        and OUTPUT_PATH.stat().st_size > 0
    )

    with open(
        OUTPUT_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nScores saved to: {OUTPUT_PATH}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("COMBINED EXPERIMENT QUALITY SCORING")

    try:

        rows = load_results()

        experiment_ids = {
            row["experiment_id"] for row in rows
        }

        if len(experiment_ids) != 1:
            raise ValueError(
                "This scoring run expects one experiment ID. "
                "Filter the CSV by experiment first."
            )

        results = []

        for row in rows:
            results.append(score_response(row))

        save_scores(results)

        print("\nQuality scoring completed.")
        print(f"Responses scored: {len(results)}")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")
