
import csv
from pathlib import Path
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "logs" / "quality_responses.csv"

OUTPUT_PATH = PROJECT_ROOT / "logs" / "quality_scores.csv"

DIMENSIONS = [
    "correctness",
    "completeness",
    "relevance",
    "clarity",
    "instruction_following"
]


# ==========================================
# LOAD MODEL RESPONSES
# ==========================================

def load_responses():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Response file not found: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", encoding="utf-8") as file:

        reader = csv.DictReader(file)
        responses = list(reader)

    if not responses:
        raise ValueError("No model responses found.")

    return responses


# ==========================================
# COLLECT HUMAN SCORES
# ==========================================

def get_score(dimension):

    while True:

        try:

            score = int(input(
                f"Enter {dimension} score (0, 1, or 2): "
            ))

            if score in [0, 1, 2]:
                return score

            print("Please enter only 0, 1, or 2.")

        except ValueError:

            print("Invalid input. Enter a number.")


# ==========================================
# EVALUATE ONE RESPONSE
# ==========================================

def evaluate_response(response):

    print("\n" + "=" * 60)

    print(f"Question ID: {response['question_id']}")
    print(f"Task type: {response['task_type']}")

    print("\nQuestion:")
    print(response["prompt"])

    print("\nReference answer:")
    print(response["reference_answer"])

    print("\nModel answer:")
    print(response["model_answer"])

    print("\nScore the model answer:")

    scores = {}

    for dimension in DIMENSIONS:

        scores[dimension] = get_score(dimension)

    total_score = sum(scores.values())

    percentage = (total_score / 10) * 100

    print(f"\nTotal score: {total_score}/10")
    print(f"Quality percentage: {percentage:.1f}%")

    return {
        "question_id": response["question_id"],
        "task_type": response["task_type"],
        **scores,
        "total_score": total_score,
        "quality_percentage": round(percentage, 2),
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

    print("LOCAL LLM QUALITY SCORING SYSTEM")

    try:

        responses = load_responses()

        print(f"\nLoaded {len(responses)} model responses.")

        results = []

        for response in responses:

            result = evaluate_response(response)

            results.append(result)

            # Save after each response so progress is preserved.
            save_scores([result])

        print("\nAll responses have been scored.")

    except (FileNotFoundError, ValueError) as error:

        print(f"\nError: {error}")
