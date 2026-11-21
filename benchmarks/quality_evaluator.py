
import csv
import json
import time
import requests

from pathlib import Path
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "quality_eval.json"

RESULTS_PATH = PROJECT_ROOT / "logs" / "quality_responses.csv"

SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"

MAX_OUTPUT_TOKENS = 256

PAYLOAD_SETTINGS = {
    "n_predict": MAX_OUTPUT_TOKENS,
    "temperature": 0,
    "stream": False
}


# ==========================================
# LOAD EVALUATION DATASET
# ==========================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty JSON list.")

    print(f"Loaded {len(dataset)} evaluation questions.")

    return dataset


# ==========================================
# GENERATE MODEL RESPONSE
# ==========================================


def generate_response(question):

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and accurate assistant. "
                    "Answer the user's question directly. "
                    "Be clear and concise. "
                    "Follow the requested format. "
                    "Do not add unnecessary explanations."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "stream": False
    }

    start_time = time.perf_counter()

    response = requests.post(
        SERVER_URL,
        json=payload,
        timeout=(10, 180)
    )

    response.raise_for_status()

    data = response.json()

    end_time = time.perf_counter()

    answer = data["choices"][0]["message"]["content"].strip()

    usage = data.get("usage", {})

    return {
        "answer": answer,
        "request_time_seconds": round(
            end_time - start_time, 6
        ),
        "prompt_tokens": usage.get("prompt_tokens"),
        "generated_tokens": usage.get("completion_tokens"),
        "prompt_ms": None,
        "generation_ms": None,
        "generation_tokens_per_second": None
    }


# ==========================================
# EVALUATE ALL QUESTIONS
# ==========================================

def evaluate_dataset(dataset):

    results = []

    for index, item in enumerate(dataset, start=1):

        question_id = item["id"]

        print("\n" + "=" * 50)
        print(f"Question {index}/{len(dataset)}: {question_id}")
        print(f"Task type: {item['task_type']}")
        print(f"Prompt: {item['prompt']}")

        result = generate_response(item["prompt"])

        record = {
            "question_id": question_id,
            "task_type": item["task_type"],
            "prompt": item["prompt"],
            "reference_answer": item["reference_answer"],
            "required_points": json.dumps(
                item["required_points"],
                ensure_ascii=False
            ),
            "model_answer": result["answer"],
            "request_time_seconds": result[
                "request_time_seconds"
            ],
            "prompt_tokens": result["prompt_tokens"],
            "generated_tokens": result["generated_tokens"],
            "prompt_ms": result["prompt_ms"],
            "generation_ms": result["generation_ms"],
            "generation_tokens_per_second": result[
                "generation_tokens_per_second"
            ]
        }

        results.append(record)

        print("\nModel answer:")
        print(result["answer"])

        print(
            f"\nRequest time: "
            f"{result['request_time_seconds']} seconds"
        )

        print(
            f"Generation speed: "
            f"{result['generation_tokens_per_second']} tokens/s"
        )

    return results


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(results):

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "timestamp",
        "model",
        "question_id",
        "task_type",
        "prompt",
        "reference_answer",
        "required_points",
        "model_answer",
        "request_time_seconds",
        "prompt_tokens",
        "generated_tokens",
        "prompt_ms",
        "generation_ms",
        "generation_tokens_per_second"
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for result in results:

            row = {
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "model": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
                **result
            }

            writer.writerow(row)

    print(f"\nResults saved to: {RESULTS_PATH}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("LOCAL LLM QUALITY EVALUATION")

    try:

        dataset = load_dataset()

        print("\nConnecting to llama-server...")

        health = requests.get(
            "http://127.0.0.1:9931/health",
            timeout=10
        )

        health.raise_for_status()

        results = evaluate_dataset(dataset)

        save_results(results)

        print("\nQuality response collection completed.")
        print(f"Total questions processed: {len(results)}")

    except requests.RequestException as error:

        print(f"\nServer or network error: {error}")

    except (ValueError, KeyError, FileNotFoundError) as error:

        print(f"\nData or configuration error: {error}")
