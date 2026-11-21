
import csv
import json
import time
import uuid
import requests

from pathlib import Path
from datetime import datetime


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "datasets" / "quality_eval.json"

RESULTS_PATH = PROJECT_ROOT / "logs" / "combined_results.csv"

SERVER_URL = "http://127.0.0.1:9931/v1/chat/completions"

MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

CACHE_MODE = "disabled"

MAX_OUTPUT_TOKENS = 256

TEMPERATURE = 0

EXPERIMENT_ID = str(uuid.uuid4())[:8]


# ==========================================
# LOAD DATASET
# ==========================================

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        dataset = json.load(file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Dataset must be a non-empty list.")

    print(f"Loaded {len(dataset)} questions.")

    return dataset


# ==========================================
# GENERATE ANSWER AND MEASURE PERFORMANCE
# ==========================================

def generate_response(question):

    payload = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful and accurate assistant. "
                    "Answer the user's question directly and clearly. "
                    "Follow all requested instructions. "
                    "Avoid unnecessary repetition."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": TEMPERATURE,
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

    answer = (
        data["choices"][0]["message"]["content"] or ""
    ).strip()

    usage = data.get("usage", {})

    prompt_tokens = usage.get("prompt_tokens")
    generated_tokens = usage.get("completion_tokens")

    elapsed = end_time - start_time

    # This is client-observed request throughput.
    # It is not pure model generation throughput.
    request_tokens_per_second = None

    if generated_tokens and elapsed > 0:
        request_tokens_per_second = round(
            generated_tokens / elapsed, 4
        )

    finish_reason = data["choices"][0].get("finish_reason")

    return {
        "answer": answer,
        "request_time_seconds": round(elapsed, 6),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "request_tokens_per_second": request_tokens_per_second,
        "finish_reason": finish_reason
    }


# ==========================================
# RUN COMBINED EXPERIMENT
# ==========================================

def run_experiment(dataset):

    results = []

    print(f"\nExperiment ID: {EXPERIMENT_ID}")
    print(f"Model: {MODEL_NAME}")
    print(f"Cache mode: {CACHE_MODE}")

    for index, item in enumerate(dataset, start=1):

        print("\n" + "=" * 55)
        print(f"Question {index}/{len(dataset)}: {item['id']}")
        print(f"Task: {item['task_type']}")

        result = generate_response(item["prompt"])

        record = {
            "experiment_id": EXPERIMENT_ID,
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "model": MODEL_NAME,
            "cache_mode": CACHE_MODE,
            "question_id": item["id"],
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
            "request_tokens_per_second": result[
                "request_tokens_per_second"
            ],
            "finish_reason": result["finish_reason"],
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "threads": 4,
            "context_size": 2048,
            "gpu_layers": 0
        }

        results.append(record)

        print("\nModel answer:")
        print(result["answer"])

        print(
            f"\nRequest latency: "
            f"{result['request_time_seconds']} seconds"
        )

        print(f"Prompt tokens: {result['prompt_tokens']}")
        print(f"Generated tokens: {result['generated_tokens']}")
        print(f"Finish reason: {result['finish_reason']}")

    return results


# ==========================================
# SAVE EXPERIMENT RESULTS
# ==========================================

def save_results(results):

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    fieldnames = [
        "experiment_id",
        "timestamp",
        "model",
        "cache_mode",
        "question_id",
        "task_type",
        "prompt",
        "reference_answer",
        "required_points",
        "model_answer",
        "request_time_seconds",
        "prompt_tokens",
        "generated_tokens",
        "request_tokens_per_second",
        "finish_reason",
        "temperature",
        "max_output_tokens",
        "threads",
        "context_size",
        "gpu_layers"
    ]

    file_exists = (
        RESULTS_PATH.exists()
        and RESULTS_PATH.stat().st_size > 0
    )

    with open(
        RESULTS_PATH,
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

    print(f"\nResults saved to: {RESULTS_PATH}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("COMBINED QUALITY-PERFORMANCE EVALUATOR")

    try:

        health = requests.get(
            "http://127.0.0.1:9931/health",
            timeout=10
        )

        health.raise_for_status()

        dataset = load_dataset()

        results = run_experiment(dataset)

        save_results(results)

        print("\nCombined experiment completed.")
        print(f"Experiment ID: {EXPERIMENT_ID}")
        print(f"Questions processed: {len(results)}")

    except requests.RequestException as error:

        print(f"\nServer or network error: {error}")

    except (ValueError, KeyError, FileNotFoundError) as error:

        print(f"\nConfiguration or data error: {error}")
