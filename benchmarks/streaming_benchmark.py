
import csv
import json
import statistics
import uuid
import time

from pathlib import Path
from datetime import datetime

import requests


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Use a new CSV because the old file has a different column structure.
RESULTS_PATH = (
    PROJECT_ROOT
    / "logs"
    / "streaming_results_v2.csv"
)

SERVER_URL = "http://127.0.0.1:9931/completion"

NUMBER_OF_TRIALS = 5

# Change this label if you later restart the server
# with prompt caching enabled.
CACHE_MODE = "enabled"

MODEL_NAME = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

PAYLOAD = {
    "prompt": "Explain what a Large Language Model is in simple terms.",
    "n_predict": 64,
    "temperature": 0,
    "stream": True
}


# ==========================================
# SINGLE STREAMING TRIAL
# ==========================================

def run_trial(trial_number):

    start_time = time.perf_counter()

    first_content_time = None
    generated_content = ""
    final_data = None

    with requests.post(
        SERVER_URL,
        json=PAYLOAD,
        stream=True,
        timeout=(10, 180)
    ) as response:

        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):

            if not line or not line.startswith("data:"):
                continue

            data_text = line[len("data:"):].strip()

            if data_text == "[DONE]":
                break

            try:
                event = json.loads(data_text)
            except ValueError:
                continue

            content = event.get("content", "")

            if content:

                if first_content_time is None:
                    first_content_time = time.perf_counter()

                generated_content += content

            if event.get("stop") is True:
                final_data = event

    end_time = time.perf_counter()

    if first_content_time is None:
        raise RuntimeError(
            f"Trial {trial_number}: no generated content received."
        )

    if not final_data or "timings" not in final_data:
        raise RuntimeError(
            f"Trial {trial_number}: server timing data missing."
        )

    timings = final_data["timings"]

    result = {
        "trial_number": trial_number,
        "cache_mode": CACHE_MODE,

        # Client-observed time until first generated text arrives.
        "ttft_seconds": round(
            first_content_time - start_time, 6
        ),

        "total_request_seconds": round(
            end_time - start_time, 6
        ),

        # Server-side timing information.
        "cache_tokens": timings.get("cache_n"),
        "prompt_tokens": timings.get("prompt_n"),
        "prompt_ms": timings.get("prompt_ms"),

        "generated_tokens": timings.get("predicted_n"),
        "generation_ms": timings.get("predicted_ms"),

        "prompt_tokens_per_second": timings.get(
            "prompt_per_second"
        ),

        "generation_tokens_per_second": timings.get(
            "predicted_per_second"
        ),

        "tpot_ms": timings.get("predicted_per_token_ms"),

        "generated_characters": len(generated_content)
    }

    print(f"\nTrial {trial_number} completed.")

    print(f"Cache mode: {CACHE_MODE}")
    print(f"Cache tokens reused: {result['cache_tokens']}")
    print(f"Prompt tokens processed: {result['prompt_tokens']}")
    print(f"TTFT: {result['ttft_seconds']:.4f} seconds")
    print(
        f"Total request: "
        f"{result['total_request_seconds']:.4f} seconds"
    )
    print(f"Prompt processing: {result['prompt_ms']} ms")
    print(f"Generation time: {result['generation_ms']} ms")
    print(f"TPOT: {result['tpot_ms']} ms/token")
    print(
        f"Generation speed: "
        f"{result['generation_tokens_per_second']} tokens/s"
    )

    return result


# ==========================================
# SAVE RESULTS
# ==========================================

def save_results(results, experiment_id):

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "timestamp",
        "trial_number",
        "model",
        "cache_mode",
        "cache_tokens",
        "ttft_seconds",
        "total_request_seconds",
        "prompt_ms",
        "generation_ms",
        "prompt_tokens",
        "generated_tokens",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "tpot_ms",
        "generated_characters"
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

        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for result in results:

            row = {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "model": MODEL_NAME,
                **result
            }

            writer.writerow(row)

    print(f"\nResults saved to: {RESULTS_PATH}")


# ==========================================
# STATISTICAL SUMMARY
# ==========================================

def print_summary(results):

    metrics = [
        "ttft_seconds",
        "total_request_seconds",
        "prompt_ms",
        "generation_ms",
        "generation_tokens_per_second",
        "tpot_ms"
    ]

    print("\n========== FIVE-TRIAL SUMMARY ==========")

    for metric in metrics:

        values = [
            result[metric]
            for result in results
            if result[metric] is not None
        ]

        if not values:
            continue

        print(f"\n{metric}")
        print(f"Mean: {statistics.mean(values):.4f}")
        print(f"Median: {statistics.median(values):.4f}")

        if len(values) >= 2:
            print(
                f"Standard deviation: "
                f"{statistics.stdev(values):.4f}"
            )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("LOCAL LLM STREAMING BENCHMARK")

    experiment_id = str(uuid.uuid4())[:8]

    print(f"Experiment ID: {experiment_id}")
    print(f"Cache mode: {CACHE_MODE}")
    print(f"Trials: {NUMBER_OF_TRIALS}")

    results = []

    for trial in range(1, NUMBER_OF_TRIALS + 1):

        results.append(run_trial(trial))

    save_results(results, experiment_id)

    print_summary(results)

    print("\nStreaming benchmark completed.")
