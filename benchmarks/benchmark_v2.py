
import csv
import re
import time
import shutil
import subprocess
import statistics
import uuid

from pathlib import Path
from datetime import datetime

import psutil


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "logs"
    / "benchmark_results_v2.csv"
)

THREADS = 4
CONTEXT_SIZE = 2048
MAX_OUTPUT_TOKENS = 128

# CPU-only inference
GPU_LAYERS = 0

# Number of repeated trials
NUMBER_OF_TRIALS = 5


# ==========================================
# RESOURCE MONITORING
# ==========================================

def measure_system_resources(process):

    peak_ram = 0

    try:
        monitored_process = psutil.Process(process.pid)

        while process.poll() is None:

            try:
                current_ram = monitored_process.memory_info().rss
                peak_ram = max(peak_ram, current_ram)

            except psutil.NoSuchProcess:
                break

            time.sleep(0.05)

    except psutil.NoSuchProcess:
        pass

    return round(peak_ram / (1024 ** 2), 2)


# ==========================================
# PERFORMANCE METRIC EXTRACTION
# ==========================================

def extract_performance_metrics(output):

    pattern = (
        r"Prompt:\s*([\d.]+)\s*t/s\s*\|\s*"
        r"Generation:\s*([\d.]+)\s*t/s"
    )

    match = re.search(pattern, output)

    if match:
        return {
            "prompt_tokens_per_second": float(match.group(1)),
            "generation_tokens_per_second": float(match.group(2))
        }

    return {
        "prompt_tokens_per_second": None,
        "generation_tokens_per_second": None
    }


# ==========================================
# SINGLE INFERENCE TRIAL
# ==========================================

def run_inference(trial_number):

    llama_path = shutil.which("llama-cli")

    if llama_path is None:
        raise RuntimeError("llama-cli was not found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    prompt = (
        "Explain what a Large Language Model "
        "is in simple terms."
    )

    command = [
        llama_path,
        "-m", str(MODEL_PATH),
        "-p", prompt,
        "-n", str(MAX_OUTPUT_TOKENS),
        "-c", str(CONTEXT_SIZE),
        "-t", str(THREADS),
        "-ngl", str(GPU_LAYERS),
        "--single-turn",
        "--perf",
        "--show-timings",
        "--no-display-prompt",
    ]

    print(f"\nStarting trial {trial_number}/{NUMBER_OF_TRIALS}...")

    start_time = time.perf_counter()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    peak_ram = measure_system_resources(process)

    output, _ = process.communicate()

    end_time = time.perf_counter()

    total_time = end_time - start_time

    if process.returncode != 0:
        print(output)
        raise RuntimeError(f"Trial {trial_number} failed.")

    performance = extract_performance_metrics(output)

    result = {
        "trial_number": trial_number,
        "wall_time_seconds": round(total_time, 3),
        "prompt_tokens_per_second":
            performance["prompt_tokens_per_second"],
        "generation_tokens_per_second":
            performance["generation_tokens_per_second"],
        "peak_process_ram_mb": peak_ram
    }

    print(f"Trial {trial_number} completed.")
    print(f"Execution time: {total_time:.3f} seconds")
    print(f"Peak process RAM: {peak_ram} MiB")
    print(
        "Generation speed:",
        performance["generation_tokens_per_second"],
        "tokens/second"
    )

    return result


# ==========================================
# SAVE INDIVIDUAL TRIAL RESULTS
# ==========================================

def save_results(results, experiment_id):

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "experiment_id",
        "timestamp",
        "trial_number",
        "model",
        "threads",
        "context_size",
        "max_output_tokens",
        "wall_time_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_process_ram_mb"
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
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "trial_number": result["trial_number"],
                "model": MODEL_PATH.name,
                "threads": THREADS,
                "context_size": CONTEXT_SIZE,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "wall_time_seconds": result["wall_time_seconds"],
                "prompt_tokens_per_second":
                    result["prompt_tokens_per_second"],
                "generation_tokens_per_second":
                    result["generation_tokens_per_second"],
                "peak_process_ram_mb":
                    result["peak_process_ram_mb"]
            }

            writer.writerow(row)

    print(f"\nIndividual trial results saved to:\n{RESULTS_PATH}")


# ==========================================
# STATISTICAL SUMMARY
# ==========================================

def print_summary(results):

    metrics = [
        "wall_time_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_process_ram_mb"
    ]

    print("\n========================================")
    print("       BENCHMARK STATISTICAL SUMMARY")
    print("========================================")

    for metric in metrics:

        values = [
            result[metric]
            for result in results
            if result[metric] is not None
        ]

        if not values:
            print(f"\n{metric}: No measurements available")
            continue

        print(f"\n{metric}")
        print(f"Mean:   {statistics.mean(values):.3f}")
        print(f"Median: {statistics.median(values):.3f}")

        if len(values) >= 2:
            print(f"Std Dev: {statistics.stdev(values):.3f}")


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\n========================================")
    print("     LOCAL LLM REPEATED BENCHMARK")
    print("========================================")

    experiment_id = str(uuid.uuid4())[:8]

    print(f"Experiment ID: {experiment_id}")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Number of trials: {NUMBER_OF_TRIALS}")

    all_results = []

    for trial in range(1, NUMBER_OF_TRIALS + 1):

        result = run_inference(trial)
        all_results.append(result)

    save_results(all_results, experiment_id)

    print_summary(all_results)

    print("\nBenchmark completed successfully!")
