import csv
import re
import time
import shutil
import subprocess

from pathlib import Path
from datetime import datetime

import psutil
# ==========================================
# PROJECT CONFIGURATION
# ==========================================

# Locate the root project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Define the model location
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
)

# Define where benchmark results will be stored
RESULTS_PATH = (
    PROJECT_ROOT
    / "logs"
    / "benchmark_results.csv"
)

# Initial inference configuration
THREADS = 4
CONTEXT_SIZE = 2048
MAX_OUTPUT_TOKENS = 128

# CPU-only inference
GPU_LAYERS = 0
def measure_system_resources(process):

    peak_ram = 0

    try:
        monitored_process = psutil.Process(process.pid)

        while process.poll() is None:

            try:
                current_ram = monitored_process.memory_info().rss

                peak_ram = max(
                    peak_ram,
                    current_ram
                )

            except psutil.NoSuchProcess:
                break

            time.sleep(0.05)

    except psutil.NoSuchProcess:
        pass

    return round(peak_ram / (1024 ** 2), 2)

def run_inference():

    llama_path = shutil.which("llama-cli")

    if llama_path is None:
        raise RuntimeError("llama-cli was not found.")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

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

    print("\nStarting benchmark...\n")

    # Record starting time
    start_time = time.perf_counter()

    # Start llama.cpp as a separate process
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace"
    )

    # Monitor RAM while llama.cpp is running
    peak_ram = measure_system_resources(process)

    # Collect generated output
    output, _ = process.communicate()

    # Record ending time
    end_time = time.perf_counter()

    total_time = end_time - start_time

    if process.returncode != 0:
        print(output)
        raise RuntimeError("Inference execution failed.")

    print("\n========== MODEL OUTPUT ==========\n")
    print(output)

    print("\n========== RESOURCE MEASUREMENTS ==========\n")
    print(f"Total execution time: {total_time:.3f} seconds")
    print(f"Peak process RAM: {peak_ram} MB")

    return output, total_time, peak_ram
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
def save_results(total_time, peak_ram, output):

    performance = extract_performance_metrics(output)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": MODEL_PATH.name,
        "threads": THREADS,
        "context_size": CONTEXT_SIZE,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "wall_time_seconds": round(total_time, 3),
        "prompt_tokens_per_second":
            performance["prompt_tokens_per_second"],
        "generation_tokens_per_second":
            performance["generation_tokens_per_second"],
        "peak_process_ram_mb": peak_ram
    }

    # Ensure logs directory exists
    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Check whether CSV already contains data
    file_exists = (
        RESULTS_PATH.exists()
        and RESULTS_PATH.stat().st_size > 0
    )

    # Append results to CSV
    with open(
        RESULTS_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)

    print("\n========== SAVED BENCHMARK ==========\n")

    for key, value in result.items():
        print(f"{key}: {value}")

    print(f"\nResults saved at: {RESULTS_PATH}")
# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print(" LOCAL LLM BENCHMARK SYSTEM")
    print("================================")

    # Run inference and collect measurements
    output, total_time, peak_ram = run_inference()

    # Save experimental results
    save_results(
        total_time,
        peak_ram,
        output
    )

    print("\nBenchmark completed successfully!")
