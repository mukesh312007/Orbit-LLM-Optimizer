
import csv
import statistics

from pathlib import Path


# ==========================================
# CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = PROJECT_ROOT / "logs" / "streaming_results_v2.csv"

OUTPUT_PATH = PROJECT_ROOT / "logs" / "cache_comparison.csv"

METRICS = [
    "ttft_seconds",
    "total_request_seconds",
    "prompt_ms",
    "generation_ms",
    "generation_tokens_per_second",
    "tpot_ms"
]


# ==========================================
# LOAD CSV DATA
# ==========================================

def load_results():

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found: {INPUT_PATH}"
        )

    with open(INPUT_PATH, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise ValueError("The results CSV is empty.")

    return rows


# ==========================================
# CONVERT CSV VALUES
# ==========================================

def convert_rows(rows):

    converted = []

    for row in rows:

        for metric in METRICS + ["cache_tokens"]:
            value = row.get(metric)

            row[metric] = (
                float(value)
                if value not in (None, "")
                else None
            )

        row["trial_number"] = int(row["trial_number"])

        converted.append(row)

    return converted


# ==========================================
# CLASSIFY CACHE CONDITIONS
# ==========================================

def classify_trials(rows):

    disabled = []
    enabled_cold = []
    enabled_warm = []

    for row in rows:

        if row["cache_mode"] == "disabled":
            disabled.append(row)

        elif row["cache_mode"] == "enabled":

            if row["cache_tokens"] and row["cache_tokens"] > 0:
                enabled_warm.append(row)
            else:
                enabled_cold.append(row)

    return disabled, enabled_cold, enabled_warm


# ==========================================
# CALCULATE STATISTICS
# ==========================================

def calculate_statistics(rows, metric):

    values = [
        row[metric]
        for row in rows
        if row[metric] is not None
    ]

    if not values:
        return None

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "std_dev": (
            statistics.stdev(values)
            if len(values) >= 2
            else 0
        )
    }


# ==========================================
# PRINT GROUP SUMMARY
# ==========================================

def print_group_summary(name, rows):

    print(f"\n========== {name} ==========")
    print(f"Number of trials: {len(rows)}")

    if not rows:
        print("No trials available.")
        return

    experiment_ids = sorted({
        row["experiment_id"] for row in rows
    })

    print(f"Experiment IDs: {', '.join(experiment_ids)}")

    for metric in METRICS:

        stats = calculate_statistics(rows, metric)

        if stats is None:
            continue

        print(f"\n{metric}")
        print(f"Mean: {stats['mean']:.4f}")
        print(f"Median: {stats['median']:.4f}")
        print(f"Standard deviation: {stats['std_dev']:.4f}")


# ==========================================
# COMPARE PERFORMANCE
# ==========================================

def print_comparison(disabled, warm):

    print("\n========== UNCACHED vs WARM CACHE ==========")

    if not disabled or not warm:
        print("Cannot compare: one or both groups are empty.")
        return

    for metric in METRICS:

        uncached_stats = calculate_statistics(disabled, metric)
        cached_stats = calculate_statistics(warm, metric)

        if not uncached_stats or not cached_stats:
            continue

        uncached = uncached_stats["mean"]
        cached = cached_stats["mean"]

        if uncached == 0:
            continue

        if metric == "generation_tokens_per_second":
            improvement = ((cached - uncached) / uncached) * 100
            direction = "higher"
        else:
            improvement = ((uncached - cached) / uncached) * 100
            direction = "lower"

        print(f"\n{metric}")
        print(f"Uncached mean: {uncached:.4f}")
        print(f"Warm-cache mean: {cached:.4f}")
        print(f"Change: {abs(improvement):.2f}% {direction}")

        if improvement < 0:
            print("Result: performance worsened for this metric.")
        else:
            print("Result: performance improved for this metric.")


# ==========================================
# SAVE COMPARISON REPORT
# ==========================================

def save_comparison(disabled, enabled_cold, enabled_warm):

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    groups = [
        ("uncached", disabled),
        ("enabled_cold", enabled_cold),
        ("enabled_warm", enabled_warm)
    ]

    fieldnames = [
        "condition",
        "metric",
        "trial_count",
        "mean",
        "median",
        "standard_deviation"
    ]

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for condition, group in groups:

            for metric in METRICS:

                stats = calculate_statistics(group, metric)

                if stats is None:
                    continue

                writer.writerow({
                    "condition": condition,
                    "metric": metric,
                    "trial_count": stats["count"],
                    "mean": round(stats["mean"], 6),
                    "median": round(stats["median"], 6),
                    "standard_deviation": round(
                        stats["std_dev"], 6
                    )
                })

    print(f"\nComparison report saved to: {OUTPUT_PATH}")


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("LOCAL LLM CACHE COMPARISON ANALYSER")

    rows = convert_rows(load_results())

    disabled, enabled_cold, enabled_warm = classify_trials(rows)

    print(f"\nTotal records loaded: {len(rows)}")

    print_group_summary("PROMPT CACHING DISABLED", disabled)

    print_group_summary("PROMPT CACHING ENABLED - COLD", enabled_cold)

    print_group_summary("PROMPT CACHING ENABLED - WARM", enabled_warm)

    print_comparison(disabled, enabled_warm)

    save_comparison(disabled, enabled_cold, enabled_warm)

    print("\nAnalysis completed successfully.")
