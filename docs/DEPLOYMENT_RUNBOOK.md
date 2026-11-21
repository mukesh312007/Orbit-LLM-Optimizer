# Orbit local deployment and benchmark runbook

All commands run locally from the project root. The dashboard remains loopback
only at `http://127.0.0.1:8502`; llama.cpp remains loopback only at port 9931.

## Start Orbit

```powershell
.\Start-Agent.ps1
```

The command starts the Orbit API at `http://127.0.0.1:8001/docs` and refuses
to reuse a port belonging to another application. Use `-ApiPort 8010` to pick
a different local port, or `-NoApi` for dashboard-only operation.

## Inspect models and safe full-live plan

```powershell
.\.venv\Scripts\python.exe -c "from core.model_registry import load_models; print([m['model_id'] for m in load_models() if m.get('available_locally')])"
```

The API accepts `minimum_quality`, `selection_weights`,
`full_configuration_matrix`, `max_live_candidates`, and `dry_run`. Always use
`dry_run: true` before submitting a full live matrix: it may include hundreds
of sequential generations.

## Run the full Spider matrix

Start with a small safe validation run, then remove the configuration limit
only when sufficient RAM is available:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.full_matrix_benchmark --model-id qwen2.5-0.5b-q4_k_m --limit 3 --trials 2 --max-configurations 2
.\.venv\Scripts\python.exe -m benchmarks.full_matrix_benchmark --limit 10 --trials 3
```

The runner records safety skips instead of violating its 1.5 GiB RAM reserve.

### Complete reproducible evidence campaign

First inspect the fully local plan (it does not run a model or download data):

```powershell
.\.venv\Scripts\python.exe -m benchmarks.coverage_campaign --spider-limit 25 --trials 3 --max-configurations 6
```

Run the campaign only when the machine can remain available for the sequential
workload. It runs larger Spider coverage for 0.5B/1.5B/3B, thread/context and
cache studies for 1.5B/3B, every prepared general dataset, then generates
calibrated profiles and an auditable manifest in `logs/demonstrations/`.

```powershell
.\.venv\Scripts\python.exe -m benchmarks.coverage_campaign --execute --spider-limit 25 --trials 3 --max-configurations 6
```

Spider raw JSONL checkpoints after every configuration. You can safely inspect
partial evidence if the campaign is stopped; its manifest records every
completed, skipped, and awaiting-local-dataset stage.

### Low-memory staged evidence mode

When the 0.5B model is already running but the full matrix has no safe startup
candidates, benchmark prompt compression and retrieval variants without
restarting or reloading the resident runtime:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.full_matrix_benchmark --model-id qwen2.5-0.5b-q4_k_m --limit 3 --trials 3 --max-configurations 2 --resident-only
```

This records real quality, latency, throughput, RAM and proxy-energy evidence
for the existing thread/context configuration. It does **not** replace the
full matrix: it cannot claim context, thread, cache-flag, model-size, or
quantization comparisons because it never changes the active runtime.

## Build threshold-selectable profiles

```powershell
.\.venv\Scripts\python.exe -m optimizer.calibrated_profiles --minimum-samples 3
```

Only repeated successful measured trials become profiles. Quality must meet the
requested threshold before cost/efficiency selection is applied.

To also build profiles for locally prepared general-task datasets, run:

```powershell
.\.venv\Scripts\python.exe -m optimizer.calibrated_profiles --minimum-samples 3 --include-standard
```

## Run local project-document retrieval evaluation

```powershell
.\.venv\Scripts\python.exe -m benchmarks.general_retrieval_evaluation --trials 3
```

This compares retrieval off/on across all available local models using the five
documents in `datasets/retrieval/` and ten labelled questions in
`datasets/retrieval_eval.json`.

## Run a task-specific quality dataset

First prepare a locally downloaded, licensed official source. Preparation does
not download a dataset automatically. Then run repeated local trials:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.prepare_datasets gsm8k path\to\official.jsonl --limit 50
.\.venv\Scripts\python.exe -m benchmarks.standard_quality_benchmark gsm8k --trials 3
```

The same workflow supports `mmlu`, `mbpp`, `ifeval`, and `cnn_dailymail`.
Scores unavailable from a safe evaluator are retained as unavailable rather
than treated as quality passes.

## MySQL telemetry

Import historical JSONL evidence after a new benchmark type is added:

```powershell
.\.venv\Scripts\python.exe -m benchmarks.import_jsonl_telemetry --dry-run
.\.venv\Scripts\python.exe -m benchmarks.import_jsonl_telemetry
```

JSONL remains the primary local source of evidence. MySQL failure must never
block inference or remove JSONL logging.

Verify the configured MySQL connection and required telemetry tables without
creating, changing, or inserting any data:

```powershell
.\.venv\Scripts\python.exe -m database.verify_telemetry
```

## Optional physical energy meter

Configure `ORBIT_ENERGY_METER_READING_PATH` in local `.env` only when a real
meter export updates a JSON file such as:

```json
{"source":"wall power meter","measurement_kind":"physical_meter","cumulative_joules":12345.6}
```

Orbit reads two cumulative values around inference and records actual joules
only when they are valid, monotonic, and not labelled as simulated, estimated,
or proxy data. CPU utilisation remains an estimated proxy, never an actual
energy reading or an automatic ranking metric.
