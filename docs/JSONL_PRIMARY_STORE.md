# JSONL Operational Primary Store

## Overview & Architecture

Orbit (Local LLM Runtime Optimizer) adopts an **evidence-first, local-first** storage architecture where **JSON Lines (JSONL)** is the authoritative, operational primary store for all telemetry, benchmark evaluations, and runtime decisions.

```
       [ Local llama.cpp / Benchmarks / Dashboard ]
                           |
                           v
        [ Primary Store: Append-only JSONL Evidence ]
         ├── logs/full_matrix_benchmarks/*.jsonl
         ├── logs/quantization_comparisons/*.jsonl
         ├── logs/context_thread_calibrations/*.jsonl
         ├── logs/cache_prompt_experiments/*.jsonl
         ├── logs/standard_quality_benchmarks/*.jsonl
         └── logs/runtime_selection_history.jsonl
                           |
               (Asynchronous / Best-effort)
                           v
        [ Optional Mirror: Relational MySQL Store ]
         (Failure or absence never blocks runtime)
```

## Why JSONL is the Justified Source of Truth

1. **Append-Only Immutability & Crash Resilience**:
   During multi-hour benchmark campaigns, partial results are flushed after every configuration. An unexpected process interrupt preserves all completed evidence up to that instant.
2. **Zero Infrastructure Overhead**:
   A local LLM optimizer must operate on developer laptops without requiring background services, root privileges, or external network configurations.
3. **Reproducibility & Provenance**:
   Each JSONL line records complete experiment metadata: model SHA-256, runtime thread/context configuration, llama.cpp version, hardware CPU/RAM snapshot, and exact input/output token counts.
4. **Air-Gapped & Privacy-Preserving**:
   Telemetry files reside directly within the user-controlled workspace. By default, prompt response text is excluded unless `ORBIT_RETAIN_LIVE_RESPONSE_TEXT` is explicitly enabled.
5. **Direct Aggregation Pipeline**:
   The calibration and profile optimizer (`optimizer/calibrated_profiles.py`) directly consumes JSONL records to compute non-dominated Pareto frontiers and conservative lower confidence bounds without relational overhead.

## Optional MySQL Telemetry Mirror

MySQL support exists as an optional auxiliary mirror for environments with an existing centralized database:
- Configured via `DATABASE_URL` in `.env` (e.g., `mysql+pymysql://...`).
- Mirrored operations: `mirror_live_selection`, `mirror_benchmark_trial`, `mirror_hardware_profile`.
- **Fault-Isolated**: Any database timeout, connection error, or schema discrepancy is caught and reported gracefully without raising exceptions or degrading local inference.
- Read-only connection health can be inspected at any time via `database/verify_telemetry.py` or the `/api/v1/store-status` endpoint.
