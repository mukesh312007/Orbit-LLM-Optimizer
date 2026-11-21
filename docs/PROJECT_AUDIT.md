# Project audit

Audited on 2026-11-18 before implementing the integrated system. The repository began as independent Python scripts around `llama-cli` and a locally hosted llama.cpp server. It contained five hand-authored quality prompts, three local Qwen GGUF files, and historical CSV measurements.

## Existing working components

- `hardware/hardware_profiler.py` reports CPU, RAM, GPU adapter metadata, storage, Python, and command availability.
- `benchmarks/benchmark.py` and `benchmark_v2.py` launch `llama-cli`, sample process RSS, and append basic benchmark CSVs.
- Streaming and combined evaluator scripts preserve individual experiment IDs and append historical logs.
- CSV evidence includes CPU-only Qwen 1.5B runs, streaming TTFT/TPOT runs, cache comparison summaries, and manually entered quality scores.

## Gaps and technical debt found

- There was no dependency manifest, package structure, model registry, database, API, optimizer, dashboard, automated tests, or reproducible external dataset pipeline.
- Scripts duplicate constants and directly hard-code server URLs/configuration. Existing CSV schemas are inconsistent; they are retained as raw evidence rather than rewritten.
- RAM monitoring did not include child processes; energy, GPU utilisation, model loading time, confidence intervals, and benchmark orchestration were absent.
- Existing generic human scoring is valuable but does not prove factual correctness. It is retained separately from automated evaluation.

## Environment and constraints

- The local virtual environment initially contained `psutil` and `requests`; project dependencies are now declared in `requirements.txt`.
- `llama-server.exe` and `ollama.exe` are installed. The audited llama.cpp endpoint `127.0.0.1:9931` was not responding. Ollama listed `qwen2.5:0.5b`.
- MySQL executable/service was not detected. The production repository therefore remains uninitialized until MySQL credentials are supplied; SQLite is used only by isolated tests.
- GPU is integrated/shared-memory class hardware. GPU offload and energy claims must be measured on this machine, never inferred from utilisation.

## Implementation sequence

1. Preserve logs and add package/configuration boundaries.
2. Add datasets, metrics, telemetry, MySQL schema and optimizer.
3. Add API/dashboard with explicit unavailable-backend errors.
4. Validate with unit/API tests; run real inference only after a local server is started.
