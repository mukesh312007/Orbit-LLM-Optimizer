# Experiment results

Raw observations remain in `logs/` untouched. The prior project recorded Qwen 1.5B CPU-only benchmark/streaming data and manually scored five custom quality prompts. These files do not constitute cross-model, controlled quantization, external-dataset, energy, or confidence-interval results. Run new experiments with a distinct experiment ID and preserve conditions before presenting comparisons.

On 2026-11-18, a bounded integration smoke test started the local 0.5B Qwen GGUF server on `127.0.0.1`, queried `2 + 2`, and received `4` (43 prompt tokens, 2 completion tokens). The FastAPI generation route also returned `4` with the registered 0.5B model. The temporary server was then stopped. This is a functionality check, not a benchmark result.
