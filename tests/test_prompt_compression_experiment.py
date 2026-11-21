from benchmarks.prompt_compression_experiment import summarize


def _record(compression: str, quality: float) -> dict:
    return {
        "phase": "measured", "success": True, "model_id": "qwen", "compression": compression,
        "task_quality": {"score": quality}, "input_tokens": 12,
        "preparation": {"compression": {"output_characters": 40}},
        "telemetry": {"end_to_end_seconds": 1, "ttft_seconds": 0.2, "generation_tokens_per_second": 10, "peak_ram_mb": 100, "energy_proxy_joules": 5},
    }


def test_compression_summary_calculates_quality_delta_from_no_compression():
    summary = summarize([_record("none", 0.5), _record("whitespace", 1.0), _record("conservative", 0.5)])
    assert summary["qwen:whitespace"]["quality_delta_vs_none"] == 0.5
    assert summary["qwen:conservative"]["quality_delta_vs_none"] == 0
    assert summary["qwen:none"]["exact_input_tokens"]["mean"] == 12
