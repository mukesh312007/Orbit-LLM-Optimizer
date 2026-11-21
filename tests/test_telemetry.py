from benchmarks.telemetry import derive_performance


def test_derived_throughput_and_tpot():
    value = derive_performance(0, 2, 6, 20, 8)
    assert value.prompt_tokens_per_second == 10
    assert value.generation_tokens_per_second == 2
    assert value.tpot_seconds == .5
