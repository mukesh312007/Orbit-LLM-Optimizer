from core.model_registry import load_models
from optimizer.configuration_catalog import benchmark_configurations
from optimizer.live_matrix import full_live_candidates, stratified_sample


def test_local_quantization_variants_are_discovered_without_registry_edits():
    models = load_models()
    ids = {model["model_id"] for model in models if model.get("available_locally")}
    assert "qwen2.5-1.5b-q5_k_m" in ids
    assert "qwen2.5-1.5b-q6_k" in ids


def test_generic_gguf_discovery_labels_unknown_architecture_without_guessing(tmp_path, monkeypatch):
    from core import model_registry
    path = tmp_path / "custom-q4_k_m.gguf"
    path.write_bytes(b"GGUF" + b"metadata")
    monkeypatch.setattr(model_registry, "ROOT", tmp_path)
    model = model_registry._model_from_file(path)
    assert model["family"] == "Unknown GGUF"
    assert model["quantization"] == "Q4_K_M"


def test_full_benchmark_matrix_covers_every_requested_runtime_dimension():
    configurations = benchmark_configurations({"model_id": "test-model"})
    assert len(configurations) == 4 * 3 * 2 * 3 * 2
    assert {item["threads"] for item in configurations} == {2, 4, 6, 8}
    assert {item["context_size"] for item in configurations} == {1024, 2048, 4096}
    assert {item["cache"] for item in configurations} == {"prompt-cache-enabled", "prompt-cache-disabled"}
    assert {item["retrieval"]["enabled"] for item in configurations} == {False, True}


def test_full_live_matrix_skips_memory_unsafe_configurations(monkeypatch):
    from optimizer import live_matrix
    monkeypatch.setattr(live_matrix, "ROOT", __import__("pathlib").Path("."))
    # The actual model path is irrelevant because required_memory_mb is mocked.
    monkeypatch.setattr(live_matrix, "required_memory_mb", lambda *_: 500)
    model = {"model_id": "test", "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    candidates, skipped = full_live_candidates([model], available_mb=2500)
    assert len(candidates) == 144
    assert not skipped


def test_full_live_matrix_reclaims_orbit_runtime_ram(monkeypatch):
    from optimizer import live_matrix
    monkeypatch.setattr(live_matrix.psutil, "virtual_memory", lambda: type("Memory", (), {"available": 1_000 * 1024**2})())
    monkeypatch.setattr(live_matrix, "_reclaimable_managed_runtime_mb", lambda: 2_000)
    monkeypatch.setattr(live_matrix, "required_memory_mb", lambda *_: 1_000)
    model = {"model_id": "test", "path": "models/qwen2.5-0.5b-instruct-q4_k_m.gguf", "available_locally": True, "backends": ["llama.cpp"]}
    candidates, skipped = full_live_candidates([model])
    assert len(candidates) == 144
    assert not skipped



# ---------------------------------------------------------------------------
# stratified_sample tests
# ---------------------------------------------------------------------------

def _make_candidate(model_id: str, cache: str, compression: str, retrieval: bool, threads: int = 2, context: int = 1024) -> dict:
    """Build a minimal candidate dict that stratified_sample can key on."""
    return {
        "model": {"model_id": model_id},
        "configuration": {
            "threads": threads,
            "context_size": context,
            "cache": cache,
            "prompt_compression": compression,
            "retrieval": {"enabled": retrieval},
        },
        "required_mb": 500.0,
    }


def _make_diverse_candidates() -> list[dict]:
    """Return a small set covering multiple distinct strata."""
    candidates = []
    for model_id in ("model-a", "model-b"):
        for cache in ("prompt-cache-enabled", "prompt-cache-disabled"):
            for compression in ("none", "conservative"):
                for retrieval in (False, True):
                    candidates.append(_make_candidate(model_id, cache, compression, retrieval))
    return candidates  # 2*2*2*2 = 16 candidates across 16 strata


def test_stratified_sample_picks_from_multiple_groups():
    """Stratified sampling should draw from more than one stratum."""
    candidates = _make_diverse_candidates()
    sample = stratified_sample(candidates, n=4)
    assert len(sample) == 4
    # The 4 chosen items must come from at least 2 distinct (model_id, cache, compression, retrieval) groups.
    keys = {
        (c["model"]["model_id"],
         c["configuration"]["cache"],
         c["configuration"]["prompt_compression"],
         c["configuration"]["retrieval"]["enabled"])
        for c in sample
    }
    assert len(keys) >= 2


def test_stratified_sample_covers_each_model_before_repeating_one():
    candidates = _make_diverse_candidates()

    sample = stratified_sample(candidates, n=2)

    assert {candidate["model"]["model_id"] for candidate in sample} == {"model-a", "model-b"}


def test_stratified_sample_returns_all_when_n_gte_total():
    """When n >= total candidates, all candidates are returned (shuffled)."""
    candidates = _make_diverse_candidates()
    result = stratified_sample(candidates, n=len(candidates))
    assert len(result) == len(candidates)
    # Same items, possibly different order.
    assert set(id(c) for c in result) == set(id(c) for c in candidates)


def test_stratified_sample_exact_n_returned():
    """The function must return exactly n candidates when n < total."""
    candidates = _make_diverse_candidates()
    for n in (1, 5, 10, 15):
        result = stratified_sample(candidates, n=n)
        assert len(result) == n, f"Expected {n} but got {len(result)}"


def test_stratified_sample_reproducibility():
    """Same seed must produce identical results across calls."""
    candidates = _make_diverse_candidates()
    result_a = stratified_sample(candidates, n=6, seed=42)
    result_b = stratified_sample(candidates, n=6, seed=42)
    assert [id(c) for c in result_a] == [id(c) for c in result_b]


def test_stratified_sample_different_seeds_differ():
    """Different seeds should (almost certainly) produce different orderings for multi-item groups."""
    # Build candidates with multiple items per stratum so the intra-group shuffle is meaningful.
    candidates = []
    for model_id in ("model-a", "model-b"):
        for cache in ("prompt-cache-enabled", "prompt-cache-disabled"):
            for retrieval in (False, True):
                # 4 candidates per stratum so seed affects intra-group order.
                for threads in (2, 4, 6, 8):
                    candidates.append(_make_candidate(model_id, cache, "none", retrieval, threads=threads))
    # 2*2*2*4 = 32 candidates across 8 strata (4 per group)
    result_42 = stratified_sample(candidates, n=16, seed=42)
    result_99 = stratified_sample(candidates, n=16, seed=99)
    # It is astronomically unlikely that both seeds produce identical orders with 32 items.
    assert [id(c) for c in result_42] != [id(c) for c in result_99]


def test_stratified_sample_fewer_candidates_than_groups():
    """When there is only one candidate per group and n < group count, picks n distinct groups."""
    candidates = _make_diverse_candidates()  # exactly 1 candidate per stratum (16 total)
    n = 5
    result = stratified_sample(candidates, n=n)
    assert len(result) == n
    keys = {
        (c["model"]["model_id"],
         c["configuration"]["cache"],
         c["configuration"]["prompt_compression"],
         c["configuration"]["retrieval"]["enabled"])
        for c in result
    }
    # Each selected candidate comes from a unique group because each group has exactly one item.
    assert len(keys) == n


def test_stratified_sample_no_duplicates():
    """Stratified sampling must not return the same candidate twice."""
    candidates = _make_diverse_candidates()
    for n in (4, 8, 12, 16):
        result = stratified_sample(candidates, n=n)
        assert len(set(id(c) for c in result)) == len(result), f"Duplicates found for n={n}"


def test_stratified_sample_single_group_falls_back_to_shuffle():
    """When all candidates share one stratum, still returns exactly n without duplicates."""
    candidates = [_make_candidate("model-x", "prompt-cache-enabled", "none", False, threads=t) for t in range(1, 11)]
    result = stratified_sample(candidates, n=5)
    assert len(result) == 5
    assert len(set(id(c) for c in result)) == 5
