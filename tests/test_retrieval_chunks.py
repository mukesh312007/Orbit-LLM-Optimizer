from runtime.prompt_pipeline import _chunks, retrieve


def test_retrieval_chunks_are_bounded_and_keep_source_traceability(tmp_path, monkeypatch):
    from runtime import prompt_pipeline

    corpus = tmp_path / "datasets" / "retrieval"
    corpus.mkdir(parents=True)
    (corpus / "guide.md").write_text("alpha database\n\n" + "beta " * 600, encoding="utf-8")
    monkeypatch.setattr(prompt_pipeline, "ROOT", tmp_path)

    chunks = _chunks((corpus / "guide.md").read_text(encoding="utf-8"))
    matches = retrieve("alpha database", "datasets/retrieval", 2)

    assert all(len(chunk) <= 1600 for chunk in chunks)
    assert matches[0]["path"].endswith("datasets\\retrieval\\guide.md")
    assert matches[0]["chunk_index"] == 0
