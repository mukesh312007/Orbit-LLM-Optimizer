"""Local-only prompt compression and optional lexical retrieval."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from core.settings import ROOT


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


def _chunks(text: str, maximum_characters: int = 1600) -> list[str]:
    """Create deterministic paragraph chunks without adding remote embeddings."""
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > maximum_characters:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(paragraph[index:index + maximum_characters] for index in range(0, len(paragraph), maximum_characters))
        elif current and len(current) + len(paragraph) + 2 > maximum_characters:
            chunks.append(current)
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current)
    return chunks or [text[:maximum_characters]]


def compress_prompt(prompt: str, mode: str) -> tuple[str, dict[str, Any]]:
    if mode == "none":
        return prompt, {"mode": mode, "input_characters": len(prompt), "output_characters": len(prompt)}
    if mode == "whitespace":
        compressed = "\n".join(" ".join(line.split()) for line in prompt.splitlines() if line.strip())
        return compressed, {"mode": mode, "input_characters": len(prompt), "output_characters": len(compressed)}
    if mode == "conservative":
        # Preserve all nonblank lines byte-for-byte (including code/SQL
        # indentation) and only collapse redundant blank-line runs.
        lines = prompt.splitlines()
        compressed_lines: list[str] = []
        previous_blank = True
        for line in lines:
            blank = not line.strip()
            if blank and previous_blank:
                continue
            compressed_lines.append(line)
            previous_blank = blank
        if compressed_lines and not compressed_lines[-1].strip():
            compressed_lines.pop()
        compressed = "\n".join(compressed_lines)
        return compressed, {"mode": mode, "input_characters": len(prompt), "output_characters": len(compressed)}
    raise ValueError(f"Unsupported prompt compression mode: {mode}")


def retrieve(prompt: str, corpus_dir: str, top_k: int) -> list[dict[str, Any]]:
    root = ROOT / corpus_dir
    if not root.exists():
        return []
    query = _tokens(prompt)
    matches: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")[:20_000]
        for chunk_index, chunk in enumerate(_chunks(text)):
            score = len(query & _tokens(chunk))
            if score:
                matches.append({
                    "path": str(path.relative_to(ROOT)), "chunk_index": chunk_index,
                    "text": chunk, "score": score,
                })
    return sorted(matches, key=lambda item: item["score"], reverse=True)[:top_k]


def prepare_prompt(prompt: str, configuration: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    compressed, compression = compress_prompt(prompt, configuration.get("prompt_compression", "none"))
    retrieval = configuration.get("retrieval", {})
    documents = retrieve(compressed, retrieval.get("corpus_dir", "datasets/retrieval"), int(retrieval.get("top_k", 3))) if retrieval.get("enabled") else []
    if not documents:
        return compressed, {"compression": compression, "retrieval_documents": [], "retrieval_enabled": bool(retrieval.get("enabled"))}
    context = "\n\n".join(f"SOURCE: {item['path']} [chunk {item['chunk_index']}]\n{item['text']}" for item in documents)
    return f"Use only this local reference context when it is relevant.\n\n{context}\n\nUSER REQUEST:\n{compressed}", {
        "compression": compression,
        "retrieval_documents": [{"path": item["path"], "chunk_index": item["chunk_index"], "score": item["score"]} for item in documents],
        "retrieval_enabled": True,
        "retrieval_method": "local lexical-overlap over deterministic paragraph chunks",
    }
