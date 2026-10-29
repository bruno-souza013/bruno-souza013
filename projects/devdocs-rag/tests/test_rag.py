from pathlib import Path

import numpy as np
import pytest

from app.rag import Chunk, chunk_text, load_chunks, retrieve


class KeywordEmbedder:
    """Deterministic test double; this is not a real language embedding model."""

    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            value = text.lower()
            vectors.append([
                float("python" in value or "pytest" in value),
                float("api" in value or "http" in value),
                float("rag" in value or "retrieval" in value),
            ])
        return np.asarray(vectors, dtype=np.float32)


def test_chunk_text_returns_source_ids_and_overlapping_chunks():
    text = "alpha " * 100
    chunks = chunk_text(text, "sample.md", chunk_size=80, overlap=10)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "sample.md#chunk-1"
    assert chunks[1].chunk_id == "sample.md#chunk-2"
    assert chunks[0].source == "sample.md"


def test_chunk_text_handles_empty_text():
    assert chunk_text("  \n ", "empty.md") == []


@pytest.mark.parametrize(
    "chunk_size, overlap",
    [(0, 0), (10, 10), (10, 11), (10, -1)],
)
def test_chunk_text_rejects_invalid_parameters(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some content", "sample.md", chunk_size, overlap)


def test_load_chunks_reads_markdown_and_text(tmp_path: Path):
    (tmp_path / "guide.md").write_text("Python and pytest.", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("REST API with HTTP.", encoding="utf-8")
    (tmp_path / "ignored.csv").write_text("rag,retrieval", encoding="utf-8")

    chunks = load_chunks(tmp_path, chunk_size=100, overlap=10)
    assert {chunk.source for chunk in chunks} == {"guide.md", "notes.txt"}


def test_retrieve_ranks_relevant_chunk_first():
    chunks = [
        Chunk("one#chunk-1", "one", "Python projects use pytest."),
        Chunk("two#chunk-1", "two", "REST API uses HTTP methods."),
        Chunk("three#chunk-1", "three", "RAG combines retrieval and generation."),
    ]
    results = retrieve("How do I test Python?", chunks, KeywordEmbedder(), top_k=2)
    assert results[0].chunk.source == "one"
    assert len(results) == 2


def test_retrieve_returns_empty_for_empty_question_or_chunks():
    embedder = KeywordEmbedder()
    chunks = [Chunk("one#chunk-1", "one", "Python and pytest.")]
    assert retrieve("  ", chunks, embedder) == []
    assert retrieve("question", [], embedder) == []
