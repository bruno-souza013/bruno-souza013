from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], **kwargs: object) -> np.ndarray:
        """Encode a sequence of texts into dense vectors."""


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Split text into overlapping character windows with stable source IDs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: list[Chunk] = []
    start = 0
    index = 1
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        content = normalized[start:end].strip()
        if content:
            chunks.append(Chunk(
                chunk_id=f"{source}#chunk-{index}",
                source=source,
                text=content,
            ))
        if end == len(normalized):
            break
        start = end - overlap
        index += 1
    return chunks


def load_chunks(
    docs_dir: str | Path,
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    """Load local Markdown and TXT files; paths become source identifiers."""
    root = Path(docs_dir)
    if not root.exists():
        return []

    chunks: list[Chunk] = []
    paths = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}
    )
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        relative_source = path.relative_to(root).as_posix()
        chunks.extend(chunk_text(text, relative_source, chunk_size, overlap))
    return chunks


def retrieve(
    question: str,
    chunks: Sequence[Chunk],
    embedder: Embedder,
    top_k: int = 3,
) -> list[RetrievedChunk]:
    """Rank chunks by cosine similarity to the question embedding."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not question.strip() or not chunks:
        return []

    texts = [chunk.text for chunk in chunks]
    vectors = np.asarray(embedder.encode([question, *texts]), dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) + 1:
        raise ValueError("Embedder returned an invalid embedding matrix")

    query_vector = vectors[0]
    document_vectors = vectors[1:]
    query_norm = np.linalg.norm(query_vector)
    document_norms = np.linalg.norm(document_vectors, axis=1)
    denominator = document_norms * query_norm
    similarities = np.divide(
        document_vectors @ query_vector,
        denominator,
        out=np.zeros(len(chunks), dtype=np.float32),
        where=denominator > 0,
    )

    order = np.argsort(similarities)[::-1][: min(top_k, len(chunks))]
    return [
        RetrievedChunk(chunk=chunks[int(i)], score=float(similarities[int(i)]))
        for i in order
    ]
