from __future__ import annotations

import json
import os
from pathlib import Path

from sentence_transformers import SentenceTransformer

from app.rag import load_chunks, retrieve


PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(PROJECT_DIR / "data" / "docs")))
QUESTIONS_PATH = PROJECT_DIR / "data" / "evaluation" / "questions.jsonl"
MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
TOP_K = 3


def main() -> None:
    chunks = load_chunks(DOCS_DIR)
    if not chunks:
        raise SystemExit(f"Nenhum documento foi encontrado em {DOCS_DIR}")

    embedder = SentenceTransformer(MODEL_NAME)
    document_vectors = embedder.encode([chunk.text for chunk in chunks])
    rows = [
        json.loads(line)
        for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise SystemExit("O conjunto de avaliação está vazio.")

    hits = 0
    for row in rows:
        results = retrieve(
            row["question"],
            chunks,
            embedder,
            top_k=TOP_K,
            document_vectors=document_vectors,
        )
        found = any(
            result.chunk.source == row["expected_source"] for result in results
        )
        hits += int(found)
        print(
            f"{'HIT ' if found else 'MISS'} | expected={row['expected_source']} | "
            f"question={row['question']}"
        )

    recall_at_k = hits / len(rows)
    print(f"\nRecall@{TOP_K}: {recall_at_k:.3f} ({hits}/{len(rows)})")
    print("Interprete o resultado como avaliação inicial do pequeno conjunto curado.")


if __name__ == "__main__":
    main()
