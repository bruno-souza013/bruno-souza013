from __future__ import annotations

import json
import os

import numpy as np
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from app.rag import Chunk, load_chunks, retrieve


BASE_DIR = Path(__file__).resolve().parents[1]
DOCS_DIR = Path(os.getenv("DOCS_DIR", str(BASE_DIR / "data" / "docs")))
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

embedder: SentenceTransformer | None = None
chunks: list[Chunk] = []
document_vectors: np.ndarray | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=8)


class Source(BaseModel):
    source: str
    chunk_id: str
    score: float
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, chunks, document_vectors
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chunks = load_chunks(DOCS_DIR)
    document_vectors = (
        np.asarray(embedder.encode([chunk.text for chunk in chunks]), dtype=np.float32)
        if chunks else None
    )
    yield
    embedder = None
    chunks = []
    document_vectors = None


app = FastAPI(
    title="DevDocs RAG",
    description="Local document retrieval with source-grounded answers via Ollama.",
    version="0.1.0",
    lifespan=lifespan,
)


def generate_with_ollama(question: str, contexts: list[Source]) -> str:
    context_block = "\n\n".join(
        f"[{item.chunk_id}] (source: {item.source})\n{item.excerpt}"
        for item in contexts
    )
    prompt = f"""You are a technical documentation assistant.
Answer in Brazilian Portuguese.
Use only the context below. If the context does not contain the answer, say
that you could not find enough information in the indexed documents.
Do not invent APIs, behavior, configuration values, or sources.
Cite supporting chunk IDs inline, in the form [filename.md#chunk-1].

CONTEXT:
{context_block}

QUESTION:
{question}

ANSWER:"""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode("utf-8")
    request = Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Não foi possível consultar o Ollama. Confirme que ele está "
                "em execução e que o modelo configurado foi baixado."
            ),
        ) from exc

    answer = result.get("response", "").strip()
    if not answer:
        raise HTTPException(status_code=502, detail="O modelo retornou uma resposta vazia.")
    return answer


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "indexed_chunks": len(chunks),
        "embedding_model": EMBEDDING_MODEL,
        "llm_model": OLLAMA_MODEL,
    }


@app.get("/documents")
def documents() -> dict[str, list[str]]:
    return {"documents": sorted({chunk.source for chunk in chunks})}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if embedder is None:
        raise HTTPException(status_code=503, detail="O modelo de embeddings ainda não foi carregado.")
    if not chunks:
        raise HTTPException(
            status_code=503,
            detail=f"Nenhum documento .md ou .txt foi encontrado em {DOCS_DIR}.",
        )

    matches = retrieve(
        request.question,
        chunks,
        embedder,
        top_k=request.top_k,
        document_vectors=document_vectors,
    )
    sources = [
        Source(
            source=item.chunk.source,
            chunk_id=item.chunk.chunk_id,
            score=round(item.score, 4),
            excerpt=item.chunk.text,
        )
        for item in matches
    ]
    answer = generate_with_ollama(request.question, sources)
    return AskResponse(answer=answer, sources=sources)
