# Retrieval-augmented generation design notes

A retrieval-augmented generation pipeline commonly has four stages: document ingestion, chunking, embedding and indexing, and query-time retrieval followed by generation. The generator receives the user question and selected context snippets.

Chunk IDs should preserve a reference to their source document so that a user can inspect the evidence used to construct an answer. Chunk size and overlap affect retrieval quality and should be treated as experimental parameters, not universal constants.

Evaluate retrieval separately from answer quality. Recall at k can measure whether an expected source appears among the top k retrieved items for a curated question set. It does not establish that the generated answer is correct. Answer evaluation should also inspect factual support, source citation quality, refusal behavior when context is insufficient, and latency.

A RAG system can still hallucinate. Clear prompts, citations, evaluation, and explicit handling of insufficient evidence reduce risk but do not guarantee correctness.
