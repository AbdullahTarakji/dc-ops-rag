# Architecture

The system is layered, with dependencies pointing inwards only. The rationale is
[ADR 0001](adr/0001-clean-architecture.md); the teaching walkthrough is
[Lesson 01](learn/01-ports-and-adapters.md).

## The layers

| Layer | Package | May import | Responsibility |
|---|---|---|---|
| Domain | `dcrag.domain` | standard library only | What things *are*: entities, and the ports the system needs |
| Application | `dcrag.application` | domain | What the system *does*: ingestion, retrieval, evaluation |
| Infrastructure | `dcrag.infrastructure` | anything | *How* it is done: Qdrant, FAISS, bge-m3, BM25, Ollama, fakes |
| Delivery | `dcrag.api`, `dcrag.agent`, `dcrag.eval` | all of the above | Who asks: HTTP, the agent, the evaluation harness |

The rule is mechanically checkable: nothing in `domain/` imports a third-party package, and
nothing in `application/` imports a client library. If either ever does, the abstraction has
leaked.

## The ports

Defined in `src/dcrag/domain/ports.py`, each as a `typing.Protocol`.

| Port | Real adapters (by iteration) | Fake |
|---|---|---|
| `DocumentLoader` | PyMuPDF4LLM, Docling (1) | `MarkdownLoader` |
| `Chunker` | heading-aware token chunker (1) | `ParagraphChunker` |
| `Embedder` | bge-m3 via sentence-transformers (2), Azure OpenAI (8) | `HashEmbedder` |
| `VectorStore` | Qdrant, FAISS (2), Azure AI Search (8) | `InMemoryVectorStore` |
| `LexicalIndex` | rank_bm25 (2) | `InMemoryLexicalIndex` |
| `Reranker` | bge-reranker-v2-m3 (2) | `KeywordReranker` |
| `Generator` | Ollama (3), OpenAI-compatible (3), Azure OpenAI (8) | `ScriptedGenerator` |

## The query path

```
question
  │
  ├─ embed ─────────────► VectorStore.search      (dense candidates)
  ├─────────────────────► LexicalIndex.search     (lexical candidates)
  │                            │
  │                            ▼
  │                    Reciprocal Rank Fusion     (application/fusion.py)
  │                            │
  │                            ▼
  │                    Reranker.rerank            (cross-encoder, top 20 → top 5)
  │                            │
  │                            ▼
  └──────────────────► numbered context blocks ──► Generator.generate
                                                       │
                                                       ▼
                                          Answer{text, citations, abstained}
```

Two invariants hold along that path:

1. **Every chunk carries provenance.** A `Chunk` cannot be constructed without a `SourceRef`,
   so every retrieved chunk can be cited to a document, page and section.
2. **Citations are validated against what was actually retrieved.** A generated citation
   pointing at a block that was never in the context is an error
   (`UngroundedCitationError`), not a warning. This is the check that makes "grounded" mean
   something.

Per-stage latency is recorded as the pipeline runs (`RetrievalOutcome.stage_ms`), so the API
can report a breakdown and the evaluation harness can report p95 per stage without a global
metrics singleton.

## Configuration and composition

Settings come from the environment (see `.env.example`). One composition root —
`dcrag.api.deps` — reads them and decides which adapters to build. Experiment configuration
lives in `configs/*.yaml`, one file per experiment, and every evaluation run records the git
commit alongside its results so any number in the README can be reproduced.
