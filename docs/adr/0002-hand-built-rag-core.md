# ADR 0002 — Build the RAG core by hand, use a framework only for the agent

* **Status:** accepted
* **Date:** 2026-08-18
* **Iteration:** 0

## Context

LangChain or LlamaIndex would provide loaders, splitters, retrievers, a fusion retriever and
a QA chain out of the box. Dutch job postings name LangChain and LangGraph explicitly, so
some exposure to them is worth having. Against that, this project exists to teach how
retrieval-augmented generation actually works and to measure it honestly — and a framework
that hides chunking, fusion and prompt assembly hides exactly the parts an interviewer will
probe and the parts the evaluation needs to vary.

## Decision

Build the retrieval core directly on the primitive libraries: `sentence-transformers`,
`qdrant-client`, `faiss`, `rank_bm25`, and plain HTTP clients for the generators. Fusion,
chunking, prompt assembly and citation parsing are our code, in the application layer.

Use LangGraph for the agent layer only, where the value is real: a tested graph runtime with
tool-call plumbing, step limits and state handling that would be tedious and bug-prone to
reimplement.

## Consequences

**Good.** Every hyperparameter the evaluation wants to vary — chunk size, overlap, `top_k`,
the RRF constant, the rerank pool — is an explicit argument in our own code rather than a
framework default nobody can name. The reasoning behind each choice is in the repository and
can be defended in an interview. Fewer transitive dependencies means a smaller supply-chain
surface and faster installs.

**Costly.** More code to write and maintain: about 120 lines of fusion and retrieval
orchestration that a framework would have supplied. Framework users get new tricks for free;
we do not.

**Risk.** Reimplementing a primitive slightly wrong is a real hazard — hence the hand-worked
arithmetic in `tests/unit/test_fusion.py`, which pins RRF against numbers computed on paper
rather than against whatever the implementation happens to produce.

## Alternatives considered

* **LangChain end to end.** Fastest path to a working demo. Rejected: it would make the
  learning goal harder to reach, and "we used the default retriever" is a weak answer to
  "why this retrieval strategy?".
* **LlamaIndex end to end.** Strong ingestion and indexing abstractions, and it appears in
  the IBM RAG courses. Rejected as the core for the same reason, but worth adding later as an
  alternative `DocumentLoader`/`VectorStore` adapter pair — the ports make that a contained
  experiment rather than a rewrite.
* **No framework at all, agent included.** Rejected: a hand-rolled ReAct loop with step
  limits and tool-call parsing is a week of yak-shaving that teaches little the ports have
  not already taught, and LangGraph is what the postings ask for.
