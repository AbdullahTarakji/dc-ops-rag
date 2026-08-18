# ADR 0001 — Clean Architecture with ports as Protocols

* **Status:** accepted
* **Date:** 2026-08-18
* **Iteration:** 0

## Context

This system has to swap components repeatedly and on purpose. The evaluation harness
compares two PDF loaders, three vector stores, two generators and four retrieval strategies;
a later iteration re-runs the whole evaluation against Azure OpenAI and Azure AI Search. If
those choices are baked into the call sites, every comparison becomes a refactor and most of
them silently do not happen.

There is a second pressure. A pipeline that only runs when a 7B model is loaded and Qdrant is
up cannot be tested on every commit, so it will not be tested on every commit.

## Decision

Layer the codebase, with dependencies pointing inwards only:

| Layer | May import | Contains |
|---|---|---|
| `domain` | standard library only | Entities and ports (`typing.Protocol`) |
| `application` | `domain` | Use cases: ingestion, retrieval, evaluation |
| `infrastructure` | anything | Adapters: Qdrant, FAISS, bge-m3, Ollama, fakes |
| `api` / `agent` / `eval` | all of the above | Delivery mechanisms and the composition root |

Ports are `typing.Protocol` rather than `abc.ABC`. Adapters do **not** inherit from them:
`QdrantStore` never imports `VectorStore`, it simply has methods of the right shape, and
mypy checks the match where the adapter is wired into a service. Composition happens in one
place, `api/deps.py`.

Entities are frozen dataclasses, not Pydantic models. Pydantic guards the HTTP boundary,
where untrusted JSON arrives; the domain should not depend on a validation library to
describe what a chunk is.

## Consequences

**Good.** Every use case is testable with in-memory fakes: the full suite runs in about a
second with no GPU, no Docker and no network, and CI can therefore run it on every push.
Swapping a backend is one new adapter plus one line in the composition root. The Azure
iteration changes no application code, which is a claim the repository can demonstrate
rather than assert.

**Costly.** More files and one more indirection than a single-module script. Every new
capability needs a port defined before an adapter can implement it. For a system whose whole
selling point is comparing implementations, that cost is the point.

**Risk.** A `Protocol` mismatch is only caught where a concrete adapter meets a typed
parameter, so the composition root and the conformance tests in
`tests/unit/test_ports_conformance.py` are load-bearing. Both static and runtime checks are
kept there deliberately.

## Alternatives considered

* **A flat pipeline module.** Fastest to write, and the standard shape of a RAG tutorial.
  Rejected: the ablation matrix in the evaluation iteration would turn into a pile of `if`
  statements, and nothing would be testable without models.
* **`abc.ABC` base classes.** Familiar, and gives a runtime error on a missing method.
  Rejected: inheritance forces adapters to import the abstraction, which points the
  dependency the wrong way; third-party classes could then never satisfy a port directly.
* **A dependency-injection framework** (`dependency-injector`, `wired`). Rejected as
  ceremony: a single `deps.py` is enough at this size, and a reviewer can read it in a
  minute.
