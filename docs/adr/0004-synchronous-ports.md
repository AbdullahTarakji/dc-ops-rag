# ADR 0004 — Synchronous ports, with the thread hop at the HTTP edge

* **Status:** accepted
* **Date:** 2026-08-18
* **Iteration:** 0

## Context

A query touches four kinds of work: embedding (GPU or CPU bound), vector search (network for
Qdrant, in-process for FAISS), reranking (GPU or CPU bound), and generation (a long network
call). FastAPI is asynchronous, and the obvious instinct is to make every port `async`.

But `sentence-transformers` and `faiss` are blocking, CPU/GPU-bound libraries with no async
API. Declaring `async def embed_query` around a blocking call does not make it concurrent —
it blocks the event loop while pretending not to, which is worse than being honest about it.

Defining both variants of every port doubles the surface of the abstraction and every fake.

## Decision

Every port is synchronous. Concurrency is handled at the edge: FastAPI route handlers
declared with `def` (not `async def`) are run by Starlette in a worker thread, so a slow
generation call cannot stall the event loop. Where finer control is needed, the API layer
calls `anyio.to_thread.run_sync` explicitly.

## Consequences

**Good.** The domain, the application layer, every adapter and every test stay plain
synchronous Python — readable, debuggable, and free of `asyncio.run` scattered through the
evaluation scripts, which are batch jobs with no event loop of their own. Fakes stay trivial.

**Costly.** Each request occupies a worker thread for its whole lifetime, so the practical
concurrency limit is the thread pool size rather than the number of in-flight sockets. For a
system whose bottleneck is a single local GPU serving one generation at a time, that ceiling
is far above the real one.

**Deferred.** Streaming responses (server-sent events) do need async at the API layer. That
is contained: the route is async, and it consumes a synchronous generator running in a
worker thread.

**Revisit when:** generation moves to a hosted API and the workload becomes many concurrent
network-bound requests. At that point an `AsyncGenerator` port next to the sync one is
justified — and because everything is behind ports, it is an additive change.

## Alternatives considered

* **Async everywhere.** Idiomatic for a modern FastAPI service. Rejected: it would wrap
  blocking libraries in `async` signatures that lie about their behaviour, and it would
  complicate the offline evaluation harness for no gain.
* **Both sync and async ports.** Complete, and what some libraries do. Rejected as premature:
  two of everything, including two of every fake, to serve a load profile this project does
  not have.
