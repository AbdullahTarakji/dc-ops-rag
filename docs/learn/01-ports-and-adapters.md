# Lesson 01 — Ports and adapters in Python

**Iteration 0.** Read this alongside `src/dcrag/domain/ports.py` and
`tests/unit/test_ports_conformance.py`.

---

## 1. The concept

A **port** is an interface owned by the inside of your system, describing what it needs from
the outside world. An **adapter** is a concrete implementation of that need.

The direction of ownership is the whole idea. The retrieval service does not say "I use
Qdrant." It says "I need something that can store vectors and find the nearest ones," and
Qdrant happens to qualify. The interface belongs to the *consumer*, not the provider.

That single inversion is what "dependencies point inwards" means:

```
api / agent / eval   ──►  application  ──►  domain  ◄──  infrastructure
   (composition)            (use cases)      (ports)        (adapters)
```

Everything points at the domain. Nothing points out of it. `src/dcrag/domain/` imports
nothing but the standard library — and that is a rule you can check mechanically, which is
why it is worth having.

## 2. Why it matters here

Three concrete payoffs in this repository, none of them theoretical:

**The test suite runs in about a second with no GPU.** `uv run pytest` exercises ingestion,
dense retrieval, BM25, fusion and reranking end to end — against
`src/dcrag/infrastructure/fakes/`. No model download, no Docker, no network. A pipeline that
can only be tested with a 7B model loaded will not be tested on every commit; this one is.

**The evaluation is possible at all.** The harness compares four retrieval modes, three chunk
sizes, three values of `top_k` and two generators. Behind ports that is a loop over
configurations. Without them it is a nest of `if` statements that nobody trusts.

**The Azure iteration changes no application code.** `AzureOpenAIGenerator` satisfies the
same `Generator` port as `OllamaGenerator`; the composition root picks one. The claim "this
architecture makes the cloud swap trivial" is the sort of thing everyone says in an
interview — here it will be demonstrable from the diff.

## 3. How it is implemented here

### The port

```python
# src/dcrag/domain/ports.py
@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...
```

Two methods rather than one, because modern retrieval models are **asymmetric**: they expect
an instruction prefix on the query side and none on the document side. Getting that backwards
does not crash anything, it just quietly costs recall — the kind of bug the type system can
prevent by refusing to let you call one where the other belongs.

### The adapter

```python
# src/dcrag/infrastructure/fakes/inmemory.py
class HashEmbedder:  # note: inherits from nothing
    @property
    def dimension(self) -> int: ...
    def embed_documents(self, texts): ...
    def embed_query(self, text): ...
```

`HashEmbedder` never imports `Embedder`. It satisfies the protocol **structurally** — by
shape. This is Python's static duck typing, and it is stronger than it looks: mypy verifies
the full signature the moment the adapter is passed to something typed as the port.

### The use case

```python
# src/dcrag/application/retrieval_service.py
class RetrievalService:
    def __init__(self, *, embedder: Embedder, vector_store: VectorStore, ...) -> None:
```

Dependencies arrive through the constructor. The service never constructs a collaborator, and
so it never needs to know which one it got. Everything is keyword-only, because
`RetrievalService(a, b, c, d)` at a call site tells a reader nothing.

### The composition root

One place — `src/dcrag/api/deps.py`, built in the API iteration — reads configuration and
decides that `vector_store` means Qdrant today and FAISS on a laptop. If a second place ever
makes that decision, the pattern is broken.

## 4. `Protocol` versus `ABC`

You know this pattern from C#, where `IEmbedder` is an interface and `SentenceTransformerEmbedder : IEmbedder`
declares the relationship. Python offers both styles.

| | `abc.ABC` (nominal) | `typing.Protocol` (structural) |
|---|---|---|
| Adapter must inherit | yes | no |
| Adapter must import the abstraction | yes | no |
| Third-party class can satisfy it | only via a wrapper | yes, if the shape matches |
| Missing method caught | at instantiation, runtime | at type-check time |
| `isinstance` support | always | only with `@runtime_checkable`, and it checks names only |

This project uses `Protocol` because inheritance would force the dependency to point outwards
— `infrastructure` importing an abstraction is fine, but the *reason* to avoid it is that a
class you do not own (a vendor SDK client) can then never satisfy the port directly.

The catch worth knowing: `@runtime_checkable` `isinstance` checks **only that the attribute
names exist**. It will happily accept an `embed_query` that takes six arguments and returns a
string. That is why `tests/unit/test_ports_conformance.py` does both:

```python
def test_fakes_satisfy_their_ports_at_runtime():
    assert isinstance(HashEmbedder(), Embedder)  # names exist


def test_fakes_satisfy_their_ports_statically():
    embedder: Embedder = HashEmbedder()  # mypy checks the signatures
```

The second test looks like it asserts nothing. It asserts a great deal — at type-check time.

## 5. Trade-offs, honestly

* **More files.** A tutorial does this in one module. You are paying indirection for
  substitutability, which is worth it precisely because this project substitutes things on
  purpose. In a script that will only ever call OpenAI, it would be waste.
* **The abstraction can be wrong.** A port that leaks its implementation — say, a
  `search(..., hnsw_ef: int)` parameter — buys you nothing, because only one backend can
  implement it. Watch for backend-specific arguments creeping into a port signature.
* **Structural typing is quiet.** Nothing announces "this class implements a port." The
  conformance tests and the composition root are where that becomes visible, which is why
  both exist.

## 6. Interview questions

**"Why Protocol instead of ABC?"**
Structural typing keeps the dependency pointing inwards: the adapter never imports the
abstraction, so a class I do not own can satisfy a port directly. mypy checks conformance
statically at the wiring point. I keep `@runtime_checkable` for a coarse runtime assertion,
knowing it only verifies attribute names.

**"How do you test a RAG pipeline without a GPU?"**
Every dependency is behind a port, so the suite wires in-memory fakes: a hash-based embedder,
a dictionary vector store doing exact cosine search, a term-overlap lexical index and a
scripted generator. They are honest fakes, not mocks, so tests assert on ranking behaviour
rather than on which methods were called. The full run takes about a second, which is why it
runs on every push.

**"Where does dependency injection happen?"**
In a single composition root. Services take their collaborators as keyword-only constructor
arguments and never construct one themselves. No framework — at this size a DI container adds
ceremony without adding capability.

**"What is the cost of this architecture?"**
More files, one extra hop of indirection, and a port must exist before an adapter can. And an
abstraction leak is a real failure mode: the moment a port grows a backend-specific
parameter, it stops being substitutable and starts being a header file.

**"Why are the entities dataclasses rather than Pydantic models?"**
Pydantic validates untrusted input, which is a boundary concern; it lives in the API layer.
The domain describes what a chunk *is* and should not depend on a validation library. The
entities are frozen because a reranker that could mutate a retrieved chunk would break the
audit trail from question to citation.

## 7. Your turn

1. **Break a port on purpose.** Rename `embed_query` to `encode_query` in `HashEmbedder`,
   then run `uv run mypy` and `uv run pytest`. Which one tells you first, and what exactly
   does each say? Now rename it in `Embedder` too and watch the error move.
2. **Add a fake that lies.** Write an `Embedder` whose `embed_query` returns a `str`. Confirm
   that `isinstance(..., Embedder)` still passes and that mypy still catches it. This is the
   single most useful thing to internalise about `Protocol`.
3. **Predict, then check.** Before running anything: in `test_agreement_beats_a_single_strong_hit`,
   why does `c` beat `b`? Work out both RRF scores on paper, then read the comment in the
   test.
4. **Find the leak.** Look at `VectorStore.search`. Which parameter would become a leak if a
   backend needed it — and how would you keep it out of the port?

## 8. Further reading

* Alistair Cockburn, *Hexagonal Architecture* — the original ports-and-adapters write-up.
* PEP 544, *Protocols: Structural subtyping* — the specification, short and readable.
* Robert C. Martin, *Clean Architecture*, chapters 21–22 — the dependency rule.
* Percival & Gregory, *Architecture Patterns with Python* — the same ideas, in Python.
