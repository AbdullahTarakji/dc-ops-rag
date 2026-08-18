# Architecture decision records

A decision worth making is worth being able to defend six months later. Each record states
the context, the decision, its consequences — including the costs — and the alternatives that
were rejected, with the reason.

Records are immutable once accepted. A change of mind means a new record that supersedes the
old one, so the reasoning stays legible in history.

| # | Decision | Status |
|---|---|---|
| [0001](0001-clean-architecture.md) | Clean Architecture with ports as `Protocol` | accepted |
| [0002](0002-hand-built-rag-core.md) | Build the RAG core by hand, framework only for the agent | accepted |
| [0003](0003-uv-and-python-311.md) | uv with a pinned Python 3.11, not conda | accepted |
| [0004](0004-synchronous-ports.md) | Synchronous ports, thread hop at the HTTP edge | accepted |

## Format

```markdown
# ADR NNNN — Title

* Status: proposed | accepted | superseded by ADR-XXXX
* Date: YYYY-MM-DD
* Iteration: N

## Context     — the forces at play, including the constraints you did not choose
## Decision    — what was decided, in the present tense
## Consequences — good, costly, risky; and what would make you revisit
## Alternatives considered — each with the reason it lost
```
