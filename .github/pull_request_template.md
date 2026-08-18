## What and why

<!-- One paragraph: what changed, and what problem it solves. Link the iteration in the
     README roadmap this belongs to. -->

## Evidence

<!-- Numbers, not adjectives. Retrieval or evaluation deltas, benchmark output, a screenshot
     of the MLflow comparison. If a metric moved, say on which split it was measured. -->

## Checklist

- [ ] `uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest` passes locally
- [ ] Behaviour is covered by a test that fails without this change
- [ ] Decisions with alternatives worth recording have an ADR in `docs/adr/`
- [ ] The iteration lesson in `docs/learn/` is written or updated
- [ ] No corpus files, model weights, or secrets are committed
- [ ] Tuning was done on the dev split only; the test split remains sealed
