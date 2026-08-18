# Contributing

This is a personal portfolio and learning project, but it is run like a team repository —
that is part of the point. If you want to open an issue or a pull request, you are welcome.

## Setup

```bash
uv sync && uv run pre-commit install
```

Optional extras are installed per iteration, so you only download what you need:

```bash
uv sync --extra retrieval --extra ingest
```

## The quality gate

Exactly what CI runs:

```bash
uv run ruff format --check . && uv run ruff check . && uv run mypy && uv run pytest
```

Tests run against in-memory fakes: no GPU, no Docker, no network. If a change makes the suite
need any of those, the dependency has escaped its port — fix that instead of marking the test
`slow`.

## Conventions

* **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/):
  `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`, `perf:`, `ci:`.
* **Layering**: `domain/` imports only the standard library; `application/` imports only the
  domain. Client libraries live in `infrastructure/`.
* **Docstrings** explain *why*. What the code does should be legible from the code.
* **Decisions** with real alternatives get an ADR in `docs/adr/`.
* **Iterations** ship a lesson in `docs/learn/`.

## Evaluation discipline

Two rules that are not negotiable, because breaking either makes every published number
meaningless:

1. **Tuning happens on the dev split.** The test split is run once, with the frozen
   configuration, at the end.
2. **Every reported score names the judge.** If a model graded an answer, the model is named
   and its agreement with human labels is reported.

## Corpus

Corpus documents are never committed. `scripts/download_corpus.py` fetches them, and
`docs/corpus.md` records each source's licence and retrieval date. Do not add a source whose
licence forbids the use, and do not commit a PDF "just for testing" — use a fixture in
`tests/fixtures/` instead.
