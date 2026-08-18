# ADR 0003 — uv with a pinned Python 3.11, not a conda environment

* **Status:** accepted
* **Date:** 2026-08-18
* **Iteration:** 0

## Context

The development machine runs Python 3.14 system-wide and has a conda installation with an
existing `ml` environment used by another project. Neither is a good base here: PyTorch and
`sentence-transformers` do not support 3.14 yet, and reusing `ml` risks breaking a project
that already works.

The repository also has to install identically in GitHub Actions, and a reviewer cloning it
should reach a green test run without reading a wiki page.

## Decision

Use [uv](https://docs.astral.sh/uv/) with a repository-local `.venv` and Python pinned to
3.11 in `.python-version`. uv downloads that interpreter itself, so no system Python has to
change. Dependencies are declared in `pyproject.toml` and resolved into `uv.lock`, which is
committed; CI installs with `UV_FROZEN=1` so the lockfile decides versions.

Heavy dependencies live in optional extras (`ingest`, `retrieval`, `api`, `agent`, `eval`,
`ui`) rather than in the base install. `uv sync` gives a working test environment without
downloading a machine-learning stack.

## Consequences

**Good.** One command — `uv sync` — reproduces the environment on any machine and in CI, and
the same lockfile is used in both. The existing conda environments are untouched. CI stays
fast because the default install has no torch in it, which is possible only because the test
suite runs on fakes.

**Costly.** uv must be installed first. Extras have to be requested explicitly, so a
missing `--extra retrieval` shows up as an `ImportError` rather than a slow implicit
install; the README documents which iteration needs which extra.

**Risk.** uv is young and moves quickly. Mitigated by the lockfile and by uv being a drop-in
replacement for pip: `pip install -e .` remains a fallback if it ever stops being the right
tool.

## Alternatives considered

* **Conda environment `rag`, as the original project plan proposed.** Familiar on this
  machine, and good at non-Python binaries. Rejected: environments live outside the
  repository, are not lockfile-reproducible by default, and are awkward in GitHub Actions.
* **Plain venv plus `requirements.txt`.** Universally understood. Rejected: no dependency
  groups, no lockfile without extra tooling, and much slower installs.
* **Poetry.** Mature, with a real lockfile. Rejected: slower, and it does not manage the
  Python interpreter itself, which is the specific problem here.
