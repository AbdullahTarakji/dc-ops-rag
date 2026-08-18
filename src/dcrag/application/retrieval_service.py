"""Retrieval use case: find the chunks most likely to answer a question.

This module contains no library imports beyond the standard library and the domain. It is
the pipeline described in the README — dense, lexical, fusion, rerank — expressed purely in
terms of ports, which is why it can be tested end to end with fakes in milliseconds.

Per-stage timings are collected as the pipeline runs rather than bolted on later, because
"where did the 6 seconds go?" is a question the observability iteration must answer and the
evaluation harness reports p95 latency per stage.
"""

from __future__ import annotations

import time
from collections.abc import Collection
from dataclasses import dataclass, field

from dcrag.application.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion
from dcrag.domain.entities import RetrievalMode, RetrievedChunk
from dcrag.domain.errors import ConfigurationError
from dcrag.domain.ports import Embedder, LexicalIndex, Reranker, VectorStore

DEFAULT_TOP_K = 5
"""Chunks handed to the generator. Ablated over 3 / 5 / 8 in the evaluation harness."""

DEFAULT_RERANK_CANDIDATES = 20
"""Candidates fetched before reranking. Retrieve wide, rerank narrow."""


@dataclass(frozen=True, slots=True)
class RetrievalOutcome:
    """What retrieval produced, and what it cost.

    Carrying the stage timings alongside the results means the API can report a latency
    breakdown without a global metrics singleton, and evaluation runs get the numbers for
    free.
    """

    results: tuple[RetrievedChunk, ...]
    """Final ranked chunks, best first."""

    mode: RetrievalMode
    """Retrieval mode used."""

    reranked: bool
    """Whether a cross-encoder reordered the candidates."""

    stage_ms: dict[str, float] = field(default_factory=dict)
    """Wall-clock milliseconds per stage: ``embed``, ``dense``, ``bm25``, ``fuse``, ``rerank``."""

    @property
    def total_ms(self) -> float:
        """Return the summed stage latency.

        Returns:
            Total retrieval latency in milliseconds.
        """
        return sum(self.stage_ms.values())

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        """Return the retrieved chunk ids in rank order.

        Returns:
            Chunk ids, best first. This is what Recall@k and MRR are computed against.
        """
        return tuple(result.chunk.chunk_id for result in self.results)


class RetrievalService:
    """Runs the retrieval half of the pipeline.

    The service owns no state beyond its collaborators, so a single instance is safe to
    share across requests.
    """

    def __init__(
        self,
        *,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_index: LexicalIndex | None = None,
        reranker: Reranker | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            embedder: Turns the question into a vector.
            vector_store: Dense nearest-neighbour search.
            lexical_index: BM25 index. Required for ``bm25`` and ``hybrid`` modes.
            reranker: Cross-encoder. Required when ``rerank=True``.
            rrf_k: Damping constant for fusion.
            rerank_candidates: How many candidates to gather before reranking.
        """
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._rerank_candidates = rerank_candidates

    def retrieve(
        self,
        question: str,
        *,
        mode: RetrievalMode = RetrievalMode.HYBRID,
        top_k: int = DEFAULT_TOP_K,
        rerank: bool = False,
        doc_ids: Collection[str] | None = None,
    ) -> RetrievalOutcome:
        """Retrieve the chunks most likely to answer a question.

        Args:
            question: The user question, in any language the embedder supports.
            mode: Dense, lexical, or hybrid retrieval.
            top_k: How many chunks to return.
            rerank: Whether to apply the cross-encoder to the candidate pool.
            doc_ids: Optional restriction to specific documents.

        Returns:
            The ranked results plus per-stage timings.

        Raises:
            ValueError: If ``question`` is blank or ``top_k`` is not positive.
            ConfigurationError: If the requested mode needs a collaborator that was not
                injected.
        """
        if not question.strip():
            raise ValueError("question must not be blank")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self._check_wiring(mode=mode, rerank=rerank)

        # Fetch a wider pool when a reranker will narrow it again; otherwise exactly top_k.
        pool = max(top_k, self._rerank_candidates) if rerank else top_k
        stage_ms: dict[str, float] = {}

        dense: list[RetrievedChunk] = []
        lexical: list[RetrievedChunk] = []

        if mode in (RetrievalMode.DENSE, RetrievalMode.HYBRID):
            with _StageTimer(stage_ms, "embed"):
                vector = self._embedder.embed_query(question)
            with _StageTimer(stage_ms, "dense"):
                dense = self._vector_store.search(vector, top_k=pool, doc_ids=doc_ids)

        if mode in (RetrievalMode.BM25, RetrievalMode.HYBRID):
            assert self._lexical_index is not None  # noqa: S101 - guaranteed by _check_wiring
            with _StageTimer(stage_ms, "bm25"):
                lexical = self._lexical_index.search(question, top_k=pool, doc_ids=doc_ids)

        if mode is RetrievalMode.HYBRID:
            with _StageTimer(stage_ms, "fuse"):
                candidates = reciprocal_rank_fusion((dense, lexical), k=self._rrf_k, top_k=pool)
        else:
            candidates = dense if mode is RetrievalMode.DENSE else lexical

        if rerank:
            assert self._reranker is not None  # noqa: S101 - guaranteed by _check_wiring
            with _StageTimer(stage_ms, "rerank"):
                results = self._reranker.rerank(question, candidates, top_k=top_k)
        else:
            results = list(candidates[:top_k])

        return RetrievalOutcome(
            results=tuple(results),
            mode=mode,
            reranked=rerank,
            stage_ms=stage_ms,
        )

    def _check_wiring(self, *, mode: RetrievalMode, rerank: bool) -> None:
        """Fail fast when the requested pipeline cannot be assembled.

        Args:
            mode: Requested retrieval mode.
            rerank: Whether reranking was requested.

        Raises:
            ConfigurationError: If a required collaborator is missing.
        """
        needs_lexical = mode in (RetrievalMode.BM25, RetrievalMode.HYBRID)
        if needs_lexical and self._lexical_index is None:
            raise ConfigurationError(
                f"retrieval mode {mode.value!r} needs a lexical index, but none was injected"
            )
        if rerank and self._reranker is None:
            raise ConfigurationError("rerank was requested, but no reranker was injected")


class _StageTimer:
    """Context manager recording elapsed milliseconds into a dictionary.

    A tiny helper rather than a decorator so the stage name stays visible at the call site,
    where a reader is trying to follow the pipeline.
    """

    __slots__ = ("_stage", "_started", "_target")

    def __init__(self, target: dict[str, float], stage: str) -> None:
        """Record into ``target`` under the key ``stage``.

        Args:
            target: Dictionary collecting the timings.
            stage: Stage name used as the key.
        """
        self._target = target
        self._stage = stage
        self._started = 0.0

    def __enter__(self) -> None:
        """Start the clock."""
        self._started = time.perf_counter()

    def __exit__(self, *exc_info: object) -> None:
        """Stop the clock and record the elapsed time."""
        elapsed_ms = (time.perf_counter() - self._started) * 1000
        self._target[self._stage] = self._target.get(self._stage, 0.0) + elapsed_ms
