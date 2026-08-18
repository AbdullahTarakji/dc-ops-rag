"""Reciprocal Rank Fusion: merging result lists that do not share a score scale.

A cosine similarity of 0.83 and a BM25 score of 14.2 say nothing about each other. Any
attempt to blend them needs normalisation, and normalisation needs score distributions that
neither backend guarantees to be stable across queries. RRF sidesteps the problem entirely
by throwing the scores away and keeping only the ranks::

    score(d) = sum over result lists L of  1 / (k + rank_L(d))

A document ranked first in one list and absent from the other still scores well; a document
ranked mid-table in both can outscore it. That "agreement between different retrievers is
evidence" behaviour is what makes hybrid retrieval work.

``k`` damps the influence of the very top ranks. With the conventional ``k = 60`` from
Cormack et al. (2009), rank 1 contributes 1/61 and rank 2 contributes 1/62 — close enough
that one list cannot dictate the fused order on its own.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from dcrag.domain.entities import RetrievedChunk

DEFAULT_RRF_K = 60
"""Conventional damping constant. Exposed in the experiment config so it can be ablated."""


def reciprocal_rank_fusion(
    result_lists: Iterable[Sequence[RetrievedChunk]],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse ranked result lists into one ranking.

    Ranks are taken from each list position rather than from ``RetrievedChunk.rank``, so a
    caller may pass an already-truncated list without the maths going wrong.

    Ties are broken by ``chunk_id`` so the output is deterministic: evaluation runs that
    differ only in dictionary ordering would be impossible to compare.

    Args:
        result_lists: Ranked lists to fuse, each ordered best first.
        k: Damping constant. Larger values flatten the contribution of top ranks.
        top_k: Optional truncation of the fused list.

    Returns:
        Fused results ordered best first, with ``retriever`` set to ``rrf``.

    Raises:
        ValueError: If ``k`` is not positive.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    scores: dict[str, float] = {}
    seen: dict[str, RetrievedChunk] = {}

    for results in result_lists:
        for position, result in enumerate(results):
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + position + 1)
            # Keep the first sighting: identical chunks carry identical text, and holding
            # the earliest one makes the fused output stable across list ordering.
            seen.setdefault(chunk_id, result)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if top_k is not None:
        ordered = ordered[:top_k]

    return [
        seen[chunk_id].with_score(score=score, retriever="rrf", rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered)
    ]
