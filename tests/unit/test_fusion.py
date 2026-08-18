"""Reciprocal Rank Fusion.

The interesting property is the one the README claims: agreement between two retrievers
outranks a strong showing in only one. If that ever stops being true, hybrid retrieval stops
being justified, so it is asserted rather than assumed.
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from dcrag.application import reciprocal_rank_fusion
from dcrag.domain import Chunk, RetrievedChunk, SourceRef


def make_result(chunk_id: str, *, rank: int, retriever: str = "dense") -> RetrievedChunk:
    """Build a retrieval result with a given id and rank."""
    return RetrievedChunk(
        chunk=Chunk(
            chunk_id=chunk_id,
            doc_id="doc",
            text=f"text for {chunk_id}",
            ordinal=rank,
            source=SourceRef(doc_id="doc", title="Doc"),
        ),
        score=1.0 / (rank + 1),
        retriever=retriever,
        rank=rank,
    )


def ranked(ids: list[str], retriever: str = "dense") -> list[RetrievedChunk]:
    """Build a ranked result list from ids, best first."""
    return [make_result(cid, rank=rank, retriever=retriever) for rank, cid in enumerate(ids)]


def test_agreement_beats_a_single_strong_hit():
    dense = ranked(["a", "b", "c"])
    lexical = ranked(["c", "b", "d"], retriever="bm25")

    fused = reciprocal_rank_fusion((dense, lexical))
    order = [result.chunk.chunk_id for result in fused]

    # Worked out by hand with k = 60, because a fusion rule you cannot compute on paper is a
    # fusion rule you cannot debug:
    #   c = 1/63 + 1/61 = 0.032266   (third in one list, first in the other)
    #   b = 1/62 + 1/62 = 0.032258   (second in both)
    #   a = 1/61        = 0.016393   (first in one list, absent from the other)
    #   d = 1/63        = 0.015873
    assert order == ["c", "b", "a", "d"]

    # The headline property: appearing in both lists beats a single strong hit. "a" tops the
    # dense list outright and still loses to "b", which is merely second in each.
    assert order.index("b") < order.index("a")

    # And the runner-up margin is razor thin — 0.032266 against 0.032258. Do not read a
    # first-versus-second place in a fused ranking as a meaningful difference.
    assert fused[0].score - fused[1].score < 0.001


def test_scores_are_descending_and_retriever_is_marked():
    fused = reciprocal_rank_fusion((ranked(["a", "b"]), ranked(["b", "c"], "bm25")))

    assert [result.rank for result in fused] == [0, 1, 2]
    assert all(result.retriever == "rrf" for result in fused)
    assert [result.score for result in fused] == sorted(
        (result.score for result in fused), reverse=True
    )


def test_k_damps_the_influence_of_top_ranks():
    dense = ranked(["a", "b"])
    lexical = ranked(["b", "a"], retriever="bm25")

    # Perfectly mirrored lists: the fused scores must be equal whatever k is, and the tie
    # must be broken deterministically by chunk id.
    for k in (1, 60, 1000):
        fused = reciprocal_rank_fusion((dense, lexical), k=k)
        assert [result.chunk.chunk_id for result in fused] == ["a", "b"]
        assert fused[0].score == pytest.approx(fused[1].score)


def test_top_k_truncates_after_fusing_not_before():
    fused = reciprocal_rank_fusion((ranked(["a", "b", "c"]), ranked(["c", "d"], "bm25")), top_k=2)
    assert [result.chunk.chunk_id for result in fused] == ["c", "a"]


def test_empty_input_is_not_an_error():
    assert reciprocal_rank_fusion(()) == []
    assert reciprocal_rank_fusion(([], [])) == []


def test_rejects_non_positive_k():
    with pytest.raises(ValueError, match="positive"):
        reciprocal_rank_fusion((ranked(["a"]),), k=0)


@given(
    left=st.lists(st.sampled_from("abcdef"), unique=True, max_size=6),
    right=st.lists(st.sampled_from("abcdef"), unique=True, max_size=6),
)
def test_fusion_preserves_every_candidate_and_is_deterministic(left, right):
    lists = (ranked(left), ranked(right, "bm25"))

    first = reciprocal_rank_fusion(lists)
    second = reciprocal_rank_fusion(lists)

    assert [r.chunk.chunk_id for r in first] == [r.chunk.chunk_id for r in second]
    assert {r.chunk.chunk_id for r in first} == set(left) | set(right)
