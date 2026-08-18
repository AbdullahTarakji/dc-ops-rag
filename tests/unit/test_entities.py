"""Entity invariants.

These tests document what the domain refuses to represent. An entity that cannot be built
in an invalid state removes a whole category of downstream bug: no service needs to check
whether a chunk has provenance, because a chunk without provenance cannot exist.
"""

from __future__ import annotations

import dataclasses

import pytest

from dcrag.domain import (
    Answer,
    Chunk,
    Citation,
    Document,
    DomainValidationError,
    GenerationResult,
    RetrievalMode,
    RetrievedChunk,
    SourceRef,
    content_hash,
)


class TestSourceRef:
    def test_describe_includes_page_and_section(self):
        ref = SourceRef(doc_id="eu-coc", title="EU CoC 2025", page=42, section_path=("Cooling",))
        assert ref.describe() == "EU CoC 2025 - p.42 - Cooling"

    def test_describe_without_page_or_section(self):
        assert SourceRef(doc_id="d", title="Title").describe() == "Title"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"doc_id": " ", "title": "t"}, "doc_id"),
            ({"doc_id": "d", "title": ""}, "title"),
            ({"doc_id": "d", "title": "t", "page": 0}, "1-based"),
        ],
    )
    def test_rejects_invalid_state(self, kwargs, message):
        with pytest.raises(DomainValidationError, match=message):
            SourceRef(**kwargs)


class TestChunk:
    def test_fingerprint_is_stable_and_content_addressed(self, chunk):
        assert chunk.fingerprint == content_hash(chunk.text)
        assert len(chunk.fingerprint) == 16

    def test_is_immutable(self, chunk):
        with pytest.raises(dataclasses.FrozenInstanceError):
            chunk.text = "rewritten"

    def test_provenance_must_agree_with_owner(self, source_ref):
        with pytest.raises(DomainValidationError, match="disagrees"):
            Chunk(
                chunk_id="other#0000",
                doc_id="other",
                text="text",
                ordinal=0,
                source=source_ref,
            )

    @pytest.mark.parametrize("text", ["", "   ", "\n"])
    def test_rejects_empty_text(self, source_ref, text):
        with pytest.raises(DomainValidationError, match="text"):
            Chunk(chunk_id="doc#0000", doc_id="doc", text=text, ordinal=0, source=source_ref)


class TestDocument:
    def test_rejects_negative_page_count(self):
        with pytest.raises(DomainValidationError, match="page_count"):
            Document(doc_id="d", title="t", source_uri="file:///d", markdown="", page_count=-1)


class TestRetrievedChunk:
    def test_rescoring_produces_a_new_object(self, chunk):
        original = RetrievedChunk(chunk=chunk, score=0.4, retriever="dense", rank=3)
        rescored = original.with_score(score=0.9, retriever="rerank", rank=0)

        assert (rescored.score, rescored.retriever, rescored.rank) == (0.9, "rerank", 0)
        assert original.score == 0.4, "rescoring must not mutate the original ranking"
        assert rescored.chunk is chunk

    def test_rejects_negative_rank(self, chunk):
        with pytest.raises(DomainValidationError, match="rank"):
            RetrievedChunk(chunk=chunk, score=1.0, retriever="dense", rank=-1)


class TestAnswer:
    def test_abstention_may_not_carry_citations(self, source_ref):
        citation = Citation(block=1, chunk_id="doc#0000", source=source_ref)
        with pytest.raises(DomainValidationError, match="abstaining"):
            Answer(text="Not in the documentation.", citations=(citation,), abstained=True)

    def test_cited_chunk_ids(self, source_ref):
        answer = Answer(
            text="Containment is effective. [1][2]",
            citations=(
                Citation(block=1, chunk_id="doc#0000", source=source_ref),
                Citation(block=2, chunk_id="doc#0001", source=source_ref),
            ),
        )
        assert answer.cited_chunk_ids == frozenset({"doc#0000", "doc#0001"})

    def test_citation_blocks_are_one_based(self, source_ref):
        with pytest.raises(DomainValidationError, match="1-based"):
            Citation(block=0, chunk_id="doc#0000", source=source_ref)


class TestGenerationResult:
    def test_total_tokens(self):
        result = GenerationResult(text="ok", model="m", prompt_tokens=120, completion_tokens=30)
        assert result.total_tokens == 150

    def test_rejects_negative_accounting(self):
        with pytest.raises(DomainValidationError, match="latency_ms"):
            GenerationResult(text="ok", model="m", latency_ms=-1.0)


def test_retrieval_modes_are_stable_strings():
    # The API contract, the config files and the evaluation tables all key off these values,
    # so renaming one is a breaking change and this test is here to make that visible.
    assert [mode.value for mode in RetrievalMode] == ["dense", "bm25", "hybrid"]
