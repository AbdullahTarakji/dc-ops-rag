"""The retrieval pipeline, exercised end to end against fakes.

Every retrieval mode described in the README runs here in milliseconds, with no model
loaded. When the real Qdrant, BM25 and cross-encoder adapters arrive, these tests do not
change: they are written against the ports.
"""

from __future__ import annotations

import pytest

from dcrag.application import RetrievalService
from dcrag.domain import ConfigurationError, RetrievalMode
from dcrag.infrastructure.fakes import HashEmbedder, InMemoryVectorStore

CONTAINMENT_QUESTION = "What does hot aisle and cold aisle containment separate?"
FOULING_QUESTION = "condenser fouling approach temperature TCA"


@pytest.mark.usefixtures("ingested")
class TestRetrievalModes:
    @pytest.mark.parametrize("mode", list(RetrievalMode))
    def test_every_mode_returns_ranked_results(self, retrieval_service, mode):
        outcome = retrieval_service.retrieve(CONTAINMENT_QUESTION, mode=mode, top_k=3)

        assert outcome.mode is mode
        assert 0 < len(outcome.results) <= 3
        assert [result.rank for result in outcome.results] == list(range(len(outcome.results)))

    def test_lexical_retrieval_finds_the_exact_identifier(self, retrieval_service):
        outcome = retrieval_service.retrieve(FOULING_QUESTION, mode=RetrievalMode.BM25, top_k=1)

        # "TCA" is the kind of token dense retrieval blurs and lexical retrieval nails.
        assert "TCA" in outcome.results[0].chunk.text
        assert outcome.results[0].chunk.doc_id == "chiller-manual"

    def test_hybrid_marks_results_as_fused(self, retrieval_service):
        outcome = retrieval_service.retrieve(FOULING_QUESTION, mode=RetrievalMode.HYBRID, top_k=3)

        assert all(result.retriever == "rrf" for result in outcome.results)

    def test_reranking_relabels_and_truncates(self, retrieval_service):
        outcome = retrieval_service.retrieve(
            CONTAINMENT_QUESTION, mode=RetrievalMode.HYBRID, top_k=2, rerank=True
        )

        assert outcome.reranked is True
        assert len(outcome.results) == 2
        assert all(result.retriever == "rerank" for result in outcome.results)

    def test_document_filter_restricts_the_search(self, retrieval_service):
        outcome = retrieval_service.retrieve(
            "temperature", mode=RetrievalMode.HYBRID, top_k=5, doc_ids={"chiller-manual"}
        )

        assert outcome.results, "the filter must not empty a corpus that contains the term"
        assert {result.chunk.doc_id for result in outcome.results} == {"chiller-manual"}


@pytest.mark.usefixtures("ingested")
class TestObservability:
    def test_stage_timings_are_recorded_for_the_stages_that_ran(self, retrieval_service):
        outcome = retrieval_service.retrieve(
            CONTAINMENT_QUESTION, mode=RetrievalMode.HYBRID, top_k=2, rerank=True
        )

        assert set(outcome.stage_ms) == {"embed", "dense", "bm25", "fuse", "rerank"}
        assert outcome.total_ms == pytest.approx(sum(outcome.stage_ms.values()))

    def test_dense_mode_does_not_time_stages_it_never_ran(self, retrieval_service):
        outcome = retrieval_service.retrieve(CONTAINMENT_QUESTION, mode=RetrievalMode.DENSE)

        assert set(outcome.stage_ms) == {"embed", "dense"}

    def test_chunk_ids_expose_the_ranking_for_metrics(self, retrieval_service):
        outcome = retrieval_service.retrieve(CONTAINMENT_QUESTION, top_k=3)

        assert outcome.chunk_ids == tuple(r.chunk.chunk_id for r in outcome.results)


class TestWiringAndValidation:
    def test_hybrid_without_a_lexical_index_fails_loudly(self, embedder, vector_store):
        service = RetrievalService(embedder=embedder, vector_store=vector_store)

        with pytest.raises(ConfigurationError, match="lexical index"):
            service.retrieve("question", mode=RetrievalMode.HYBRID)

    def test_rerank_without_a_reranker_fails_loudly(self, embedder, vector_store, lexical_index):
        service = RetrievalService(
            embedder=embedder, vector_store=vector_store, lexical_index=lexical_index
        )

        with pytest.raises(ConfigurationError, match="no reranker"):
            service.retrieve("question", rerank=True)

    @pytest.mark.parametrize(("question", "top_k"), [("", 5), ("   ", 5), ("valid", 0)])
    def test_rejects_nonsense_arguments(self, retrieval_service, question, top_k):
        with pytest.raises(ValueError, match="must"):
            retrieval_service.retrieve(question, top_k=top_k)

    def test_changing_embedding_model_without_reingesting_is_caught(self):
        # A 64-dimensional index cannot serve a 128-dimensional query. In production this
        # shows up as silently terrible retrieval; here it is a loud error.
        store = InMemoryVectorStore()
        store.ensure_collection(dimension=64)

        with pytest.raises(ConfigurationError, match="re-ingest"):
            store.ensure_collection(dimension=128)

        assert HashEmbedder(dimension=128).dimension == 128
