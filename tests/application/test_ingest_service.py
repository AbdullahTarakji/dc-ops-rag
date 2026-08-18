"""Ingestion policy.

The interesting behaviour is not "does it store chunks" but "what happens the second time":
re-ingesting a document that changed must not leave stale chunks behind, because a corpus
that quietly accumulates orphans produces citations to text that no longer exists.
"""

from __future__ import annotations

import pytest

from dcrag.application import IngestService
from dcrag.domain import Document
from dcrag.infrastructure.fakes import (
    HashEmbedder,
    InMemoryVectorStore,
    MarkdownLoader,
    ParagraphChunker,
)


class TestIngestion:
    def test_reports_what_it_did(self, ingested, vector_store, lexical_index):
        report = ingested[0]

        assert report.doc_id == "chiller-manual"
        assert report.chunks > 0
        assert report.tokens > 0
        assert report.embedder == "fake-hash-embedder"
        assert report.chunks_per_second >= 0
        assert vector_store.count() == sum(r.chunks for r in ingested)
        assert lexical_index.count() == vector_store.count()

    def test_chunks_carry_provenance(self, ingested, retrieval_service):
        del ingested
        outcome = retrieval_service.retrieve("supply air temperature range", top_k=1)
        source = outcome.results[0].chunk.source

        assert source.doc_id == "eu-coc-2025"
        assert source.title.startswith("EU Code of Conduct")
        assert source.section_path, "a chunk without a heading trail cannot be cited precisely"

    def test_reingesting_replaces_rather_than_duplicates(
        self, corpus_dir, ingest_service, vector_store
    ):
        path = corpus_dir / "eu-coc-2025.md"
        first = ingest_service.ingest_file(path)
        after_first = vector_store.count()

        second = ingest_service.ingest_file(path)

        assert second.chunks == first.chunks
        assert vector_store.count() == after_first, "re-ingestion must be idempotent"

    def test_a_shortened_document_leaves_no_orphans(self, ingest_service, vector_store):
        long_doc = Document(
            doc_id="notes",
            title="Notes",
            source_uri="file:///notes.md",
            markdown="# Notes\n\nfirst paragraph\n\nsecond paragraph\n\nthird paragraph",
        )
        ingest_service.ingest_document(long_doc)
        assert vector_store.count() == 3

        short_doc = Document(
            doc_id="notes",
            title="Notes",
            source_uri="file:///notes.md",
            markdown="# Notes\n\nfirst paragraph",
        )
        ingest_service.ingest_document(short_doc)

        assert vector_store.count() == 1, "stale chunks would keep being cited"

    @pytest.mark.parametrize("batch_size", [1, 2, 50])
    def test_batch_size_does_not_change_the_result(self, corpus_dir, embedder, batch_size):
        store = InMemoryVectorStore()
        service = IngestService(
            loader=MarkdownLoader(),
            chunker=ParagraphChunker(),
            embedder=embedder,
            vector_store=store,
            batch_size=batch_size,
        )

        report = service.ingest_file(corpus_dir / "chiller-manual.md")

        assert report.chunks == store.count()

    def test_works_without_a_lexical_index(self, corpus_dir, vector_store):
        service = IngestService(
            loader=MarkdownLoader(),
            chunker=ParagraphChunker(),
            embedder=HashEmbedder(),
            vector_store=vector_store,
        )

        assert service.ingest_file(corpus_dir / "chiller-manual.md").chunks > 0


class TestValidation:
    def test_rejects_unsupported_file_types(self, tmp_path, ingest_service):
        pdf = tmp_path / "manual.pdf"
        pdf.write_bytes(b"%PDF-1.7")

        with pytest.raises(ValueError, match="no loader supports"):
            ingest_service.ingest_file(pdf)

    def test_rejects_a_non_positive_batch_size(self, embedder, vector_store):
        with pytest.raises(ValueError, match="batch_size"):
            IngestService(
                loader=MarkdownLoader(),
                chunker=ParagraphChunker(),
                embedder=embedder,
                vector_store=vector_store,
                batch_size=0,
            )
