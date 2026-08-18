"""Shared fixtures.

The whole suite runs against in-memory fakes: no model downloads, no Docker, no network.
That is not a compromise for speed, it is the point of the architecture — see
docs/learn/01-ports-and-adapters.md.
"""

from __future__ import annotations

import pytest

from dcrag.application import IngestService, RetrievalService
from dcrag.domain import Chunk, Document, SourceRef
from dcrag.infrastructure.fakes import (
    HashEmbedder,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    KeywordReranker,
    MarkdownLoader,
    ParagraphChunker,
)

COOLING_DOC = """# EU Code of Conduct on Data Centre Energy Efficiency

## Air flow management

Hot aisle and cold aisle containment separates the supply air from the exhaust air.
Containment is the single most effective air flow management measure in most facilities.

## Temperature and humidity

The recommended supply air temperature range for data centre equipment is 18 to 27 degrees
Celsius. Operating above the recommended range is allowed within the allowable envelope.
"""

CHILLER_DOC = """# Water Cooled Chiller Operation Manual

## Condenser fouling

Condenser fouling raises the condenser approach temperature TCA and increases compressor
power consumption. Clean the condenser tubes when the approach temperature exceeds the
commissioning baseline by more than 2 degrees Fahrenheit.

## Refrigerant charge

A refrigerant leak lowers the evaporator pressure and raises the suction superheat Tsh suc.
Recover and weigh the charge before recharging the circuit.
"""


@pytest.fixture
def corpus_dir(tmp_path):
    """Write the two sample documents to a temporary directory."""
    (tmp_path / "eu-coc-2025.md").write_text(COOLING_DOC, encoding="utf-8")
    (tmp_path / "chiller-manual.md").write_text(CHILLER_DOC, encoding="utf-8")
    return tmp_path


@pytest.fixture
def source_ref():
    """A minimal valid source reference."""
    return SourceRef(doc_id="doc", title="A document", page=1, section_path=("Cooling",))


@pytest.fixture
def chunk(source_ref):
    """A minimal valid chunk."""
    return Chunk(
        chunk_id="doc#0000",
        doc_id="doc",
        text="Containment separates supply air from exhaust air.",
        ordinal=0,
        source=source_ref,
        token_count=7,
    )


@pytest.fixture
def document():
    """A minimal valid document."""
    return Document(
        doc_id="doc",
        title="A document",
        source_uri="file:///doc.md",
        markdown=COOLING_DOC,
        page_count=1,
    )


@pytest.fixture
def vector_store():
    """An empty in-memory vector store."""
    return InMemoryVectorStore()


@pytest.fixture
def lexical_index():
    """An empty in-memory lexical index."""
    return InMemoryLexicalIndex()


@pytest.fixture
def embedder():
    """A deterministic fake embedder."""
    return HashEmbedder(dimension=64)


@pytest.fixture
def ingest_service(embedder, vector_store, lexical_index):
    """An ingestion service wired entirely from fakes."""
    return IngestService(
        loader=MarkdownLoader(),
        chunker=ParagraphChunker(),
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
    )


@pytest.fixture
def retrieval_service(embedder, vector_store, lexical_index):
    """A retrieval service with every stage available."""
    return RetrievalService(
        embedder=embedder,
        vector_store=vector_store,
        lexical_index=lexical_index,
        reranker=KeywordReranker(),
    )


@pytest.fixture
def ingested(corpus_dir, ingest_service):
    """Ingest the sample corpus and return the reports."""
    return [ingest_service.ingest_file(path) for path in sorted(corpus_dir.glob("*.md"))]
