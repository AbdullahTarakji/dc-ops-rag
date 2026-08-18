"""Ingestion use case: source file to searchable chunks.

The pipeline is load, split, embed, store, index. Every step sits behind a port, so this
module is where the *policy* lives (batching, what counts as a successful ingestion, what
gets reported) while the *mechanism* lives in adapters.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from dcrag.domain.entities import Chunk, Document
from dcrag.domain.ports import Chunker, DocumentLoader, Embedder, LexicalIndex, VectorStore

DEFAULT_BATCH_SIZE = 32
"""Chunks embedded per forward pass. Sized for 8 GB of VRAM with bge-m3 loaded."""


@dataclass(frozen=True, slots=True)
class IngestReport:
    """What one ingestion run did, in numbers worth putting in a log line."""

    doc_id: str
    """Document that was ingested."""

    chunks: int
    """Chunks written to the vector store."""

    tokens: int
    """Total tokens across those chunks, when the chunker counted them."""

    duration_ms: float
    """Wall-clock duration of the whole run."""

    embedder: str
    """Embedding model used, so a stale index can be spotted after a model change."""

    @property
    def chunks_per_second(self) -> float:
        """Return ingestion throughput.

        Returns:
            Chunks per second, or 0.0 when the run was too fast to measure.
        """
        if self.duration_ms <= 0:
            return 0.0
        return self.chunks / (self.duration_ms / 1000)


class IngestService:
    """Loads documents and makes them searchable."""

    def __init__(
        self,
        *,
        loader: DocumentLoader,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        lexical_index: LexicalIndex | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Wire the service to its collaborators.

        Args:
            loader: Converts source files to Markdown documents.
            chunker: Splits documents into chunks with provenance.
            embedder: Produces vectors for the chunks.
            vector_store: Persists chunks and vectors.
            lexical_index: Optional BM25 index kept in step with the vector store.
            batch_size: Chunks embedded per batch.

        Raises:
            ValueError: If ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self._loader = loader
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store
        self._lexical_index = lexical_index
        self._batch_size = batch_size

    def ingest_file(self, path: Path) -> IngestReport:
        """Load a file and ingest it.

        Args:
            path: Source file.

        Returns:
            A report describing the run.

        Raises:
            ValueError: If the loader does not support this file type.
        """
        if not self._loader.supports(path):
            raise ValueError(f"no loader supports {path.name}")
        return self.ingest_document(self._loader.load(path))

    def ingest_document(self, document: Document) -> IngestReport:
        """Chunk, embed, store and index an already-loaded document.

        Re-ingesting the same document replaces its chunks rather than duplicating them:
        the vector store upserts by chunk id, and stale chunks are deleted first so a
        document that got *shorter* does not leave orphans behind.

        Args:
            document: The document to ingest.

        Returns:
            A report describing the run.
        """
        started = time.perf_counter()

        chunks = self._chunker.split(document)
        self._vector_store.ensure_collection(dimension=self._embedder.dimension)
        self._vector_store.delete_document(document.doc_id)

        written = 0
        for batch in _batched(chunks, self._batch_size):
            vectors = self._embedder.embed_documents([chunk.text for chunk in batch])
            written += self._vector_store.upsert(batch, vectors)

        if self._lexical_index is not None:
            self._lexical_index.index(chunks)

        return IngestReport(
            doc_id=document.doc_id,
            chunks=written,
            tokens=sum(chunk.token_count for chunk in chunks),
            duration_ms=(time.perf_counter() - started) * 1000,
            embedder=self._embedder.model_id,
        )


def _batched(chunks: list[Chunk], size: int) -> list[list[Chunk]]:
    """Split a list of chunks into fixed-size batches.

    ``itertools.batched`` would do this in 3.12; the project targets 3.11 for library
    compatibility, so the four-line version stays.

    Args:
        chunks: Chunks to batch.
        size: Maximum batch size.

    Returns:
        A list of batches, the last possibly shorter.
    """
    return [chunks[start : start + size] for start in range(0, len(chunks), size)]
