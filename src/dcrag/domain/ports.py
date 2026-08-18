"""Ports: the interfaces the application layer is allowed to depend on.

Each port is a ``typing.Protocol`` — structural typing, the Python counterpart of a C#
interface, with one important difference: an adapter does **not** inherit from the port.
``QdrantStore`` never imports ``VectorStore``; it simply has methods with the right shapes,
and mypy verifies the match where the adapter is wired in. Dependencies therefore point
inwards only, which is the whole point of the architecture.

Two consequences worth internalising:

* Every use case can be tested with in-memory fakes, with no GPU, no Docker and no network
  (see ``dcrag.infrastructure.fakes``). The test suite runs in seconds.
* Swapping Qdrant for Azure AI Search, or Ollama for Azure OpenAI, is one new adapter plus
  one line in the composition root. No application code changes.

All ports are synchronous. Blocking work is pushed to a worker thread at the HTTP edge
rather than colouring the whole codebase with ``async``; see
docs/adr/0004-synchronous-ports.md.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from dcrag.domain.entities import (
    Chunk,
    Document,
    GenerationResult,
    RetrievedChunk,
    Vector,
)


@runtime_checkable
class DocumentLoader(Protocol):
    """Turns a source file into a ``Document`` with headings preserved.

    Implemented by the PyMuPDF and Docling adapters. Keeping this behind a port is what
    makes "which PDF parser handles these tables better?" an experiment rather than a
    rewrite.
    """

    def supports(self, path: Path) -> bool:
        """Report whether this loader can handle the given file.

        Args:
            path: Candidate source file.

        Returns:
            True when the loader recognises the file type.
        """
        ...

    def load(self, path: Path) -> Document:
        """Load a file and convert it to Markdown.

        Args:
            path: Source file to load.

        Returns:
            The loaded document.

        Raises:
            DcragError: If the file cannot be parsed.
        """
        ...


@runtime_checkable
class Chunker(Protocol):
    """Splits a document into retrievable chunks carrying provenance."""

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into chunks.

        Implementations must attach a ``SourceRef`` to every chunk: a chunk without
        provenance cannot be cited, and an answer without citations cannot be verified.

        Args:
            document: The document to split.

        Returns:
            Chunks in document order, ``ordinal`` starting at 0.
        """
        ...


@runtime_checkable
class Embedder(Protocol):
    """Maps text to vectors.

    Documents and queries get separate methods on purpose. Modern retrieval models are
    asymmetric: they expect an instruction prefix on the query side and none on the
    document side, and getting that backwards quietly costs recall.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model, recorded in every evaluation run."""
        ...

    @property
    def dimension(self) -> int:
        """Dimensionality of the vectors this embedder produces."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed chunk texts for indexing.

        Args:
            texts: Chunk texts, batched by the caller.

        Returns:
            One vector per input text, in the same order.
        """
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for searching.

        Args:
            text: The question.

        Returns:
            The query vector.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Stores chunk vectors and answers nearest-neighbour queries.

    Implemented by the Qdrant, FAISS and in-memory adapters.
    """

    def ensure_collection(self, *, dimension: int) -> None:
        """Create the collection if it does not exist yet.

        Args:
            dimension: Vector dimensionality the collection must accept.

        Raises:
            ConfigurationError: If a collection exists with a different dimensionality.
        """
        ...

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> int:
        """Insert or replace chunks and their vectors.

        Upsert rather than insert so re-ingesting a changed document is idempotent.

        Args:
            chunks: Chunks to store.
            vectors: Vectors aligned positionally with ``chunks``.

        Returns:
            Number of chunks written.

        Raises:
            DomainValidationError: If the two sequences have different lengths.
        """
        ...

    def search(
        self,
        vector: Vector,
        *,
        top_k: int,
        doc_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks to a query vector.

        Args:
            vector: The query vector.
            top_k: Maximum number of results.
            doc_ids: Optional restriction to specific documents. The agent uses this to
                answer questions about one manual rather than the whole corpus.

        Returns:
            Results ordered best first, ``retriever`` set to ``dense``.
        """
        ...

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk belonging to a document.

        Args:
            doc_id: Document to remove.

        Returns:
            Number of chunks deleted.
        """
        ...

    def count(self) -> int:
        """Return the number of stored chunks.

        Returns:
            Total chunk count.
        """
        ...


@runtime_checkable
class LexicalIndex(Protocol):
    """Scores chunks by term overlap, typically BM25.

    Technical documentation is full of exact identifiers, model numbers and acronyms, which
    is where lexical search rescues dense search. Whether that holds on this corpus is an
    empirical question the evaluation harness answers.
    """

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks to the lexical index.

        Args:
            chunks: Chunks to index.

        Returns:
            Number of chunks indexed.
        """
        ...

    def search(
        self,
        query: str,
        *,
        top_k: int,
        doc_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the best lexical matches for a query.

        Args:
            query: The raw question text.
            top_k: Maximum number of results.
            doc_ids: Optional restriction to specific documents.

        Returns:
            Results ordered best first, ``retriever`` set to ``bm25``.
        """
        ...

    def count(self) -> int:
        """Return the number of indexed chunks.

        Returns:
            Total chunk count.
        """
        ...


@runtime_checkable
class Reranker(Protocol):
    """Re-scores candidates with a cross-encoder.

    A bi-encoder embeds question and chunk separately, so it never sees them together. A
    cross-encoder reads both at once and is far more accurate — and far too slow to run
    over a whole corpus. Hence the standard shape: retrieve wide, rerank narrow.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model."""
        ...

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Re-score and truncate a candidate list.

        Args:
            query: The question.
            candidates: Candidates from earlier retrieval stages.
            top_k: How many to keep.

        Returns:
            The best ``top_k`` candidates, ``retriever`` set to ``rerank``.
        """
        ...


@runtime_checkable
class Generator(Protocol):
    """Produces a completion from a system and user prompt.

    Implemented by the Ollama, OpenAI-compatible, Azure OpenAI and fake adapters. The port
    is intentionally narrow: no chat history, no tools. Multi-turn behaviour belongs to the
    agent layer, and keeping it out of here means the RAG pipeline stays deterministic and
    easy to evaluate.
    """

    @property
    def model_id(self) -> str:
        """Identifier of the underlying model."""
        ...

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Generate a completion.

        Args:
            system: System prompt establishing the grounding contract.
            user: User prompt containing the question and the numbered context blocks.
            temperature: Sampling temperature. Defaults to 0 because evaluation runs must
                be as reproducible as a non-deterministic system allows.
            max_tokens: Optional cap on generated tokens.

        Returns:
            The completion plus token accounting and latency.

        Raises:
            GenerationError: If the backend fails.
        """
        ...
