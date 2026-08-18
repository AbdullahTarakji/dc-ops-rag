"""Domain entities and value objects.

These are plain frozen dataclasses, not Pydantic models, and that is deliberate. Pydantic
belongs at the edges of the system where untrusted input arrives (the HTTP layer). The
domain describes what a chunk *is*; it should not depend on a validation library or on the
shape of any JSON payload. See docs/adr/0001-clean-architecture.md.

Everything here is immutable. A retrieved chunk that a reranker could mutate would make the
audit trail from question to citation untrustworthy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TypeAlias

from dcrag.domain.errors import DomainValidationError

Vector: TypeAlias = Sequence[float]
"""An embedding. A sequence rather than a list so adapters may hand back numpy views."""


class RetrievalMode(StrEnum):
    """How candidate chunks are found for a question.

    The modes exist to be compared against each other in the evaluation harness, which is
    why this enum lives in the domain rather than being a loose string in the API layer.
    """

    DENSE = "dense"
    """Embedding similarity only."""

    BM25 = "bm25"
    """Lexical scoring only. Strong on exact identifiers, part numbers and acronyms."""

    HYBRID = "hybrid"
    """Dense and BM25 result lists merged with Reciprocal Rank Fusion."""


def _require(condition: bool, message: str) -> None:
    """Raise ``DomainValidationError`` when an invariant does not hold.

    Args:
        condition: The invariant that must be true.
        message: What was violated, phrased for a developer reading a stack trace.

    Raises:
        DomainValidationError: If ``condition`` is false.
    """
    if not condition:
        raise DomainValidationError(message)


def content_hash(text: str) -> str:
    """Return a stable short hash of chunk text.

    Used to key the embedding cache and to detect that a source document changed between
    ingestion runs. Truncated to 16 hex characters: collision risk is negligible at corpus
    scale and short hashes keep logs and payloads readable.

    Args:
        text: The text to hash.

    Returns:
        The first 16 hex characters of the SHA-256 digest.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Where a piece of text came from, precisely enough to cite it.

    This is what turns a plausible sentence into a verifiable one. Without ``page`` and
    ``section_path`` a reader cannot check the claim, and an unverifiable answer is worth
    little in a domain where being wrong costs money.
    """

    doc_id: str
    """Stable identifier of the source document, e.g. ``eu-coc-2025``."""

    title: str
    """Human-readable document title, shown in the UI next to the citation."""

    page: int | None = None
    """1-based page number in the original PDF, when the loader could determine one."""

    section_path: tuple[str, ...] = ()
    """Heading trail, outermost first, e.g. ``("Cooling", "Air flow management")``."""

    def __post_init__(self) -> None:
        """Validate identifiers and page numbering."""
        _require(bool(self.doc_id.strip()), "SourceRef.doc_id must not be empty")
        _require(bool(self.title.strip()), "SourceRef.title must not be empty")
        _require(self.page is None or self.page >= 1, "SourceRef.page is 1-based")

    def describe(self) -> str:
        """Return a compact human-readable reference.

        Returns:
            A single line such as ``EU CoC 2025 - p.42 - Cooling > Air flow``, suitable for
            a prompt block header or a UI label.
        """
        parts = [self.title]
        if self.page is not None:
            parts.append(f"p.{self.page}")
        if self.section_path:
            parts.append(" > ".join(self.section_path))
        return " - ".join(parts)


@dataclass(frozen=True, slots=True)
class Document:
    """A source document after loading and conversion to Markdown."""

    doc_id: str
    """Stable identifier, derived from the filename in the corpus manifest."""

    title: str
    """Document title."""

    source_uri: str
    """Where the document came from: a URL for downloaded corpora, else a file path."""

    markdown: str
    """Full document text with headings preserved."""

    page_count: int = 0
    """Number of pages in the original file, 0 when not applicable."""

    checksum: str = ""
    """SHA-256 of the original bytes, recorded so ingestion runs are reproducible."""

    language: str | None = None
    """ISO 639-1 code when known. The corpus is mostly English, queries are often Dutch."""

    licence: str | None = None
    """Licence of the source, copied from the corpus manifest. Never guessed."""

    def __post_init__(self) -> None:
        """Validate identity and page count."""
        _require(bool(self.doc_id.strip()), "Document.doc_id must not be empty")
        _require(bool(self.title.strip()), "Document.title must not be empty")
        _require(self.page_count >= 0, "Document.page_count must not be negative")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit of text with the provenance needed to cite it.

    Chunk size is a hyperparameter, not a constant: it is set in the experiment config and
    appears as a column in the ablation table.
    """

    chunk_id: str
    """Unique id, conventionally ``{doc_id}#{ordinal:04d}``."""

    doc_id: str
    """Owning document."""

    text: str
    """The chunk text as it will be embedded and as it will be shown to the generator."""

    ordinal: int
    """0-based position of this chunk within its document, used for stable ordering."""

    source: SourceRef
    """Provenance for citation."""

    token_count: int = 0
    """Token count under the tokenizer used at ingestion time, 0 when not computed."""

    def __post_init__(self) -> None:
        """Validate identity, ordering and non-emptiness."""
        _require(bool(self.chunk_id.strip()), "Chunk.chunk_id must not be empty")
        _require(bool(self.text.strip()), "Chunk.text must not be empty")
        _require(self.ordinal >= 0, "Chunk.ordinal must not be negative")
        _require(self.token_count >= 0, "Chunk.token_count must not be negative")
        _require(
            self.doc_id == self.source.doc_id,
            f"Chunk.doc_id {self.doc_id!r} disagrees with SourceRef {self.source.doc_id!r}",
        )

    @property
    def fingerprint(self) -> str:
        """Return the content hash of this chunk text.

        Returns:
            A short stable hash, used as the embedding-cache key.
        """
        return content_hash(self.text)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk together with why and how strongly it was retrieved.

    ``score`` is only comparable within a single ``retriever``: a BM25 score and a cosine
    similarity live on different scales. That is exactly why fusion is done by rank
    (Reciprocal Rank Fusion) rather than by blending raw scores.
    """

    chunk: Chunk
    """The retrieved chunk."""

    score: float
    """Backend-specific relevance score."""

    retriever: str
    """Which stage produced this result: ``dense``, ``bm25``, ``rrf`` or ``rerank``."""

    rank: int
    """0-based position in the result list this chunk came from."""

    def __post_init__(self) -> None:
        """Validate ranking metadata."""
        _require(self.rank >= 0, "RetrievedChunk.rank must not be negative")
        _require(bool(self.retriever.strip()), "RetrievedChunk.retriever must not be empty")

    def with_score(self, *, score: float, retriever: str, rank: int) -> RetrievedChunk:
        """Return a copy rescored by a later stage.

        Fusion and reranking do not mutate results, they produce new ones, so the original
        dense and lexical rankings stay available for evaluation and debugging.

        Args:
            score: The new score.
            retriever: The stage that produced it.
            rank: The new 0-based rank.

        Returns:
            A new ``RetrievedChunk`` wrapping the same chunk.
        """
        return RetrievedChunk(chunk=self.chunk, score=score, retriever=retriever, rank=rank)


@dataclass(frozen=True, slots=True)
class Citation:
    """A claim in an answer, tied to the context block that supports it."""

    block: int
    """1-based index of the numbered context block the generator cited."""

    chunk_id: str
    """The chunk that block contained."""

    source: SourceRef
    """Provenance, copied so a citation is self-contained in API responses and logs."""

    score: float = 0.0
    """Retrieval score of the cited chunk, useful when triaging weak answers."""

    def __post_init__(self) -> None:
        """Validate block numbering and identity."""
        _require(self.block >= 1, "Citation.block is 1-based")
        _require(bool(self.chunk_id.strip()), "Citation.chunk_id must not be empty")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Raw output of a generator, before it is parsed into an answer.

    Token counts and latency are carried here rather than logged ad hoc, because cost and
    p95 latency are reported metrics, not afterthoughts.
    """

    text: str
    """The completion text."""

    model: str
    """Model identifier, e.g. ``qwen2.5:7b-instruct``. Recorded in every evaluation run."""

    prompt_tokens: int = 0
    """Tokens consumed by the prompt, 0 when the backend does not report it."""

    completion_tokens: int = 0
    """Tokens produced, 0 when the backend does not report it."""

    latency_ms: float = 0.0
    """Wall-clock generation time in milliseconds."""

    def __post_init__(self) -> None:
        """Validate accounting fields."""
        _require(self.prompt_tokens >= 0, "GenerationResult.prompt_tokens must not be negative")
        _require(
            self.completion_tokens >= 0,
            "GenerationResult.completion_tokens must not be negative",
        )
        _require(self.latency_ms >= 0, "GenerationResult.latency_ms must not be negative")

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens.

        Returns:
            Total tokens attributable to this generation.
        """
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class Answer:
    """A grounded answer: text, the citations supporting it, and whether it abstained.

    ``abstained`` is a first-class field rather than a magic string inside ``text`` because
    correct abstention is a measured metric. In this domain, "the documentation does not
    cover this" is a good answer, and good answers must be countable.
    """

    text: str
    """The answer as shown to the user."""

    citations: tuple[Citation, ...] = ()
    """Supporting citations, empty when the system abstained."""

    abstained: bool = False
    """True when the system declined to answer because the context did not support one."""

    generation: GenerationResult | None = None
    """The raw generation, kept for evaluation, tracing and cost accounting."""

    retrieved: tuple[RetrievedChunk, ...] = field(default=(), repr=False)
    """The context the answer was produced from, kept for faithfulness scoring."""

    def __post_init__(self) -> None:
        """Validate the relationship between abstention and citations."""
        _require(bool(self.text.strip()), "Answer.text must not be empty")
        _require(
            not (self.abstained and self.citations),
            "An abstaining answer must not carry citations",
        )

    @property
    def cited_chunk_ids(self) -> frozenset[str]:
        """Return the set of chunk ids this answer cites.

        Returns:
            Chunk ids referenced by the citations.
        """
        return frozenset(citation.chunk_id for citation in self.citations)
