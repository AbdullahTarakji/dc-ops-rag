"""In-memory test doubles for every port.

These are shipped in ``src`` rather than hidden in ``tests`` for two reasons. The API test
suite wires the real FastAPI app against them, and the continuous-integration smoke
evaluation runs the whole pipeline with them — no GPU, no Docker, no network, a few
seconds. A pipeline that only works when a 7B model is loaded cannot be regression-tested
on every commit; this one can.

They are honest fakes, not mocks: the vector store really does cosine search, the lexical
index really does score term overlap. Tests can therefore assert on ranking behaviour, not
merely on "was this method called".
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Collection, Sequence
from pathlib import Path

from dcrag.domain.entities import (
    Chunk,
    Document,
    GenerationResult,
    RetrievedChunk,
    SourceRef,
    Vector,
)
from dcrag.domain.errors import ConfigurationError, DomainValidationError

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split text into alphanumeric tokens.

    Deliberately naive: the real pipeline uses a model tokenizer. This exists so the fakes
    behave predictably and so tests read clearly.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercased alphanumeric tokens.
    """
    return _TOKEN_RE.findall(text.lower())


class HashEmbedder:
    """A deterministic bag-of-words embedder.

    Each token is hashed to a dimension and accumulated, then the vector is L2-normalised.
    Texts sharing vocabulary end up with a high cosine similarity, which is enough for tests
    to assert that the *right* chunk ranks first, while being instant and dependency-free.

    It is not semantic: "chiller" and "koelmachine" are unrelated here. Cross-lingual
    behaviour is a property of the real bge-m3 adapter and is measured in the evaluation
    harness, never assumed from a fake.
    """

    def __init__(self, *, dimension: int = 64, model_id: str = "fake-hash-embedder") -> None:
        """Configure the fake embedder.

        Args:
            dimension: Vector dimensionality.
            model_id: Identifier reported to callers.

        Raises:
            ValueError: If ``dimension`` is not positive.
        """
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Identifier of this embedder."""
        return self._model_id

    @property
    def dimension(self) -> int:
        """Dimensionality of the produced vectors."""
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed several texts.

        Args:
            texts: Texts to embed.

        Returns:
            One vector per text, in order.
        """
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """Embed a query.

        Args:
            text: The query.

        Returns:
            The query vector.
        """
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """Hash tokens into a normalised vector.

        Args:
            text: Text to embed.

        Returns:
            An L2-normalised vector; the zero vector when the text has no tokens.
        """
        vector = [0.0] * self._dimension
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimension] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class InMemoryVectorStore:
    """A dictionary-backed vector store doing exact cosine search.

    Exact rather than approximate: with a few hundred test chunks brute force is
    instantaneous, and tests should not have to reason about recall loss from an ANN index.
    """

    def __init__(self) -> None:
        """Create an empty store."""
        self._vectors: dict[str, list[float]] = {}
        self._chunks: dict[str, Chunk] = {}
        self._dimension: int | None = None

    def ensure_collection(self, *, dimension: int) -> None:
        """Fix the dimensionality of this store.

        Args:
            dimension: Expected vector dimensionality.

        Raises:
            ConfigurationError: If the store already holds vectors of another size — the
                failure mode you hit in production by changing embedding model without
                re-ingesting.
        """
        if self._dimension is not None and self._dimension != dimension:
            raise ConfigurationError(
                f"store holds {self._dimension}-dimensional vectors, got {dimension}; "
                "re-ingest the corpus after changing embedding model"
            )
        self._dimension = dimension

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[Vector]) -> int:
        """Insert or replace chunks and their vectors.

        Args:
            chunks: Chunks to store.
            vectors: Vectors aligned with ``chunks``.

        Returns:
            Number of chunks written.

        Raises:
            DomainValidationError: If the sequences differ in length or a vector has the
                wrong dimensionality.
        """
        if len(chunks) != len(vectors):
            raise DomainValidationError(f"got {len(chunks)} chunks but {len(vectors)} vectors")
        for chunk, vector in zip(chunks, vectors, strict=True):
            if self._dimension is not None and len(vector) != self._dimension:
                raise DomainValidationError(
                    f"chunk {chunk.chunk_id} has a {len(vector)}-dimensional vector, "
                    f"expected {self._dimension}"
                )
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = list(vector)
        return len(chunks)

    def search(
        self,
        vector: Vector,
        *,
        top_k: int,
        doc_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the nearest chunks by cosine similarity.

        Args:
            vector: Query vector.
            top_k: Maximum number of results.
            doc_ids: Optional document filter.

        Returns:
            Results ordered best first.
        """
        scored: list[tuple[float, str]] = []
        for chunk_id, stored in self._vectors.items():
            if doc_ids is not None and self._chunks[chunk_id].doc_id not in doc_ids:
                continue
            scored.append((_cosine(vector, stored), chunk_id))
        # Ties broken by chunk id so results are reproducible run to run.
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(chunk=self._chunks[chunk_id], score=score, retriever="dense", rank=rank)
            for rank, (score, chunk_id) in enumerate(scored[:top_k])
        ]

    def delete_document(self, doc_id: str) -> int:
        """Remove every chunk of a document.

        Args:
            doc_id: Document to remove.

        Returns:
            Number of chunks deleted.
        """
        doomed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            del self._vectors[chunk_id]
        return len(doomed)

    def count(self) -> int:
        """Return the number of stored chunks.

        Returns:
            Chunk count.
        """
        return len(self._chunks)


class InMemoryLexicalIndex:
    """A term-overlap index standing in for BM25.

    Scoring is inverse-document-frequency weighted term overlap: enough to prefer chunks
    containing rare query terms, which is the behaviour tests care about, without
    reimplementing BM25 twice in one repository. The real adapter arrives with rank_bm25.
    """

    def __init__(self) -> None:
        """Create an empty index."""
        self._chunks: dict[str, Chunk] = {}
        self._tokens: dict[str, set[str]] = {}

    def index(self, chunks: Sequence[Chunk]) -> int:
        """Add chunks to the index.

        Args:
            chunks: Chunks to index.

        Returns:
            Number of chunks indexed.
        """
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk
            self._tokens[chunk.chunk_id] = set(tokenize(chunk.text))
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        doc_ids: Collection[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the best lexical matches.

        Args:
            query: Raw question text.
            top_k: Maximum number of results.
            doc_ids: Optional document filter.

        Returns:
            Results ordered best first, chunks with no overlap omitted.
        """
        query_tokens = set(tokenize(query))
        total = len(self._tokens) or 1
        scored: list[tuple[float, str]] = []
        for chunk_id, tokens in self._tokens.items():
            if doc_ids is not None and self._chunks[chunk_id].doc_id not in doc_ids:
                continue
            score = 0.0
            for token in query_tokens & tokens:
                containing = sum(1 for other in self._tokens.values() if token in other)
                score += math.log(1 + total / containing)
            if score > 0:
                scored.append((score, chunk_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            RetrievedChunk(chunk=self._chunks[chunk_id], score=score, retriever="bm25", rank=rank)
            for rank, (score, chunk_id) in enumerate(scored[:top_k])
        ]

    def count(self) -> int:
        """Return the number of indexed chunks.

        Returns:
            Chunk count.
        """
        return len(self._chunks)


class KeywordReranker:
    """A deterministic stand-in for the cross-encoder.

    Scores a candidate by the fraction of query tokens it contains. Crude, but it reorders
    results in a way a test can predict, which is what a fake reranker is for.
    """

    def __init__(self, *, model_id: str = "fake-keyword-reranker") -> None:
        """Configure the fake reranker.

        Args:
            model_id: Identifier reported to callers.
        """
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        """Identifier of this reranker."""
        return self._model_id

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        *,
        top_k: int,
    ) -> list[RetrievedChunk]:
        """Re-score candidates by query-token coverage.

        Args:
            query: The question.
            candidates: Candidates to reorder.
            top_k: How many to keep.

        Returns:
            The best ``top_k`` candidates.
        """
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return list(candidates[:top_k])
        scored = [
            (len(query_tokens & set(tokenize(candidate.chunk.text))) / len(query_tokens), candidate)
            for candidate in candidates
        ]
        scored.sort(key=lambda item: (-item[0], item[1].chunk.chunk_id))
        return [
            candidate.with_score(score=score, retriever="rerank", rank=rank)
            for rank, (score, candidate) in enumerate(scored[:top_k])
        ]


class ScriptedGenerator:
    """A generator that replays canned completions and records what it was asked.

    Tests assert on the recorded prompts — for example that retrieved text was fenced as
    data rather than pasted in as instructions, which is the prompt-injection defence.
    """

    def __init__(
        self,
        responses: Sequence[str] | None = None,
        *,
        model_id: str = "fake-scripted-generator",
    ) -> None:
        """Configure the fake generator.

        Args:
            responses: Completions returned in order. The last one repeats once exhausted.
                Defaults to a single grounded answer citing block 1.
            model_id: Identifier reported to callers.
        """
        self._responses = list(responses) if responses else ["Grounded answer. [1]"]
        self._model_id = model_id
        self.calls: list[tuple[str, str]] = []
        """Every ``(system, user)`` prompt pair this generator received."""

    @property
    def model_id(self) -> str:
        """Identifier of this generator."""
        return self._model_id

    def generate(
        self,
        *,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> GenerationResult:
        """Return the next scripted completion.

        Args:
            system: System prompt, recorded for assertions.
            user: User prompt, recorded for assertions.
            temperature: Ignored; accepted to satisfy the port.
            max_tokens: Ignored; accepted to satisfy the port.

        Returns:
            The scripted completion with plausible token accounting.
        """
        del temperature, max_tokens
        self.calls.append((system, user))
        index = min(len(self.calls) - 1, len(self._responses) - 1)
        text = self._responses[index]
        return GenerationResult(
            text=text,
            model=self._model_id,
            prompt_tokens=len(tokenize(system)) + len(tokenize(user)),
            completion_tokens=len(tokenize(text)),
            latency_ms=0.0,
        )


class MarkdownLoader:
    """Loads a Markdown file straight from disk.

    Real ingestion starts from PDFs; this loader lets tests and fixtures start from text,
    keeping the fixtures readable and reviewable in a pull request.
    """

    def supports(self, path: Path) -> bool:
        """Report whether the file is Markdown.

        Args:
            path: Candidate file.

        Returns:
            True for ``.md`` and ``.markdown`` files.
        """
        return path.suffix.lower() in {".md", ".markdown"}

    def load(self, path: Path) -> Document:
        """Read a Markdown file into a document.

        The title is the first level-1 heading when present, otherwise the file stem.

        Args:
            path: File to read.

        Returns:
            The loaded document.
        """
        text = path.read_text(encoding="utf-8")
        title = path.stem
        for line in text.splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                break
        return Document(
            doc_id=path.stem,
            title=title,
            source_uri=path.as_uri(),
            markdown=text,
            checksum=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        )


class ParagraphChunker:
    """Splits Markdown on blank lines, tracking the heading trail.

    A placeholder for the heading-aware, token-budgeted chunker built in the ingestion
    iteration. It already produces real provenance, so services and tests written against
    it keep working when the real chunker replaces it.
    """

    def __init__(self, *, max_words: int = 120) -> None:
        """Configure the chunker.

        Args:
            max_words: Word budget per chunk; longer paragraphs are split.

        Raises:
            ValueError: If ``max_words`` is not positive.
        """
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        self._max_words = max_words

    def split(self, document: Document) -> list[Chunk]:
        """Split a document into paragraph-sized chunks.

        Args:
            document: Document to split.

        Returns:
            Chunks in document order.
        """
        chunks: list[Chunk] = []
        headings: list[str] = []
        for block in document.markdown.split("\n\n"):
            text = block.strip()
            if not text:
                continue
            if text.startswith("#"):
                level = len(text) - len(text.lstrip("#"))
                headings = [*headings[: level - 1], text.lstrip("# ").strip()]
                continue
            for piece in self._wrap(text):
                ordinal = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=f"{document.doc_id}#{ordinal:04d}",
                        doc_id=document.doc_id,
                        text=piece,
                        ordinal=ordinal,
                        token_count=len(piece.split()),
                        source=SourceRef(
                            doc_id=document.doc_id,
                            title=document.title,
                            page=None,
                            section_path=tuple(headings),
                        ),
                    )
                )
        return chunks

    def _wrap(self, text: str) -> list[str]:
        """Split a paragraph into pieces within the word budget.

        Args:
            text: Paragraph text.

        Returns:
            One or more pieces.
        """
        words = text.split()
        if len(words) <= self._max_words:
            return [text]
        return [
            " ".join(words[start : start + self._max_words])
            for start in range(0, len(words), self._max_words)
        ]


def _cosine(left: Vector, right: Vector) -> float:
    """Return the cosine similarity of two vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        Similarity in ``[-1, 1]``, or 0.0 when either vector is all zeros.
    """
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
