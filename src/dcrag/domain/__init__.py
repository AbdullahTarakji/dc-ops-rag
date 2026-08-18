"""Domain layer: entities, value objects, ports, and errors.

This package deliberately imports nothing outside the standard library. If a third-party
import ever becomes necessary here, the abstraction is wrong.
"""

from dcrag.domain.entities import (
    Answer,
    Chunk,
    Citation,
    Document,
    GenerationResult,
    RetrievalMode,
    RetrievedChunk,
    SourceRef,
    Vector,
    content_hash,
)
from dcrag.domain.errors import (
    ConfigurationError,
    DcragError,
    DomainValidationError,
    GenerationError,
    RetrievalError,
    UngroundedCitationError,
)
from dcrag.domain.ports import (
    Chunker,
    DocumentLoader,
    Embedder,
    Generator,
    LexicalIndex,
    Reranker,
    VectorStore,
)

__all__ = [
    "Answer",
    "Chunk",
    "Chunker",
    "Citation",
    "ConfigurationError",
    "DcragError",
    "Document",
    "DocumentLoader",
    "DomainValidationError",
    "Embedder",
    "GenerationError",
    "GenerationResult",
    "Generator",
    "LexicalIndex",
    "Reranker",
    "RetrievalError",
    "RetrievalMode",
    "RetrievedChunk",
    "SourceRef",
    "UngroundedCitationError",
    "Vector",
    "VectorStore",
    "content_hash",
]
