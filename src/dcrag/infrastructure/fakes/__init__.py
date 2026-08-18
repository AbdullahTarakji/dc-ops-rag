"""In-memory test doubles for every port.

Importable from application and API tests, and used by the continuous-integration smoke
evaluation so the whole pipeline is exercised on every commit without a GPU.
"""

from dcrag.infrastructure.fakes.inmemory import (
    HashEmbedder,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    KeywordReranker,
    MarkdownLoader,
    ParagraphChunker,
    ScriptedGenerator,
    tokenize,
)

__all__ = [
    "HashEmbedder",
    "InMemoryLexicalIndex",
    "InMemoryVectorStore",
    "KeywordReranker",
    "MarkdownLoader",
    "ParagraphChunker",
    "ScriptedGenerator",
    "tokenize",
]
