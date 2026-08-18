"""Every fake must satisfy the port it stands in for.

Two checks run here, and they are not redundant:

* the ``isinstance`` assertions are *runtime* structural checks — the fake has the members;
* the annotated assignments are *static* checks — mypy verifies the full signatures, which
  ``isinstance`` against a Protocol never does.

Note that no fake inherits from a port. That is dependency inversion working: the adapter
does not know the abstraction exists, yet the type checker still holds it to the contract.
"""

from __future__ import annotations

from dcrag.domain import (
    Chunker,
    DocumentLoader,
    Embedder,
    Generator,
    LexicalIndex,
    Reranker,
    VectorStore,
)
from dcrag.infrastructure.fakes import (
    HashEmbedder,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    KeywordReranker,
    MarkdownLoader,
    ParagraphChunker,
    ScriptedGenerator,
)


def test_fakes_satisfy_their_ports_at_runtime():
    assert isinstance(HashEmbedder(), Embedder)
    assert isinstance(InMemoryVectorStore(), VectorStore)
    assert isinstance(InMemoryLexicalIndex(), LexicalIndex)
    assert isinstance(KeywordReranker(), Reranker)
    assert isinstance(ScriptedGenerator(), Generator)
    assert isinstance(MarkdownLoader(), DocumentLoader)
    assert isinstance(ParagraphChunker(), Chunker)


def test_fakes_satisfy_their_ports_statically():
    embedder: Embedder = HashEmbedder()
    store: VectorStore = InMemoryVectorStore()
    lexical: LexicalIndex = InMemoryLexicalIndex()
    reranker: Reranker = KeywordReranker()
    generator: Generator = ScriptedGenerator()
    loader: DocumentLoader = MarkdownLoader()
    chunker: Chunker = ParagraphChunker()

    assert embedder.dimension > 0
    assert store.count() == 0
    assert lexical.count() == 0
    assert reranker.model_id
    assert generator.model_id
    assert loader.supports.__name__ == "supports"
    assert chunker.split.__name__ == "split"


def test_a_class_missing_a_method_does_not_satisfy_the_port():
    class NotAnEmbedder:
        """Has a dimension but no embedding methods."""

        dimension = 8

    assert not isinstance(NotAnEmbedder(), Embedder)
