"""The fakes themselves.

Test doubles that silently misbehave are worse than no doubles at all: every downstream test
inherits the lie. These are cheap tests that keep the doubles honest.
"""

from __future__ import annotations

import pytest

from dcrag.domain import Chunk, DomainValidationError, SourceRef
from dcrag.infrastructure.fakes import (
    HashEmbedder,
    InMemoryLexicalIndex,
    InMemoryVectorStore,
    KeywordReranker,
    MarkdownLoader,
    ParagraphChunker,
    ScriptedGenerator,
    tokenize,
)


def make_chunk(chunk_id: str, text: str, *, doc_id: str = "doc", ordinal: int = 0) -> Chunk:
    """Build a chunk for store and index tests."""
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        ordinal=ordinal,
        source=SourceRef(doc_id=doc_id, title="Doc"),
    )


class TestHashEmbedder:
    def test_is_deterministic_and_normalised(self):
        embedder = HashEmbedder(dimension=32)
        first = embedder.embed_query("condenser fouling")
        second = embedder.embed_query("condenser fouling")

        assert first == second
        assert sum(value * value for value in first) == pytest.approx(1.0)

    def test_shared_vocabulary_scores_higher_than_unrelated_text(self):
        embedder = HashEmbedder()
        store = InMemoryVectorStore()
        store.ensure_collection(dimension=embedder.dimension)
        chunks = [
            make_chunk("doc#0000", "condenser fouling raises approach temperature", ordinal=0),
            make_chunk("doc#0001", "refrigerant leak lowers evaporator pressure", ordinal=1),
        ]
        store.upsert(chunks, embedder.embed_documents([c.text for c in chunks]))

        results = store.search(embedder.embed_query("condenser fouling"), top_k=2)

        assert results[0].chunk.chunk_id == "doc#0000"
        assert results[0].score > results[1].score

    def test_text_without_tokens_yields_the_zero_vector(self):
        assert HashEmbedder(dimension=4).embed_query("!!! ???") == [0.0, 0.0, 0.0, 0.0]

    def test_rejects_a_non_positive_dimension(self):
        with pytest.raises(ValueError, match="dimension"):
            HashEmbedder(dimension=0)


class TestInMemoryVectorStore:
    def test_rejects_mismatched_chunk_and_vector_counts(self):
        store = InMemoryVectorStore()
        with pytest.raises(DomainValidationError, match="2 chunks but 1 vectors"):
            store.upsert([make_chunk("a", "a"), make_chunk("b", "b", ordinal=1)], [[1.0]])

    def test_rejects_a_vector_of_the_wrong_size(self):
        store = InMemoryVectorStore()
        store.ensure_collection(dimension=3)
        with pytest.raises(DomainValidationError, match="expected 3"):
            store.upsert([make_chunk("a", "a")], [[1.0, 0.0]])

    def test_delete_document_reports_what_it_removed(self):
        store = InMemoryVectorStore()
        store.upsert(
            [make_chunk("a#0", "one", doc_id="a"), make_chunk("b#0", "two", doc_id="b")],
            [[1.0], [1.0]],
        )

        assert store.delete_document("a") == 1
        assert store.count() == 1
        assert store.delete_document("missing") == 0

    def test_zero_vectors_score_zero_rather_than_dividing_by_zero(self):
        store = InMemoryVectorStore()
        store.upsert([make_chunk("a", "text")], [[0.0, 0.0]])

        assert store.search([1.0, 0.0], top_k=1)[0].score == 0.0


class TestInMemoryLexicalIndex:
    def test_ignores_chunks_with_no_overlap(self):
        index = InMemoryLexicalIndex()
        index.index([make_chunk("a", "condenser fouling"), make_chunk("b", "oil sump", ordinal=1)])

        results = index.search("condenser", top_k=5)

        assert [result.chunk.chunk_id for result in results] == ["a"]

    def test_rarer_terms_contribute_more(self):
        index = InMemoryLexicalIndex()
        index.index(
            [
                make_chunk("a", "the chiller uses refrigerant", ordinal=0),
                make_chunk("b", "the chiller uses water", ordinal=1),
                make_chunk("c", "the pump uses water", ordinal=2),
            ]
        )

        results = index.search("refrigerant chiller", top_k=3)

        assert results[0].chunk.chunk_id == "a"


class TestKeywordReranker:
    def test_promotes_the_chunk_covering_more_query_terms(self):
        index = InMemoryLexicalIndex()
        chunks = [
            make_chunk("a", "approach temperature", ordinal=0),
            make_chunk("b", "condenser fouling raises approach temperature", ordinal=1),
        ]
        index.index(chunks)
        candidates = index.search("condenser fouling approach temperature", top_k=2)

        reranked = KeywordReranker().rerank("condenser fouling approach", candidates, top_k=2)

        assert reranked[0].chunk.chunk_id == "b"
        assert reranked[0].retriever == "rerank"

    def test_a_query_without_tokens_leaves_the_order_alone(self):
        index = InMemoryLexicalIndex()
        index.index([make_chunk("a", "text one"), make_chunk("b", "text two", ordinal=1)])
        candidates = index.search("text", top_k=2)

        assert KeywordReranker().rerank("???", candidates, top_k=2) == list(candidates[:2])


class TestScriptedGenerator:
    def test_replays_responses_then_repeats_the_last(self):
        generator = ScriptedGenerator(["first", "second"])

        assert generator.generate(system="s", user="u").text == "first"
        assert generator.generate(system="s", user="u").text == "second"
        assert generator.generate(system="s", user="u").text == "second"

    def test_records_prompts_for_assertions(self):
        generator = ScriptedGenerator()
        generator.generate(system="answer only from context", user="Q: why?")

        assert generator.calls == [("answer only from context", "Q: why?")]

    def test_counts_tokens_so_cost_accounting_has_something_to_read(self):
        result = ScriptedGenerator(["a grounded answer"]).generate(system="sys", user="question")

        assert result.prompt_tokens == 2
        assert result.completion_tokens == 3
        assert result.total_tokens == 5


class TestMarkdownLoader:
    def test_supports_only_markdown(self, tmp_path):
        loader = MarkdownLoader()
        assert loader.supports(tmp_path / "a.md")
        assert loader.supports(tmp_path / "a.MARKDOWN")
        assert not loader.supports(tmp_path / "a.pdf")

    def test_takes_the_title_from_the_first_heading(self, tmp_path):
        path = tmp_path / "notes.md"
        path.write_text("# Real Title\n\nbody", encoding="utf-8")

        document = MarkdownLoader().load(path)

        assert document.title == "Real Title"
        assert document.doc_id == "notes"
        assert len(document.checksum) == 64

    def test_falls_back_to_the_filename(self, tmp_path):
        path = tmp_path / "no-heading.md"
        path.write_text("body only", encoding="utf-8")

        assert MarkdownLoader().load(path).title == "no-heading"


class TestParagraphChunker:
    def test_tracks_the_heading_trail(self, document):
        chunks = ParagraphChunker().split(document)

        assert chunks[0].source.section_path[0].startswith("EU Code of Conduct")
        assert chunks[0].source.section_path[-1] == "Air flow management"
        assert [chunk.ordinal for chunk in chunks] == list(range(len(chunks)))

    def test_splits_paragraphs_over_the_word_budget(self, document):
        chunks = ParagraphChunker(max_words=5).split(document)

        assert all(len(chunk.text.split()) <= 5 for chunk in chunks)

    def test_rejects_a_non_positive_budget(self):
        with pytest.raises(ValueError, match="max_words"):
            ParagraphChunker(max_words=0)


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("TCA rose 2.5 degF!") == ["tca", "rose", "2", "5", "degf"]
