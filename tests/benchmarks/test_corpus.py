"""Tests for benchmark corpus data models.

Tests BenchmarkDocument, DocumentCorpus, and CorpusMetadata classes
following TASK-070 acceptance criteria.
"""

import json
import tempfile
from pathlib import Path

import pytest

from draagon_ai.testing.benchmarks import (
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
    DocumentCorpus,
    CorpusMetadata,
)


class TestBenchmarkDocument:
    """Tests for BenchmarkDocument dataclass."""

    def test_content_hash_deterministic(self) -> None:
        """Content hash is deterministic for same content."""
        content = "This is test content for hashing."
        doc1 = BenchmarkDocument(
            doc_id="doc_1",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/1.md",
            content=content,
        )
        doc2 = BenchmarkDocument(
            doc_id="doc_2",
            source=DocumentSource.ONLINE,  # Different source
            category=DocumentCategory.NARRATIVE,  # Different category
            domain="java",  # Different domain
            file_path="/path/2.md",  # Different path
            content=content,  # Same content
        )

        assert doc1.content_hash == doc2.content_hash
        assert len(doc1.content_hash) == 16  # First 16 chars of SHA256

    def test_content_hash_different_for_different_content(self) -> None:
        """Different content produces different hashes."""
        doc1 = BenchmarkDocument(
            doc_id="doc_1",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/1.md",
            content="Content A",
        )
        doc2 = BenchmarkDocument(
            doc_id="doc_2",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/2.md",
            content="Content B",
        )

        assert doc1.content_hash != doc2.content_hash

    def test_size_bytes_calculated(self) -> None:
        """Size in bytes is automatically calculated."""
        content = "Hello, World!"  # 13 ASCII chars = 13 bytes
        doc = BenchmarkDocument(
            doc_id="doc_1",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/test.md",
            content=content,
        )

        assert doc.size_bytes == 13

    def test_size_bytes_unicode(self) -> None:
        """Size handles unicode content correctly."""
        content = "Hello 🌍"  # 6 ASCII + space + 4-byte emoji = 11 bytes
        doc = BenchmarkDocument(
            doc_id="doc_1",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/test.md",
            content=content,
        )

        assert doc.size_bytes == len(content.encode())

    def test_to_dict_serialization(self, sample_document: BenchmarkDocument) -> None:
        """to_dict() includes all fields."""
        data = sample_document.to_dict()

        assert data["doc_id"] == "doc_001"
        assert data["source"] == "local"
        assert data["category"] == "technical"
        assert data["domain"] == "python"
        assert data["file_path"] == "/home/user/project/README.md"
        assert "sample Python project" in data["content"]
        assert data["chunk_ids"] == ["chunk_001", "chunk_002"]
        assert data["metadata"]["author"] == "test_user"
        assert data["is_distractor"] is False
        assert "python" in data["semantic_tags"]
        assert len(data["content_hash"]) == 16
        assert data["size_bytes"] > 0
        assert "created_at" in data

    def test_from_dict_deserialization(self, sample_document: BenchmarkDocument) -> None:
        """from_dict() restores all fields correctly."""
        data = sample_document.to_dict()
        restored = BenchmarkDocument.from_dict(data)

        assert restored.doc_id == sample_document.doc_id
        assert restored.source == sample_document.source
        assert restored.category == sample_document.category
        assert restored.domain == sample_document.domain
        assert restored.content == sample_document.content
        assert restored.content_hash == sample_document.content_hash
        assert restored.is_distractor == sample_document.is_distractor

    def test_roundtrip_json_serialization(self, sample_document: BenchmarkDocument) -> None:
        """Document survives JSON round-trip."""
        data = sample_document.to_dict()
        json_str = json.dumps(data)
        restored_data = json.loads(json_str)
        restored = BenchmarkDocument.from_dict(restored_data)

        assert restored.content == sample_document.content
        assert restored.content_hash == sample_document.content_hash

    def test_default_values(self) -> None:
        """Default values are applied correctly."""
        doc = BenchmarkDocument(
            doc_id="minimal",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="test",
            file_path="/path/test.md",
            content="Minimal content",
        )

        assert doc.chunk_ids == []
        assert doc.metadata == {}
        assert doc.is_distractor is False
        assert doc.semantic_tags == []


class TestCorpusMetadata:
    """Tests for CorpusMetadata calculations."""

    def test_calculate_empty_corpus(self) -> None:
        """Empty corpus doesn't crash."""
        metadata = CorpusMetadata.calculate([])

        assert metadata.total_documents == 0
        assert metadata.distractor_count == 0
        assert metadata.distractor_ratio == 0.0
        assert metadata.size_stats["min"] == 0

    def test_calculate_single_document(self, sample_document: BenchmarkDocument) -> None:
        """Single document calculates correctly."""
        metadata = CorpusMetadata.calculate([sample_document])

        assert metadata.total_documents == 1
        assert metadata.source_distribution["local"] == 1
        assert metadata.category_distribution["technical"] == 1
        assert metadata.domain_distribution["python"] == 1
        assert metadata.distractor_count == 0
        assert metadata.distractor_ratio == 0.0

    def test_calculate_diverse_documents(
        self,
        sample_document: BenchmarkDocument,
        sample_distractor: BenchmarkDocument,
        sample_legal_document: BenchmarkDocument,
    ) -> None:
        """Diverse documents tracked correctly."""
        docs = [sample_document, sample_distractor, sample_legal_document]
        metadata = CorpusMetadata.calculate(docs)

        assert metadata.total_documents == 3
        assert metadata.source_distribution["local"] == 1
        assert metadata.source_distribution["synthetic"] == 1
        assert metadata.source_distribution["online"] == 1
        assert metadata.category_distribution["technical"] == 1
        assert metadata.category_distribution["synthetic"] == 1
        assert metadata.category_distribution["legal"] == 1
        assert metadata.distractor_count == 1
        assert metadata.distractor_ratio == pytest.approx(1 / 3, rel=0.01)

    def test_size_stats_calculation(self) -> None:
        """Size statistics are correct."""
        docs = [
            BenchmarkDocument(
                doc_id=f"doc_{i}",
                source=DocumentSource.LOCAL,
                category=DocumentCategory.TECHNICAL,
                domain="test",
                file_path=f"/path/{i}.md",
                content="x" * size,
            )
            for i, size in enumerate([100, 200, 300, 400])
        ]

        metadata = CorpusMetadata.calculate(docs)

        assert metadata.size_stats["min"] == 100
        assert metadata.size_stats["max"] == 400
        assert metadata.size_stats["mean"] == 250
        assert metadata.size_stats["median"] == 250  # (200 + 300) / 2

    def test_to_dict_from_dict_roundtrip(
        self, sample_document: BenchmarkDocument
    ) -> None:
        """Metadata survives serialization round-trip."""
        metadata = CorpusMetadata.calculate([sample_document])
        data = metadata.to_dict()
        restored = CorpusMetadata.from_dict(data)

        assert restored.total_documents == metadata.total_documents
        assert restored.distractor_ratio == metadata.distractor_ratio


class TestDocumentCorpus:
    """Tests for DocumentCorpus collection."""

    def test_len(self, diverse_corpus: DocumentCorpus) -> None:
        """__len__ returns document count."""
        assert len(diverse_corpus) == 3

    def test_empty_corpus(self) -> None:
        """Empty corpus handles gracefully."""
        corpus = DocumentCorpus(documents=[])

        assert len(corpus) == 0
        assert corpus.metadata.total_documents == 0
        assert corpus.get_document("nonexistent") is None

    def test_get_document(
        self, diverse_corpus: DocumentCorpus, sample_document: BenchmarkDocument
    ) -> None:
        """get_document retrieves by ID."""
        found = diverse_corpus.get_document("doc_001")
        assert found is not None
        assert found.content == sample_document.content

        not_found = diverse_corpus.get_document("nonexistent")
        assert not_found is None

    def test_get_by_category(self, diverse_corpus: DocumentCorpus) -> None:
        """get_by_category filters correctly."""
        technical = diverse_corpus.get_by_category(DocumentCategory.TECHNICAL)
        assert len(technical) == 1
        assert technical[0].doc_id == "doc_001"

        legal = diverse_corpus.get_by_category(DocumentCategory.LEGAL)
        assert len(legal) == 1
        assert legal[0].doc_id == "legal_001"

    def test_get_by_domain(self, diverse_corpus: DocumentCorpus) -> None:
        """get_by_domain filters correctly."""
        python_docs = diverse_corpus.get_by_domain("python")
        assert len(python_docs) == 1
        assert python_docs[0].category == DocumentCategory.TECHNICAL

    def test_get_by_source(self, diverse_corpus: DocumentCorpus) -> None:
        """get_by_source filters correctly."""
        local_docs = diverse_corpus.get_by_source(DocumentSource.LOCAL)
        assert len(local_docs) == 1

        online_docs = diverse_corpus.get_by_source(DocumentSource.ONLINE)
        assert len(online_docs) == 1

    def test_get_distractors(self, diverse_corpus: DocumentCorpus) -> None:
        """get_distractors returns only distractor documents."""
        distractors = diverse_corpus.get_distractors()
        assert len(distractors) == 1
        assert all(d.is_distractor for d in distractors)

    def test_get_distractors_with_count(self, diverse_corpus: DocumentCorpus) -> None:
        """get_distractors respects count limit."""
        # Add more distractors
        for i in range(5):
            diverse_corpus.documents.append(
                BenchmarkDocument(
                    doc_id=f"distractor_{i}",
                    source=DocumentSource.SYNTHETIC,
                    category=DocumentCategory.SYNTHETIC,
                    domain="random",
                    file_path=f"synthetic://gen_{i}",
                    content=f"Distractor content {i}",
                    is_distractor=True,
                )
            )

        limited = diverse_corpus.get_distractors(count=3)
        assert len(limited) == 3

    def test_get_non_distractors(self, diverse_corpus: DocumentCorpus) -> None:
        """get_non_distractors excludes distractors."""
        real_docs = diverse_corpus.get_non_distractors()
        assert len(real_docs) == 2  # technical + legal
        assert not any(d.is_distractor for d in real_docs)

    def test_save_and_load(self, diverse_corpus: DocumentCorpus) -> None:
        """Corpus survives save/load round-trip."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_corpus.json"
            diverse_corpus.save(path)

            assert path.exists()

            loaded = DocumentCorpus.load(path)
            assert len(loaded) == len(diverse_corpus)
            assert loaded.version == diverse_corpus.version
            assert loaded.description == diverse_corpus.description

            # Verify content preserved
            for orig in diverse_corpus.documents:
                loaded_doc = loaded.get_document(orig.doc_id)
                assert loaded_doc is not None
                assert loaded_doc.content == orig.content
                assert loaded_doc.content_hash == orig.content_hash

    def test_deduplicate(self) -> None:
        """deduplicate removes duplicate content."""
        docs = [
            BenchmarkDocument(
                doc_id="doc_1",
                source=DocumentSource.LOCAL,
                category=DocumentCategory.TECHNICAL,
                domain="python",
                file_path="/path/1.md",
                content="Duplicate content",
            ),
            BenchmarkDocument(
                doc_id="doc_2",
                source=DocumentSource.ONLINE,
                category=DocumentCategory.TECHNICAL,
                domain="python",
                file_path="/path/2.md",
                content="Duplicate content",  # Same content
            ),
            BenchmarkDocument(
                doc_id="doc_3",
                source=DocumentSource.LOCAL,
                category=DocumentCategory.TECHNICAL,
                domain="python",
                file_path="/path/3.md",
                content="Unique content",
            ),
        ]

        corpus = DocumentCorpus(documents=docs)
        removed = corpus.deduplicate()

        assert removed == 1
        assert len(corpus) == 2

    def test_add_document(self) -> None:
        """add_document respects deduplication."""
        corpus = DocumentCorpus(documents=[])

        doc1 = BenchmarkDocument(
            doc_id="doc_1",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/1.md",
            content="Content A",
        )
        doc2 = BenchmarkDocument(
            doc_id="doc_2",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/2.md",
            content="Content A",  # Same content = duplicate
        )

        added1 = corpus.add_document(doc1)
        assert added1 is True
        assert len(corpus) == 1

        added2 = corpus.add_document(doc2)
        assert added2 is False  # Duplicate rejected
        assert len(corpus) == 1

    def test_get_category_distribution(self, diverse_corpus: DocumentCorpus) -> None:
        """get_category_distribution returns correct counts."""
        dist = diverse_corpus.get_category_distribution()

        assert dist[DocumentCategory.TECHNICAL] == 1
        assert dist[DocumentCategory.SYNTHETIC] == 1
        assert dist[DocumentCategory.LEGAL] == 1


class TestLargeDocuments:
    """Tests for handling large documents (500KB+)."""

    def test_large_document_serialization(self) -> None:
        """Large documents serialize without truncation."""
        large_content = "x" * 500_000  # 500KB

        doc = BenchmarkDocument(
            doc_id="large_doc",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="test",
            file_path="/path/large.md",
            content=large_content,
        )

        assert doc.size_bytes == 500_000

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large_corpus.json"
            corpus = DocumentCorpus(documents=[doc])
            corpus.save(path)

            loaded = DocumentCorpus.load(path)
            loaded_doc = loaded.get_document("large_doc")

            assert loaded_doc is not None
            assert len(loaded_doc.content) == 500_000
            assert loaded_doc.content_hash == doc.content_hash


class TestUnicodeContent:
    """Tests for unicode and emoji content."""

    def test_unicode_content_hash(self) -> None:
        """Unicode content hashes correctly."""
        doc = BenchmarkDocument(
            doc_id="unicode_doc",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.NARRATIVE,
            domain="fiction",
            file_path="/path/story.md",
            content="日本語テキスト with emoji 🎉🔥💯",
        )

        assert len(doc.content_hash) == 16
        assert doc.size_bytes > 0

    def test_unicode_roundtrip(self) -> None:
        """Unicode content survives serialization."""
        content = "Ελληνικά, 中文, العربية, emoji: 🌍🌎🌏"

        doc = BenchmarkDocument(
            doc_id="multilingual",
            source=DocumentSource.ONLINE,
            category=DocumentCategory.ACADEMIC,
            domain="linguistics",
            file_path="https://example.com/multilingual",
            content=content,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "unicode_corpus.json"
            corpus = DocumentCorpus(documents=[doc])
            corpus.save(path)

            loaded = DocumentCorpus.load(path)
            loaded_doc = loaded.get_document("multilingual")

            assert loaded_doc is not None
            assert loaded_doc.content == content
