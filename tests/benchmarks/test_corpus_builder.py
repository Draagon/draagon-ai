"""Tests for CorpusBuilder orchestrator.

Tests corpus assembly from multiple sources including local scanning,
online fetching, legal documents, and synthetic distractor generation.
"""

import tempfile
from pathlib import Path

import pytest

from draagon_ai.testing.benchmarks import (
    CorpusBuilder,
    CorpusBuilderConfig,
    SourceConfig,
    BuildProgress,
    build_default_corpus,
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
    DocumentCorpus,
)


class TestCorpusBuilderConfig:
    """Tests for CorpusBuilderConfig dataclass."""

    def test_default_config(self) -> None:
        """Default config has sensible values."""
        config = CorpusBuilderConfig()

        assert config.min_documents == 500
        assert config.distractor_ratio == 0.4
        assert config.local.enabled is True
        assert config.online.enabled is True
        assert config.legal.enabled is True
        assert config.distractors.enabled is True

    def test_custom_config(self) -> None:
        """Custom config values are respected."""
        config = CorpusBuilderConfig(
            min_documents=100,
            distractor_ratio=0.3,
            local=SourceConfig(enabled=False),
            online=SourceConfig(max_docs=50),
        )

        assert config.min_documents == 100
        assert config.distractor_ratio == 0.3
        assert config.local.enabled is False
        assert config.online.max_docs == 50


class TestSourceConfig:
    """Tests for SourceConfig dataclass."""

    def test_default_source_config(self) -> None:
        """Default source config is enabled with no limits."""
        config = SourceConfig()

        assert config.enabled is True
        assert config.max_docs is None
        assert config.weight == 1.0

    def test_disabled_source(self) -> None:
        """Source can be disabled."""
        config = SourceConfig(enabled=False)
        assert config.enabled is False

    def test_limited_source(self) -> None:
        """Source can have max_docs limit."""
        config = SourceConfig(max_docs=100)
        assert config.max_docs == 100


class TestBuildProgress:
    """Tests for BuildProgress dataclass."""

    def test_default_progress(self) -> None:
        """Default progress is all zeros."""
        progress = BuildProgress()

        assert progress.local_docs == 0
        assert progress.online_docs == 0
        assert progress.legal_docs == 0
        assert progress.distractor_docs == 0
        assert progress.total == 0
        assert progress.duplicates_removed == 0

    def test_progress_str(self) -> None:
        """Progress has readable string representation."""
        progress = BuildProgress(
            local_docs=10,
            online_docs=20,
            legal_docs=5,
            distractor_docs=15,
            total=50,
            duplicates_removed=3,
        )

        s = str(progress)
        assert "local=10" in s
        assert "online=20" in s
        assert "total=50" in s


class TestCorpusBuilder:
    """Tests for CorpusBuilder class."""

    def test_builder_initialization(self) -> None:
        """Builder initializes without local paths."""
        builder = CorpusBuilder()

        assert builder.local_paths == []
        assert builder.config.min_documents == 500

    def test_builder_with_local_paths(self, tmp_path: Path) -> None:
        """Builder accepts local paths."""
        builder = CorpusBuilder(local_paths=[tmp_path])

        assert tmp_path in builder.local_paths

    def test_builder_with_config(self) -> None:
        """Builder accepts custom config."""
        config = CorpusBuilderConfig(min_documents=100)
        builder = CorpusBuilder(config=config)

        assert builder.config.min_documents == 100

    def test_builder_invalid_path_skipped(self) -> None:
        """Builder skips invalid local paths."""
        builder = CorpusBuilder(
            local_paths=[Path("/nonexistent/path/that/does/not/exist")]
        )

        # Should not raise, just skip the invalid path
        assert len(builder._local_scanners) == 0


class TestCorpusBuildLocal:
    """Tests for local document collection."""

    @pytest.mark.asyncio
    async def test_build_with_local_only(self, tmp_path: Path) -> None:
        """Builds corpus from local documents only."""
        # Create test files (min 500 bytes for local scanner)
        content1 = "# Test Document 1\n\n" + ("This is test content for document one. " * 20)
        content2 = "# Test Document 2\n\n" + ("This is test content for document two. " * 20)
        (tmp_path / "doc1.md").write_text(content1)
        (tmp_path / "doc2.md").write_text(content2)

        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,  # No distractors
            local=SourceConfig(enabled=True),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=False),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(
            local_paths=[tmp_path],
            config=config,
        )

        corpus = await builder.build()

        assert len(corpus) >= 1
        assert all(doc.source == DocumentSource.LOCAL for doc in corpus.documents)

    @pytest.mark.asyncio
    async def test_build_respects_local_max_docs(self, tmp_path: Path) -> None:
        """Build respects local max_docs limit."""
        # Create many test files (min 500 bytes for local scanner)
        for i in range(10):
            content = f"# Document {i}\n\n" + (f"Unique content here for document {i}. " * 20)
            (tmp_path / f"doc{i}.md").write_text(content)

        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=True, max_docs=3),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=False),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(
            local_paths=[tmp_path],
            config=config,
        )

        corpus = await builder.build()

        # Should be limited to 3 local docs
        local_docs = [d for d in corpus.documents if d.source == DocumentSource.LOCAL]
        assert len(local_docs) <= 3


class TestCorpusBuildDisabled:
    """Tests for disabled sources."""

    @pytest.mark.asyncio
    async def test_build_local_disabled(self, tmp_path: Path) -> None:
        """Build skips local when disabled."""
        content = "# Test Document\n\n" + ("Content here for testing. " * 30)
        (tmp_path / "doc.md").write_text(content)

        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=False),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(
            local_paths=[tmp_path],
            config=config,
        )

        corpus = await builder.build()

        # No documents since everything is disabled
        assert len(corpus) == 0

    @pytest.mark.asyncio
    async def test_build_online_disabled(self) -> None:
        """Build skips online when disabled."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True),  # Just legal enabled
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        # Should have only legal docs
        for doc in corpus.documents:
            assert doc.category == DocumentCategory.LEGAL


class TestCorpusBuildLegal:
    """Tests for legal document collection."""

    @pytest.mark.asyncio
    async def test_build_with_legal_only(self) -> None:
        """Builds corpus with legal documents only."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=5),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        assert len(corpus) >= 1
        assert all(doc.category == DocumentCategory.LEGAL for doc in corpus.documents)


class TestCorpusBuildDistractors:
    """Tests for distractor generation."""

    @pytest.mark.asyncio
    async def test_build_generates_distractors(self) -> None:
        """Build generates distractors to meet ratio."""
        config = CorpusBuilderConfig(
            min_documents=10,
            distractor_ratio=0.4,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=6),
            distractors=SourceConfig(enabled=True),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        # Should have distractors
        distractors = corpus.get_distractors()
        assert len(distractors) >= 1

        # All distractors should be marked correctly
        for doc in distractors:
            assert doc.is_distractor is True
            assert doc.source == DocumentSource.SYNTHETIC

    @pytest.mark.asyncio
    async def test_distractor_ratio_calculation(self) -> None:
        """Distractor ratio is approximately correct."""
        config = CorpusBuilderConfig(
            min_documents=20,
            distractor_ratio=0.4,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=12),
            distractors=SourceConfig(enabled=True),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        total = len(corpus)
        distractor_count = len(corpus.get_distractors())
        ratio = distractor_count / total if total > 0 else 0

        # Should be approximately 0.4 (allow some variance)
        assert 0.2 <= ratio <= 0.6

    @pytest.mark.asyncio
    async def test_build_no_distractors_when_disabled(self) -> None:
        """No distractors when disabled."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.4,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=5),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        assert len(corpus.get_distractors()) == 0


class TestCorpusBuildDeduplication:
    """Tests for content deduplication."""

    @pytest.mark.asyncio
    async def test_build_deduplicates(self, tmp_path: Path) -> None:
        """Build removes duplicate content."""
        # Create duplicate files (min 500 bytes for local scanner)
        content = "# Duplicate Content\n\n" + ("This is the same content in multiple files. " * 20)
        (tmp_path / "doc1.md").write_text(content)
        (tmp_path / "doc2.md").write_text(content)
        (tmp_path / "doc3.md").write_text(content)
        unique_content = "# Unique Content\n\n" + ("This is different content that is unique. " * 20)
        (tmp_path / "unique.md").write_text(unique_content)

        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=True),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=False),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(
            local_paths=[tmp_path],
            config=config,
        )

        corpus = await builder.build()

        # Should only have 2 unique documents
        assert len(corpus) == 2


class TestCorpusValidation:
    """Tests for corpus validation."""

    @pytest.mark.asyncio
    async def test_validate_corpus_meets_requirements(self) -> None:
        """Valid corpus passes validation."""
        config = CorpusBuilderConfig(
            min_documents=10,
            distractor_ratio=0.4,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=10),
            distractors=SourceConfig(enabled=True),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        issues = builder.validate_corpus(corpus)

        # May have issues about category coverage, but should have docs
        if len(corpus) >= 10:
            assert "Document count" not in str(issues)

    @pytest.mark.asyncio
    async def test_validate_detects_insufficient_docs(self) -> None:
        """Validation detects insufficient document count."""
        config = CorpusBuilderConfig(
            min_documents=1000,  # Very high requirement
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=5),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        issues = builder.validate_corpus(corpus)

        # Should detect insufficient docs
        assert any("Document count" in issue for issue in issues)

    def test_get_category_coverage(self) -> None:
        """Category coverage tracks present categories."""
        builder = CorpusBuilder()

        # Create a corpus with specific categories
        docs = [
            BenchmarkDocument(
                doc_id="tech_001",
                source=DocumentSource.LOCAL,
                category=DocumentCategory.TECHNICAL,
                domain="python",
                file_path="/test.py",
                content="Technical content",
            ),
            BenchmarkDocument(
                doc_id="legal_001",
                source=DocumentSource.SYNTHETIC,
                category=DocumentCategory.LEGAL,
                domain="license",
                file_path="/license.md",
                content="Legal content",
            ),
        ]
        corpus = DocumentCorpus(documents=docs)

        coverage = builder.get_category_coverage(corpus)

        assert coverage["technical"] is True
        assert coverage["legal"] is True
        assert coverage["narrative"] is False
        assert coverage["academic"] is False


class TestCorpusSave:
    """Tests for corpus persistence."""

    @pytest.mark.asyncio
    async def test_build_saves_corpus(self, tmp_path: Path) -> None:
        """Build saves corpus to output_path."""
        output_file = tmp_path / "corpus.json"

        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=3),
            distractors=SourceConfig(enabled=False),
            output_path=output_file,
        )

        builder = CorpusBuilder(config=config)
        await builder.build()

        # File should exist
        assert output_file.exists()

        # Should be loadable
        loaded = DocumentCorpus.load(output_file)
        assert len(loaded) >= 1


class TestBuildDefaultCorpus:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_build_default_corpus_function(self, tmp_path: Path) -> None:
        """build_default_corpus convenience function works."""
        # Create minimal test data (min 500 bytes for local scanner)
        local_dir = tmp_path / "dev"
        local_dir.mkdir()
        content = "# Test\n\n" + ("Content for testing purposes. " * 30)
        (local_dir / "test.md").write_text(content)

        corpus = await build_default_corpus(
            local_paths=[local_dir],
            min_documents=1,
        )

        assert isinstance(corpus, DocumentCorpus)
        assert len(corpus) >= 1

    @pytest.mark.asyncio
    async def test_build_default_corpus_no_paths(self) -> None:
        """build_default_corpus works without local paths."""
        corpus = await build_default_corpus(
            local_paths=[],
            min_documents=1,
        )

        # Should still work (will use distractors to meet minimum)
        assert isinstance(corpus, DocumentCorpus)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_corpus(self) -> None:
        """Handles empty corpus gracefully."""
        config = CorpusBuilderConfig(
            min_documents=0,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=False),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        assert len(corpus) == 0
        assert corpus.metadata.total_documents == 0

    @pytest.mark.asyncio
    async def test_zero_distractor_ratio(self) -> None:
        """Zero distractor ratio produces no distractors."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=5),
            distractors=SourceConfig(enabled=True),  # Enabled but ratio is 0
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        assert len(corpus.get_distractors()) == 0

    @pytest.mark.asyncio
    async def test_invalid_distractor_ratio(self) -> None:
        """Invalid distractor ratio (>1) produces no distractors."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=1.5,  # Invalid ratio
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=5),
            distractors=SourceConfig(enabled=True),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build()

        # Should not crash, just skip distractors
        assert isinstance(corpus, DocumentCorpus)

    @pytest.mark.asyncio
    async def test_build_with_version_and_description(self) -> None:
        """Build accepts version and description."""
        config = CorpusBuilderConfig(
            min_documents=1,
            distractor_ratio=0.0,
            local=SourceConfig(enabled=False),
            online=SourceConfig(enabled=False),
            legal=SourceConfig(enabled=True, max_docs=2),
            distractors=SourceConfig(enabled=False),
        )

        builder = CorpusBuilder(config=config)
        corpus = await builder.build(
            version="2.0.0",
            description="Test corpus",
        )

        assert corpus.version == "2.0.0"
        assert corpus.description == "Test corpus"
