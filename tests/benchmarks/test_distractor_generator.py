"""Tests for DistractorGenerator.

Tests synthetic distractor document generation including
template-based generation, similarity levels, and LLM generation.
"""

import pytest

from draagon_ai.testing.benchmarks import (
    DistractorGenerator,
    DistractorConfig,
    SimilarityLevel,
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
)


class TestSimilarityLevel:
    """Tests for SimilarityLevel enum."""

    def test_all_levels_exist(self) -> None:
        """All expected similarity levels exist."""
        assert SimilarityLevel.VERY_DIFFERENT
        assert SimilarityLevel.SOMEWHAT_SIMILAR
        assert SimilarityLevel.VERY_SIMILAR

    def test_level_values(self) -> None:
        """Levels have string values."""
        assert SimilarityLevel.VERY_DIFFERENT.value == "very_different"
        assert SimilarityLevel.SOMEWHAT_SIMILAR.value == "somewhat_similar"
        assert SimilarityLevel.VERY_SIMILAR.value == "very_similar"


class TestDistractorConfig:
    """Tests for DistractorConfig dataclass."""

    def test_config_creation(self) -> None:
        """Config can be created with all fields."""
        config = DistractorConfig(
            category=DocumentCategory.TECHNICAL,
            domain="python",
            similarity_level=SimilarityLevel.VERY_SIMILAR,
            keywords=["async", "await", "coroutine"],
            target_length=500,
        )

        assert config.category == DocumentCategory.TECHNICAL
        assert config.domain == "python"
        assert config.similarity_level == SimilarityLevel.VERY_SIMILAR
        assert "async" in config.keywords

    def test_config_defaults(self) -> None:
        """Config has sensible defaults."""
        config = DistractorConfig(
            category=DocumentCategory.TECHNICAL,
            domain="test",
            similarity_level=SimilarityLevel.VERY_DIFFERENT,
        )

        assert config.keywords == []
        assert config.target_length == 500


class TestDistractorGenerator:
    """Tests for DistractorGenerator class."""

    def test_generator_initialization(self) -> None:
        """Generator initializes without LLM provider."""
        generator = DistractorGenerator()
        assert generator.llm_provider is None

    def test_generator_with_seed(self) -> None:
        """Generator with seed produces reproducible results."""
        gen1 = DistractorGenerator(seed=42)
        gen2 = DistractorGenerator(seed=42)

        docs1 = gen1.generate(count=5)
        docs2 = gen2.generate(count=5)

        # Same seed should produce same content
        assert docs1[0].content == docs2[0].content
        assert docs1[0].doc_id == docs2[0].doc_id

    def test_generator_different_seeds(self) -> None:
        """Different seeds produce different results."""
        gen1 = DistractorGenerator(seed=42)
        gen2 = DistractorGenerator(seed=123)

        docs1 = gen1.generate(count=5)
        docs2 = gen2.generate(count=5)

        # Different seeds should produce different content
        assert docs1[0].content != docs2[0].content


class TestTemplateGeneration:
    """Tests for template-based distractor generation."""

    def test_generates_requested_count(self) -> None:
        """Generator produces exact number of documents."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=10)

        assert len(docs) == 10

    def test_all_marked_as_distractors(self) -> None:
        """All generated docs are marked as distractors."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        assert all(doc.is_distractor for doc in docs)

    def test_all_have_synthetic_source(self) -> None:
        """All generated docs have synthetic source."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        assert all(doc.source == DocumentSource.SYNTHETIC for doc in docs)

    def test_all_have_synthetic_category(self) -> None:
        """All generated docs have synthetic category."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        assert all(doc.category == DocumentCategory.SYNTHETIC for doc in docs)

    def test_unique_doc_ids(self) -> None:
        """All generated docs have unique IDs."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=20)

        doc_ids = [doc.doc_id for doc in docs]
        assert len(doc_ids) == len(set(doc_ids))

    def test_docs_have_content(self) -> None:
        """All generated docs have non-empty content."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        assert all(len(doc.content) > 100 for doc in docs)

    def test_similarity_level_in_tags(self) -> None:
        """Similarity level is recorded in semantic tags."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(
            count=3,
            similarity_distribution={"very_different": 1.0, "somewhat_similar": 0.0, "very_similar": 0.0}
        )

        for doc in docs:
            assert "very_different" in doc.semantic_tags

    def test_distractor_tag_present(self) -> None:
        """All distractors have 'distractor' tag."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        for doc in docs:
            assert "distractor" in doc.semantic_tags


class TestSimilarityDistribution:
    """Tests for similarity level distribution."""

    def test_default_distribution(self) -> None:
        """Default distribution is 50/30/20."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=100)

        # Count by similarity level from metadata
        counts = {"very_different": 0, "somewhat_similar": 0, "very_similar": 0}
        for doc in docs:
            level = doc.metadata.get("similarity_level")
            if level:
                counts[level] += 1

        # Should be approximately 50/30/20
        assert 45 <= counts["very_different"] <= 55
        assert 25 <= counts["somewhat_similar"] <= 35
        assert 15 <= counts["very_similar"] <= 25

    def test_custom_distribution(self) -> None:
        """Custom distribution is respected."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(
            count=100,
            similarity_distribution={
                "very_different": 0.2,
                "somewhat_similar": 0.5,
                "very_similar": 0.3,
            }
        )

        counts = {"very_different": 0, "somewhat_similar": 0, "very_similar": 0}
        for doc in docs:
            level = doc.metadata.get("similarity_level")
            if level:
                counts[level] += 1

        # Should be approximately 20/50/30
        assert 15 <= counts["very_different"] <= 25
        assert 45 <= counts["somewhat_similar"] <= 55
        assert 25 <= counts["very_similar"] <= 35

    def test_single_level_only(self) -> None:
        """Can generate all docs at one similarity level."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(
            count=10,
            similarity_distribution={"very_similar": 1.0, "very_different": 0.0, "somewhat_similar": 0.0}
        )

        for doc in docs:
            assert doc.metadata.get("similarity_level") == "very_similar"


class TestCategoryDistribution:
    """Tests for category distribution in generated distractors."""

    def test_uses_all_categories_by_default(self) -> None:
        """Generator uses all non-SYNTHETIC categories."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=100)

        base_categories = {doc.metadata.get("base_category") for doc in docs}

        # Should have sampled from multiple categories
        assert len(base_categories) > 3

    def test_custom_categories(self) -> None:
        """Generator respects custom category list."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(
            count=20,
            categories=[DocumentCategory.TECHNICAL, DocumentCategory.LEGAL],
        )

        base_categories = {doc.metadata.get("base_category") for doc in docs}

        # Should only have technical and legal
        assert base_categories <= {"technical", "legal"}

    def test_single_category(self) -> None:
        """Can generate all from single category."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(
            count=10,
            categories=[DocumentCategory.NARRATIVE],
        )

        for doc in docs:
            assert doc.metadata.get("base_category") == "narrative"


class TestMetadata:
    """Tests for distractor metadata."""

    def test_has_generated_at(self) -> None:
        """Metadata includes generation timestamp."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert "generated_at" in docs[0].metadata

    def test_has_similarity_level(self) -> None:
        """Metadata includes similarity level."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert "similarity_level" in docs[0].metadata

    def test_has_base_category(self) -> None:
        """Metadata includes base category."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert "base_category" in docs[0].metadata

    def test_has_topic(self) -> None:
        """Metadata includes topic."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert "topic" in docs[0].metadata

    def test_has_generator_type(self) -> None:
        """Metadata includes generator type."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert docs[0].metadata.get("generator") == "template"


class TestDocumentStructure:
    """Tests for generated document structure."""

    def test_doc_id_format(self) -> None:
        """Document IDs follow expected format."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=5)

        for i, doc in enumerate(docs):
            assert doc.doc_id.startswith("distractor_")
            assert doc.doc_id == f"distractor_{i:04d}"

    def test_file_path_format(self) -> None:
        """File paths follow expected format."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert docs[0].file_path.startswith("synthetic://distractor/")

    def test_domain_includes_base_category(self) -> None:
        """Domain includes base category."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert docs[0].domain.startswith("synthetic_")


class TestLLMGeneration:
    """Tests for LLM-based generation."""

    @pytest.mark.asyncio
    async def test_requires_llm_provider(self) -> None:
        """generate_with_llm raises without LLM provider."""
        generator = DistractorGenerator()

        with pytest.raises(ValueError, match="LLM provider required"):
            await generator.generate_with_llm(count=1)

    @pytest.mark.asyncio
    async def test_with_mock_llm(self) -> None:
        """LLM generation works with mock provider."""

        class MockLLM:
            async def chat(self, messages, **kwargs):
                return {"content": "This is generated content for testing purposes. " * 20}

        generator = DistractorGenerator(llm_provider=MockLLM())
        docs = await generator.generate_with_llm(count=3)

        assert len(docs) == 3
        assert all(doc.is_distractor for doc in docs)
        assert all("llm_generated" in doc.semantic_tags for doc in docs)
        assert all(doc.metadata.get("generator") == "llm" for doc in docs)

    @pytest.mark.asyncio
    async def test_llm_with_reference_docs(self) -> None:
        """LLM generation can use reference documents."""

        class MockLLM:
            async def chat(self, messages, **kwargs):
                # Check that reference was included in prompt
                prompt = messages[-1]["content"]
                return {"content": f"Generated based on prompt length: {len(prompt)}"}

        reference = BenchmarkDocument(
            doc_id="ref_001",
            source=DocumentSource.LOCAL,
            category=DocumentCategory.TECHNICAL,
            domain="python",
            file_path="/path/ref.md",
            content="Reference content for generation context",
        )

        generator = DistractorGenerator(llm_provider=MockLLM())
        docs = await generator.generate_with_llm(
            count=2,
            reference_docs=[reference],
        )

        assert len(docs) == 2

    @pytest.mark.asyncio
    async def test_llm_fallback_on_error(self) -> None:
        """Falls back to template on LLM error."""

        class FailingLLM:
            async def chat(self, messages, **kwargs):
                raise RuntimeError("LLM API error")

        generator = DistractorGenerator(llm_provider=FailingLLM(), seed=42)
        docs = await generator.generate_with_llm(count=2)

        # Should still generate docs via template fallback
        assert len(docs) == 2
        assert all(doc.metadata.get("generator") == "template" for doc in docs)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_generate_zero_docs(self) -> None:
        """Generating zero docs returns empty list."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=0)

        assert docs == []

    def test_generate_one_doc(self) -> None:
        """Can generate exactly one doc."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=1)

        assert len(docs) == 1

    def test_large_count(self) -> None:
        """Can generate large number of docs."""
        generator = DistractorGenerator(seed=42)
        docs = generator.generate(count=500)

        assert len(docs) == 500
        # All should have unique IDs
        doc_ids = [doc.doc_id for doc in docs]
        assert len(doc_ids) == len(set(doc_ids))

    def test_empty_categories_list_raises(self) -> None:
        """Empty categories list raises when trying to generate."""
        generator = DistractorGenerator(seed=42)

        # Empty list means no categories to choose from
        with pytest.raises(IndexError):
            generator.generate(count=5, categories=[])
