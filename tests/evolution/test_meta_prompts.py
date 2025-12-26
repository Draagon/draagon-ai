"""Tests for self-referential meta prompts."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.evolution.meta_prompts import (
    MetaPrompt,
    MutationResult,
    MetaPromptManager,
    DEFAULT_MUTATION_PROMPTS,
)


class TestMetaPrompt:
    """Test MetaPrompt dataclass."""

    def test_create_basic_meta_prompt(self):
        """Test creating a basic meta prompt."""
        mp = MetaPrompt(
            prompt_id="test_1",
            content="Improve clarity.",
            target_type="mutation",
        )

        assert mp.prompt_id == "test_1"
        assert mp.content == "Improve clarity."
        assert mp.target_type == "mutation"
        assert mp.generation == 0
        assert mp.fitness == 0.5

    def test_success_rate_no_usage(self):
        """Test success rate with no usage."""
        mp = MetaPrompt(
            prompt_id="test_1",
            content="Test",
            target_type="mutation",
        )

        assert mp.success_rate == 0.5  # Default

    def test_success_rate_with_usage(self):
        """Test success rate calculation."""
        mp = MetaPrompt(
            prompt_id="test_1",
            content="Test",
            target_type="mutation",
            usage_count=10,
            success_count=7,
        )

        assert mp.success_rate == 0.7

    def test_record_usage_success(self):
        """Test recording successful usage."""
        mp = MetaPrompt(
            prompt_id="test_1",
            content="Test",
            target_type="mutation",
        )

        mp.record_usage(successful=True)

        assert mp.usage_count == 1
        assert mp.success_count == 1

    def test_record_usage_failure(self):
        """Test recording failed usage."""
        mp = MetaPrompt(
            prompt_id="test_1",
            content="Test",
            target_type="mutation",
        )

        mp.record_usage(successful=False)

        assert mp.usage_count == 1
        assert mp.success_count == 0


class TestDefaultMutationPrompts:
    """Test default mutation prompts."""

    def test_has_mutation_prompts(self):
        """Test that we have default mutation prompts."""
        assert len(DEFAULT_MUTATION_PROMPTS) > 0

    def test_all_are_mutation_type(self):
        """Test that all defaults are mutation type."""
        for mp in DEFAULT_MUTATION_PROMPTS:
            assert mp.target_type == "mutation"

    def test_has_expansion_prompts(self):
        """Test that we have expansion prompts."""
        expansion_keywords = ["add", "expand", "example"]
        has_expansion = any(
            any(kw in mp.content.lower() for kw in expansion_keywords)
            for mp in DEFAULT_MUTATION_PROMPTS
        )
        assert has_expansion

    def test_has_compression_prompts(self):
        """Test that we have compression prompts."""
        compression_keywords = ["remove", "consolidate", "compress"]
        has_compression = any(
            any(kw in mp.content.lower() for kw in compression_keywords)
            for mp in DEFAULT_MUTATION_PROMPTS
        )
        assert has_compression


class TestMetaPromptManager:
    """Test MetaPromptManager."""

    @pytest.fixture
    def mock_llm(self):
        """Create a mock LLM provider."""
        llm = MagicMock()
        llm.chat = AsyncMock(return_value={
            "content": "This is the mutated prompt with improvements."
        })
        return llm

    @pytest.fixture
    def manager(self, mock_llm):
        """Create a MetaPromptManager."""
        return MetaPromptManager(llm=mock_llm)

    def test_init_with_defaults(self, manager):
        """Test initialization with default prompts."""
        assert len(manager.mutation_prompts) == len(DEFAULT_MUTATION_PROMPTS)

    def test_init_with_custom_prompts(self, mock_llm):
        """Test initialization with custom prompts."""
        custom = [
            MetaPrompt(prompt_id="custom_1", content="Custom", target_type="mutation"),
        ]
        manager = MetaPromptManager(llm=mock_llm, initial_prompts=custom)

        assert len(manager.mutation_prompts) == 1
        assert manager.mutation_prompts[0].prompt_id == "custom_1"

    def test_select_mutation_prompt(self, manager):
        """Test selecting a mutation prompt."""
        selected = manager.select_mutation_prompt()

        assert selected is not None
        assert isinstance(selected, MetaPrompt)
        assert selected in manager.mutation_prompts

    def test_select_favors_successful(self, mock_llm):
        """Test that selection favors successful prompts."""
        prompts = [
            MetaPrompt(
                prompt_id="low",
                content="Low success",
                target_type="mutation",
                usage_count=100,
                success_count=10,  # 10% success
            ),
            MetaPrompt(
                prompt_id="high",
                content="High success",
                target_type="mutation",
                usage_count=100,
                success_count=90,  # 90% success
            ),
        ]
        manager = MetaPromptManager(llm=mock_llm, initial_prompts=prompts)

        # Run many selections
        high_wins = 0
        for _ in range(100):
            selected = manager.select_mutation_prompt()
            if selected.prompt_id == "high":
                high_wins += 1

        # High success rate should win more often
        assert high_wins > 50

    @pytest.mark.asyncio
    async def test_mutate_success(self, manager):
        """Test successful mutation."""
        result = await manager.mutate("Original prompt here.")

        assert result.success
        assert result.mutated_prompt is not None
        assert result.meta_prompt_used is not None

    @pytest.mark.asyncio
    async def test_mutate_with_failure_cases(self, manager):
        """Test mutation with failure cases."""
        failure_cases = [
            {"query": "What time?", "issue": "Wrong action"},
        ]

        result = await manager.mutate("Original prompt", failure_cases)

        assert result.success

    @pytest.mark.asyncio
    async def test_mutate_no_llm(self):
        """Test mutation without LLM."""
        manager = MetaPromptManager(llm=None)

        result = await manager.mutate("prompt")

        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_mutate_removes_markdown(self, mock_llm):
        """Test that mutation removes markdown code blocks."""
        mock_llm.chat = AsyncMock(return_value={
            "content": "```\nClean content here\n```"
        })
        manager = MetaPromptManager(llm=mock_llm)

        result = await manager.mutate("original")

        assert result.success
        assert "```" not in result.mutated_prompt

    @pytest.mark.asyncio
    async def test_crossover_success(self, manager):
        """Test successful crossover."""
        result = await manager.crossover("Parent 1", "Parent 2")

        assert result.success
        assert result.mutated_prompt is not None

    @pytest.mark.asyncio
    async def test_crossover_no_llm(self):
        """Test crossover without LLM."""
        manager = MetaPromptManager(llm=None)

        result = await manager.crossover("p1", "p2")

        assert not result.success

    @pytest.mark.asyncio
    async def test_evolve_meta_prompts(self, manager, mock_llm):
        """Test self-referential meta prompt evolution."""
        mock_llm.chat = AsyncMock(return_value={
            "content": "Improved mutation instruction here."
        })

        evolved_count = await manager.evolve_meta_prompts()

        assert evolved_count > 0
        assert manager._evolution_count == 1

    @pytest.mark.asyncio
    async def test_evolve_tracks_lineage(self, manager, mock_llm):
        """Test that evolution tracks lineage."""
        mock_llm.chat = AsyncMock(return_value={
            "content": "Evolved instruction."
        })

        original_id = manager.mutation_prompts[0].prompt_id

        await manager.evolve_meta_prompts()

        # Find the evolved version
        evolved = next(
            (mp for mp in manager.mutation_prompts if mp.parent_id == original_id),
            None
        )
        if evolved:
            assert evolved.generation == 1
            assert len(evolved.mutation_history) > 0

    def test_record_result(self, manager):
        """Test recording results for meta prompts."""
        mp = manager.mutation_prompts[0]
        original_usage = mp.usage_count

        manager.record_result(mp, successful=True)

        assert mp.usage_count == original_usage + 1
        assert mp.success_count >= 1

    def test_get_stats(self, manager):
        """Test getting statistics."""
        stats = manager.get_stats()

        assert "total_prompts" in stats
        assert "evolution_count" in stats
        assert "prompts" in stats
        assert len(stats["prompts"]) == len(manager.mutation_prompts)

    def test_serialize_deserialize(self, manager):
        """Test serialization round-trip."""
        # Create fresh prompts to avoid shared state from DEFAULT_MUTATION_PROMPTS
        fresh_prompts = [
            MetaPrompt(
                prompt_id="test_prompt_1",
                content="Test mutation instruction 1.",
                target_type="mutation",
            ),
            MetaPrompt(
                prompt_id="test_prompt_2",
                content="Test mutation instruction 2.",
                target_type="mutation",
            ),
        ]
        manager.mutation_prompts = fresh_prompts

        # Record some usage
        manager.mutation_prompts[0].record_usage(True)
        manager._evolution_count = 5

        # Serialize
        data = manager.serialize()

        # Create new manager and deserialize
        new_manager = MetaPromptManager(llm=None)
        new_manager.deserialize(data)

        assert new_manager._evolution_count == 5
        assert len(new_manager.mutation_prompts) == len(manager.mutation_prompts)
        assert new_manager.mutation_prompts[0].usage_count == 1
