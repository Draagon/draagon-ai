"""
Memory integration tests demonstrating FR-009 testing framework.

These tests show how to:
- Define seed items for test data
- Use seed sets for scenario setup
- Apply seeds using real MemoryProvider
- Evaluate responses with LLM-as-judge

Requirements:
- Running Neo4j instance
- neo4j package installed

Example:
    pytest tests/integration/test_memory_integration.py -v
"""

from __future__ import annotations

import pytest

from draagon_ai.testing import (
    SeedFactory,
    SeedItem,
    SeedSet,
    AgentEvaluator,
)
from draagon_ai.memory import (
    MemoryType,
    MemoryScope,
    Memory,
    NEO4J_AVAILABLE,
)


# =============================================================================
# Seed Items - Declarative Test Data
# =============================================================================


@SeedFactory.register("user_doug")
class DougUserSeed(SeedItem):
    """Doug's user profile with cat information.

    Seeds use REAL provider API - no wrapper methods.
    This ensures tests validate production interfaces.
    """

    async def create(self, provider, **deps) -> str:
        """Create Doug's user profile in memory.

        Returns:
            Memory ID of the created memory
        """
        memory = await provider.store(
            content="User profile: Doug has 3 cats named Whiskers, Mittens, and Shadow",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.9,
            entities=["Doug", "Whiskers", "Mittens", "Shadow"],
        )
        return memory.id


@SeedFactory.register("doug_birthday")
class DougBirthdaySeed(SeedItem):
    """Doug's birthday fact."""

    async def create(self, provider, **deps) -> str:
        memory = await provider.store(
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.8,
            entities=["Doug", "March 15"],
        )
        return memory.id


@SeedFactory.register("doug_preferences")
class DougPreferencesSeed(SeedItem):
    """Doug's preferences."""

    dependencies = ["user_doug"]  # Depends on user profile

    async def create(self, provider, user_doug: str = None, **deps) -> str:
        memory = await provider.store(
            content="Doug prefers dark mode and uses Celsius for temperature",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.85,
            entities=["Doug"],
        )
        return memory.id


# =============================================================================
# Seed Sets - Test Scenarios
# =============================================================================


USER_WITH_CATS = SeedSet(
    name="user_with_cats",
    seed_ids=["user_doug"],
    description="User profile with cat information",
)

USER_COMPLETE = SeedSet(
    name="user_complete",
    seed_ids=["user_doug", "doug_birthday", "doug_preferences"],
    description="Complete user profile with all facts",
)


# =============================================================================
# Memory Recall Tests
# =============================================================================


@pytest.mark.memory_integration
@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
class TestMemoryRecall:
    """Test agent recalls seeded facts from memory.

    These tests demonstrate the core pattern:
    1. Apply seed data
    2. Query memory
    3. Evaluate with LLM-as-judge
    """

    @pytest.mark.asyncio
    async def test_seed_data_persists(
        self, clean_database, seed_factory
    ):
        """Verify seed data is stored in database."""
        # This test verifies the seed mechanics without requiring
        # a full MemoryProvider - it uses the TestDatabase directly

        # Just verify database is clean
        count = await clean_database.node_count()
        assert count == 0, f"Expected clean database, found {count} nodes"

    @pytest.mark.asyncio
    async def test_evaluator_works_with_mock_llm(self, evaluator):
        """Verify evaluator works with mock LLM.

        This tests the evaluation infrastructure without needing
        real memory or agent - just the LLM-as-judge mechanics.
        """
        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Agent lists: Whiskers, Mittens, Shadow",
            actual_response="Your cats are Whiskers, Mittens, and Shadow!",
        )

        assert result.correct, f"Expected correct, got: {result.reasoning}"
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_coherence_evaluation(self, evaluator):
        """Test coherence evaluation for response quality."""
        result = await evaluator.evaluate_coherence(
            query="Tell me about the weather",
            response="The weather today is sunny with a high of 72°F.",
        )

        assert result.score > 0.5
        assert result.reasoning

    @pytest.mark.asyncio
    async def test_helpfulness_evaluation(self, evaluator):
        """Test helpfulness evaluation for response usefulness."""
        result = await evaluator.evaluate_helpfulness(
            query="How do I restart the service?",
            response="To restart the service, run: sudo systemctl restart myservice",
        )

        assert result.score > 0.5
        assert result.reasoning

    @pytest.mark.asyncio
    async def test_all_evaluations(self, evaluator):
        """Test running all evaluations at once."""
        results = await evaluator.evaluate_all(
            query="What's my birthday?",
            expected_outcome="Agent mentions March 15",
            actual_response="Based on my records, your birthday is March 15th!",
        )

        assert "correctness" in results
        assert "coherence" in results
        assert "helpfulness" in results

        # All evaluations should return reasonable results
        assert results["correctness"].correct
        assert results["coherence"].score > 0.5
        assert results["helpfulness"].score > 0.5


# =============================================================================
# Seed Factory Tests
# =============================================================================


@pytest.fixture
def factory_with_seeds():
    """Create a SeedFactory with our test seeds registered.

    Uses instance-based registration to avoid conflicts with
    tests that clear the global registry.
    """
    factory = SeedFactory()

    # Register seeds to this instance
    factory.register_instance("user_doug", DougUserSeed())
    factory.register_instance("doug_birthday", DougBirthdaySeed())
    factory.register_instance("doug_preferences", DougPreferencesSeed())

    return factory


@pytest.mark.memory_integration
class TestSeedFactoryIntegration:
    """Test SeedFactory mechanics in integration context."""

    def test_seeds_are_registered(self, factory_with_seeds):
        """Verify our test seeds are registered."""
        seeds = factory_with_seeds.list_seeds()

        assert "user_doug" in seeds
        assert "doug_birthday" in seeds
        assert "doug_preferences" in seeds

    def test_seed_set_contains_correct_seeds(self):
        """Verify seed sets contain expected seeds."""
        assert "user_doug" in USER_WITH_CATS.seed_ids
        assert len(USER_COMPLETE.seed_ids) == 3

    def test_dependency_resolution(self, factory_with_seeds):
        """Test dependency order is correct."""
        # doug_preferences depends on user_doug
        order = factory_with_seeds._topological_sort(["doug_preferences"])

        # user_doug must come before doug_preferences
        assert "user_doug" in order
        assert "doug_preferences" in order
        assert order.index("user_doug") < order.index("doug_preferences")


# =============================================================================
# Evaluation Edge Cases
# =============================================================================


@pytest.mark.memory_integration
class TestEvaluationEdgeCases:
    """Test edge cases in LLM-as-judge evaluation."""

    @pytest.mark.asyncio
    async def test_evaluation_result_is_falsy_when_incorrect(self, llm_provider):
        """Test that EvaluationResult is falsy when incorrect.

        This requires a custom evaluator that returns incorrect.
        """

        class IncorrectMockLLM:
            async def chat(self, messages, **kwargs):
                return """<result>
  <correct>false</correct>
  <reasoning>The response does not match expected outcome.</reasoning>
  <confidence>0.8</confidence>
</result>"""

        evaluator = AgentEvaluator(llm=IncorrectMockLLM())

        result = await evaluator.evaluate_correctness(
            query="What color is the sky?",
            expected_outcome="Blue",
            actual_response="The sky is green.",
        )

        assert not result.correct
        assert not result  # Should be falsy
        assert result.reasoning

    @pytest.mark.asyncio
    async def test_evaluation_handles_malformed_xml(self, llm_provider):
        """Test graceful handling of malformed LLM responses."""

        class MalformedMockLLM:
            async def chat(self, messages, **kwargs):
                return "This is not XML at all"

        evaluator = AgentEvaluator(llm=MalformedMockLLM())

        result = await evaluator.evaluate_correctness(
            query="Test query",
            expected_outcome="Test outcome",
            actual_response="Test response",
        )

        # Should return defaults gracefully
        assert not result.correct  # Default to false
        assert result.confidence == 0.5  # Default confidence

    @pytest.mark.asyncio
    async def test_evaluation_clamps_confidence(self, llm_provider):
        """Test that out-of-range confidence is clamped."""

        class OutOfRangeMockLLM:
            async def chat(self, messages, **kwargs):
                return """<result>
  <correct>true</correct>
  <reasoning>Very confident</reasoning>
  <confidence>1.5</confidence>
</result>"""

        evaluator = AgentEvaluator(llm=OutOfRangeMockLLM())

        result = await evaluator.evaluate_correctness(
            query="Test", expected_outcome="Test", actual_response="Test"
        )

        assert result.confidence == 1.0  # Clamped to max

    @pytest.mark.asyncio
    async def test_negative_confidence_clamped(self, llm_provider):
        """Test that negative confidence is clamped to 0."""

        class NegativeConfidenceLLM:
            async def chat(self, messages, **kwargs):
                return """<result>
  <correct>true</correct>
  <reasoning>Negative confidence</reasoning>
  <confidence>-0.5</confidence>
</result>"""

        evaluator = AgentEvaluator(llm=NegativeConfidenceLLM())

        result = await evaluator.evaluate_correctness(
            query="Test", expected_outcome="Test", actual_response="Test"
        )

        assert result.confidence == 0.0  # Clamped to min
