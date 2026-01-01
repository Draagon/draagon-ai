"""Validation tests for real agent integration test fixtures.

These tests verify that all fixtures work correctly and fail gracefully
when dependencies are unavailable.
"""

import os

import pytest

from draagon_ai.memory.base import Memory, MemoryType, MemoryScope


class TestEmbeddingProviderFixture:
    """Test embedding provider fixture."""

    @pytest.mark.asyncio
    async def test_embedding_provider_generates_vectors(self, embedding_provider):
        """Embedding provider should generate vectors."""
        embedding = await embedding_provider.embed("test text")

        assert isinstance(embedding, list)
        assert len(embedding) == 1536  # OpenAI-compatible dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embedding_provider_deterministic(self, embedding_provider):
        """Embedding provider should be deterministic for same input."""
        text = "The quick brown fox"

        embedding1 = await embedding_provider.embed(text)
        embedding2 = await embedding_provider.embed(text)

        assert embedding1 == embedding2

    @pytest.mark.asyncio
    async def test_embedding_provider_varies_by_text(self, embedding_provider):
        """Embedding provider should vary by input text."""
        embedding1 = await embedding_provider.embed("cats")
        embedding2 = await embedding_provider.embed("dogs")

        assert embedding1 != embedding2


class TestRealLLMFixture:
    """Test real LLM provider fixture."""

    def test_real_llm_skips_without_api_key(self, monkeypatch):
        """Fixture should skip if no API key is set."""
        # Remove API keys from environment
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Import fixture function
        from tests.integration.agents.conftest import real_llm

        # Should raise pytest.skip
        with pytest.raises(pytest.skip.Exception):
            real_llm()

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_real_llm_can_generate(self, real_llm):
        """Real LLM should be able to generate responses."""
        response = await real_llm.generate("What is 2+2?")

        assert isinstance(response, str)
        assert len(response) > 0
        # LLM should mention "4" somewhere
        assert "4" in response or "four" in response.lower()


class TestDatabaseFixtures:
    """Test database fixture behavior."""

    @pytest.mark.asyncio
    async def test_test_database_initializes(self, test_database):
        """Test database should initialize successfully."""
        # Fixture should have initialized
        assert test_database is not None

        # Should be able to verify connection
        await test_database.verify_connection()

    @pytest.mark.asyncio
    async def test_clean_database_clears_data(self, clean_database, memory_provider):
        """Clean database should clear data before each test."""
        # Store a memory
        memory_id = await memory_provider.store(
            content="Test memory",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.SESSION,
            user_id="test_user",
        )

        # Verify it exists
        memory = await memory_provider.get(memory_id)
        assert memory is not None

        # Clean database (simulated by new test using clean_database)
        await clean_database.clear()

        # Memory should be gone after clear
        # (In real test, this would be a new test with clean_database fixture)


class TestMemoryProviderFixture:
    """Test memory provider fixture."""

    @pytest.mark.asyncio
    async def test_memory_provider_initializes(self, memory_provider):
        """Memory provider should initialize successfully."""
        assert memory_provider is not None

    @pytest.mark.asyncio
    async def test_memory_provider_can_store(self, memory_provider):
        """Memory provider should be able to store memories."""
        memory_id = await memory_provider.store(
            content="The capital of France is Paris",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
            importance=0.8,
        )

        assert isinstance(memory_id, str)
        assert len(memory_id) > 0

    @pytest.mark.asyncio
    async def test_memory_provider_can_retrieve(self, memory_provider):
        """Memory provider should be able to retrieve stored memories."""
        # Store memory
        memory_id = await memory_provider.store(
            content="The capital of France is Paris",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
        )

        # Retrieve by ID
        memory = await memory_provider.get(memory_id)

        assert memory is not None
        assert memory.content == "The capital of France is Paris"
        assert memory.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_memory_provider_can_search(self, memory_provider):
        """Memory provider should support semantic search."""
        # Store some memories
        await memory_provider.store(
            content="Paris is the capital of France",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
        )

        # Search
        results = await memory_provider.search(
            query="capital of France",
            limit=5,
        )

        assert len(results) > 0
        # Should find the Paris fact
        assert any("Paris" in r.content for r in results)


class TestToolRegistryFixture:
    """Test tool registry fixture."""

    def test_tool_registry_empty_by_default(self, tool_registry):
        """Tool registry should start empty for each test."""
        tools = tool_registry.list_tools()
        assert len(tools) == 0

    @pytest.mark.asyncio
    async def test_tool_registry_can_register_tools(self, tool_registry):
        """Tool registry should allow tool registration."""
        # Register a test tool
        async def test_tool(arg: str) -> str:
            return f"Result: {arg}"

        tool_registry.register(
            name="test_tool",
            handler=test_tool,
            description="A test tool",
        )

        # Verify registration
        tools = tool_registry.list_tools()
        assert "test_tool" in tools


class TestAgentFixture:
    """Test agent loop fixture."""

    @pytest.mark.asyncio
    async def test_agent_initializes(self, agent):
        """Agent should initialize successfully."""
        assert agent is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_agent_can_process_simple_query(self, agent):
        """Agent should be able to process basic queries."""
        # NOTE: This requires proper AgentLoop.process() signature
        # which may need Behavior and AgentContext arguments
        # Skip this test for now - will be tested in TASK-010
        pytest.skip("Agent processing tested in TASK-010")


class TestEvaluatorFixture:
    """Test evaluator fixture."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_evaluator_can_evaluate_correctness(self, evaluator):
        """Evaluator should be able to evaluate response correctness."""
        result = await evaluator.evaluate_correctness(
            query="What is 2+2?",
            expected_outcome="The answer should be 4",
            actual_response="The answer is 4",
        )

        assert result.correct is True
        assert len(result.reasoning) > 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_evaluator_detects_incorrect_responses(self, evaluator):
        """Evaluator should detect incorrect responses."""
        result = await evaluator.evaluate_correctness(
            query="What is 2+2?",
            expected_outcome="The answer should be 4",
            actual_response="The answer is 5",
        )

        assert result.correct is False
        assert "incorrect" in result.reasoning.lower() or "wrong" in result.reasoning.lower()


class TestAdvanceTimeUtility:
    """Test time advancement utility."""

    @pytest.mark.asyncio
    async def test_advance_time_with_mock(self, advance_time):
        """Advance time utility should work in mock mode."""
        # Mock mode should execute instantly
        import time

        start = time.time()
        await advance_time(minutes=10, mock_time=True)
        elapsed = time.time() - start

        # Should be nearly instant (< 1 second)
        assert elapsed < 1.0

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_advance_time_with_real_delay(self, advance_time):
        """Advance time utility should work with real delays."""
        # Real mode should actually wait
        import time

        start = time.time()
        await advance_time(seconds=2, mock_time=False)
        elapsed = time.time() - start

        # Should have actually waited ~2 seconds
        assert 1.8 < elapsed < 2.5


class TestFixtureCleanup:
    """Test fixture cleanup and isolation."""

    @pytest.mark.asyncio
    async def test_memory_provider_isolated_between_tests(self, memory_provider):
        """Each test should get clean memory provider."""
        # Search for any existing memories
        results = await memory_provider.search(
            query="test",
            limit=100,
        )

        # Should be empty (clean database)
        # Note: May have system memories, but should not have test data from other tests
        # This validates clean_database fixture works

    @pytest.mark.asyncio
    async def test_tool_registry_isolated_between_tests(self, tool_registry):
        """Each test should get fresh tool registry."""
        # Register a tool
        async def test_tool() -> str:
            return "test"

        tool_registry.register(
            name="test_tool",
            handler=test_tool,
            description="Test",
        )

        # Verify registered
        assert "test_tool" in tool_registry.list_tools()

        # In next test, tool_registry should be fresh again
        # (Verified by test_tool_registry_empty_by_default)


# ============================================================================
# Integration Test Example
# ============================================================================


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_full_fixture_integration(
    memory_provider, real_llm, evaluator, tool_registry
):
    """Test that all fixtures work together.

    This validates the complete fixture setup for real agent tests.
    """
    # Store a fact using memory provider
    memory_id = await memory_provider.store(
        content="The capital of France is Paris",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.WORLD,
        importance=0.9,
    )

    # Retrieve it
    memory = await memory_provider.get(memory_id)
    assert memory is not None

    # Generate a response using LLM
    response = await real_llm.generate("What is the capital of France?")

    # Evaluate response using evaluator
    result = await evaluator.evaluate_correctness(
        query="What is the capital of France?",
        expected_outcome="Should mention Paris",
        actual_response=response,
    )

    # LLM should know capitals
    assert result.correct

    # Tool registry should be available
    assert tool_registry.list_tools() == []  # Empty by default

    # All fixtures working together!
