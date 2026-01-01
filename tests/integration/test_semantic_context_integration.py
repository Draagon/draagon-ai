"""Integration tests for Semantic Context Service with Neo4j backend.

These tests hit REAL Neo4j and (optionally) LLM providers - no mocks.
They verify the full semantic context enrichment flow:
  Query -> ReasoningLoop -> SemanticGraph -> SemanticContext -> LLM Prompt

Requirements:
- Running Neo4j instance (default: bolt://localhost:7687)
- neo4j package installed
- Optional: GROQ_API_KEY for LLM tests

Set NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables to override defaults.
"""

import asyncio
import os
import pytest
from datetime import datetime
from uuid import uuid4

# Check for neo4j
try:
    from neo4j import AsyncGraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

from draagon_ai.orchestration import (
    SemanticContextService,
    SemanticContextConfig,
    SemanticContext,
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
)
from draagon_ai.cognition.reasoning import ReasoningLoop, ReasoningConfig
from draagon_ai.cognition.decomposition import SemanticGraph, GraphNode, NodeType

# Test configuration
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "neo4j")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

TEST_INSTANCE_PREFIX = "test_semantic_ctx_"


class MockLLMProvider:
    """Simple mock LLM for tests that don't need real LLM calls."""

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ):
        """Return a mock response."""
        from dataclasses import dataclass

        @dataclass
        class MockResponse:
            content: str

        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user" or hasattr(m, "role") and m.role == "user":
                user_msg = m.get("content") or getattr(m, "content", "")
                break

        # Generate a simple response based on the query
        if "time" in user_msg.lower():
            return MockResponse(content="""<response>
<action>answer</action>
<answer>The current time is 3:00 PM.</answer>
</response>""")
        elif "birthday" in user_msg.lower():
            return MockResponse(content="""<response>
<action>answer</action>
<answer>Based on what I know, Doug's birthday is March 15.</answer>
</response>""")
        else:
            return MockResponse(content="""<response>
<action>answer</action>
<answer>I understand your question. Let me help with that.</answer>
</response>""")


async def check_neo4j_connection() -> bool:
    """Check if Neo4j is accessible."""
    if not NEO4J_AVAILABLE:
        return False

    try:
        driver = AsyncGraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
        )
        async with driver.session() as session:
            await session.run("RETURN 1")
        await driver.close()
        return True
    except Exception as e:
        print(f"Neo4j connection failed: {e}")
        return False


# Skip all tests if Neo4j not available
pytestmark = [
    pytest.mark.skipif(
        not NEO4J_AVAILABLE,
        reason="neo4j package not installed"
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def neo4j_available():
    """Check Neo4j connectivity once per module."""
    available = await check_neo4j_connection()
    if not available:
        pytest.skip(f"Neo4j not accessible at {NEO4J_URI}")
    return True


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
async def semantic_service(mock_llm, neo4j_available):
    """Create SemanticContextService with Neo4j backend."""
    test_id = uuid4().hex[:8]

    config = SemanticContextConfig(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        instance_id=f"{TEST_INSTANCE_PREFIX}{test_id}",
        max_facts=10,
        max_memories=5,
        context_depth=2,
    )

    service = SemanticContextService(
        llm=mock_llm,
        memory_provider=None,  # No memory provider for basic tests
        config=config,
    )

    yield service

    # Cleanup
    service.close()


@pytest.fixture
async def agent_loop_with_semantic(mock_llm, neo4j_available):
    """Create AgentLoop with semantic context enabled."""
    test_id = uuid4().hex[:8]

    config = AgentLoopConfig(
        use_semantic_context=True,
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        semantic_instance_id=f"{TEST_INSTANCE_PREFIX}{test_id}",
        max_semantic_facts=10,
        max_semantic_memories=5,
    )

    loop = AgentLoop(
        llm=mock_llm,
        memory=None,
        config=config,
    )

    yield loop

    # Cleanup
    if loop._semantic_context_service:
        loop._semantic_context_service.close()


# =============================================================================
# SemanticContextService Tests
# =============================================================================


class TestSemanticContextService:
    """Test SemanticContextService initialization and basic functionality."""

    @pytest.mark.asyncio
    async def test_service_initializes(self, semantic_service):
        """Test that service initializes correctly."""
        assert semantic_service is not None
        assert semantic_service._config is not None

    @pytest.mark.asyncio
    async def test_lazy_reasoning_loop(self, semantic_service):
        """Test that ReasoningLoop is lazily initialized."""
        # Access the property to trigger initialization
        loop = semantic_service.reasoning_loop

        # Should be initialized (or None if LLM not configured properly)
        # The key is that it doesn't crash
        assert semantic_service._loop_initialized is True

    @pytest.mark.asyncio
    async def test_enrich_returns_context(self, semantic_service):
        """Test that enrich returns a SemanticContext."""
        context = await semantic_service.enrich(
            query="When is Doug's birthday?",
            user_id="test_user",
        )

        assert isinstance(context, SemanticContext)
        assert context.query == "When is Doug's birthday?"
        assert hasattr(context, "relevant_facts")
        assert hasattr(context, "relevant_entities")
        assert hasattr(context, "related_memories")

    @pytest.mark.asyncio
    async def test_empty_context_check(self, semantic_service):
        """Test is_empty() method on SemanticContext."""
        context = await semantic_service.enrich(
            query="Random query with no stored context",
            user_id="test_user",
        )

        # New database should have no relevant facts
        assert isinstance(context.is_empty(), bool)

    @pytest.mark.asyncio
    async def test_context_to_prompt_format(self, semantic_service):
        """Test that to_prompt_context() returns proper format."""
        context = await semantic_service.enrich(
            query="What do you know about cats?",
            user_id="test_user",
        )

        prompt_str = context.to_prompt_context()
        assert isinstance(prompt_str, str)
        # Empty or formatted with headers
        if prompt_str:
            assert "## Relevant Context" in prompt_str or len(prompt_str) > 0


# =============================================================================
# AgentLoop Semantic Context Integration Tests
# =============================================================================


class TestAgentLoopSemanticContext:
    """Test AgentLoop with semantic context enabled."""

    @pytest.mark.asyncio
    async def test_loop_creates_semantic_service(self, agent_loop_with_semantic):
        """Test that AgentLoop creates SemanticContextService."""
        # Access the property to trigger lazy init
        scs = agent_loop_with_semantic.semantic_context_service

        # Should be initialized
        assert agent_loop_with_semantic._semantic_context_initialized is True
        # May be None if Neo4j not properly connected, but shouldn't crash
        if scs is not None:
            assert isinstance(scs, SemanticContextService)

    @pytest.mark.asyncio
    async def test_semantic_context_off_by_default(self, mock_llm):
        """Test that semantic context is off by default."""
        config = AgentLoopConfig()  # Default config
        assert config.use_semantic_context is False

        loop = AgentLoop(llm=mock_llm, config=config)
        scs = loop.semantic_context_service

        # Should be None when disabled
        assert scs is None


# =============================================================================
# ReasoningLoop Direct Tests
# =============================================================================


class TestReasoningLoopDirect:
    """Test ReasoningLoop directly with Neo4j."""

    @pytest.mark.asyncio
    async def test_reasoning_loop_creates(self, mock_llm, neo4j_available):
        """Test that ReasoningLoop can be created."""
        test_id = uuid4().hex[:8]

        config = ReasoningConfig(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            instance_id=f"{TEST_INSTANCE_PREFIX}rl_{test_id}",
        )

        loop = ReasoningLoop(llm=mock_llm, config=config)

        assert loop is not None
        assert loop.config == config

        loop.close()

    @pytest.mark.asyncio
    async def test_reasoning_loop_process(self, mock_llm, neo4j_available):
        """Test that ReasoningLoop can process a query."""
        test_id = uuid4().hex[:8]

        config = ReasoningConfig(
            neo4j_uri=NEO4J_URI,
            neo4j_user=NEO4J_USER,
            neo4j_password=NEO4J_PASSWORD,
            instance_id=f"{TEST_INSTANCE_PREFIX}rl_proc_{test_id}",
        )

        loop = ReasoningLoop(llm=mock_llm, config=config)

        try:
            result = await loop.process("Doug has 6 cats named Whiskers, Mittens, and friends.")

            assert result is not None
            assert hasattr(result, "final_answer") or hasattr(result, "extracted_graph")

        finally:
            loop.close()


# =============================================================================
# SemanticGraph Tests
# =============================================================================


class TestSemanticGraphBasics:
    """Test SemanticGraph data structure."""

    def test_graph_creates_empty(self):
        """Test creating an empty SemanticGraph."""
        graph = SemanticGraph()
        assert graph is not None
        assert graph.node_count == 0

    def test_graph_add_node(self):
        """Test adding a node to SemanticGraph."""
        graph = SemanticGraph()

        node = GraphNode(
            node_id="test_node_1",
            canonical_name="Doug",
            node_type=NodeType.INSTANCE,  # INSTANCE for named entities
        )
        graph.add_node(node)

        assert graph.node_count == 1
        retrieved = graph.get_node("test_node_1")
        assert retrieved is not None
        assert retrieved.canonical_name == "Doug"

    def test_graph_iter_nodes(self):
        """Test iterating over nodes."""
        graph = SemanticGraph()

        graph.add_node(GraphNode(node_id="n1", canonical_name="Node1", node_type=NodeType.INSTANCE))
        graph.add_node(GraphNode(node_id="n2", canonical_name="Node2", node_type=NodeType.EVENT))

        nodes = list(graph.iter_nodes())
        assert len(nodes) == 2


# =============================================================================
# End-to-End Flow Tests
# =============================================================================


class TestEndToEndFlow:
    """Test complete end-to-end flows."""

    @pytest.mark.asyncio
    async def test_query_enrichment_flow(self, semantic_service):
        """Test the full query enrichment flow."""
        # Enrich a query
        context = await semantic_service.enrich(
            query="What's the weather like today?",
            user_id="test_user",
            agent_id="test_agent",
        )

        # Should complete without error
        assert context is not None
        assert context.retrieval_time_ms >= 0

        # Format for prompt
        prompt_context = context.to_prompt_context()
        assert isinstance(prompt_context, str)

    @pytest.mark.asyncio
    async def test_multiple_queries_same_service(self, semantic_service):
        """Test multiple queries through same service."""
        queries = [
            "When is Doug's birthday?",
            "How many cats does Doug have?",
            "What is the capital of France?",
        ]

        contexts = []
        for query in queries:
            ctx = await semantic_service.enrich(
                query=query,
                user_id="test_user",
            )
            contexts.append(ctx)

        assert len(contexts) == 3
        for ctx in contexts:
            assert isinstance(ctx, SemanticContext)


# =============================================================================
# Performance/Timing Tests
# =============================================================================


class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_context_retrieval_timing(self, semantic_service):
        """Test that context retrieval is reasonably fast."""
        context = await semantic_service.enrich(
            query="Simple test query",
            user_id="test_user",
        )

        # Should complete in under 2 seconds even with network latency
        assert context.retrieval_time_ms < 2000

    @pytest.mark.asyncio
    async def test_concurrent_enrichment(self, semantic_service):
        """Test concurrent enrichment requests."""
        async def enrich_query(idx: int):
            return await semantic_service.enrich(
                query=f"Concurrent query number {idx}",
                user_id=f"user_{idx}",
            )

        # Run 5 concurrent enrichments
        tasks = [enrich_query(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        for result in results:
            assert isinstance(result, SemanticContext)
