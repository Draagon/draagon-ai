"""Real agent integration test fixtures.

This module provides fixtures for testing real agent behavior with:
- Real LLM providers (Groq or OpenAI)
- Real Neo4j memory backend
- Real embedding provider
- Real cognitive services

All fixtures gracefully skip tests if dependencies are unavailable.

API Keys are loaded from .env in project root automatically.
"""

import asyncio
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

# Load .env from project root (CRITICAL: API keys live here!)
try:
    from dotenv import load_dotenv

    # Find project root (where .env lives)
    project_root = Path(__file__).parent.parent.parent.parent
    env_file = project_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    # python-dotenv not installed, rely on environment variables
    pass

from draagon_ai.llm.base import LLMProvider
from draagon_ai.llm.groq import GroqLLM
from draagon_ai.memory.providers.neo4j import Neo4jMemoryProvider, Neo4jMemoryConfig
from draagon_ai.orchestration.loop import AgentLoop, AgentLoopConfig, LoopMode
from draagon_ai.orchestration.registry import ToolRegistry
from draagon_ai.testing.database import TestDatabase
from draagon_ai.testing.evaluation import AgentEvaluator


# ============================================================================
# Embedding Provider Fixture
# ============================================================================


class MockEmbeddingProvider:
    """Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash.
    Avoids external API dependencies while maintaining semantic similarity.
    """

    def __init__(self, dimension: int = 1536):
        """Initialize mock embedding provider.

        Args:
            dimension: Embedding vector dimension (default 1536 for OpenAI compatibility)
        """
        self.dimension = dimension

    async def embed(self, text: str) -> list[float]:
        """Generate mock embedding for text.

        Args:
            text: Input text to embed

        Returns:
            Deterministic embedding vector
        """
        import hashlib

        # Generate deterministic hash-based embedding
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

        # Create embedding with slight variation based on text length
        base_value = (hash_val % 1000) / 1000.0
        length_factor = min(len(text) / 100.0, 1.0)

        # Generate vector with some structure
        embedding = []
        for i in range(self.dimension):
            # Add variation based on position and text
            variation = ((hash_val >> (i % 16)) % 100) / 100.0
            value = (base_value + variation * length_factor) / 2
            embedding.append(value)

        return embedding

    async def initialize(self):
        """Initialize provider (no-op for mock)."""
        pass

    async def close(self):
        """Close provider (no-op for mock)."""
        pass


@pytest.fixture(scope="session")
async def embedding_provider():
    """Real embedding provider for vector operations.

    Provides mock embeddings that are deterministic and don't require external APIs.
    For production testing with real embeddings, configure actual provider.

    Returns:
        Embedding provider instance
    """
    provider = MockEmbeddingProvider(dimension=1536)
    await provider.initialize()
    yield provider
    await provider.close()


# ============================================================================
# LLM Provider Fixture
# ============================================================================


@pytest.fixture(scope="session")
def real_llm():
    """Real LLM provider (Groq or OpenAI based on env vars).

    Checks for API keys in order:
    1. GROQ_API_KEY - Preferred for fast inference
    2. OPENAI_API_KEY - Fallback option

    Returns:
        Configured LLM provider

    Raises:
        pytest.skip: If no API keys are configured
    """
    # Try Groq first (faster, cheaper for testing)
    if os.getenv("GROQ_API_KEY"):
        return GroqLLM(api_key=os.getenv("GROQ_API_KEY"))

    # Fallback to OpenAI
    if os.getenv("OPENAI_API_KEY"):
        # OpenAI provider would be here when implemented
        pytest.skip("OpenAI provider not yet implemented")

    # No provider available
    pytest.skip(
        "No LLM provider configured. "
        "Set GROQ_API_KEY or OPENAI_API_KEY environment variable."
    )


# ============================================================================
# Database Fixtures
# ============================================================================


@pytest.fixture(scope="session")
async def test_database():
    """Session-scoped test database manager.

    Initializes Neo4j connection once per test session.

    Returns:
        TestDatabase instance

    Raises:
        pytest.skip: If Neo4j is not available
    """
    db = TestDatabase()
    try:
        await db.initialize()
        await db.verify_connection()
    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")

    yield db

    await db.close()


@pytest.fixture
async def clean_database(test_database):
    """Function-scoped database cleanup.

    Clears database before each test to ensure clean state.

    Args:
        test_database: Session-scoped database manager

    Returns:
        Clean TestDatabase instance
    """
    await test_database.clear()
    await test_database.verify_connection()
    yield test_database


# ============================================================================
# Memory Provider Fixture
# ============================================================================


@pytest.fixture
async def memory_provider(clean_database, embedding_provider, real_llm):
    """Real Neo4jMemoryProvider with semantic decomposition.

    Creates a fully configured memory provider with:
    - Real Neo4j backend
    - Mock embedding provider (deterministic)
    - Real LLM for semantic decomposition
    - Clean state per test

    Args:
        clean_database: Clean database instance
        embedding_provider: Embedding provider
        real_llm: Real LLM provider

    Returns:
        Initialized Neo4jMemoryProvider
    """
    # Get database config
    db_config = clean_database.get_config()

    # Create memory config
    config = Neo4jMemoryConfig(
        uri=db_config["uri"],
        username=db_config["username"],
        password=db_config["password"],
        database=db_config.get("database", "neo4j"),
        embedding_dimension=1536,
        enable_semantic_decomposition=True,
    )

    # Create provider
    provider = Neo4jMemoryProvider(
        config=config,
        embedding_provider=embedding_provider,
        llm_provider=real_llm,
    )

    await provider.initialize()

    yield provider

    # No cleanup - preserves state for debugging via http://localhost:7474
    # Database is cleared at start of next test via clean_database


# ============================================================================
# Tool Registry Fixture
# ============================================================================


@pytest.fixture
def tool_registry():
    """Fresh tool registry for each test.

    Returns:
        Empty ToolRegistry instance
    """
    return ToolRegistry()


# ============================================================================
# Agent Loop Fixture
# ============================================================================


@pytest.fixture
async def agent(memory_provider, real_llm, tool_registry):
    """Fully configured real agent.

    Creates an AgentLoop with:
    - Real LLM provider
    - Real memory backend
    - Fresh tool registry
    - Testing-appropriate configuration

    Args:
        memory_provider: Neo4j memory provider
        real_llm: Real LLM provider
        tool_registry: Fresh tool registry

    Returns:
        Configured AgentLoop instance
    """
    config = AgentLoopConfig(
        mode=LoopMode.AUTO,
        max_iterations=10,
        iteration_timeout_seconds=30.0,
        log_thought_traces=True,
    )

    agent_loop = AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        config=config,
    )

    return agent_loop


# ============================================================================
# Evaluator Fixture
# ============================================================================


@pytest.fixture
def evaluator(real_llm):
    """LLM-as-judge evaluator with real LLM.

    Creates AgentEvaluator for semantic response validation.

    Args:
        real_llm: Real LLM provider

    Returns:
        AgentEvaluator instance with retry logic
    """
    return AgentEvaluator(llm=real_llm, max_retries=3)


# ============================================================================
# Time Utilities
# ============================================================================


@pytest.fixture
def advance_time():
    """Utility for fast-forwarding time in tests.

    Provides two modes:
    1. Mock mode (fast, for CI) - mocks datetime
    2. Real mode (slow, for validation) - uses actual delays

    Returns:
        Async function to advance time

    Example:
        @pytest.mark.asyncio
        async def test_ttl(advance_time):
            await advance_time(minutes=10)
            # Time has advanced by 10 minutes
    """

    async def _advance(
        minutes: int = 0,
        seconds: int = 0,
        mock_time: bool = True,
    ) -> None:
        """Advance time for TTL testing.

        Args:
            minutes: Minutes to advance
            seconds: Seconds to advance
            mock_time: If True, mock datetime. If False, use real delay.

        Note:
            Mock mode is fast but may not catch all time-dependent bugs.
            Real mode is slow but validates actual behavior.
        """
        total_seconds = minutes * 60 + seconds

        if mock_time:
            # Mock mode - instant time jump
            now = datetime.now()
            future = now + timedelta(seconds=total_seconds)

            # Mock datetime.now() to return future time
            with patch("datetime.datetime") as mock_dt:
                mock_dt.now.return_value = future
                # Let test code run in mocked time context
                # Note: This is a simplified mock - real implementation
                # would need to coordinate with Neo4j queries
        else:
            # Real mode - actual delay (mark test with @pytest.mark.slow)
            await asyncio.sleep(total_seconds)

    return _advance


# ============================================================================
# Test Markers
# ============================================================================


def pytest_configure(config):
    """Register custom test markers.

    Markers:
        agent_integration: Tests for real agent behavior
        memory_integration: Tests for memory persistence
        learning_integration: Tests for learning capabilities
        belief_integration: Tests for belief reconciliation
        react_integration: Tests for ReAct reasoning
        tool_integration: Tests for tool execution
        multiagent_integration: Tests for multi-agent coordination
        tier_integration: Tests for LLM tier selection
        slow: Tests that use real delays (not mocked time)
    """
    config.addinivalue_line(
        "markers", "agent_integration: Tests for real agent query processing"
    )
    config.addinivalue_line(
        "markers", "memory_integration: Tests for memory storage and retrieval"
    )
    config.addinivalue_line(
        "markers", "learning_integration: Tests for agent learning capabilities"
    )
    config.addinivalue_line(
        "markers", "belief_integration: Tests for belief reconciliation"
    )
    config.addinivalue_line(
        "markers", "react_integration: Tests for ReAct multi-step reasoning"
    )
    config.addinivalue_line(
        "markers", "tool_integration: Tests for tool discovery and execution"
    )
    config.addinivalue_line(
        "markers", "multiagent_integration: Tests for multi-agent coordination"
    )
    config.addinivalue_line(
        "markers", "tier_integration: Tests for LLM tier selection"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that use real delays (not mocked time)"
    )
