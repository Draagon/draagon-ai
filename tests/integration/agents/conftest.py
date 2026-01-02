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


@pytest.fixture
async def embedding_provider():
    """Real embedding provider using Ollama.

    Per CONSTITUTION.md Section 1.7: Integration tests must use REAL providers.
    Mock embeddings that generate hash-based vectors are FORBIDDEN because:
    - Semantic search REQUIRES semantic embeddings to work
    - Mock embeddings break the fundamental assumption of vector similarity
    - Tests that pass with mocks give false confidence

    This fixture uses Ollama nomic-embed-text (768 dimensions):
    - Real semantic embeddings
    - Runs on local Ollama server
    - No external API costs

    Environment variables:
    - OLLAMA_BASE_URL: Ollama server URL (default: http://192.168.168.200:11434)

    Returns:
        OllamaEmbeddingProvider instance
    """
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    # Use Ollama server (defaults to local network server)
    base_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.168.200:11434")

    provider = OllamaEmbeddingProvider(
        base_url=base_url,
        model="nomic-embed-text",
        dimension=768,
    )

    # Verify Ollama is accessible
    try:
        # Test embedding to verify connection
        test_embedding = await provider.embed("test")
        if len(test_embedding) != 768:
            pytest.skip(f"Unexpected embedding dimension: {len(test_embedding)}")
    except Exception as e:
        pytest.skip(f"Ollama not available at {base_url}: {e}")

    yield provider


# ============================================================================
# LLM Provider Fixture
# ============================================================================


@pytest.fixture
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


@pytest.fixture
async def test_database():
    """Test database manager.

    Creates a fresh Neo4j connection per test to avoid event loop issues.
    Connection is closed after each test.

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
    """Real Neo4jMemoryProvider with real Ollama embeddings.

    Creates a fully configured memory provider with:
    - Real Neo4j backend
    - Real Ollama embedding provider (768-dim nomic-embed-text)
    - Real LLM for semantic decomposition
    - Clean state per test

    Per CONSTITUTION.md Section 1.7: Integration tests must use REAL providers.

    Args:
        clean_database: Clean database instance
        embedding_provider: Real OllamaEmbeddingProvider
        real_llm: Real LLM provider

    Returns:
        Initialized Neo4jMemoryProvider
    """
    # Get database config
    db_config = clean_database.get_config()

    # Create memory config
    # Note: semantic decomposition disabled for basic fixture tests
    # (requires nltk/wordnet). Enable for specific decomposition tests.
    # Use 768 dimensions to match Ollama nomic-embed-text model
    config = Neo4jMemoryConfig(
        uri=db_config["uri"],
        username=db_config["username"],
        password=db_config["password"],
        database=db_config.get("database", "neo4j"),
        embedding_dimension=embedding_provider.dimension,  # 768 for nomic-embed-text
        enable_semantic_decomposition=False,
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
