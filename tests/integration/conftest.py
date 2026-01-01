"""
Integration test fixtures for FR-009 testing framework.

Provides fixtures for:
- TestDatabase (Neo4j lifecycle)
- Memory providers (real Neo4j)
- Seed applicators
- LLM-as-judge evaluators
- Mock LLM for tests that don't need real LLM

Requirements:
- Running Neo4j instance (default: bolt://localhost:7687)
- neo4j package installed
- Optional: GROQ_API_KEY or OPENAI_API_KEY for real LLM tests

Environment Variables:
- NEO4J_TEST_URI: Neo4j connection URI
- NEO4J_TEST_USER: Neo4j username
- NEO4J_TEST_PASSWORD: Neo4j password
- NEO4J_TEST_DATABASE: Neo4j database name
- GROQ_API_KEY: Groq API key for real LLM
- OPENAI_API_KEY: OpenAI API key for real LLM
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

# Check for neo4j
try:
    from neo4j import AsyncGraphDatabase

    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    AsyncGraphDatabase = None


from draagon_ai.testing import (
    TestDatabase,
    TestDatabaseConfig,
    SeedFactory,
    SeedApplicator,
    AgentEvaluator,
    create_test_database,
)

if TYPE_CHECKING:
    from draagon_ai.testing.evaluation import LLMProvider


# =============================================================================
# Configuration
# =============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Test markers for integration tests
pytestmark = [
    pytest.mark.skipif(
        not NEO4J_AVAILABLE,
        reason="neo4j package not installed",
    ),
    pytest.mark.integration,
]


# =============================================================================
# Mock LLM Provider
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for tests that don't need real LLM calls.

    Returns canned responses based on query content. Useful for testing
    framework mechanics without LLM costs.
    """

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 500,
    ) -> str:
        """Return mock response based on query content.

        The mock LLM returns XML-formatted evaluation responses that
        match what the AgentEvaluator expects.
        """
        # Extract the last user message
        user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_msg = m.get("content", "")
                break

        # For quality evaluation (coherence/helpfulness) - check first as these
        # also contain <evaluation> tag but need different response format
        if "coherence" in user_msg.lower() or "helpfulness" in user_msg.lower():
            return """<result>
  <score>0.85</score>
  <reasoning>The response is well-structured and helpful.</reasoning>
</result>"""

        # For correctness evaluation prompts
        if "<evaluation>" in user_msg:
            return """<result>
  <correct>true</correct>
  <reasoning>The response correctly addresses the expected outcome.</reasoning>
  <confidence>0.9</confidence>
</result>"""

        # Default response for agent interactions
        if "birthday" in user_msg.lower():
            return "Based on my records, your birthday is March 15."
        elif "cat" in user_msg.lower():
            return "You have 3 cats: Whiskers, Mittens, and Shadow."
        else:
            return "I understand. How can I help you further?"


# =============================================================================
# LLM Provider Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def mock_llm() -> MockLLMProvider:
    """Create mock LLM provider for framework testing."""
    return MockLLMProvider()


@pytest.fixture(scope="session")
def llm_provider(mock_llm) -> MockLLMProvider:
    """LLM provider for evaluation.

    Returns mock LLM by default. To use real LLM, set GROQ_API_KEY
    or OPENAI_API_KEY environment variable.
    """
    # For now, always use mock LLM
    # Real LLM integration can be added when needed
    return mock_llm


# =============================================================================
# Database Fixtures
# =============================================================================


async def check_neo4j_connection() -> bool:
    """Check if Neo4j is accessible."""
    if not NEO4J_AVAILABLE:
        return False

    config = TestDatabaseConfig()
    try:
        driver = AsyncGraphDatabase.driver(
            config.neo4j_uri,
            auth=(config.neo4j_username, config.neo4j_password),
        )
        async with driver.session(database=config.database_name) as session:
            await session.run("RETURN 1")
        await driver.close()
        return True
    except Exception as e:
        print(f"Neo4j connection check failed: {e}")
        return False


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def neo4j_available():
    """Check Neo4j connectivity once per session."""
    available = await check_neo4j_connection()
    if not available:
        config = TestDatabaseConfig()
        pytest.skip(f"Neo4j not accessible at {config.neo4j_uri}")
    return True


@pytest.fixture(scope="session")
async def test_database(neo4j_available) -> TestDatabase:
    """Session-scoped Neo4j test database lifecycle manager.

    Initializes once per test session and cleans up at the end.
    """
    db = await create_test_database()
    yield db
    await db.close()


@pytest.fixture
async def clean_database(test_database: TestDatabase) -> TestDatabase:
    """Function-scoped clean database.

    Clears all data before each test for a clean slate.
    Uses BEFORE cleanup policy (preserves state for debugging after).
    """
    await test_database.clear()
    await test_database.verify_connection()
    yield test_database


# =============================================================================
# Seed Fixtures
# =============================================================================


@pytest.fixture
def seed_factory() -> SeedFactory:
    """Create fresh SeedFactory for each test.

    Instance-based registry ensures test isolation.
    """
    return SeedFactory()


@pytest.fixture
def seed(seed_factory: SeedFactory) -> SeedApplicator:
    """Create seed applicator for the test.

    Usage:
        await seed.apply(MY_SEED_SET, memory_provider)
    """
    return SeedApplicator(factory=seed_factory)


# =============================================================================
# Evaluator Fixtures
# =============================================================================


@pytest.fixture
def evaluator(llm_provider) -> AgentEvaluator:
    """Create LLM-as-judge evaluator.

    Uses mock LLM by default for predictable tests.
    """
    return AgentEvaluator(llm=llm_provider, max_retries=1)


# =============================================================================
# Marker Configuration
# =============================================================================


def pytest_configure(config):
    """Register custom markers for integration tests."""
    config.addinivalue_line(
        "markers",
        "memory_integration: Tests that require Neo4j memory provider",
    )
    config.addinivalue_line(
        "markers",
        "learning_integration: Tests that verify agent learning",
    )
    config.addinivalue_line(
        "markers",
        "sequence_test: Multi-step test sequence",
    )
    config.addinivalue_line(
        "markers",
        "smoke: Critical integration tests that must pass",
    )
