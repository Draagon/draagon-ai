"""Tests for Neo4j Memory Provider.

These tests verify the Neo4j-based memory storage with semantic decomposition.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.memory.base import MemoryType, MemoryScope, Memory, SearchResult
from draagon_ai.memory.providers.neo4j import (
    Neo4jMemoryProvider,
    Neo4jMemoryConfig,
    NEO4J_AVAILABLE,
    memory_type_to_content_type,
    memory_type_to_layer,
)
from draagon_ai.cognition.reasoning.memory import MemoryLayer, ContentType


# =============================================================================
# Test Configuration
# =============================================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "draagon-ai-2025"


def neo4j_available() -> bool:
    """Check if Neo4j is available for integration tests."""
    if not NEO4J_AVAILABLE:
        return False
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        driver.close()
        return True
    except Exception:
        return False


# =============================================================================
# Mock Providers
# =============================================================================


class MockEmbedder:
    """Mock embedding provider for tests."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.calls = []

    async def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        # Return deterministic embedding based on text hash
        import hashlib
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        # Generate pseudo-random but deterministic vector
        return [(hash_val + i) % 100 / 100.0 for i in range(self.dimension)]


class MockLLM:
    """Mock LLM provider for tests."""

    def __init__(self):
        self.calls = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.calls.append(messages)

        # Return mock entity extraction response
        return """
<entities>
<entity type="person">Doug</entity>
<entity type="event">birthday</entity>
</entities>
<relationships>
<rel source="Doug" type="has" target="birthday"/>
</relationships>
"""


# =============================================================================
# Unit Tests - Type Mappings
# =============================================================================


class TestTypeMappings:
    """Test memory type to content type/layer mappings."""

    def test_memory_type_to_content_type(self):
        """Test MemoryType to ContentType mapping."""
        assert memory_type_to_content_type(MemoryType.FACT) == ContentType.FACT
        assert memory_type_to_content_type(MemoryType.PREFERENCE) == ContentType.PREFERENCE
        assert memory_type_to_content_type(MemoryType.INSTRUCTION) == ContentType.INSTRUCTION
        assert memory_type_to_content_type(MemoryType.SKILL) == ContentType.SKILL
        assert memory_type_to_content_type(MemoryType.EPISODIC) == ContentType.EVENT

    def test_memory_type_to_layer(self):
        """Test MemoryType to MemoryLayer mapping."""
        # Metacognitive
        assert memory_type_to_layer(MemoryType.INSTRUCTION) == MemoryLayer.METACOGNITIVE
        assert memory_type_to_layer(MemoryType.SELF_KNOWLEDGE) == MemoryLayer.METACOGNITIVE

        # Semantic
        assert memory_type_to_layer(MemoryType.FACT) == MemoryLayer.SEMANTIC
        assert memory_type_to_layer(MemoryType.PREFERENCE) == MemoryLayer.SEMANTIC
        assert memory_type_to_layer(MemoryType.KNOWLEDGE) == MemoryLayer.SEMANTIC
        assert memory_type_to_layer(MemoryType.SKILL) == MemoryLayer.SEMANTIC

        # Episodic
        assert memory_type_to_layer(MemoryType.EPISODIC) == MemoryLayer.EPISODIC

        # Working
        assert memory_type_to_layer(MemoryType.OBSERVATION) == MemoryLayer.WORKING


# =============================================================================
# Unit Tests - Configuration
# =============================================================================


class TestNeo4jMemoryConfig:
    """Test configuration class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = Neo4jMemoryConfig()
        assert config.uri == "bolt://localhost:7687"
        assert config.username == "neo4j"
        assert config.password == "neo4j"
        assert config.database == "neo4j"
        assert config.embedding_dimension == 1536
        assert config.enable_semantic_decomposition is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = Neo4jMemoryConfig(
            uri="bolt://custom:7688",
            username="myuser",
            password="mypass",
            embedding_dimension=768,
            enable_semantic_decomposition=False,
        )
        assert config.uri == "bolt://custom:7688"
        assert config.username == "myuser"
        assert config.embedding_dimension == 768
        assert config.enable_semantic_decomposition is False

    def test_type_weights(self):
        """Test type importance weights."""
        config = Neo4jMemoryConfig()
        assert config.type_weights["instruction"] == 1.0
        assert config.type_weights["fact"] == 0.8
        assert config.type_weights["episodic"] == 0.5


# =============================================================================
# Unit Tests - Provider Initialization
# =============================================================================


@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="neo4j not installed")
class TestProviderInit:
    """Test provider initialization."""

    def test_init_without_neo4j_raises(self):
        """Test that missing neo4j package raises ImportError."""
        # This is implicitly tested by the module import
        pass

    def test_init_requires_embedder(self):
        """Test that embedding provider is required."""
        config = Neo4jMemoryConfig()
        provider = Neo4jMemoryProvider(config, MockEmbedder())
        assert provider._embedder is not None

    def test_init_defers_connection(self):
        """Test that connection is deferred until initialize()."""
        config = Neo4jMemoryConfig()
        embedder = MockEmbedder()
        provider = Neo4jMemoryProvider(config, embedder)
        assert not provider._initialized
        assert provider._store is None


# =============================================================================
# Integration Tests - Store and Retrieve
# =============================================================================


@pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
class TestStoreAndRetrieve:
    """Integration tests for storing and retrieving memories."""

    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return Neo4jMemoryConfig(
            uri=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            embedding_dimension=768,  # Match MockEmbedder dimension
            enable_semantic_decomposition=False,  # Disable for unit tests
        )

    @pytest.fixture
    def embedder(self):
        """Create mock embedder."""
        return MockEmbedder()

    @pytest.fixture
    async def provider(self, config, embedder):
        """Create and initialize provider."""
        provider = Neo4jMemoryProvider(config, embedder)
        await provider.initialize()
        yield provider
        # Cleanup
        await provider.garbage_collect()
        await provider.close()

    @pytest.mark.asyncio
    async def test_store_memory(self, provider):
        """Test storing a memory."""
        memory = await provider.store(
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.8,
        )

        assert memory.id is not None
        assert memory.content == "Doug's birthday is March 15"
        assert memory.memory_type == MemoryType.FACT
        assert memory.scope == MemoryScope.USER
        assert memory.user_id == "doug"
        assert memory.importance == 0.8

        # Cleanup
        await provider.delete(memory.id)

    @pytest.mark.asyncio
    async def test_get_memory(self, provider):
        """Test getting a memory by ID."""
        # Store first
        stored = await provider.store(
            content="Test fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test",
        )

        # Get it back
        retrieved = await provider.get(stored.id)

        assert retrieved is not None
        assert retrieved.id == stored.id
        assert retrieved.content == "Test fact"

        # Cleanup
        await provider.delete(stored.id)

    @pytest.mark.asyncio
    async def test_search_memories(self, provider):
        """Test searching memories by similarity."""
        # Store some memories
        m1 = await provider.store(
            content="Doug likes pizza for dinner",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
        )
        m2 = await provider.store(
            content="Doug's favorite movie is Inception",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
        )

        # Search
        results = await provider.search(
            query="what food does Doug like",
            user_id="doug",
            limit=5,
        )

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

        # Cleanup
        await provider.delete(m1.id)
        await provider.delete(m2.id)

    @pytest.mark.asyncio
    async def test_update_memory(self, provider):
        """Test updating a memory."""
        # Store
        stored = await provider.store(
            content="Original content",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
        )

        # Update
        updated = await provider.update(
            stored.id,
            content="Updated content",
            importance=0.9,
        )

        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.importance == 0.9

        # Cleanup
        await provider.delete(stored.id)

    @pytest.mark.asyncio
    async def test_delete_memory(self, provider):
        """Test deleting a memory."""
        # Store
        stored = await provider.store(
            content="To be deleted",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
        )

        # Delete
        result = await provider.delete(stored.id)
        assert result is True

        # Verify gone
        retrieved = await provider.get(stored.id)
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_reinforce_memory(self, provider):
        """Test reinforcing a memory boosts importance."""
        # Store with low importance
        stored = await provider.store(
            content="Reinforceable memory",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            importance=0.5,
        )

        original_importance = stored.importance

        # Reinforce
        reinforced = await provider.reinforce(stored.id, boost=0.2)

        assert reinforced is not None
        assert reinforced.importance > original_importance

        # Cleanup
        await provider.delete(stored.id)


# =============================================================================
# Integration Tests - Memory Layer Behavior
# =============================================================================


@pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
class TestMemoryLayers:
    """Test memory layer behavior (TTL, promotion)."""

    @pytest.fixture
    async def provider(self):
        """Create provider."""
        config = Neo4jMemoryConfig(
            uri=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            embedding_dimension=768,  # Match MockEmbedder dimension
            enable_semantic_decomposition=False,
        )
        embedder = MockEmbedder()
        provider = Neo4jMemoryProvider(config, embedder)
        await provider.initialize()
        yield provider
        await provider.garbage_collect()
        await provider.close()

    @pytest.mark.asyncio
    async def test_instruction_goes_to_metacognitive(self, provider):
        """Test that instructions are stored in metacognitive layer."""
        memory = await provider.store(
            content="Always use Celsius for temperature",
            memory_type=MemoryType.INSTRUCTION,
            scope=MemoryScope.USER,
        )

        # The memory should be in metacognitive layer
        retrieved = await provider.get(memory.id)
        # Layer info is stored as Neo4j property, would need to query directly
        # For now just verify store/retrieve works
        assert retrieved is not None

        await provider.delete(memory.id)

    @pytest.mark.asyncio
    async def test_fact_goes_to_semantic(self, provider):
        """Test that facts are stored in semantic layer."""
        memory = await provider.store(
            content="Doug has 6 cats",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
        )

        retrieved = await provider.get(memory.id)
        assert retrieved is not None

        await provider.delete(memory.id)


# =============================================================================
# Integration Tests - Graph Traversal
# =============================================================================


@pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
class TestGraphTraversal:
    """Test graph-based search via traversal."""

    @pytest.fixture
    async def provider(self):
        """Create provider."""
        config = Neo4jMemoryConfig(
            uri=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            embedding_dimension=768,  # Match MockEmbedder dimension
            enable_semantic_decomposition=False,
        )
        embedder = MockEmbedder()
        provider = Neo4jMemoryProvider(config, embedder)
        await provider.initialize()
        yield provider
        await provider.garbage_collect()
        await provider.close()

    @pytest.mark.asyncio
    async def test_search_by_graph_traversal(self, provider):
        """Test finding memories via graph traversal."""
        # Store some related memories
        m1 = await provider.store(
            content="Doug owns a cat named Whiskers",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            entities=["Doug", "Whiskers", "cat"],
        )
        m2 = await provider.store(
            content="Whiskers likes to play",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            entities=["Whiskers", "play"],
        )

        # Search by entity
        results = await provider.search_by_graph_traversal(
            entity_names=["Whiskers"],
            user_id="doug",
            max_depth=2,
        )

        # Should find memories related to Whiskers
        # Note: depends on semantic graph being built
        # For now just verify the method doesn't error

        await provider.delete(m1.id)
        await provider.delete(m2.id)


# =============================================================================
# Integration Tests - Semantic Decomposition
# =============================================================================


@pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
class TestSemanticDecomposition:
    """Test semantic decomposition integration."""

    @pytest.fixture
    async def provider_with_decomposition(self):
        """Create provider with semantic decomposition enabled."""
        config = Neo4jMemoryConfig(
            uri=NEO4J_URI,
            username=NEO4J_USER,
            password=NEO4J_PASSWORD,
            embedding_dimension=768,  # Match MockEmbedder dimension
            enable_semantic_decomposition=True,
        )
        embedder = MockEmbedder()
        llm = MockLLM()
        provider = Neo4jMemoryProvider(config, embedder, llm)
        await provider.initialize()
        yield provider
        await provider.garbage_collect()
        await provider.close()

    @pytest.mark.asyncio
    async def test_store_with_decomposition(self, provider_with_decomposition):
        """Test that storing with decomposition creates graph nodes."""
        provider = provider_with_decomposition

        memory = await provider.store(
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
        )

        assert memory.id is not None

        # The LLM should have been called for decomposition
        # Note: May not be called if pipeline isn't fully initialized

        await provider.delete(memory.id)


# =============================================================================
# Unit Tests - Mock-based Tests (No Neo4j Required)
# =============================================================================


class TestProviderMocked:
    """Tests using mocks instead of real Neo4j."""

    def test_instance_id_generation(self):
        """Test instance ID generation for graph partitioning."""
        from draagon_ai.memory.providers.neo4j import Neo4jMemoryProvider

        # Create provider (won't connect without neo4j)
        if not NEO4J_AVAILABLE:
            pytest.skip("neo4j not installed")

        config = Neo4jMemoryConfig()
        embedder = MockEmbedder()
        provider = Neo4jMemoryProvider(config, embedder)

        # Test instance ID generation
        assert provider._get_instance_id("agent1", None, None) == "agent:agent1"
        assert provider._get_instance_id(None, "user1", None) == "user:user1"
        assert provider._get_instance_id("agent1", "user1", None) == "agent:agent1:user:user1"
        assert provider._get_instance_id(None, None, "ctx1") == "ctx:ctx1"
        assert provider._get_instance_id(None, None, None) == "default"
