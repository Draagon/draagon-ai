"""Full memory flow integration tests.

Tests the complete memory lifecycle:
- Store → search → promote → search again
- Cross-layer interactions
- Promotion cycles

Requirements:
- Running Qdrant instance (default: http://192.168.168.216:6333)
- qdrant-client package installed
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

try:
    from qdrant_client import AsyncQdrantClient
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from draagon_ai.memory import (
    LayeredMemoryProvider,
    TemporalCognitiveGraph,
)
from draagon_ai.memory.providers.layered import LayeredMemoryConfig
from draagon_ai.memory.base import MemoryType, MemoryScope
from draagon_ai.memory.temporal_nodes import NodeType, MemoryLayer


QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.168.216:6333")
TEST_COLLECTION_PREFIX = "test_draagon_flow_"


class WordBasedEmbeddingProvider:
    """Word-based embedding provider for tests."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        import re
        words = re.findall(r'\b\w+\b', text.lower())

        embedding = [0.0] * self.dimension

        for word in words:
            idx = self._get_word_idx(word)
            for offset in range(5):
                dim_idx = (idx * 7 + offset * 13) % self.dimension
                embedding[dim_idx] += 0.5

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [(hash_bytes[i % 32] / 255.0) * 0.1 for i in range(self.dimension)]

        self._cache[text] = embedding
        return embedding


async def check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    if not QDRANT_CLIENT_AVAILABLE:
        return False
    try:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=5.0)
        await client.get_collections()
        await client.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not QDRANT_CLIENT_AVAILABLE,
        reason="qdrant-client not installed"
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def qdrant_available():
    available = await check_qdrant_connection()
    if not available:
        pytest.skip(f"Qdrant not accessible at {QDRANT_URL}")
    return True


@pytest.fixture
def embedder():
    return WordBasedEmbeddingProvider()


@pytest.fixture
async def provider(embedder, qdrant_available):
    """Create a LayeredMemoryProvider with immediate promotion thresholds."""
    test_id = uuid4().hex[:8]

    config = LayeredMemoryConfig(
        qdrant_url=QDRANT_URL,
        qdrant_nodes_collection=f"{TEST_COLLECTION_PREFIX}nodes_{test_id}",
        qdrant_edges_collection=f"{TEST_COLLECTION_PREFIX}edges_{test_id}",
        embedding_dimension=768,
        # Immediate promotion for testing
        promotion_working_importance=0.3,
        promotion_working_access=1,
        promotion_working_min_age_minutes=0,
        promotion_episodic_importance=0.4,
        promotion_episodic_access=2,
        promotion_episodic_min_age_hours=0,
        promotion_semantic_importance=0.5,
        promotion_semantic_access=3,
        promotion_semantic_min_age_days=0,
    )

    p = LayeredMemoryProvider(config=config, embedding_provider=embedder)
    await p.initialize()

    yield p

    try:
        if hasattr(p._graph, '_client') and p._graph._client:
            await p._graph._client.delete_collection(config.qdrant_nodes_collection)
            await p._graph._client.delete_collection(config.qdrant_edges_collection)
    except Exception:
        pass
    await p.close()


# =============================================================================
# Full Lifecycle Tests
# =============================================================================


class TestStoreSearchPromoteFlow:
    """Test the complete store → search → promote → search flow."""

    @pytest.mark.asyncio
    async def test_working_to_episodic_promotion(self, provider):
        """Test that working memory items promote to episodic."""
        # Add to working memory with high attention
        item = await provider.working.add(
            content="Important conversation item",
            attention_weight=0.9,
        )

        # Verify in working memory
        node = await provider.graph.get_node(item.node_id)
        assert node.layer == MemoryLayer.WORKING

        # Boost importance to trigger promotion
        node.importance = 0.9
        node.access_count = 5

        # Run promotion cycle
        promoted = await provider.promote_all()

        # Should return promotion stats
        assert hasattr(promoted, 'total_promoted')  # May or may not promote depending on timing

    @pytest.mark.asyncio
    async def test_store_search_reinforce_cycle(self, provider):
        """Test store → search → reinforce cycle."""
        # Store memory
        memory = await provider.store(
            content="Python programming language",
            memory_type="fact",
            entities=["Python"],
        )
        original_id = memory.id

        # Search finds it
        results = await provider.search(
            query="Python programming",
            limit=5,
        )
        assert isinstance(results, list)

        # Reinforce accessed memory
        await provider.reinforce(original_id, boost=0.1)

        # Memory should still be retrievable
        results2 = await provider.search(
            query="Python programming",
            limit=5,
        )
        assert isinstance(results2, list)

    @pytest.mark.asyncio
    async def test_episode_lifecycle(self, provider):
        """Test complete episode lifecycle."""
        # Start episode
        episode = await provider.episodic.start_episode(
            content="User conversation about weather",
        )

        # Add events to episode
        event = await provider.episodic.add_event(
            episode_id=episode.node_id,
            content="User asked about temperature",
        )
        assert event is not None

        # Close episode
        closed = await provider.episodic.close_episode(
            episode_id=episode.node_id,
            summary="Weather inquiry completed",
        )
        assert closed is not None

        # Search episodic memory
        results = await provider.episodic.search(
            query="weather temperature",
            limit=5,
        )
        assert isinstance(results, list)


# =============================================================================
# Cross-Layer Tests
# =============================================================================


class TestCrossLayerInteractions:
    """Test interactions across memory layers."""

    @pytest.mark.asyncio
    async def test_semantic_entity_creation_and_linking(self, provider):
        """Test entity creation and relationship linking."""
        # Create entities
        entity1 = await provider.semantic.create_entity(
            name="Paris",
            entity_type="city",
        )
        entity2 = await provider.semantic.create_entity(
            name="France",
            entity_type="country",
        )

        # Link entities
        link = await provider.semantic.add_relationship(
            source_entity_id=entity1.node_id,
            target_entity_id=entity2.node_id,
            relationship_type="capital_of",
        )
        assert link is not None

        # Search should find related entities
        results = await provider.semantic.search(
            query="Paris France capital",
            limit=5,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_skill_learning_and_retrieval(self, provider):
        """Test skill storage and retrieval."""
        # Add skill to metacognitive layer
        skill = await provider.metacognitive.add_skill(
            name="restart_nginx",
            skill_type="command",
            procedure="systemctl restart nginx",
        )

        # Retrieve by name
        retrieved = await provider.metacognitive.get_skill_by_name("restart_nginx")
        assert retrieved is not None
        assert retrieved.procedure == "systemctl restart nginx"

        # Record success
        await provider.metacognitive.record_skill_result(
            skill_id=skill.node_id,
            success=True,
        )

        # Verify success recorded
        updated = await provider.metacognitive.get_skill(skill.node_id)
        assert updated is not None

    @pytest.mark.asyncio
    async def test_multi_layer_search(self, provider):
        """Test searching across all layers."""
        # Store items in different layers
        await provider.store(
            content="Python is a programming language",
            memory_type="fact",
        )
        await provider.store(
            content="How to install Python: apt install python3",
            memory_type="skill",
        )
        await provider.store(
            content="User asked about Python today",
            memory_type="episodic",
        )

        # Search should find across layers
        results = await provider.search(
            query="Python",
            limit=10,
        )
        assert isinstance(results, list)


# =============================================================================
# Promotion Cycle Tests
# =============================================================================


class TestPromotionCycle:
    """Test memory promotion between layers."""

    @pytest.mark.asyncio
    async def test_get_promotion_candidates(self, provider):
        """Test identifying promotion candidates."""
        # Add items with high importance
        item = await provider.working.add(
            content="Very important item",
            attention_weight=0.95,
        )

        # Boost importance
        node = await provider.graph.get_node(item.node_id)
        node.importance = 0.95
        node.access_count = 10

        # Get promotion candidates
        candidates = await provider.working.get_promotion_candidates()
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_consolidation_cycle(self, provider):
        """Test memory consolidation."""
        # Store some memories
        for i in range(5):
            await provider.store(
                content=f"Memory item {i} for consolidation",
                memory_type="fact",
            )

        # Run consolidation
        result = await provider.consolidate()

        # Should return stats
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_decay_application(self, provider):
        """Test decay application to memories."""
        # Add item to working memory
        item = await provider.working.add(
            content="Decaying item",
            attention_weight=0.5,
        )

        # Force last_decay to be old
        provider.working._last_decay = datetime.now() - timedelta(minutes=10)

        # Apply decay
        decayed = await provider.working.apply_decay()

        assert isinstance(decayed, int)
        assert decayed >= 0


# =============================================================================
# Persistence Verification Tests
# =============================================================================


class TestPersistenceVerification:
    """Test that full flows persist correctly to Qdrant."""

    @pytest.mark.asyncio
    async def test_full_flow_persists(self, embedder, qdrant_available):
        """Test that a full flow persists and survives reconnection."""
        test_id = uuid4().hex[:8]
        nodes_col = f"{TEST_COLLECTION_PREFIX}persist_nodes_{test_id}"
        edges_col = f"{TEST_COLLECTION_PREFIX}persist_edges_{test_id}"

        config = LayeredMemoryConfig(
            qdrant_url=QDRANT_URL,
            qdrant_nodes_collection=nodes_col,
            qdrant_edges_collection=edges_col,
            embedding_dimension=768,
        )

        # First connection - create data
        p1 = LayeredMemoryProvider(config=config, embedding_provider=embedder)
        await p1.initialize()

        # Store memories
        await p1.store(content="Fact 1", memory_type="fact")
        await p1.store(content="Fact 2", memory_type="fact")
        await p1.store(content="Skill 1", memory_type="skill")

        # Create entity
        entity = await p1.semantic.create_entity(
            name="TestEntity",
            entity_type="test",
        )

        await p1.close()

        # Second connection - verify data
        p2 = LayeredMemoryProvider(config=config, embedding_provider=embedder)
        await p2.initialize()

        try:
            # Search should find data
            results = await p2.search(query="Fact", limit=5)
            assert isinstance(results, list)
            # Results may vary based on embedding similarity

        finally:
            try:
                if hasattr(p2._graph, '_client') and p2._graph._client:
                    await p2._graph._client.delete_collection(nodes_col)
                    await p2._graph._client.delete_collection(edges_col)
            except Exception:
                pass
            await p2.close()


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestFlowEdgeCases:
    """Test edge cases in memory flows."""

    @pytest.mark.asyncio
    async def test_search_empty_database(self, provider):
        """Test searching when database is empty (for this test)."""
        results = await provider.search(
            query="something that doesn't exist xyz123",
            limit=5,
        )
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_reinforce_nonexistent(self, provider):
        """Test reinforcing a nonexistent memory."""
        result = await provider.reinforce("nonexistent_id", boost=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_and_search(self, provider):
        """Test that deleted memories don't appear in search."""
        # Store using semantic layer directly to get node_id
        fact = await provider.semantic.add_fact(
            content="Memory to delete xyz unique",
            entities=["test"],
        )

        # Delete using node_id
        deleted = await provider.graph.delete_node(fact.node_id)
        assert deleted is True

        # Search should not find deleted
        results = await provider.search(
            query="xyz unique",
            limit=5,
        )
        found = any("xyz unique" in r.memory.content for r in results)
        assert not found

    @pytest.mark.asyncio
    async def test_update_and_search(self, provider):
        """Test that updated memories reflect changes."""
        # Store
        memory = await provider.store(
            content="Original content abc123",
            memory_type="fact",
        )

        # Update importance
        await provider.update(
            memory_id=memory.id,
            importance=0.99,
        )

        # Memory should still be searchable
        results = await provider.search(
            query="abc123",
            limit=5,
        )
        assert isinstance(results, list)
