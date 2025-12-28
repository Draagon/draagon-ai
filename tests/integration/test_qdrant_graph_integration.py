"""Integration tests for QdrantGraphStore with real Qdrant (REQ-001-08).

These tests hit a REAL Qdrant instance - no mocks.
They verify actual graph persistence, node/edge operations, and search.

Requirements:
- Running Qdrant instance (default: http://192.168.168.216:6333)
- qdrant-client package installed

Set QDRANT_URL environment variable to override the default.
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from uuid import uuid4

# Check for qdrant-client
try:
    from qdrant_client import AsyncQdrantClient
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from draagon_ai.memory import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
    reset_scope_registry,
)

# Import conditionally
try:
    from draagon_ai.memory.providers.qdrant_graph import (
        QdrantGraphStore,
        QdrantGraphConfig,
    )
    QDRANT_GRAPH_AVAILABLE = True
except ImportError:
    QDRANT_GRAPH_AVAILABLE = False
    QdrantGraphStore = None
    QdrantGraphConfig = None


# Test configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.168.216:6333")
TEST_COLLECTION_PREFIX = "test_draagon_graph_"


class WordBasedEmbeddingProvider:
    """Word-based embedding provider for tests.

    Generates embeddings based on word overlap so that semantically
    similar texts produce similar vectors.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        """Get or assign a vocab index for a word."""
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        """Generate word-based embedding."""
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


# Skip all tests if Qdrant not available
pytestmark = [
    pytest.mark.skipif(
        not QDRANT_GRAPH_AVAILABLE,
        reason="qdrant-client or QdrantGraphStore not available"
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
async def qdrant_available():
    """Check Qdrant connectivity once per module."""
    available = await check_qdrant_connection()
    if not available:
        pytest.skip(f"Qdrant not accessible at {QDRANT_URL}")
    return True


@pytest.fixture
def embedder():
    """Create embedding provider."""
    return WordBasedEmbeddingProvider()


@pytest.fixture
async def graph_store(embedder, qdrant_available):
    """Create and cleanup a QdrantGraphStore with test collections."""
    reset_scope_registry()

    test_id = uuid4().hex[:8]
    config = QdrantGraphConfig(
        url=QDRANT_URL,
        nodes_collection=f"{TEST_COLLECTION_PREFIX}nodes_{test_id}",
        edges_collection=f"{TEST_COLLECTION_PREFIX}edges_{test_id}",
        embedding_dimension=768,
        similarity_threshold=0.05,  # Low threshold for word-based mock embeddings
    )

    store = QdrantGraphStore(config, embedding_provider=embedder)
    await store.initialize()

    yield store

    # Cleanup: delete test collections
    try:
        await store._client.delete_collection(config.nodes_collection)
        await store._client.delete_collection(config.edges_collection)
    except Exception:
        pass
    await store.close()


# =============================================================================
# Node Operations Tests
# =============================================================================


class TestNodeOperationsIntegration:
    """Tests for node CRUD operations with real Qdrant."""

    @pytest.mark.asyncio
    async def test_add_node_persists(self, graph_store):
        """Test that adding a node actually persists to Qdrant."""
        node = await graph_store.add_node(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            entities=["Doug", "birthday"],
            importance=0.8,
        )

        assert node.node_id is not None
        assert node.content == "Doug's birthday is March 15"
        assert node.node_type == NodeType.FACT
        assert node.entities == ["Doug", "birthday"]
        assert node.importance == 0.8

        # Verify it exists in Qdrant
        retrieved = await graph_store.get_node(node.node_id)
        assert retrieved is not None
        assert retrieved.content == node.content

    @pytest.mark.asyncio
    async def test_node_bitemporal_timestamps(self, graph_store):
        """Test that bi-temporal timestamps are correctly stored."""
        event_time = datetime(2020, 3, 15, 12, 0, 0)

        node = await graph_store.add_node(
            content="Test event with custom time",
            node_type=NodeType.EVENT,
            event_time=event_time,
        )

        retrieved = await graph_store.get_node(node.node_id)

        assert retrieved is not None
        assert retrieved.event_time.year == 2020
        assert retrieved.event_time.month == 3
        assert retrieved.event_time.day == 15
        assert retrieved.ingestion_time is not None
        assert retrieved.valid_from is not None

    @pytest.mark.asyncio
    async def test_update_node_persists(self, graph_store):
        """Test that updating a node persists changes."""
        node = await graph_store.add_node(
            content="Original content",
            node_type=NodeType.FACT,
            importance=0.5,
        )

        updated = await graph_store.update_node(
            node.node_id,
            content="Updated content",
            importance=0.9,
        )

        assert updated.content == "Updated content"
        assert updated.importance == 0.9

        # Verify persistence
        retrieved = await graph_store.get_node(node.node_id)
        assert retrieved.content == "Updated content"
        assert retrieved.importance == 0.9

    @pytest.mark.asyncio
    async def test_delete_node(self, graph_store):
        """Test that deleting a node actually removes it."""
        node = await graph_store.add_node(
            content="To be deleted",
            node_type=NodeType.FACT,
        )

        # Verify exists
        assert await graph_store.get_node(node.node_id) is not None

        # Delete
        result = await graph_store.delete_node(node.node_id)
        assert result is True

        # Verify gone
        assert await graph_store.get_node(node.node_id) is None

    @pytest.mark.asyncio
    async def test_supersede_node(self, graph_store):
        """Test superseding a node with a new version."""
        old_node = await graph_store.add_node(
            content="WiFi password is abc123",
            node_type=NodeType.FACT,
        )

        new_node = await graph_store.supersede_node(
            old_node.node_id,
            "WiFi password is xyz789",
        )

        # Both should exist
        old_retrieved = await graph_store.get_node(old_node.node_id)
        new_retrieved = await graph_store.get_node(new_node.node_id)

        assert old_retrieved is not None
        assert new_retrieved is not None

        # Old should be marked superseded
        assert old_retrieved.superseded_by == new_node.node_id
        assert old_node.node_id in new_retrieved.supersedes


# =============================================================================
# Edge Operations Tests
# =============================================================================


class TestEdgeOperationsIntegration:
    """Tests for edge CRUD operations with real Qdrant."""

    @pytest.mark.asyncio
    async def test_add_edge_persists(self, graph_store):
        """Test that adding an edge persists to Qdrant."""
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
            label="test relationship",
            weight=0.8,
        )

        assert edge.edge_id is not None
        assert edge.source_id == node1.node_id
        assert edge.target_id == node2.node_id
        assert edge.weight == 0.8

        # Verify it exists
        retrieved = await graph_store.get_edge(edge.edge_id)
        assert retrieved is not None
        assert retrieved.label == "test relationship"

    @pytest.mark.asyncio
    async def test_delete_edge(self, graph_store):
        """Test that deleting an edge removes it."""
        node1 = await graph_store.add_node(content="N1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="N2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        # Delete
        result = await graph_store.delete_edge(edge.edge_id)
        assert result is True

        # Verify gone
        assert await graph_store.get_edge(edge.edge_id) is None

    @pytest.mark.asyncio
    async def test_delete_node_cascades_to_edges(self, graph_store):
        """Test that deleting a node also deletes its edges."""
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        # Delete node1
        await graph_store.delete_node(node1.node_id)

        # Edge should also be gone
        assert await graph_store.get_edge(edge.edge_id) is None


# =============================================================================
# Search Tests
# =============================================================================


class TestSearchIntegration:
    """Tests for semantic search with real Qdrant."""

    @pytest.mark.asyncio
    async def test_search_returns_results(self, graph_store):
        """Test that search returns relevant results."""
        await graph_store.add_node(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
        )
        await graph_store.add_node(
            content="Lisa's birthday is June 20",
            node_type=NodeType.FACT,
            scope_id="user:roxy:lisa",
        )
        await graph_store.add_node(
            content="The weather is sunny today",
            node_type=NodeType.FACT,
        )

        results = await graph_store.search("When is Doug's birthday?", limit=5)

        assert len(results) > 0
        # Birthday-related content should be found
        contents = [r.node.content for r in results]
        assert any("birthday" in c.lower() for c in contents)

    @pytest.mark.asyncio
    async def test_search_filters_by_node_type(self, graph_store):
        """Test that search can filter by node type."""
        await graph_store.add_node(
            content="A fact about cats",
            node_type=NodeType.FACT,
        )
        await graph_store.add_node(
            content="A skill about cats",
            node_type=NodeType.SKILL,
        )

        results = await graph_store.search(
            "cats",
            node_types=[NodeType.FACT],
            limit=10,
        )

        for result in results:
            assert result.node.node_type == NodeType.FACT

    @pytest.mark.asyncio
    async def test_search_filters_by_scope(self, graph_store):
        """Test that search respects scope filtering."""
        await graph_store.add_node(
            content="User secret",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
        )
        await graph_store.add_node(
            content="World fact",
            node_type=NodeType.FACT,
            scope_id="world:global",
        )

        results = await graph_store.search(
            "fact",
            scope_ids=["user:roxy:doug"],
            limit=10,
        )

        # Results should be from accessible scopes
        for result in results:
            assert "user:roxy:doug" in result.node.scope_id or "world" in result.node.scope_id

    @pytest.mark.asyncio
    async def test_search_filters_by_layer(self, graph_store):
        """Test that search can filter by memory layer."""
        await graph_store.add_node(
            content="A semantic fact",
            node_type=NodeType.FACT,  # Maps to SEMANTIC layer
        )
        await graph_store.add_node(
            content="A metacognitive skill",
            node_type=NodeType.SKILL,  # Maps to METACOGNITIVE layer
        )

        results = await graph_store.search(
            "knowledge",
            layers=[MemoryLayer.SEMANTIC],
            limit=10,
        )

        for result in results:
            assert result.node.layer == MemoryLayer.SEMANTIC


# =============================================================================
# Persistence Across Reconnects Tests
# =============================================================================


class TestPersistenceIntegration:
    """Tests verifying data persists across provider instances."""

    @pytest.mark.asyncio
    async def test_graph_survives_reconnect(self, embedder, qdrant_available):
        """Test that graph data persists after closing and reopening."""
        reset_scope_registry()

        test_id = uuid4().hex[:8]
        config = QdrantGraphConfig(
            url=QDRANT_URL,
            nodes_collection=f"{TEST_COLLECTION_PREFIX}persist_{test_id}",
            edges_collection=f"{TEST_COLLECTION_PREFIX}persist_edges_{test_id}",
        )

        # Create store and add data
        store1 = QdrantGraphStore(config, embedding_provider=embedder)
        await store1.initialize()

        node1 = await store1.add_node(
            content="Persistent node 1",
            node_type=NodeType.FACT,
        )
        node2 = await store1.add_node(
            content="Persistent node 2",
            node_type=NodeType.FACT,
        )
        edge = await store1.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        await store1.close()

        # Create new store with same collections
        store2 = QdrantGraphStore(config, embedding_provider=embedder)
        await store2.initialize()

        # Load from Qdrant
        nodes_loaded, edges_loaded = await store2.load_from_qdrant()

        assert nodes_loaded == 2
        assert edges_loaded == 1

        # Data should be accessible
        retrieved1 = await store2.get_node(node1.node_id)
        retrieved2 = await store2.get_node(node2.node_id)
        retrieved_edge = await store2.get_edge(edge.edge_id)

        assert retrieved1 is not None
        assert retrieved1.content == "Persistent node 1"
        assert retrieved2 is not None
        assert retrieved_edge is not None

        # Cleanup
        await store2._client.delete_collection(config.nodes_collection)
        await store2._client.delete_collection(config.edges_collection)
        await store2.close()

    @pytest.mark.asyncio
    async def test_large_scale_persistence(self, embedder, qdrant_available):
        """Test persistence with many nodes (100+)."""
        reset_scope_registry()

        test_id = uuid4().hex[:8]
        config = QdrantGraphConfig(
            url=QDRANT_URL,
            nodes_collection=f"{TEST_COLLECTION_PREFIX}large_{test_id}",
            edges_collection=f"{TEST_COLLECTION_PREFIX}large_edges_{test_id}",
        )

        store = QdrantGraphStore(config, embedding_provider=embedder)
        await store.initialize()

        # Add 100 nodes
        node_ids = []
        for i in range(100):
            node = await store.add_node(
                content=f"Fact number {i} about testing",
                node_type=NodeType.FACT,
            )
            node_ids.append(node.node_id)

        # Verify all persisted
        store.clear()
        nodes_loaded, _ = await store.load_from_qdrant()

        assert nodes_loaded == 100

        # Cleanup
        await store._client.delete_collection(config.nodes_collection)
        await store._client.delete_collection(config.edges_collection)
        await store.close()


# =============================================================================
# Concurrent Access Tests
# =============================================================================


class TestConcurrentAccessIntegration:
    """Tests for concurrent write handling."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, graph_store):
        """Test that concurrent writes don't cause data loss."""

        async def add_node(content: str):
            return await graph_store.add_node(
                content=content,
                node_type=NodeType.FACT,
            )

        # Add 20 nodes concurrently
        tasks = [add_node(f"Concurrent fact {i}") for i in range(20)]
        nodes = await asyncio.gather(*tasks)

        # All should be created
        assert len(nodes) == 20
        assert all(n is not None for n in nodes)

        # Verify all exist
        for node in nodes:
            retrieved = await graph_store.get_node(node.node_id)
            assert retrieved is not None


# =============================================================================
# Graph Traversal Tests
# =============================================================================


class TestGraphTraversalIntegration:
    """Tests for graph traversal with real Qdrant."""

    @pytest.mark.asyncio
    async def test_traverse_neighbors(self, graph_store):
        """Test traversing to neighbor nodes."""
        # Create a chain: A -> B -> C
        node_a = await graph_store.add_node(content="Node A", node_type=NodeType.ENTITY)
        node_b = await graph_store.add_node(content="Node B", node_type=NodeType.ENTITY)
        node_c = await graph_store.add_node(content="Node C", node_type=NodeType.ENTITY)

        await graph_store.add_edge(
            source_id=node_a.node_id,
            target_id=node_b.node_id,
            edge_type=EdgeType.CAUSES,
        )
        await graph_store.add_edge(
            source_id=node_b.node_id,
            target_id=node_c.node_id,
            edge_type=EdgeType.CAUSES,
        )

        # Traverse from A
        result = await graph_store.traverse(node_a.node_id, max_hops=2)

        assert result is not None
        # Should find B and C within 2 hops
        found_ids = [n.node_id for n in result.nodes]
        assert node_b.node_id in found_ids
        assert node_c.node_id in found_ids

    @pytest.mark.asyncio
    async def test_get_edges_from(self, graph_store):
        """Test getting edges from a node to find neighbors."""
        center = await graph_store.add_node(content="Center", node_type=NodeType.ENTITY)
        neighbor1 = await graph_store.add_node(content="N1", node_type=NodeType.ENTITY)
        neighbor2 = await graph_store.add_node(content="N2", node_type=NodeType.ENTITY)

        await graph_store.add_edge(
            source_id=center.node_id,
            target_id=neighbor1.node_id,
            edge_type=EdgeType.RELATED_TO,
        )
        await graph_store.add_edge(
            source_id=center.node_id,
            target_id=neighbor2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        edges_out = await graph_store.get_edges_from(center.node_id)

        assert len(edges_out) == 2
        neighbor_ids = [e.target_id for e in edges_out]
        assert neighbor1.node_id in neighbor_ids
        assert neighbor2.node_id in neighbor_ids
