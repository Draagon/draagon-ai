"""Tests for QdrantGraphStore (REQ-001-01).

This tests the Qdrant-backed TemporalCognitiveGraph implementation.
Uses mocked Qdrant client for unit tests.
"""

import pytest
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from draagon_ai.memory import (
    TemporalNode,
    TemporalEdge,
    EmbeddingProvider,
    NodeType,
    EdgeType,
    MemoryLayer,
    reset_scope_registry,
)

# Import conditionally since qdrant-client may not be installed
try:
    from draagon_ai.memory.providers.qdrant_graph import (
        QdrantGraphStore,
        QdrantGraphConfig,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantGraphStore = None
    QdrantGraphConfig = None


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text content."""
        self.call_count += 1

        embedding = [0.0] * self.dimension
        words = text.lower().split()
        for i, word in enumerate(words):
            h = hash(word)
            for j in range(10):
                idx = abs(h + j) % self.dimension
                embedding[idx] += 0.1 * (1 / (i + 1))

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


@dataclass
class MockQdrantRecord:
    """Mock Qdrant record for testing."""
    id: str
    payload: dict
    vector: list[float] | None = None
    score: float = 1.0


class MockAsyncQdrantClient:
    """Mock Qdrant client for testing."""

    def __init__(self):
        self.collections: dict[str, dict] = {}
        self.points: dict[str, dict[str, MockQdrantRecord]] = {}
        self._closed = False

    async def get_collections(self):
        """Return mock collections."""
        @dataclass
        class Collection:
            name: str

        @dataclass
        class CollectionsResponse:
            collections: list[Collection]

        return CollectionsResponse(
            collections=[Collection(name=name) for name in self.collections.keys()]
        )

    async def create_collection(self, collection_name: str, **kwargs):
        """Create a mock collection."""
        self.collections[collection_name] = kwargs
        self.points[collection_name] = {}

    async def upsert(self, collection_name: str, points: list):
        """Upsert points to mock collection."""
        if collection_name not in self.points:
            self.points[collection_name] = {}

        for point in points:
            self.points[collection_name][point.id] = MockQdrantRecord(
                id=point.id,
                payload=point.payload,
                vector=point.vector,
            )

    async def delete(self, collection_name: str, points_selector):
        """Delete points from mock collection."""
        if collection_name not in self.points:
            return

        # Handle HasIdCondition
        if hasattr(points_selector, 'has_id'):
            for point_id in points_selector.has_id:
                if point_id in self.points[collection_name]:
                    del self.points[collection_name][point_id]

    async def retrieve(self, collection_name: str, ids: list, **kwargs):
        """Retrieve points by ID."""
        if collection_name not in self.points:
            return []

        return [
            self.points[collection_name][pid]
            for pid in ids
            if pid in self.points[collection_name]
        ]

    async def scroll(self, collection_name: str, **kwargs):
        """Scroll through points."""
        if collection_name not in self.points:
            return [], None

        limit = kwargs.get('limit', 100)
        scroll_filter = kwargs.get('scroll_filter')

        records = list(self.points[collection_name].values())

        # Apply filter if provided
        if scroll_filter:
            # Simple filter implementation for tests
            records = self._apply_filter(records, scroll_filter)

        return records[:limit], None

    async def search(self, collection_name: str, query_vector: list, **kwargs):
        """Search for similar points."""
        if collection_name not in self.points:
            return []

        limit = kwargs.get('limit', 10)
        score_threshold = kwargs.get('score_threshold', 0.0)
        query_filter = kwargs.get('query_filter')

        records = list(self.points[collection_name].values())

        # Apply filter if provided
        if query_filter:
            records = self._apply_filter(records, query_filter)

        # Mock scoring - just return all matching records with fake scores
        results = []
        for record in records:
            score = 0.8  # Mock score
            if score >= score_threshold:
                record.score = score
                results.append(record)

        return results[:limit]

    def _apply_filter(self, records: list, filter_obj) -> list:
        """Apply a mock filter to records."""
        if not hasattr(filter_obj, 'must') or not filter_obj.must:
            return records

        filtered = records
        for condition in filter_obj.must:
            if hasattr(condition, 'key') and hasattr(condition, 'match'):
                key = condition.key
                match = condition.match

                if hasattr(match, 'value'):
                    # MatchValue
                    filtered = [
                        r for r in filtered
                        if r.payload.get(key) == match.value
                    ]
                elif hasattr(match, 'any'):
                    # MatchAny
                    filtered = [
                        r for r in filtered
                        if r.payload.get(key) in match.any
                    ]

        return filtered

    async def get_collection(self, collection_name: str):
        """Get collection info."""
        @dataclass
        class CollectionInfo:
            points_count: int
            vectors_count: int
            indexed_vectors_count: int

        count = len(self.points.get(collection_name, {}))
        return CollectionInfo(
            points_count=count,
            vectors_count=count,
            indexed_vectors_count=count,
        )

    async def close(self):
        """Close the mock client."""
        self._closed = True


@pytest.fixture
def mock_embedder():
    """Fixture for mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def mock_qdrant_client():
    """Fixture for mock Qdrant client."""
    return MockAsyncQdrantClient()


@pytest.fixture
def config():
    """Fixture for QdrantGraphStore config."""
    if not QDRANT_AVAILABLE:
        pytest.skip("qdrant-client not installed")
    return QdrantGraphConfig(
        url="http://localhost:6333",
        nodes_collection="test_nodes",
        edges_collection="test_edges",
    )


@pytest.fixture
async def graph_store(config, mock_embedder, mock_qdrant_client):
    """Fixture for QdrantGraphStore with mocked client."""
    reset_scope_registry()

    store = QdrantGraphStore(config, embedding_provider=mock_embedder)

    # Replace the client with our mock
    with patch.object(store, '_client', mock_qdrant_client):
        store._initialized = True
        # Create mock collections
        await mock_qdrant_client.create_collection(config.nodes_collection)
        await mock_qdrant_client.create_collection(config.edges_collection)
        store._client = mock_qdrant_client
        yield store


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantGraphStoreNodeOperations:
    """Tests for node CRUD operations with Qdrant persistence."""

    @pytest.mark.anyio
    async def test_add_node_persists_to_qdrant(self, graph_store, mock_qdrant_client, config):
        """Test that adding a node persists it to Qdrant."""
        node = await graph_store.add_node(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            entities=["Doug", "birthday"],
        )

        # Check node was persisted
        assert node.node_id in mock_qdrant_client.points[config.nodes_collection]

        # Check payload
        record = mock_qdrant_client.points[config.nodes_collection][node.node_id]
        assert record.payload["content"] == "Doug's birthday is March 15"
        assert record.payload["node_type"] == "fact"
        assert record.payload["scope_id"] == "user:roxy:doug"
        assert record.payload["entities"] == ["Doug", "birthday"]

    @pytest.mark.anyio
    async def test_add_node_stores_bitemporal_timestamps(self, graph_store, mock_qdrant_client, config):
        """Test that bi-temporal timestamps are stored correctly."""
        event_time = datetime(2020, 3, 15, 12, 0, 0)

        node = await graph_store.add_node(
            content="Test event",
            node_type=NodeType.EVENT,
            event_time=event_time,
        )

        record = mock_qdrant_client.points[config.nodes_collection][node.node_id]

        # Check bi-temporal timestamps
        assert record.payload["event_time"] == event_time.isoformat()
        assert "ingestion_time" in record.payload
        assert "valid_from" in record.payload
        assert record.payload["valid_until"] is None  # Current node

    @pytest.mark.anyio
    async def test_update_node_persists_changes(self, graph_store, mock_qdrant_client, config):
        """Test that updating a node persists changes to Qdrant."""
        node = await graph_store.add_node(
            content="Original content",
            node_type=NodeType.FACT,
        )

        # Update the node
        updated = await graph_store.update_node(
            node.node_id,
            content="Updated content",
            importance=0.9,
        )

        # Check update was persisted
        record = mock_qdrant_client.points[config.nodes_collection][node.node_id]
        assert record.payload["content"] == "Updated content"
        assert record.payload["importance"] == 0.9

    @pytest.mark.anyio
    async def test_delete_node_removes_from_qdrant(self, graph_store, mock_qdrant_client, config):
        """Test that deleting a node removes it from Qdrant."""
        node = await graph_store.add_node(
            content="To be deleted",
            node_type=NodeType.FACT,
        )

        # Verify it exists
        assert node.node_id in mock_qdrant_client.points[config.nodes_collection]

        # Delete it
        result = await graph_store.delete_node(node.node_id)

        assert result is True
        assert node.node_id not in mock_qdrant_client.points[config.nodes_collection]

    @pytest.mark.anyio
    async def test_supersede_node_persists_both_nodes(self, graph_store, mock_qdrant_client, config):
        """Test that superseding a node persists both old and new nodes."""
        old_node = await graph_store.add_node(
            content="WiFi password is abc123",
            node_type=NodeType.FACT,
        )

        new_node = await graph_store.supersede_node(
            old_node.node_id,
            "WiFi password is xyz789",
        )

        # Both nodes should be persisted
        assert old_node.node_id in mock_qdrant_client.points[config.nodes_collection]
        assert new_node.node_id in mock_qdrant_client.points[config.nodes_collection]

        # Old node should be marked as superseded
        old_record = mock_qdrant_client.points[config.nodes_collection][old_node.node_id]
        assert old_record.payload["superseded_by"] == new_node.node_id
        assert old_record.payload["is_current"] is False

        # New node should reference old
        new_record = mock_qdrant_client.points[config.nodes_collection][new_node.node_id]
        assert old_node.node_id in new_record.payload["supersedes"]


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantGraphStoreEdgeOperations:
    """Tests for edge CRUD operations with Qdrant persistence."""

    @pytest.mark.anyio
    async def test_add_edge_persists_to_qdrant(self, graph_store, mock_qdrant_client, config):
        """Test that adding an edge persists it to Qdrant."""
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
            label="test relationship",
            weight=0.8,
        )

        # Check edge was persisted
        assert edge.edge_id in mock_qdrant_client.points[config.edges_collection]

        record = mock_qdrant_client.points[config.edges_collection][edge.edge_id]
        assert record.payload["source_id"] == node1.node_id
        assert record.payload["target_id"] == node2.node_id
        assert record.payload["edge_type"] == "related_to"
        assert record.payload["label"] == "test relationship"
        assert record.payload["weight"] == 0.8

    @pytest.mark.anyio
    async def test_delete_edge_removes_from_qdrant(self, graph_store, mock_qdrant_client, config):
        """Test that deleting an edge removes it from Qdrant."""
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        # Delete it
        result = await graph_store.delete_edge(edge.edge_id)

        assert result is True
        assert edge.edge_id not in mock_qdrant_client.points[config.edges_collection]

    @pytest.mark.anyio
    async def test_delete_node_cascades_to_edges(self, graph_store, mock_qdrant_client, config):
        """Test that deleting a node also deletes its edges."""
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.ENTITY)

        edge = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        # Delete node1 - should also delete the edge
        await graph_store.delete_node(node1.node_id)

        assert edge.edge_id not in mock_qdrant_client.points[config.edges_collection]


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantGraphStoreSearch:
    """Tests for search operations using Qdrant."""

    @pytest.mark.anyio
    async def test_search_returns_results(self, graph_store):
        """Test that search returns matching nodes."""
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

        results = await graph_store.search("When is Doug's birthday?", limit=5)

        assert len(results) > 0
        # Results should include the birthday facts

    @pytest.mark.anyio
    async def test_search_filters_by_scope(self, graph_store):
        """Test that search respects scope filtering."""
        await graph_store.add_node(
            content="User fact",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
        )
        await graph_store.add_node(
            content="World fact",
            node_type=NodeType.FACT,
            scope_id="world:global",
        )

        # Search only in user scope
        results = await graph_store.search(
            "fact",
            scope_ids=["user:roxy:doug"],
            limit=10,
        )

        # Should only find the user fact (and maybe ancestor scopes)
        for result in results:
            assert "user" in result.node.scope_id or "world" in result.node.scope_id

    @pytest.mark.anyio
    async def test_search_filters_by_node_type(self, graph_store):
        """Test that search filters by node type."""
        await graph_store.add_node(
            content="A fact",
            node_type=NodeType.FACT,
        )
        await graph_store.add_node(
            content="A skill",
            node_type=NodeType.SKILL,
        )

        results = await graph_store.search(
            "test",
            node_types=[NodeType.FACT],
            limit=10,
        )

        for result in results:
            assert result.node.node_type == NodeType.FACT


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantGraphStoreLoading:
    """Tests for loading graph from Qdrant."""

    @pytest.mark.anyio
    async def test_load_from_qdrant(self, graph_store, mock_qdrant_client, config):
        """Test loading graph data from Qdrant."""
        # Add some nodes directly to the mock
        node1 = await graph_store.add_node(content="Node 1", node_type=NodeType.FACT)
        node2 = await graph_store.add_node(content="Node 2", node_type=NodeType.FACT)
        await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.RELATED_TO,
        )

        # Clear in-memory state
        graph_store.clear()

        # Reload from Qdrant
        nodes_loaded, edges_loaded = await graph_store.load_from_qdrant()

        assert nodes_loaded == 2
        assert edges_loaded == 1

        # Nodes should be back in memory
        assert node1.node_id in graph_store._nodes
        assert node2.node_id in graph_store._nodes

    @pytest.mark.anyio
    async def test_get_node_loads_from_qdrant_if_not_in_memory(self, graph_store, mock_qdrant_client):
        """Test that get_node loads from Qdrant if not cached."""
        # Add a node
        node = await graph_store.add_node(content="Test", node_type=NodeType.FACT)
        node_id = node.node_id

        # Remove from memory but not Qdrant
        del graph_store._nodes[node_id]

        # Get should load from Qdrant
        retrieved = await graph_store.get_node(node_id)

        assert retrieved is not None
        assert retrieved.node_id == node_id
        assert retrieved.content == "Test"


@pytest.mark.skipif(not QDRANT_AVAILABLE, reason="qdrant-client not installed")
class TestQdrantGraphStoreSerialization:
    """Tests for node/edge serialization."""

    @pytest.mark.anyio
    async def test_node_round_trip(self, graph_store, mock_qdrant_client, config):
        """Test that nodes survive round-trip serialization."""
        original = await graph_store.add_node(
            content="Test content",
            node_type=NodeType.SKILL,
            scope_id="agent:test",
            entities=["entity1", "entity2"],
            confidence=0.9,
            importance=0.8,
            metadata={"key": "value"},
        )

        # Get from payload
        record = mock_qdrant_client.points[config.nodes_collection][original.node_id]
        restored = graph_store._payload_to_node(record.payload, record.vector)

        assert restored.node_id == original.node_id
        assert restored.content == original.content
        assert restored.node_type == original.node_type
        assert restored.scope_id == original.scope_id
        assert restored.entities == original.entities
        assert restored.confidence == original.confidence
        assert restored.importance == original.importance
        assert restored.metadata == original.metadata

    @pytest.mark.anyio
    async def test_edge_round_trip(self, graph_store, mock_qdrant_client, config):
        """Test that edges survive round-trip serialization."""
        node1 = await graph_store.add_node(content="N1", node_type=NodeType.ENTITY)
        node2 = await graph_store.add_node(content="N2", node_type=NodeType.ENTITY)

        original = await graph_store.add_edge(
            source_id=node1.node_id,
            target_id=node2.node_id,
            edge_type=EdgeType.CAUSES,
            label="test label",
            weight=0.75,
            confidence=0.85,
            metadata={"reason": "testing"},
        )

        # Get from payload
        record = mock_qdrant_client.points[config.edges_collection][original.edge_id]
        restored = graph_store._payload_to_edge(record.payload)

        assert restored.edge_id == original.edge_id
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.edge_type == original.edge_type
        assert restored.label == original.label
        assert restored.weight == original.weight
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata
