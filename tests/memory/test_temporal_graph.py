"""Tests for TemporalCognitiveGraph (Phase C.1)."""

import pytest
from datetime import datetime, timedelta
from typing import Any

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    TemporalNode,
    TemporalEdge,
    EmbeddingProvider,
    NodeType,
    EdgeType,
    MemoryLayer,
    GraphSearchResult,
    GraphTraversalResult,
    reset_scope_registry,
)


class MockEmbeddingProvider(EmbeddingProvider):
    """Mock embedding provider for testing.

    Uses simple word-based embeddings for deterministic testing.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self.call_count = 0

    async def embed(self, text: str) -> list[float]:
        """Generate a deterministic embedding based on text content."""
        self.call_count += 1

        # Create a simple hash-based embedding
        embedding = [0.0] * self.dimension

        # Use word positions to create embedding
        words = text.lower().split()
        for i, word in enumerate(words):
            # Hash each word to positions in the embedding
            h = hash(word)
            for j in range(10):  # Spread across 10 positions
                idx = abs(h + j) % self.dimension
                embedding[idx] += 0.1 * (1 / (i + 1))  # Decay by position

        # Normalize
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding


@pytest.fixture
def mock_embedder():
    """Fixture for mock embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
def graph(mock_embedder):
    """Fixture for graph with mock embedder."""
    reset_scope_registry()
    return TemporalCognitiveGraph(embedding_provider=mock_embedder)


class TestGraphNodeOperations:
    """Tests for node CRUD operations."""

    @pytest.mark.anyio
    async def test_add_node(self, graph, mock_embedder):
        """Test adding a node."""
        node = await graph.add_node(
            content="Doug's birthday is March 15",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            entities=["Doug", "birthday"],
        )

        assert node.node_id is not None
        assert node.content == "Doug's birthday is March 15"
        assert node.node_type == NodeType.FACT
        assert node.embedding is not None
        assert len(node.embedding) == mock_embedder.dimension
        assert mock_embedder.call_count == 1

    @pytest.mark.anyio
    async def test_add_node_with_precomputed_embedding(self, graph, mock_embedder):
        """Test adding a node with pre-computed embedding."""
        embedding = [0.1] * 768

        node = await graph.add_node(
            content="Test content",
            node_type=NodeType.FACT,
            embedding=embedding,
        )

        assert node.embedding == embedding
        assert mock_embedder.call_count == 0  # Embedder not called

    @pytest.mark.anyio
    async def test_get_node(self, graph):
        """Test getting a node by ID."""
        node = await graph.add_node(
            content="Test content",
            node_type=NodeType.FACT,
        )

        retrieved = await graph.get_node(node.node_id)

        assert retrieved is not None
        assert retrieved.node_id == node.node_id
        assert retrieved.content == node.content

    @pytest.mark.anyio
    async def test_get_nonexistent_node(self, graph):
        """Test getting a non-existent node."""
        result = await graph.get_node("nonexistent-id")
        assert result is None

    @pytest.mark.anyio
    async def test_update_node(self, graph):
        """Test updating a node."""
        node = await graph.add_node(
            content="Original content",
            node_type=NodeType.FACT,
            importance=0.5,
        )

        updated = await graph.update_node(
            node.node_id,
            content="Updated content",
            importance=0.8,
            metadata={"updated": True},
        )

        assert updated is not None
        assert updated.content == "Updated content"
        assert updated.importance == 0.8
        assert updated.metadata["updated"] is True

    @pytest.mark.anyio
    async def test_delete_node(self, graph):
        """Test deleting a node."""
        node = await graph.add_node(
            content="To be deleted",
            node_type=NodeType.FACT,
        )

        result = await graph.delete_node(node.node_id)
        assert result is True

        # Node should be gone
        retrieved = await graph.get_node(node.node_id)
        assert retrieved is None

    @pytest.mark.anyio
    async def test_delete_node_removes_edges(self, graph):
        """Test that deleting a node removes connected edges."""
        node1 = await graph.add_node("Node 1", NodeType.ENTITY)
        node2 = await graph.add_node("Node 2", NodeType.ENTITY)

        edge = await graph.add_edge(
            node1.node_id,
            node2.node_id,
            EdgeType.RELATED_TO,
        )

        await graph.delete_node(node1.node_id)

        # Edge should be gone
        retrieved_edge = await graph.get_edge(edge.edge_id)
        assert retrieved_edge is None

    @pytest.mark.anyio
    async def test_supersede_node(self, graph):
        """Test superseding a node with a newer version."""
        old_node = await graph.add_node(
            content="WiFi password is abc123",
            node_type=NodeType.FACT,
            scope_id="context:home",
        )

        new_node = await graph.supersede_node(
            old_node.node_id,
            new_content="WiFi password is xyz789",
        )

        assert new_node is not None
        assert old_node.node_id in new_node.supersedes
        assert new_node.node_id == old_node.superseded_by
        assert old_node.is_current is False
        assert new_node.is_current is True


class TestGraphEdgeOperations:
    """Tests for edge CRUD operations."""

    @pytest.mark.anyio
    async def test_add_edge(self, graph):
        """Test adding an edge."""
        node1 = await graph.add_node("Doug", NodeType.ENTITY)
        node2 = await graph.add_node("Philadelphia", NodeType.ENTITY)

        edge = await graph.add_edge(
            node1.node_id,
            node2.node_id,
            EdgeType.LOCATED_IN,
            label="lives in",
        )

        assert edge is not None
        assert edge.source_id == node1.node_id
        assert edge.target_id == node2.node_id
        assert edge.edge_type == EdgeType.LOCATED_IN
        assert edge.label == "lives in"

    @pytest.mark.anyio
    async def test_add_edge_nonexistent_nodes(self, graph):
        """Test that edges require existing nodes."""
        edge = await graph.add_edge(
            "nonexistent1",
            "nonexistent2",
            EdgeType.RELATED_TO,
        )

        assert edge is None

    @pytest.mark.anyio
    async def test_get_edges_from(self, graph):
        """Test getting edges from a node."""
        node1 = await graph.add_node("Center", NodeType.ENTITY)
        node2 = await graph.add_node("Connected 1", NodeType.ENTITY)
        node3 = await graph.add_node("Connected 2", NodeType.ENTITY)

        await graph.add_edge(node1.node_id, node2.node_id, EdgeType.HAS)
        await graph.add_edge(node1.node_id, node3.node_id, EdgeType.HAS)

        edges = await graph.get_edges_from(node1.node_id)

        assert len(edges) == 2
        target_ids = {e.target_id for e in edges}
        assert node2.node_id in target_ids
        assert node3.node_id in target_ids

    @pytest.mark.anyio
    async def test_get_edges_to(self, graph):
        """Test getting edges pointing to a node."""
        node1 = await graph.add_node("Source 1", NodeType.ENTITY)
        node2 = await graph.add_node("Source 2", NodeType.ENTITY)
        node3 = await graph.add_node("Target", NodeType.ENTITY)

        await graph.add_edge(node1.node_id, node3.node_id, EdgeType.CAUSES)
        await graph.add_edge(node2.node_id, node3.node_id, EdgeType.CAUSES)

        edges = await graph.get_edges_to(node3.node_id)

        assert len(edges) == 2
        source_ids = {e.source_id for e in edges}
        assert node1.node_id in source_ids
        assert node2.node_id in source_ids

    @pytest.mark.anyio
    async def test_delete_edge(self, graph):
        """Test deleting an edge."""
        node1 = await graph.add_node("Node 1", NodeType.ENTITY)
        node2 = await graph.add_node("Node 2", NodeType.ENTITY)

        edge = await graph.add_edge(
            node1.node_id,
            node2.node_id,
            EdgeType.RELATED_TO,
        )

        result = await graph.delete_edge(edge.edge_id)
        assert result is True

        # Edge should be gone
        retrieved = await graph.get_edge(edge.edge_id)
        assert retrieved is None


class TestGraphSearch:
    """Tests for semantic search operations."""

    @pytest.mark.anyio
    async def test_basic_search(self, graph):
        """Test basic semantic search."""
        await graph.add_node("Doug's birthday is March 15", NodeType.FACT)
        await graph.add_node("Maya's birthday is June 20", NodeType.FACT)
        await graph.add_node("The weather is sunny", NodeType.FACT)

        results = await graph.search("When is Doug's birthday?", limit=3)

        assert len(results) > 0
        # Doug's birthday should be most relevant
        assert "Doug" in results[0].node.content or "birthday" in results[0].node.content

    @pytest.mark.anyio
    async def test_search_with_scope_filter(self, graph):
        """Test search with scope filtering."""
        await graph.add_node(
            "Doug's fact",
            NodeType.FACT,
            scope_id="user:roxy:doug",
        )
        await graph.add_node(
            "Maya's fact",
            NodeType.FACT,
            scope_id="user:roxy:maya",
        )

        results = await graph.search(
            "fact",
            scope_ids=["user:roxy:doug"],
        )

        assert len(results) == 1
        assert "Doug" in results[0].node.content

    @pytest.mark.anyio
    async def test_search_with_type_filter(self, graph):
        """Test search with node type filtering."""
        await graph.add_node("A fact about something", NodeType.FACT)
        await graph.add_node("A skill for doing something", NodeType.SKILL)
        await graph.add_node("An episode about something", NodeType.EPISODE)

        results = await graph.search(
            "something",
            node_types=[NodeType.FACT, NodeType.SKILL],
        )

        for result in results:
            assert result.node.node_type in [NodeType.FACT, NodeType.SKILL]

    @pytest.mark.anyio
    async def test_search_with_layer_filter(self, graph):
        """Test search with memory layer filtering."""
        await graph.add_node("Semantic fact", NodeType.FACT)  # Semantic layer
        await graph.add_node("Episodic event", NodeType.EPISODE)  # Episodic layer
        await graph.add_node("Metacognitive skill", NodeType.SKILL)  # Metacognitive layer

        results = await graph.search(
            "test",
            layers=[MemoryLayer.SEMANTIC],
        )

        for result in results:
            assert result.node.layer == MemoryLayer.SEMANTIC

    @pytest.mark.anyio
    async def test_search_excludes_superseded_by_default(self, graph):
        """Test that superseded nodes are excluded by default."""
        old = await graph.add_node("Old version", NodeType.FACT)
        new = await graph.supersede_node(old.node_id, "New version")

        results = await graph.search("version", limit=10)

        node_ids = {r.node.node_id for r in results}
        assert new.node_id in node_ids
        assert old.node_id not in node_ids

    @pytest.mark.anyio
    async def test_search_include_superseded(self, graph):
        """Test searching with superseded nodes included."""
        old = await graph.add_node("Old version", NodeType.FACT)
        new = await graph.supersede_node(old.node_id, "New version")

        results = await graph.search(
            "version",
            include_superseded=True,
            limit=10,
        )

        node_ids = {r.node.node_id for r in results}
        assert new.node_id in node_ids
        assert old.node_id in node_ids

    @pytest.mark.anyio
    async def test_search_with_edges(self, graph):
        """Test search including connected edges."""
        node1 = await graph.add_node("Node with edges", NodeType.ENTITY)
        node2 = await graph.add_node("Connected node", NodeType.ENTITY)
        await graph.add_edge(node1.node_id, node2.node_id, EdgeType.RELATED_TO)

        results = await graph.search(
            "Node with edges",
            include_edges=True,
        )

        assert len(results) > 0
        assert len(results[0].edges) > 0

    @pytest.mark.anyio
    async def test_search_no_embedder(self):
        """Test search without embedding provider."""
        graph = TemporalCognitiveGraph(embedding_provider=None)

        results = await graph.search("test")
        assert results == []


class TestGraphTraversal:
    """Tests for graph traversal operations."""

    @pytest.mark.anyio
    async def test_basic_traverse(self, graph):
        """Test basic graph traversal."""
        center = await graph.add_node("Center", NodeType.ENTITY)
        connected1 = await graph.add_node("Connected 1", NodeType.ENTITY)
        connected2 = await graph.add_node("Connected 2", NodeType.ENTITY)

        await graph.add_edge(center.node_id, connected1.node_id, EdgeType.HAS)
        await graph.add_edge(center.node_id, connected2.node_id, EdgeType.HAS)

        result = await graph.traverse(center.node_id, max_hops=1)

        assert len(result.nodes) == 3  # Center + 2 connected
        assert len(result.edges) == 2

    @pytest.mark.anyio
    async def test_traverse_with_depth(self, graph):
        """Test traversal with depth limit."""
        a = await graph.add_node("A", NodeType.ENTITY)
        b = await graph.add_node("B", NodeType.ENTITY)
        c = await graph.add_node("C", NodeType.ENTITY)
        d = await graph.add_node("D", NodeType.ENTITY)

        await graph.add_edge(a.node_id, b.node_id, EdgeType.RELATED_TO)
        await graph.add_edge(b.node_id, c.node_id, EdgeType.RELATED_TO)
        await graph.add_edge(c.node_id, d.node_id, EdgeType.RELATED_TO)

        # 1 hop should get A and B
        result1 = await graph.traverse(a.node_id, max_hops=1)
        assert len(result1.nodes) == 2

        # 2 hops should get A, B, C
        result2 = await graph.traverse(a.node_id, max_hops=2)
        assert len(result2.nodes) == 3

        # 3 hops should get all
        result3 = await graph.traverse(a.node_id, max_hops=3)
        assert len(result3.nodes) == 4

    @pytest.mark.anyio
    async def test_traverse_with_edge_type_filter(self, graph):
        """Test traversal with edge type filtering."""
        a = await graph.add_node("A", NodeType.ENTITY)
        b = await graph.add_node("B", NodeType.ENTITY)
        c = await graph.add_node("C", NodeType.ENTITY)

        await graph.add_edge(a.node_id, b.node_id, EdgeType.HAS)
        await graph.add_edge(a.node_id, c.node_id, EdgeType.CAUSES)

        result = await graph.traverse(
            a.node_id,
            edge_types=[EdgeType.HAS],
            max_hops=1,
        )

        assert len(result.nodes) == 2  # A and B only
        node_ids = {n.node_id for n in result.nodes}
        assert a.node_id in node_ids
        assert b.node_id in node_ids
        assert c.node_id not in node_ids

    @pytest.mark.anyio
    async def test_find_path(self, graph):
        """Test finding a path between nodes."""
        a = await graph.add_node("A", NodeType.ENTITY)
        b = await graph.add_node("B", NodeType.ENTITY)
        c = await graph.add_node("C", NodeType.ENTITY)

        await graph.add_edge(a.node_id, b.node_id, EdgeType.RELATED_TO)
        await graph.add_edge(b.node_id, c.node_id, EdgeType.RELATED_TO)

        path = await graph.find_path(a.node_id, c.node_id)

        assert path is not None
        assert len(path) == 3
        assert path[0] == a.node_id
        assert path[1] == b.node_id
        assert path[2] == c.node_id

    @pytest.mark.anyio
    async def test_find_path_no_connection(self, graph):
        """Test finding path when nodes aren't connected."""
        a = await graph.add_node("A", NodeType.ENTITY)
        b = await graph.add_node("B", NodeType.ENTITY)

        path = await graph.find_path(a.node_id, b.node_id)
        assert path is None

    @pytest.mark.anyio
    async def test_find_path_same_node(self, graph):
        """Test finding path to same node."""
        a = await graph.add_node("A", NodeType.ENTITY)

        path = await graph.find_path(a.node_id, a.node_id)

        assert path == [a.node_id]


class TestScopeOperations:
    """Tests for scope-aware operations."""

    @pytest.mark.anyio
    async def test_get_nodes_in_scope(self, graph):
        """Test getting nodes in a specific scope."""
        await graph.add_node("In scope 1", NodeType.FACT, scope_id="scope:1")
        await graph.add_node("In scope 1 again", NodeType.FACT, scope_id="scope:1")
        await graph.add_node("In scope 2", NodeType.FACT, scope_id="scope:2")

        nodes = await graph.get_nodes_in_scope("scope:1")

        assert len(nodes) == 2
        for node in nodes:
            assert node.scope_id == "scope:1"

    @pytest.mark.anyio
    async def test_promote_node(self, graph):
        """Test promoting a node to parent scope."""
        # Set up scope hierarchy
        from draagon_ai.memory import get_scope_registry, ScopeType

        registry = get_scope_registry()
        registry.create_scope(ScopeType.AGENT, "roxy", scope_id="agent:roxy")
        registry.create_scope(
            ScopeType.USER, "doug",
            scope_id="user:doug",
            parent_scope_id="agent:roxy",
        )

        # Create node in user scope
        node = await graph.add_node(
            "User-level fact",
            NodeType.FACT,
            scope_id="user:doug",
        )

        # Promote to agent scope
        result = await graph.promote_node(node.node_id)

        assert result is True
        assert node.scope_id == "agent:roxy"


class TestTemporalQueries:
    """Tests for temporal query operations."""

    @pytest.mark.anyio
    async def test_get_nodes_at_time(self, graph):
        """Test getting nodes valid at a specific time."""
        # Create node valid from a week ago
        past = datetime.now() - timedelta(days=7)
        node1 = await graph.add_node("Past fact", NodeType.FACT)
        node1.valid_from = past

        # Create node valid from yesterday
        yesterday = datetime.now() - timedelta(days=1)
        node2 = await graph.add_node("Recent fact", NodeType.FACT)
        node2.valid_from = yesterday

        # Query for 3 days ago
        three_days_ago = datetime.now() - timedelta(days=3)
        nodes = await graph.get_nodes_at_time(three_days_ago)

        # Only node1 should be included
        node_ids = {n.node_id for n in nodes}
        assert node1.node_id in node_ids
        assert node2.node_id not in node_ids

    @pytest.mark.anyio
    async def test_get_node_history(self, graph):
        """Test getting version history of a node."""
        v1 = await graph.add_node("Version 1", NodeType.FACT)
        v2 = await graph.supersede_node(v1.node_id, "Version 2")
        v3 = await graph.supersede_node(v2.node_id, "Version 3")

        history = await graph.get_node_history(v1.node_id)

        assert len(history) == 3
        # Newest first
        assert history[0].content == "Version 3"
        assert history[1].content == "Version 2"
        assert history[2].content == "Version 1"


class TestGraphStatistics:
    """Tests for graph statistics."""

    @pytest.mark.anyio
    async def test_stats(self, graph):
        """Test getting graph statistics."""
        await graph.add_node("Fact 1", NodeType.FACT, scope_id="scope:1")
        await graph.add_node("Fact 2", NodeType.FACT, scope_id="scope:1")
        await graph.add_node("Episode 1", NodeType.EPISODE, scope_id="scope:2")
        node = await graph.add_node("Old", NodeType.FACT)
        await graph.supersede_node(node.node_id, "New")

        stats = graph.stats()

        assert stats["total_nodes"] == 5
        assert stats["nodes_by_type"]["fact"] == 4
        assert stats["nodes_by_type"]["episode"] == 1
        assert stats["nodes_by_scope"]["scope:1"] == 2
        assert stats["superseded_nodes"] == 1

    @pytest.mark.anyio
    async def test_clear(self, graph):
        """Test clearing the graph."""
        await graph.add_node("Node 1", NodeType.FACT)
        await graph.add_node("Node 2", NodeType.FACT)

        graph.clear()

        stats = graph.stats()
        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0


class TestHierarchicalScopeInheritance:
    """Tests for hierarchical scope inheritance in search.

    When searching with a user scope like "user:roxy:doug", nodes from
    ancestor scopes (agent:roxy, world:global) should also be included.
    This implements the hierarchical memory model where child scopes
    can see parent scope nodes.
    """

    @pytest.fixture
    def graph_with_hierarchy(self, mock_embedder):
        """Create a graph with a proper scope hierarchy."""
        from draagon_ai.memory import get_scope_registry, ScopeType

        reset_scope_registry()
        registry = get_scope_registry()

        # Build hierarchy: world:global -> context:home -> agent:roxy -> user:roxy:doug
        registry.create_scope(ScopeType.CONTEXT, "home", scope_id="context:home")
        registry.create_scope(
            ScopeType.AGENT, "roxy",
            scope_id="agent:roxy",
            parent_scope_id="context:home",
        )
        registry.create_scope(
            ScopeType.USER, "doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )

        return TemporalCognitiveGraph(
            embedding_provider=mock_embedder,
            scope_registry=registry,
        )

    @pytest.mark.anyio
    async def test_search_includes_parent_scopes(self, graph_with_hierarchy):
        """Test that search includes nodes from parent scopes."""
        # Add nodes at different scope levels
        world_node = await graph_with_hierarchy.add_node(
            "The Earth orbits the Sun",
            NodeType.FACT,
            scope_id="world:global",
        )
        context_node = await graph_with_hierarchy.add_node(
            "Home WiFi password is abc123",
            NodeType.FACT,
            scope_id="context:home",
        )
        agent_node = await graph_with_hierarchy.add_node(
            "Roxy knows about cooking",
            NodeType.FACT,
            scope_id="agent:roxy",
        )
        user_node = await graph_with_hierarchy.add_node(
            "Doug's birthday is March 15",
            NodeType.FACT,
            scope_id="user:roxy:doug",
        )

        # Search from user scope - should include all ancestor scopes
        results = await graph_with_hierarchy.search(
            "fact",
            scope_ids=["user:roxy:doug"],
            limit=10,
        )

        # Should find all 4 nodes (user + agent + context + world)
        found_ids = {r.node.node_id for r in results}
        assert world_node.node_id in found_ids, "World scope nodes should be visible"
        assert context_node.node_id in found_ids, "Context scope nodes should be visible"
        assert agent_node.node_id in found_ids, "Agent scope nodes should be visible"
        assert user_node.node_id in found_ids, "User scope nodes should be visible"

    @pytest.mark.anyio
    async def test_search_from_agent_scope(self, graph_with_hierarchy):
        """Test that agent scope search includes parent scopes but not child scopes."""
        # Add nodes at different levels
        world_node = await graph_with_hierarchy.add_node(
            "Universal truth",
            NodeType.FACT,
            scope_id="world:global",
        )
        context_node = await graph_with_hierarchy.add_node(
            "Home context fact",
            NodeType.FACT,
            scope_id="context:home",
        )
        agent_node = await graph_with_hierarchy.add_node(
            "Agent level fact",
            NodeType.FACT,
            scope_id="agent:roxy",
        )
        user_node = await graph_with_hierarchy.add_node(
            "User level fact",
            NodeType.FACT,
            scope_id="user:roxy:doug",
        )

        # Search from agent scope
        results = await graph_with_hierarchy.search(
            "fact",
            scope_ids=["agent:roxy"],
            limit=10,
        )

        found_ids = {r.node.node_id for r in results}
        # Agent scope sees world, context, and agent nodes
        assert world_node.node_id in found_ids
        assert context_node.node_id in found_ids
        assert agent_node.node_id in found_ids
        # But NOT user nodes (child scope)
        assert user_node.node_id not in found_ids

    @pytest.mark.anyio
    async def test_search_can_disable_ancestor_scopes(self, graph_with_hierarchy):
        """Test that ancestor scope inclusion can be disabled."""
        # Add nodes
        await graph_with_hierarchy.add_node(
            "World fact",
            NodeType.FACT,
            scope_id="world:global",
        )
        user_node = await graph_with_hierarchy.add_node(
            "User fact",
            NodeType.FACT,
            scope_id="user:roxy:doug",
        )

        # Search with ancestor scopes disabled
        results = await graph_with_hierarchy.search(
            "fact",
            scope_ids=["user:roxy:doug"],
            include_ancestor_scopes=False,
            limit=10,
        )

        found_ids = {r.node.node_id for r in results}
        # Only user scope node should be found
        assert len(found_ids) == 1
        assert user_node.node_id in found_ids

    @pytest.mark.anyio
    async def test_search_multiple_scopes_with_inheritance(self, graph_with_hierarchy):
        """Test searching multiple scopes with inheritance."""
        from draagon_ai.memory import get_scope_registry, ScopeType

        registry = get_scope_registry()

        # Create another user under the same agent
        registry.create_scope(
            ScopeType.USER, "maya",
            scope_id="user:roxy:maya",
            parent_scope_id="agent:roxy",
        )

        # Add nodes for both users
        doug_node = await graph_with_hierarchy.add_node(
            "Doug's personal fact",
            NodeType.FACT,
            scope_id="user:roxy:doug",
        )
        maya_node = await graph_with_hierarchy.add_node(
            "Maya's personal fact",
            NodeType.FACT,
            scope_id="user:roxy:maya",
        )
        agent_node = await graph_with_hierarchy.add_node(
            "Shared agent fact",
            NodeType.FACT,
            scope_id="agent:roxy",
        )

        # Search from both user scopes
        results = await graph_with_hierarchy.search(
            "fact",
            scope_ids=["user:roxy:doug", "user:roxy:maya"],
            limit=10,
        )

        found_ids = {r.node.node_id for r in results}
        # Both user nodes should be found
        assert doug_node.node_id in found_ids
        assert maya_node.node_id in found_ids
        # Agent node should also be found (inherited by both)
        assert agent_node.node_id in found_ids


class TestPermissionEnforcement:
    """Tests for permission enforcement in graph operations.

    When `enforce_permissions=True`, graph operations should check that
    the agent/user has the required permission before allowing the operation.
    """

    @pytest.fixture
    def graph_with_permissions(self, mock_embedder):
        """Create a graph with permission structure for testing."""
        from draagon_ai.memory import (
            get_scope_registry,
            ScopeType,
            Permission,
            PermissionDeniedError,
        )

        reset_scope_registry()
        registry = get_scope_registry()

        # Create scope hierarchy with permissions
        registry.create_scope(
            ScopeType.AGENT, "roxy",
            scope_id="agent:roxy",
        )

        # Get the agent scope and grant permissions
        agent_scope = registry.get("agent:roxy")
        agent_scope.grant_permission(
            "roxy", "agent",
            {Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN},
        )

        # Create user scope under agent
        registry.create_scope(
            ScopeType.USER, "doug",
            scope_id="user:roxy:doug",
            parent_scope_id="agent:roxy",
        )

        # Grant doug permissions on his user scope
        user_scope = registry.get("user:roxy:doug")
        user_scope.grant_permission(
            "doug", "user",
            {Permission.READ, Permission.WRITE},
        )

        return TemporalCognitiveGraph(
            embedding_provider=mock_embedder,
            scope_registry=registry,
        )

    @pytest.mark.anyio
    async def test_add_node_with_permission(self, graph_with_permissions):
        """Test that add_node succeeds with proper permissions."""
        # Agent has WRITE permission on agent:roxy
        node = await graph_with_permissions.add_node(
            content="Test content",
            node_type=NodeType.FACT,
            scope_id="agent:roxy",
            agent_id="roxy",
            enforce_permissions=True,
        )

        assert node is not None
        assert node.content == "Test content"

    @pytest.mark.anyio
    async def test_add_node_without_permission_raises(self, graph_with_permissions):
        """Test that add_node raises PermissionDeniedError without proper permissions."""
        from draagon_ai.memory import PermissionDeniedError

        # "other_agent" has no permissions on agent:roxy
        with pytest.raises(PermissionDeniedError) as exc_info:
            await graph_with_permissions.add_node(
                content="Test content",
                node_type=NodeType.FACT,
                scope_id="agent:roxy",
                agent_id="other_agent",
                enforce_permissions=True,
            )

        assert exc_info.value.operation == "add_node"
        assert exc_info.value.scope_id == "agent:roxy"
        assert exc_info.value.agent_id == "other_agent"

    @pytest.mark.anyio
    async def test_add_node_enforcement_disabled_by_default(self, graph_with_permissions):
        """Test that permission enforcement is disabled by default."""
        # No agent_id provided, but no error raised because enforcement is off
        node = await graph_with_permissions.add_node(
            content="Test content",
            node_type=NodeType.FACT,
            scope_id="agent:roxy",
        )

        assert node is not None

    @pytest.mark.anyio
    async def test_add_node_user_with_inherited_permission(self, graph_with_permissions):
        """Test that users can add nodes using inherited permissions."""
        # doug has WRITE on user:roxy:doug
        node = await graph_with_permissions.add_node(
            content="Doug's fact",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            agent_id="roxy",
            user_id="doug",
            enforce_permissions=True,
        )

        assert node is not None

    @pytest.mark.anyio
    async def test_delete_node_with_permission(self, graph_with_permissions):
        """Test that delete_node succeeds with proper permissions."""
        # First add a node
        node = await graph_with_permissions.add_node(
            content="To be deleted",
            node_type=NodeType.FACT,
            scope_id="agent:roxy",
        )

        # Agent has DELETE permission on agent:roxy
        result = await graph_with_permissions.delete_node(
            node.node_id,
            agent_id="roxy",
            enforce_permissions=True,
        )

        assert result is True

        # Verify node is gone
        retrieved = await graph_with_permissions.get_node(node.node_id)
        assert retrieved is None

    @pytest.mark.anyio
    async def test_delete_node_unauthorized_agent_raises(self, graph_with_permissions):
        """Test that delete_node raises PermissionDeniedError for unauthorized agent."""
        from draagon_ai.memory import PermissionDeniedError

        # First add a node
        node = await graph_with_permissions.add_node(
            content="Cannot be deleted by other agents",
            node_type=NodeType.FACT,
            scope_id="agent:roxy",
        )

        # "other_agent" has no permissions on agent:roxy
        with pytest.raises(PermissionDeniedError) as exc_info:
            await graph_with_permissions.delete_node(
                node.node_id,
                agent_id="other_agent",
                enforce_permissions=True,
            )

        assert exc_info.value.operation == "delete_node"
        assert exc_info.value.scope_id == "agent:roxy"
        assert exc_info.value.agent_id == "other_agent"

    @pytest.mark.anyio
    async def test_delete_node_user_only_without_agent_permission(self, graph_with_permissions):
        """Test that user-only permission check fails when user lacks DELETE."""
        from draagon_ai.memory import (
            get_scope_registry,
            ScopeType,
            Permission,
            PermissionDeniedError,
        )

        # Create an isolated scope where only user has permissions (no agent fallback)
        registry = get_scope_registry()
        registry.create_scope(
            ScopeType.USER, "isolated",
            scope_id="user:isolated",
            parent_scope_id="world:global",  # Parent is world, not agent
        )

        # Grant only READ/WRITE to isolated_user (no DELETE)
        isolated_scope = registry.get("user:isolated")
        isolated_scope.grant_permission(
            "isolated_user", "user",
            {Permission.READ, Permission.WRITE},
        )

        # Add a node in the isolated scope
        node = await graph_with_permissions.add_node(
            content="Isolated user's node",
            node_type=NodeType.FACT,
            scope_id="user:isolated",
        )

        # isolated_user has READ/WRITE but NOT DELETE
        with pytest.raises(PermissionDeniedError):
            await graph_with_permissions.delete_node(
                node.node_id,
                agent_id="isolated_agent",  # Agent with no permissions
                user_id="isolated_user",    # User without DELETE
                enforce_permissions=True,
            )

    @pytest.mark.anyio
    async def test_agent_inherits_delete_permission_to_child_scope(self, graph_with_permissions):
        """Test that agent with DELETE on parent can delete in child scope.

        This verifies that permission inheritance works correctly:
        - roxy has DELETE on agent:roxy
        - user:roxy:doug is a child of agent:roxy
        - Therefore roxy can delete nodes in user:roxy:doug
        """
        # Add a node in doug's scope
        node = await graph_with_permissions.add_node(
            content="Doug's node",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
        )

        # roxy (the agent) has DELETE permission via inheritance from agent:roxy
        result = await graph_with_permissions.delete_node(
            node.node_id,
            agent_id="roxy",
            enforce_permissions=True,
        )

        assert result is True
        retrieved = await graph_with_permissions.get_node(node.node_id)
        assert retrieved is None

    @pytest.mark.anyio
    async def test_delete_node_nonexistent_returns_false(self, graph_with_permissions):
        """Test that deleting non-existent node returns False without checking permissions."""
        # Should return False, not raise PermissionDeniedError
        result = await graph_with_permissions.delete_node(
            "nonexistent-id",
            agent_id="unauthorized_agent",
            enforce_permissions=True,
        )

        assert result is False

    @pytest.mark.anyio
    async def test_permission_denied_error_message(self, graph_with_permissions):
        """Test that PermissionDeniedError has useful message."""
        from draagon_ai.memory import PermissionDeniedError

        with pytest.raises(PermissionDeniedError) as exc_info:
            await graph_with_permissions.add_node(
                content="Test",
                node_type=NodeType.FACT,
                scope_id="agent:roxy",
                agent_id="evil_agent",
                user_id="evil_user",
                enforce_permissions=True,
            )

        error = exc_info.value
        message = str(error)

        assert "add_node" in message
        assert "agent:roxy" in message
        assert "evil_agent" in message
        assert "evil_user" in message

    @pytest.mark.anyio
    async def test_agent_inherited_permission_from_parent(self, graph_with_permissions):
        """Test that agent can access child scopes via inherited permissions."""
        # roxy has WRITE on agent:roxy, which should inherit to user:roxy:doug
        node = await graph_with_permissions.add_node(
            content="Agent writing to user scope",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            agent_id="roxy",
            enforce_permissions=True,
        )

        assert node is not None
        assert node.scope_id == "user:roxy:doug"

    @pytest.mark.anyio
    async def test_permission_check_helper_method(self, graph_with_permissions):
        """Test the _check_permission helper directly."""
        from draagon_ai.memory import Permission, PermissionDeniedError

        # Should return True for valid permission
        result = graph_with_permissions._check_permission(
            scope_id="agent:roxy",
            agent_id="roxy",
            user_id=None,
            permission=Permission.WRITE,
            operation="test_operation",
            enforce=False,  # Don't raise, just return
        )
        assert result is True

        # Should return False for invalid permission (without raising)
        result = graph_with_permissions._check_permission(
            scope_id="agent:roxy",
            agent_id="unauthorized",
            user_id=None,
            permission=Permission.WRITE,
            operation="test_operation",
            enforce=False,
        )
        assert result is False

        # Should raise when enforce=True
        with pytest.raises(PermissionDeniedError):
            graph_with_permissions._check_permission(
                scope_id="agent:roxy",
                agent_id="unauthorized",
                user_id=None,
                permission=Permission.WRITE,
                operation="test_operation",
                enforce=True,
            )
