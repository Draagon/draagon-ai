"""Unit tests for SemanticGraph class.

Comprehensive tests for graph operations including:
- Node and edge management
- Query operations
- Traversal
- Bi-temporal queries
- Serialization
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from draagon_ai.cognition.decomposition.graph import (
    SemanticGraph,
    GraphNode,
    GraphEdge,
    NodeType,
    TraversalResult,
)
from draagon_ai.cognition.decomposition.identifiers import EntityType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_graph() -> SemanticGraph:
    """Create an empty graph."""
    return SemanticGraph()


@pytest.fixture
def simple_graph() -> SemanticGraph:
    """Create a simple graph with Doug and his cats.

    Structure:
        Doug --[has]--> Whiskers
        Doug --[has]--> Mittens
        Whiskers --[is_a]--> Cat
        Mittens --[is_a]--> Cat
    """
    graph = SemanticGraph()

    # Create nodes
    doug = graph.create_node("Doug", NodeType.INSTANCE, EntityType.INSTANCE)
    whiskers = graph.create_node("Whiskers", NodeType.INSTANCE, EntityType.INSTANCE)
    mittens = graph.create_node("Mittens", NodeType.INSTANCE, EntityType.INSTANCE)
    cat = graph.create_node("Cat", NodeType.CLASS, EntityType.CLASS)

    # Create relationships
    graph.create_edge(doug.node_id, whiskers.node_id, "has")
    graph.create_edge(doug.node_id, mittens.node_id, "has")
    graph.create_edge(whiskers.node_id, cat.node_id, "is_a")
    graph.create_edge(mittens.node_id, cat.node_id, "is_a")

    return graph


@pytest.fixture
def temporal_graph() -> SemanticGraph:
    """Create a graph with temporal edges for historical queries."""
    graph = SemanticGraph()

    doug = graph.create_node("Doug", NodeType.INSTANCE)

    # Past addresses
    portland = graph.create_node("Portland", NodeType.INSTANCE)
    seattle = graph.create_node("Seattle", NodeType.INSTANCE)

    # Edge from 2020-2023
    edge1 = graph.create_edge(
        doug.node_id,
        portland.node_id,
        "lives_in",
        valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
    )
    graph.invalidate_edge(edge1.edge_id, datetime(2023, 6, 1, tzinfo=timezone.utc))

    # Edge from 2023-current
    graph.create_edge(
        doug.node_id,
        seattle.node_id,
        "lives_in",
        valid_from=datetime(2023, 6, 1, tzinfo=timezone.utc),
    )

    return graph


# =============================================================================
# Node Operations Tests
# =============================================================================


class TestNodeOperations:
    """Tests for node CRUD operations."""

    def test_add_node(self, empty_graph):
        """Test adding a node to the graph."""
        node = GraphNode(canonical_name="Test")
        empty_graph.add_node(node)

        assert empty_graph.node_count == 1
        assert node.node_id in empty_graph

    def test_create_node(self, empty_graph):
        """Test creating a node directly."""
        node = empty_graph.create_node(
            canonical_name="Doug",
            node_type=NodeType.INSTANCE,
            properties={"age": 35},
        )

        assert empty_graph.node_count == 1
        assert node.canonical_name == "Doug"
        assert node.properties["age"] == 35

    def test_get_node(self, simple_graph):
        """Test getting a node by ID."""
        doug = simple_graph.find_node("Doug")
        assert doug is not None

        retrieved = simple_graph.get_node(doug.node_id)
        assert retrieved == doug

    def test_get_nonexistent_node(self, empty_graph):
        """Test getting a node that doesn't exist."""
        result = empty_graph.get_node("nonexistent-id")
        assert result is None

    def test_remove_node(self, simple_graph):
        """Test removing a node and its edges."""
        doug = simple_graph.find_node("Doug")
        initial_edge_count = simple_graph.edge_count

        result = simple_graph.remove_node(doug.node_id)

        assert result is True
        assert doug.node_id not in simple_graph
        # Edges connected to Doug should be removed
        assert simple_graph.edge_count < initial_edge_count

    def test_remove_nonexistent_node(self, empty_graph):
        """Test removing a node that doesn't exist."""
        result = empty_graph.remove_node("nonexistent")
        assert result is False

    def test_replace_node(self, empty_graph):
        """Test that adding a node with same ID replaces it."""
        node1 = GraphNode(node_id="same-id", canonical_name="Original")
        node2 = GraphNode(node_id="same-id", canonical_name="Updated")

        empty_graph.add_node(node1)
        empty_graph.add_node(node2)

        assert empty_graph.node_count == 1
        retrieved = empty_graph.get_node("same-id")
        assert retrieved.canonical_name == "Updated"


# =============================================================================
# Node Search Tests
# =============================================================================


class TestNodeSearch:
    """Tests for node search operations."""

    def test_find_node_by_name(self, simple_graph):
        """Test finding a node by exact name."""
        doug = simple_graph.find_node("Doug")
        assert doug is not None
        assert doug.canonical_name == "Doug"

    def test_find_node_case_insensitive(self, simple_graph):
        """Test case-insensitive name search."""
        assert simple_graph.find_node("doug") is not None
        assert simple_graph.find_node("DOUG") is not None
        assert simple_graph.find_node("Doug") is not None

    def test_find_node_not_found(self, simple_graph):
        """Test finding a node that doesn't exist."""
        result = simple_graph.find_node("Bob")
        assert result is None

    def test_find_node_with_type_filter(self, simple_graph):
        """Test finding nodes with type filter."""
        cat_entity = simple_graph.find_node("Cat", node_type=NodeType.CLASS)
        assert cat_entity is not None

        # Should not find Cat if filtering for ENTITY
        cat_as_entity = simple_graph.find_node("Cat", node_type=NodeType.INSTANCE)
        assert cat_as_entity is None

    def test_find_nodes_multiple_matches(self):
        """Test finding multiple nodes with same name."""
        graph = SemanticGraph()
        graph.create_node("Test", NodeType.INSTANCE)
        graph.create_node("Test", NodeType.CLASS)

        results = graph.find_nodes("Test")
        assert len(results) == 2

    def test_find_nodes_by_type(self, simple_graph):
        """Test finding all nodes of a type."""
        entities = simple_graph.find_nodes_by_type(NodeType.INSTANCE)
        concepts = simple_graph.find_nodes_by_type(NodeType.CLASS)

        assert len(entities) == 3  # Doug, Whiskers, Mittens
        assert len(concepts) == 1  # Cat

    def test_find_nodes_by_property(self, empty_graph):
        """Test finding nodes by property value."""
        empty_graph.create_node("Cat1", properties={"color": "orange"})
        empty_graph.create_node("Cat2", properties={"color": "orange"})
        empty_graph.create_node("Cat3", properties={"color": "black"})

        orange_cats = empty_graph.find_nodes_by_property("color", "orange")
        assert len(orange_cats) == 2

    def test_find_nodes_by_synset(self, empty_graph):
        """Test finding nodes by WordNet synset.

        With the new ontology, synset nodes are CLASS type. The find_nodes_by_synset
        defaults to class_only=True to enable proper deduplication.
        """
        # Create CLASS nodes (synset nodes) with synset_id
        empty_graph.create_node("cat.n.01", node_type=NodeType.CLASS, synset_id="cat.n.01")
        empty_graph.create_node("dog.n.01", node_type=NodeType.CLASS, synset_id="dog.n.01")
        empty_graph.create_node("feline.n.01", node_type=NodeType.CLASS, synset_id="cat.n.01")

        # Default behavior: only return CLASS nodes
        cats = empty_graph.find_nodes_by_synset("cat.n.01")
        assert len(cats) == 2

        # Can also find all nodes with synset_id if needed
        cats_all = empty_graph.find_nodes_by_synset("cat.n.01", class_only=False)
        assert len(cats_all) == 2

    def test_find_node_by_alias(self):
        """Test finding nodes by alias."""
        graph = SemanticGraph()
        graph.create_node(
            "Douglas",
            properties={"aliases": ["Doug", "Dougie"]},
        )

        assert graph.find_node("Douglas") is not None
        assert graph.find_node("Doug") is not None
        assert graph.find_node("Dougie") is not None
        assert graph.find_node("Bob") is None


# =============================================================================
# Edge Operations Tests
# =============================================================================


class TestEdgeOperations:
    """Tests for edge CRUD operations."""

    def test_add_edge(self, empty_graph):
        """Test adding an edge."""
        node1 = empty_graph.create_node("A")
        node2 = empty_graph.create_node("B")

        edge = GraphEdge(
            source_node_id=node1.node_id,
            target_node_id=node2.node_id,
            relation_type="connects_to",
        )
        empty_graph.add_edge(edge)

        assert empty_graph.edge_count == 1

    def test_add_edge_missing_source(self, empty_graph):
        """Test adding edge with missing source node."""
        node = empty_graph.create_node("A")
        edge = GraphEdge(
            source_node_id="nonexistent",
            target_node_id=node.node_id,
            relation_type="test",
        )

        with pytest.raises(ValueError, match="Source node"):
            empty_graph.add_edge(edge)

    def test_add_edge_missing_target(self, empty_graph):
        """Test adding edge with missing target node."""
        node = empty_graph.create_node("A")
        edge = GraphEdge(
            source_node_id=node.node_id,
            target_node_id="nonexistent",
            relation_type="test",
        )

        with pytest.raises(ValueError, match="Target node"):
            empty_graph.add_edge(edge)

    def test_create_edge(self, empty_graph):
        """Test creating an edge directly."""
        a = empty_graph.create_node("A")
        b = empty_graph.create_node("B")

        edge = empty_graph.create_edge(
            a.node_id, b.node_id, "connects_to", properties={"weight": 1.0}
        )

        assert edge.source_node_id == a.node_id
        assert edge.target_node_id == b.node_id
        assert edge.relation_type == "connects_to"
        assert edge.properties["weight"] == 1.0

    def test_get_edge(self, simple_graph):
        """Test getting an edge by ID."""
        doug = simple_graph.find_node("Doug")
        edges = simple_graph.get_outgoing_edges(doug.node_id)
        assert len(edges) > 0

        edge = edges[0]
        retrieved = simple_graph.get_edge(edge.edge_id)
        assert retrieved == edge

    def test_remove_edge(self, simple_graph):
        """Test removing an edge."""
        doug = simple_graph.find_node("Doug")
        edges = simple_graph.get_outgoing_edges(doug.node_id)
        initial_count = len(edges)

        result = simple_graph.remove_edge(edges[0].edge_id)

        assert result is True
        new_edges = simple_graph.get_outgoing_edges(doug.node_id)
        assert len(new_edges) == initial_count - 1

    def test_remove_nonexistent_edge(self, empty_graph):
        """Test removing a nonexistent edge."""
        result = empty_graph.remove_edge("nonexistent")
        assert result is False


# =============================================================================
# Edge Query Tests
# =============================================================================


class TestEdgeQueries:
    """Tests for edge query operations."""

    def test_get_outgoing_edges(self, simple_graph):
        """Test getting outgoing edges from a node."""
        doug = simple_graph.find_node("Doug")
        edges = simple_graph.get_outgoing_edges(doug.node_id)

        assert len(edges) == 2  # has -> Whiskers, has -> Mittens
        for edge in edges:
            assert edge.source_node_id == doug.node_id

    def test_get_outgoing_edges_with_type_filter(self, simple_graph):
        """Test filtering outgoing edges by relation type."""
        doug = simple_graph.find_node("Doug")

        has_edges = simple_graph.get_outgoing_edges(doug.node_id, relation_type="has")
        assert len(has_edges) == 2

        is_a_edges = simple_graph.get_outgoing_edges(doug.node_id, relation_type="is_a")
        assert len(is_a_edges) == 0

    def test_get_incoming_edges(self, simple_graph):
        """Test getting incoming edges to a node."""
        cat = simple_graph.find_node("Cat")
        edges = simple_graph.get_incoming_edges(cat.node_id)

        assert len(edges) == 2  # is_a from Whiskers, is_a from Mittens
        for edge in edges:
            assert edge.target_node_id == cat.node_id

    def test_get_edges_between(self, simple_graph):
        """Test getting edges between two specific nodes."""
        doug = simple_graph.find_node("Doug")
        whiskers = simple_graph.find_node("Whiskers")

        edges = simple_graph.get_edges_between(doug.node_id, whiskers.node_id)
        assert len(edges) == 1
        assert edges[0].relation_type == "has"

    def test_get_edges_at_time(self, temporal_graph):
        """Test getting edges valid at a specific time."""
        doug = temporal_graph.find_node("Doug")

        # In 2021: Should be in Portland
        edges_2021 = temporal_graph.get_edges_at_time(
            datetime(2021, 1, 1, tzinfo=timezone.utc),
            node_id=doug.node_id,
        )
        assert len(edges_2021) == 1
        target_node = temporal_graph.get_node(edges_2021[0].target_node_id)
        assert target_node.canonical_name == "Portland"

        # In 2024: Should be in Seattle
        edges_2024 = temporal_graph.get_edges_at_time(
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            node_id=doug.node_id,
        )
        assert len(edges_2024) == 1
        target_node = temporal_graph.get_node(edges_2024[0].target_node_id)
        assert target_node.canonical_name == "Seattle"

    def test_invalidate_edge(self, simple_graph):
        """Test invalidating an edge."""
        doug = simple_graph.find_node("Doug")
        edges = simple_graph.get_outgoing_edges(doug.node_id)
        edge_id = edges[0].edge_id

        result = simple_graph.invalidate_edge(edge_id)
        assert result is True

        # Current edges should exclude the invalidated one
        current_edges = simple_graph.get_outgoing_edges(
            doug.node_id, current_only=True
        )
        assert len(current_edges) == 1

        # All edges should include the invalidated one
        all_edges = simple_graph.get_outgoing_edges(doug.node_id, current_only=False)
        assert len(all_edges) == 2


# =============================================================================
# Relation Counting Tests
# =============================================================================


class TestRelationCounting:
    """Tests for count_relations operation (how many queries)."""

    def test_count_relations(self, simple_graph):
        """Test counting relations of a specific type."""
        doug = simple_graph.find_node("Doug")
        count = simple_graph.count_relations(doug.node_id, "has")
        assert count == 2

    def test_count_relations_with_target_type(self, simple_graph):
        """Test counting relations with target type filter."""
        doug = simple_graph.find_node("Doug")
        count = simple_graph.count_relations(
            doug.node_id, "has", target_type=NodeType.INSTANCE
        )
        assert count == 2

        count_concepts = simple_graph.count_relations(
            doug.node_id, "has", target_type=NodeType.CLASS
        )
        assert count_concepts == 0

    def test_count_relations_no_matches(self, simple_graph):
        """Test counting when no relations match."""
        doug = simple_graph.find_node("Doug")
        count = simple_graph.count_relations(doug.node_id, "unknown_relation")
        assert count == 0

    def test_count_excludes_expired_edges(self, temporal_graph):
        """Test that counting excludes expired edges by default."""
        doug = temporal_graph.find_node("Doug")
        count = temporal_graph.count_relations(doug.node_id, "lives_in")
        assert count == 1  # Only current (Seattle)

        count_all = temporal_graph.count_relations(
            doug.node_id, "lives_in", current_only=False
        )
        assert count_all == 2  # Both Portland and Seattle


# =============================================================================
# Related Nodes Tests
# =============================================================================


class TestRelatedNodes:
    """Tests for getting related nodes."""

    def test_get_related_nodes_outgoing(self, simple_graph):
        """Test getting nodes related via outgoing edges."""
        doug = simple_graph.find_node("Doug")
        related = simple_graph.get_related_nodes(
            doug.node_id, direction="outgoing"
        )

        names = {n.canonical_name for n in related}
        assert "Whiskers" in names
        assert "Mittens" in names

    def test_get_related_nodes_incoming(self, simple_graph):
        """Test getting nodes related via incoming edges."""
        cat = simple_graph.find_node("Cat")
        related = simple_graph.get_related_nodes(
            cat.node_id, direction="incoming"
        )

        names = {n.canonical_name for n in related}
        assert "Whiskers" in names
        assert "Mittens" in names

    def test_get_related_nodes_both_directions(self, simple_graph):
        """Test getting nodes related in both directions."""
        whiskers = simple_graph.find_node("Whiskers")
        related = simple_graph.get_related_nodes(
            whiskers.node_id, direction="both"
        )

        names = {n.canonical_name for n in related}
        assert "Doug" in names  # incoming "has" from Doug
        assert "Cat" in names   # outgoing "is_a" to Cat

    def test_get_related_nodes_with_type_filter(self, simple_graph):
        """Test filtering related nodes by relation type."""
        doug = simple_graph.find_node("Doug")

        # Only "has" relations
        has_related = simple_graph.get_related_nodes(
            doug.node_id, relation_type="has"
        )
        assert len(has_related) == 2

        # Only "is_a" relations (Doug has none)
        is_a_related = simple_graph.get_related_nodes(
            doug.node_id, relation_type="is_a"
        )
        assert len(is_a_related) == 0


# =============================================================================
# Graph Traversal Tests
# =============================================================================


class TestTraversal:
    """Tests for graph traversal operations."""

    def test_traverse_basic(self, simple_graph):
        """Test basic graph traversal."""
        doug = simple_graph.find_node("Doug")
        result = simple_graph.traverse(doug.node_id, max_depth=2)

        assert isinstance(result, TraversalResult)
        assert len(result.nodes) > 0
        assert result.depth <= 2

    def test_traverse_with_relation_path(self, simple_graph):
        """Test traversal following specific relation path."""
        doug = simple_graph.find_node("Doug")

        # Follow has -> is_a
        result = simple_graph.traverse(
            doug.node_id,
            relation_path=["has", "is_a"],
        )

        # Should find Cat via Whiskers/Mittens
        node_names = {n.canonical_name for n in result.nodes}
        assert "Cat" in node_names

    def test_traverse_max_depth(self, simple_graph):
        """Test traversal respects max depth."""
        doug = simple_graph.find_node("Doug")

        # Depth 1: should only get immediate connections
        result_1 = simple_graph.traverse(doug.node_id, max_depth=1)
        # Depth 2: should get connections of connections
        result_2 = simple_graph.traverse(doug.node_id, max_depth=2)

        assert result_1.depth <= 1
        assert len(result_2.nodes) >= len(result_1.nodes)

    def test_traverse_with_node_filter(self, simple_graph):
        """Test traversal with custom node filter."""
        doug = simple_graph.find_node("Doug")

        # Only include ENTITY nodes
        result = simple_graph.traverse(
            doug.node_id,
            max_depth=3,
            node_filter=lambda n: n.node_type == NodeType.INSTANCE,
        )

        for node in result.nodes:
            assert node.node_type == NodeType.INSTANCE

    def test_traverse_nonexistent_start(self, empty_graph):
        """Test traversal from nonexistent node."""
        result = empty_graph.traverse("nonexistent")
        assert len(result.nodes) == 0


# =============================================================================
# Path Finding Tests
# =============================================================================


class TestPathFinding:
    """Tests for path finding operations."""

    def test_find_path_same_node(self, simple_graph):
        """Test finding path to same node."""
        doug = simple_graph.find_node("Doug")
        path = simple_graph.find_path(doug.node_id, doug.node_id)
        assert path == []

    def test_find_path_direct(self, simple_graph):
        """Test finding direct path."""
        doug = simple_graph.find_node("Doug")
        whiskers = simple_graph.find_node("Whiskers")

        path = simple_graph.find_path(doug.node_id, whiskers.node_id)

        assert path is not None
        assert len(path) == 1
        assert path[0].source_node_id == doug.node_id
        assert path[0].target_node_id == whiskers.node_id

    def test_find_path_multi_hop(self, simple_graph):
        """Test finding multi-hop path."""
        doug = simple_graph.find_node("Doug")
        cat = simple_graph.find_node("Cat")

        path = simple_graph.find_path(doug.node_id, cat.node_id)

        assert path is not None
        assert len(path) == 2  # Doug -> Whiskers/Mittens -> Cat

    def test_find_path_no_path(self):
        """Test when no path exists."""
        graph = SemanticGraph()
        a = graph.create_node("A")
        b = graph.create_node("B")
        # No edge between them

        path = graph.find_path(a.node_id, b.node_id)
        assert path is None

    def test_find_path_max_depth(self, simple_graph):
        """Test path finding respects max depth."""
        doug = simple_graph.find_node("Doug")
        cat = simple_graph.find_node("Cat")

        # Path requires 2 hops, max_depth=1 should fail
        path = simple_graph.find_path(doug.node_id, cat.node_id, max_depth=1)
        assert path is None

        # max_depth=2 should succeed
        path = simple_graph.find_path(doug.node_id, cat.node_id, max_depth=2)
        assert path is not None


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for graph statistics."""

    def test_node_count(self, simple_graph):
        """Test node count property."""
        assert simple_graph.node_count == 4  # Doug, Whiskers, Mittens, Cat

    def test_edge_count(self, simple_graph):
        """Test edge count property."""
        assert simple_graph.edge_count == 4  # 2 has + 2 is_a

    def test_current_edge_count(self, temporal_graph):
        """Test current edge count (excludes expired)."""
        assert temporal_graph.edge_count == 2  # Total
        assert temporal_graph.current_edge_count == 1  # Only Seattle

    def test_node_type_counts(self, simple_graph):
        """Test counting nodes by type."""
        counts = simple_graph.get_node_type_counts()

        assert counts[NodeType.INSTANCE] == 3
        assert counts[NodeType.CLASS] == 1

    def test_relation_type_counts(self, simple_graph):
        """Test counting edges by relation type."""
        counts = simple_graph.get_relation_type_counts()

        assert counts["has"] == 2
        assert counts["is_a"] == 2


# =============================================================================
# Iteration Tests
# =============================================================================


class TestIteration:
    """Tests for graph iteration."""

    def test_iter_nodes(self, simple_graph):
        """Test iterating over all nodes."""
        nodes = list(simple_graph.iter_nodes())
        assert len(nodes) == 4

    def test_iter_nodes_with_type_filter(self, simple_graph):
        """Test iterating with type filter."""
        entities = list(simple_graph.iter_nodes(node_type=NodeType.INSTANCE))
        assert len(entities) == 3

    def test_iter_edges(self, simple_graph):
        """Test iterating over all edges."""
        edges = list(simple_graph.iter_edges())
        assert len(edges) == 4

    def test_iter_edges_with_type_filter(self, simple_graph):
        """Test iterating edges with relation filter."""
        has_edges = list(simple_graph.iter_edges(relation_type="has"))
        assert len(has_edges) == 2

    def test_iter_edges_current_only(self, temporal_graph):
        """Test iterating only current edges."""
        current = list(temporal_graph.iter_edges(current_only=True))
        all_edges = list(temporal_graph.iter_edges(current_only=False))

        assert len(current) == 1
        assert len(all_edges) == 2


# =============================================================================
# Serialization Tests
# =============================================================================


class TestSerialization:
    """Tests for graph serialization."""

    def test_to_dict(self, simple_graph):
        """Test serializing graph to dictionary."""
        data = simple_graph.to_dict()

        assert "nodes" in data
        assert "edges" in data
        assert len(data["nodes"]) == 4
        assert len(data["edges"]) == 4

    def test_from_dict(self, simple_graph):
        """Test deserializing graph from dictionary."""
        data = simple_graph.to_dict()
        restored = SemanticGraph.from_dict(data)

        assert restored.node_count == simple_graph.node_count
        assert restored.edge_count == simple_graph.edge_count

        # Verify Doug exists
        doug = restored.find_node("Doug")
        assert doug is not None

    def test_to_json(self, simple_graph):
        """Test serializing to JSON."""
        json_str = simple_graph.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert "nodes" in data
        assert "edges" in data

    def test_from_json(self, simple_graph):
        """Test deserializing from JSON."""
        json_str = simple_graph.to_json()
        restored = SemanticGraph.from_json(json_str)

        assert restored.node_count == simple_graph.node_count
        assert restored.edge_count == simple_graph.edge_count

    def test_round_trip_preserves_data(self, simple_graph):
        """Test that serialization round-trip preserves all data."""
        json_str = simple_graph.to_json()
        restored = SemanticGraph.from_json(json_str)

        # Check node data
        for node_id, original in simple_graph.nodes.items():
            restored_node = restored.get_node(node_id)
            assert restored_node is not None
            assert restored_node.canonical_name == original.canonical_name
            assert restored_node.node_type == original.node_type

        # Check edge data
        for edge_id, original in simple_graph.edges.items():
            restored_edge = restored.get_edge(edge_id)
            assert restored_edge is not None
            assert restored_edge.relation_type == original.relation_type


# =============================================================================
# Utility Operations Tests
# =============================================================================


class TestUtilityOperations:
    """Tests for utility operations."""

    def test_clear(self, simple_graph):
        """Test clearing the graph."""
        assert simple_graph.node_count > 0

        simple_graph.clear()

        assert simple_graph.node_count == 0
        assert simple_graph.edge_count == 0

    def test_copy(self, simple_graph):
        """Test copying the graph."""
        copy = simple_graph.copy()

        # Same content
        assert copy.node_count == simple_graph.node_count
        assert copy.edge_count == simple_graph.edge_count

        # Independent - modifying copy doesn't affect original
        copy.create_node("NewNode")
        assert copy.node_count != simple_graph.node_count

    def test_len(self, simple_graph):
        """Test __len__ returns node count."""
        assert len(simple_graph) == simple_graph.node_count

    def test_contains(self, simple_graph):
        """Test __contains__ for node ID."""
        doug = simple_graph.find_node("Doug")
        assert doug.node_id in simple_graph
        assert "nonexistent" not in simple_graph

    def test_repr(self, simple_graph):
        """Test string representation."""
        repr_str = repr(simple_graph)
        assert "SemanticGraph" in repr_str
        assert "nodes=" in repr_str
        assert "edges=" in repr_str
