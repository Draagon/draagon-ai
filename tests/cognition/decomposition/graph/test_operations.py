"""Tests for graph update operations.

Tests incremental knowledge integration scenarios:
- Single → Collection promotion
- Anonymous → Named resolution
- Cardinality updates
"""

import pytest

from draagon_ai.cognition.decomposition.graph import (
    SemanticGraph,
    GraphNode,
    NodeType,
    EdgeRelationType,
)
from draagon_ai.cognition.decomposition.graph.operations import (
    GraphUpdater,
    UpdateResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def graph() -> SemanticGraph:
    """Create an empty graph for testing."""
    return SemanticGraph()


@pytest.fixture
def graph_with_doug_and_cat(graph: SemanticGraph) -> SemanticGraph:
    """Create a graph with Doug owning one cat (Whiskers)."""
    # Create nodes
    doug = graph.create_node("Doug", NodeType.INSTANCE)
    whiskers = graph.create_node("Whiskers", NodeType.INSTANCE)
    cat_class = graph.create_node("cat.n.01", NodeType.CLASS, synset_id="cat.n.01")

    # Create relationships
    graph.create_edge(doug.node_id, whiskers.node_id, "owns")
    graph.create_edge(whiskers.node_id, cat_class.node_id, "instance_of")

    return graph


# =============================================================================
# Single to Collection Promotion Tests
# =============================================================================


class TestCollectionPromotion:
    """Tests for promoting single relations to collections."""

    def test_add_first_relation_stays_direct(self, graph: SemanticGraph):
        """Test that first relation doesn't create a collection."""
        doug = graph.create_node("Doug", NodeType.INSTANCE)
        whiskers = graph.create_node("Whiskers", NodeType.INSTANCE)

        updater = GraphUpdater(graph)
        result = updater.add_relation(doug.node_id, whiskers.node_id, "owns")

        assert result.success
        assert result.collection_id is None
        assert len(result.edges_created) == 1
        assert "direct" in result.message.lower()

        # Verify direct edge exists
        edges = graph.get_outgoing_edges(doug.node_id, relation_type="owns")
        assert len(edges) == 1
        assert edges[0].target_node_id == whiskers.node_id

    def test_add_second_relation_creates_collection(
        self, graph_with_doug_and_cat: SemanticGraph
    ):
        """Test that second relation promotes to collection."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")
        whiskers = graph.find_node("Whiskers")

        # Add a second cat
        new_cat = graph.create_node("[Doug's cat #2]", NodeType.INSTANCE)
        cat_class = graph.find_node("cat.n.01")
        graph.create_edge(new_cat.node_id, cat_class.node_id, "instance_of")

        updater = GraphUpdater(graph)
        result = updater.add_relation(doug.node_id, new_cat.node_id, "owns")

        assert result.success
        assert result.collection_id is not None
        assert "collection" in result.message.lower()
        assert "2 members" in result.message

        # Verify collection was created
        collection = graph.get_node(result.collection_id)
        assert collection is not None
        assert collection.node_type == NodeType.COLLECTION
        assert collection.properties["count"] == 2
        assert "Doug's cats" in collection.canonical_name

        # Verify both cats are members
        member_edges = graph.get_incoming_edges(
            collection.node_id, relation_type=EdgeRelationType.MEMBER_OF
        )
        member_ids = [e.source_node_id for e in member_edges]
        assert whiskers.node_id in member_ids
        assert new_cat.node_id in member_ids

        # Verify direct edge was invalidated
        direct_edges = graph.get_outgoing_edges(doug.node_id, relation_type="owns")
        current_direct = [e for e in direct_edges if e.is_current]
        # Should have one edge to collection, not to Whiskers directly
        assert len(current_direct) == 1
        assert current_direct[0].target_node_id == collection.node_id

    def test_add_third_relation_adds_to_collection(
        self, graph_with_doug_and_cat: SemanticGraph
    ):
        """Test that third relation adds to existing collection."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")

        # Add second cat (creates collection)
        cat2 = graph.create_node("[Doug's cat #2]", NodeType.INSTANCE)
        updater = GraphUpdater(graph)
        result1 = updater.add_relation(doug.node_id, cat2.node_id, "owns")
        collection_id = result1.collection_id

        # Add third cat (should add to existing collection)
        cat3 = graph.create_node("[Doug's cat #3]", NodeType.INSTANCE)
        result2 = updater.add_relation(doug.node_id, cat3.node_id, "owns")

        assert result2.success
        assert result2.collection_id == collection_id  # Same collection
        assert "add" in result2.message.lower() or "member" in result2.message.lower()

        # Verify collection count updated
        collection = graph.get_node(collection_id)
        assert collection.properties["count"] == 3

    def test_disable_collections(self, graph_with_doug_and_cat: SemanticGraph):
        """Test adding relations without collection promotion."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")

        cat2 = graph.create_node("Mittens", NodeType.INSTANCE)

        updater = GraphUpdater(graph)
        result = updater.add_relation(
            doug.node_id, cat2.node_id, "owns", use_collections=False
        )

        assert result.success
        assert result.collection_id is None

        # Should have two direct edges
        edges = graph.get_outgoing_edges(doug.node_id, relation_type="owns")
        current_edges = [e for e in edges if e.is_current]
        assert len(current_edges) == 2


# =============================================================================
# Anonymous Resolution Tests
# =============================================================================


class TestAnonymousResolution:
    """Tests for resolving anonymous instances to named ones."""

    def test_resolve_anonymous_to_named(self, graph: SemanticGraph):
        """Test resolving an anonymous node to a named one."""
        anon_cat = graph.create_node(
            "[Doug's cat]",
            NodeType.INSTANCE,
            properties={"is_anonymous": True},
        )

        updater = GraphUpdater(graph)
        result = updater.resolve_anonymous(
            anon_cat.node_id,
            "Whiskers",
            additional_properties={"color": "orange", "breed": "tabby"},
        )

        assert result.success
        assert anon_cat.node_id in result.nodes_modified

        # Verify node was updated
        updated = graph.get_node(anon_cat.node_id)
        assert updated.canonical_name == "Whiskers"
        assert updated.properties["is_anonymous"] is False
        assert updated.properties["resolved_from"] == "[Doug's cat]"
        assert updated.properties["color"] == "orange"
        assert updated.properties["breed"] == "tabby"

    def test_resolve_nonexistent_node(self, graph: SemanticGraph):
        """Test resolving a node that doesn't exist."""
        updater = GraphUpdater(graph)
        result = updater.resolve_anonymous("nonexistent-id", "Whiskers")

        assert not result.success
        assert "not found" in result.message.lower()


# =============================================================================
# Cardinality Update Tests
# =============================================================================


class TestCardinalityUpdates:
    """Tests for updating collection cardinality."""

    def test_increase_cardinality_creates_placeholders(
        self, graph_with_doug_and_cat: SemanticGraph
    ):
        """Test that increasing cardinality creates anonymous placeholders."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")

        # First, create a collection by adding second cat
        cat2 = graph.create_node("[Doug's cat #2]", NodeType.INSTANCE)
        cat_class = graph.find_node("cat.n.01")
        graph.create_edge(cat2.node_id, cat_class.node_id, "instance_of")

        updater = GraphUpdater(graph)
        result1 = updater.add_relation(doug.node_id, cat2.node_id, "owns")
        collection_id = result1.collection_id

        # Now update cardinality to 4
        result2 = updater.update_cardinality(doug.node_id, "owns", 4)

        assert result2.success
        assert len(result2.nodes_created) == 2  # 2 new placeholders
        assert "2 anonymous placeholders" in result2.message

        # Verify collection count updated
        collection = graph.get_node(collection_id)
        assert collection.properties["count"] == 4

        # Verify placeholders were created
        member_edges = graph.get_incoming_edges(
            collection_id, relation_type=EdgeRelationType.MEMBER_OF
        )
        assert len(member_edges) == 4

    def test_decrease_cardinality_updates_count(
        self, graph_with_doug_and_cat: SemanticGraph
    ):
        """Test that decreasing cardinality updates count without removing members."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")

        # Create collection with 3 cats
        cat2 = graph.create_node("Mittens", NodeType.INSTANCE)
        cat3 = graph.create_node("Shadow", NodeType.INSTANCE)

        updater = GraphUpdater(graph)
        updater.add_relation(doug.node_id, cat2.node_id, "owns")
        result = updater.add_relation(doug.node_id, cat3.node_id, "owns")
        collection_id = result.collection_id

        # Update cardinality to 2 (one cat ran away?)
        result2 = updater.update_cardinality(doug.node_id, "owns", 2)

        assert result2.success
        assert len(result2.nodes_created) == 0  # No new nodes

        # Count updated but members still exist
        collection = graph.get_node(collection_id)
        assert collection.properties["count"] == 2

        # All 3 member edges still exist (we don't know which to remove)
        member_edges = graph.get_incoming_edges(
            collection_id, relation_type=EdgeRelationType.MEMBER_OF
        )
        assert len(member_edges) == 3

    def test_cardinality_no_collection(self, graph_with_doug_and_cat: SemanticGraph):
        """Test updating cardinality when no collection exists."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")

        updater = GraphUpdater(graph)
        result = updater.update_cardinality(doug.node_id, "owns", 3)

        assert not result.success
        assert "no collection" in result.message.lower()


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_add_relation_nonexistent_source(self, graph: SemanticGraph):
        """Test adding relation with nonexistent source."""
        target = graph.create_node("Whiskers", NodeType.INSTANCE)

        updater = GraphUpdater(graph)
        result = updater.add_relation("nonexistent", target.node_id, "owns")

        assert not result.success
        assert "not found" in result.message.lower()

    def test_add_relation_nonexistent_target(self, graph: SemanticGraph):
        """Test adding relation with nonexistent target."""
        source = graph.create_node("Doug", NodeType.INSTANCE)

        updater = GraphUpdater(graph)
        result = updater.add_relation(source.node_id, "nonexistent", "owns")

        assert not result.success
        assert "not found" in result.message.lower()

    def test_add_duplicate_to_collection(
        self, graph_with_doug_and_cat: SemanticGraph
    ):
        """Test adding the same member to a collection twice."""
        graph = graph_with_doug_and_cat
        doug = graph.find_node("Doug")
        whiskers = graph.find_node("Whiskers")

        # Create collection
        cat2 = graph.create_node("Mittens", NodeType.INSTANCE)
        updater = GraphUpdater(graph)
        result1 = updater.add_relation(doug.node_id, cat2.node_id, "owns")
        collection_id = result1.collection_id

        # Try to add Whiskers again (already in collection)
        result2 = updater.add_relation(doug.node_id, whiskers.node_id, "owns")

        assert result2.success
        assert "already a member" in result2.message.lower()

        # Count should not have increased
        collection = graph.get_node(collection_id)
        assert collection.properties["count"] == 2
