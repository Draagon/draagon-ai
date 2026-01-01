"""Unit tests for graph data models.

Tests for GraphNode, GraphEdge, and related data structures.
"""

import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from draagon_ai.cognition.decomposition.graph import (
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeRelationType,
    ConflictType,
    MergeConflict,
    MergeResult,
)
from draagon_ai.cognition.decomposition.identifiers import EntityType


# =============================================================================
# GraphNode Tests
# =============================================================================


class TestGraphNode:
    """Tests for GraphNode dataclass."""

    # -------------------------------------------------------------------------
    # Creation and Basic Properties
    # -------------------------------------------------------------------------

    def test_create_node_with_defaults(self):
        """Test creating a node with default values."""
        node = GraphNode()
        assert node.node_id is not None
        assert len(node.node_id) == 36  # UUID format
        assert node.node_type == NodeType.INSTANCE
        assert node.canonical_name == ""
        assert node.properties == {}
        assert node.confidence == 1.0

    def test_create_node_with_all_fields(self):
        """Test creating a node with all fields specified."""
        node = GraphNode(
            node_id="test-node-001",
            node_type=NodeType.CLASS,
            canonical_name="Cat",
            entity_type=EntityType.CLASS,
            properties={"breed": "tabby", "color": "orange"},
            synset_id="cat.n.01",
            wikidata_qid="Q146",
            confidence=0.95,
            source_ids=["decomp-001"],
        )
        assert node.node_id == "test-node-001"
        assert node.node_type == NodeType.CLASS
        assert node.canonical_name == "Cat"
        assert node.entity_type == EntityType.CLASS
        assert node.properties["breed"] == "tabby"
        assert node.synset_id == "cat.n.01"
        assert node.wikidata_qid == "Q146"
        assert node.confidence == 0.95
        assert "decomp-001" in node.source_ids

    def test_node_hash_and_equality(self):
        """Test node hashing and equality based on node_id."""
        node1 = GraphNode(node_id="same-id", canonical_name="Node 1")
        node2 = GraphNode(node_id="same-id", canonical_name="Node 2")
        node3 = GraphNode(node_id="different-id", canonical_name="Node 1")

        assert node1 == node2  # Same ID
        assert node1 != node3  # Different ID
        assert hash(node1) == hash(node2)
        assert hash(node1) != hash(node3)

        # Can be used in sets
        node_set = {node1, node2, node3}
        assert len(node_set) == 2  # node1 and node2 are same

    def test_node_not_equal_to_non_node(self):
        """Test that node is not equal to non-node objects."""
        node = GraphNode(node_id="test")
        assert node != "test"
        assert node != 123
        assert node != None

    # -------------------------------------------------------------------------
    # Properties Management
    # -------------------------------------------------------------------------

    def test_add_property(self):
        """Test adding properties to a node."""
        node = GraphNode(canonical_name="Test")
        original_updated = node.updated_at

        node.add_property("color", "blue")
        assert node.properties["color"] == "blue"
        assert node.updated_at >= original_updated

    def test_get_property(self):
        """Test getting properties with default value."""
        node = GraphNode(properties={"name": "Test"})

        assert node.get_property("name") == "Test"
        assert node.get_property("missing") is None
        assert node.get_property("missing", "default") == "default"

    def test_has_property(self):
        """Test checking if property exists."""
        node = GraphNode(properties={"exists": True})

        assert node.has_property("exists") is True
        assert node.has_property("missing") is False

    def test_add_source(self):
        """Test adding source decomposition IDs."""
        node = GraphNode()
        node.add_source("decomp-001")
        node.add_source("decomp-002")
        node.add_source("decomp-001")  # Duplicate

        assert len(node.source_ids) == 2
        assert "decomp-001" in node.source_ids
        assert "decomp-002" in node.source_ids

    # -------------------------------------------------------------------------
    # Name Matching
    # -------------------------------------------------------------------------

    def test_matches_name_case_insensitive(self):
        """Test case-insensitive name matching."""
        node = GraphNode(canonical_name="Doug")

        assert node.matches_name("Doug") is True
        assert node.matches_name("doug") is True
        assert node.matches_name("DOUG") is True
        assert node.matches_name("Bob") is False

    def test_matches_name_case_sensitive(self):
        """Test case-sensitive name matching."""
        node = GraphNode(canonical_name="Doug")

        assert node.matches_name("Doug", case_sensitive=True) is True
        assert node.matches_name("doug", case_sensitive=True) is False

    def test_matches_name_with_aliases(self):
        """Test matching against aliases."""
        node = GraphNode(
            canonical_name="Douglas",
            properties={"aliases": ["Doug", "Dougie"]},
        )

        assert node.matches_name("Douglas") is True
        assert node.matches_name("Doug") is True
        assert node.matches_name("Dougie") is True
        assert node.matches_name("Bob") is False

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def test_to_dict(self):
        """Test serializing node to dictionary."""
        node = GraphNode(
            node_id="test-001",
            node_type=NodeType.INSTANCE,
            canonical_name="Test",
            entity_type=EntityType.INSTANCE,
            properties={"key": "value"},
        )
        data = node.to_dict()

        assert data["node_id"] == "test-001"
        assert data["node_type"] == "instance"
        assert data["canonical_name"] == "Test"
        assert data["entity_type"] == "instance"
        assert data["properties"]["key"] == "value"

    def test_from_dict(self):
        """Test deserializing node from dictionary."""
        data = {
            "node_id": "test-001",
            "node_type": "concept",
            "canonical_name": "Test",
            "entity_type": "class",
            "properties": {"key": "value"},
            "synset_id": "test.n.01",
            "confidence": 0.9,
        }
        node = GraphNode.from_dict(data)

        assert node.node_id == "test-001"
        assert node.node_type == NodeType.CLASS
        assert node.canonical_name == "Test"
        assert node.entity_type == EntityType.CLASS
        assert node.synset_id == "test.n.01"
        assert node.confidence == 0.9

    def test_to_json_and_back(self):
        """Test JSON round-trip serialization."""
        original = GraphNode(
            node_id="round-trip",
            canonical_name="Test Node",
            properties={"nested": {"data": [1, 2, 3]}},
        )
        json_str = original.to_json()
        restored = GraphNode.from_json(json_str)

        assert restored.node_id == original.node_id
        assert restored.canonical_name == original.canonical_name
        assert restored.properties == original.properties

    def test_node_repr(self):
        """Test string representation."""
        node = GraphNode(node_type=NodeType.INSTANCE, canonical_name="Test")
        repr_str = repr(node)

        assert "instance" in repr_str
        assert "Test" in repr_str


# =============================================================================
# GraphEdge Tests
# =============================================================================


class TestGraphEdge:
    """Tests for GraphEdge dataclass."""

    # -------------------------------------------------------------------------
    # Creation and Basic Properties
    # -------------------------------------------------------------------------

    def test_create_edge_with_defaults(self):
        """Test creating an edge with default values."""
        edge = GraphEdge()
        assert edge.edge_id is not None
        assert edge.source_node_id == ""
        assert edge.target_node_id == ""
        assert edge.relation_type == "related_to"
        assert edge.valid_to is None
        assert edge.is_current is True
        assert edge.confidence == 1.0

    def test_create_edge_with_all_fields(self):
        """Test creating an edge with all fields."""
        now = datetime.now(timezone.utc)
        edge = GraphEdge(
            edge_id="edge-001",
            source_node_id="node-a",
            target_node_id="node-b",
            relation_type="has",
            properties={"count": 6},
            valid_from=now,
            confidence=0.95,
            source_decomposition_id="decomp-001",
        )

        assert edge.edge_id == "edge-001"
        assert edge.source_node_id == "node-a"
        assert edge.target_node_id == "node-b"
        assert edge.relation_type == "has"
        assert edge.properties["count"] == 6
        assert edge.valid_from == now
        assert edge.confidence == 0.95

    def test_edge_hash_and_equality(self):
        """Test edge hashing and equality based on edge_id."""
        edge1 = GraphEdge(edge_id="same-id", relation_type="has")
        edge2 = GraphEdge(edge_id="same-id", relation_type="owns")
        edge3 = GraphEdge(edge_id="different-id", relation_type="has")

        assert edge1 == edge2
        assert edge1 != edge3
        assert hash(edge1) == hash(edge2)

    # -------------------------------------------------------------------------
    # Bi-Temporal Operations
    # -------------------------------------------------------------------------

    def test_is_current(self):
        """Test checking if edge is currently valid."""
        current_edge = GraphEdge(valid_to=None)
        expired_edge = GraphEdge(valid_to=datetime.now(timezone.utc))

        assert current_edge.is_current is True
        assert expired_edge.is_current is False

    def test_invalidate(self):
        """Test invalidating an edge."""
        edge = GraphEdge()
        assert edge.is_current is True

        edge.invalidate()
        assert edge.is_current is False
        assert edge.valid_to is not None

    def test_invalidate_with_specific_time(self):
        """Test invalidating an edge at a specific time."""
        edge = GraphEdge()
        specific_time = datetime(2024, 6, 15, tzinfo=timezone.utc)

        edge.invalidate(specific_time)
        assert edge.valid_to == specific_time

    def test_is_valid_at_past_time(self):
        """Test checking validity at a past time."""
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_time = datetime(2024, 6, 1, tzinfo=timezone.utc)

        edge = GraphEdge(valid_from=start_time, valid_to=end_time)

        # Before start: not valid
        before_start = datetime(2023, 12, 1, tzinfo=timezone.utc)
        assert edge.is_valid_at(before_start) is False

        # During validity period: valid
        during = datetime(2024, 3, 1, tzinfo=timezone.utc)
        assert edge.is_valid_at(during) is True

        # After end: not valid
        after_end = datetime(2024, 12, 1, tzinfo=timezone.utc)
        assert edge.is_valid_at(after_end) is False

    def test_is_valid_at_with_current_edge(self):
        """Test validity check for a currently valid edge."""
        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        edge = GraphEdge(valid_from=start_time, valid_to=None)

        # Future date should still be valid
        future = datetime(2030, 1, 1, tzinfo=timezone.utc)
        assert edge.is_valid_at(future) is True

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    def test_add_property(self):
        """Test adding properties to an edge."""
        edge = GraphEdge()
        edge.add_property("count", 5)
        edge.add_property("confidence", 0.9)

        assert edge.properties["count"] == 5
        assert edge.properties["confidence"] == 0.9

    def test_get_property(self):
        """Test getting properties with default."""
        edge = GraphEdge(properties={"exists": True})

        assert edge.get_property("exists") is True
        assert edge.get_property("missing") is None
        assert edge.get_property("missing", "default") == "default"

    # -------------------------------------------------------------------------
    # Serialization
    # -------------------------------------------------------------------------

    def test_to_dict(self):
        """Test serializing edge to dictionary."""
        edge = GraphEdge(
            edge_id="edge-001",
            source_node_id="node-a",
            target_node_id="node-b",
            relation_type="has",
            properties={"count": 6},
        )
        data = edge.to_dict()

        assert data["edge_id"] == "edge-001"
        assert data["source_node_id"] == "node-a"
        assert data["target_node_id"] == "node-b"
        assert data["relation_type"] == "has"
        assert data["properties"]["count"] == 6
        assert data["valid_to"] is None  # Currently valid

    def test_from_dict(self):
        """Test deserializing edge from dictionary."""
        data = {
            "edge_id": "edge-001",
            "source_node_id": "node-a",
            "target_node_id": "node-b",
            "relation_type": "owns",
            "valid_from": "2024-01-01T00:00:00+00:00",
            "valid_to": "2024-06-01T00:00:00+00:00",
            "confidence": 0.85,
        }
        edge = GraphEdge.from_dict(data)

        assert edge.edge_id == "edge-001"
        assert edge.relation_type == "owns"
        assert edge.valid_to is not None
        assert edge.confidence == 0.85

    def test_to_json_and_back(self):
        """Test JSON round-trip serialization."""
        original = GraphEdge(
            edge_id="round-trip",
            source_node_id="a",
            target_node_id="b",
            relation_type="custom_relation",
            properties={"metadata": {"nested": True}},
        )
        json_str = original.to_json()
        restored = GraphEdge.from_json(json_str)

        assert restored.edge_id == original.edge_id
        assert restored.relation_type == original.relation_type
        assert restored.properties == original.properties

    def test_edge_repr(self):
        """Test string representation."""
        edge = GraphEdge(
            source_node_id="12345678-1234-1234-1234-123456789012",
            target_node_id="87654321-4321-4321-4321-210987654321",
            relation_type="has",
        )
        repr_str = repr(edge)

        assert "has" in repr_str
        assert "current" in repr_str


# =============================================================================
# MergeConflict Tests
# =============================================================================


class TestMergeConflict:
    """Tests for MergeConflict dataclass."""

    def test_create_conflict(self):
        """Test creating a merge conflict."""
        conflict = MergeConflict(
            conflict_type=ConflictType.CONTRADICTS,
            existing_edge_id="edge-001",
            new_content="Doug has 5 cats",
            existing_content="Doug has 6 cats",
            confidence=0.95,
        )

        assert conflict.conflict_type == ConflictType.CONTRADICTS
        assert conflict.existing_edge_id == "edge-001"
        assert "5 cats" in conflict.new_content
        assert "6 cats" in conflict.existing_content

    def test_conflict_types(self):
        """Test different conflict types."""
        assert ConflictType.CONTRADICTS.value == "contradicts"
        assert ConflictType.SUPERSEDES.value == "supersedes"
        assert ConflictType.AMBIGUOUS.value == "ambiguous"

    def test_conflict_serialization(self):
        """Test conflict serialization round-trip."""
        original = MergeConflict(
            conflict_type=ConflictType.SUPERSEDES,
            existing_edge_id="edge-old",
            new_content="new info",
            existing_content="old info",
            resolution="auto-resolved",
        )
        data = original.to_dict()
        restored = MergeConflict.from_dict(data)

        assert restored.conflict_type == original.conflict_type
        assert restored.resolution == "auto-resolved"


# =============================================================================
# MergeResult Tests
# =============================================================================


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_empty_merge_result(self):
        """Test creating an empty merge result."""
        result = MergeResult()

        assert result.nodes_created == []
        assert result.nodes_updated == []
        assert result.edges_created == []
        assert result.edges_invalidated == []
        assert result.conflicts == []
        assert result.success is True
        assert result.has_conflicts is False
        assert result.total_changes == 0

    def test_merge_result_with_changes(self):
        """Test merge result with changes."""
        result = MergeResult(
            nodes_created=["node-1", "node-2"],
            nodes_updated=["node-3"],
            edges_created=["edge-1", "edge-2", "edge-3"],
            edges_invalidated=["edge-old"],
        )

        assert len(result.nodes_created) == 2
        assert len(result.nodes_updated) == 1
        assert len(result.edges_created) == 3
        assert len(result.edges_invalidated) == 1
        assert result.total_changes == 7

    def test_merge_result_with_conflicts(self):
        """Test merge result with conflicts."""
        conflict = MergeConflict(
            conflict_type=ConflictType.CONTRADICTS,
            new_content="new",
            existing_content="old",
        )
        result = MergeResult(conflicts=[conflict])

        assert result.has_conflicts is True
        assert len(result.conflicts) == 1

    def test_merge_result_failure(self):
        """Test merge result indicating failure."""
        result = MergeResult(
            success=False,
            error_message="Failed to resolve entity",
        )

        assert result.success is False
        assert "Failed" in result.error_message

    def test_merge_result_serialization(self):
        """Test merge result serialization."""
        conflict = MergeConflict(conflict_type=ConflictType.AMBIGUOUS)
        result = MergeResult(
            nodes_created=["n1"],
            conflicts=[conflict],
        )

        data = result.to_dict()
        restored = MergeResult.from_dict(data)

        assert restored.nodes_created == ["n1"]
        assert len(restored.conflicts) == 1


# =============================================================================
# NodeType and EdgeRelationType Tests
# =============================================================================


class TestEnums:
    """Tests for enumeration types."""

    def test_node_types(self):
        """Test all node types are defined."""
        assert NodeType.INSTANCE.value == "instance"
        assert NodeType.CLASS.value == "class"
        assert NodeType.EVENT.value == "event"
        assert NodeType.ATTRIBUTE.value == "attribute"
        assert NodeType.COLLECTION.value == "collection"

    def test_edge_relation_types(self):
        """Test common edge relation types."""
        assert EdgeRelationType.HAS.value == "has"
        assert EdgeRelationType.IS_A.value == "is_a"
        assert EdgeRelationType.PART_OF.value == "part_of"
        assert EdgeRelationType.LOCATED_IN.value == "located_in"
        assert EdgeRelationType.CAUSES.value == "causes"

    def test_node_type_string_conversion(self):
        """Test node type string conversion."""
        # New values
        node_type = NodeType("instance")
        assert node_type == NodeType.INSTANCE

        # Backwards compatibility - old values still work
        node_type_legacy = NodeType("entity")
        assert node_type_legacy == NodeType.INSTANCE

        # Can be used as string
        assert f"Type: {node_type.value}" == "Type: instance"
