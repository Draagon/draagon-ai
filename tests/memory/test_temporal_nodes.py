"""Tests for temporal nodes (Phase C.1)."""

import pytest
from datetime import datetime, timedelta

from draagon_ai.memory import (
    TemporalNode,
    TemporalEdge,
    NodeType,
    EdgeType,
    MemoryLayer,
    NODE_TYPE_TO_LAYER,
    LAYER_DEFAULT_TTL,
    create_fact_node,
    create_episode_node,
    create_skill_node,
    create_entity_node,
    create_behavior_node,
)


class TestTemporalNode:
    """Tests for the TemporalNode class."""

    def test_create_basic_node(self):
        """Test basic node creation."""
        node = TemporalNode(
            content="Test content",
            node_type=NodeType.FACT,
        )

        assert node.content == "Test content"
        assert node.node_type == NodeType.FACT
        assert node.node_id is not None
        assert node.confidence == 1.0
        assert node.importance == 0.5
        assert node.layer == MemoryLayer.SEMANTIC  # Facts are semantic

    def test_node_layer_assignment(self):
        """Test that layers are correctly assigned based on type."""
        for node_type, expected_layer in NODE_TYPE_TO_LAYER.items():
            node = TemporalNode(content="test", node_type=node_type)
            assert node.layer == expected_layer, f"Expected {node_type} to have layer {expected_layer}"

    def test_node_bi_temporal_tracking(self):
        """Test bi-temporal tracking (event time vs ingestion time)."""
        # Event happened in the past, ingested now
        past_time = datetime(2023, 1, 15)
        node = TemporalNode(
            content="Event from the past",
            node_type=NodeType.EVENT,
            event_time=past_time,
        )

        assert node.event_time == past_time
        assert node.ingestion_time > past_time  # Ingested later

    def test_node_validity_interval(self):
        """Test validity interval (valid_from, valid_until)."""
        node = TemporalNode(
            content="Currently valid fact",
            node_type=NodeType.FACT,
        )

        assert node.valid_from is not None
        assert node.valid_until is None  # Currently valid
        assert node.is_current is True

        # Mark as superseded
        node.valid_until = datetime.now()
        assert node.is_current is False

    def test_node_supersede(self):
        """Test superseding a node."""
        old_node = TemporalNode(content="Old info", node_type=NodeType.FACT)
        new_node = TemporalNode(content="New info", node_type=NodeType.FACT)

        old_node.supersede(new_node.node_id)

        assert old_node.superseded_by == new_node.node_id
        assert old_node.valid_until is not None
        assert old_node.is_current is False

    def test_node_reinforce(self):
        """Test reinforcing a node (access boost)."""
        node = TemporalNode(
            content="Reinforced fact",
            node_type=NodeType.FACT,
            importance=0.5,
        )

        original_importance = node.importance
        original_stated_count = node.stated_count
        node.reinforce(boost=0.1)

        assert node.importance == original_importance + 0.1
        assert node.access_count == 1
        # reinforce() should NOT increment stated_count - that's for restate()
        assert node.stated_count == original_stated_count
        assert node.last_accessed is not None

    def test_node_restate(self):
        """Test restating a node (user says same fact again)."""
        node = TemporalNode(
            content="User's birthday",
            node_type=NodeType.FACT,
            importance=0.5,
        )

        original_importance = node.importance
        original_stated_count = node.stated_count
        original_access_count = node.access_count

        node.restate(boost=0.05)

        assert node.stated_count == original_stated_count + 1
        assert node.importance == original_importance + 0.05
        # restate() should NOT increment access_count
        assert node.access_count == original_access_count

    def test_reinforce_vs_restate_are_independent(self):
        """Test that reinforce and restate track different things."""
        node = TemporalNode(
            content="Test fact",
            node_type=NodeType.FACT,
            importance=0.5,
        )

        # Access 3 times (retrieval during queries)
        node.reinforce()
        node.reinforce()
        node.reinforce()

        # User re-stated it 2 times
        node.restate()
        node.restate()

        assert node.access_count == 3
        assert node.stated_count == 3  # 1 initial + 2 restates

    def test_node_reinforce_caps_at_one(self):
        """Test that reinforcement doesn't exceed 1.0."""
        node = TemporalNode(
            content="High importance fact",
            node_type=NodeType.FACT,
            importance=0.95,
        )

        node.reinforce(boost=0.2)
        assert node.importance == 1.0

    def test_node_decay(self):
        """Test temporal decay."""
        node = TemporalNode(
            content="Decaying memory",
            node_type=NodeType.EPISODE,
            importance=1.0,
        )

        node.decay(factor=0.9)
        assert node.importance == 0.9

        node.decay(factor=0.9)
        assert abs(node.importance - 0.81) < 0.001

    def test_node_age_seconds(self):
        """Test age calculation."""
        node = TemporalNode(content="test", node_type=NodeType.FACT)
        # Node was just created, age should be very small
        assert node.age_seconds < 1.0

    def test_node_default_ttl(self):
        """Test default TTL by layer."""
        fact_node = TemporalNode(content="fact", node_type=NodeType.FACT)
        assert fact_node.default_ttl == LAYER_DEFAULT_TTL[MemoryLayer.SEMANTIC]

        episode_node = TemporalNode(content="episode", node_type=NodeType.EPISODE)
        assert episode_node.default_ttl == LAYER_DEFAULT_TTL[MemoryLayer.EPISODIC]

        skill_node = TemporalNode(content="skill", node_type=NodeType.SKILL)
        assert skill_node.default_ttl is None  # Metacognitive = permanent

    def test_node_serialization(self):
        """Test to_dict and from_dict round-trip."""
        original = TemporalNode(
            content="Test content",
            node_type=NodeType.FACT,
            scope_id="user:roxy:doug",
            entities=["Doug", "test"],
            confidence=0.9,
            importance=0.7,
            metadata={"key": "value"},
        )

        # Round-trip
        data = original.to_dict()
        restored = TemporalNode.from_dict(data)

        assert restored.node_id == original.node_id
        assert restored.content == original.content
        assert restored.node_type == original.node_type
        assert restored.scope_id == original.scope_id
        assert restored.entities == original.entities
        assert restored.confidence == original.confidence
        assert restored.importance == original.importance
        assert restored.metadata == original.metadata


class TestTemporalEdge:
    """Tests for the TemporalEdge class."""

    def test_create_basic_edge(self):
        """Test basic edge creation."""
        edge = TemporalEdge(
            source_id="node1",
            target_id="node2",
            edge_type=EdgeType.RELATED_TO,
        )

        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.edge_type == EdgeType.RELATED_TO
        assert edge.weight == 1.0
        assert edge.confidence == 1.0
        assert edge.is_current is True

    def test_edge_with_label(self):
        """Test edge with custom label."""
        edge = TemporalEdge(
            source_id="doug",
            target_id="maya",
            edge_type=EdgeType.RELATED_TO,
            label="father of",
        )

        assert edge.label == "father of"

    def test_edge_invalidate(self):
        """Test invalidating an edge."""
        edge = TemporalEdge(
            source_id="node1",
            target_id="node2",
            edge_type=EdgeType.HAS,
        )

        assert edge.is_current is True
        edge.invalidate()
        assert edge.is_current is False
        assert edge.valid_until is not None

    def test_edge_serialization(self):
        """Test to_dict and from_dict round-trip."""
        original = TemporalEdge(
            source_id="node1",
            target_id="node2",
            edge_type=EdgeType.CAUSES,
            label="triggered by",
            weight=0.8,
            confidence=0.9,
            metadata={"reason": "observed correlation"},
        )

        data = original.to_dict()
        restored = TemporalEdge.from_dict(data)

        assert restored.edge_id == original.edge_id
        assert restored.source_id == original.source_id
        assert restored.target_id == original.target_id
        assert restored.edge_type == original.edge_type
        assert restored.label == original.label
        assert restored.weight == original.weight
        assert restored.confidence == original.confidence
        assert restored.metadata == original.metadata


class TestNodeFactoryFunctions:
    """Tests for convenience factory functions."""

    def test_create_fact_node(self):
        """Test create_fact_node factory."""
        node = create_fact_node(
            content="Doug's birthday is March 15",
            scope_id="user:roxy:doug",
            entities=["Doug", "birthday", "March 15"],
            confidence=0.95,
        )

        assert node.node_type == NodeType.FACT
        assert node.layer == MemoryLayer.SEMANTIC
        assert node.content == "Doug's birthday is March 15"
        assert "Doug" in node.entities
        assert node.confidence == 0.95
        assert node.importance == 0.7  # Facts default to 0.7

    def test_create_episode_node(self):
        """Test create_episode_node factory."""
        node = create_episode_node(
            content="Discussed vacation plans",
            conversation_id="conv_123",
        )

        assert node.node_type == NodeType.EPISODE
        assert node.layer == MemoryLayer.EPISODIC
        assert node.metadata["conversation_id"] == "conv_123"
        assert node.importance == 0.5

    def test_create_skill_node(self):
        """Test create_skill_node factory."""
        node = create_skill_node(
            content="To restart Roxy: systemctl restart roxy",
            skill_name="restart_roxy",
        )

        assert node.node_type == NodeType.SKILL
        assert node.layer == MemoryLayer.METACOGNITIVE
        assert node.metadata["skill_name"] == "restart_roxy"
        assert node.importance == 0.85

    def test_create_entity_node(self):
        """Test create_entity_node factory."""
        node = create_entity_node(
            name="Doug",
            entity_type="person",
            attributes={"role": "owner"},
        )

        assert node.node_type == NodeType.ENTITY
        assert node.layer == MemoryLayer.SEMANTIC
        assert node.metadata["entity_name"] == "Doug"
        assert node.metadata["entity_type"] == "person"
        assert node.metadata["attributes"]["role"] == "owner"
        assert "Doug" in node.entities

    def test_create_behavior_node(self):
        """Test create_behavior_node factory."""
        node = create_behavior_node(
            behavior_id="storyteller",
            behavior_name="Story Teller",
            description="Interactive storytelling behavior",
            triggers=["tell me a story"],
            actions=["generate_character", "narrate"],
        )

        assert node.node_type == NodeType.BEHAVIOR
        assert node.layer == MemoryLayer.METACOGNITIVE
        assert node.metadata["behavior_id"] == "storyteller"
        assert node.metadata["behavior_name"] == "Story Teller"
        assert "tell me a story" in node.metadata["triggers"]
        assert node.importance == 0.9


class TestNodeTypeToLayer:
    """Tests for NODE_TYPE_TO_LAYER mapping."""

    def test_all_node_types_mapped(self):
        """Ensure all node types have a layer mapping."""
        for node_type in NodeType:
            assert node_type in NODE_TYPE_TO_LAYER, f"Missing mapping for {node_type}"

    def test_working_memory_nodes(self):
        """Test working memory layer nodes."""
        assert NODE_TYPE_TO_LAYER[NodeType.CONTEXT] == MemoryLayer.WORKING
        assert NODE_TYPE_TO_LAYER[NodeType.GOAL] == MemoryLayer.WORKING

    def test_episodic_memory_nodes(self):
        """Test episodic memory layer nodes."""
        assert NODE_TYPE_TO_LAYER[NodeType.EPISODE] == MemoryLayer.EPISODIC
        assert NODE_TYPE_TO_LAYER[NodeType.EVENT] == MemoryLayer.EPISODIC

    def test_semantic_memory_nodes(self):
        """Test semantic memory layer nodes."""
        assert NODE_TYPE_TO_LAYER[NodeType.ENTITY] == MemoryLayer.SEMANTIC
        assert NODE_TYPE_TO_LAYER[NodeType.RELATIONSHIP] == MemoryLayer.SEMANTIC
        assert NODE_TYPE_TO_LAYER[NodeType.FACT] == MemoryLayer.SEMANTIC
        assert NODE_TYPE_TO_LAYER[NodeType.BELIEF] == MemoryLayer.SEMANTIC

    def test_metacognitive_memory_nodes(self):
        """Test metacognitive memory layer nodes."""
        assert NODE_TYPE_TO_LAYER[NodeType.SKILL] == MemoryLayer.METACOGNITIVE
        assert NODE_TYPE_TO_LAYER[NodeType.STRATEGY] == MemoryLayer.METACOGNITIVE
        assert NODE_TYPE_TO_LAYER[NodeType.INSIGHT] == MemoryLayer.METACOGNITIVE
        assert NODE_TYPE_TO_LAYER[NodeType.BEHAVIOR] == MemoryLayer.METACOGNITIVE


class TestLayerDefaultTTL:
    """Tests for LAYER_DEFAULT_TTL mapping."""

    def test_all_layers_have_ttl(self):
        """Ensure all layers have TTL defined."""
        for layer in MemoryLayer:
            assert layer in LAYER_DEFAULT_TTL, f"Missing TTL for {layer}"

    def test_working_memory_short_ttl(self):
        """Working memory should have short TTL."""
        ttl = LAYER_DEFAULT_TTL[MemoryLayer.WORKING]
        assert ttl == timedelta(minutes=30)

    def test_episodic_memory_medium_ttl(self):
        """Episodic memory should have medium TTL."""
        ttl = LAYER_DEFAULT_TTL[MemoryLayer.EPISODIC]
        assert ttl == timedelta(days=7)

    def test_semantic_memory_longer_ttl(self):
        """Semantic memory should have longer TTL."""
        ttl = LAYER_DEFAULT_TTL[MemoryLayer.SEMANTIC]
        assert ttl == timedelta(days=90)

    def test_metacognitive_permanent(self):
        """Metacognitive memory should be permanent."""
        ttl = LAYER_DEFAULT_TTL[MemoryLayer.METACOGNITIVE]
        assert ttl is None
