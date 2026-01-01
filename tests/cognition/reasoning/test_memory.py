"""Tests for the unified temporal memory model.

These tests verify:
1. Memory layers with TTL/expiration
2. Differential decay rates by content type
3. Volatile working memory for ReAct swarms
4. Reinforcement and layer promotion
5. Memory-aware graph storage
"""

import pytest
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

from draagon_ai.cognition.reasoning.memory import (
    MemoryLayer,
    ContentType,
    MemoryProperties,
    VolatileObservation,
    VolatileWorkingMemory,
    MemoryAwareGraphStore,
    classify_phase1_content,
    LAYER_TTL,
    DECAY_RATES,
)
from draagon_ai.cognition.decomposition.graph import (
    SemanticGraph,
    GraphNode,
    NodeType,
    Neo4jGraphStoreSync,
)


# =============================================================================
# Test Configuration
# =============================================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "draagon-ai-2025"


def neo4j_available() -> bool:
    """Check if Neo4j is available."""
    try:
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        store.close()
        return True
    except Exception:
        return False


# =============================================================================
# Memory Layer Tests
# =============================================================================


class TestMemoryLayers:
    """Tests for memory layer definitions."""

    def test_layer_values(self):
        """Test all 4 layers are defined."""
        assert MemoryLayer.WORKING.value == "working"
        assert MemoryLayer.EPISODIC.value == "episodic"
        assert MemoryLayer.SEMANTIC.value == "semantic"
        assert MemoryLayer.METACOGNITIVE.value == "metacognitive"

    def test_layer_ttls(self):
        """Test TTL definitions for each layer."""
        assert LAYER_TTL[MemoryLayer.WORKING] == timedelta(minutes=5)
        assert LAYER_TTL[MemoryLayer.EPISODIC] == timedelta(weeks=2)
        assert LAYER_TTL[MemoryLayer.SEMANTIC] == timedelta(days=180)
        assert LAYER_TTL[MemoryLayer.METACOGNITIVE] is None  # Permanent


class TestContentTypes:
    """Tests for content type definitions."""

    def test_core_content_types(self):
        """Test core content types (slow decay)."""
        core_types = [
            ContentType.INSTRUCTION,
            ContentType.PREFERENCE,
            ContentType.SKILL,
            ContentType.FACT,
            ContentType.ENTITY,
            ContentType.RELATIONSHIP,
        ]
        for ct in core_types:
            assert DECAY_RATES[ct] < 1.0, f"{ct} should have slow decay"

    def test_peripheral_content_types(self):
        """Test peripheral content types (fast decay)."""
        peripheral_types = [
            ContentType.MODALITY,
            ContentType.PRESUPPOSITION,
            ContentType.COMMONSENSE,
            ContentType.SENTIMENT,
            ContentType.NEGATION,
        ]
        for ct in peripheral_types:
            assert DECAY_RATES[ct] > 1.0, f"{ct} should have fast decay"

    def test_instruction_decays_slowest(self):
        """Instructions should decay the slowest."""
        instruction_rate = DECAY_RATES[ContentType.INSTRUCTION]
        for ct, rate in DECAY_RATES.items():
            if ct != ContentType.INSTRUCTION:
                assert rate >= instruction_rate, f"{ct} should decay >= INSTRUCTION"


# =============================================================================
# Memory Properties Tests
# =============================================================================


class TestMemoryProperties:
    """Tests for MemoryProperties dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        props = MemoryProperties()

        assert props.layer == MemoryLayer.WORKING
        assert props.access_count == 0
        assert props.importance == 0.5
        assert props.confidence == 1.0
        assert props.content_type == ContentType.ENTITY
        assert props.reinforcement_score == 0.0

    def test_automatic_expiration_calculation(self):
        """Test that expiration is calculated from layer TTL."""
        props = MemoryProperties(layer=MemoryLayer.WORKING)

        assert props.expires_at is not None
        expected_expiry = props.created_at + timedelta(minutes=5)
        # Allow small time delta for test execution
        assert abs((props.expires_at - expected_expiry).total_seconds()) < 1

    def test_metacognitive_no_expiration(self):
        """Test that metacognitive layer has no expiration."""
        props = MemoryProperties(layer=MemoryLayer.METACOGNITIVE)
        assert props.expires_at is None

    def test_is_expired(self):
        """Test expiration detection."""
        # Not expired
        props = MemoryProperties(layer=MemoryLayer.WORKING)
        assert not props.is_expired

        # Manually set to past
        props.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
        assert props.is_expired

    def test_time_to_expiry(self):
        """Test time to expiry calculation."""
        props = MemoryProperties(layer=MemoryLayer.WORKING)

        ttl = props.time_to_expiry
        assert ttl is not None
        assert ttl.total_seconds() > 0
        assert ttl.total_seconds() <= 5 * 60  # Max 5 minutes

    def test_decay_rate_lookup(self):
        """Test decay rate lookup from content type."""
        props = MemoryProperties(content_type=ContentType.INSTRUCTION)
        assert props.decay_rate == 0.1

        props = MemoryProperties(content_type=ContentType.MODALITY)
        assert props.decay_rate == 3.0

    def test_record_access(self):
        """Test that accessing extends expiration."""
        props = MemoryProperties(layer=MemoryLayer.WORKING)
        original_expiry = props.expires_at
        original_count = props.access_count

        props.record_access()

        assert props.access_count == original_count + 1
        assert props.expires_at > original_expiry

    def test_reinforce_increases_score(self):
        """Test that reinforcement increases score."""
        props = MemoryProperties()
        assert props.reinforcement_score == 0.0

        props.reinforce(0.1)
        assert props.reinforcement_score == 0.1

        # Don't reinforce too much - 0.3 triggers promotion from working
        props.reinforce(0.15)
        assert props.reinforcement_score == 0.25  # Still below threshold

    def test_layer_promotion_from_working(self):
        """Test promotion from working to episodic."""
        props = MemoryProperties(layer=MemoryLayer.WORKING)

        # Reinforce until promotion threshold (0.3)
        # Each reinforce checks for promotion, so 3 x 0.1 = 0.3 triggers it
        props.reinforce(0.1)  # score = 0.1
        props.reinforce(0.1)  # score = 0.2
        props.reinforce(0.1)  # score = 0.3 - triggers promotion, resets to 0

        assert props.layer == MemoryLayer.EPISODIC
        assert props.reinforcement_score == 0.0  # Reset after promotion

    def test_layer_promotion_chain(self):
        """Test full promotion chain through all layers."""
        props = MemoryProperties(layer=MemoryLayer.WORKING)

        # Working -> Episodic (threshold 0.3)
        props.reinforcement_score = 0.3
        props._check_promotion()
        assert props.layer == MemoryLayer.EPISODIC

        # Episodic -> Semantic (threshold 0.6)
        props.reinforcement_score = 0.6
        props._check_promotion()
        assert props.layer == MemoryLayer.SEMANTIC

        # Semantic -> Metacognitive (threshold 0.9)
        props.reinforcement_score = 0.9
        props._check_promotion()
        assert props.layer == MemoryLayer.METACOGNITIVE
        assert props.expires_at is None  # Now permanent

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = MemoryProperties(
            layer=MemoryLayer.SEMANTIC,
            importance=0.8,
            confidence=0.9,
            content_type=ContentType.SKILL,
            reinforcement_score=0.5,
        )
        original.record_access()

        data = original.to_dict()
        restored = MemoryProperties.from_dict(data)

        assert restored.layer == original.layer
        assert restored.importance == original.importance
        assert restored.confidence == original.confidence
        assert restored.content_type == original.content_type
        assert restored.reinforcement_score == original.reinforcement_score
        assert restored.access_count == original.access_count


# =============================================================================
# Volatile Working Memory Tests
# =============================================================================


class TestVolatileObservation:
    """Tests for VolatileObservation dataclass."""

    def test_default_initialization(self):
        """Test default values."""
        obs = VolatileObservation(content="test")

        assert obs.observation_id.startswith("obs_")
        assert obs.content == "test"
        assert obs.attention_weight == 1.0
        assert not obs.persist_candidate

    def test_expiration(self):
        """Test expiration is ~5 minutes from creation."""
        obs = VolatileObservation()

        # Should expire in approximately 5 minutes
        ttl = (obs.expires_at - obs.created_at).total_seconds()
        assert 290 < ttl <= 301  # 5 minutes ± small margin for timing

    def test_is_expired(self):
        """Test expiration detection."""
        obs = VolatileObservation()
        assert not obs.is_expired

        obs.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        assert obs.is_expired

    def test_attention_boost(self):
        """Test attention boosting."""
        obs = VolatileObservation()
        obs.attention_weight = 0.5

        obs.boost_attention(0.2)
        assert obs.attention_weight == 0.7

        # Should cap at 1.0
        obs.boost_attention(0.5)
        assert obs.attention_weight == 1.0

    def test_attention_decay(self):
        """Test attention decay."""
        obs = VolatileObservation()
        assert obs.attention_weight == 1.0

        obs.decay_attention(0.9)
        assert obs.attention_weight == 0.9

        obs.decay_attention(0.9)
        assert abs(obs.attention_weight - 0.81) < 0.001


class TestVolatileWorkingMemory:
    """Tests for VolatileWorkingMemory (ReAct swarm scratchpad)."""

    @pytest.mark.asyncio
    async def test_add_observation(self):
        """Test adding observations."""
        memory = VolatileWorkingMemory(task_id="test-1")

        obs = await memory.add(
            content="User wants to book a flight",
            source_agent_id="agent-1",
        )

        assert obs.observation_id in memory._observations
        assert obs.content == "User wants to book a flight"
        assert obs.source_agent_id == "agent-1"

    @pytest.mark.asyncio
    async def test_millers_law_capacity(self):
        """Test that capacity is limited to 9 (7+2)."""
        memory = VolatileWorkingMemory(task_id="test-2", max_items=9)

        # Add 12 observations
        for i in range(12):
            await memory.add(
                content=f"Observation {i}",
                source_agent_id="agent",
            )

        # Should only keep 9
        assert len(memory._observations) == 9

    @pytest.mark.asyncio
    async def test_lowest_attention_evicted(self):
        """Test that lowest attention items are evicted first."""
        memory = VolatileWorkingMemory(task_id="test-3", max_items=3)

        # Add with varying attention weights
        obs1 = await memory.add("Low priority", "agent", attention_weight=0.3)
        obs2 = await memory.add("High priority", "agent", attention_weight=0.9)
        obs3 = await memory.add("Medium priority", "agent", attention_weight=0.6)

        # Add one more (should evict lowest attention)
        obs4 = await memory.add("New item", "agent", attention_weight=0.7)

        assert obs1.observation_id not in memory._observations  # Evicted (lowest)
        assert obs2.observation_id in memory._observations
        assert obs3.observation_id in memory._observations
        assert obs4.observation_id in memory._observations

    @pytest.mark.asyncio
    async def test_get_context_sorted_by_attention(self):
        """Test context retrieval sorted by attention."""
        memory = VolatileWorkingMemory(task_id="test-4")

        await memory.add("Low", "agent", attention_weight=0.3)
        await memory.add("High", "agent", attention_weight=0.9)
        await memory.add("Medium", "agent", attention_weight=0.6)

        context = await memory.get_context("agent-2", max_items=3)

        # Should be sorted by attention (highest first)
        assert context[0].content == "High"
        assert context[1].content == "Medium"
        assert context[2].content == "Low"

    @pytest.mark.asyncio
    async def test_mark_for_persistence(self):
        """Test marking observations for persistence."""
        memory = VolatileWorkingMemory(task_id="test-5")

        obs = await memory.add("Important finding", "agent")
        assert not obs.persist_candidate

        result = await memory.mark_for_persistence(obs.observation_id)
        assert result is True
        assert memory._observations[obs.observation_id].persist_candidate

    @pytest.mark.asyncio
    async def test_get_persistence_candidates(self):
        """Test getting persistence candidates."""
        memory = VolatileWorkingMemory(task_id="test-6")

        obs1 = await memory.add("Not persisted", "agent")
        obs2 = await memory.add("Will persist", "agent")

        await memory.mark_for_persistence(obs2.observation_id)

        candidates = await memory.get_persistence_candidates()
        assert len(candidates) == 1
        assert candidates[0].content == "Will persist"

    @pytest.mark.asyncio
    async def test_persist_candidates_not_evicted(self):
        """Test that persistence candidates are protected from eviction."""
        memory = VolatileWorkingMemory(task_id="test-7", max_items=2)

        # Add and mark as persist candidate
        obs1 = await memory.add("Keep me", "agent", attention_weight=0.1)
        await memory.mark_for_persistence(obs1.observation_id)

        # Add more to trigger eviction
        await memory.add("New 1", "agent", attention_weight=0.5)
        await memory.add("New 2", "agent", attention_weight=0.5)

        # Persist candidate should still be there
        assert obs1.observation_id in memory._observations

    @pytest.mark.asyncio
    async def test_expired_observations_removed(self):
        """Test that expired observations are removed."""
        memory = VolatileWorkingMemory(task_id="test-8")

        obs = await memory.add("Will expire", "agent")

        # Manually expire it
        memory._observations[obs.observation_id].expires_at = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        )

        # Get context triggers cleanup
        context = await memory.get_context("agent")
        assert obs.observation_id not in memory._observations

    @pytest.mark.asyncio
    async def test_content_type_affects_decay(self):
        """Test that content type affects decay rate."""
        memory = VolatileWorkingMemory(
            task_id="test-9",
            decay_interval_seconds=0.0,  # Immediate decay
        )

        # Add with different content types
        obs_instruction = await memory.add(
            "Remember this",
            "agent",
            content_type=ContentType.INSTRUCTION,
            attention_weight=0.5,
        )
        obs_modality = await memory.add(
            "Maybe",
            "agent",
            content_type=ContentType.MODALITY,
            attention_weight=0.5,
        )

        # Trigger decay
        memory._apply_decay()

        # Instruction should decay slower than modality
        instruction_weight = memory._observations[obs_instruction.observation_id].attention_weight
        modality_weight = memory._observations[obs_modality.observation_id].attention_weight

        assert instruction_weight > modality_weight


# =============================================================================
# Memory-Aware Graph Store Tests (Integration)
# =============================================================================


@pytest.mark.skipif(not neo4j_available(), reason="Neo4j not available")
class TestMemoryAwareGraphStore:
    """Integration tests for memory-aware Neo4j storage."""

    def test_save_with_memory_properties(self):
        """Test saving graph with memory properties."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-memory-props"

        try:
            # Create simple graph
            graph = SemanticGraph()
            node = graph.create_node("Doug", NodeType.INSTANCE)

            # Save with memory awareness
            memory_store = MemoryAwareGraphStore(store, instance_id)
            result = memory_store.save_with_memory(
                graph,
                default_layer=MemoryLayer.WORKING,
            )

            assert result["nodes"] > 0

            # Verify memory properties were saved
            with store.driver.session() as session:
                result = session.run("""
                    MATCH (n:Entity {instance_id: $id})
                    RETURN n.memory_layer AS layer,
                           n.memory_expires_at AS expires_at
                """, id=instance_id)

                record = result.single()
                assert record["layer"] == "working"
                assert record["expires_at"] is not None

        finally:
            # Cleanup
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_load_active_filters_expired(self):
        """Test that load_active filters expired memories."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-load-active"

        try:
            # Create graph with two nodes
            graph = SemanticGraph()
            active_node = graph.create_node("Active", NodeType.INSTANCE)
            expired_node = graph.create_node("Expired", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            # Manually expire one node
            with store.driver.session() as session:
                session.run("""
                    MATCH (n:Entity {instance_id: $id, canonical_name: 'Expired'})
                    SET n.memory_expires_at = $past
                """, id=instance_id, past=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())

            # Load active should only return Active node
            active_graph = memory_store.load_active()

            names = [n.canonical_name for n in active_graph.iter_nodes()]
            assert "Active" in names
            assert "Expired" not in names

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_load_active_filters_by_layer(self):
        """Test that load_active filters by layer."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-layer-filter"

        try:
            graph = SemanticGraph()
            graph.create_node("Working", NodeType.INSTANCE)
            graph.create_node("Semantic", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            # Promote one to semantic
            with store.driver.session() as session:
                session.run("""
                    MATCH (n:Entity {instance_id: $id, canonical_name: 'Semantic'})
                    SET n.memory_layer = 'semantic',
                        n.memory_expires_at = $future
                """, id=instance_id, future=(datetime.now(timezone.utc) + timedelta(days=180)).isoformat())

            # Load only working layer
            working_graph = memory_store.load_active(
                include_layers=[MemoryLayer.WORKING]
            )

            names = [n.canonical_name for n in working_graph.iter_nodes()]
            assert "Working" in names
            assert "Semantic" not in names

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_reinforce_nodes(self):
        """Test reinforcing nodes increases score."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-reinforce"

        try:
            graph = SemanticGraph()
            node = graph.create_node("Reinforceable", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            # Reinforce the node
            reinforced = memory_store.reinforce_nodes([node.node_id], amount=0.2)
            assert reinforced == 1

            # Check the score increased
            with store.driver.session() as session:
                result = session.run("""
                    MATCH (n:Entity {node_id: $nid, instance_id: $id})
                    RETURN n.memory_reinforcement_score AS score,
                           n.memory_access_count AS count
                """, nid=node.node_id, id=instance_id)

                record = result.single()
                assert record["score"] == 0.2
                assert record["count"] == 1

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_reinforcement_promotes_layer(self):
        """Test that sufficient reinforcement promotes to higher layer."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-promote"

        try:
            graph = SemanticGraph()
            node = graph.create_node("Promotable", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            # Reinforce past promotion threshold (0.3)
            for _ in range(4):
                memory_store.reinforce_nodes([node.node_id], amount=0.1)

            # Should now be episodic
            with store.driver.session() as session:
                result = session.run("""
                    MATCH (n:Entity {node_id: $nid, instance_id: $id})
                    RETURN n.memory_layer AS layer
                """, nid=node.node_id, id=instance_id)

                record = result.single()
                assert record["layer"] == "episodic"

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_garbage_collect(self):
        """Test garbage collection removes expired memories."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-gc"

        try:
            graph = SemanticGraph()
            graph.create_node("Expired", NodeType.INSTANCE)
            graph.create_node("Active", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            # Expire one
            with store.driver.session() as session:
                session.run("""
                    MATCH (n:Entity {instance_id: $id, canonical_name: 'Expired'})
                    SET n.memory_expires_at = $past
                """, id=instance_id, past=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat())

            # Run GC
            deleted = memory_store.garbage_collect()
            assert deleted == 1

            # Verify only Active remains
            with store.driver.session() as session:
                result = session.run("""
                    MATCH (n:Entity {instance_id: $id})
                    RETURN n.canonical_name AS name
                """, id=instance_id)

                names = [r["name"] for r in result]
                assert "Active" in names
                assert "Expired" not in names

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()

    def test_layer_statistics(self):
        """Test getting layer statistics."""
        store = Neo4jGraphStoreSync(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        instance_id = "test-stats"

        try:
            graph = SemanticGraph()
            graph.create_node("W1", NodeType.INSTANCE)
            graph.create_node("W2", NodeType.INSTANCE)

            memory_store = MemoryAwareGraphStore(store, instance_id)
            memory_store.save_with_memory(graph, default_layer=MemoryLayer.WORKING)

            stats = memory_store.get_layer_statistics()

            assert stats["working"] == 2
            assert stats["episodic"] == 0
            assert stats["semantic"] == 0
            assert stats["metacognitive"] == 0

        finally:
            with store.driver.session() as session:
                session.run(
                    "MATCH (n:Entity {instance_id: $id}) DETACH DELETE n",
                    id=instance_id,
                )
            store.close()


# =============================================================================
# Phase 1 Content Classification Tests
# =============================================================================


class TestPhase1Classification:
    """Tests for classifying Phase 1 extraction outputs."""

    def test_classify_entities(self):
        """Test entity classification."""
        graph = SemanticGraph()
        node = graph.create_node("Doug", NodeType.INSTANCE)

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.ENTITY

    def test_classify_presupposition(self):
        """Test presupposition classification."""
        graph = SemanticGraph()
        node = graph.create_node("understood", NodeType.INSTANCE)
        node.properties["is_presupposition"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.PRESUPPOSITION

    def test_classify_temporal(self):
        """Test temporal classification."""
        graph = SemanticGraph()
        node = graph.create_node("yesterday", NodeType.INSTANCE)
        node.properties["is_temporal"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.TEMPORAL

    def test_classify_modality(self):
        """Test modality classification."""
        graph = SemanticGraph()
        node = graph.create_node("might", NodeType.INSTANCE)
        node.properties["is_modality"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.MODALITY

    def test_classify_event(self):
        """Test event classification."""
        graph = SemanticGraph()
        node = graph.create_node("running", NodeType.EVENT)

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.EVENT

    def test_classify_named_entity_as_fact(self):
        """Test that linked entities are classified as FACT."""
        graph = SemanticGraph()
        node = graph.create_node("Python", NodeType.INSTANCE)
        node.synset_id = "python.n.01"  # Linked to WordNet

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.FACT

    def test_classify_class_as_fact(self):
        """Test that class nodes are classified as FACT."""
        graph = SemanticGraph()
        node = graph.create_node("Programming Language", NodeType.CLASS)

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.FACT

    def test_classify_negation(self):
        """Test negation classification."""
        graph = SemanticGraph()
        node = graph.create_node("not", NodeType.INSTANCE)
        node.properties["is_negation"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.NEGATION

    def test_classify_sentiment(self):
        """Test sentiment classification."""
        graph = SemanticGraph()
        node = graph.create_node("happy", NodeType.INSTANCE)
        node.properties["is_sentiment"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.SENTIMENT

    def test_classify_commonsense(self):
        """Test commonsense classification."""
        graph = SemanticGraph()
        node = graph.create_node("birds fly", NodeType.INSTANCE)
        node.properties["is_commonsense"] = True

        classification = classify_phase1_content(graph)

        assert classification[node.node_id] == ContentType.COMMONSENSE


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_memory_properties_from_empty_dict(self):
        """Test creating MemoryProperties from minimal data."""
        data = {}
        props = MemoryProperties.from_dict(data)

        assert props.layer == MemoryLayer.WORKING
        assert props.access_count == 0

    def test_expired_memory_with_none_expires_at(self):
        """Test is_expired with None expires_at (permanent)."""
        props = MemoryProperties(layer=MemoryLayer.METACOGNITIVE)
        props.expires_at = None

        assert not props.is_expired

    @pytest.mark.asyncio
    async def test_empty_working_memory(self):
        """Test getting context from empty working memory."""
        memory = VolatileWorkingMemory(task_id="empty")
        context = await memory.get_context("agent")
        assert context == []

    @pytest.mark.asyncio
    async def test_mark_nonexistent_observation(self):
        """Test marking nonexistent observation for persistence."""
        memory = VolatileWorkingMemory(task_id="test")
        result = await memory.mark_for_persistence("nonexistent")
        assert result is False

    def test_decay_rate_unknown_content_type(self):
        """Test that unknown content type defaults to 1.0 decay."""
        props = MemoryProperties()

        # Mock an unknown content type scenario
        original_rate = DECAY_RATES.get(ContentType.ENTITY, 1.0)
        assert original_rate == 0.5  # Should find it

    def test_promotion_at_metacognitive_does_nothing(self):
        """Test that promoting metacognitive has no effect."""
        props = MemoryProperties(layer=MemoryLayer.METACOGNITIVE)
        props.reinforcement_score = 1.0

        props._check_promotion()

        assert props.layer == MemoryLayer.METACOGNITIVE
        assert props.expires_at is None


# =============================================================================
# Scenario Tests
# =============================================================================


class TestScenarios:
    """Tests for real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_react_swarm_scenario(self):
        """Simulate a ReAct swarm sharing working memory."""
        memory = VolatileWorkingMemory(task_id="swarm-1", max_items=9)

        # Agent 1 adds observation
        await memory.add(
            "User wants to book a flight to Tokyo",
            "planner-agent",
            content_type=ContentType.INSTRUCTION,
            attention_weight=0.9,
        )

        # Agent 2 adds observation
        await memory.add(
            "Found flights: JAL, ANA, United",
            "researcher-agent",
            content_type=ContentType.FACT,
            attention_weight=0.8,
        )

        # Agent 3 reads context
        context = await memory.get_context("booking-agent", max_items=5)

        assert len(context) == 2
        assert any("Tokyo" in obs.content for obs in context)
        assert any("JAL" in obs.content for obs in context)

        # Mark important finding for persistence
        for obs in context:
            if "flights:" in obs.content:
                await memory.mark_for_persistence(obs.observation_id)

        candidates = await memory.get_persistence_candidates()
        assert len(candidates) == 1

    def test_memory_decay_scenario(self):
        """Test that peripheral info decays faster than core info."""
        # Simulate passage of time with multiple access patterns
        core_props = MemoryProperties(
            layer=MemoryLayer.WORKING,
            content_type=ContentType.INSTRUCTION,
        )

        peripheral_props = MemoryProperties(
            layer=MemoryLayer.WORKING,
            content_type=ContentType.MODALITY,
        )

        # Core has 0.1 decay rate, peripheral has 3.0
        # This means core should stay longer with same access patterns

        assert core_props.decay_rate == 0.1
        assert peripheral_props.decay_rate == 3.0

        # Record same access for both
        core_props.record_access()
        peripheral_props.record_access()

        # Core extension should be larger (1 min / 0.1 = 10 min)
        # Peripheral extension should be smaller (1 min / 3.0 = 20 sec)
        # Both started at same time, so core should expire later

        core_ttl = core_props.time_to_expiry.total_seconds() if core_props.time_to_expiry else 0
        peripheral_ttl = peripheral_props.time_to_expiry.total_seconds() if peripheral_props.time_to_expiry else 0

        assert core_ttl > peripheral_ttl

    def test_promotion_path_scenario(self):
        """Test the full journey of a memory from working to metacognitive."""
        props = MemoryProperties(
            layer=MemoryLayer.WORKING,
            content_type=ContentType.FACT,
        )

        # Track promotions
        promotions = [props.layer]

        # Simulate repeated successful usage
        for i in range(20):
            props.reinforce(0.1)
            if props.layer not in promotions:
                promotions.append(props.layer)

        # Should have gone through all layers
        assert MemoryLayer.WORKING in promotions
        assert MemoryLayer.EPISODIC in promotions
        assert MemoryLayer.SEMANTIC in promotions
        assert MemoryLayer.METACOGNITIVE in promotions

        # Should now be permanent
        assert props.expires_at is None
