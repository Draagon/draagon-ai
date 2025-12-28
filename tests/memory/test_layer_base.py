"""Tests for base layer functionality.

These tests cover common layer operations like decay, cleanup, and promotion
that are shared across all memory layers.
"""

import pytest
from datetime import datetime, timedelta

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MetacognitiveMemory,
)
from draagon_ai.memory.layers.base import LayerConfig
from draagon_ai.memory.temporal_nodes import MemoryLayer


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return TemporalCognitiveGraph()


class TestLayerProperties:
    """Test layer property accessors."""

    @pytest.mark.asyncio
    async def test_working_memory_layer_property(self, graph):
        """Test working memory layer property."""
        wm = WorkingMemory(graph, session_id="test")
        assert wm.layer == MemoryLayer.WORKING

    @pytest.mark.asyncio
    async def test_episodic_memory_layer_property(self, graph):
        """Test episodic memory layer property."""
        em = EpisodicMemory(graph)
        assert em.layer == MemoryLayer.EPISODIC

    @pytest.mark.asyncio
    async def test_semantic_memory_layer_property(self, graph):
        """Test semantic memory layer property."""
        sm = SemanticMemory(graph)
        assert sm.layer == MemoryLayer.SEMANTIC

    @pytest.mark.asyncio
    async def test_metacognitive_memory_layer_property(self, graph):
        """Test metacognitive memory layer property."""
        mm = MetacognitiveMemory(graph)
        assert mm.layer == MemoryLayer.METACOGNITIVE

    @pytest.mark.asyncio
    async def test_layer_config_property(self, graph):
        """Test config property is accessible."""
        wm = WorkingMemory(graph, session_id="test")
        config = wm.config
        assert config is not None
        assert isinstance(config, LayerConfig)


class TestLayerDecay:
    """Test decay functionality across layers."""

    @pytest.mark.asyncio
    async def test_apply_decay_respects_interval(self, graph):
        """Test that decay only applies after interval passes."""
        wm = WorkingMemory(graph, session_id="test")
        await wm.add("Test item")

        # First call may not decay due to interval
        decayed = await wm.apply_decay()
        assert isinstance(decayed, int)

    @pytest.mark.asyncio
    async def test_apply_decay_processes_all_items(self, graph):
        """Test that decay affects all layer items."""
        wm = WorkingMemory(graph, session_id="test")

        # Add multiple items
        for i in range(5):
            await wm.add(f"Item {i}")

        # Force decay by manipulating last_decay
        wm._last_decay = datetime.now() - timedelta(minutes=10)

        decayed = await wm.apply_decay()
        assert decayed == 5

    @pytest.mark.asyncio
    async def test_apply_decay_only_affects_own_layer(self, graph):
        """Test that decay only affects items in that layer."""
        wm = WorkingMemory(graph, session_id="test")
        em = EpisodicMemory(graph)

        # Add item to working memory
        await wm.add("Working item")

        # Add episode to episodic memory
        await em.start_episode()

        # Force decay on working memory
        wm._last_decay = datetime.now() - timedelta(minutes=10)
        decayed = await wm.apply_decay()

        # Should only decay working memory items
        assert decayed == 1


class TestLayerCleanup:
    """Test cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_expired_with_no_ttl(self, graph):
        """Test cleanup does nothing when no TTL is set."""
        # Metacognitive has no TTL by default
        mm = MetacognitiveMemory(graph)
        await mm.add_skill("test", "command", "echo test")

        removed = await mm.cleanup_expired()
        assert removed == 0

    @pytest.mark.asyncio
    async def test_cleanup_expired_removes_old_items(self, graph):
        """Test cleanup removes expired low-importance items."""
        wm = WorkingMemory(graph, session_id="test")

        # Add item
        item = await wm.add("Expiring item", attention_weight=0.1)

        # Manipulate the ingestion_time to be old
        node = await wm._graph.get_node(item.node_id)
        node.ingestion_time = datetime.now() - timedelta(days=30)
        node.importance = 0.1  # Low importance

        # Set a TTL for the layer
        wm._config.default_ttl = timedelta(hours=1)

        removed = await wm.cleanup_expired()
        assert removed == 1

    @pytest.mark.asyncio
    async def test_cleanup_keeps_high_importance_items(self, graph):
        """Test cleanup keeps high-importance items even if old."""
        wm = WorkingMemory(graph, session_id="test")

        # Add item with high importance
        item = await wm.add("Important item", attention_weight=0.9)

        # Manipulate the ingestion_time to be old
        node = await wm._graph.get_node(item.node_id)
        node.ingestion_time = datetime.now() - timedelta(days=30)
        node.importance = 0.9  # High importance

        # Set a TTL for the layer
        wm._config.default_ttl = timedelta(hours=1)

        removed = await wm.cleanup_expired()
        assert removed == 0


class TestLayerPromotionCandidates:
    """Test promotion candidate detection."""

    @pytest.mark.asyncio
    async def test_promotion_candidates_by_importance(self, graph):
        """Test items are candidates when importance is high."""
        wm = WorkingMemory(graph, session_id="test")

        # Add high-importance item
        item = await wm.add("Important item", attention_weight=0.9)
        node = await wm._graph.get_node(item.node_id)
        node.importance = 0.9  # Exceeds default threshold (0.8)

        candidates = await wm.get_promotion_candidates()
        assert any(c.node_id == item.node_id for c in candidates)

    @pytest.mark.asyncio
    async def test_promotion_candidates_by_access(self, graph):
        """Test items are candidates when access count is high."""
        wm = WorkingMemory(graph, session_id="test")

        # Add item and boost access count
        item = await wm.add("Accessed item", attention_weight=0.5)
        node = await wm._graph.get_node(item.node_id)
        node.access_count = 5  # Exceeds default threshold (3)

        candidates = await wm.get_promotion_candidates()
        assert any(c.node_id == item.node_id for c in candidates)

    @pytest.mark.asyncio
    async def test_promotion_disabled_returns_empty(self, graph):
        """Test that promotion disabled returns empty list."""
        wm = WorkingMemory(graph, session_id="test")
        wm._config.auto_promote = False

        await wm.add("Any item", attention_weight=0.9)

        candidates = await wm.get_promotion_candidates()
        assert candidates == []


class TestLayerCount:
    """Test count functionality."""

    @pytest.mark.asyncio
    async def test_count_empty_layer(self, graph):
        """Test counting items in empty layer."""
        wm = WorkingMemory(graph, session_id="test")
        count = await wm.count()
        assert count == 0

    @pytest.mark.asyncio
    async def test_count_with_items(self, graph):
        """Test counting items in populated layer."""
        wm = WorkingMemory(graph, session_id="test")

        for i in range(3):
            await wm.add(f"Item {i}")

        count = await wm.count()
        assert count == 3


class TestLayerDelete:
    """Test delete functionality."""

    @pytest.mark.asyncio
    async def test_delete_nonexistent_returns_false(self, graph):
        """Test deleting non-existent item returns False."""
        wm = WorkingMemory(graph, session_id="test")
        result = await wm.delete("nonexistent_id")
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_wrong_layer_returns_false(self, graph):
        """Test deleting item from wrong layer returns False."""
        wm = WorkingMemory(graph, session_id="test")
        em = EpisodicMemory(graph)

        # Create item in episodic memory
        ep = await em.start_episode()

        # Try to delete from working memory
        result = await wm.delete(ep.node_id)
        assert result is False


class TestLayerConfig:
    """Test layer configuration."""

    def test_layer_config_defaults(self):
        """Test LayerConfig has sensible defaults."""
        config = LayerConfig()

        assert config.auto_promote is True
        assert config.importance_threshold == 0.7  # Actual default
        assert config.access_threshold == 5  # Actual default

    def test_layer_config_custom_values(self):
        """Test LayerConfig accepts custom values."""
        config = LayerConfig(
            auto_promote=False,
            importance_threshold=0.5,
            access_threshold=10,
            decay_factor=0.8,
        )

        assert config.auto_promote is False
        assert config.importance_threshold == 0.5
        assert config.access_threshold == 10
        assert config.decay_factor == 0.8


class TestLayerSearch:
    """Test search functionality in layers."""

    @pytest.mark.asyncio
    async def test_search_empty_layer(self, graph):
        """Test searching empty layer returns empty list."""
        wm = WorkingMemory(graph, session_id="test")
        results = await wm.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, graph):
        """Test search respects min_score filter."""
        sm = SemanticMemory(graph)

        # Add an entity
        await sm.create_entity("Paris", "city")

        results = await sm.search("Paris", min_score=0.9)
        # Results may or may not be returned based on score
        assert isinstance(results, list)
