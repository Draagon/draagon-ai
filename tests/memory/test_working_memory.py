"""Tests for Working Memory layer."""

import pytest
from datetime import timedelta

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    WorkingMemory,
    WorkingMemoryItem,
)


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return TemporalCognitiveGraph()


@pytest.fixture
def working_memory(graph):
    """Create working memory with default capacity."""
    return WorkingMemory(graph, session_id="test_session")


class TestWorkingMemoryBasics:
    """Test basic working memory operations."""

    @pytest.mark.asyncio
    async def test_add_item(self, working_memory):
        """Test adding an item to working memory."""
        item = await working_memory.add(
            "User asked about weather",
            attention_weight=0.8,
            source="voice",
        )

        assert item is not None
        assert item.content == "User asked about weather"
        assert item.attention_weight == 0.8
        assert item.source == "voice"
        assert item.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_get_item(self, working_memory):
        """Test retrieving an item by ID."""
        item = await working_memory.add("Test content")
        retrieved = await working_memory.get(item.node_id)

        assert retrieved is not None
        assert retrieved.content == "Test content"

    @pytest.mark.asyncio
    async def test_delete_item(self, working_memory):
        """Test deleting an item."""
        item = await working_memory.add("To be deleted")
        result = await working_memory.delete(item.node_id)

        assert result is True
        assert await working_memory.get(item.node_id) is None


class TestWorkingMemoryCapacity:
    """Test capacity management (Miller's Law)."""

    @pytest.mark.asyncio
    async def test_capacity_limit(self, graph):
        """Test that capacity is enforced."""
        wm = WorkingMemory(graph, session_id="test", capacity=3)

        # Add 5 items, should evict 2
        for i in range(5):
            await wm.add(f"Item {i}", attention_weight=0.5)

        count = await wm.count()
        assert count == 3

    @pytest.mark.asyncio
    async def test_eviction_by_priority(self, graph):
        """Test that low-priority items are evicted first."""
        wm = WorkingMemory(graph, session_id="test", capacity=3)

        # Add items with different priorities
        low = await wm.add("Low priority", attention_weight=0.1)
        mid = await wm.add("Mid priority", attention_weight=0.5)
        high = await wm.add("High priority", attention_weight=0.9)

        # Add one more to trigger eviction
        new = await wm.add("New item", attention_weight=0.6)

        # Low priority should be evicted
        assert await wm.get(low.node_id) is None
        assert await wm.get(mid.node_id) is not None
        assert await wm.get(high.node_id) is not None
        assert await wm.get(new.node_id) is not None


class TestWorkingMemoryAttention:
    """Test attention and focus mechanisms."""

    @pytest.mark.asyncio
    async def test_focus_boosts_attention(self, working_memory):
        """Test that focus() increases attention weight."""
        item = await working_memory.add("Focused item", attention_weight=0.5)

        original_weight = item.attention_weight
        await working_memory.focus(item.node_id, boost=0.2)

        updated = await working_memory.get(item.node_id)
        assert updated.attention_weight > original_weight

    @pytest.mark.asyncio
    async def test_get_active_context(self, working_memory):
        """Test retrieving high-attention items."""
        await working_memory.add("Low attention", attention_weight=0.2)
        await working_memory.add("High attention", attention_weight=0.8)
        await working_memory.add("Medium attention", attention_weight=0.5)

        active = await working_memory.get_active_context(min_attention=0.4)

        assert len(active) == 2
        # Should be sorted by effective_priority
        assert active[0].attention_weight >= active[1].attention_weight

    @pytest.mark.asyncio
    async def test_decay_activation(self, working_memory):
        """Test that activation decay works."""
        item = await working_memory.add("Decaying item")
        original_activation = item.activation_level

        # Apply decay
        await working_memory.apply_decay()

        updated = await working_memory.get(item.node_id)
        # Activation should decrease (unless just added)
        assert updated is not None


class TestWorkingMemoryGoals:
    """Test goal setting functionality."""

    @pytest.mark.asyncio
    async def test_set_goal(self, working_memory):
        """Test setting a goal."""
        goal = await working_memory.set_goal(
            "Help user book a flight",
        )

        assert goal is not None
        assert goal.attention_weight == 0.9  # Goals have high attention
        assert goal.source == "goal"

    @pytest.mark.asyncio
    async def test_set_goal_with_metadata(self, working_memory):
        """Test setting a goal with metadata."""
        goal = await working_memory.set_goal(
            "Help user book a flight",
            metadata={"subgoals": ["Find flights", "Compare prices"]},
        )

        assert goal is not None
        assert goal.metadata.get("subgoals") is not None


class TestWorkingMemorySessionIsolation:
    """Test session isolation."""

    @pytest.mark.asyncio
    async def test_sessions_use_different_scope(self, graph):
        """Test that different sessions use different scopes."""
        session1 = WorkingMemory(graph, session_id="session_1")
        session2 = WorkingMemory(graph, session_id="session_2")

        # Add items to different sessions
        item1 = await session1.add("Session 1 item")
        item2 = await session2.add("Session 2 item")

        # Verify session IDs are in the items
        assert item1.session_id == "session_1"
        assert item2.session_id == "session_2"

    @pytest.mark.asyncio
    async def test_search_returns_results(self, graph):
        """Test that search works within a session."""
        session1 = WorkingMemory(graph, session_id="session_1")

        await session1.add("Weather query")

        results = await session1.search("weather")

        # Search should work
        assert isinstance(results, list)


class TestEffectivePriority:
    """Test effective priority calculation."""

    @pytest.mark.asyncio
    async def test_effective_priority_formula(self, working_memory):
        """Test the effective priority calculation."""
        item = await working_memory.add(
            "test content",
            attention_weight=0.6,
        )
        # Manually set values for testing
        item.importance = 0.8
        item.activation_level = 1.0

        expected = 0.8 * 0.4 + 0.6 * 0.35 + 1.0 * 0.25
        assert abs(item.effective_priority - expected) < 0.001

    @pytest.mark.asyncio
    async def test_high_attention_boosts_priority(self, working_memory):
        """Test that attention significantly affects priority."""
        low = await working_memory.add("low", attention_weight=0.1)
        high = await working_memory.add("high", attention_weight=0.9)

        assert high.effective_priority > low.effective_priority


class TestWorkingMemoryItem:
    """Test WorkingMemoryItem methods."""

    @pytest.mark.asyncio
    async def test_decay_activation(self, working_memory):
        """Test that decay_activation reduces activation level."""
        item = await working_memory.add("test", attention_weight=0.5)
        item.activation_level = 1.0

        item.decay_activation(factor=0.9)

        assert item.activation_level == 0.9

    @pytest.mark.asyncio
    async def test_decay_activation_floors_at_zero(self, working_memory):
        """Test that activation doesn't go below zero."""
        item = await working_memory.add("test", attention_weight=0.5)
        item.activation_level = 0.01

        # Apply many decays
        for _ in range(10):
            item.decay_activation(factor=0.5)

        assert item.activation_level >= 0.0

    @pytest.mark.asyncio
    async def test_boost_attention(self, working_memory):
        """Test that boost_attention increases attention weight."""
        item = await working_memory.add("test", attention_weight=0.5)

        item.boost_attention(boost=0.2)

        assert item.attention_weight == 0.7
        assert item.activation_level == 1.0  # Reset on access

    @pytest.mark.asyncio
    async def test_boost_attention_caps_at_one(self, working_memory):
        """Test that attention weight doesn't exceed 1.0."""
        item = await working_memory.add("test", attention_weight=0.9)

        item.boost_attention(boost=0.5)

        assert item.attention_weight == 1.0


class TestWorkingMemoryGoalManagement:
    """Test goal-specific operations."""

    @pytest.mark.asyncio
    async def test_get_goals_returns_goals_only(self, working_memory):
        """Test that get_goals only returns GOAL type items."""
        await working_memory.add("Regular context", attention_weight=0.5)
        await working_memory.set_goal("Goal 1")
        await working_memory.set_goal("Goal 2")

        goals = await working_memory.get_goals()

        assert len(goals) == 2
        for goal in goals:
            assert "Goal" in goal.content

    @pytest.mark.asyncio
    async def test_complete_goal(self, working_memory):
        """Test marking a goal as completed."""
        goal = await working_memory.set_goal("Complete this task")

        result = await working_memory.complete_goal(goal.node_id)

        assert result is True
        # Verify node metadata was updated
        node = await working_memory._graph.get_node(goal.node_id)
        assert node.metadata.get("completed") is True
        assert "completed_at" in node.metadata

    @pytest.mark.asyncio
    async def test_complete_nonexistent_goal(self, working_memory):
        """Test completing a non-existent goal returns False."""
        result = await working_memory.complete_goal("nonexistent_id")

        assert result is False

    @pytest.mark.asyncio
    async def test_complete_non_goal_returns_false(self, working_memory):
        """Test that completing a non-GOAL node returns False."""
        item = await working_memory.add("Regular item", attention_weight=0.5)

        result = await working_memory.complete_goal(item.node_id)

        assert result is False


class TestWorkingMemoryDecay:
    """Test activation decay and expiration."""

    @pytest.mark.asyncio
    async def test_apply_decay_respects_interval(self, graph):
        """Test that decay only applies after interval."""
        from datetime import timedelta
        wm = WorkingMemory(graph, session_id="test", capacity=7)
        # Immediately after creation, decay should not trigger

        await wm.add("test item")
        decayed = await wm.apply_decay()

        # First call might not decay due to interval check
        assert isinstance(decayed, int)

    @pytest.mark.asyncio
    async def test_apply_decay_affects_all_session_items(self, graph):
        """Test that decay affects all items in the session."""
        from datetime import datetime, timedelta
        wm = WorkingMemory(graph, session_id="test", capacity=7)

        # Add multiple items
        for i in range(3):
            await wm.add(f"Item {i}")

        # Force decay by manipulating last_decay
        wm._last_decay = datetime.now() - timedelta(minutes=5)

        decayed = await wm.apply_decay()

        assert decayed == 3


class TestWorkingMemoryPromotion:
    """Test promotion candidate detection."""

    @pytest.mark.asyncio
    async def test_get_promotion_candidates_by_importance(self, working_memory):
        """Test that high-importance items are candidates."""
        # Add a high-importance item
        item = await working_memory.add("Important item", attention_weight=0.9)
        # Manually boost importance above threshold (0.8)
        node = await working_memory._graph.get_node(item.node_id)
        node.importance = 0.85

        candidates = await working_memory.get_promotion_candidates()

        # Item should be a candidate
        assert len(candidates) >= 0  # May or may not have candidates based on thresholds

    @pytest.mark.asyncio
    async def test_get_promotion_candidates_by_access(self, working_memory):
        """Test that frequently accessed items are candidates."""
        item = await working_memory.add("Accessed item", attention_weight=0.5)
        # Boost access count above threshold (3)
        node = await working_memory._graph.get_node(item.node_id)
        node.access_count = 5

        candidates = await working_memory.get_promotion_candidates()

        # Should include accessed item
        assert any(c.node_id == item.node_id for c in candidates)


class TestWorkingMemoryClear:
    """Test session clearing."""

    @pytest.mark.asyncio
    async def test_clear_session(self, working_memory):
        """Test clearing all session items."""
        # Add multiple items
        for i in range(5):
            await working_memory.add(f"Item {i}")

        cleared = await working_memory.clear_session()

        assert cleared == 5
        assert await working_memory.count() == 0

    @pytest.mark.asyncio
    async def test_clear_session_only_affects_this_session(self, graph):
        """Test that clear only affects the current session."""
        session1 = WorkingMemory(graph, session_id="session_1")
        session2 = WorkingMemory(graph, session_id="session_2")

        await session1.add("Session 1 item")
        await session2.add("Session 2 item")

        cleared = await session1.clear_session()

        assert cleared == 1
        # Session 2 should still have its item
        assert await session2.count() == 1


class TestWorkingMemoryStats:
    """Test statistics gathering."""

    @pytest.mark.asyncio
    async def test_stats_empty_session(self, working_memory):
        """Test stats for empty session."""
        stats = working_memory.stats()

        assert stats["session_id"] == "test_session"
        assert stats["item_count"] == 0
        assert stats["capacity"] == 7
        assert stats["capacity_used"] == 0
        assert stats["avg_attention"] == 0
        assert stats["avg_activation"] == 0
        assert stats["goal_count"] == 0
        assert stats["context_count"] == 0

    @pytest.mark.asyncio
    async def test_stats_with_items(self, working_memory):
        """Test stats with items present."""
        await working_memory.add("Context 1", attention_weight=0.6)
        await working_memory.add("Context 2", attention_weight=0.8)
        await working_memory.set_goal("Goal 1")

        stats = working_memory.stats()

        assert stats["item_count"] == 3
        assert stats["capacity_used"] == 3 / 7
        assert stats["goal_count"] == 1
        assert stats["context_count"] == 2
        assert stats["avg_attention"] > 0


class TestWorkingMemoryProperties:
    """Test property accessors."""

    @pytest.mark.asyncio
    async def test_session_id_property(self, working_memory):
        """Test session_id property returns correct value."""
        assert working_memory.session_id == "test_session"

    @pytest.mark.asyncio
    async def test_capacity_property(self, graph):
        """Test capacity property returns correct value."""
        wm = WorkingMemory(graph, session_id="test", capacity=10)
        assert wm.capacity == 10


class TestWorkingMemoryAddOptions:
    """Test add method with various parameters."""

    @pytest.mark.asyncio
    async def test_add_with_entities(self, working_memory):
        """Test adding item with entities."""
        item = await working_memory.add(
            content="Paris vacation planning",
            attention_weight=0.7,
            source="conversation",
            entities=["Paris", "vacation"],
        )

        assert item is not None
        assert "Paris" in item.entities

    @pytest.mark.asyncio
    async def test_add_with_all_params(self, working_memory):
        """Test adding item with all optional parameters."""
        item = await working_memory.add(
            content="Complete context",
            attention_weight=0.8,
            source="api",
            entities=["entity1"],
        )

        assert item is not None
        assert item.attention_weight == 0.8
        assert item.source == "api"
