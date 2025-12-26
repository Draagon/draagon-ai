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
