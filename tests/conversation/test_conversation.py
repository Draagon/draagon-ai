"""Tests for conversation state management."""

import pytest
import time

from draagon_ai.conversation import (
    ConversationState,
    ConversationManager,
    ConversationManagerConfig,
    ConversationMode,
    ConversationModeType,
    UndoableAction,
    UndoResult,
    ToolCallInfo,
    SentimentResult,
)


class TestConversationState:
    """Tests for ConversationState class."""

    def test_create_state(self):
        """Test creating a conversation state."""
        state = ConversationState()

        assert state.history == []
        assert state.max_history == 10
        assert state.timeout_seconds == 300
        assert state.pending_command is None
        assert state.mode.primary == ConversationModeType.TASK

    def test_add_turn(self):
        """Test adding conversation turns."""
        state = ConversationState()

        state.add_turn("Hello", "Hi there!")
        state.add_turn("What time is it?", "It's 3:00 PM.")

        assert len(state.history) == 2
        assert state.history[0]["user"] == "Hello"
        assert state.history[1]["assistant"] == "It's 3:00 PM."

    def test_add_turn_with_tool_calls(self):
        """Test adding turns with tool call results."""
        state = ConversationState()

        tool_calls = [
            ToolCallInfo(tool="get_time", result="3:00 PM"),
            ToolCallInfo(tool="failed_tool", error="Not found"),
        ]

        state.add_turn("What time is it?", "It's 3:00 PM.", tool_calls=tool_calls)

        assert len(state.history) == 1
        # Only successful tool calls are stored
        assert len(state.history[0]["tool_results"]) == 1
        assert state.history[0]["tool_results"][0]["tool"] == "get_time"

    def test_max_history(self):
        """Test that history is trimmed to max."""
        state = ConversationState(max_history=3)

        for i in range(5):
            state.add_turn(f"User {i}", f"Assistant {i}")

        assert len(state.history) == 3
        assert state.history[0]["user"] == "User 2"

    def test_is_expired(self):
        """Test expiration detection."""
        state = ConversationState(timeout_seconds=1)

        assert not state.is_expired()

        # Simulate time passing
        state.last_access = time.time() - 2

        assert state.is_expired()

    def test_get_messages(self):
        """Test getting history as LLM messages."""
        state = ConversationState()

        state.add_turn("Hello", "Hi!")
        state.add_turn("How are you?", "I'm good!")

        messages = state.get_messages()

        assert len(messages) == 4
        assert messages[0] == {"role": "user", "content": "Hello"}
        assert messages[1] == {"role": "assistant", "content": "Hi!"}

    def test_clear(self):
        """Test clearing state."""
        state = ConversationState()

        state.add_turn("Test", "Response")
        state.pending_command = {"action": "test"}
        state.mode.update(ConversationModeType.BRAINSTORM)

        state.clear()

        assert state.history == []
        assert state.pending_command is None
        assert state.mode.primary == ConversationModeType.TASK

    def test_to_dict_from_dict(self):
        """Test serialization."""
        state = ConversationState()

        state.add_turn("Hello", "Hi!")
        state.current_area_id = "living_room"
        state.mode.update(ConversationModeType.CASUAL)

        data = state.to_dict()
        restored = ConversationState.from_dict(data)

        assert len(restored.history) == 1
        assert restored.current_area_id == "living_room"
        assert restored.mode.primary == ConversationModeType.CASUAL


class TestConversationMode:
    """Tests for ConversationMode class."""

    def test_default_mode(self):
        """Test default mode is TASK."""
        mode = ConversationMode()

        assert mode.primary == ConversationModeType.TASK
        assert mode.is_engaged() is False

    def test_update_mode(self):
        """Test updating conversation mode."""
        mode = ConversationMode()

        mode.update(ConversationModeType.BRAINSTORM, confidence=0.9)

        assert mode.primary == ConversationModeType.BRAINSTORM
        assert mode.confidence == 0.9
        assert mode.turns_in_mode == 1
        assert mode.is_engaged() is True

    def test_update_same_mode(self):
        """Test updating with same mode increments turns."""
        mode = ConversationMode()

        mode.update(ConversationModeType.SUPPORT)
        mode.update(ConversationModeType.SUPPORT)
        mode.update(ConversationModeType.SUPPORT)

        assert mode.turns_in_mode == 3

    def test_exit_detected(self):
        """Test exit detection resets mode."""
        mode = ConversationMode()

        mode.update(ConversationModeType.BRAINSTORM)
        mode.update(ConversationModeType.TASK, exit_detected=True)

        assert mode.primary == ConversationModeType.TASK
        assert mode.turns_in_mode == 0


class TestUndoableAction:
    """Tests for UndoableAction class."""

    def test_create_action(self):
        """Test creating an undoable action."""
        action = UndoableAction(
            action_type="calendar_create",
            details={"summary": "Meeting", "event_id": "123"},
        )

        assert action.action_type == "calendar_create"
        assert action.details["summary"] == "Meeting"
        assert action.expires_at is not None

    def test_describe_calendar_create(self):
        """Test describing calendar create action."""
        action = UndoableAction(
            action_type="calendar_create",
            details={"summary": "Team Meeting"},
        )

        assert "Team Meeting" in action.describe()
        assert "creation" in action.describe()

    def test_describe_ha_control(self):
        """Test describing home assistant action."""
        action = UndoableAction(
            action_type="ha_control",
            details={"entity_id": "light.living_room", "service": "turn_on"},
        )

        assert "light.living_room" in action.describe()
        assert "turn_on" in action.describe()

    def test_is_expired(self):
        """Test expiration detection."""
        action = UndoableAction(
            action_type="test",
            details={},
            expires_at=time.time() - 1,
        )

        assert action.is_expired() is True

        action.expires_at = time.time() + 100
        assert action.is_expired() is False

    def test_to_dict_from_dict(self):
        """Test serialization."""
        action = UndoableAction(
            action_type="memory_store",
            details={"content": "Test memory"},
            user_id="user123",
        )

        data = action.to_dict()
        restored = UndoableAction.from_dict(data)

        assert restored.action_type == "memory_store"
        assert restored.details["content"] == "Test memory"
        assert restored.user_id == "user123"


class TestToolCallInfo:
    """Tests for ToolCallInfo class."""

    def test_create_tool_call(self):
        """Test creating tool call info."""
        info = ToolCallInfo(
            tool="get_time",
            args={"format": "12h"},
            result="3:00 PM",
            elapsed_ms=50.5,
        )

        assert info.tool == "get_time"
        assert info.result == "3:00 PM"
        assert info.error is None

    def test_to_dict_from_dict(self):
        """Test serialization."""
        info = ToolCallInfo(
            tool="web_search",
            args={"query": "weather"},
            result={"temp": 72},
        )

        data = info.to_dict()
        restored = ToolCallInfo.from_dict(data)

        assert restored.tool == "web_search"
        assert restored.result["temp"] == 72


class TestSentimentResult:
    """Tests for SentimentResult class."""

    def test_default_sentiment(self):
        """Test default sentiment."""
        sentiment = SentimentResult()

        assert sentiment.sentiment == "neutral"
        assert sentiment.needs_support is False

    def test_custom_sentiment(self):
        """Test custom sentiment."""
        sentiment = SentimentResult(
            sentiment="frustrated",
            confidence=0.9,
            emotions={"anger": 0.3, "sadness": 0.5},
            needs_support=True,
        )

        assert sentiment.sentiment == "frustrated"
        assert sentiment.needs_support is True
        assert sentiment.emotions["sadness"] == 0.5


class TestConversationManager:
    """Tests for ConversationManager class."""

    def test_create_manager(self):
        """Test creating a manager."""
        manager = ConversationManager()

        assert manager.active_count() == 0

    def test_get_or_create(self):
        """Test getting or creating conversations."""
        manager = ConversationManager()

        state1 = manager.get_or_create("conv1")
        state2 = manager.get_or_create("conv1")
        state3 = manager.get_or_create("conv2")

        assert state1 is state2  # Same conversation
        assert state1 is not state3  # Different conversations
        assert manager.active_count() == 2

    def test_get_nonexistent(self):
        """Test getting non-existent conversation."""
        manager = ConversationManager()

        state = manager.get("nonexistent")

        assert state is None

    def test_delete(self):
        """Test deleting conversations."""
        manager = ConversationManager()

        manager.get_or_create("conv1")
        assert manager.active_count() == 1

        result = manager.delete("conv1")
        assert result is True
        assert manager.active_count() == 0

        result = manager.delete("nonexistent")
        assert result is False

    def test_clear(self):
        """Test clearing conversation history."""
        manager = ConversationManager()

        state = manager.get_or_create("conv1")
        state.add_turn("Test", "Response")
        state.pending_command = {"action": "test"}

        result = manager.clear("conv1")

        assert result is True
        assert len(state.history) == 0
        assert state.pending_command is None

    def test_expired_conversation(self):
        """Test expired conversation handling."""
        config = ConversationManagerConfig(timeout_seconds=1, cleanup_interval=0)
        manager = ConversationManager(config=config)

        state = manager.get_or_create("conv1")
        state.add_turn("Test", "Response")

        # Simulate time passing
        state.last_access = time.time() - 2

        # Try to get expired conversation
        result = manager.get("conv1")

        assert result is None  # Expired
        assert manager.active_count() == 0

    def test_cleanup_expired(self):
        """Test cleanup of expired conversations."""
        config = ConversationManagerConfig(timeout_seconds=1, cleanup_interval=0)
        manager = ConversationManager(config=config)

        state1 = manager.get_or_create("conv1")
        state2 = manager.get_or_create("conv2")

        # Expire one conversation
        state1.last_access = time.time() - 2

        expired = manager.cleanup_expired()

        assert len(expired) == 1
        assert expired[0].conversation_id == "conv1"
        assert manager.active_count() == 1

    def test_get_stats(self):
        """Test getting manager stats."""
        manager = ConversationManager()

        manager.get_or_create("conv1")
        manager.get_or_create("conv2")

        stats = manager.get_stats()

        assert stats["active_conversations"] == 2
        assert "config" in stats

    def test_list_active_ids(self):
        """Test listing active conversation IDs."""
        manager = ConversationManager()

        manager.get_or_create("conv1")
        manager.get_or_create("conv2")

        ids = manager.list_active_ids()

        assert "conv1" in ids
        assert "conv2" in ids
        assert len(ids) == 2


class TestUndoResult:
    """Tests for UndoResult class."""

    def test_successful_undo(self):
        """Test successful undo result."""
        result = UndoResult(
            success=True,
            message="Undid calendar event creation",
            details={"event_id": "123"},
        )

        assert result.success is True
        assert "calendar" in result.message

    def test_failed_undo(self):
        """Test failed undo result."""
        result = UndoResult(
            success=False,
            message="Could not undo - action expired",
        )

        assert result.success is False
