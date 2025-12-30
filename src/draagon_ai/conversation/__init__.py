"""Conversation state management for draagon-ai.

This module provides conversation tracking and state management for AI agents:

- **ConversationState**: Tracks history, pending actions, and session data
- **ConversationMode**: Detects and adapts to conversation modes (task, brainstorm, support)
- **UndoableAction**: Enables undoing recent actions
- **ConversationManager**: Manages multiple concurrent conversations

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Conversation Manager                          │
    │                                                                  │
    │  User → [Get/Create State] → [Update History] → [Track State]  │
    │                                                                  │
    │  Features:                                                       │
    │  - Multi-turn history with tool results                         │
    │  - Timeout-based expiration                                     │
    │  - Pending action tracking (confirmations, passcodes)          │
    │  - Undo support for reversible actions                          │
    │  - Conversation mode detection (task/brainstorm/support)        │
    │  - Episode summarization on expiration                          │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.conversation import (
        ConversationState,
        ConversationManager,
        ConversationMode,
        ConversationModeType,
        UndoableAction,
    )

    # Create a manager
    manager = ConversationManager()

    # Get or create conversation state
    state = manager.get_or_create(conversation_id="user123")

    # Add a turn
    state.add_turn(
        user="Turn on the lights",
        assistant="Done! I turned on the lights.",
        tool_calls=[{"tool": "home_assistant", "result": {"success": True}}],
    )

    # Track undoable action
    state.last_undoable_action = UndoableAction(
        action_type="ha_control",
        details={"entity_id": "light.living_room", "previous_state": "off"},
    )
"""

from .state import (
    ConversationState,
    ConversationMode,
    ConversationModeType,
    ToolCallInfo,
    SentimentResult,
)
from .undo import (
    UndoableAction,
    UndoResult,
)
from .manager import (
    ConversationManager,
    ConversationManagerConfig,
    ExpiredConversation,
)

__all__ = [
    # State
    "ConversationState",
    "ConversationMode",
    "ConversationModeType",
    "ToolCallInfo",
    "SentimentResult",
    # Undo
    "UndoableAction",
    "UndoResult",
    # Manager
    "ConversationManager",
    "ConversationManagerConfig",
    "ExpiredConversation",
]
