"""Conversation state tracking.

This module provides the core data structures for tracking conversation state,
including history, pending actions, and conversation modes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ConversationModeType(str, Enum):
    """Types of conversation modes the agent can detect and adapt to."""

    TASK = "task"  # Default: direct commands, execute efficiently
    BRAINSTORM = "brainstorm"  # Creative exploration, ask probing questions
    SUPPORT = "support"  # Emotional support, validate feelings first
    CASUAL = "casual"  # Small talk, show personality
    LEARNING = "learning"  # Educational, explain thoroughly


@dataclass
class ConversationMode:
    """Tracks the current conversation mode and engagement state.

    Attributes:
        primary: The primary conversation mode
        secondary: Additional active modes
        confidence: Confidence in mode detection
        turns_in_mode: Number of turns in this mode
        exit_detected: Whether mode exit was detected
    """

    primary: ConversationModeType = ConversationModeType.TASK
    secondary: list[ConversationModeType] = field(default_factory=list)
    confidence: float = 1.0
    turns_in_mode: int = 0
    exit_detected: bool = False

    def reset(self) -> None:
        """Reset to default task mode."""
        self.primary = ConversationModeType.TASK
        self.secondary = []
        self.confidence = 1.0
        self.turns_in_mode = 0
        self.exit_detected = False

    def update(
        self,
        primary: ConversationModeType,
        secondary: list[ConversationModeType] | None = None,
        confidence: float = 0.8,
        exit_detected: bool = False,
    ) -> None:
        """Update the conversation mode.

        Args:
            primary: New primary mode
            secondary: New secondary modes
            confidence: Detection confidence
            exit_detected: Whether to exit current mode
        """
        if exit_detected:
            self.reset()
            return

        if primary == self.primary:
            self.turns_in_mode += 1
        else:
            self.primary = primary
            self.turns_in_mode = 1

        self.secondary = secondary or []
        self.confidence = confidence
        self.exit_detected = False

    def is_engaged(self) -> bool:
        """Check if we're in an engaged (non-task) conversation mode."""
        return self.primary != ConversationModeType.TASK

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "primary": self.primary.value,
            "secondary": [m.value for m in self.secondary],
            "confidence": self.confidence,
            "turns_in_mode": self.turns_in_mode,
            "exit_detected": self.exit_detected,
        }


@dataclass
class ToolCallInfo:
    """Information about a tool call.

    Attributes:
        tool: Name of the tool
        args: Arguments passed to the tool
        result: Result from the tool
        error: Error message if the tool failed
        elapsed_ms: Execution time in milliseconds
    """

    tool: str
    args: Any = None
    result: Any = None
    error: str | None = None
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCallInfo:
        """Create from dictionary."""
        return cls(
            tool=data.get("tool", "unknown"),
            args=data.get("args"),
            result=data.get("result"),
            error=data.get("error"),
            elapsed_ms=data.get("elapsed_ms", 0.0),
        )


@dataclass
class SentimentResult:
    """Result of sentiment analysis.

    Attributes:
        sentiment: Overall sentiment (positive, neutral, negative, frustrated)
        confidence: Confidence in the detection
        emotions: Detected emotions with intensities
        needs_support: Whether emotional support is needed
    """

    sentiment: str = "neutral"
    confidence: float = 0.8
    emotions: dict[str, float] = field(default_factory=dict)
    needs_support: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sentiment": self.sentiment,
            "confidence": self.confidence,
            "emotions": self.emotions,
            "needs_support": self.needs_support,
        }


class ConversationState:
    """Tracks conversation history and state.

    This class manages all state for a single conversation, including:
    - Message history with tool results
    - Pending actions (confirmations, passcode verification)
    - Undoable action tracking
    - Conversation mode
    - User sentiment

    Attributes:
        history: List of conversation turns
        max_history: Maximum turns to keep
        timeout_seconds: Seconds until conversation expires
        last_access: Timestamp of last access
        pending_command: Pending command awaiting confirmation
        pending_sensitive_op: Pending operation requiring authentication
        last_undoable_action: Most recent undoable action
        pending_details: Full response when condensed for voice
        current_area_id: Current room/area for context
        current_device_id: Device ID for targeted responses
        current_timezone: User's timezone
        mode: Current conversation mode
        sentiment: Detected user sentiment
        retrieved_memory_ids: Memory IDs to reinforce after response
    """

    def __init__(
        self,
        max_history: int = 10,
        timeout_seconds: int = 300,
    ):
        """Initialize conversation state.

        Args:
            max_history: Maximum number of turns to keep
            timeout_seconds: Seconds until conversation expires
        """
        self.history: list[dict[str, Any]] = []
        self.max_history = max_history
        self.timeout_seconds = timeout_seconds
        self.last_access = time.time()

        # Pending actions
        self.pending_command: dict[str, Any] | None = None
        self.pending_sensitive_op: dict[str, Any] | None = None
        self.last_undoable_action: Any | None = None  # UndoableAction

        # Response state
        self.pending_details: str | None = None

        # Context
        self.current_area_id: str | None = None
        self.current_device_id: str | None = None
        self.current_timezone: str | None = None
        self.current_identification: Any | None = None

        # Mode and sentiment
        self.mode: ConversationMode = ConversationMode()
        self.sentiment: SentimentResult | None = None

        # Memory tracking
        self.retrieved_memory_ids: list[str] = []

        # Metadata
        self.metadata: dict[str, Any] = {}

    def add_turn(
        self,
        user: str,
        assistant: str,
        tool_calls: list[ToolCallInfo | dict[str, Any]] | None = None,
    ) -> None:
        """Add a conversation turn with optional tool results.

        Tool results are stored so that later turns can reference details
        from previous tool calls (e.g., "add that event to my calendar"
        after a web search returned event details).

        Args:
            user: User's message
            assistant: Assistant's response
            tool_calls: List of tool calls made during this turn
        """
        turn: dict[str, Any] = {"user": user, "assistant": assistant}

        if tool_calls:
            # Store simplified tool results
            results = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    if tc.get("result") is not None:
                        results.append({
                            "tool": tc.get("tool", "unknown"),
                            "args": tc.get("args"),
                            "result": tc.get("result"),
                        })
                elif hasattr(tc, "result") and tc.result is not None:
                    results.append({
                        "tool": tc.tool,
                        "args": tc.args,
                        "result": tc.result,
                    })
            if results:
                turn["tool_results"] = results

        self.history.append(turn)

        # Trim to max history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self.last_access = time.time()

    def is_expired(self) -> bool:
        """Check if conversation has timed out."""
        return time.time() - self.last_access > self.timeout_seconds

    def touch(self) -> None:
        """Update last access time."""
        self.last_access = time.time()

    def get_messages(self) -> list[dict[str, str]]:
        """Get history as LLM messages format.

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        messages = []
        for turn in self.history:
            messages.append({"role": "user", "content": turn["user"]})
            messages.append({"role": "assistant", "content": turn["assistant"]})
        return messages

    def get_recent_tool_results(self, n: int = 3) -> list[dict[str, Any]]:
        """Get recent tool results for context.

        Args:
            n: Number of recent turns to check

        Returns:
            List of tool result dicts
        """
        results = []
        for turn in self.history[-n:]:
            if "tool_results" in turn:
                results.extend(turn["tool_results"])
        return results

    def clear(self) -> None:
        """Clear all state except configuration."""
        self.history = []
        self.pending_command = None
        self.pending_sensitive_op = None
        self.last_undoable_action = None
        self.pending_details = None
        self.mode.reset()
        self.sentiment = None
        self.retrieved_memory_ids = []
        self.metadata = {}
        self.last_access = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "history": self.history,
            "max_history": self.max_history,
            "timeout_seconds": self.timeout_seconds,
            "last_access": self.last_access,
            "pending_command": self.pending_command,
            "pending_details": self.pending_details,
            "current_area_id": self.current_area_id,
            "current_device_id": self.current_device_id,
            "current_timezone": self.current_timezone,
            "mode": self.mode.to_dict(),
            "sentiment": self.sentiment.to_dict() if self.sentiment else None,
            "retrieved_memory_ids": self.retrieved_memory_ids,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversationState:
        """Create from dictionary."""
        state = cls(
            max_history=data.get("max_history", 10),
            timeout_seconds=data.get("timeout_seconds", 300),
        )
        state.history = data.get("history", [])
        state.last_access = data.get("last_access", time.time())
        state.pending_command = data.get("pending_command")
        state.pending_details = data.get("pending_details")
        state.current_area_id = data.get("current_area_id")
        state.current_device_id = data.get("current_device_id")
        state.current_timezone = data.get("current_timezone")
        state.retrieved_memory_ids = data.get("retrieved_memory_ids", [])
        state.metadata = data.get("metadata", {})

        if data.get("mode"):
            mode_data = data["mode"]
            state.mode = ConversationMode(
                primary=ConversationModeType(mode_data.get("primary", "task")),
                secondary=[ConversationModeType(m) for m in mode_data.get("secondary", [])],
                confidence=mode_data.get("confidence", 1.0),
                turns_in_mode=mode_data.get("turns_in_mode", 0),
            )

        return state
