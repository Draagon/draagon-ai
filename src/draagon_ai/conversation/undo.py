"""Undo support for reversible actions.

This module provides data structures and utilities for tracking and undoing
actions that can be reversed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class UndoableAction:
    """Stores information needed to undo an action.

    Attributes:
        action_type: Type of action (calendar_create, ha_control, memory_store, etc.)
        timestamp: When the action was performed
        details: Action-specific details needed for undo
        user_id: User who performed the action
        expires_at: When this undo option expires (default: 5 minutes)
    """

    action_type: str
    details: dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    user_id: str | None = None
    expires_at: float | None = None

    def __post_init__(self):
        """Set expiration if not provided."""
        if self.expires_at is None:
            self.expires_at = self.timestamp + 300  # 5 minutes

    def is_expired(self) -> bool:
        """Check if the undo window has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def describe(self) -> str:
        """Get a human-readable description of what can be undone.

        Returns:
            Description of the undoable action
        """
        if self.action_type == "calendar_delete":
            event_name = self.details.get("summary", "the event")
            return f"deletion of '{event_name}' from your calendar"

        elif self.action_type == "calendar_create":
            event_name = self.details.get("summary", "the event")
            return f"creation of '{event_name}' on your calendar"

        elif self.action_type == "memory_store":
            content = self.details.get("content", "")[:50]
            return f"storing '{content}...'"

        elif self.action_type == "memory_delete":
            content = self.details.get("content", "")[:50]
            return f"deleting '{content}...'"

        elif self.action_type == "ha_control":
            entity = self.details.get("entity_id", "device")
            action = self.details.get("service", "change")
            return f"{action} on {entity}"

        elif self.action_type == "tool_execution":
            tool = self.details.get("tool", "unknown")
            return f"execution of {tool}"

        return f"the last {self.action_type} action"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "action_type": self.action_type,
            "details": self.details,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UndoableAction:
        """Create from dictionary."""
        return cls(
            action_type=data["action_type"],
            details=data.get("details", {}),
            timestamp=data.get("timestamp", time.time()),
            user_id=data.get("user_id"),
            expires_at=data.get("expires_at"),
        )


@dataclass
class UndoResult:
    """Result of an undo operation.

    Attributes:
        success: Whether the undo was successful
        message: Human-readable result message
        details: Additional details about the undo
        new_state: The new state after undo (if applicable)
    """

    success: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    new_state: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "details": self.details,
        }


# Common action type constants
class ActionTypes:
    """Constants for common undoable action types."""

    CALENDAR_CREATE = "calendar_create"
    CALENDAR_DELETE = "calendar_delete"
    CALENDAR_UPDATE = "calendar_update"
    MEMORY_STORE = "memory_store"
    MEMORY_DELETE = "memory_delete"
    MEMORY_UPDATE = "memory_update"
    HA_CONTROL = "ha_control"
    TOOL_EXECUTION = "tool_execution"
    SETTING_CHANGE = "setting_change"
