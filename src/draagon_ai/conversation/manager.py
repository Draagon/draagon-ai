"""Conversation manager for handling multiple concurrent conversations.

This module provides a manager class that handles creation, tracking, and
cleanup of multiple conversation states.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from .state import ConversationState

logger = logging.getLogger(__name__)


@dataclass
class ConversationManagerConfig:
    """Configuration for conversation manager.

    Attributes:
        max_history: Default max history per conversation
        timeout_seconds: Default timeout for conversations
        cleanup_interval: How often to check for expired conversations (seconds)
        max_concurrent: Maximum concurrent conversations (0 = unlimited)
    """

    max_history: int = 10
    timeout_seconds: int = 300
    cleanup_interval: int = 60
    max_concurrent: int = 0


@dataclass
class ExpiredConversation:
    """Information about an expired conversation.

    Attributes:
        conversation_id: The conversation ID
        state: The conversation state at expiration
        expired_at: Timestamp when it expired
    """

    conversation_id: str
    state: ConversationState
    expired_at: float = field(default_factory=time.time)

    @property
    def history(self) -> list[dict[str, Any]]:
        """Get the conversation history."""
        return self.state.history


class ConversationManager:
    """Manages multiple concurrent conversations.

    This class handles:
    - Creating and retrieving conversation states
    - Automatic expiration and cleanup
    - Episode summarization on expiration (optional)

    Usage:
        manager = ConversationManager()

        # Get or create a conversation
        state = manager.get_or_create("user123")

        # Use the state
        state.add_turn("Hello", "Hi there!")

        # Cleanup expired conversations
        expired = manager.cleanup_expired()
    """

    def __init__(
        self,
        config: ConversationManagerConfig | None = None,
        on_expire: Callable[[ExpiredConversation], Awaitable[None]] | None = None,
    ):
        """Initialize conversation manager.

        Args:
            config: Configuration options
            on_expire: Optional async callback when conversations expire
        """
        self.config = config or ConversationManagerConfig()
        self.on_expire = on_expire

        self._conversations: dict[str, ConversationState] = {}
        self._expired: list[ExpiredConversation] = []
        self._last_cleanup = time.time()

    def get(self, conversation_id: str) -> ConversationState | None:
        """Get a conversation by ID.

        Args:
            conversation_id: The conversation ID

        Returns:
            The conversation state, or None if not found or expired
        """
        state = self._conversations.get(conversation_id)

        if state is None:
            return None

        if state.is_expired():
            # Move to expired list
            self._expired.append(ExpiredConversation(
                conversation_id=conversation_id,
                state=state,
            ))
            del self._conversations[conversation_id]
            return None

        state.touch()
        return state

    def get_or_create(
        self,
        conversation_id: str,
        **kwargs: Any,
    ) -> ConversationState:
        """Get existing conversation or create new one.

        Args:
            conversation_id: The conversation ID
            **kwargs: Additional arguments for new ConversationState

        Returns:
            The conversation state
        """
        state = self.get(conversation_id)

        if state is None:
            state = ConversationState(
                max_history=kwargs.get("max_history", self.config.max_history),
                timeout_seconds=kwargs.get("timeout_seconds", self.config.timeout_seconds),
            )
            self._conversations[conversation_id] = state

        return state

    def create(
        self,
        conversation_id: str,
        **kwargs: Any,
    ) -> ConversationState:
        """Create a new conversation, replacing any existing one.

        Args:
            conversation_id: The conversation ID
            **kwargs: Additional arguments for ConversationState

        Returns:
            The new conversation state
        """
        # Expire existing conversation if present
        if conversation_id in self._conversations:
            old_state = self._conversations[conversation_id]
            self._expired.append(ExpiredConversation(
                conversation_id=conversation_id,
                state=old_state,
            ))

        state = ConversationState(
            max_history=kwargs.get("max_history", self.config.max_history),
            timeout_seconds=kwargs.get("timeout_seconds", self.config.timeout_seconds),
        )
        self._conversations[conversation_id] = state

        return state

    def delete(self, conversation_id: str) -> bool:
        """Delete a conversation.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if deleted, False if not found
        """
        if conversation_id in self._conversations:
            del self._conversations[conversation_id]
            return True
        return False

    def clear(self, conversation_id: str) -> bool:
        """Clear a conversation's history but keep the state.

        Args:
            conversation_id: The conversation ID

        Returns:
            True if cleared, False if not found
        """
        state = self._conversations.get(conversation_id)
        if state:
            state.clear()
            return True
        return False

    def cleanup_expired(self) -> list[ExpiredConversation]:
        """Check for and cleanup expired conversations.

        Returns:
            List of expired conversations that were cleaned up
        """
        now = time.time()

        # Only cleanup periodically
        if now - self._last_cleanup < self.config.cleanup_interval:
            return []

        self._last_cleanup = now
        expired = []

        # Find expired conversations
        to_remove = []
        for conv_id, state in self._conversations.items():
            if state.is_expired():
                expired.append(ExpiredConversation(
                    conversation_id=conv_id,
                    state=state,
                ))
                to_remove.append(conv_id)

        # Remove expired
        for conv_id in to_remove:
            del self._conversations[conv_id]

        # Add to expired list
        self._expired.extend(expired)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired conversations")

        return expired

    def get_expired(self, clear: bool = True) -> list[ExpiredConversation]:
        """Get list of expired conversations.

        Args:
            clear: Whether to clear the expired list after returning

        Returns:
            List of expired conversations
        """
        expired = list(self._expired)
        if clear:
            self._expired = []
        return expired

    def active_count(self) -> int:
        """Get count of active conversations."""
        return len(self._conversations)

    def list_active_ids(self) -> list[str]:
        """Get list of active conversation IDs."""
        return list(self._conversations.keys())

    def get_stats(self) -> dict[str, Any]:
        """Get manager statistics.

        Returns:
            Dict with stats about conversations
        """
        return {
            "active_conversations": len(self._conversations),
            "pending_expired": len(self._expired),
            "last_cleanup": self._last_cleanup,
            "config": {
                "max_history": self.config.max_history,
                "timeout_seconds": self.config.timeout_seconds,
                "cleanup_interval": self.config.cleanup_interval,
            },
        }
