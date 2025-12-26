"""Learning Channel for cross-agent knowledge sharing.

The Learning Channel enables agents to share learnings with each other:
- When one agent learns something, others can benefit
- Learnings are scoped (not all agents see all learnings)
- Asynchronous pub/sub pattern

This is Phase C.1 - a stub implementation that logs but doesn't
actually distribute learnings. Full implementation in Phase C.4.

Based on research from:
- Multi-agent systems literature
- Pub/sub patterns for AI coordination
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable
import asyncio
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class LearningType(str, Enum):
    """Types of learnings that can be shared."""

    FACT = "fact"              # Learned factual information
    SKILL = "skill"            # Learned how to do something
    INSIGHT = "insight"        # Meta-learning about patterns
    PREFERENCE = "preference"  # User preference discovered
    CORRECTION = "correction"  # Corrected a previous belief
    BEHAVIOR = "behavior"      # New/improved behavior pattern


class LearningScope(str, Enum):
    """Scope at which a learning is shared.

    Determines which agents receive the learning.
    """

    PRIVATE = "private"      # Only the learning agent (no sharing)
    CONTEXT = "context"      # All agents in the same context
    GLOBAL = "global"        # All agents everywhere


@dataclass
class Learning:
    """A unit of knowledge to be shared.

    When an agent learns something, it creates a Learning object
    and broadcasts it through the channel.
    """

    # Identity
    learning_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Content
    learning_type: LearningType = LearningType.FACT
    content: str = ""
    entities: list[str] = field(default_factory=list)

    # Source
    source_agent_id: str = ""
    source_context_id: str | None = None

    # Scope
    scope: LearningScope = LearningScope.CONTEXT

    # Quality signals
    confidence: float = 1.0
    importance: float = 0.5
    verified: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_id": self.learning_id,
            "learning_type": self.learning_type.value,
            "content": self.content,
            "entities": self.entities,
            "source_agent_id": self.source_agent_id,
            "source_context_id": self.source_context_id,
            "scope": self.scope.value,
            "confidence": self.confidence,
            "importance": self.importance,
            "verified": self.verified,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Learning":
        """Create from dictionary."""
        return cls(
            learning_id=data.get("learning_id", str(uuid.uuid4())),
            learning_type=LearningType(data.get("learning_type", "fact")),
            content=data.get("content", ""),
            entities=data.get("entities", []),
            source_agent_id=data.get("source_agent_id", ""),
            source_context_id=data.get("source_context_id"),
            scope=LearningScope(data.get("scope", "context")),
            confidence=data.get("confidence", 1.0),
            importance=data.get("importance", 0.5),
            verified=data.get("verified", False),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
        )


# Type for learning handlers
LearningHandler = Callable[[Learning], Awaitable[None]]


@dataclass
class Subscription:
    """A subscription to learnings."""

    subscription_id: str
    agent_id: str
    context_id: str | None
    learning_types: set[LearningType] | None  # None = all types
    handler: LearningHandler
    created_at: datetime = field(default_factory=datetime.now)


class LearningChannel(ABC):
    """Abstract base class for learning channels.

    Subclasses implement the actual distribution mechanism.
    """

    @abstractmethod
    async def broadcast(self, learning: Learning) -> None:
        """Broadcast a learning to subscribers.

        Args:
            learning: The learning to share
        """
        ...

    @abstractmethod
    async def subscribe(
        self,
        agent_id: str,
        handler: LearningHandler,
        context_id: str | None = None,
        learning_types: set[LearningType] | None = None,
    ) -> str:
        """Subscribe to learnings.

        Args:
            agent_id: Subscribing agent
            handler: Async function to call with learnings
            context_id: Optional context filter
            learning_types: Optional type filter

        Returns:
            Subscription ID
        """
        ...

    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from learnings.

        Args:
            subscription_id: Subscription to cancel

        Returns:
            True if subscription existed
        """
        ...


class StubLearningChannel(LearningChannel):
    """Stub implementation of the learning channel for Phase C.1.

    This implementation:
    - Logs all broadcasts
    - Stores learnings in memory
    - Does NOT actually distribute to subscribers
    - Useful for testing and development

    The full implementation in Phase C.4 will use Redis pub/sub
    or similar for actual distribution.
    """

    def __init__(self):
        """Initialize the stub channel."""
        self._subscriptions: dict[str, Subscription] = {}
        self._learning_log: list[Learning] = []
        self._max_log_size = 1000

    async def broadcast(self, learning: Learning) -> None:
        """Log the learning (stub - doesn't actually distribute).

        Args:
            learning: The learning to share
        """
        logger.info(
            f"[STUB] Learning broadcast: type={learning.learning_type.value}, "
            f"content='{learning.content[:50]}...', "
            f"scope={learning.scope.value}, "
            f"from={learning.source_agent_id}"
        )

        # Store in log
        self._learning_log.append(learning)

        # Trim log if too large
        if len(self._learning_log) > self._max_log_size:
            self._learning_log = self._learning_log[-self._max_log_size // 2:]

        # NOTE: In Phase C.4, we would:
        # 1. Find matching subscriptions
        # 2. Call each subscriber's handler
        # 3. Handle errors gracefully
        # 4. Maybe use Redis pub/sub for cross-process

    async def subscribe(
        self,
        agent_id: str,
        handler: LearningHandler,
        context_id: str | None = None,
        learning_types: set[LearningType] | None = None,
    ) -> str:
        """Register a subscription (stub - handlers won't be called).

        Args:
            agent_id: Subscribing agent
            handler: Handler function
            context_id: Optional context filter
            learning_types: Optional type filter

        Returns:
            Subscription ID
        """
        sub_id = str(uuid.uuid4())

        self._subscriptions[sub_id] = Subscription(
            subscription_id=sub_id,
            agent_id=agent_id,
            context_id=context_id,
            learning_types=learning_types,
            handler=handler,
        )

        logger.info(
            f"[STUB] Subscription created: id={sub_id}, "
            f"agent={agent_id}, context={context_id}"
        )

        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription.

        Args:
            subscription_id: Subscription to remove

        Returns:
            True if existed
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            logger.info(f"[STUB] Subscription removed: id={subscription_id}")
            return True
        return False

    def get_learning_log(self) -> list[Learning]:
        """Get the log of all broadcasts (for testing).

        Returns:
            List of all learnings broadcast
        """
        return self._learning_log.copy()

    def get_subscriptions(self) -> list[Subscription]:
        """Get all active subscriptions (for testing).

        Returns:
            List of subscriptions
        """
        return list(self._subscriptions.values())

    def clear(self) -> None:
        """Clear all state (for testing)."""
        self._subscriptions.clear()
        self._learning_log.clear()


class InMemoryLearningChannel(LearningChannel):
    """In-memory implementation that actually delivers learnings.

    This is a step up from the stub - it actually calls handlers,
    but only works within a single process. Useful for testing
    multi-agent scenarios.

    For production, use a distributed implementation (Redis, etc.).
    """

    def __init__(self):
        """Initialize the channel."""
        self._subscriptions: dict[str, Subscription] = {}
        self._learning_log: list[Learning] = []
        self._max_log_size = 1000

    async def broadcast(self, learning: Learning) -> None:
        """Broadcast to matching subscribers.

        Handlers are executed concurrently using asyncio.gather() to prevent
        slow handlers from blocking other subscribers.

        Args:
            learning: The learning to share
        """
        logger.debug(
            f"Broadcasting learning: type={learning.learning_type.value}, "
            f"scope={learning.scope.value}"
        )

        # Log the learning
        self._learning_log.append(learning)
        if len(self._learning_log) > self._max_log_size:
            self._learning_log = self._learning_log[-self._max_log_size // 2:]

        # Find matching subscriptions and create handler tasks
        async def safe_handler(sub: Subscription) -> None:
            """Wrapper to catch and log handler exceptions."""
            try:
                await sub.handler(learning)
            except Exception as e:
                logger.error(
                    f"Error in learning handler for agent {sub.agent_id}: {e}"
                )

        # Collect matching handlers
        tasks = [
            safe_handler(sub)
            for sub in self._subscriptions.values()
            if self._matches(learning, sub)
        ]

        # Run all handlers concurrently
        if tasks:
            await asyncio.gather(*tasks)

    def _matches(self, learning: Learning, sub: Subscription) -> bool:
        """Check if a learning matches a subscription."""
        # Don't send to source agent
        if learning.source_agent_id == sub.agent_id:
            return False

        # Check scope
        if learning.scope == LearningScope.PRIVATE:
            return False

        if learning.scope == LearningScope.CONTEXT:
            if sub.context_id and learning.source_context_id != sub.context_id:
                return False

        # Check type filter
        if sub.learning_types and learning.learning_type not in sub.learning_types:
            return False

        return True

    async def subscribe(
        self,
        agent_id: str,
        handler: LearningHandler,
        context_id: str | None = None,
        learning_types: set[LearningType] | None = None,
    ) -> str:
        """Register a subscription.

        Args:
            agent_id: Subscribing agent
            handler: Handler function
            context_id: Optional context filter
            learning_types: Optional type filter

        Returns:
            Subscription ID
        """
        sub_id = str(uuid.uuid4())

        self._subscriptions[sub_id] = Subscription(
            subscription_id=sub_id,
            agent_id=agent_id,
            context_id=context_id,
            learning_types=learning_types,
            handler=handler,
        )

        logger.info(f"Subscription created: agent={agent_id}, id={sub_id}")
        return sub_id

    async def unsubscribe(self, subscription_id: str) -> bool:
        """Remove a subscription."""
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            return True
        return False

    def get_learning_log(self) -> list[Learning]:
        """Get learning log (for testing)."""
        return self._learning_log.copy()

    def clear(self) -> None:
        """Clear all state."""
        self._subscriptions.clear()
        self._learning_log.clear()


# Factory function
def create_learning_channel(channel_type: str = "stub") -> LearningChannel:
    """Create a learning channel.

    Args:
        channel_type: Type of channel ("stub", "memory")

    Returns:
        Learning channel instance
    """
    if channel_type == "stub":
        return StubLearningChannel()
    elif channel_type == "memory":
        return InMemoryLearningChannel()
    else:
        raise ValueError(f"Unknown channel type: {channel_type}")


# Singleton for easy access with thread-safe initialization
_default_channel: LearningChannel | None = None
_default_channel_lock = threading.Lock()


def get_learning_channel() -> LearningChannel:
    """Get the default learning channel (singleton).

    Thread-safe: Uses double-checked locking to ensure only one
    instance is created even when called from multiple threads.
    """
    global _default_channel

    # Fast path: channel already exists
    if _default_channel is not None:
        return _default_channel

    # Slow path: need to create channel
    with _default_channel_lock:
        # Double-check after acquiring lock
        if _default_channel is None:
            _default_channel = StubLearningChannel()
        return _default_channel


def set_learning_channel(channel: LearningChannel) -> None:
    """Set the default learning channel.

    Thread-safe: Acquires lock before modifying singleton.
    """
    global _default_channel
    with _default_channel_lock:
        _default_channel = channel


def reset_learning_channel() -> None:
    """Reset the learning channel (for testing).

    Thread-safe: Acquires lock before modifying singleton.
    """
    global _default_channel
    with _default_channel_lock:
        _default_channel = None
