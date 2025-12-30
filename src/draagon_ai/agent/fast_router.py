"""Fast-path routing for common queries.

This module provides a fast-path routing mechanism that bypasses
expensive context gathering and LLM decision making for common,
unambiguous queries like greetings, time, and simple commands.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Protocol

from .models import FastRouteResult, ToolCallInfo

logger = logging.getLogger(__name__)


class FastRouteHandler(ABC):
    """Abstract base class for fast route handlers.

    Implement this to create custom fast-path handlers for specific
    types of queries.

    Example:
        class TimeHandler(FastRouteHandler):
            @property
            def route_type(self) -> str:
                return "time"

            def can_handle(self, query: str, context: dict) -> bool:
                time_words = {"time", "date", "day", "today"}
                return any(w in query.lower() for w in time_words)

            async def handle(self, query: str, context: dict) -> FastRouteResult:
                current_time = datetime.now().strftime("%I:%M %p")
                return FastRouteResult(
                    response=f"It's {current_time}.",
                    route_type="time",
                )
    """

    @property
    @abstractmethod
    def route_type(self) -> str:
        """Unique identifier for this route type."""
        pass

    @property
    def priority(self) -> int:
        """Priority for route matching (higher = checked first)."""
        return 0

    @abstractmethod
    def can_handle(self, query: str, context: dict[str, Any]) -> bool:
        """Check if this handler can process the query.

        Args:
            query: User's query text
            context: Additional context (user_id, area_id, etc.)

        Returns:
            True if this handler should process the query
        """
        pass

    @abstractmethod
    async def handle(
        self,
        query: str,
        context: dict[str, Any],
    ) -> FastRouteResult:
        """Process the query and return a response.

        Args:
            query: User's query text
            context: Additional context

        Returns:
            FastRouteResult with response and metadata
        """
        pass


# Words that indicate context is required (skip fast routing)
CONTEXT_REQUIRED_WORDS = {
    "previous", "before", "earlier", "last", "again",
    "that", "those", "it", "them", "they",
    "mentioned", "said", "told", "discussed",
    "more", "else", "also", "another",
    "why", "how come", "explain",
}


@dataclass
class FastRouterConfig:
    """Configuration for the fast router.

    Attributes:
        enabled: Whether fast routing is enabled
        context_required_words: Words that indicate context is needed
        max_query_length: Max query length for fast routing
    """

    enabled: bool = True
    context_required_words: set[str] = field(
        default_factory=lambda: CONTEXT_REQUIRED_WORDS.copy()
    )
    max_query_length: int = 200


class FastRouter:
    """Routes common queries through fast-path handlers.

    The fast router skips expensive operations (context gathering,
    decision LLM) for simple, unambiguous queries like:
    - Greetings ("hello", "good morning")
    - Time/date queries ("what time is it?")
    - Simple commands ("turn off the lights")
    - Undo requests ("undo that")

    Usage:
        router = FastRouter()
        router.register_handler(GreetingHandler())
        router.register_handler(TimeHandler())

        result = await router.try_route(query, context)
        if result:
            return result.to_agent_response()
        # Fall through to full processing
    """

    def __init__(self, config: FastRouterConfig | None = None):
        """Initialize the fast router.

        Args:
            config: Configuration options
        """
        self.config = config or FastRouterConfig()
        self._handlers: list[FastRouteHandler] = []

    def register_handler(self, handler: FastRouteHandler) -> None:
        """Register a fast route handler.

        Args:
            handler: Handler to register
        """
        self._handlers.append(handler)
        # Keep sorted by priority (highest first)
        self._handlers.sort(key=lambda h: h.priority, reverse=True)
        logger.debug(f"Registered fast route handler: {handler.route_type}")

    def unregister_handler(self, route_type: str) -> bool:
        """Unregister a handler by route type.

        Args:
            route_type: Route type to unregister

        Returns:
            True if handler was removed
        """
        before = len(self._handlers)
        self._handlers = [h for h in self._handlers if h.route_type != route_type]
        return len(self._handlers) < before

    def list_handlers(self) -> list[str]:
        """List all registered handler route types."""
        return [h.route_type for h in self._handlers]

    def _requires_context(self, query: str) -> bool:
        """Check if query requires conversation context.

        Queries with references to previous conversation or ambiguous
        pronouns should not be fast-routed.

        Args:
            query: User's query text

        Returns:
            True if context is required
        """
        query_lower = query.lower()

        # Check for context-required words
        for word in self.config.context_required_words:
            if word in query_lower:
                return True

        # Long queries usually need context
        if len(query) > self.config.max_query_length:
            return True

        return False

    async def try_route(
        self,
        query: str,
        context: dict[str, Any],
    ) -> FastRouteResult | None:
        """Try to route query through fast-path handlers.

        Args:
            query: User's query text
            context: Additional context (user_id, area_id, conversation, etc.)

        Returns:
            FastRouteResult if routed, None if should use full processing
        """
        if not self.config.enabled:
            return None

        # Skip if query requires context
        if self._requires_context(query):
            logger.debug(f"Query requires context, skipping fast route: {query[:50]}")
            return None

        # Try handlers in priority order
        for handler in self._handlers:
            try:
                if handler.can_handle(query, context):
                    logger.debug(f"Fast routing via {handler.route_type}: {query[:50]}")
                    return await handler.handle(query, context)
            except Exception as e:
                logger.error(f"Fast route handler {handler.route_type} failed: {e}")
                # Continue to next handler

        return None

    def get_stats(self) -> dict[str, Any]:
        """Get router statistics.

        Returns:
            Dict with router stats
        """
        return {
            "enabled": self.config.enabled,
            "handler_count": len(self._handlers),
            "handlers": [
                {"route_type": h.route_type, "priority": h.priority}
                for h in self._handlers
            ],
        }


# Example built-in handlers that can be used or extended

class GreetingHandler(FastRouteHandler):
    """Handler for greeting queries."""

    GREETINGS = {
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "howdy", "greetings", "yo", "sup",
    }

    @property
    def route_type(self) -> str:
        return "greeting"

    @property
    def priority(self) -> int:
        return 100  # High priority

    def can_handle(self, query: str, context: dict) -> bool:
        query_lower = query.lower().strip()
        # Check if query is just a greeting
        for greeting in self.GREETINGS:
            if query_lower == greeting or query_lower.startswith(f"{greeting} "):
                return True
        return False

    async def handle(self, query: str, context: dict) -> FastRouteResult:
        # Simple greeting responses
        responses = [
            "Hello! How can I help you?",
            "Hi there! What can I do for you?",
            "Hey! What's on your mind?",
        ]
        import random
        return FastRouteResult(
            response=random.choice(responses),
            route_type="greeting",
        )


class UndoHandler(FastRouteHandler):
    """Handler for undo requests.

    Requires conversation state with last_undoable_action in context.
    """

    UNDO_PHRASES = {"undo", "undo that", "reverse that", "take that back", "cancel that"}

    @property
    def route_type(self) -> str:
        return "undo"

    @property
    def priority(self) -> int:
        return 90  # High priority

    def can_handle(self, query: str, context: dict) -> bool:
        query_lower = query.lower().strip()
        return query_lower in self.UNDO_PHRASES

    async def handle(self, query: str, context: dict) -> FastRouteResult:
        # Get undoable action from context
        conversation = context.get("conversation")
        if not conversation:
            return FastRouteResult(
                response="Nothing to undo.",
                route_type="undo",
            )

        undoable_action = getattr(conversation, "last_undoable_action", None)
        if not undoable_action:
            return FastRouteResult(
                response="Nothing to undo.",
                route_type="undo",
            )

        # Check if expired (default 5 min)
        if hasattr(undoable_action, "is_expired") and undoable_action.is_expired():
            return FastRouteResult(
                response="That action is too old to undo.",
                route_type="undo",
            )

        # Perform undo - this should be implemented by the application
        undo_handler = context.get("undo_handler")
        if undo_handler:
            result = await undo_handler(undoable_action)
            return FastRouteResult(
                response=result.get("message", "Undone."),
                route_type="undo",
                tool_calls=result.get("tool_calls", []),
            )

        return FastRouteResult(
            response="Undo is not configured.",
            route_type="undo",
        )


class ClearHistoryHandler(FastRouteHandler):
    """Handler for clearing conversation history."""

    CLEAR_PHRASES = {
        "start fresh", "start over", "new conversation",
        "new topic", "clear history", "fresh start",
        "reset conversation", "clear context",
    }

    @property
    def route_type(self) -> str:
        return "clear_history"

    @property
    def priority(self) -> int:
        return 85

    def can_handle(self, query: str, context: dict) -> bool:
        query_lower = query.lower().strip()
        return query_lower in self.CLEAR_PHRASES

    async def handle(self, query: str, context: dict) -> FastRouteResult:
        # Clear conversation state
        conversation = context.get("conversation")
        if conversation and hasattr(conversation, "clear"):
            conversation.clear()

        return FastRouteResult(
            response="Starting fresh! How can I help you?",
            route_type="clear_history",
        )
