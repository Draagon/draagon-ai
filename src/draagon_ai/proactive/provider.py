"""Abstract base class for suggestion providers.

This module defines the interface that all suggestion providers must implement.
Providers are the pluggable sources of proactive suggestions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from .models import Suggestion, SuggestionCategory


@dataclass
class ProviderContext:
    """Context passed to providers when generating suggestions.

    Attributes:
        user_id: Current user identifier
        area_id: Current physical area/room (if known)
        device_id: Current device identifier (if known)
        timezone: User's timezone
        current_time: Current timestamp
        metadata: Additional context-specific data
    """

    user_id: str
    area_id: str | None = None
    device_id: str | None = None
    timezone: str | None = None
    current_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class SuggestionProvider(ABC):
    """Abstract base class for suggestion providers.

    Providers are pluggable sources of proactive suggestions. Each provider
    focuses on a specific domain (calendar, security, memory, etc.) and
    generates suggestions based on current context.

    To create a custom provider:
    1. Subclass SuggestionProvider
    2. Implement get_suggestions() to return domain-specific suggestions
    3. Register with ProactiveEngine

    Example:
        class CalendarProvider(SuggestionProvider):
            @property
            def name(self) -> str:
                return "calendar"

            @property
            def categories(self) -> list[SuggestionCategory]:
                return [SuggestionCategory.CALENDAR]

            async def get_suggestions(self, context: ProviderContext) -> list[Suggestion]:
                # Check upcoming events, generate reminders
                events = await self.calendar.get_upcoming(context.user_id)
                return [self._event_to_suggestion(e) for e in events]
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this provider."""
        pass

    @property
    def categories(self) -> list[SuggestionCategory]:
        """Categories this provider can generate suggestions for.

        Override to specify which categories this provider handles.
        Default returns CUSTOM.
        """
        return [SuggestionCategory.CUSTOM]

    @property
    def enabled(self) -> bool:
        """Whether this provider is currently enabled.

        Override to implement dynamic enable/disable logic.
        """
        return True

    @abstractmethod
    async def get_suggestions(self, context: ProviderContext) -> list[Suggestion]:
        """Generate suggestions based on current context.

        Args:
            context: Current context including user, location, time

        Returns:
            List of suggestions from this provider
        """
        pass

    async def initialize(self) -> None:
        """Initialize provider resources.

        Override to perform async initialization (connect to services, etc.).
        Called once when provider is registered.
        """
        pass

    async def shutdown(self) -> None:
        """Clean up provider resources.

        Override to perform cleanup when engine shuts down.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
