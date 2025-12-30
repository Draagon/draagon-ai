"""Models for proactive suggestions.

This module defines the core data structures for the proactive suggestions
framework, including suggestions, priorities, and categories.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SuggestionPriority(str, Enum):
    """Priority levels for suggestions.

    Higher priority suggestions should be delivered first and may
    interrupt the user more aggressively.
    """

    URGENT = "urgent"      # Immediate attention (e.g., security alert)
    HIGH = "high"          # Soon (e.g., meeting in 30 min)
    MEDIUM = "medium"      # Good to know (e.g., meeting in 2 hours)
    LOW = "low"            # Nice to have (e.g., routine reminder)

    @property
    def weight(self) -> int:
        """Get numeric weight for sorting (higher = more urgent)."""
        weights = {
            SuggestionPriority.URGENT: 100,
            SuggestionPriority.HIGH: 75,
            SuggestionPriority.MEDIUM: 50,
            SuggestionPriority.LOW: 25,
        }
        return weights.get(self, 0)


class SuggestionCategory(str, Enum):
    """Categories for suggestions.

    Categories help group related suggestions and enable
    category-specific handling or filtering.
    """

    CALENDAR = "calendar"       # Schedule-related
    REMINDER = "reminder"       # User-set reminders
    MEMORY = "memory"           # Memory-based (birthdays, etc.)
    SECURITY = "security"       # Security alerts
    HOME = "home"               # Smart home state
    WEATHER = "weather"         # Weather alerts
    ROUTINE = "routine"         # Daily routines
    CONTEXT = "context"         # Context-aware suggestions
    SYSTEM = "system"           # System notifications
    CUSTOM = "custom"           # Custom/plugin suggestions


@dataclass
class Suggestion:
    """A single proactive suggestion.

    Attributes:
        id: Unique identifier for deduplication
        message: Human-readable suggestion text
        priority: Urgency level
        category: Suggestion category
        source: Provider that generated this suggestion
        created_at: Timestamp when suggestion was created
        expires_at: Optional expiration timestamp
        metadata: Additional context-specific data
        action: Optional action to execute if user accepts
        actionable: Whether user can act on this suggestion
    """

    id: str
    message: str
    priority: SuggestionPriority = SuggestionPriority.MEDIUM
    category: SuggestionCategory = SuggestionCategory.CUSTOM
    source: str = "unknown"
    created_at: float = field(default_factory=time.time)
    expires_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    action: dict[str, Any] | None = None
    actionable: bool = False

    def is_expired(self) -> bool:
        """Check if suggestion has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "message": self.message,
            "priority": self.priority.value,
            "category": self.category.value,
            "source": self.source,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "metadata": self.metadata,
            "action": self.action,
            "actionable": self.actionable,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Suggestion:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            message=data["message"],
            priority=SuggestionPriority(data.get("priority", "medium")),
            category=SuggestionCategory(data.get("category", "custom")),
            source=data.get("source", "unknown"),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            metadata=data.get("metadata", {}),
            action=data.get("action"),
            actionable=data.get("actionable", False),
        )


@dataclass
class AggregatedSuggestions:
    """Collection of suggestions from multiple providers.

    Attributes:
        suggestions: List of suggestions sorted by priority
        by_category: Suggestions grouped by category
        by_priority: Suggestions grouped by priority
        generated_at: When aggregation was performed
        provider_stats: Stats about each provider's contribution
    """

    suggestions: list[Suggestion] = field(default_factory=list)
    by_category: dict[SuggestionCategory, list[Suggestion]] = field(default_factory=dict)
    by_priority: dict[SuggestionPriority, list[Suggestion]] = field(default_factory=dict)
    generated_at: float = field(default_factory=time.time)
    provider_stats: dict[str, int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        """Total number of suggestions."""
        return len(self.suggestions)

    @property
    def has_urgent(self) -> bool:
        """Check if there are urgent suggestions."""
        return SuggestionPriority.URGENT in self.by_priority and len(self.by_priority[SuggestionPriority.URGENT]) > 0

    def get_top(self, n: int = 3) -> list[Suggestion]:
        """Get top N suggestions by priority."""
        return self.suggestions[:n]

    def filter_by_category(self, category: SuggestionCategory) -> list[Suggestion]:
        """Get suggestions for a specific category."""
        return self.by_category.get(category, [])

    def filter_by_priority(self, min_priority: SuggestionPriority) -> list[Suggestion]:
        """Get suggestions at or above a priority level."""
        min_weight = min_priority.weight
        return [s for s in self.suggestions if s.priority.weight >= min_weight]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "count": self.count,
            "has_urgent": self.has_urgent,
            "generated_at": self.generated_at,
            "provider_stats": self.provider_stats,
        }
