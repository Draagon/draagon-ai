"""Proactive suggestions framework.

This module provides a generic framework for generating proactive suggestions
from multiple providers, with deduplication and priority management.
"""

from .models import (
    Suggestion,
    SuggestionPriority,
    SuggestionCategory,
    AggregatedSuggestions,
)
from .provider import SuggestionProvider, ProviderContext
from .cache import DeduplicationCache
from .engine import ProactiveEngine, ProactiveEngineConfig

__all__ = [
    # Models
    "Suggestion",
    "SuggestionPriority",
    "SuggestionCategory",
    "AggregatedSuggestions",
    # Provider
    "SuggestionProvider",
    "ProviderContext",
    # Cache
    "DeduplicationCache",
    # Engine
    "ProactiveEngine",
    "ProactiveEngineConfig",
]
