"""Proactive engine for aggregating suggestions from providers.

This module provides the main orchestrator that collects suggestions
from all registered providers, deduplicates them, and returns a
prioritized list.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .cache import DeduplicationCache
from .models import (
    AggregatedSuggestions,
    Suggestion,
    SuggestionCategory,
    SuggestionPriority,
)
from .provider import ProviderContext, SuggestionProvider

logger = logging.getLogger(__name__)


@dataclass
class ProactiveEngineConfig:
    """Configuration for the proactive engine.

    Attributes:
        cache_ttl: TTL for deduplication cache (seconds)
        provider_timeout: Timeout for each provider (seconds)
        max_suggestions: Maximum suggestions to return
        parallel_providers: Whether to run providers in parallel
    """

    cache_ttl: float = 3600  # 1 hour
    provider_timeout: float = 5.0  # 5 seconds
    max_suggestions: int = 10
    parallel_providers: bool = True


class ProactiveEngine:
    """Orchestrates suggestion generation from multiple providers.

    The engine collects suggestions from all registered providers,
    filters out duplicates and recently-shown suggestions, and
    returns a prioritized list.

    Usage:
        engine = ProactiveEngine()

        # Register providers
        engine.register_provider(CalendarProvider())
        engine.register_provider(SecurityProvider())

        # Generate suggestions
        context = ProviderContext(user_id="user123")
        suggestions = await engine.get_suggestions(context)

        # Access results
        for s in suggestions.get_top(3):
            print(f"[{s.priority}] {s.message}")
    """

    def __init__(self, config: ProactiveEngineConfig | None = None):
        """Initialize engine.

        Args:
            config: Configuration options
        """
        self.config = config or ProactiveEngineConfig()
        self._providers: dict[str, SuggestionProvider] = {}
        self._cache = DeduplicationCache(default_ttl=self.config.cache_ttl)
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize all registered providers."""
        if self._initialized:
            return

        for provider in self._providers.values():
            try:
                await provider.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize provider {provider.name}: {e}")

        self._initialized = True

    async def shutdown(self) -> None:
        """Shut down all providers."""
        for provider in self._providers.values():
            try:
                await provider.shutdown()
            except Exception as e:
                logger.error(f"Failed to shutdown provider {provider.name}: {e}")

        self._initialized = False

    def register_provider(self, provider: SuggestionProvider) -> None:
        """Register a suggestion provider.

        Args:
            provider: Provider instance to register

        Raises:
            ValueError: If provider with same name already registered
        """
        if provider.name in self._providers:
            raise ValueError(f"Provider '{provider.name}' already registered")

        self._providers[provider.name] = provider
        logger.debug(f"Registered provider: {provider.name}")

    def unregister_provider(self, name: str) -> bool:
        """Unregister a provider by name.

        Args:
            name: Provider name to unregister

        Returns:
            True if provider was removed, False if not found
        """
        if name in self._providers:
            del self._providers[name]
            return True
        return False

    def get_provider(self, name: str) -> SuggestionProvider | None:
        """Get a registered provider by name."""
        return self._providers.get(name)

    def list_providers(self) -> list[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    async def get_suggestions(
        self,
        context: ProviderContext,
        categories: list[SuggestionCategory] | None = None,
        min_priority: SuggestionPriority | None = None,
        include_shown: bool = False,
    ) -> AggregatedSuggestions:
        """Get suggestions from all providers.

        Args:
            context: Current context for suggestion generation
            categories: Optional filter by categories
            min_priority: Optional minimum priority filter
            include_shown: If True, include recently shown suggestions

        Returns:
            Aggregated suggestions from all providers
        """
        if not self._initialized:
            await self.initialize()

        # Collect from all providers
        all_suggestions: list[Suggestion] = []
        provider_stats: dict[str, int] = {}

        if self.config.parallel_providers:
            results = await self._collect_parallel(context, categories)
        else:
            results = await self._collect_sequential(context, categories)

        for provider_name, suggestions in results.items():
            provider_stats[provider_name] = len(suggestions)
            all_suggestions.extend(suggestions)

        # Filter and deduplicate
        filtered = self._filter_suggestions(
            all_suggestions,
            context.user_id,
            min_priority=min_priority,
            include_shown=include_shown,
        )

        # Sort by priority
        sorted_suggestions = sorted(
            filtered,
            key=lambda s: (s.priority.weight, -s.created_at),
            reverse=True,
        )

        # Limit results
        limited = sorted_suggestions[:self.config.max_suggestions]

        # Mark as shown (unless include_shown is True)
        if not include_shown:
            for s in limited:
                self._cache.mark_shown(s.id, context.user_id)

        # Build aggregated result
        return self._build_aggregated(limited, provider_stats)

    async def _collect_parallel(
        self,
        context: ProviderContext,
        categories: list[SuggestionCategory] | None,
    ) -> dict[str, list[Suggestion]]:
        """Collect suggestions from providers in parallel."""
        results: dict[str, list[Suggestion]] = {}
        tasks = []
        provider_names = []

        for name, provider in self._providers.items():
            if not provider.enabled:
                continue

            if categories and not any(c in categories for c in provider.categories):
                continue

            tasks.append(self._call_provider(provider, context))
            provider_names.append(name)

        if not tasks:
            return results

        # Run with timeout
        try:
            completed = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=self.config.provider_timeout * 2,  # Extra buffer for all
            )
        except asyncio.TimeoutError:
            logger.warning("Provider collection timed out")
            return results

        for name, result in zip(provider_names, completed):
            if isinstance(result, Exception):
                logger.error(f"Provider {name} failed: {result}")
                results[name] = []
            else:
                results[name] = result

        return results

    async def _collect_sequential(
        self,
        context: ProviderContext,
        categories: list[SuggestionCategory] | None,
    ) -> dict[str, list[Suggestion]]:
        """Collect suggestions from providers sequentially."""
        results: dict[str, list[Suggestion]] = {}

        for name, provider in self._providers.items():
            if not provider.enabled:
                continue

            if categories and not any(c in categories for c in provider.categories):
                continue

            try:
                suggestions = await self._call_provider(provider, context)
                results[name] = suggestions
            except Exception as e:
                logger.error(f"Provider {name} failed: {e}")
                results[name] = []

        return results

    async def _call_provider(
        self,
        provider: SuggestionProvider,
        context: ProviderContext,
    ) -> list[Suggestion]:
        """Call a single provider with timeout."""
        try:
            return await asyncio.wait_for(
                provider.get_suggestions(context),
                timeout=self.config.provider_timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(f"Provider {provider.name} timed out")
            return []

    def _filter_suggestions(
        self,
        suggestions: list[Suggestion],
        user_id: str,
        min_priority: SuggestionPriority | None = None,
        include_shown: bool = False,
    ) -> list[Suggestion]:
        """Filter suggestions based on criteria."""
        filtered = []

        for s in suggestions:
            # Skip expired
            if s.is_expired():
                continue

            # Skip below minimum priority
            if min_priority and s.priority.weight < min_priority.weight:
                continue

            # Skip recently shown (unless include_shown)
            if not include_shown and not self._cache.should_show(s.id, user_id):
                continue

            filtered.append(s)

        return filtered

    def _build_aggregated(
        self,
        suggestions: list[Suggestion],
        provider_stats: dict[str, int],
    ) -> AggregatedSuggestions:
        """Build aggregated suggestions result."""
        by_category: dict[SuggestionCategory, list[Suggestion]] = {}
        by_priority: dict[SuggestionPriority, list[Suggestion]] = {}

        for s in suggestions:
            by_category.setdefault(s.category, []).append(s)
            by_priority.setdefault(s.priority, []).append(s)

        return AggregatedSuggestions(
            suggestions=suggestions,
            by_category=by_category,
            by_priority=by_priority,
            generated_at=time.time(),
            provider_stats=provider_stats,
        )

    def clear_cache(self, user_id: str | None = None) -> int:
        """Clear deduplication cache.

        Args:
            user_id: If provided, only clear for this user

        Returns:
            Number of entries cleared
        """
        if user_id:
            return self._cache.clear_for_user(user_id)
        return self._cache.clear_all()

    def get_stats(self) -> dict[str, Any]:
        """Get engine statistics.

        Returns:
            Dict with engine stats
        """
        return {
            "providers": {
                name: {
                    "enabled": p.enabled,
                    "categories": [c.value for c in p.categories],
                }
                for name, p in self._providers.items()
            },
            "cache": self._cache.get_stats(),
            "config": {
                "cache_ttl": self.config.cache_ttl,
                "provider_timeout": self.config.provider_timeout,
                "max_suggestions": self.config.max_suggestions,
                "parallel_providers": self.config.parallel_providers,
            },
            "initialized": self._initialized,
        }
