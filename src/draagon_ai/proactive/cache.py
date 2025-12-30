"""Deduplication cache for suggestions.

This module provides TTL-based caching to prevent suggestion spam.
The same suggestion won't be shown to a user within the TTL window.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheEntry:
    """A single cache entry tracking when a suggestion was shown."""

    suggestion_id: str
    user_id: str
    shown_at: float
    expires_at: float


class DeduplicationCache:
    """TTL-based cache for deduplicating suggestions.

    Prevents the same suggestion from being shown to a user within
    a configurable time window.

    Usage:
        cache = DeduplicationCache(default_ttl=3600)  # 1 hour

        if cache.should_show(suggestion.id, user_id):
            # Show suggestion
            cache.mark_shown(suggestion.id, user_id)

    Attributes:
        default_ttl: Default time-to-live in seconds
        max_entries: Maximum cache entries (for memory management)
    """

    def __init__(
        self,
        default_ttl: float = 3600,  # 1 hour
        max_entries: int = 10000,
    ):
        """Initialize cache.

        Args:
            default_ttl: Default TTL in seconds for cache entries
            max_entries: Maximum number of entries before cleanup
        """
        self.default_ttl = default_ttl
        self.max_entries = max_entries
        self._cache: dict[str, CacheEntry] = {}
        self._last_cleanup = time.time()
        self._cleanup_interval = 300  # 5 minutes

    def _make_key(self, suggestion_id: str, user_id: str) -> str:
        """Create cache key from suggestion and user IDs."""
        return f"{user_id}:{suggestion_id}"

    def should_show(self, suggestion_id: str, user_id: str) -> bool:
        """Check if suggestion should be shown to user.

        Args:
            suggestion_id: Unique suggestion identifier
            user_id: User identifier

        Returns:
            True if suggestion should be shown (not recently shown)
        """
        self._maybe_cleanup()

        key = self._make_key(suggestion_id, user_id)
        entry = self._cache.get(key)

        if entry is None:
            return True

        # Check if expired
        if time.time() > entry.expires_at:
            del self._cache[key]
            return True

        return False

    def mark_shown(
        self,
        suggestion_id: str,
        user_id: str,
        ttl: float | None = None,
    ) -> None:
        """Mark suggestion as shown to user.

        Args:
            suggestion_id: Unique suggestion identifier
            user_id: User identifier
            ttl: Optional custom TTL (uses default if not provided)
        """
        key = self._make_key(suggestion_id, user_id)
        now = time.time()
        expires_at = now + (ttl if ttl is not None else self.default_ttl)

        self._cache[key] = CacheEntry(
            suggestion_id=suggestion_id,
            user_id=user_id,
            shown_at=now,
            expires_at=expires_at,
        )

    def clear_for_user(self, user_id: str) -> int:
        """Clear all cache entries for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of entries cleared
        """
        prefix = f"{user_id}:"
        to_remove = [k for k in self._cache if k.startswith(prefix)]

        for key in to_remove:
            del self._cache[key]

        return len(to_remove)

    def clear_all(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries cleared
        """
        count = len(self._cache)
        self._cache.clear()
        return count

    def _maybe_cleanup(self) -> None:
        """Periodically clean up expired entries."""
        now = time.time()

        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        self._cleanup_expired()

        # If still over limit, remove oldest entries
        if len(self._cache) > self.max_entries:
            self._evict_oldest(len(self._cache) - self.max_entries)

    def _cleanup_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries removed
        """
        now = time.time()
        expired = [k for k, v in self._cache.items() if now > v.expires_at]

        for key in expired:
            del self._cache[key]

        return len(expired)

    def _evict_oldest(self, count: int) -> None:
        """Evict oldest entries to free space.

        Args:
            count: Number of entries to evict
        """
        if count <= 0:
            return

        # Sort by shown_at and remove oldest
        sorted_entries = sorted(
            self._cache.items(),
            key=lambda x: x[1].shown_at,
        )

        for key, _ in sorted_entries[:count]:
            del self._cache[key]

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with cache stats
        """
        now = time.time()
        active = sum(1 for v in self._cache.values() if now <= v.expires_at)

        return {
            "total_entries": len(self._cache),
            "active_entries": active,
            "expired_entries": len(self._cache) - active,
            "max_entries": self.max_entries,
            "default_ttl": self.default_ttl,
        }

    def __len__(self) -> int:
        return len(self._cache)
