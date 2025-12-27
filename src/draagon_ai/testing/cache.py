"""
Unified Cache Manager for Test Optimization.

Provides multi-layer caching for LLM responses, HTTP requests, embeddings,
and other external service calls during testing.

Example:
    # Configure caching for tests
    config = CacheConfig(
        mode=CacheMode.READ_WRITE,
        ttl_seconds=3600,
        storage_path=".test_cache"
    )
    UnifiedCacheManager.configure(config)

    # Use in service code
    cache = UnifiedCacheManager.get_instance()
    result = await cache.get_or_call(
        service="llm",
        key_parts=(model, prompt),
        call_fn=lambda: llm.chat(prompt)
    )
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


class CacheMode(str, Enum):
    """Cache operation modes."""

    DISABLED = "disabled"  # No caching (production default)
    READ_ONLY = "read_only"  # Use cache, don't write new entries
    WRITE_THROUGH = "write"  # Always call API, cache response
    READ_WRITE = "read_write"  # Full caching (test default)
    RECORD = "record"  # Record mode for building fixtures
    REPLAY = "replay"  # Replay only, fail on miss


class CacheMissError(Exception):
    """Raised when cache miss occurs in replay mode."""

    pass


@dataclass
class CacheConfig:
    """Configuration for cache behavior."""

    mode: CacheMode = CacheMode.DISABLED
    ttl_seconds: int = 3600  # 1 hour default
    max_entries: int = 10000
    storage_path: str = ".test_cache"
    persist_on_exit: bool = True

    # Per-service mode overrides
    llm_mode: CacheMode | None = None
    http_mode: CacheMode | None = None
    mcp_mode: CacheMode | None = None
    embedding_mode: CacheMode | None = None

    @classmethod
    def from_env(cls) -> "CacheConfig":
        """Create config from environment variables."""
        mode_str = os.environ.get("TEST_CACHE_MODE", "disabled")
        try:
            mode = CacheMode(mode_str)
        except ValueError:
            mode = CacheMode.DISABLED

        return cls(
            mode=mode,
            ttl_seconds=int(os.environ.get("TEST_CACHE_TTL", "3600")),
            storage_path=os.environ.get("TEST_CACHE_PATH", ".test_cache"),
            llm_mode=cls._get_service_mode("LLM"),
            http_mode=cls._get_service_mode("HTTP"),
            mcp_mode=cls._get_service_mode("MCP"),
            embedding_mode=cls._get_service_mode("EMBEDDING"),
        )

    @staticmethod
    def _get_service_mode(service: str) -> CacheMode | None:
        """Get mode override for a specific service."""
        mode_str = os.environ.get(f"TEST_CACHE_{service}_MODE")
        if mode_str:
            try:
                return CacheMode(mode_str)
            except ValueError:
                pass
        return None


@dataclass
class CacheEntry:
    """A single cache entry with metadata."""

    value: Any
    timestamp: float
    hits: int = 0
    service: str = ""
    key_hash: str = ""


class UnifiedCacheManager:
    """
    Central cache manager for all external service calls.

    Singleton pattern ensures consistent caching across the application.

    Usage:
        # Configure once at startup
        UnifiedCacheManager.configure(CacheConfig(mode=CacheMode.READ_WRITE))

        # Use anywhere
        cache = UnifiedCacheManager.get_instance()
        result = await cache.get_or_call("llm", (prompt,), call_fn)
    """

    _instance: "UnifiedCacheManager | None" = None

    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._caches: dict[str, dict[str, CacheEntry]] = {}
        self._stats = {"hits": 0, "misses": 0, "writes": 0, "evictions": 0}
        self._load_persistent_cache()

    @classmethod
    def get_instance(cls) -> "UnifiedCacheManager":
        """Get the singleton cache manager instance."""
        if cls._instance is None:
            cls._instance = cls(CacheConfig.from_env())
        return cls._instance

    @classmethod
    def configure(cls, config: CacheConfig) -> "UnifiedCacheManager":
        """Configure the singleton instance with specific config."""
        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (for testing)."""
        if cls._instance and cls._instance.config.persist_on_exit:
            cls._instance._save_persistent_cache()
        cls._instance = None

    def get_mode(self, service: str) -> CacheMode:
        """Get effective mode for a service."""
        override = getattr(self.config, f"{service}_mode", None)
        return override or self.config.mode

    def cache_key(self, service: str, key_parts: tuple) -> str:
        """Generate deterministic cache key from service and key parts."""
        # Handle non-serializable objects
        def make_serializable(obj: Any) -> Any:
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, (list, tuple)):
                return [make_serializable(x) for x in obj]
            if isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in sorted(obj.items())}
            # For complex objects, use repr
            return repr(obj)

        data = json.dumps(
            {"service": service, "keys": make_serializable(key_parts)},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:32]

    async def get_or_call(
        self,
        service: str,
        key_parts: tuple,
        call_fn: Callable[[], Awaitable[T]],
        ttl: int | None = None,
    ) -> T:
        """
        Get from cache or call function and cache the result.

        Args:
            service: Service identifier (e.g., "llm", "http", "mcp")
            key_parts: Tuple of values that uniquely identify this call
            call_fn: Async function to call on cache miss
            ttl: Optional TTL override in seconds

        Returns:
            Cached or fresh result

        Raises:
            CacheMissError: If in replay mode and no cache entry exists
        """
        mode = self.get_mode(service)

        if mode == CacheMode.DISABLED:
            return await call_fn()

        key = self.cache_key(service, key_parts)
        cache = self._caches.setdefault(service, {})
        effective_ttl = ttl or self.config.ttl_seconds

        # Check cache
        if key in cache:
            entry = cache[key]
            age = time.time() - entry.timestamp
            if age < effective_ttl:
                entry.hits += 1
                self._stats["hits"] += 1
                return entry.value

        # Cache miss
        self._stats["misses"] += 1

        if mode == CacheMode.REPLAY:
            raise CacheMissError(
                f"Cache miss in replay mode: service={service}, key={key[:8]}..."
            )

        if mode == CacheMode.READ_ONLY:
            return await call_fn()

        # Call and cache
        result = await call_fn()

        # Evict if over limit
        if len(cache) >= self.config.max_entries:
            self._evict_oldest(service)

        cache[key] = CacheEntry(
            value=result,
            timestamp=time.time(),
            service=service,
            key_hash=key,
        )
        self._stats["writes"] += 1

        return result

    def get_sync(
        self,
        service: str,
        key_parts: tuple,
        call_fn: Callable[[], T],
        ttl: int | None = None,
    ) -> T:
        """Synchronous version of get_or_call for non-async code."""
        mode = self.get_mode(service)

        if mode == CacheMode.DISABLED:
            return call_fn()

        key = self.cache_key(service, key_parts)
        cache = self._caches.setdefault(service, {})
        effective_ttl = ttl or self.config.ttl_seconds

        if key in cache:
            entry = cache[key]
            age = time.time() - entry.timestamp
            if age < effective_ttl:
                entry.hits += 1
                self._stats["hits"] += 1
                return entry.value

        self._stats["misses"] += 1

        if mode == CacheMode.REPLAY:
            raise CacheMissError(
                f"Cache miss in replay mode: service={service}, key={key[:8]}..."
            )

        if mode == CacheMode.READ_ONLY:
            return call_fn()

        result = call_fn()

        if len(cache) >= self.config.max_entries:
            self._evict_oldest(service)

        cache[key] = CacheEntry(
            value=result,
            timestamp=time.time(),
            service=service,
            key_hash=key,
        )
        self._stats["writes"] += 1

        return result

    def invalidate(self, service: str, key_parts: tuple | None = None) -> int:
        """
        Invalidate cache entries.

        Args:
            service: Service to invalidate
            key_parts: Specific key to invalidate, or None for all service entries

        Returns:
            Number of entries invalidated
        """
        if service not in self._caches:
            return 0

        if key_parts is None:
            count = len(self._caches[service])
            self._caches[service] = {}
            return count

        key = self.cache_key(service, key_parts)
        if key in self._caches[service]:
            del self._caches[service][key]
            return 1
        return 0

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_entries = sum(len(c) for c in self._caches.values())
        hit_rate = (
            self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
            if (self._stats["hits"] + self._stats["misses"]) > 0
            else 0.0
        )

        return {
            **self._stats,
            "total_entries": total_entries,
            "hit_rate": f"{hit_rate:.1%}",
            "services": list(self._caches.keys()),
            "entries_per_service": {k: len(v) for k, v in self._caches.items()},
        }

    def _evict_oldest(self, service: str) -> None:
        """Evict oldest entry from a service cache."""
        cache = self._caches.get(service, {})
        if not cache:
            return

        oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
        del cache[oldest_key]
        self._stats["evictions"] += 1

    def _load_persistent_cache(self) -> None:
        """Load cache from disk if it exists."""
        cache_file = Path(self.config.storage_path) / "cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                    self._caches = data.get("caches", {})
                    # Don't restore stats - start fresh
            except Exception:
                # If load fails, start fresh
                pass

    def _save_persistent_cache(self) -> None:
        """Save cache to disk."""
        cache_dir = Path(self.config.storage_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / "cache.pkl"

        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"caches": self._caches}, f)
        except Exception:
            # If save fails, that's okay
            pass

    def __del__(self):
        """Save cache on exit if configured."""
        if self.config.persist_on_exit:
            self._save_persistent_cache()
