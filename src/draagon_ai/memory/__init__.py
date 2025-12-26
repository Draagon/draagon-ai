"""Memory abstraction layer for Draagon AI.

This module defines abstract interfaces for memory systems. The cognitive
engine uses these interfaces without knowing about specific vector databases.
"""

from draagon_ai.memory.base import (
    MemoryProvider,
    MemoryType,
    MemoryScope,
    Memory,
    SearchResult,
    MemoryConfig,
)

__all__ = [
    "MemoryProvider",
    "MemoryType",
    "MemoryScope",
    "Memory",
    "SearchResult",
    "MemoryConfig",
]
