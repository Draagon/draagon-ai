"""Adapters for integrating draagon-ai with various applications.

This module contains reference adapter implementations that show how to
bridge draagon-ai's protocols with existing application services.

Usage:
    # In Roxy
    from draagon_ai.adapters.roxy import RoxyLLMAdapter, RoxyMemoryAdapter

    llm_provider = RoxyLLMAdapter(roxy_llm_service)
    memory_provider = RoxyMemoryAdapter(roxy_memory_service)
"""

from .roxy import (
    RoxyLLMAdapter,
    RoxyMemoryAdapter,
    RoxyToolAdapter,
)

__all__ = [
    "RoxyLLMAdapter",
    "RoxyMemoryAdapter",
    "RoxyToolAdapter",
]
