"""Adapters for integrating draagon-ai with various applications.

This module provides adapter implementations and base protocols that applications
can use to bridge their existing services with draagon-ai's cognitive framework.

Built-in Adapters:
    - RoxyLayeredAdapter: Allows Roxy voice assistant to use LayeredMemoryProvider

Adapter Patterns (for custom implementations):
    - LLMAdapter: Wrap your LLM service to implement draagon_ai.llm.LLMProvider
    - MemoryAdapter: Wrap your memory service to implement MemoryProvider
    - ToolAdapter: Wrap your tool executor to implement ToolProvider

Example (Roxy integration):
    from draagon_ai.adapters import RoxyLayeredAdapter
    from draagon_ai.memory.providers import LayeredMemoryProvider, LayeredMemoryConfig

    config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
    provider = LayeredMemoryProvider(config=config, embedding_provider=embedder)
    await provider.initialize()

    # Drop-in replacement for Roxy's MemoryService
    memory_adapter = RoxyLayeredAdapter(provider)
    result = await memory_adapter.store(content="...", user_id="doug")

Example (custom adapter):
    from draagon_ai.llm import LLMProvider, ChatMessage, ChatResponse
    from your_app.services.llm import YourLLMService

    class YourLLMAdapter(LLMProvider):
        def __init__(self, llm_service: YourLLMService):
            self._llm = llm_service

        async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
            # Convert and call your service
            ...
"""

from draagon_ai.adapters.roxy import (
    RoxyLayeredAdapter,
    RoxyMemoryType,
    RoxyMemoryScope,
    ROXY_TYPE_MAPPING,
    ROXY_SCOPE_MAPPING,
)

__all__ = [
    "RoxyLayeredAdapter",
    "RoxyMemoryType",
    "RoxyMemoryScope",
    "ROXY_TYPE_MAPPING",
    "ROXY_SCOPE_MAPPING",
]
