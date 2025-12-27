"""Adapters for integrating draagon-ai with various applications.

This module provides base adapter protocols that applications can implement
to bridge their existing services with draagon-ai's cognitive framework.

Applications should create their own adapter implementations in their codebase.

Adapter Patterns:
    - LLMAdapter: Wrap your LLM service to implement draagon_ai.llm.LLMProvider
    - MemoryAdapter: Wrap your memory service to implement MemoryProvider
    - ToolAdapter: Wrap your tool executor to implement ToolProvider

Example (in your application):
    from draagon_ai.llm import LLMProvider, ChatMessage, ChatResponse
    from your_app.services.llm import YourLLMService

    class YourLLMAdapter(LLMProvider):
        def __init__(self, llm_service: YourLLMService):
            self._llm = llm_service

        async def chat(self, messages: list[ChatMessage], **kwargs) -> ChatResponse:
            # Convert and call your service
            ...
"""

# No exports - applications implement their own adapters
__all__: list[str] = []
