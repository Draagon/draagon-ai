"""Adapters module - patterns for integrating applications with draagon-ai.

This module documents adapter patterns that applications can use to integrate
with draagon-ai's cognitive framework. Applications should implement these
patterns in their own codebase, not in draagon-ai.

Adapter Patterns:
-----------------

1. LLM Provider Adapter:
   Implement draagon_ai.llm.LLMProvider to wrap your LLM service.

   Example:
       from draagon_ai.llm import LLMProvider, ChatResponse, ModelTier

       class MyLLMProvider(LLMProvider):
           def __init__(self, my_llm_service):
               self._llm = my_llm_service

           async def chat(self, messages, *, tier=ModelTier.LOCAL, **kwargs) -> ChatResponse:
               result = await self._llm.complete(messages)
               return ChatResponse(content=result.text)

2. Memory Provider Adapter:
   Use draagon_ai.memory.LayeredMemoryProvider directly, or implement
   draagon_ai.memory.MemoryProvider for custom backends.

   Example:
       from draagon_ai.memory import LayeredMemoryProvider, LayeredMemoryConfig

       config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
       memory = LayeredMemoryProvider(config=config, embedding_provider=my_embedder)
       await memory.initialize()

3. Credibility Provider Adapter:
   Implement draagon_ai.cognition.beliefs.CredibilityProvider for user trust.

   Example:
       from draagon_ai.cognition.beliefs import CredibilityProvider

       class MyCredibilityProvider(CredibilityProvider):
           def get_user_credibility(self, user_id: str) -> float | None:
               return self._user_service.get_trust_score(user_id)

4. Tool Provider Adapter:
   Use draagon_ai.orchestration.Tool directly to define tools.

   Example:
       from draagon_ai.orchestration import Tool, ToolParameter

       tools = [
           Tool(
               name="get_weather",
               description="Get current weather",
               handler=my_weather_handler,
               parameters=[
                   ToolParameter(name="location", type="string", required=True),
               ],
           ),
       ]

5. Extension Integration:
   Create custom extensions by subclassing draagon_ai.extensions.Extension.

   Example:
       from draagon_ai.extensions import Extension, ExtensionInfo

       class MyExtension(Extension):
           @property
           def info(self) -> ExtensionInfo:
               return ExtensionInfo(name="my_extension", version="1.0.0")

           def get_tools(self) -> list[Tool]:
               return [...]

Reference Implementation:
------------------------
See the Roxy voice assistant (github.com/dmealing/roxy-voice-assistant)
for a complete reference implementation showing how to:
- Implement LLMProvider with multi-tier model routing
- Add domain-specific extensions (Home Assistant, Calendar)
- Create custom tools
- Implement CredibilityProvider with multi-dimensional tracking
"""

# No exports - this module documents patterns, doesn't provide implementations.
# Applications implement these patterns in their own codebase.

__all__: list[str] = []
