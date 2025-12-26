"""Abstract base classes for LLM providers.

These protocols define what the cognitive engine needs from LLM providers.
Host applications implement these interfaces with their specific providers.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal


class ModelTier(str, Enum):
    """Model capability tiers for routing queries.

    Cognitive services use these tiers to request appropriate model capabilities.
    The host application maps these to actual providers/models.
    """

    LOCAL = "local"  # Fast, simple queries (8B models)
    COMPLEX = "complex"  # Complex reasoning (70B models)
    DEEP = "deep"  # Nuanced judgment (Claude Opus, GPT-4)


@dataclass
class ChatMessage:
    """A message in a conversation."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: str | None = None  # For tool messages
    tool_calls: list["ToolCall"] | None = None
    tool_call_id: str | None = None  # For tool responses


@dataclass
class ToolCall:
    """A tool call from the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolDefinition:
    """Definition of a tool the model can call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@dataclass
class ChatResponse:
    """Response from a chat completion."""

    content: str
    role: str = "assistant"
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None  # tokens used

    # Timing
    latency_ms: float | None = None


class LLMProvider(ABC):
    """Abstract interface for LLM chat providers.

    Cognitive services depend on this interface, not concrete implementations.
    This allows the engine to work with any LLM backend.
    """

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[ToolDefinition] | None = None,
        tier: ModelTier = ModelTier.LOCAL,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Complete a chat conversation.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt to prepend
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens in response
            tools: Optional tool definitions for function calling
            tier: Model tier to use for this request
            response_format: Optional structured output format (JSON schema)

        Returns:
            ChatResponse with the model's reply
        """
        ...

    @abstractmethod
    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> AsyncIterator[str]:
        """Stream a chat completion token by token.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tier: Model tier to use

        Yields:
            Individual tokens as they're generated
        """
        ...

    async def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> str:
        """Simple text completion (convenience wrapper around chat).

        Args:
            prompt: The prompt to complete
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tier: Model tier to use

        Returns:
            The completion text
        """
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )
        return response.content

    async def chat_json(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        max_tokens: int = 200,
        temperature: float = 0.1,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> dict[str, Any]:
        """Chat with JSON output parsing.

        Args:
            messages: Chat messages
            max_tokens: Max tokens to generate
            temperature: Sampling temperature (lower for structured output)
            tier: Model tier to use

        Returns:
            Dict with 'parsed' (the JSON object), 'content' (raw)
        """
        import json
        import re

        response = await self.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            tier=tier,
        )

        content = response.content

        # Try to parse JSON from response
        parsed = None
        try:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(1))
            else:
                # Try direct parse
                parsed = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

        return {
            "content": content,
            "parsed": parsed,
        }


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers.

    Used by memory systems for semantic search.
    """

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        ...


@dataclass
class LLMConfig:
    """Configuration for LLM providers.

    Host applications can use this to configure the cognitive engine's
    LLM access. The engine doesn't need to know about specific providers.
    """

    # Model tiers (host maps these to actual models)
    local_model: str = "llama-3.1-8b-instant"
    complex_model: str = "llama-3.3-70b-versatile"
    deep_model: str = "claude-opus-4"

    # Embedding model
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # Defaults
    default_temperature: float = 0.7
    default_max_tokens: int = 1024

    # Rate limiting
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
