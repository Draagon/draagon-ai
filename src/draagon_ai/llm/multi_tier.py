"""Multi-tier LLM router.

Routes requests to appropriate LLM providers based on model tier.
This allows using different providers for different complexity levels.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .base import (
    ChatMessage,
    ChatResponse,
    EmbeddingProvider,
    LLMProvider,
    ModelTier,
    ToolDefinition,
)


@dataclass
class MultiTierConfig:
    """Configuration for multi-tier router."""

    # Default tiers if not all providers specified
    default_tier: ModelTier = ModelTier.LOCAL

    # Fallback behavior
    fallback_enabled: bool = True  # Fall back to available provider if requested unavailable


class MultiTierRouter(LLMProvider):
    """Routes requests to different LLM providers based on tier.

    This allows using:
    - Fast local models (Groq 8B, Ollama) for simple queries
    - Complex models (Groq 70B) for reasoning tasks
    - Deep models (Claude Opus) for nuanced judgment

    Usage:
        from draagon_ai.llm import GroqLLM, AnthropicLLM, MultiTierRouter

        router = MultiTierRouter(
            fast=GroqLLM(model="llama-3.1-8b-instant"),
            complex=GroqLLM(model="llama-3.3-70b-versatile"),
            deep=AnthropicLLM(model="claude-opus-4-20250514"),
        )

        # Automatically routes to appropriate provider
        response = await router.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tier=ModelTier.DEEP,  # Routes to Claude
        )
    """

    def __init__(
        self,
        fast: LLMProvider | None = None,
        complex: LLMProvider | None = None,
        deep: LLMProvider | None = None,
        embedding: EmbeddingProvider | None = None,
        config: MultiTierConfig | None = None,
    ):
        """Initialize multi-tier router.

        Args:
            fast: Provider for fast/local tier (8B models)
            complex: Provider for complex tier (70B models)
            deep: Provider for deep tier (Claude Opus)
            embedding: Provider for embeddings
            config: Optional configuration
        """
        self.config = config or MultiTierConfig()

        # Map tiers to providers
        self._providers: dict[ModelTier, LLMProvider] = {}

        if fast:
            self._providers[ModelTier.LOCAL] = fast

        if complex:
            self._providers[ModelTier.COMPLEX] = complex

        if deep:
            self._providers[ModelTier.DEEP] = deep

        self._embedding = embedding

        if not self._providers:
            raise ValueError("At least one LLM provider must be specified")

    def _get_provider(self, tier: ModelTier) -> LLMProvider:
        """Get provider for tier, with fallback if enabled."""
        if tier in self._providers:
            return self._providers[tier]

        if not self.config.fallback_enabled:
            raise ValueError(f"No provider configured for tier: {tier}")

        # Fallback logic: prefer higher capability
        fallback_order = {
            ModelTier.LOCAL: [ModelTier.COMPLEX, ModelTier.DEEP],
            ModelTier.COMPLEX: [ModelTier.DEEP, ModelTier.LOCAL],
            ModelTier.DEEP: [ModelTier.COMPLEX, ModelTier.LOCAL],
        }

        for fallback_tier in fallback_order.get(tier, []):
            if fallback_tier in self._providers:
                return self._providers[fallback_tier]

        # Last resort: use any available
        return next(iter(self._providers.values()))

    @property
    def providers(self) -> dict[ModelTier, LLMProvider]:
        """Get all configured providers."""
        return self._providers.copy()

    @property
    def available_tiers(self) -> list[ModelTier]:
        """Get list of available tiers."""
        return list(self._providers.keys())

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
        """Route chat request to appropriate provider.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Optional tool definitions
            tier: Model tier to use (determines provider)
            response_format: Optional JSON response format

        Returns:
            ChatResponse from the routed provider
        """
        provider = self._get_provider(tier)
        return await provider.chat(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tools=tools,
            tier=tier,
            response_format=response_format,
        )

    async def chat_stream(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> AsyncIterator[str]:
        """Route streaming request to appropriate provider.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tier: Model tier to use

        Yields:
            Tokens from the routed provider
        """
        provider = self._get_provider(tier)
        async for token in provider.chat_stream(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        ):
            yield token

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using configured provider.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ValueError: If no embedding provider configured
        """
        if self._embedding:
            return await self._embedding.embed(text)

        # Try to find an embedding-capable provider
        for provider in self._providers.values():
            if isinstance(provider, EmbeddingProvider):
                return await provider.embed(text)

        raise ValueError("No embedding provider configured")

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        if self._embedding:
            return await self._embedding.embed_batch(texts)

        # Try to find an embedding-capable provider
        for provider in self._providers.values():
            if isinstance(provider, EmbeddingProvider):
                return await provider.embed_batch(texts)

        raise ValueError("No embedding provider configured")
