"""Ollama LLM provider implementation.

Provides access to local LLM models via Ollama.
"""

import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .base import (
    ChatMessage,
    ChatResponse,
    EmbeddingProvider,
    LLMProvider,
    ModelTier,
    ToolCall,
    ToolDefinition,
)


@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM provider."""

    base_url: str = "http://localhost:11434"

    # Model mapping
    local_model: str = "llama3.2:3b"
    complex_model: str = "llama3.1:8b"
    deep_model: str = "llama3.1:70b"

    # Embedding model
    embedding_model: str = "nomic-embed-text"

    # Timeouts
    timeout_seconds: float = 300.0  # Local models can be slow

    # Options
    num_ctx: int = 4096  # Context window
    num_gpu: int = -1  # Use all GPUs


class OllamaLLM(LLMProvider, EmbeddingProvider):
    """Ollama LLM provider for local model inference.

    Usage:
        llm = OllamaLLM()  # Uses default localhost:11434

        response = await llm.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tier=ModelTier.LOCAL,
        )
        print(response.content)
    """

    def __init__(
        self,
        base_url: str | None = None,
        config: OllamaConfig | None = None,
    ):
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL (default: http://localhost:11434)
            config: Optional configuration
        """
        self.config = config or OllamaConfig()
        if base_url:
            self.config.base_url = base_url

        self._client = None

    @property
    def client(self):
        """Lazy-load the Ollama client."""
        if self._client is None:
            try:
                from ollama import AsyncClient

                self._client = AsyncClient(host=self.config.base_url)
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                )
        return self._client

    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension (depends on model)."""
        # nomic-embed-text produces 768-dim embeddings
        return 768

    def _get_model(self, tier: ModelTier) -> str:
        """Get the model for a tier."""
        if tier == ModelTier.LOCAL:
            return self.config.local_model
        elif tier == ModelTier.COMPLEX:
            return self.config.complex_model
        elif tier == ModelTier.DEEP:
            return self.config.deep_model
        return self.config.local_model

    def _convert_messages(
        self,
        messages: list[ChatMessage] | list[dict[str, Any]],
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convert messages to Ollama format."""
        result = []

        # Add system prompt if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if isinstance(msg, ChatMessage):
                converted = {"role": msg.role, "content": msg.content}
            else:
                converted = {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            result.append(converted)

        return result

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
        """Complete a chat conversation using Ollama.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Optional tool definitions (limited support)
            tier: Model tier to use
            response_format: Optional JSON response format

        Returns:
            ChatResponse with the model's reply
        """
        start_time = time.time()
        model = self._get_model(tier)
        converted_messages = self._convert_messages(messages, system_prompt)

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.config.num_ctx,
        }

        if self.config.num_gpu >= 0:
            options["num_gpu"] = self.config.num_gpu

        # Make request
        format_type = "json" if response_format else None

        response = await self.client.chat(
            model=model,
            messages=converted_messages,
            options=options,
            format=format_type,
        )

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            content=response["message"]["content"],
            role="assistant",
            tool_calls=None,  # Ollama tool support is limited
            finish_reason="stop",
            model=model,
            usage={
                "prompt_tokens": response.get("prompt_eval_count", 0),
                "completion_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0)
                + response.get("eval_count", 0),
            },
            latency_ms=latency_ms,
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
        model = self._get_model(tier)
        converted_messages = self._convert_messages(messages, system_prompt)

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": self.config.num_ctx,
        }

        async for chunk in await self.client.chat(
            model=model,
            messages=converted_messages,
            options=options,
            stream=True,
        ):
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        response = await self.client.embeddings(
            model=self.config.embedding_model,
            prompt=text,
        )
        return response["embedding"]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: Texts to embed

        Returns:
            List of embedding vectors
        """
        # Ollama doesn't have native batch embedding, so we do it sequentially
        return [await self.embed(text) for text in texts]
