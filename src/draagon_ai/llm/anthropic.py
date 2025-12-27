"""Anthropic LLM provider implementation.

Provides access to Claude models via Anthropic's API.
"""

import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from .base import (
    ChatMessage,
    ChatResponse,
    LLMProvider,
    ModelTier,
    ToolCall,
    ToolDefinition,
)


@dataclass
class AnthropicConfig:
    """Configuration for Anthropic LLM provider."""

    api_key: str | None = None
    base_url: str | None = None  # For proxies

    # Model mapping
    local_model: str = "claude-3-5-haiku-20241022"
    complex_model: str = "claude-sonnet-4-20250514"
    deep_model: str = "claude-opus-4-20250514"

    # Timeouts
    timeout_seconds: float = 120.0

    # Rate limiting
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class AnthropicLLM(LLMProvider):
    """Anthropic LLM provider for Claude models.

    Usage:
        llm = AnthropicLLM(api_key="your-key")

        response = await llm.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tier=ModelTier.DEEP,
        )
        print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: AnthropicConfig | None = None,
    ):
        """Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
            config: Optional configuration
        """
        self.config = config or AnthropicConfig()
        self.config.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

        if not self.config.api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY or pass api_key."
            )

        self._client = None

    @property
    def client(self):
        """Lazy-load the Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                kwargs = {"api_key": self.config.api_key}
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url

                self._client = AsyncAnthropic(**kwargs)
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
        return self._client

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
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Convert messages to Anthropic format.

        Returns system prompt separately as Anthropic handles it differently.
        """
        result = []
        extracted_system = system_prompt

        for msg in messages:
            if isinstance(msg, ChatMessage):
                role = msg.role
                content = msg.content
            else:
                role = msg.get("role", "user")
                content = msg.get("content", "")

            # Anthropic handles system prompt separately
            if role == "system":
                if not extracted_system:
                    extracted_system = content
                continue

            # Map tool role to user with tool_result
            if role == "tool":
                result.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.get("tool_call_id") if isinstance(msg, dict) else getattr(msg, "tool_call_id", ""),
                            "content": content,
                        }
                    ],
                })
                continue

            result.append({"role": role, "content": content})

        return extracted_system, result

    def _convert_tools(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[dict] | None:
        """Convert tools to Anthropic format."""
        if not tools:
            return None

        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in tools
        ]

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
        """Complete a chat conversation using Anthropic.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Optional tool definitions
            tier: Model tier to use
            response_format: Optional JSON response format (not directly supported)

        Returns:
            ChatResponse with the model's reply
        """
        start_time = time.time()
        model = self._get_model(tier)
        extracted_system, converted_messages = self._convert_messages(
            messages, system_prompt
        )
        converted_tools = self._convert_tools(tools)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if extracted_system:
            kwargs["system"] = extracted_system

        if converted_tools:
            kwargs["tools"] = converted_tools

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.messages.create(**kwargs)
                break
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await self._async_sleep(
                        self.config.retry_delay_seconds * (attempt + 1)
                    )
        else:
            raise last_error

        # Parse response
        content = ""
        tool_calls = None

        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            content=content,
            role="assistant",
            tool_calls=tool_calls,
            finish_reason=response.stop_reason,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
        extracted_system, converted_messages = self._convert_messages(
            messages, system_prompt
        )

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if extracted_system:
            kwargs["system"] = extracted_system

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep for retries."""
        import asyncio

        await asyncio.sleep(seconds)
