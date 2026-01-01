"""Groq LLM provider implementation.

Provides high-speed inference using Groq's API.
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
class GroqConfig:
    """Configuration for Groq LLM provider."""

    api_key: str | None = None
    base_url: str = "https://api.groq.com/openai/v1"

    # Model mapping
    local_model: str = "llama-3.1-8b-instant"
    complex_model: str = "llama-3.3-70b-versatile"
    deep_model: str = "llama-3.3-70b-versatile"  # Groq doesn't have Claude

    # Timeouts
    timeout_seconds: float = 60.0

    # Rate limiting
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class GroqLLM(LLMProvider):
    """Groq LLM provider with high-speed inference.

    Usage:
        llm = GroqLLM(api_key="your-key")

        response = await llm.chat(
            messages=[{"role": "user", "content": "Hello"}],
            tier=ModelTier.COMPLEX,
        )
        print(response.content)
    """

    def __init__(
        self,
        api_key: str | None = None,
        config: GroqConfig | None = None,
    ):
        """Initialize Groq provider.

        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            config: Optional configuration
        """
        self.config = config or GroqConfig()
        self.config.api_key = api_key or os.environ.get("GROQ_API_KEY")

        if not self.config.api_key:
            raise ValueError("Groq API key required. Set GROQ_API_KEY or pass api_key.")

        self._client = None

    @property
    def client(self):
        """Lazy-load the Groq client."""
        if self._client is None:
            try:
                from groq import AsyncGroq
                self._client = AsyncGroq(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout_seconds,
                )
            except ImportError:
                raise ImportError(
                    "groq package required. Install with: pip install groq"
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
    ) -> list[dict[str, Any]]:
        """Convert messages to Groq format."""
        result = []

        # Add system prompt if provided
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        for msg in messages:
            if isinstance(msg, ChatMessage):
                converted = {"role": msg.role, "content": msg.content}
                if msg.name:
                    converted["name"] = msg.name
                if msg.tool_call_id:
                    converted["tool_call_id"] = msg.tool_call_id
            elif isinstance(msg, dict):
                converted = dict(msg)
            else:
                # Handle dataclass-like objects (e.g., LLMMessage from protocols)
                converted = {"role": msg.role, "content": msg.content}
                if hasattr(msg, "name") and msg.name:
                    converted["name"] = msg.name
                if hasattr(msg, "tool_call_id") and msg.tool_call_id:
                    converted["tool_call_id"] = msg.tool_call_id
            result.append(converted)

        return result

    def _convert_tools(
        self,
        tools: list[ToolDefinition] | None,
    ) -> list[dict] | None:
        """Convert tools to Groq format."""
        if not tools:
            return None

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
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
        model: str | None = None,  # Accept but ignore - use tier instead
        **kwargs: Any,  # Accept any additional kwargs for protocol compatibility
    ) -> ChatResponse:
        """Complete a chat conversation using Groq.

        Args:
            messages: Conversation history
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Optional tool definitions
            tier: Model tier to use
            response_format: Optional JSON response format

        Returns:
            ChatResponse with the model's reply
        """
        start_time = time.time()
        model = self._get_model(tier)
        converted_messages = self._convert_messages(messages, system_prompt)
        converted_tools = self._convert_tools(tools)

        # Build request kwargs
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if converted_tools:
            kwargs["tools"] = converted_tools
            kwargs["tool_choice"] = "auto"

        if response_format:
            kwargs["response_format"] = response_format

        # Make request with retries
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    await self._async_sleep(self.config.retry_delay_seconds * (attempt + 1))
        else:
            raise last_error

        # Parse response
        choice = response.choices[0]
        message = choice.message

        # Parse tool calls if present
        tool_calls = None
        if message.tool_calls:
            import json
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments) if tc.function.arguments else {},
                )
                for tc in message.tool_calls
            ]

        latency_ms = (time.time() - start_time) * 1000

        return ChatResponse(
            content=message.content or "",
            role=message.role,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
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

        stream = await self.client.chat.completions.create(
            model=model,
            messages=converted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> str:
        """Simple text generation (compatibility method).

        This matches the interface expected by BehaviorArchitectService.

        Args:
            prompt: The prompt to complete
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tier: Model tier to use

        Returns:
            Generated text
        """
        response = await self.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )
        return response.content

    async def _async_sleep(self, seconds: float) -> None:
        """Async sleep for retries."""
        import asyncio
        await asyncio.sleep(seconds)
