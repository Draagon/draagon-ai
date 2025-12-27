"""LLM abstraction layer for Draagon AI.

This module provides both abstract interfaces and concrete implementations
of LLM providers.

Implementations:
    - GroqLLM: High-speed inference via Groq API
    - AnthropicLLM: Claude models via Anthropic API
    - OllamaLLM: Local models via Ollama
    - MultiTierRouter: Routes to different providers by tier
    - MockLLM: Simple mock for unit tests
    - RealisticMockLLM: Context-aware mock for integration tests

Usage:
    from draagon_ai.llm import create_llm, GroqLLM, MultiTierRouter

    # Auto-detect provider
    llm = create_llm()

    # Specific provider
    llm = GroqLLM(api_key="your-key")

    # Multi-tier routing
    llm = MultiTierRouter(
        fast=GroqLLM(api_key="..."),
        deep=AnthropicLLM(api_key="..."),
    )

    # Mock for tests
    llm = create_llm("realistic-mock")
"""

from draagon_ai.llm.base import (
    LLMProvider,
    EmbeddingProvider,
    ModelTier,
    ChatMessage,
    ChatResponse,
    ToolCall,
    ToolDefinition,
    LLMConfig,
)
from draagon_ai.llm.factory import create_llm, create_multi_tier_llm

# Import implementations (lazy to avoid import errors if deps missing)
try:
    from draagon_ai.llm.groq import GroqLLM, GroqConfig
except ImportError:
    GroqLLM = None  # type: ignore
    GroqConfig = None  # type: ignore

try:
    from draagon_ai.llm.anthropic import AnthropicLLM, AnthropicConfig
except ImportError:
    AnthropicLLM = None  # type: ignore
    AnthropicConfig = None  # type: ignore

try:
    from draagon_ai.llm.ollama import OllamaLLM, OllamaConfig
except ImportError:
    OllamaLLM = None  # type: ignore
    OllamaConfig = None  # type: ignore

from draagon_ai.llm.multi_tier import MultiTierRouter, MultiTierConfig
from draagon_ai.llm.mock import MockLLM, RealisticMockLLM

__all__ = [
    # Abstract
    "LLMProvider",
    "EmbeddingProvider",
    "ModelTier",
    "ChatMessage",
    "ChatResponse",
    "ToolCall",
    "ToolDefinition",
    "LLMConfig",
    # Factory
    "create_llm",
    "create_multi_tier_llm",
    # Implementations
    "GroqLLM",
    "GroqConfig",
    "AnthropicLLM",
    "AnthropicConfig",
    "OllamaLLM",
    "OllamaConfig",
    "MultiTierRouter",
    "MultiTierConfig",
    "MockLLM",
    "RealisticMockLLM",
]
