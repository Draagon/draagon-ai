"""LLM abstraction layer for Draagon AI.

This module provides both abstract interfaces and concrete implementations
of LLM providers.

Implementations:
    - GroqLLM: High-speed inference via Groq API
    - MockLLM: Simple mock for unit tests
    - RealisticMockLLM: Context-aware mock for integration tests

Usage:
    from draagon_ai.llm import create_llm, GroqLLM

    # Auto-detect provider
    llm = create_llm()

    # Specific provider
    llm = GroqLLM(api_key="your-key")

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
from draagon_ai.llm.factory import create_llm

# Import implementations (lazy to avoid import errors if deps missing)
try:
    from draagon_ai.llm.groq import GroqLLM, GroqConfig
except ImportError:
    GroqLLM = None  # type: ignore
    GroqConfig = None  # type: ignore

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
    # Implementations
    "GroqLLM",
    "GroqConfig",
    "MockLLM",
    "RealisticMockLLM",
]
