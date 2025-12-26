"""Factory functions for creating LLM providers."""

import os
from typing import Any

from .base import LLMProvider


def create_llm(
    provider: str = "auto",
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        provider: Provider name ("groq", "mock", "realistic-mock", "auto")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider

    Examples:
        # Auto-detect based on environment
        llm = create_llm()

        # Specific provider
        llm = create_llm("groq", api_key="your-key")

        # Mock for testing
        llm = create_llm("mock", responses=["Response 1", "Response 2"])

        # Realistic mock for integration tests
        llm = create_llm("realistic-mock")
    """
    if provider == "auto":
        # Auto-detect based on available credentials
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        else:
            provider = "realistic-mock"

    if provider == "groq":
        from .groq import GroqLLM
        return GroqLLM(**kwargs)

    elif provider == "mock":
        from .mock import MockLLM
        return MockLLM(**kwargs)

    elif provider == "realistic-mock":
        from .mock import RealisticMockLLM
        return RealisticMockLLM(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")
