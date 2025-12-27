"""Factory functions for creating LLM providers."""

import os
from typing import Any

from .base import LLMProvider, ModelTier


def create_llm(
    provider: str = "auto",
    **kwargs: Any,
) -> LLMProvider:
    """Create an LLM provider by name.

    Args:
        provider: Provider name ("groq", "anthropic", "ollama", "multi-tier",
                  "mock", "realistic-mock", "auto")
        **kwargs: Provider-specific arguments

    Returns:
        Configured LLM provider

    Examples:
        # Auto-detect based on environment
        llm = create_llm()

        # Specific provider
        llm = create_llm("groq", api_key="your-key")

        # Anthropic/Claude
        llm = create_llm("anthropic", api_key="your-key")

        # Local Ollama
        llm = create_llm("ollama", base_url="http://localhost:11434")

        # Multi-tier routing
        llm = create_llm("multi-tier", fast=groq_llm, deep=anthropic_llm)

        # Mock for testing
        llm = create_llm("mock", responses=["Response 1", "Response 2"])

        # Realistic mock for integration tests
        llm = create_llm("realistic-mock")
    """
    if provider == "auto":
        # Auto-detect based on available credentials
        if os.environ.get("GROQ_API_KEY"):
            provider = "groq"
        elif os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            provider = "realistic-mock"

    if provider == "groq":
        from .groq import GroqLLM
        return GroqLLM(**kwargs)

    elif provider == "anthropic":
        from .anthropic import AnthropicLLM
        return AnthropicLLM(**kwargs)

    elif provider == "ollama":
        from .ollama import OllamaLLM
        return OllamaLLM(**kwargs)

    elif provider == "multi-tier":
        from .multi_tier import MultiTierRouter
        return MultiTierRouter(**kwargs)

    elif provider == "mock":
        from .mock import MockLLM
        return MockLLM(**kwargs)

    elif provider == "realistic-mock":
        from .mock import RealisticMockLLM
        return RealisticMockLLM(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_multi_tier_llm(
    fast: LLMProvider | str | None = None,
    complex: LLMProvider | str | None = None,
    deep: LLMProvider | str | None = None,
    **kwargs: Any,
) -> LLMProvider:
    """Create a multi-tier LLM router.

    Convenience function for creating a router with string provider names.

    Args:
        fast: Provider for fast/local tier (LLMProvider or provider name)
        complex: Provider for complex tier
        deep: Provider for deep tier
        **kwargs: Passed to provider creation

    Returns:
        Configured MultiTierRouter

    Examples:
        # Using provider names
        llm = create_multi_tier_llm(
            fast="groq",
            complex="groq",
            deep="anthropic",
        )

        # Mixed providers and names
        llm = create_multi_tier_llm(
            fast=my_groq_instance,
            deep="anthropic",
        )
    """
    from .multi_tier import MultiTierRouter

    def resolve_provider(p: LLMProvider | str | None) -> LLMProvider | None:
        if p is None:
            return None
        if isinstance(p, str):
            return create_llm(p, **kwargs)
        return p

    return MultiTierRouter(
        fast=resolve_provider(fast),
        complex=resolve_provider(complex),
        deep=resolve_provider(deep),
    )
