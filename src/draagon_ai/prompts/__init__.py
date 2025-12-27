"""Prompt management system for draagon-ai.

This module provides a versioned, evolvable prompt system that can be stored
in Qdrant and evolved using Promptbreeder-style evolution.

Core Components:
    - PromptRegistry: Version-controlled prompt storage in Qdrant
    - PromptBuilder: Dynamic prompt construction from capabilities/extensions
    - PromptLoader: Hierarchical loading (core → capabilities → extensions → app)

Usage:
    from draagon_ai.prompts import PromptRegistry, PromptDomain, Prompt

    registry = PromptRegistry(qdrant_provider)
    await registry.initialize()

    # Get a prompt
    decision = await registry.get_prompt("decision", "DECISION_PROMPT")

    # List prompts by domain
    routing_prompts = await registry.list_prompts(domain="routing")

Dynamic Prompt Building:
    from draagon_ai.prompts import PromptBuilder, create_default_builder
    from draagon_ai.prompts.domains.templates import DECISION_TEMPLATE

    # Create builder with capabilities
    builder = create_default_builder(capabilities=["calendar", "home_assistant"])

    # Build prompt with dynamic actions
    prompt = builder.build(DECISION_TEMPLATE, assistant_name="MyBot")
"""

from .types import (
    Prompt,
    PromptVersion,
    PromptDomain,
    PromptMetadata,
    PromptStatus,
)
from .registry import PromptRegistry
from .loader import PromptLoader
from .builder import (
    PromptBuilder,
    ActionDef,
    CapabilityDef,
    create_default_builder,
    CORE_ACTIONS,
    CALENDAR_CAPABILITY,
    HOME_ASSISTANT_CAPABILITY,
    COMMANDS_CAPABILITY,
    TIMERS_CAPABILITY,
    CODE_DOCS_CAPABILITY,
)

__all__ = [
    # Types
    "Prompt",
    "PromptVersion",
    "PromptDomain",
    "PromptMetadata",
    "PromptStatus",
    # Registry
    "PromptRegistry",
    "PromptLoader",
    # Builder
    "PromptBuilder",
    "ActionDef",
    "CapabilityDef",
    "create_default_builder",
    # Pre-built capabilities
    "CORE_ACTIONS",
    "CALENDAR_CAPABILITY",
    "HOME_ASSISTANT_CAPABILITY",
    "COMMANDS_CAPABILITY",
    "TIMERS_CAPABILITY",
    "CODE_DOCS_CAPABILITY",
]
