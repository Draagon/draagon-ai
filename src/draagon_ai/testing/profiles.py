"""
App profiles for configurable agent testing.

AppProfile defines agent configurations for tests, allowing different
personalities, tool sets, and memory configurations to be easily tested.

Example:
    from draagon_ai.testing import AppProfile, ToolSet

    MY_PROFILE = AppProfile(
        name="researcher",
        personality="You are a research assistant focused on accuracy.",
        tool_set=ToolSet.FULL,
        memory_config={"working_ttl": 600},
        llm_model_tier="advanced",
    )

    # In tests
    async def test_with_profile(agent_factory):
        agent = await agent_factory.create(MY_PROFILE)
        response = await agent.process("Find information about X")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolSet(Enum):
    """Tool availability levels for testing.

    Use to configure which tools an agent has access to.
    """

    MINIMAL = "minimal"  # No tools, just answer
    BASIC = "basic"      # Answer + memory search
    FULL = "full"        # All available tools


@dataclass
class AppProfile:
    """Configuration for agent creation in tests.

    Defines the personality, capabilities, and configuration
    for an agent instance during testing.

    Attributes:
        name: Profile identifier
        personality: System prompt / personality for the agent
        tool_set: Level of tool access (MINIMAL, BASIC, FULL)
        memory_config: Memory provider configuration overrides
        llm_model_tier: LLM model tier ("fast", "standard", "advanced")
        extra_config: Additional app-specific configuration

    Example:
        DEFAULT_PROFILE = AppProfile(
            name="default",
            personality="You are a helpful assistant.",
            tool_set=ToolSet.BASIC,
        )

        RESEARCHER_PROFILE = AppProfile(
            name="researcher",
            personality="You focus on accuracy and cite sources.",
            tool_set=ToolSet.FULL,
            llm_model_tier="advanced",
        )
    """

    name: str
    personality: str
    tool_set: ToolSet = ToolSet.BASIC
    memory_config: dict[str, Any] = field(default_factory=dict)
    llm_model_tier: str = "standard"
    extra_config: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Pre-defined Profiles
# =============================================================================


DEFAULT_PROFILE = AppProfile(
    name="default",
    personality="You are a helpful assistant.",
    tool_set=ToolSet.BASIC,
)

RESEARCHER_PROFILE = AppProfile(
    name="researcher",
    personality=(
        "You are a research assistant focused on gathering and analyzing information. "
        "You prioritize accuracy and cite sources when possible."
    ),
    tool_set=ToolSet.FULL,
    llm_model_tier="advanced",
)

ASSISTANT_PROFILE = AppProfile(
    name="assistant",
    personality=(
        "You are a personal assistant helping with daily tasks. "
        "You are friendly, efficient, and proactive."
    ),
    tool_set=ToolSet.BASIC,
    memory_config={"working_ttl": 600},  # 10 minute working memory
)

MINIMAL_PROFILE = AppProfile(
    name="minimal",
    personality="You answer questions directly without using tools.",
    tool_set=ToolSet.MINIMAL,
    llm_model_tier="fast",
)
