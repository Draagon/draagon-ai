"""Behavior system for defining what agents CAN DO.

Behaviors are pluggable modules that define:
- Actions: The specific things an agent can do
- Triggers: When a behavior should activate
- Prompts: How the LLM should reason about actions
- Constraints: Rules and safety limits
- Tests: Validation cases
- Metrics: Performance tracking

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     BehaviorRegistry                             │
    │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────────┐   │
    │  │   CORE   │ │  ADDON   │ │   APP    │ │    GENERATED     │   │
    │  │ behaviors│ │ behaviors│ │ behaviors│ │    behaviors     │   │
    │  └──────────┘ └──────────┘ └──────────┘ └──────────────────┘   │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.behaviors import (
        Behavior,
        BehaviorRegistry,
        VOICE_ASSISTANT_TEMPLATE,
    )

    # Create a registry
    registry = BehaviorRegistry()

    # Register the voice assistant template
    registry.register(VOICE_ASSISTANT_TEMPLATE)

    # Or create a customized version
    from draagon_ai.behaviors.templates import create_voice_assistant_behavior
    my_assistant = create_voice_assistant_behavior(
        behavior_id="my_assistant",
        exclude_actions=["execute_command"],  # Disable shell commands
    )
    registry.register(my_assistant)
"""

# Core types
from .types import (
    # Enums
    ActivationScope,
    BehaviorStatus,
    BehaviorTier,
    # Action types
    Action,
    ActionExample,
    ActionMetrics,
    ActionParameter,
    # Behavior types
    Behavior,
    BehaviorConstraints,
    BehaviorMetrics,
    BehaviorPrompts,
    # Testing types
    BehaviorTestCase,
    TestOutcome,
    TestResults,
    # Trigger types
    Trigger,
    # Activation types
    ActivationContext,
    ActiveBehaviorSet,
    # Evolution types
    BehaviorEvolutionResult,
    EvolutionConfig,
    # Research types
    DomainResearchResult,
    FailureAnalysis,
    # Validation types
    ValidationIssue,
)

# Registry
from .registry import BehaviorRegistry

# Templates
from .templates import (
    # Voice Assistant
    VOICE_ASSISTANT_TEMPLATE,
    create_voice_assistant_behavior,
    # Story Teller
    STORY_TELLER_TEMPLATE,
    STORY_TELLER_CHARACTER_TEMPLATE,
    create_story_character,
    create_story_teller,
    StoryState,
    CharacterProfile,
)

__all__ = [
    # Enums
    "ActivationScope",
    "BehaviorStatus",
    "BehaviorTier",
    # Action types
    "Action",
    "ActionExample",
    "ActionMetrics",
    "ActionParameter",
    # Behavior types
    "Behavior",
    "BehaviorConstraints",
    "BehaviorMetrics",
    "BehaviorPrompts",
    # Testing types
    "BehaviorTestCase",
    "TestOutcome",
    "TestResults",
    # Trigger types
    "Trigger",
    # Activation types
    "ActivationContext",
    "ActiveBehaviorSet",
    # Evolution types
    "BehaviorEvolutionResult",
    "EvolutionConfig",
    # Research types
    "DomainResearchResult",
    "FailureAnalysis",
    # Validation types
    "ValidationIssue",
    # Registry
    "BehaviorRegistry",
    # Voice Assistant Templates
    "VOICE_ASSISTANT_TEMPLATE",
    "create_voice_assistant_behavior",
    # Story Teller Templates
    "STORY_TELLER_TEMPLATE",
    "STORY_TELLER_CHARACTER_TEMPLATE",
    "create_story_character",
    "create_story_teller",
    "StoryState",
    "CharacterProfile",
]
