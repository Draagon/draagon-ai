"""Behavior templates for common agent patterns.

Templates are pre-built behavior definitions that can be customized
for specific applications. They provide a starting point for common
agent types like voice assistants and behavior architects.

Usage:
    from draagon_ai.behaviors.templates import VOICE_ASSISTANT_TEMPLATE

    # Customize for your application
    my_assistant = VOICE_ASSISTANT_TEMPLATE.customize(
        behavior_id="my_assistant",
        name="My Assistant",
        # Add custom actions, remove unwanted ones, etc.
    )

    # Or use the Behavior Architect to create new behaviors
    from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
    from draagon_ai.services import BehaviorArchitectService

Note:
    Storytelling templates are provided by the draagon-ai-ext-storytelling
    extension. Install with: pip install draagon-ai-ext-storytelling
"""

from .assistant import (
    VOICE_ASSISTANT_TEMPLATE,
    VOICE_ASSISTANT_ACTIONS,
    VOICE_ASSISTANT_TRIGGERS,
    VOICE_ASSISTANT_CONSTRAINTS,
    VOICE_ASSISTANT_TEST_CASES,
    create_voice_assistant_behavior,
)

from .architect import (
    BEHAVIOR_ARCHITECT_TEMPLATE,
    create_behavior_architect,
)

__all__ = [
    # Voice Assistant
    "VOICE_ASSISTANT_TEMPLATE",
    "VOICE_ASSISTANT_ACTIONS",
    "VOICE_ASSISTANT_TRIGGERS",
    "VOICE_ASSISTANT_CONSTRAINTS",
    "VOICE_ASSISTANT_TEST_CASES",
    "create_voice_assistant_behavior",
    # Behavior Architect
    "BEHAVIOR_ARCHITECT_TEMPLATE",
    "create_behavior_architect",
]
