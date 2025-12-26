"""Behavior templates for common agent patterns.

Templates are pre-built behavior definitions that can be customized
for specific applications. They provide a starting point for common
agent types like voice assistants, story tellers, and NPCs.

Usage:
    from draagon_ai.behaviors.templates import VOICE_ASSISTANT_TEMPLATE

    # Customize for your application
    my_assistant = VOICE_ASSISTANT_TEMPLATE.customize(
        behavior_id="my_assistant",
        name="My Assistant",
        # Add custom actions, remove unwanted ones, etc.
    )

    # Or use the story teller for interactive fiction
    from draagon_ai.behaviors.templates import (
        STORY_TELLER_TEMPLATE,
        create_story_character,
    )
"""

from .assistant import (
    VOICE_ASSISTANT_TEMPLATE,
    create_voice_assistant_behavior,
)

from .storyteller import (
    # Templates
    STORY_TELLER_TEMPLATE,
    STORY_TELLER_CHARACTER_TEMPLATE,
    # Factory functions
    create_story_character,
    create_story_teller,
    # State types
    StoryState,
    CharacterProfile,
)

__all__ = [
    # Voice Assistant
    "VOICE_ASSISTANT_TEMPLATE",
    "create_voice_assistant_behavior",
    # Story Teller
    "STORY_TELLER_TEMPLATE",
    "STORY_TELLER_CHARACTER_TEMPLATE",
    "create_story_character",
    "create_story_teller",
    "StoryState",
    "CharacterProfile",
]
