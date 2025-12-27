"""Storytelling extension implementation.

This module contains the StorytellingExtension class that registers
the storytelling behaviors with draagon-ai.
"""

from __future__ import annotations

from typing import Any

from draagon_ai.extensions import Extension, ExtensionInfo


class StorytellingExtension(Extension):
    """Interactive storytelling extension for draagon-ai.

    This extension provides:
    - Story Teller behavior for narrating interactive stories
    - Story Character behavior for NPCs with personalities
    - Drama Manager integration for pacing

    Configuration options:
    - drama_intensity: How dramatic the narration (0.0-1.0)
    - default_narrator: Which narrator persona to use
    """

    def __init__(self) -> None:
        self._drama_intensity: float = 0.7
        self._default_narrator: str = "warm"
        self._initialized: bool = False

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="storytelling",
            version="0.1.0",
            description="Interactive fiction and storytelling behaviors",
            author="draagon-ai",
            requires_core=">=0.1.0",
            provides_behaviors=[
                "story_teller",
                "story_character",
            ],
            config_schema={
                "type": "object",
                "properties": {
                    "drama_intensity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7,
                    },
                    "default_narrator": {
                        "type": "string",
                        "enum": ["warm", "mysterious", "dramatic", "sardonic"],
                        "default": "warm",
                    },
                },
            },
            homepage="https://github.com/draagon-ai/draagon-ai",
            license="MIT",
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize extension with configuration.

        Args:
            config: Extension configuration from draagon.yaml.
        """
        self._drama_intensity = config.get("drama_intensity", 0.7)
        self._default_narrator = config.get("default_narrator", "warm")
        self._initialized = True

    def shutdown(self) -> None:
        """Clean up on shutdown."""
        self._initialized = False

    def get_behaviors(self) -> list:
        """Return storytelling behaviors.

        Returns:
            List containing Story Teller and Story Character behaviors.
        """
        from .behavior import (
            STORY_TELLER_TEMPLATE,
            STORY_TELLER_CHARACTER_TEMPLATE,
        )

        return [
            STORY_TELLER_TEMPLATE,
            STORY_TELLER_CHARACTER_TEMPLATE,
        ]

    def get_services(self) -> dict[str, Any]:
        """Return storytelling services.

        Returns:
            Dict with drama manager and related services.
        """
        # Services could be added here for drama manager, etc.
        return {}
