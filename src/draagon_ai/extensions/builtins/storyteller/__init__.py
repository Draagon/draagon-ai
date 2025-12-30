"""Storyteller Extension for draagon-ai.

Provides interactive storytelling capabilities for AI assistants.
Users can engage in text adventures, have stories told, or explore
narrative experiences with the AI as narrator.

Features:
- Interactive text adventures with user choices
- Multiple story genres (adventure, mystery, fantasy, etc.)
- Story state management with history tracking
- Mood-aware storytelling
- Time-context themes (morning adventures, bedtime stories)
- Extensible via adapters for personalization

Configuration (draagon.yaml):
    extensions:
      storyteller:
        enabled: true
        config:
          drama_intensity: 0.7
          default_narrator: "warm"
          max_story_length: 50

Usage:
    from draagon_ai.extensions.builtins.storyteller import (
        StorytellerExtension,
        StoryElements,
        StoryGenre,
    )

    ext = StorytellerExtension()
    ext.initialize({"drama_intensity": 0.8})
    tools = ext.get_tools()
"""

from .extension import StorytellerExtension
from .story import (
    StoryState,
    StoryElements,
    StoryBeat,
    StoryGenre,
    NarratorStyle,
)
from .adapter import StoryAdapter

__all__ = [
    # Extension
    "StorytellerExtension",
    # Story types
    "StoryState",
    "StoryElements",
    "StoryBeat",
    "StoryGenre",
    "NarratorStyle",
    # Adapter
    "StoryAdapter",
]
