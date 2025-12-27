"""Interactive storytelling extension for draagon-ai.

This extension provides interactive fiction / text adventure capabilities,
including:
- Story Teller behavior for narrating stories
- Story Character behavior for NPCs
- Drama Manager for pacing and narrative quality
- Dynamic character generation

Example:
    from draagon_ai.extensions import get_extension_manager

    manager = get_extension_manager()
    behaviors = manager.get_all_behaviors()

    # Find the story teller behavior
    story_teller = next(
        b for b in behaviors if b.behavior_id == "story_teller"
    )
"""

from .extension import StorytellingExtension

__version__ = "0.1.0"

__all__ = [
    "StorytellingExtension",
    "__version__",
]
