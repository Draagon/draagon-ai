"""Story adapter for personalization.

This module provides the StoryAdapter class that can be subclassed
to provide application-specific personalization for stories.

The base adapter provides sensible defaults. Applications like Roxy
can subclass to add specific personalization (e.g., user's pets,
locations, preferences from memory).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, TYPE_CHECKING

from .story import StoryElements, NarratorStyle

if TYPE_CHECKING:
    from draagon_ai.memory import MemoryProvider

logger = logging.getLogger(__name__)


class StoryAdapter:
    """Base adapter for story personalization.

    Subclass this to provide application-specific personalization.
    The base implementation provides generic defaults.

    Example:
        class MyStoryAdapter(StoryAdapter):
            async def get_characters(self, user_id: str) -> list[dict]:
                # Load from user's pets/family
                return [{"name": "Fluffy", "role": "wise guide"}]

            def get_narrator_style(self, user_id: str) -> NarratorStyle:
                # Load from user preferences
                return NarratorStyle.WHIMSICAL
    """

    def __init__(
        self,
        memory_provider: "MemoryProvider | None" = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            memory_provider: Optional memory provider for personalization
        """
        self._memory = memory_provider
        self._story_sessions: dict[str, list[dict[str, Any]]] = {}

    async def get_story_elements(
        self,
        user_id: str,
        mood: str = "neutral",
    ) -> StoryElements:
        """Get personalization elements for a story.

        Override this to provide custom personalization.

        Args:
            user_id: User to personalize for
            mood: Current user mood

        Returns:
            StoryElements with personalization data
        """
        elements = StoryElements()

        # Set mood-appropriate themes
        elements.themes = self._get_mood_themes(mood)

        # Set time context
        elements.time_context = self._get_time_context()

        # Default characters (override to customize)
        elements.characters = await self.get_characters(user_id)

        # Default locations (override to customize)
        elements.locations = await self.get_locations(user_id)

        # Get user interests from memory if available
        if self._memory:
            try:
                memories = await self._memory.search(
                    query=f"interests hobbies preferences for {user_id}",
                    user_id=user_id,
                    limit=5,
                )
                for mem in memories:
                    if hasattr(mem, "content"):
                        elements.interests.append(str(mem.content))
            except Exception as e:
                logger.warning(f"Could not fetch interests: {e}")

        # Get past story callbacks
        if user_id in self._story_sessions:
            past = self._story_sessions[user_id][-3:]
            for story in past:
                if "memorable_moment" in story:
                    elements.callbacks.append(story["memorable_moment"])

        return elements

    async def get_characters(self, user_id: str) -> list[dict[str, str]]:
        """Get characters for stories.

        Override to provide custom characters (e.g., user's pets).

        Args:
            user_id: User to get characters for

        Returns:
            List of character dicts with name, role, description
        """
        # Default generic characters
        return [
            {"name": "Sage", "role": "wise mentor", "description": "An ancient guide"},
            {"name": "Spark", "role": "trickster companion", "description": "Mischievous helper"},
        ]

    async def get_locations(self, user_id: str) -> list[str]:
        """Get locations for story settings.

        Override to provide custom locations.

        Args:
            user_id: User to get locations for

        Returns:
            List of location descriptions
        """
        # Default fantasy locations
        return [
            "a mystical forest",
            "an ancient library",
            "a bustling marketplace",
            "a hidden cave",
        ]

    def get_narrator_style(self, user_id: str) -> NarratorStyle:
        """Get narrator style for a user.

        Override to provide user-specific styles.

        Args:
            user_id: User to get style for

        Returns:
            NarratorStyle enum value
        """
        return NarratorStyle.WARM

    def _get_mood_themes(self, mood: str) -> list[str]:
        """Get themes appropriate for mood."""
        mood_themes = {
            "stressed": ["comfort", "gentle adventure", "found family", "cozy"],
            "excited": ["action", "discovery", "magic", "quest"],
            "sad": ["hope", "friendship", "healing", "warmth"],
            "playful": ["whimsy", "humor", "silly magic", "pranks"],
            "curious": ["mystery", "exploration", "secrets", "wonder"],
            "calm": ["peaceful", "reflection", "nature", "harmony"],
            "neutral": ["adventure", "friendship", "discovery"],
        }
        return mood_themes.get(mood, mood_themes["neutral"])

    def _get_time_context(self) -> str:
        """Get time-of-day context for story themes."""
        hour = datetime.now().hour

        if 6 <= hour < 12:
            return "morning - stories can be energetic and set up adventures"
        elif 12 <= hour < 17:
            return "afternoon - good for exploration and discovery"
        elif 17 <= hour < 20:
            return "evening - winding down, comfort stories"
        else:
            return "night - dreamlike, magical, calming"

    async def record_story_moment(
        self,
        user_id: str,
        moment_type: str,
        content: str,
        memorable: bool = False,
    ) -> None:
        """Record a story moment for future callbacks.

        Args:
            user_id: User having the story
            moment_type: Type of moment (choice, twist, conclusion)
            content: What happened
            memorable: Whether to save for callbacks
        """
        if user_id not in self._story_sessions:
            self._story_sessions[user_id] = []

        moment: dict[str, Any] = {
            "type": moment_type,
            "content": content[:200],
        }

        if memorable:
            moment["memorable_moment"] = content[:100]

        self._story_sessions[user_id].append(moment)

        # Keep only last 50 moments per user
        if len(self._story_sessions[user_id]) > 50:
            self._story_sessions[user_id] = self._story_sessions[user_id][-50:]
