"""Story types and state management.

This module provides the core types for storytelling:
- StoryGenre: Available story genres
- NarratorStyle: Narrator personality styles
- StoryElements: Personalization elements for stories
- StoryBeat: A single story moment/turn
- StoryState: Complete state of an active story
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class StoryGenre(Enum):
    """Available story genres."""

    ADVENTURE = "adventure"
    MYSTERY = "mystery"
    FANTASY = "fantasy"
    SCIFI = "scifi"
    COMEDY = "comedy"
    HEARTWARMING = "heartwarming"
    HORROR = "horror"
    ROMANCE = "romance"


class NarratorStyle(Enum):
    """Narrator personality styles."""

    WARM = "warm"
    """Friendly, engaging, personal."""

    DRAMATIC = "dramatic"
    """Theatrical, heightened tension."""

    WHIMSICAL = "whimsical"
    """Playful, magical, wonder-filled."""

    MYSTERIOUS = "mysterious"
    """Enigmatic, atmospheric, suspenseful."""

    WIT = "wit"
    """Clever wordplay, dry humor, smart."""


@dataclass
class StoryElements:
    """Personalization elements for stories.

    These elements can be injected into story generation to make
    stories feel personal and grounded in the user's world.
    """

    # Characters that can appear (e.g., pets as magical creatures)
    characters: list[dict[str, str]] = field(default_factory=list)
    """List of character dicts with name, role, description."""

    # Familiar locations for story settings
    locations: list[str] = field(default_factory=list)
    """Locations the user knows (can be real or fantastical versions)."""

    # User interests for plot elements
    interests: list[str] = field(default_factory=list)
    """Topics the user cares about."""

    # Previous story callbacks
    callbacks: list[str] = field(default_factory=list)
    """References to previous stories for continuity."""

    # Mood-appropriate themes
    themes: list[str] = field(default_factory=list)
    """Suggested themes based on current mood/context."""

    # Time context
    time_context: str = "any"
    """Time of day context (morning, afternoon, evening, night)."""

    def to_prompt_text(self) -> str:
        """Format elements for injection into prompts."""
        parts = []

        if self.characters:
            char_lines = []
            for c in self.characters:
                char_lines.append(f"- {c.get('name', 'Unknown')}: {c.get('role', 'character')}")
            parts.append(f"CHARACTERS:\n" + "\n".join(char_lines))

        if self.locations:
            parts.append(f"LOCATIONS:\n" + "\n".join(f"- {loc}" for loc in self.locations))

        if self.interests:
            parts.append(f"INTERESTS:\n" + "\n".join(f"- {i}" for i in self.interests))

        if self.callbacks:
            parts.append(f"PREVIOUS STORY REFERENCES:\n" + "\n".join(f"- {cb}" for cb in self.callbacks))

        if self.themes:
            parts.append(f"SUGGESTED THEMES: {', '.join(self.themes)}")

        if self.time_context != "any":
            parts.append(f"TIME CONTEXT: {self.time_context}")

        return "\n\n".join(parts) if parts else "No personalization elements provided."


@dataclass
class StoryBeat:
    """A single moment/turn in a story.

    Represents one narrative unit - could be narration, a choice
    presented to the user, or the user's response.
    """

    beat_type: str
    """Type: opening, narration, choice, user_action, conclusion."""

    content: str
    """The narrative content."""

    user_input: str | None = None
    """User's input that led to this beat (if any)."""

    choices_offered: list[str] | None = None
    """Choices presented to user (if this is a choice beat)."""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this beat occurred."""

    memorable: bool = False
    """Whether this beat should be saved for future callbacks."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "beat_type": self.beat_type,
            "content": self.content,
            "user_input": self.user_input,
            "choices_offered": self.choices_offered,
            "timestamp": self.timestamp.isoformat(),
            "memorable": self.memorable,
        }


@dataclass
class StoryState:
    """Complete state of an active story.

    Tracks everything needed to continue a story session.
    """

    story_id: str
    """Unique identifier for this story session."""

    user_id: str
    """User who started the story."""

    genre: StoryGenre
    """Story genre."""

    narrator_style: NarratorStyle
    """Narrator personality."""

    elements: StoryElements
    """Personalization elements."""

    beats: list[StoryBeat] = field(default_factory=list)
    """History of story beats."""

    beat_count: int = 0
    """Number of beats so far."""

    mood: str = "neutral"
    """User's mood for tone adjustment."""

    started_at: datetime = field(default_factory=datetime.now)
    """When story started."""

    concluded: bool = False
    """Whether story has ended."""

    max_beats: int = 50
    """Maximum beats before auto-conclusion."""

    def add_beat(self, beat: StoryBeat) -> None:
        """Add a story beat."""
        self.beats.append(beat)
        self.beat_count += 1

    def get_last_beat(self) -> StoryBeat | None:
        """Get the most recent beat."""
        return self.beats[-1] if self.beats else None

    def should_conclude(self) -> bool:
        """Check if story should wrap up."""
        return self.beat_count >= self.max_beats

    def get_memorable_moments(self) -> list[str]:
        """Get memorable moments for future callbacks."""
        return [beat.content[:100] for beat in self.beats if beat.memorable]

    def to_context(self) -> str:
        """Format story state for LLM context."""
        recent_beats = self.beats[-5:] if len(self.beats) > 5 else self.beats

        context_parts = [
            f"Genre: {self.genre.value}",
            f"Narrator: {self.narrator_style.value}",
            f"Mood: {self.mood}",
            f"Beat: {self.beat_count}/{self.max_beats}",
            "",
            "Recent story:",
        ]

        for beat in recent_beats:
            if beat.user_input:
                context_parts.append(f"[User]: {beat.user_input}")
            context_parts.append(f"[Story]: {beat.content[:200]}")

        return "\n".join(context_parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "story_id": self.story_id,
            "user_id": self.user_id,
            "genre": self.genre.value,
            "narrator_style": self.narrator_style.value,
            "beat_count": self.beat_count,
            "mood": self.mood,
            "started_at": self.started_at.isoformat(),
            "concluded": self.concluded,
            "beats": [beat.to_dict() for beat in self.beats],
        }
