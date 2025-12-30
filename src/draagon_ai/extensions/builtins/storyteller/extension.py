"""Storyteller Extension for draagon-ai.

This extension provides interactive storytelling capabilities.
Users can engage in text adventures, have stories told, or explore
narrative experiences with the AI as narrator.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from draagon_ai.extensions import Extension, ExtensionInfo
from draagon_ai.orchestration.registry import Tool, ToolParameter

from .story import (
    StoryState,
    StoryBeat,
    StoryElements,
    StoryGenre,
    NarratorStyle,
)
from .adapter import StoryAdapter

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Prompts for Story Generation
# =============================================================================

STORY_DECISION_PROMPT = """You are an interactive storyteller. Your task is to decide the next story beat based on the current state and user's action.

STORY CONTEXT:
{story_context}

PERSONALIZATION:
{personal_elements}

USER'S ACTION:
{user_action}

YOUR TASK: Decide the next story beat. Consider:
1. What would engage this user?
2. How can you weave in personalization naturally?
3. What tone matches their current mood?

Available narrative actions:
- describe_scene: Set the scene with vivid details
- offer_choice: Give the user 2-3 meaningful choices
- reveal_twist: Add an unexpected development
- emotional_moment: A touching or tense scene
- comic_relief: Lighten the mood
- advance_plot: Move the main story forward
- conclude: Begin wrapping up the story

Output JSON:
{{"action": "<narrative action>", "content": "<the story content - 2-4 sentences>", "choices": ["option1", "option2"] (optional, if offering choices)}}"""

STORY_OPENING_PROMPT = """You are starting an interactive story. Create an engaging opening that hooks the reader.

GENRE: {genre}
NARRATOR STYLE: {narrator_style}
USER MOOD: {mood}
PERSONALIZATION:
{personal_elements}

Create an opening that:
1. Sets the scene vividly in 2-3 sentences
2. Introduces intrigue or a call to adventure
3. Feels appropriate for the genre and time of day
4. Uses personalization naturally if available

Output JSON:
{{"opening": "<the story opening - 2-4 sentences>", "hook": "<what makes the reader want to continue>"}}"""

STORY_CONCLUSION_PROMPT = """You are concluding an interactive story. Create a satisfying ending.

STORY CONTEXT:
{story_context}

GENRE: {genre}
BEAT COUNT: {beat_count}

Create a conclusion that:
1. Wraps up the immediate adventure
2. Leaves room for future stories
3. References memorable moments if any
4. Feels emotionally satisfying

Output JSON:
{{"conclusion": "<the story conclusion - 2-4 sentences>", "callback": "<a memorable element for future stories>"}}"""


class StorytellerExtension(Extension):
    """Interactive storytelling extension.

    Provides tools for:
    - start_story: Begin a new interactive story
    - continue_story: Continue with a user choice/action
    - get_story_status: Check current story state
    - end_story: Conclude the current story

    The storyteller can adapt to:
    - User preferences and interests
    - Current mood (calm vs excited)
    - Time of day (morning adventures vs bedtime stories)
    - Custom personalization via StoryAdapter

    Configuration (draagon.yaml):
        extensions:
          storyteller:
            enabled: true
            config:
              drama_intensity: 0.7
              default_narrator: "warm"
              max_story_length: 50

    Example:
        ext = StorytellerExtension()
        ext.initialize({"drama_intensity": 0.8})

        # Optionally set custom adapter for personalization
        ext.set_adapter(MyCustomAdapter())

        tools = ext.get_tools()
    """

    def __init__(self) -> None:
        """Initialize the extension."""
        self._adapter: StoryAdapter = StoryAdapter()
        self._active_stories: dict[str, StoryState] = {}
        self._drama_intensity: float = 0.7
        self._default_narrator: NarratorStyle = NarratorStyle.WARM
        self._max_story_length: int = 50
        self._config: dict[str, Any] = {}
        self._initialized: bool = False
        self._llm_fn: Any = None

    @property
    def info(self) -> ExtensionInfo:
        """Return extension metadata."""
        return ExtensionInfo(
            name="storyteller",
            version="1.0.0",
            description="Interactive storytelling with personalization",
            author="draagon-ai",
            requires_core=">=0.1.0",
            provides_behaviors=["storyteller"],
            provides_tools=[
                "start_story",
                "continue_story",
                "get_story_status",
                "end_story",
            ],
            provides_prompt_domains=["storytelling"],
            provides_mcp_servers=[],
            config_schema={
                "type": "object",
                "properties": {
                    "drama_intensity": {
                        "type": "number",
                        "description": "How dramatic stories are (0.0-1.0)",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "default_narrator": {
                        "type": "string",
                        "description": "Default narrator style",
                        "enum": ["warm", "dramatic", "whimsical", "mysterious", "wit"],
                        "default": "warm",
                    },
                    "max_story_length": {
                        "type": "integer",
                        "description": "Max beats before auto-conclusion",
                        "default": 50,
                    },
                },
            },
        )

    def initialize(self, config: dict[str, Any]) -> None:
        """Initialize with configuration.

        Args:
            config: Extension configuration
        """
        self._config = config
        self._drama_intensity = config.get("drama_intensity", 0.7)
        self._max_story_length = config.get("max_story_length", 50)

        narrator_str = config.get("default_narrator", "warm")
        try:
            self._default_narrator = NarratorStyle(narrator_str)
        except ValueError:
            self._default_narrator = NarratorStyle.WARM

        self._initialized = True
        logger.info("StorytellerExtension initialized")

    def set_adapter(self, adapter: StoryAdapter) -> None:
        """Set custom story adapter for personalization.

        Args:
            adapter: StoryAdapter subclass with custom personalization
        """
        self._adapter = adapter

    def set_llm_function(self, llm_fn: Any) -> None:
        """Set LLM function for story generation.

        Args:
            llm_fn: Async function(prompt, system) -> str for generation
        """
        self._llm_fn = llm_fn

    async def _start_story(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Start a new interactive story.

        Args:
            args: Tool arguments (genre, mood)
            context: Execution context (user_id)

        Returns:
            Story opening and state info
        """
        context = context or {}
        user_id = context.get("user_id", "unknown")

        # Parse genre
        genre_str = args.get("genre", "adventure").lower()
        try:
            genre = StoryGenre(genre_str)
        except ValueError:
            genre = StoryGenre.ADVENTURE

        mood = args.get("mood", "neutral")

        # Get personalization elements
        elements = await self._adapter.get_story_elements(user_id, mood)
        narrator_style = self._adapter.get_narrator_style(user_id)

        # Create story state
        story_id = str(uuid.uuid4())
        story = StoryState(
            story_id=story_id,
            user_id=user_id,
            genre=genre,
            narrator_style=narrator_style,
            elements=elements,
            mood=mood,
            max_beats=self._max_story_length,
        )

        # Generate opening
        opening = await self._generate_opening(story, elements)

        # Add opening beat
        story.add_beat(StoryBeat(
            beat_type="opening",
            content=opening,
            memorable=True,
        ))

        # Store active story
        self._active_stories[user_id] = story

        return {
            "success": True,
            "story_started": True,
            "story_id": story_id,
            "opening": opening,
            "genre": genre.value,
            "narrator_style": narrator_style.value,
        }

    async def _continue_story(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Continue the story with a user choice/action.

        Args:
            args: Tool arguments (choice, action)
            context: Execution context (user_id)

        Returns:
            Next story beat
        """
        context = context or {}
        user_id = context.get("user_id", "unknown")

        if user_id not in self._active_stories:
            return {
                "success": False,
                "error": "No active story. Start one with 'tell me a story'.",
            }

        story = self._active_stories[user_id]
        user_input = args.get("choice") or args.get("action") or ""

        # Generate next beat
        next_content = await self._generate_beat(story, user_input)

        # Mark every 5th beat as memorable
        memorable = story.beat_count % 5 == 0

        # Add beat
        story.add_beat(StoryBeat(
            beat_type="narration",
            content=next_content,
            user_input=user_input,
            memorable=memorable,
        ))

        # Record for future callbacks
        if memorable:
            await self._adapter.record_story_moment(
                user_id,
                moment_type="choice",
                content=next_content,
                memorable=True,
            )

        return {
            "success": True,
            "story_beat": next_content,
            "beat_number": story.beat_count,
            "should_conclude": story.should_conclude(),
        }

    async def _get_story_status(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Get status of current story.

        Args:
            args: Tool arguments (unused)
            context: Execution context (user_id)

        Returns:
            Story status info
        """
        context = context or {}
        user_id = context.get("user_id", "unknown")

        if user_id not in self._active_stories:
            return {
                "active": False,
                "message": "No story in progress. Say 'tell me a story' to start!",
            }

        story = self._active_stories[user_id]
        last_beat = story.get_last_beat()

        return {
            "active": True,
            "story_id": story.story_id,
            "genre": story.genre.value,
            "beat_count": story.beat_count,
            "started_at": story.started_at.isoformat(),
            "last_beat": last_beat.content if last_beat else None,
        }

    async def _end_story(
        self,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """End the current story with a conclusion.

        Args:
            args: Tool arguments (unused)
            context: Execution context (user_id)

        Returns:
            Story conclusion
        """
        context = context or {}
        user_id = context.get("user_id", "unknown")

        if user_id not in self._active_stories:
            return {
                "success": False,
                "error": "No story to end.",
            }

        story = self._active_stories[user_id]

        # Generate conclusion
        conclusion = await self._generate_conclusion(story)

        # Add conclusion beat
        story.add_beat(StoryBeat(
            beat_type="conclusion",
            content=conclusion,
            memorable=True,
        ))
        story.concluded = True

        # Record memorable moment
        await self._adapter.record_story_moment(
            user_id,
            moment_type="conclusion",
            content=conclusion,
            memorable=True,
        )

        # Clean up
        total_beats = story.beat_count
        del self._active_stories[user_id]

        return {
            "success": True,
            "conclusion": conclusion,
            "total_beats": total_beats,
        }

    async def _generate_opening(
        self,
        story: StoryState,
        elements: StoryElements,
    ) -> str:
        """Generate story opening."""
        if self._llm_fn:
            try:
                prompt = STORY_OPENING_PROMPT.format(
                    genre=story.genre.value,
                    narrator_style=story.narrator_style.value,
                    mood=story.mood,
                    personal_elements=elements.to_prompt_text(),
                )
                result = await self._llm_fn(prompt, "You are a master storyteller.")
                if isinstance(result, dict) and "opening" in result:
                    return result["opening"]
                elif isinstance(result, str):
                    return result
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback generic openings
        openings = {
            StoryGenre.ADVENTURE: (
                "The morning sun cast long shadows across the path ahead. "
                "Something stirred in the air - today would be different. "
                "An adventure awaited, calling out to those brave enough to answer."
            ),
            StoryGenre.MYSTERY: (
                "It started with a curious note, found tucked beneath the door. "
                "The message was simple but unsettling: 'They know what you found.' "
                "Some mysteries are better left unsolved - but this one demanded answers."
            ),
            StoryGenre.FANTASY: (
                "The portal shimmered into existence where nothing had been before. "
                "Through it, you could see a world where the impossible was merely improbable. "
                "Magic hung in the air like morning mist, waiting to be discovered."
            ),
            StoryGenre.HEARTWARMING: (
                "Some days feel like the center of the universe. "
                "Today was one of those days - a day for small miracles and unexpected connections. "
                "The kind of day that changes everything, in the gentlest way possible."
            ),
        }
        return openings.get(story.genre, openings[StoryGenre.ADVENTURE])

    async def _generate_beat(
        self,
        story: StoryState,
        user_input: str,
    ) -> str:
        """Generate the next story beat."""
        if self._llm_fn:
            try:
                prompt = STORY_DECISION_PROMPT.format(
                    story_context=story.to_context(),
                    personal_elements=story.elements.to_prompt_text(),
                    user_action=user_input or "continue",
                )
                result = await self._llm_fn(prompt, "You are a master storyteller.")
                if isinstance(result, dict) and "content" in result:
                    return result["content"]
                elif isinstance(result, str):
                    return result
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback simple responses
        user_lower = user_input.lower() if user_input else ""

        if "left" in user_lower:
            return (
                "You choose the path to the left, where ancient trees grow thick. "
                "Mysterious sounds echo through the canopy above. "
                "What awaits in the shadows?"
            )
        elif "right" in user_lower:
            return (
                "The right path leads toward a clearing where something glimmers. "
                "As you approach, the light grows stronger. "
                "A discovery waits to be made."
            )
        elif "wait" in user_lower:
            return (
                "You pause, listening carefully to the world around you. "
                "In the silence, a faint melody drifts through the air. "
                "Sometimes patience reveals what haste would miss."
            )
        else:
            return (
                f"You decide to {user_input.lower() if user_input else 'press forward'}. "
                "The adventure continues to unfold before you. "
                "What happens next is up to you."
            )

    async def _generate_conclusion(self, story: StoryState) -> str:
        """Generate story conclusion."""
        if self._llm_fn:
            try:
                prompt = STORY_CONCLUSION_PROMPT.format(
                    story_context=story.to_context(),
                    genre=story.genre.value,
                    beat_count=story.beat_count,
                )
                result = await self._llm_fn(prompt, "You are a master storyteller.")
                if isinstance(result, dict) and "conclusion" in result:
                    return result["conclusion"]
                elif isinstance(result, str):
                    return result
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")

        # Fallback conclusions
        if story.beat_count < 5:
            return (
                "And so ends this brief but memorable tale. "
                "Perhaps we'll continue another time?"
            )
        elif story.genre == StoryGenre.MYSTERY:
            return (
                "The mystery was solved, but you knew this was only the beginning. "
                "Some secrets run deeper than others. "
                "Until next time, keep your eyes open."
            )
        else:
            return (
                "As the sun set on this adventure, you knew these moments would be treasured. "
                "Every ending is just a new beginning in disguise. "
                "The end... or is it?"
            )

    def get_tools(self) -> list[Tool]:
        """Return tools provided by this extension."""
        return [
            Tool(
                name="start_story",
                description="Start a new interactive story adventure",
                handler=self._start_story,
                parameters=[
                    ToolParameter(
                        name="genre",
                        type="string",
                        description="Story genre: adventure, mystery, fantasy, scifi, comedy, heartwarming",
                        required=False,
                        default="adventure",
                    ),
                    ToolParameter(
                        name="mood",
                        type="string",
                        description="User's mood to adapt tone: excited, calm, playful, curious, sad",
                        required=False,
                        default="neutral",
                    ),
                ],
            ),
            Tool(
                name="continue_story",
                description="Continue the story with a choice or action",
                handler=self._continue_story,
                parameters=[
                    ToolParameter(
                        name="choice",
                        type="string",
                        description="The user's choice or decision",
                        required=False,
                    ),
                    ToolParameter(
                        name="action",
                        type="string",
                        description="The action the user wants to take",
                        required=False,
                    ),
                ],
            ),
            Tool(
                name="get_story_status",
                description="Check the status of the current story",
                handler=self._get_story_status,
                parameters=[],
            ),
            Tool(
                name="end_story",
                description="End the current story with a satisfying conclusion",
                handler=self._end_story,
                parameters=[],
            ),
        ]

    def get_prompt_domains(self) -> dict[str, dict[str, str]]:
        """Return prompt domains for this extension."""
        return {
            "storytelling": {
                "STORY_DECISION_PROMPT": STORY_DECISION_PROMPT,
                "STORY_OPENING_PROMPT": STORY_OPENING_PROMPT,
                "STORY_CONCLUSION_PROMPT": STORY_CONCLUSION_PROMPT,
            }
        }

    def shutdown(self) -> None:
        """Clean up resources."""
        self._active_stories.clear()
        self._initialized = False
