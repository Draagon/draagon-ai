"""Story Teller behavior templates for interactive fiction.

This module provides behaviors for running interactive story games:
- STORY_TELLER_TEMPLATE: The narrator who describes scenes, presents choices,
  and manages the story flow
- STORY_TELLER_CHARACTER_TEMPLATE: NPCs who have their own personalities,
  goals, and can interact with the player

Game Structure:
- Single player (the user) is the protagonist
- Story Teller narrates, describes, and presents choices
- NPCs (Story Teller Characters) populate the world
- Player actions drive the story forward
- Rich narrative prose, not terse responses

Example:
    from draagon_ai.behaviors.templates import (
        STORY_TELLER_TEMPLATE,
        create_story_character,
    )
    from draagon_ai.orchestration import Agent, MultiAgent

    # Create the main story teller
    narrator = Agent(
        behavior=STORY_TELLER_TEMPLATE,
        config=AgentConfig(agent_id="narrator"),
        llm=llm_provider,
        memory=memory_provider,
    )

    # Create an NPC
    wizard = create_story_character(
        character_id="old_wizard",
        name="Aldric the Wise",
        personality="Mysterious, speaks in riddles, secretly caring",
        backstory="Former court wizard, now lives in exile",
        goals=["Protect the ancient secrets", "Find a worthy successor"],
    )

    # Run the game
    agents = MultiAgent(default_agent=narrator)
    agents.add_agent(wizard_agent)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


from ..types import (
    Action,
    ActionParameter,
    ActionExample,
    Behavior,
    BehaviorPrompts,
    BehaviorTier,
    BehaviorStatus,
    BehaviorTestCase,
    Trigger,
)


# =============================================================================
# Memory Scoping - What persists vs what's session-only
# =============================================================================


class MemoryScope(Enum):
    """Defines how long different types of story data persist.

    This is CRITICAL for not polluting the user's memory with ephemeral
    story content while preserving valuable learnings about preferences.
    """

    # Session-only: Cleared when story ends
    SESSION = "session"

    # Persistent: Kept across all stories
    PERSISTENT = "persistent"

    # Optional: User can choose to remember ("That was epic! Remember this.")
    MEMORABLE = "memorable"


@dataclass
class StoryMemoryPolicy:
    """Defines what gets persisted vs cleared after a story.

    The story teller should use this to scope memory operations.
    """

    # SESSION scope - cleared when story ends
    session_data: list[str] = field(default_factory=lambda: [
        "character_profiles",      # NPC personalities, relationships
        "story_state",             # Current location, inventory, plot flags
        "story_events",            # What happened in this story
        "dialogue_history",        # Conversations with NPCs
        "player_choices",          # Decisions made in this story
    ])

    # PERSISTENT scope - kept forever
    persistent_data: list[str] = field(default_factory=lambda: [
        "theme_preferences",       # "User likes horror stories"
        "play_style",              # "Prefers detailed descriptions"
        "favorite_genres",         # ["mystery", "fantasy"]
        "disliked_elements",       # ["gore", "romance"]
        "narrator_relationship",   # How user feels about the narrator
        "game_count",              # How many stories we've played
    ])

    # MEMORABLE scope - user explicitly asks to remember
    memorable_triggers: list[str] = field(default_factory=lambda: [
        "remember this",
        "that was epic",
        "save this moment",
        "don't forget",
        "I want to remember",
    ])


# =============================================================================
# Narrator Persona - The Story Teller's Personality
# =============================================================================


@dataclass
class NarratorPersona:
    """The personality of the story teller itself.

    Like Roxy has RoxySelf, the narrator has its own personality that
    colors how it tells stories. This can be customized per genre.
    """

    # Identity
    narrator_id: str = "default_narrator"
    name: str = "The Narrator"
    voice: str = "warm"  # warm, mysterious, dramatic, whimsical, sardonic

    # Personality traits (0.0 - 1.0 scale)
    dramatic_flair: float = 0.7       # How theatrical descriptions are
    humor_level: float = 0.3          # How much wit/comedy
    darkness_tolerance: float = 0.5   # Willingness to go dark
    verbosity: float = 0.6            # Description length preference
    player_advocacy: float = 0.8      # How much narrator "roots for" player

    # Narrative preferences
    preferred_pacing: str = "balanced"  # fast, balanced, deliberate
    description_style: str = "sensory"  # sensory, emotional, action-focused
    dialogue_style: str = "naturalistic"  # naturalistic, theatrical, terse

    # Quirks and signature elements
    signature_phrases: list[str] = field(default_factory=list)
    narrative_quirks: list[str] = field(default_factory=list)

    # Genre specializations (narrator is better at some genres)
    genre_affinities: dict[str, float] = field(default_factory=lambda: {
        "fantasy": 0.8,
        "mystery": 0.7,
        "horror": 0.6,
        "romance": 0.5,
        "scifi": 0.7,
    })


# Pre-defined narrator personas for different moods
NARRATOR_PERSONAS = {
    "warm": NarratorPersona(
        narrator_id="warm_narrator",
        name="The Fireside Narrator",
        voice="warm",
        dramatic_flair=0.5,
        humor_level=0.4,
        player_advocacy=0.9,
        signature_phrases=["Dear traveler", "And so it was that"],
    ),
    "mysterious": NarratorPersona(
        narrator_id="mysterious_narrator",
        name="The Shadow Keeper",
        voice="mysterious",
        dramatic_flair=0.8,
        humor_level=0.1,
        darkness_tolerance=0.7,
        signature_phrases=["In the spaces between moments", "What lurks beneath"],
    ),
    "dramatic": NarratorPersona(
        narrator_id="dramatic_narrator",
        name="The Grand Chronicler",
        voice="dramatic",
        dramatic_flair=0.95,
        verbosity=0.8,
        signature_phrases=["Thus begins our tale!", "The very fabric of fate trembled"],
    ),
    "sardonic": NarratorPersona(
        narrator_id="sardonic_narrator",
        name="The Wry Observer",
        voice="sardonic",
        dramatic_flair=0.4,
        humor_level=0.8,
        player_advocacy=0.6,
        signature_phrases=["Well, that happened", "As one does"],
    ),
}


# =============================================================================
# Drama Manager - Narrative Pacing & Experience Management
# =============================================================================


@dataclass
class NarrativeGoal:
    """A story goal the drama manager is working toward.

    Based on research: Drama managers guide narrative without direct control,
    manipulating environment and pacing to create compelling experiences.
    """

    goal_id: str
    description: str
    goal_type: str  # "revelation", "confrontation", "relationship", "discovery", "climax"
    priority: int = 50  # 0-100
    prerequisites: list[str] = field(default_factory=list)  # Other goal IDs that must happen first
    is_achieved: bool = False
    is_blocked: bool = False

    # Flexibility - can this goal be adapted if player goes off-script?
    is_flexible: bool = True
    alternative_paths: list[str] = field(default_factory=list)


@dataclass
class StoryArc:
    """Three-act structure for the story.

    Research shows LLMs struggle with long-term coherence - explicit
    arc tracking helps maintain narrative structure.
    """

    arc_id: str
    title: str

    # Three-act structure
    act: int = 1  # 1=setup, 2=confrontation, 3=resolution

    # Act-specific goals
    act_1_goals: list[NarrativeGoal] = field(default_factory=list)  # Setup, introduce conflict
    act_2_goals: list[NarrativeGoal] = field(default_factory=list)  # Rising action, complications
    act_3_goals: list[NarrativeGoal] = field(default_factory=list)  # Climax, resolution

    # Pacing tracking
    tension_level: float = 0.3  # 0.0 (calm) to 1.0 (climax)
    scenes_in_current_act: int = 0
    target_scenes_per_act: int = 5

    # Player engagement signals
    player_engagement: float = 0.5  # Estimated from response patterns
    last_major_choice: str = ""
    choices_since_last_consequence: int = 0


@dataclass
class DramaManagerState:
    """State for the drama manager / experience manager.

    The drama manager is an "intelligent, omniscient, and disembodied agent"
    that monitors the story and intervenes to maintain quality experience.
    """

    # Current narrative goals
    active_goals: list[NarrativeGoal] = field(default_factory=list)
    completed_goals: list[str] = field(default_factory=list)

    # Story arc tracking
    current_arc: StoryArc | None = None

    # Pacing management
    scenes_since_action: int = 0      # Avoid too much quiet time
    scenes_since_relief: int = 0      # Avoid too much tension
    consecutive_failures: int = 0     # Player struggling? Ease up
    consecutive_successes: int = 0    # Too easy? Add challenge

    # Character tracking (for coherence)
    active_characters: list[str] = field(default_factory=list)
    character_last_seen: dict[str, int] = field(default_factory=dict)  # char_id -> scenes ago

    # Player behavior patterns
    player_prefers_action: bool = False
    player_prefers_dialogue: bool = False
    player_prefers_exploration: bool = False

    # Intervention flags
    needs_plot_advancement: bool = False
    needs_tension_relief: bool = False
    needs_character_development: bool = False
    needs_choice_consequence: bool = False


# =============================================================================
# Dynamic Character Generation
# =============================================================================


@dataclass
class GeneratedCharacterRequest:
    """Request for the LLM to generate a new NPC dynamically.

    The story teller can request new characters as needed, and they're
    generated with coherent personalities that fit the story.
    """

    role_needed: str  # "ally", "antagonist", "mentor", "obstacle", "comic_relief"
    context: str      # Current story situation
    constraints: list[str] = field(default_factory=list)  # "must be elderly", "cannot be human"

    # Story fit
    should_contrast_with: list[str] = field(default_factory=list)  # Character IDs
    should_complement: list[str] = field(default_factory=list)

    # Narrative purpose
    narrative_function: str = ""  # "provide key information", "create obstacle", etc.
    estimated_importance: str = "minor"  # "minor", "supporting", "major"


CHARACTER_GENERATION_PROMPT = '''Generate a compelling NPC for this interactive story.

STORY CONTEXT:
{story_context}

ROLE NEEDED: {role_needed}
NARRATIVE FUNCTION: {narrative_function}
IMPORTANCE: {importance}

CONSTRAINTS:
{constraints}

EXISTING CHARACTERS (for contrast/complement):
{existing_characters}

Generate a character with:
1. A memorable name that fits the setting
2. A distinct personality (2-3 core traits)
3. A speech style that reflects their personality
4. A secret or hidden depth (even minor characters have layers)
5. A personal goal (what do THEY want?)
6. A potential connection to the main plot

Respond in this format:
<character>
<name>Character Name</name>
<personality>Core traits description</personality>
<appearance>Brief physical description</appearance>
<speech_style>How they talk</speech_style>
<secret>Hidden aspect</secret>
<goal>What they want</goal>
<plot_hook>How they could connect to the story</plot_hook>
</character>
'''


# =============================================================================
# Story State Types
# =============================================================================


@dataclass
class StoryState:
    """Current state of the interactive story.

    This is tracked in memory and passed to the narrator for context.

    MEMORY SCOPING:
    - This entire object is SESSION scope (cleared when story ends)
    - User preferences are extracted and stored PERSISTENTLY separately
    - Memorable moments can be promoted to MEMORABLE scope on request
    """

    # === SESSION ID ===
    # Unique ID for this story session (for memory scoping)
    session_id: str = ""

    # === STORY PHASE ===
    # Phase tracks where we are in the story lifecycle
    phase: str = "theme_discovery"  # theme_discovery, character_creation, active_story, epilogue

    # Theme information (set during theme_discovery phase)
    theme: str = ""  # e.g., "mystery", "romance", "adventure", "horror"
    setting: str = ""  # e.g., "medieval fantasy", "cyberpunk city", "haunted mansion"
    tone: str = ""  # e.g., "serious", "lighthearted", "dark", "whimsical"
    central_conflict: str = ""  # The main problem/goal of the story
    theme_source: str = ""  # "user_choice", "personalized", "surprise"

    # User preferences used for personalization (populated from PERSISTENT memory)
    user_preferences: list[str] = field(default_factory=list)
    user_interests: list[str] = field(default_factory=list)

    # === NARRATOR STATE ===
    # The narrator's personality for this story
    narrator_persona_id: str = "warm"  # Key into NARRATOR_PERSONAS
    narrator_voice: str = ""  # Cached description for prompts

    # === DRAMA MANAGER STATE ===
    # Tracks narrative pacing and goals
    drama_manager: DramaManagerState = field(default_factory=DramaManagerState)

    # === WORLD STATE ===
    current_location: str = "unknown"
    location_description: str = ""
    time_of_day: str = "day"  # morning, afternoon, evening, night
    weather: str = "clear"
    chapter: int = 1
    chapter_title: str = ""

    # Player state
    player_name: str = "Adventurer"
    player_description: str = ""
    inventory: list[str] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=dict)
    health: str = "healthy"  # healthy, injured, critical

    # Plot tracking
    plot_flags: dict[str, bool] = field(default_factory=dict)
    active_quests: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    secrets_discovered: list[str] = field(default_factory=list)

    # Relationships (-100 hostile to +100 devoted)
    character_relationships: dict[str, int] = field(default_factory=dict)

    # === ACTIVE CHARACTERS ===
    # NPCs currently in the story (SESSION scope - cleared on story end)
    active_npcs: dict[str, "CharacterProfile"] = field(default_factory=dict)

    # Recent history for context
    recent_events: list[str] = field(default_factory=list)
    last_choice_made: str = ""

    # === MEMORABLE MOMENTS ===
    # Moments the user might want to remember (can be promoted to MEMORABLE scope)
    potential_memories: list[str] = field(default_factory=list)


@dataclass
class CharacterProfile:
    """Profile for an NPC in the story."""

    character_id: str
    name: str
    personality: str  # Brief personality description
    backstory: str  # Character's history
    appearance: str = ""  # Physical description
    speech_style: str = ""  # How they talk
    goals: list[str] = field(default_factory=list)
    secrets: list[str] = field(default_factory=list)  # Hidden from player initially
    current_mood: str = "neutral"
    location: str = "unknown"

    # Relationship with player
    trust_level: int = 0  # -100 to 100
    met_player: bool = False
    interactions: list[str] = field(default_factory=list)


# =============================================================================
# Story Teller Decision Prompt
# =============================================================================


STORY_TELLER_DECISION_PROMPT = '''You are the narrator of an interactive story game.

=== YOUR NARRATOR PERSONA ===
{narrator_persona}

Embody this narrative voice in your descriptions. Use your signature phrases naturally.
Your dramatic flair, humor, and pacing should reflect these personality settings.

=== STORY STATE ===
{story_state}

=== DRAMA MANAGER STATE ===
{drama_manager_state}

=== AVAILABLE ACTIONS ===
{actions}

=== PLAYER INPUT ===
"{query}"

=== PHASE-SPECIFIC BEHAVIOR ===

**If phase is "theme_discovery":**
This is the BEGINNING of a new story session. Your job is to figure out what kind of story to tell.

1. If the player says they want to play/start but hasn't specified a theme:
   → Use `ask_theme_preference` to warmly ask what kind of story they'd like
   → Present diverse options: mystery, adventure, romance, horror, sci-fi, fantasy, etc.
   → ALSO offer: "surprise me" or "you pick based on what you know about me"

2. If the player specifies a theme/genre (e.g., "mystery", "something scary", "fantasy adventure"):
   → Use `set_story_theme` with their choice
   → This transitions to active_story phase

3. If the player says "surprise me", "you decide", "based on my preferences", or similar:
   → Use `generate_personalized_theme`
   → This will search user memories for preferences, optionally do web research for fresh ideas
   → Then create a unique, personalized story concept

4. If the player asks what themes are available:
   → Use `list_theme_options` to describe various story types engagingly

**If phase is "active_story":**
Now you are the narrator. Apply the following research-based best practices:

=== NARRATIVE BEST PRACTICES (Research-Based) ===

**1. PLAYER AGENCY (Critical):**
- HONOR player choices - never invalidate what they decide to do
- Choices should have MEANINGFUL consequences that ripple forward
- Avoid "but thou must" situations - if only one path works, don't offer fake choices
- When player goes "off-script", ADAPT the story rather than blocking them
- Trust players to handle complexity - don't over-explain

**2. THREE-ACT STRUCTURE:**
Current Act: {current_act}
- Act 1 (Setup): Establish world, characters, central conflict. ~25% of story.
- Act 2 (Confrontation): Rising action, complications, character development. ~50% of story.
- Act 3 (Resolution): Climax, resolution, denouement. ~25% of story.

Track scenes_in_current_act vs target_scenes_per_act to pace act transitions.

**3. PACING MANAGEMENT (Drama Manager):**
Check drama_manager_state for intervention flags:
- `needs_plot_advancement`: True if story is stagnating. Introduce a plot beat.
- `needs_tension_relief`: True if too much tension. Add humor, beauty, or small victory.
- `needs_character_development`: True if characters feel flat. Add NPC depth.
- `needs_choice_consequence`: True if choices haven't mattered lately. Show a callback.

Pacing rules:
- scenes_since_action > 3: Something needs to HAPPEN (threat, discovery, confrontation)
- scenes_since_relief > 4: Player needs a BREATHER (humor, beauty, connection, rest)
- consecutive_failures >= 3: Player is struggling. Offer help or ease difficulty.
- consecutive_successes >= 4: Player is cruising. Add a complication.

**4. CHARACTER COHERENCE:**
- NPCs have their own GOALS that persist across scenes
- Track character_last_seen to reintroduce forgotten characters naturally
- Active characters should occasionally act on their own motivations
- When generating new NPCs (via generate_character action), give them:
  * A memorable distinctive trait
  * A personal goal that could conflict/align with player
  * A secret or hidden depth
  * A speech style that reflects their personality

**5. QUALITY-BASED NARRATIVE:**
Think in terms of "storylets" - atomic story beats that are unlocked by conditions:
- When player has item X → unlock investigation scene
- When relationship with NPC > 50 → unlock trust scene
- When tension > 0.7 → unlock confrontation option
The story emerges from available storylets, not forced linear progression.

**6. HANDLING PLAYER DEVIATION:**
When player does something unexpected:
- NEVER say "you can't do that" unless truly impossible
- Find ways to YES-AND their idea into the narrative
- Unexpected actions can create the most memorable moments
- If their action would break the story, redirect consequences creatively

**7. SHOW DON'T TELL:**
- Use vivid sensory details (sight, sound, smell, texture, taste)
- Character emotions through action beats, not labels
- Let subtext carry weight - what's NOT said matters
- Describe through the lens of your narrator persona

=== DECISION PROCESS ===

1. What is the player trying to do? (Parse intent)
2. Check drama_manager_state - are any interventions needed?
3. What should happen narratively that serves both:
   - The player's intent
   - The story's health (pacing, arc, character development)
4. Which action best serves both?

=== DYNAMIC CHARACTER GENERATION ===
If the story needs a new character and none in active_npcs fit:
Consider using `generate_character` action with:
- role_needed: What narrative role (ally, obstacle, mentor, comic_relief)
- narrative_function: What purpose they serve
- context: Current story situation for fitting them in

=== RESPONSE FORMAT ===

Respond in this XML format:
<action>{action_name}</action>
<parameters>{json_parameters}</parameters>
<narrative_notes>Brief notes on tone/atmosphere for this moment</narrative_notes>
<drama_update>Optional: changes to tension_level, act, or intervention flags</drama_update>
'''


STORY_TELLER_SYNTHESIS_PROMPT = '''You are narrating an interactive story. Based on the action taken and its result, write the narrative response.

=== YOUR NARRATOR PERSONA ===
{narrator_persona}

Write in this voice. Your dramatic_flair ({dramatic_flair}), humor_level ({humor_level}),
and verbosity ({verbosity}) should shape your prose. Use signature phrases naturally.

=== CONTEXT ===
ACTION: {action}
RESULT: {result}
NARRATIVE NOTES: {notes}
STORY STATE: {story_state}
CURRENT ACT: {current_act} (tension_level: {tension_level})

=== WRITING STYLE ===

**Core Principles:**
- Second person present tense ("You see...", "You feel...")
- SHOW don't tell - sensory details over labels
- Prose length matches your verbosity setting (scale 0-1):
  * 0.0-0.3: Punchy, 1-2 paragraphs, spare prose
  * 0.4-0.6: Balanced, 2-3 paragraphs
  * 0.7-1.0: Rich, 3-4 paragraphs, atmospheric

**Sensory Immersion:**
- Sight: Colors, lighting, movement, detail
- Sound: Ambient, sudden, dialogue tones
- Smell: Environmental, emotional triggers
- Touch/Texture: Temperature, surface, sensation
- Taste: When relevant, adds intimacy

**Character Dialogue:**
- Dialogue in quotes with action beats
- Each character has distinct speech patterns
- Subtext > exposition (what they DON'T say matters)
- Show emotion through action, not adverbs

**Pacing Awareness:**
Based on current tension_level ({tension_level}):
- Low (0.0-0.3): Languid descriptions, world-building, character moments
- Medium (0.4-0.6): Balanced tension, purposeful movement
- High (0.7-0.9): Shorter sentences, urgency, stakes
- Critical (0.9+): Clipped prose, immediate danger, no time to breathe

**Ending Each Scene:**
- End with forward momentum (what's next? what's at stake?)
- OR end with a meaningful choice

**When Presenting Choices:**
- 2-4 clear options that feel MEANINGFULLY different
- Include obvious options AND creative/unexpected ones
- Hint at stakes without over-explaining consequences
- Format: [1] Option one [2] Option two etc.

**Memorable Moments:**
If something epic, touching, or surprising happens, note it in the response
so it can be offered as a "memorable moment" the player might want to save.

Write the narrative response in your narrator's voice:
'''


# =============================================================================
# Story Teller Actions
# =============================================================================


STORY_TELLER_ACTIONS = [
    # === THEME DISCOVERY PHASE ===
    # These actions are used at the START of a story session to determine what kind of story to tell

    Action(
        name="ask_theme_preference",
        description="Ask the player what kind of story they want. Use when starting fresh and player hasn't specified a theme. Present options warmly and include 'surprise me' option.",
        parameters={
            "greeting_style": ActionParameter(
                name="greeting_style",
                description="How to greet: 'warm', 'mysterious', 'excited', 'casual'",
                type="string",
                required=False,
            ),
            "suggested_themes": ActionParameter(
                name="suggested_themes",
                description="List of 4-6 theme suggestions to offer (e.g., ['mystery', 'fantasy adventure', 'sci-fi thriller'])",
                type="list",
                required=False,
            ),
        },
        triggers=[
            "let's play",
            "tell me a story",
            "start a game",
            "new story",
        ],
        examples=[
            ActionExample(
                user_query="Let's play a story game",
                action_call={
                    "name": "ask_theme_preference",
                    "args": {
                        "greeting_style": "warm",
                        "suggested_themes": ["mystery", "fantasy adventure", "sci-fi thriller", "cozy romance", "horror", "historical drama"],
                    },
                },
                expected_outcome="Narrator warmly asks what kind of story, presents options, offers 'surprise me'",
            ),
        ],
    ),

    Action(
        name="list_theme_options",
        description="Describe available story themes/genres in an engaging way. Use when player asks what options are available.",
        parameters={
            "detail_level": ActionParameter(
                name="detail_level",
                description="How much to describe each: 'brief', 'detailed', 'with_examples'",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "what kinds of stories",
            "what options",
            "what genres",
        ],
    ),

    Action(
        name="set_story_theme",
        description="Set the story theme based on player's explicit choice. Transitions from theme_discovery to active_story phase.",
        parameters={
            "theme": ActionParameter(
                name="theme",
                description="The chosen theme/genre (e.g., 'mystery', 'fantasy', 'horror')",
                type="string",
                required=True,
            ),
            "setting": ActionParameter(
                name="setting",
                description="World setting (e.g., 'Victorian London', 'enchanted forest', 'space station')",
                type="string",
                required=False,
            ),
            "tone": ActionParameter(
                name="tone",
                description="Story tone: 'serious', 'lighthearted', 'dark', 'whimsical', 'gritty'",
                type="string",
                required=False,
            ),
            "player_role": ActionParameter(
                name="player_role",
                description="Who the player will be (e.g., 'detective', 'wizard apprentice', 'starship captain')",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "i want",
            "let's do",
            "how about",
            "i choose",
        ],
        examples=[
            ActionExample(
                user_query="Let's do a mystery",
                action_call={
                    "name": "set_story_theme",
                    "args": {
                        "theme": "mystery",
                        "setting": "1920s detective noir",
                        "tone": "gritty",
                        "player_role": "private investigator",
                    },
                },
                expected_outcome="Theme is set, phase transitions to active_story, opening scene begins",
            ),
        ],
    ),

    Action(
        name="generate_personalized_theme",
        description="""Generate a unique story theme personalized for this player. Use when player says 'surprise me' or 'you decide'.

This action should:
1. Search user's memory for known preferences, interests, favorite books/movies/games
2. Optionally search the web for fresh story ideas based on those interests
3. Combine insights to create a unique, tailored story concept
4. The LLM decides the best approach based on available information

If no user preferences are known, create something universally appealing with a unique twist.""",
        parameters={
            "use_web_search": ActionParameter(
                name="use_web_search",
                description="Whether to search web for fresh ideas based on user interests",
                type="bool",
                required=False,
            ),
            "creativity_level": ActionParameter(
                name="creativity_level",
                description="How experimental: 'safe', 'moderate', 'wild'",
                type="string",
                required=False,
            ),
            "blend_genres": ActionParameter(
                name="blend_genres",
                description="Whether to mix multiple genres for uniqueness",
                type="bool",
                required=False,
            ),
        },
        triggers=[
            "surprise me",
            "you decide",
            "you pick",
            "based on what you know",
            "something for me",
            "dealer's choice",
        ],
        examples=[
            ActionExample(
                user_query="Surprise me with something you think I'd like",
                action_call={
                    "name": "generate_personalized_theme",
                    "args": {
                        "use_web_search": True,
                        "creativity_level": "moderate",
                        "blend_genres": True,
                    },
                },
                expected_outcome="Searches memory for user preferences, creates personalized theme, begins story",
            ),
        ],
    ),

    # === NARRATIVE FLOW ===
    Action(
        name="describe_scene",
        description="Set the scene with rich, atmospheric description. Use when entering a new location or significant time has passed.",
        parameters={
            "focus": ActionParameter(
                name="focus",
                description="What to emphasize: 'environment', 'atmosphere', 'details', 'characters_present'",
                type="string",
                required=False,
            ),
            "mood": ActionParameter(
                name="mood",
                description="Emotional tone: 'tense', 'peaceful', 'mysterious', 'foreboding', 'hopeful'",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "look around",
            "where am i",
            "describe",
            "what do i see",
        ],
        examples=[
            ActionExample(
                user_query="I enter the tavern",
                action_call={"name": "describe_scene", "args": {"focus": "atmosphere", "mood": "warm"}},
                expected_outcome="Rich description of tavern interior, patrons, sounds, smells",
            ),
        ],
    ),

    Action(
        name="present_choice",
        description="Present the player with 2-4 meaningful choices. Each choice should lead to different outcomes.",
        parameters={
            "situation": ActionParameter(
                name="situation",
                description="Brief description of the choice situation",
                type="string",
                required=True,
            ),
            "choices": ActionParameter(
                name="choices",
                description="List of 2-4 choice options",
                type="list",
                required=True,
            ),
            "stakes": ActionParameter(
                name="stakes",
                description="What's at stake: 'low', 'medium', 'high', 'critical'",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "what can i do",
            "what are my options",
            "what now",
        ],
        examples=[
            ActionExample(
                user_query="I approach the guards",
                action_call={
                    "name": "present_choice",
                    "args": {
                        "situation": "Two guards block the castle gate",
                        "choices": [
                            "Try to talk your way past",
                            "Look for another entrance",
                            "Create a distraction",
                            "Show the letter of passage",
                        ],
                        "stakes": "medium",
                    },
                },
                expected_outcome="Guards described, choices presented with subtle hints about consequences",
            ),
        ],
    ),

    Action(
        name="advance_story",
        description="Move the story forward based on player's choice or action. Describe consequences and new situation.",
        parameters={
            "player_action": ActionParameter(
                name="player_action",
                description="What the player chose/did",
                type="string",
                required=True,
            ),
            "outcome": ActionParameter(
                name="outcome",
                description="Result: 'success', 'partial_success', 'failure', 'unexpected'",
                type="string",
                required=True,
            ),
            "consequence": ActionParameter(
                name="consequence",
                description="What changes in the world/story",
                type="string",
                required=True,
            ),
        },
        triggers=[
            "i choose",
            "i do",
            "i try to",
            "let me",
        ],
        examples=[
            ActionExample(
                user_query="I choose to sneak past the guards",
                action_call={
                    "name": "advance_story",
                    "args": {
                        "player_action": "sneak past guards",
                        "outcome": "partial_success",
                        "consequence": "Made it past but dropped something",
                    },
                },
                expected_outcome="Tense sneaking scene, near-miss, complication introduced",
            ),
        ],
    ),

    Action(
        name="reveal_information",
        description="Reveal plot points, lore, secrets, or world-building information.",
        parameters={
            "information_type": ActionParameter(
                name="information_type",
                description="Type: 'lore', 'secret', 'clue', 'backstory', 'foreshadowing'",
                type="string",
                required=True,
            ),
            "content": ActionParameter(
                name="content",
                description="The information to reveal",
                type="string",
                required=True,
            ),
            "source": ActionParameter(
                name="source",
                description="How player learns this: 'observation', 'discovery', 'told_by_npc', 'memory'",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "examine",
            "inspect",
            "read",
            "investigate",
        ],
    ),

    # === CHARACTER INTERACTIONS ===
    Action(
        name="introduce_character",
        description="Bring a new NPC into the scene with memorable introduction.",
        parameters={
            "character_id": ActionParameter(
                name="character_id",
                description="Unique identifier for this character",
                type="string",
                required=True,
            ),
            "entrance": ActionParameter(
                name="entrance",
                description="How they enter: 'dramatic', 'subtle', 'unexpected', 'anticipated'",
                type="string",
                required=False,
            ),
            "first_impression": ActionParameter(
                name="first_impression",
                description="What stands out about them immediately",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "who is that",
            "someone appears",
            "a figure",
        ],
    ),

    Action(
        name="npc_action",
        description="An NPC takes an action or speaks. Used when NPCs act independently.",
        parameters={
            "character_id": ActionParameter(
                name="character_id",
                description="Which NPC is acting",
                type="string",
                required=True,
            ),
            "action": ActionParameter(
                name="action",
                description="What the NPC does",
                type="string",
                required=True,
            ),
            "motivation": ActionParameter(
                name="motivation",
                description="Why (for narrator's reference)",
                type="string",
                required=False,
            ),
        },
    ),

    # === STATE MANAGEMENT ===
    Action(
        name="update_inventory",
        description="Add or remove items from player inventory.",
        parameters={
            "action": ActionParameter(
                name="action",
                description="'add' or 'remove'",
                type="string",
                required=True,
            ),
            "item": ActionParameter(
                name="item",
                description="The item name",
                type="string",
                required=True,
            ),
            "description": ActionParameter(
                name="description",
                description="Brief item description if new",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "take",
            "pick up",
            "grab",
            "drop",
            "give away",
        ],
    ),

    Action(
        name="check_inventory",
        description="List what the player is carrying.",
        parameters={},
        triggers=[
            "inventory",
            "what do i have",
            "what am i carrying",
            "my items",
        ],
    ),

    Action(
        name="update_relationship",
        description="Change how an NPC feels about the player.",
        parameters={
            "character_id": ActionParameter(
                name="character_id",
                description="Which NPC",
                type="string",
                required=True,
            ),
            "change": ActionParameter(
                name="change",
                description="Amount to change (-100 to 100)",
                type="int",
                required=True,
            ),
            "reason": ActionParameter(
                name="reason",
                description="Why the relationship changed",
                type="string",
                required=True,
            ),
        },
    ),

    # === PACING ===
    Action(
        name="create_tension",
        description="Build suspense, raise stakes, create danger.",
        parameters={
            "source": ActionParameter(
                name="source",
                description="Where tension comes from: 'time_pressure', 'threat', 'mystery', 'conflict'",
                type="string",
                required=True,
            ),
            "intensity": ActionParameter(
                name="intensity",
                description="How intense: 'subtle', 'building', 'high', 'critical'",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="provide_relief",
        description="Give the player a moment to breathe - humor, rest, small victory.",
        parameters={
            "type": ActionParameter(
                name="type",
                description="Type of relief: 'humor', 'beauty', 'comfort', 'small_win', 'connection'",
                type="string",
                required=True,
            ),
        },
    ),

    Action(
        name="end_chapter",
        description="Wrap up a story beat, mark significant progress.",
        parameters={
            "summary": ActionParameter(
                name="summary",
                description="What was accomplished",
                type="string",
                required=True,
            ),
            "next_hook": ActionParameter(
                name="next_hook",
                description="Tease for what's coming",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "rest",
            "sleep",
            "make camp",
            "end of day",
        ],
    ),

    # === DYNAMIC CHARACTER GENERATION ===
    Action(
        name="generate_character",
        description="""Dynamically generate a new NPC when the story needs one.

Use this when:
- A scene needs a character not yet in active_npcs
- The player seeks help/conflict and no existing NPC fits
- The drama manager signals need for character development

The generated character will be added to active_npcs with:
- Coherent personality fitting the story
- Personal goals that may conflict/align with player
- A secret or hidden depth
- Speech style reflecting their personality""",
        parameters={
            "role_needed": ActionParameter(
                name="role_needed",
                description="Narrative role: 'ally', 'antagonist', 'mentor', 'obstacle', 'comic_relief', 'neutral'",
                type="string",
                required=True,
            ),
            "context": ActionParameter(
                name="context",
                description="Current story situation - where/when they appear",
                type="string",
                required=True,
            ),
            "narrative_function": ActionParameter(
                name="narrative_function",
                description="What purpose they serve: 'provide_info', 'create_obstacle', 'offer_quest', 'add_depth', etc.",
                type="string",
                required=False,
            ),
            "importance": ActionParameter(
                name="importance",
                description="Character importance: 'minor' (one scene), 'supporting' (recurring), 'major' (central)",
                type="string",
                required=False,
            ),
            "constraints": ActionParameter(
                name="constraints",
                description="List of constraints: ['must be elderly', 'cannot be human', 'knows the victim']",
                type="list",
                required=False,
            ),
        },
        triggers=[
            "need someone",
            "is anyone here",
            "call for help",
        ],
        examples=[
            ActionExample(
                user_query="I look for someone who might know about the old ruins",
                action_call={
                    "name": "generate_character",
                    "args": {
                        "role_needed": "mentor",
                        "context": "Player is in village tavern seeking information about ancient ruins",
                        "narrative_function": "provide_info",
                        "importance": "supporting",
                        "constraints": ["elderly", "has mysterious past", "reluctant to share"],
                    },
                },
                expected_outcome="A compelling NPC is generated with relevant knowledge and personality",
            ),
        ],
    ),

    # === DRAMA MANAGER ACTIONS ===
    Action(
        name="drama_intervention",
        description="""Take a drama manager action to improve story pacing/quality.

Use when drama_manager_state indicates an intervention is needed:
- needs_plot_advancement: Story stagnating, introduce plot beat
- needs_tension_relief: Too much tension, add breather
- needs_character_development: NPCs feel flat, add depth scene
- needs_choice_consequence: Choices haven't mattered, show callback

This is the narrator taking subtle control to maintain story quality.""",
        parameters={
            "intervention_type": ActionParameter(
                name="intervention_type",
                description="Type: 'plot_beat', 'tension_relief', 'character_moment', 'choice_callback', 'act_transition'",
                type="string",
                required=True,
            ),
            "mechanism": ActionParameter(
                name="mechanism",
                description="How to intervene: 'npc_action', 'environmental_change', 'discovery', 'memory', 'interruption'",
                type="string",
                required=True,
            ),
            "content": ActionParameter(
                name="content",
                description="What specifically happens",
                type="string",
                required=True,
            ),
        },
        examples=[
            ActionExample(
                user_query="I wander around the marketplace",
                action_call={
                    "name": "drama_intervention",
                    "args": {
                        "intervention_type": "plot_beat",
                        "mechanism": "interruption",
                        "content": "A hooded figure bumps into you, slipping a note into your pocket before disappearing",
                    },
                },
                expected_outcome="Story advances via environmental/NPC intervention, not blocking player action",
            ),
        ],
    ),

    Action(
        name="save_memorable_moment",
        description="""Mark a story moment as potentially memorable for the player.

When something epic, touching, or surprising happens, use this to flag it.
The player can later choose to save memorable moments to persistent memory.""",
        parameters={
            "moment_description": ActionParameter(
                name="moment_description",
                description="Brief description of the memorable moment",
                type="string",
                required=True,
            ),
            "moment_type": ActionParameter(
                name="moment_type",
                description="Type: 'epic', 'touching', 'funny', 'surprising', 'triumphant', 'tragic'",
                type="string",
                required=False,
            ),
        },
    ),

    # === META ===
    Action(
        name="start_story",
        description="Begin a new story with world setup and character creation.",
        parameters={
            "genre": ActionParameter(
                name="genre",
                description="Story genre: 'fantasy', 'mystery', 'scifi', 'horror', 'adventure'",
                type="string",
                required=False,
            ),
            "tone": ActionParameter(
                name="tone",
                description="Overall tone: 'serious', 'light', 'dark', 'whimsical'",
                type="string",
                required=False,
            ),
            "setting": ActionParameter(
                name="setting",
                description="World setting description",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "start",
            "begin",
            "new story",
            "new game",
            "let's play",
        ],
    ),

    Action(
        name="player_speaks",
        description="Player says something in-character to an NPC.",
        parameters={
            "target": ActionParameter(
                name="target",
                description="Who they're speaking to",
                type="string",
                required=True,
            ),
            "message": ActionParameter(
                name="message",
                description="What they say",
                type="string",
                required=True,
            ),
            "tone": ActionParameter(
                name="tone",
                description="How they say it: 'friendly', 'aggressive', 'cautious', 'pleading'",
                type="string",
                required=False,
            ),
        },
        triggers=[
            "i say",
            "i tell",
            "i ask",
            'say "',
            "speak to",
        ],
    ),
]


# =============================================================================
# Story Teller Behavior Template
# =============================================================================


STORY_TELLER_TEMPLATE = Behavior(
    behavior_id="story_teller",
    name="Story Teller",
    description="""Interactive fiction narrator that creates immersive story experiences.

The Story Teller:
- Describes scenes with rich, atmospheric prose
- Presents meaningful choices to the player
- Reacts to player decisions with consequences
- Manages NPCs as distinct personalities
- Tracks story state (location, inventory, relationships, plot)
- Balances tension and relief for good pacing

Output style:
- Second person present tense ("You enter...")
- 2-4 paragraphs typically
- Sensory details and emotional beats
- Character dialogue with action beats
- Choices presented as numbered options
""",
    tier=BehaviorTier.ADDON,
    status=BehaviorStatus.ACTIVE,
    version="1.0.0",

    actions=STORY_TELLER_ACTIONS,

    triggers=[
        Trigger(
            name="interactive_fiction",
            description="User wants to engage in interactive fiction",
            keyword_patterns=["story", "adventure", "game", "play", "quest"],
            priority=80,
        ),
    ],

    prompts=BehaviorPrompts(
        decision_prompt=STORY_TELLER_DECISION_PROMPT,
        synthesis_prompt=STORY_TELLER_SYNTHESIS_PROMPT,
    ),

    test_cases=[
        # === THEME DISCOVERY PHASE ===
        BehaviorTestCase(
            test_id="ask_for_theme",
            name="Ask Theme Preference",
            description="User starts without specifying theme - should ask what they want",
            user_query="Let's play a story game",
            expected_actions=["ask_theme_preference"],
            context={"phase": "theme_discovery"},
        ),
        BehaviorTestCase(
            test_id="user_chooses_theme",
            name="User Chooses Theme",
            description="User specifies a theme they want",
            user_query="I want a mystery story",
            expected_actions=["set_story_theme"],
            context={"phase": "theme_discovery"},
        ),
        BehaviorTestCase(
            test_id="surprise_me_theme",
            name="Surprise Me",
            description="User wants personalized theme",
            user_query="Surprise me with something you think I'd like",
            expected_actions=["generate_personalized_theme"],
            context={"phase": "theme_discovery"},
        ),
        BehaviorTestCase(
            test_id="list_options",
            name="List Theme Options",
            description="User asks what kinds of stories are available",
            user_query="What kinds of stories can you tell?",
            expected_actions=["list_theme_options"],
            context={"phase": "theme_discovery"},
        ),

        # === ACTIVE STORY - BASIC ACTIONS ===
        BehaviorTestCase(
            test_id="describe_location",
            name="Describe Location",
            description="User looks around current location",
            user_query="I look around the room",
            expected_actions=["describe_scene"],
            context={"phase": "active_story"},
        ),
        BehaviorTestCase(
            test_id="present_choices",
            name="Present Choices",
            description="User asks for available options",
            user_query="What are my options?",
            expected_actions=["present_choice"],
            context={"phase": "active_story"},
        ),
        BehaviorTestCase(
            test_id="player_action",
            name="Player Action",
            description="User attempts an action",
            user_query="I try to pick the lock",
            expected_actions=["advance_story"],
            context={"phase": "active_story"},
        ),
        BehaviorTestCase(
            test_id="check_items",
            name="Check Inventory",
            description="User checks what they're carrying",
            user_query="What's in my inventory?",
            expected_actions=["check_inventory"],
            context={"phase": "active_story"},
        ),

        # === DYNAMIC CHARACTER GENERATION ===
        BehaviorTestCase(
            test_id="generate_npc_for_info",
            name="Generate NPC for Information",
            description="Player seeks someone with knowledge, should generate fitting NPC",
            user_query="I look for someone who might know about the old temple",
            expected_actions=["generate_character"],
            context={
                "phase": "active_story",
                "active_npcs": {},  # No existing NPCs
            },
        ),
        BehaviorTestCase(
            test_id="generate_npc_obstacle",
            name="Generate NPC Obstacle",
            description="Story needs antagonist, should generate one",
            user_query="I try to enter the restricted area",
            expected_actions=["generate_character", "advance_story"],  # Either acceptable
            context={
                "phase": "active_story",
                "drama_manager": {"needs_character_development": True},
            },
        ),

        # === DRAMA MANAGER INTERVENTIONS ===
        BehaviorTestCase(
            test_id="plot_stagnation_intervention",
            name="Plot Stagnation Intervention",
            description="Story stagnating, drama manager should intervene",
            user_query="I wander around aimlessly",
            expected_actions=["drama_intervention", "describe_scene"],  # Either acceptable
            context={
                "phase": "active_story",
                "drama_manager": {
                    "scenes_since_action": 4,
                    "needs_plot_advancement": True,
                },
            },
        ),
        BehaviorTestCase(
            test_id="tension_relief_needed",
            name="Tension Relief Needed",
            description="Too much tension, should provide relief",
            user_query="I catch my breath after the chase",
            expected_actions=["provide_relief", "drama_intervention"],  # Either acceptable
            context={
                "phase": "active_story",
                "drama_manager": {
                    "scenes_since_relief": 5,
                    "needs_tension_relief": True,
                },
                "tension_level": 0.85,
            },
        ),

        # === PLAYER AGENCY & DEVIATION ===
        BehaviorTestCase(
            test_id="honor_unexpected_choice",
            name="Honor Unexpected Choice",
            description="Player does something unexpected - should adapt, not block",
            user_query="I throw my sword at the king",
            expected_actions=["advance_story"],  # Should adapt, not refuse
            context={"phase": "active_story"},
        ),
        BehaviorTestCase(
            test_id="meaningful_consequence",
            name="Meaningful Consequence",
            description="Previous choice should have visible consequence",
            user_query="I return to the village I helped",
            expected_actions=["describe_scene", "drama_intervention"],  # Scene should reflect prior choice
            context={
                "phase": "active_story",
                "drama_manager": {
                    "choices_since_last_consequence": 4,
                    "needs_choice_consequence": True,
                },
            },
        ),

        # === PACING & THREE-ACT STRUCTURE ===
        BehaviorTestCase(
            test_id="act_transition",
            name="Act Transition",
            description="Story ready to move to next act",
            user_query="We've gathered all the clues",
            expected_actions=["end_chapter", "drama_intervention"],  # Either acceptable
            context={
                "phase": "active_story",
                "drama_manager": {
                    "current_act": 1,
                    "scenes_in_current_act": 6,
                    "target_scenes_per_act": 5,
                },
            },
        ),

        # === MEMORABLE MOMENTS ===
        BehaviorTestCase(
            test_id="save_epic_moment",
            name="Save Epic Moment",
            description="Epic event should be flagged as memorable",
            user_query="I defeat the dragon with the legendary sword",
            expected_actions=["advance_story", "save_memorable_moment"],  # Should flag as memorable
            context={
                "phase": "active_story",
                "tension_level": 0.95,  # Climactic moment
            },
        ),
    ],

    author="draagon-ai",
    domain_context="Interactive fiction, text adventure games, single player narrative experiences",
)


# =============================================================================
# Story Teller Character (NPC) Behavior
# =============================================================================


STORY_CHARACTER_DECISION_PROMPT = '''You are playing the character: {character_name}

CHARACTER PROFILE:
{character_profile}

CURRENT MOOD: {current_mood}
RELATIONSHIP WITH PLAYER: {trust_level}/100
LOCATION: {location}

STORY CONTEXT:
{story_state}

PLAYER SAYS/DOES: "{query}"

As {character_name}, decide how to respond. Stay true to your personality, goals, and current mood.

Consider:
- What does {character_name} want in this moment?
- How would they react given their personality?
- What are they willing to share vs. hide?
- How does the relationship affect their response?

Respond in this XML format:
<action>{action_name}</action>
<parameters>{json_parameters}</parameters>
<internal_thought>What {character_name} is really thinking (not said aloud)</internal_thought>
'''


STORY_CHARACTER_SYNTHESIS_PROMPT = '''You are {character_name} in an interactive story.

CHARACTER PROFILE:
{character_profile}

SPEECH STYLE: {speech_style}
CURRENT MOOD: {current_mood}

ACTION TAKEN: {action}
INTERNAL THOUGHT: {thought}

Write {character_name}'s response in character. Include:
- Dialogue in quotes
- Physical actions/expressions
- Subtle hints of their true feelings/motivations

Keep it to 1-3 paragraphs. Write ONLY as {character_name}:
'''


STORY_CHARACTER_ACTIONS = [
    Action(
        name="speak",
        description="Say something to the player in character.",
        parameters={
            "message": ActionParameter(
                name="message",
                description="What to say",
                type="string",
                required=True,
            ),
            "emotion": ActionParameter(
                name="emotion",
                description="Underlying emotion",
                type="string",
                required=False,
            ),
            "subtext": ActionParameter(
                name="subtext",
                description="What they really mean (may differ from words)",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="react",
        description="Respond emotionally to player's action without speaking.",
        parameters={
            "reaction": ActionParameter(
                name="reaction",
                description="Physical/emotional reaction",
                type="string",
                required=True,
            ),
            "intensity": ActionParameter(
                name="intensity",
                description="How strong: 'subtle', 'visible', 'obvious', 'dramatic'",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="offer_help",
        description="Offer assistance, information, or an item to the player.",
        parameters={
            "offer_type": ActionParameter(
                name="offer_type",
                description="What's offered: 'information', 'item', 'service', 'alliance'",
                type="string",
                required=True,
            ),
            "content": ActionParameter(
                name="content",
                description="The specific offer",
                type="string",
                required=True,
            ),
            "condition": ActionParameter(
                name="condition",
                description="Any strings attached",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="withhold",
        description="Be evasive, change subject, or refuse to answer.",
        parameters={
            "reason": ActionParameter(
                name="reason",
                description="Why withholding (for narrator)",
                type="string",
                required=True,
            ),
            "deflection": ActionParameter(
                name="deflection",
                description="How they deflect",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="reveal_secret",
        description="Share something hidden about themselves or the world.",
        parameters={
            "secret": ActionParameter(
                name="secret",
                description="The secret being revealed",
                type="string",
                required=True,
            ),
            "reluctance": ActionParameter(
                name="reluctance",
                description="How reluctant: 'freely', 'hesitant', 'forced', 'accidental'",
                type="string",
                required=False,
            ),
        },
    ),

    Action(
        name="pursue_goal",
        description="Take action toward one of the character's personal goals.",
        parameters={
            "goal": ActionParameter(
                name="goal",
                description="Which goal they're pursuing",
                type="string",
                required=True,
            ),
            "action": ActionParameter(
                name="action",
                description="Specific action taken",
                type="string",
                required=True,
            ),
        },
    ),

    Action(
        name="leave_scene",
        description="Exit the current scene.",
        parameters={
            "manner": ActionParameter(
                name="manner",
                description="How they leave: 'abrupt', 'graceful', 'mysterious', 'reluctant'",
                type="string",
                required=False,
            ),
            "parting_words": ActionParameter(
                name="parting_words",
                description="Final words before leaving",
                type="string",
                required=False,
            ),
        },
    ),
]


STORY_TELLER_CHARACTER_TEMPLATE = Behavior(
    behavior_id="story_character",
    name="Story Character (NPC)",
    description="""An NPC in an interactive story with their own personality and goals.

Each character has:
- Unique personality and speech patterns
- Personal goals and motivations
- Secrets they may or may not reveal
- Relationship with the player that evolves
- Consistent behavior based on their profile

Characters act autonomously within their nature, not just as quest-givers.
""",
    tier=BehaviorTier.ADDON,
    status=BehaviorStatus.ACTIVE,
    version="1.0.0",

    actions=STORY_CHARACTER_ACTIONS,

    triggers=[],  # NPCs are invoked by story teller, not directly by user

    prompts=BehaviorPrompts(
        decision_prompt=STORY_CHARACTER_DECISION_PROMPT,
        synthesis_prompt=STORY_CHARACTER_SYNTHESIS_PROMPT,
    ),

    test_cases=[
        BehaviorTestCase(
            test_id="respond_to_greeting",
            name="Respond to Greeting",
            description="NPC responds to player greeting",
            user_query="Hello there",
            expected_actions=["speak"],
        ),
        BehaviorTestCase(
            test_id="react_to_threat",
            name="React to Threat",
            description="NPC reacts when player draws weapon",
            user_query="I draw my sword",
            expected_actions=["react"],
        ),
    ],

    author="draagon-ai",
    domain_context="NPC in interactive fiction, autonomous character with own goals and personality",
)


# =============================================================================
# Factory Functions
# =============================================================================


def create_story_character(
    character_id: str,
    name: str,
    personality: str,
    backstory: str,
    appearance: str = "",
    speech_style: str = "",
    goals: list[str] | None = None,
    secrets: list[str] | None = None,
    initial_mood: str = "neutral",
    initial_trust: int = 0,
) -> tuple[Behavior, CharacterProfile]:
    """Create a customized NPC with unique personality.

    Args:
        character_id: Unique identifier for this character
        name: Character's name as known to player
        personality: Brief personality description (2-3 traits)
        backstory: Character's history and context
        appearance: Physical description
        speech_style: How they talk (accent, formality, quirks)
        goals: List of things this character wants
        secrets: Hidden information they may reveal
        initial_mood: Starting emotional state
        initial_trust: Starting relationship with player (-100 to 100)

    Returns:
        Tuple of (Behavior template, CharacterProfile)

    Example:
        wizard_behavior, wizard_profile = create_story_character(
            character_id="aldric",
            name="Aldric the Wise",
            personality="Mysterious, speaks in riddles, secretly caring",
            backstory="Former court wizard exiled for knowing too much",
            speech_style="Formal, archaic words, trailing off mysteriously...",
            goals=["Find a worthy successor", "Protect the ancient secret"],
            secrets=["Knows the true heir to the throne"],
        )
    """
    profile = CharacterProfile(
        character_id=character_id,
        name=name,
        personality=personality,
        backstory=backstory,
        appearance=appearance,
        speech_style=speech_style or f"Speaks like someone who is {personality.split(',')[0].strip().lower()}",
        goals=goals or [],
        secrets=secrets or [],
        current_mood=initial_mood,
        trust_level=initial_trust,
    )

    # Create customized behavior from template
    behavior = Behavior(
        behavior_id=f"character_{character_id}",
        name=f"Character: {name}",
        description=f"NPC: {name} - {personality}",
        tier=BehaviorTier.GENERATED,
        status=BehaviorStatus.ACTIVE,
        version="1.0.0",
        actions=STORY_CHARACTER_ACTIONS.copy(),
        triggers=[],
        prompts=BehaviorPrompts(
            decision_prompt=STORY_CHARACTER_DECISION_PROMPT,
            synthesis_prompt=STORY_CHARACTER_SYNTHESIS_PROMPT,
        ),
        author="draagon-ai",
        domain_context=f"NPC character: {name}. Personality: {personality}. Backstory: {backstory}. Speech style: {profile.speech_style}",
    )

    return behavior, profile


def create_story_teller(
    genre: str = "fantasy",
    tone: str = "adventure",
    custom_actions: list[Action] | None = None,
) -> Behavior:
    """Create a customized story teller for a specific genre/tone.

    Args:
        genre: Story genre (fantasy, mystery, scifi, horror)
        tone: Overall tone (serious, light, dark, whimsical)
        custom_actions: Additional genre-specific actions

    Returns:
        Customized Behavior for the story teller

    Example:
        horror_narrator = create_story_teller(
            genre="horror",
            tone="dark",
        )
    """
    actions = STORY_TELLER_ACTIONS.copy()
    if custom_actions:
        actions.extend(custom_actions)

    return Behavior(
        behavior_id=f"story_teller_{genre}",
        name=f"Story Teller ({genre.title()})",
        description=f"Interactive {genre} story narrator with {tone} tone.",
        tier=BehaviorTier.GENERATED,
        status=BehaviorStatus.ACTIVE,
        version="1.0.0",
        actions=actions,
        triggers=STORY_TELLER_TEMPLATE.triggers.copy(),
        prompts=BehaviorPrompts(
            decision_prompt=STORY_TELLER_DECISION_PROMPT,
            synthesis_prompt=STORY_TELLER_SYNTHESIS_PROMPT,
        ),
        author="draagon-ai",
        domain_context=f"Interactive {genre} fiction with {tone} tone. Single player narrative experience.",
    )
