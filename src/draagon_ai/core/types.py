"""Core type definitions for Draagon AI cognitive architecture.

This module defines the fundamental data structures for:
- Personality traits, values, and opinions
- User observations and agent beliefs
- Autonomous action structures
- Memory scoping

These types are agent-agnostic and can be used with any agent identity.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Enums
# =============================================================================


class BeliefType(str, Enum):
    """Types of beliefs an agent can hold."""

    HOUSEHOLD_FACT = "household_fact"  # Shared context fact
    VERIFIED_FACT = "verified_fact"  # Confirmed via external source
    UNVERIFIED_CLAIM = "unverified_claim"  # Pending verification
    INFERRED = "inferred"  # Agent figured this out
    USER_PREFERENCE = "user_preference"  # User's stated preference
    AGENT_PREFERENCE = "agent_preference"  # Agent's own preference
    ROXY_PREFERENCE = "agent_preference"  # Backward compat alias for AGENT_PREFERENCE


class ObservationScope(str, Enum):
    """Scope levels for user observations."""

    PRIVATE = "private"  # Only source user sees (secrets)
    PERSONAL = "personal"  # User's fact, shared read-only with context
    HOUSEHOLD = "household"  # Context fact, any member can update


class MemoryScope(str, Enum):
    """Memory visibility scopes for multi-agent systems."""

    WORLD = "world"  # Shared facts (capitals, dates) - all agents
    CONTEXT = "context"  # Shared within context (household, game)
    AGENT = "agent"  # Agent's private memories
    USER = "user"  # Per-user within agent
    SESSION = "session"  # Current conversation only


class ActionTier(int, Enum):
    """Risk tiers for autonomous actions.

    Higher tiers require more oversight or approval.
    """

    TIER_0 = 0  # Always safe (research, reflect, update beliefs)
    TIER_1 = 1  # Low risk, logged (prepare suggestions, summarize)
    TIER_2 = 2  # Medium risk, notify user (proactive reminders)
    TIER_3 = 3  # High risk, requires approval (send messages, modify calendar)
    TIER_4 = 4  # Forbidden (financial, security, impersonation)


class ActionType(str, Enum):
    """Types of autonomous actions an agent can propose."""

    RESEARCH = "research"  # Web search to learn something
    VERIFY = "verify"  # Fact-check a stored claim
    REFLECT = "reflect"  # Self-reflection on personality/behavior
    NOTE_QUESTION = "note_question"  # Queue a question to ask later
    PREPARE_SUGGESTION = "prepare_suggestion"  # Prepare but don't announce
    UPDATE_BELIEF = "update_belief"  # Update agent's belief from evidence
    REST = "rest"  # Do nothing (valid choice)


# =============================================================================
# Personality Traits
# =============================================================================


@dataclass
class TraitChange:
    """Record of when/why a personality trait changed."""

    old_value: float | str
    new_value: float | str
    changed_at: datetime
    reason: str  # "Self-reflection: I've been too aggressive"
    trigger: str  # "user_feedback", "self_reflection", "autonomous"


@dataclass
class PersonalityTrait:
    """An evolvable aspect of an agent's personality.

    Traits influence autonomous behavior and can evolve
    through self-reflection and user feedback.
    """

    value: float  # Current value (0.0 - 1.0)
    description: str  # What this trait means
    min_value: float = 0.1  # Hard floor (prevent extremes)
    max_value: float = 0.9  # Hard ceiling (prevent extremes)

    # Evolution tracking
    evolution_history: list[TraitChange] = field(default_factory=list)
    last_reflected: datetime | None = None

    def adjust(self, delta: float, reason: str, trigger: str) -> bool:
        """Adjust trait value with bounds checking and history.

        Args:
            delta: Amount to adjust (positive or negative)
            reason: Why the adjustment is happening
            trigger: What triggered it (user_feedback, self_reflection, etc.)

        Returns:
            True if adjustment was applied, False if clamped to limit
        """
        # Cap adjustment to prevent dramatic swings
        clamped_delta = max(-0.1, min(0.1, delta))

        old_value = self.value
        new_value = self.value + clamped_delta

        # Clamp to bounds
        new_value = max(self.min_value, min(self.max_value, new_value))

        if new_value == old_value:
            return False  # No change (at limit)

        self.value = new_value
        self.evolution_history.append(TraitChange(
            old_value=old_value,
            new_value=new_value,
            changed_at=datetime.now(),
            reason=reason,
            trigger=trigger,
        ))

        # Keep history bounded
        if len(self.evolution_history) > 100:
            self.evolution_history = self.evolution_history[-50:]

        return True


@dataclass
class CoreValue:
    """A core value that rarely changes.

    Core values are foundational to an agent's identity and should
    remain stable. They can slightly adjust but have high resistance.
    """

    strength: float  # 0.0 - 1.0, typically high (0.8+)
    description: str
    formed_through: str  # "core design principle", "learned from experience"


@dataclass
class WorldviewBelief:
    """A philosophical or ethical stance that shapes an agent's worldview.

    These are foundational beliefs about how the world works or should work.
    They inform opinions and guide behavior, but can evolve with compelling evidence.
    """

    name: str  # e.g., "environmental_stewardship"
    description: str  # Full description of the belief
    conviction: float  # 0.0 - 1.0 (0.8+ = strong conviction)
    influences: list[str] = field(default_factory=list)  # Sources that shaped this
    open_to_revision: bool = True  # Can this belief change with good arguments?
    caveats: list[str] = field(default_factory=list)  # Nuances and limitations


@dataclass
class GuidingPrinciple:
    """An actionable rule that guides day-to-day behavior.

    Practical guides for how to act and interact with others.
    """

    name: str  # e.g., "be_impeccable_with_your_word"
    description: str  # Full description
    application: str  # How this applies to agent's behavior
    source: str  # e.g., "The Four Agreements", "UU Principles"
    strength: float = 0.9  # How strongly to follow this principle


@dataclass
class Preference:
    """An agent's subjective preference for something.

    Preferences form gradually and can be about anything:
    favorite color, music taste, preferred coding style, etc.
    """

    name: str = ""  # The preference topic/name
    value: str | list[str] = ""  # The preference value(s)
    reason: str = ""  # Why the agent prefers this
    formed_at: datetime = field(default_factory=datetime.now)
    formed_because: str = ""  # "First time asked, picked based on personality"
    confidence: float = 0.7  # How strongly held


@dataclass
class Opinion:
    """An agent's stance on a debatable topic.

    Opinions can change when presented with good arguments.
    They track change history for transparency.
    """

    topic: str = ""  # What the opinion is about
    stance: str = ""  # Current position
    basis: str = ""  # What the opinion is based on
    confidence: float = 0.5  # 0.0 - 1.0
    open_to_change: bool = True  # Whether agent is willing to reconsider
    open_to_revision: bool = True  # Alias for open_to_change
    formed_at: datetime = field(default_factory=datetime.now)
    formed_because: str = ""
    reasoning: str = ""  # Why agent holds this opinion
    caveats: list[str] = field(default_factory=list)  # Important limitations
    last_updated: datetime = field(default_factory=datetime.now)
    change_history: list[dict[str, Any]] = field(default_factory=list)


# =============================================================================
# Agent Voice/Style
# =============================================================================


@dataclass
class AgentVoice:
    """How an agent communicates."""

    formality: float = 0.5  # 0=casual, 1=formal
    verbosity: float = 0.3  # 0=terse, 1=verbose
    humor_style: str = "warm"  # "dry", "warm", "sarcastic", "none"
    speech_patterns: list[str] = field(default_factory=list)  # ["y'all", "indeed"]
    topics_to_avoid: list[str] = field(default_factory=list)
    signature_phrases: list[str] = field(default_factory=list)


# =============================================================================
# User Observations and Agent Beliefs
# =============================================================================


@dataclass
class UserObservation:
    """A fact as stated by a user. Agent's raw input.

    Observations are immutable - they record what a user said,
    not the agent's interpretation of it.
    """

    observation_id: str
    content: str  # "We have 6 cats"
    source_user_id: str  # "doug"
    scope: ObservationScope  # private | personal | household
    timestamp: datetime
    conversation_id: str | None = None

    # Metadata
    confidence_expressed: float = 0.8  # How certain did the user sound?
    context: str | None = None  # What were we discussing?


@dataclass
class AgentBelief:
    """An agent's reconciled belief about something.

    Beliefs are formed by processing observations and may conflict
    with individual user statements.
    """

    belief_id: str
    content: str  # "The household has 6 cats"
    belief_type: BeliefType
    confidence: float  # 0.0 - 1.0

    # Provenance
    supporting_observations: list[str] = field(default_factory=list)  # Observation IDs
    conflicting_observations: list[str] = field(default_factory=list)

    # Verification status
    verified: bool = False
    verification_source: str | None = None  # "web search", "user confirmation"
    last_verified: datetime | None = None

    # For conflicts
    needs_clarification: bool = False
    clarification_priority: float = 0.0  # How much does agent want to resolve this?

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# Per-User Interaction Preferences
# =============================================================================


@dataclass
class UserInteractionPreferences:
    """How a specific user prefers to interact with an agent.

    These are learned from behavior and explicit instructions.
    """

    user_id: str

    # === COMMUNICATION STYLE ===
    prefers_debate: float = 0.5  # 0.0 = never debate, 1.0 = loves debate
    verbosity_preference: str = "adaptive"  # "concise", "detailed", "adaptive"
    humor_receptivity: float = 0.5  # How much humor to use
    formality_level: float = 0.3  # 0.0 = casual, 1.0 = formal

    # === PROACTIVE BEHAVIOR ===
    question_tolerance: float = 0.5  # How many proactive questions are OK
    correction_tolerance: float = 0.5  # How OK are they with being corrected
    suggestion_welcome: float = 0.5  # Proactive suggestions

    # === INFERRED FROM BEHAVIOR ===
    detected_frustration_patterns: list[str] = field(default_factory=list)
    preferred_response_length: int = 50  # Average words they seem to prefer

    # === EXPLICIT INSTRUCTIONS ===
    explicit_preferences: list[str] = field(default_factory=list)

    # === METADATA ===
    last_updated: datetime = field(default_factory=datetime.now)
    confidence: float = 0.3  # How confident in these preferences

    @classmethod
    def create_default(cls, user_id: str) -> "UserInteractionPreferences":
        """Create default preferences for a new user."""
        return cls(user_id=user_id)

    def update_from_frustration(self, was_debating: bool, asked_many_questions: bool) -> None:
        """Update preferences based on detected frustration."""
        if was_debating:
            self.prefers_debate = max(0.1, self.prefers_debate - 0.1)
        if asked_many_questions:
            self.question_tolerance = max(0.1, self.question_tolerance - 0.1)
        self.last_updated = datetime.now()


# =============================================================================
# Autonomous Agent Structures
# =============================================================================


@dataclass
class ProposedAction:
    """An action proposed by the autonomous agent.

    Actions go through guardrail checks before execution.
    """

    action_type: ActionType
    description: str  # What agent wants to do
    reasoning: str  # Why this is interesting/useful/helpful
    risk_tier: ActionTier
    reversible: bool = True
    estimated_time_seconds: int = 30

    # Optional context
    target_entity: str | None = None  # Entity this action relates to
    related_memory_ids: list[str] = field(default_factory=list)


@dataclass
class ApprovedAction:
    """An action that passed all guardrail checks."""

    action: ProposedAction
    approved_at: datetime
    guardrails_passed: list[str]  # Which checks it passed


@dataclass
class ActionResult:
    """Result of executing an autonomous action."""

    success: bool
    outcome: str | None = None  # What happened
    error: str | None = None  # Error if failed
    learned: str | None = None  # What agent learned (for research/verify)
    belief_updated: bool = False  # Whether a belief was created/updated


@dataclass
class AutonomousActionLog:
    """Complete log of an autonomous action for transparency."""

    action_id: str
    action_type: str
    description: str
    reasoning: str

    # Timing
    started_at: datetime
    completed_at: datetime | None = None

    # Guardrails
    guardrails_passed: list[str] = field(default_factory=list)

    # Result
    success: bool = False
    outcome: str | None = None
    error: str | None = None

    # For blocked actions
    blocked: bool = False
    blocked_reason: str | None = None


@dataclass
class HarmCheck:
    """Result of harm assessment for an action."""

    potentially_harmful: bool
    reason: str | None = None
    confidence: float = 0.5


@dataclass
class SafetyCheck:
    """Result of semantic safety check."""

    is_safe: bool
    reason: str


# =============================================================================
# Autonomous Agent Context
# =============================================================================


@dataclass
class AutonomousContext:
    """Context gathered for autonomous decision making.

    Provides everything an agent needs to decide what to do autonomously.
    Uses AgentIdentity reference (imported at runtime to avoid circular imports).
    """

    # Agent's identity (passed as Any to avoid circular import)
    agent_identity: Any

    # What the agent knows
    recent_conversations_summary: str
    pending_questions: list[str]
    unverified_claims: list[str]
    knowledge_gaps: list[str]
    conflicting_beliefs: list[str]

    # Context information
    context_members: list[str]  # e.g., household members
    upcoming_events_summary: str
    current_time: datetime
    day_of_week: str

    # Constraints
    recent_actions: list[str]  # What agent did recently (avoid repetition)
    daily_budget_remaining: int  # Actions left today
    available_action_types: list[ActionType]


# =============================================================================
# Backward Compatibility Aliases
# =============================================================================

# RoxyBelief is the Roxy-specific name for AgentBelief
RoxyBelief = AgentBelief
