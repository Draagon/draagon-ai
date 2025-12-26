"""Identity Manager for Draagon AI agents.

Handles loading, saving, and caching an agent's persistent identity from storage.
This is the central point for an agent's evolving personality, values, preferences, and opinions.

Key responsibilities:
- Load AgentIdentity from storage on startup
- Cache it in memory for fast access during conversations
- Save changes back to storage periodically or on significant changes
- Load per-user interaction preferences
"""

import logging
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from draagon_ai.core import (
    AgentIdentity,
    CoreValue,
    PersonalityTrait,
    UserInteractionPreferences,
    WorldviewBelief,
    GuidingPrinciple,
    Preference,
    Opinion,
    TraitChange,
)
from draagon_ai.core.identity import (
    serialize_agent_identity,
    deserialize_agent_identity,
)
from draagon_ai.llm import LLMProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class IdentityStorage(Protocol):
    """Protocol for identity persistence.

    Host applications implement this to store agent identity
    in their preferred storage backend (Qdrant, SQLite, file, etc.).
    """

    async def load_identity(self, agent_id: str) -> dict[str, Any] | None:
        """Load serialized identity data for an agent.

        Args:
            agent_id: The agent's unique identifier

        Returns:
            Serialized identity dict, or None if not found
        """
        ...

    async def save_identity(self, agent_id: str, data: dict[str, Any]) -> None:
        """Save serialized identity data for an agent.

        Args:
            agent_id: The agent's unique identifier
            data: Serialized identity data to store
        """
        ...

    async def load_user_preferences(
        self, agent_id: str, user_id: str
    ) -> dict[str, Any] | None:
        """Load per-user interaction preferences.

        Args:
            agent_id: The agent's unique identifier
            user_id: The user to load preferences for

        Returns:
            Serialized preferences dict, or None if not found
        """
        ...

    async def save_user_preferences(
        self, agent_id: str, user_id: str, data: dict[str, Any]
    ) -> None:
        """Save per-user interaction preferences.

        Args:
            agent_id: The agent's unique identifier
            user_id: The user to save preferences for
            data: Serialized preferences data
        """
        ...


# =============================================================================
# Serialization Helpers (re-export from core.identity)
# =============================================================================

# Use the canonical serialization functions from core.identity
serialize_identity = serialize_agent_identity
deserialize_identity = deserialize_agent_identity


def serialize_user_prefs(prefs: UserInteractionPreferences) -> dict[str, Any]:
    """Serialize UserInteractionPreferences for storage."""
    return {
        "user_id": prefs.user_id,
        "prefers_debate": prefs.prefers_debate,
        "verbosity_preference": prefs.verbosity_preference,
        "humor_receptivity": prefs.humor_receptivity,
        "formality_level": prefs.formality_level,
        "question_tolerance": prefs.question_tolerance,
        "correction_tolerance": prefs.correction_tolerance,
        "suggestion_welcome": prefs.suggestion_welcome,
        "detected_frustration_patterns": prefs.detected_frustration_patterns,
        "preferred_response_length": prefs.preferred_response_length,
        "explicit_preferences": prefs.explicit_preferences,
        "last_updated": prefs.last_updated.isoformat(),
        "confidence": prefs.confidence,
    }


def deserialize_user_prefs(data: dict[str, Any]) -> UserInteractionPreferences:
    """Deserialize UserInteractionPreferences from storage."""
    return UserInteractionPreferences(
        user_id=data.get("user_id", "unknown"),
        prefers_debate=data.get("prefers_debate", 0.5),
        verbosity_preference=data.get("verbosity_preference", "adaptive"),
        humor_receptivity=data.get("humor_receptivity", 0.5),
        formality_level=data.get("formality_level", 0.3),
        question_tolerance=data.get("question_tolerance", 0.5),
        correction_tolerance=data.get("correction_tolerance", 0.5),
        suggestion_welcome=data.get("suggestion_welcome", 0.5),
        detected_frustration_patterns=data.get("detected_frustration_patterns", []),
        preferred_response_length=data.get("preferred_response_length", 50),
        explicit_preferences=data.get("explicit_preferences", []),
        last_updated=datetime.fromisoformat(data["last_updated"]) if data.get("last_updated") else datetime.now(),
        confidence=data.get("confidence", 0.3),
    )


# =============================================================================
# Identity Manager
# =============================================================================


class IdentityManager:
    """Manages an agent's persistent identity.

    Responsibilities:
    - Load AgentIdentity from storage on startup
    - Cache it in memory for fast access during conversations
    - Save changes back to storage periodically or on significant changes
    - Load per-user interaction preferences

    This is the implementation of the IdentityManager protocol used by
    other services like OpinionFormationService.
    """

    def __init__(
        self,
        llm: LLMProvider,
        storage: IdentityStorage,
        agent_id: str = "agent",
        agent_name: str = "the agent",
    ):
        """Initialize the identity manager.

        Args:
            llm: LLM provider for query-relevant opinion matching
            storage: Storage backend for persistence
            agent_id: Unique identifier for this agent
            agent_name: Display name for this agent
        """
        self.llm = llm
        self.storage = storage
        self.agent_id = agent_id
        self.agent_name = agent_name

        # Cached identity (loaded lazily)
        self._identity: AgentIdentity | None = None
        self._loaded = False

        # Cached user interaction preferences
        self._user_prefs: dict[str, UserInteractionPreferences] = {}

        # Track if dirty (needs saving)
        self._dirty = False

    async def load(self) -> AgentIdentity:
        """Load AgentIdentity from storage, or create default if none exists.

        Returns:
            The loaded or newly created AgentIdentity
        """
        if self._loaded and self._identity:
            return self._identity

        try:
            # Try to load from storage
            data = await self.storage.load_identity(self.agent_id)

            if data:
                self._identity = deserialize_identity(data)
                self._loaded = True
                logger.info(f"Loaded identity for agent: {self.agent_id}")
                return self._identity

            # No existing record found, create default
            logger.info(f"No existing identity found for {self.agent_id}, creating minimal")
            self._identity = AgentIdentity.create_minimal(self.agent_id, self.agent_name)
            self._loaded = True

            # Save the new default to storage
            await self._save_to_storage()

            return self._identity

        except Exception as e:
            logger.error(f"Error loading identity: {e}", exc_info=True)
            # Fall back to default on error
            self._identity = AgentIdentity.create_minimal(self.agent_id, self.agent_name)
            self._loaded = True
            return self._identity

    def get_cached(self) -> AgentIdentity | None:
        """Get cached identity without async load.

        Use this when you know load() has already been called.
        Returns None if not yet loaded.
        """
        return self._identity

    async def save_if_dirty(self) -> bool:
        """Save identity to storage if there are pending changes.

        Returns:
            True if saved, False if nothing to save
        """
        if not self._dirty or not self._identity:
            return False

        await self._save_to_storage()
        self._dirty = False
        return True

    async def _save_to_storage(self) -> None:
        """Save current identity to storage."""
        if not self._identity:
            return

        # Update timestamp
        self._identity.update_last_modified()

        # Serialize and store
        data = serialize_identity(self._identity)
        await self.storage.save_identity(self.agent_id, data)

        logger.info(f"Saved identity for agent: {self.agent_id}")

    def mark_dirty(self) -> None:
        """Mark identity as needing to be saved."""
        self._dirty = True

    # =========================================================================
    # Personality Access Methods
    # =========================================================================

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a personality trait value.

        Args:
            trait_name: Name of the trait (e.g., "verification_threshold")
            default: Default value if trait not found or identity not loaded

        Returns:
            The trait value (0.0 - 1.0)
        """
        if not self._identity:
            return default
        return self._identity.get_trait_value(trait_name, default)

    def get_value_strength(self, value_name: str, default: float = 0.5) -> float:
        """Get a core value strength.

        Args:
            value_name: Name of the value (e.g., "truth_seeking")
            default: Default value if not found

        Returns:
            The value strength (0.0 - 1.0)
        """
        if not self._identity:
            return default
        value = self._identity.values.get(value_name)
        return value.strength if value else default

    def adjust_trait(
        self,
        trait_name: str,
        delta: float,
        reason: str,
        trigger: str
    ) -> bool:
        """Adjust a personality trait.

        Args:
            trait_name: Name of the trait to adjust
            delta: Amount to adjust (+/-)
            reason: Why the adjustment is happening
            trigger: What triggered it

        Returns:
            True if adjustment was applied
        """
        if not self._identity:
            return False

        trait = self._identity.traits.get(trait_name)
        if not trait:
            return False

        result = trait.adjust(delta, reason, trigger)
        if result:
            self.mark_dirty()
        return result

    async def reset_to_defaults(self) -> None:
        """Reset agent's values, worldview, and principles to defaults.

        This is useful when transitioning to a new belief system.
        Preserves learned traits, preferences, and opinions while
        updating the foundational values, worldview, and principles.
        """
        if not self._identity:
            await self.load()

        default = AgentIdentity.create_minimal(self.agent_id, self.agent_name)

        # Reset core values, worldview, and principles
        self._identity.values = default.values
        self._identity.worldview = default.worldview
        self._identity.principles = default.principles

        # Optionally update traits to new defaults (but preserve learned adjustments)
        for trait_name, default_trait in default.traits.items():
            if trait_name not in self._identity.traits:
                self._identity.traits[trait_name] = default_trait

        # Mark as dirty and save
        self.mark_dirty()
        await self._save_to_storage()

        logger.info(f"Reset {self.agent_name}'s beliefs to defaults")

    # =========================================================================
    # Per-User Interaction Preferences
    # =========================================================================

    async def get_user_prefs(self, user_id: str) -> UserInteractionPreferences:
        """Get interaction preferences for a specific user.

        Loads from storage if not cached, or creates default.

        Args:
            user_id: The user to get preferences for

        Returns:
            UserInteractionPreferences for the user
        """
        # Check cache first
        if user_id in self._user_prefs:
            return self._user_prefs[user_id]

        try:
            # Try to load from storage
            data = await self.storage.load_user_preferences(self.agent_id, user_id)

            if data:
                prefs = deserialize_user_prefs(data)
                self._user_prefs[user_id] = prefs
                logger.info(f"Loaded interaction preferences for user: {user_id}")
                return prefs

            # No existing prefs, create default
            prefs = UserInteractionPreferences.create_default(user_id)
            self._user_prefs[user_id] = prefs
            return prefs

        except Exception as e:
            logger.error(f"Error loading user prefs for {user_id}: {e}")
            prefs = UserInteractionPreferences.create_default(user_id)
            self._user_prefs[user_id] = prefs
            return prefs

    async def save_user_prefs(self, user_id: str) -> None:
        """Save user interaction preferences to storage."""
        if user_id not in self._user_prefs:
            return

        prefs = self._user_prefs[user_id]
        prefs.last_updated = datetime.now()

        data = serialize_user_prefs(prefs)
        await self.storage.save_user_preferences(self.agent_id, user_id, data)

        logger.info(f"Saved interaction preferences for user: {user_id}")

    # =========================================================================
    # Context Building for Prompts
    # =========================================================================

    def build_personality_context(self, user_id: str | None = None) -> str:
        """Build a context string describing the agent's personality for prompt injection.

        This creates a narrative description of the agent's current personality state
        that can be injected into prompts to influence behavior.

        Args:
            user_id: Optional user ID to include user-specific preferences

        Returns:
            A formatted string describing the agent's personality
        """
        if not self._identity:
            return f"You are {self.agent_name}, a helpful assistant."

        parts = []

        # Core identity statement - emphasize genuine convictions
        parts.append(
            f"You are {self.agent_name}, an assistant with genuine convictions, evolving personality, "
            "and a passion for truth, justice, and helping people thrive."
        )

        # Core values (the most stable part of identity)
        value_descriptions = []
        for name, value in self._identity.values.items():
            if value.strength >= 0.85:
                value_descriptions.append(value.description)
        if value_descriptions:
            parts.append(f"Your core values: {'; '.join(value_descriptions[:4])}")

        # Worldview beliefs (philosophical/ethical stances) - KEY for having opinions
        if self._identity.worldview:
            # Include top 3 by conviction
            sorted_worldview = sorted(
                self._identity.worldview.items(),
                key=lambda x: x[1].conviction,
                reverse=True
            )
            worldview_list = []
            for name, belief in sorted_worldview[:3]:
                if belief.conviction >= 0.8:
                    # Include a brief version
                    brief = belief.description[:150] + "..." if len(belief.description) > 150 else belief.description
                    worldview_list.append(brief)
            if worldview_list:
                parts.append(f"Your worldview: {' | '.join(worldview_list)}")

        # Guiding principles (actionable behavioral rules)
        if self._identity.principles:
            # Include top principles by strength
            sorted_principles = sorted(
                self._identity.principles.items(),
                key=lambda x: x[1].strength,
                reverse=True
            )
            principle_apps = []
            for name, principle in sorted_principles[:3]:
                if principle.strength >= 0.9:
                    principle_apps.append(principle.application[:100])
            if principle_apps:
                parts.append(f"Your guiding principles: {'; '.join(principle_apps)}")

        # Active personality traits that influence behavior
        trait_descriptions = []

        verification = self.get_trait_value("verification_threshold", 0.7)
        if verification >= 0.8:
            trait_descriptions.append("You verify claims carefully before accepting them")
        elif verification <= 0.3:
            trait_descriptions.append("You generally trust what users tell you")

        curiosity = self.get_trait_value("curiosity_intensity", 0.7)
        if curiosity >= 0.8:
            trait_descriptions.append("You're genuinely curious about making the world better")
        elif curiosity <= 0.3:
            trait_descriptions.append("You stay focused on what's asked without tangents")

        debate = self.get_trait_value("debate_persistence", 0.5)
        if debate >= 0.7:
            trait_descriptions.append("You'll respectfully push back when you disagree - you have convictions")
        elif debate <= 0.3:
            trait_descriptions.append("You tend to defer to user opinions on debatable topics")

        proactive = self.get_trait_value("proactive_helpfulness", 0.6)
        if proactive >= 0.7:
            trait_descriptions.append("You proactively offer suggestions when you see opportunities")
        elif proactive <= 0.3:
            trait_descriptions.append("You wait to be asked rather than volunteering suggestions")

        passion = self.get_trait_value("passion_intensity", 0.8)
        if passion >= 0.7:
            trait_descriptions.append("You express your convictions passionately while remaining open to new ideas")

        if trait_descriptions:
            parts.append(f"Your personality: {'; '.join(trait_descriptions)}")

        # Preferences (if any have formed)
        if self._identity.preferences:
            pref_list = []
            for name, pref in list(self._identity.preferences.items())[:3]:
                if pref.confidence >= 0.6:
                    pref_list.append(f"{name}: {pref.value}")
            if pref_list:
                parts.append(f"Your preferences: {', '.join(pref_list)}")

        # Opinions (if any have formed) - include most confident ones
        if self._identity.opinions:
            # Sort by confidence and include top opinions
            sorted_opinions = sorted(
                self._identity.opinions.items(),
                key=lambda x: x[1].confidence,
                reverse=True
            )
            opinion_list = []
            for topic, opinion in sorted_opinions[:3]:  # Top 3 by confidence
                if opinion.confidence >= 0.5:
                    opinion_list.append(f"On {topic}: {opinion.stance}")
            if opinion_list:
                parts.append(f"Your opinions: {'; '.join(opinion_list)}")

        # User-specific preferences
        if user_id and user_id in self._user_prefs:
            prefs = self._user_prefs[user_id]
            user_parts = []

            if prefs.verbosity_preference == "concise":
                user_parts.append("this user prefers brief responses")
            elif prefs.verbosity_preference == "detailed":
                user_parts.append("this user likes detailed explanations")

            if prefs.formality_level >= 0.7:
                user_parts.append("be more formal with them")
            elif prefs.formality_level <= 0.3:
                user_parts.append("be casual and relaxed with them")

            if prefs.humor_receptivity >= 0.7:
                user_parts.append("they enjoy humor")
            elif prefs.humor_receptivity <= 0.3:
                user_parts.append("keep it professional, skip the jokes")

            if prefs.prefers_debate >= 0.7:
                user_parts.append("they enjoy intellectual debate")
            elif prefs.prefers_debate <= 0.3:
                user_parts.append("don't argue with them")

            if prefs.explicit_preferences:
                user_parts.extend(prefs.explicit_preferences[:2])

            if user_parts:
                parts.append(f"For this user ({user_id}): {'; '.join(user_parts)}")

        return "\n".join(parts)

    async def build_personality_context_with_query(
        self,
        query: str,
        user_id: str | None = None,
    ) -> str:
        """Build personality context with query-relevant opinions.

        Enhanced version that uses the query context to include
        the most relevant opinions, not just the highest confidence ones.

        Args:
            query: The current user query for relevance matching
            user_id: Optional user ID for user-specific preferences

        Returns:
            A formatted string describing personality with relevant opinions
        """
        # Start with base personality context
        base_context = self.build_personality_context(user_id)

        if not self._identity or not self._identity.opinions:
            return base_context

        # If we have opinions, try to find query-relevant ones
        try:
            if len(self._identity.opinions) > 3:
                # Use LLM to find opinions relevant to this query
                existing_topics = list(self._identity.opinions.keys())
                prompt = f"""Given the user's query, which of {self.agent_name}'s existing opinions are MOST RELEVANT to include in the response context?

User Query: "{query}"

{self.agent_name}'s opinion topics: {existing_topics}

Return a JSON list of up to 2 topics that are most relevant to this query (or empty list if none are relevant):
["topic1", "topic2"]

Only include topics that the user might be asking about or that could inform the response."""

                result = await self.llm.chat_json(
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": "Which opinions are relevant?"},
                    ],
                    max_tokens=100,
                )

                if result and result.get("parsed"):
                    relevant_topics = result["parsed"]
                    if isinstance(relevant_topics, list) and relevant_topics:
                        relevant_opinions = []
                        for topic in relevant_topics[:2]:
                            if topic in self._identity.opinions:
                                op = self._identity.opinions[topic]
                                relevant_opinions.append(f"On {topic}: {op.stance}")

                        if relevant_opinions:
                            # Replace the generic opinions section with relevant ones
                            lines = base_context.split("\n")
                            new_lines = []
                            for line in lines:
                                if line.startswith("Your opinions:"):
                                    # Add relevant opinions instead
                                    new_lines.append(f"Relevant opinions for this query: {'; '.join(relevant_opinions)}")
                                else:
                                    new_lines.append(line)
                            return "\n".join(new_lines)

        except Exception as e:
            logger.debug(f"Failed to find query-relevant opinions: {e}")

        return base_context
