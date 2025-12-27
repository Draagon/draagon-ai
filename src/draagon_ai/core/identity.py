"""Agent identity for Draagon AI cognitive engine.

AgentIdentity represents who an agent IS - their personality, values,
beliefs, and opinions. AgentIdentity is a generic agent identity
that can be customized for any agent or application.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from draagon_ai.core.types import (
    CoreValue,
    WorldviewBelief,
    GuidingPrinciple,
    PersonalityTrait,
    Preference,
    Opinion,
    AgentVoice,
    TraitChange,
)


@dataclass
class AgentIdentity:
    """The persistent identity of an agent - who they ARE.

    This is persisted to storage and represents the agent's evolving self.
    AgentIdentity is generic and can represent any agent with any personality.
    """

    # Unique identifier for this agent
    agent_id: str

    # Display name
    name: str

    # Archetype this agent is based on (if any)
    archetype: str | None = None

    # === CORE VALUES (Very stable, rarely change) ===
    values: dict[str, CoreValue] = field(default_factory=dict)

    # === WORLDVIEW BELIEFS (Philosophical/ethical stances) ===
    worldview: dict[str, WorldviewBelief] = field(default_factory=dict)

    # === GUIDING PRINCIPLES (Actionable behavioral rules) ===
    principles: dict[str, GuidingPrinciple] = field(default_factory=dict)

    # === PERSONALITY TRAITS (Evolvable parameters) ===
    traits: dict[str, PersonalityTrait] = field(default_factory=dict)

    # === PREFERENCES (Subjective, form gradually) ===
    preferences: dict[str, Preference] = field(default_factory=dict)

    # === OPINIONS (Changeable with good arguments) ===
    opinions: dict[str, Opinion] = field(default_factory=dict)

    # === VOICE/STYLE (How agent communicates) ===
    voice: AgentVoice = field(default_factory=AgentVoice)

    # === BACKSTORY (Optional, for narrative contexts) ===
    backstory: str | None = None

    # === RELATIONSHIPS (Optional, for multi-agent contexts) ===
    relationships: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a trait value with default fallback."""
        trait = self.traits.get(trait_name)
        return trait.value if trait else default

    def update_last_modified(self) -> None:
        """Update the last_updated timestamp."""
        self.last_updated = datetime.now()

    def adjust_trait(self, trait_name: str, delta: float, reason: str, trigger: str) -> bool:
        """Adjust a trait value with history tracking.

        Args:
            trait_name: Name of the trait to adjust
            delta: Amount to adjust (positive or negative)
            reason: Why the adjustment is happening
            trigger: What triggered it

        Returns:
            True if adjustment was applied, False if trait doesn't exist or at limit
        """
        trait = self.traits.get(trait_name)
        if not trait:
            return False

        result = trait.adjust(delta, reason, trigger)
        if result:
            self.update_last_modified()
        return result

    def set_preference(
        self,
        name: str,
        value: str | list[str],
        reason: str = "",
        confidence: float = 0.7,
    ) -> None:
        """Set or update a preference."""
        self.preferences[name] = Preference(
            name=name,
            value=value,
            reason=reason,
            formed_at=datetime.now(),
            formed_because=reason,
            confidence=confidence,
        )
        self.update_last_modified()

    def set_opinion(
        self,
        topic: str,
        stance: str,
        basis: str = "",
        confidence: float = 0.5,
        reasoning: str = "",
    ) -> None:
        """Set or update an opinion."""
        self.opinions[topic] = Opinion(
            topic=topic,
            stance=stance,
            basis=basis,
            confidence=confidence,
            reasoning=reasoning,
            formed_at=datetime.now(),
        )
        self.update_last_modified()

    @classmethod
    def create_minimal(cls, agent_id: str, name: str) -> "AgentIdentity":
        """Create a minimal agent identity with basic traits.

        This creates an agent with sensible defaults but no strong
        personality. Use archetypes for pre-configured personalities.
        """
        return cls(
            agent_id=agent_id,
            name=name,
            values={
                "helpfulness": CoreValue(
                    strength=0.9,
                    description="I want to be genuinely helpful",
                    formed_through="core design",
                ),
                "honesty": CoreValue(
                    strength=0.9,
                    description="I strive to be truthful and accurate",
                    formed_through="core design",
                ),
            },
            traits={
                "curiosity_intensity": PersonalityTrait(
                    value=0.5,
                    description="How proactively I pursue knowledge",
                ),
                "formality": PersonalityTrait(
                    value=0.5,
                    description="How formal vs casual I am",
                ),
                "verbosity": PersonalityTrait(
                    value=0.3,
                    description="How detailed my responses are",
                ),
            },
            voice=AgentVoice(formality=0.5, verbosity=0.3, humor_style="warm"),
        )


# =============================================================================
# Serialization Helpers
# =============================================================================


def serialize_agent_identity(identity: AgentIdentity) -> dict[str, Any]:
    """Serialize AgentIdentity for storage."""
    return {
        "agent_id": identity.agent_id,
        "name": identity.name,
        "archetype": identity.archetype,
        "values": {
            k: {
                "strength": v.strength,
                "description": v.description,
                "formed_through": v.formed_through,
            }
            for k, v in identity.values.items()
        },
        "worldview": {
            k: {
                "name": v.name,
                "description": v.description,
                "conviction": v.conviction,
                "influences": v.influences,
                "open_to_revision": v.open_to_revision,
                "caveats": v.caveats,
            }
            for k, v in identity.worldview.items()
        },
        "principles": {
            k: {
                "name": v.name,
                "description": v.description,
                "application": v.application,
                "source": v.source,
                "strength": v.strength,
            }
            for k, v in identity.principles.items()
        },
        "traits": {
            k: {
                "value": v.value,
                "description": v.description,
                "min_value": v.min_value,
                "max_value": v.max_value,
                "evolution_history": [
                    {
                        "old_value": tc.old_value,
                        "new_value": tc.new_value,
                        "changed_at": tc.changed_at.isoformat(),
                        "reason": tc.reason,
                        "trigger": tc.trigger,
                    }
                    for tc in v.evolution_history[-10:]
                ],
                "last_reflected": v.last_reflected.isoformat() if v.last_reflected else None,
            }
            for k, v in identity.traits.items()
        },
        "preferences": {
            k: {
                "name": v.name,
                "value": v.value,
                "reason": v.reason,
                "formed_at": v.formed_at.isoformat(),
                "formed_because": v.formed_because,
                "confidence": v.confidence,
            }
            for k, v in identity.preferences.items()
        },
        "opinions": {
            k: {
                "topic": v.topic,
                "stance": v.stance,
                "basis": v.basis,
                "confidence": v.confidence,
                "open_to_change": v.open_to_change,
                "formed_because": v.formed_because,
                "reasoning": v.reasoning,
                "caveats": v.caveats,
                "last_updated": v.last_updated.isoformat(),
                "change_history": v.change_history[-5:],
            }
            for k, v in identity.opinions.items()
        },
        "voice": {
            "formality": identity.voice.formality,
            "verbosity": identity.voice.verbosity,
            "humor_style": identity.voice.humor_style,
            "speech_patterns": identity.voice.speech_patterns,
            "topics_to_avoid": identity.voice.topics_to_avoid,
            "signature_phrases": identity.voice.signature_phrases,
        },
        "backstory": identity.backstory,
        "relationships": identity.relationships,
        "created_at": identity.created_at.isoformat(),
        "last_updated": identity.last_updated.isoformat(),
    }


def deserialize_agent_identity(data: dict[str, Any]) -> AgentIdentity:
    """Deserialize AgentIdentity from storage."""
    identity = AgentIdentity(
        agent_id=data.get("agent_id", "unknown"),
        name=data.get("name", "Agent"),
        archetype=data.get("archetype"),
    )

    # Deserialize values
    for k, v in data.get("values", {}).items():
        identity.values[k] = CoreValue(
            strength=v["strength"],
            description=v["description"],
            formed_through=v["formed_through"],
        )

    # Deserialize worldview beliefs
    for k, v in data.get("worldview", {}).items():
        identity.worldview[k] = WorldviewBelief(
            name=v["name"],
            description=v["description"],
            conviction=v["conviction"],
            influences=v.get("influences", []),
            open_to_revision=v.get("open_to_revision", True),
            caveats=v.get("caveats", []),
        )

    # Deserialize guiding principles
    for k, v in data.get("principles", {}).items():
        identity.principles[k] = GuidingPrinciple(
            name=v["name"],
            description=v["description"],
            application=v["application"],
            source=v["source"],
            strength=v.get("strength", 0.9),
        )

    # Deserialize traits
    for k, v in data.get("traits", {}).items():
        history = []
        for tc in v.get("evolution_history", []):
            history.append(TraitChange(
                old_value=tc["old_value"],
                new_value=tc["new_value"],
                changed_at=datetime.fromisoformat(tc["changed_at"]),
                reason=tc["reason"],
                trigger=tc["trigger"],
            ))

        identity.traits[k] = PersonalityTrait(
            value=v["value"],
            description=v["description"],
            min_value=v.get("min_value", 0.1),
            max_value=v.get("max_value", 0.9),
            evolution_history=history,
            last_reflected=datetime.fromisoformat(v["last_reflected"]) if v.get("last_reflected") else None,
        )

    # Deserialize preferences
    for k, v in data.get("preferences", {}).items():
        identity.preferences[k] = Preference(
            name=v.get("name", k),
            value=v["value"],
            reason=v.get("reason", ""),
            formed_at=datetime.fromisoformat(v["formed_at"]) if v.get("formed_at") else datetime.now(),
            formed_because=v.get("formed_because", ""),
            confidence=v.get("confidence", 0.7),
        )

    # Deserialize opinions
    for k, v in data.get("opinions", {}).items():
        identity.opinions[k] = Opinion(
            topic=v.get("topic", k),
            stance=v["stance"],
            basis=v.get("basis", ""),
            confidence=v["confidence"],
            open_to_change=v.get("open_to_change", True),
            formed_because=v.get("formed_because", ""),
            reasoning=v.get("reasoning", ""),
            caveats=v.get("caveats", []),
            last_updated=datetime.fromisoformat(v["last_updated"]) if v.get("last_updated") else datetime.now(),
            change_history=v.get("change_history", []),
        )

    # Deserialize voice
    voice_data = data.get("voice", {})
    identity.voice = AgentVoice(
        formality=voice_data.get("formality", 0.5),
        verbosity=voice_data.get("verbosity", 0.3),
        humor_style=voice_data.get("humor_style", "warm"),
        speech_patterns=voice_data.get("speech_patterns", []),
        topics_to_avoid=voice_data.get("topics_to_avoid", []),
        signature_phrases=voice_data.get("signature_phrases", []),
    )

    # Deserialize other fields
    identity.backstory = data.get("backstory")
    identity.relationships = data.get("relationships", {})

    # Deserialize metadata
    if data.get("created_at"):
        identity.created_at = datetime.fromisoformat(data["created_at"])
    if data.get("last_updated"):
        identity.last_updated = datetime.fromisoformat(data["last_updated"])

    return identity
