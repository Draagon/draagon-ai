"""Base Persona types and abstract PersonaManager.

Persona represents a coherent AI identity that can be embodied.
PersonaManager is the abstract base for managing active personas.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from draagon_ai.core.identity import AgentIdentity


@dataclass
class PersonaTraits:
    """Simplified personality traits for a Persona.

    These are the key traits that affect how the persona interacts.
    For more complex cognitive traits, use the full AgentIdentity.
    """

    # Core interaction style (0.0 = low, 1.0 = high)
    traits: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        """Get a trait value, default 0.5 if not set."""
        return self.traits.get(key, 0.5)

    def __setitem__(self, key: str, value: float) -> None:
        """Set a trait value, clamped to 0.0-1.0."""
        self.traits[key] = max(0.0, min(1.0, value))

    def get(self, key: str, default: float = 0.5) -> float:
        """Get a trait value with custom default."""
        return self.traits.get(key, default)

    def items(self) -> list[tuple[str, float]]:
        """Get all trait key-value pairs."""
        return list(self.traits.items())

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> "PersonaTraits":
        """Create PersonaTraits from a dictionary."""
        return cls(traits=dict(data))


@dataclass
class PersonaRelationship:
    """Relationship between personas (for multi-persona systems)."""

    target_id: str  # ID of the persona this relationship is with
    relationship_type: str  # e.g., "ally", "rival", "mentor", "student"
    affinity: float = 0.5  # -1.0 (hostile) to 1.0 (friendly)
    trust: float = 0.5  # 0.0 (distrust) to 1.0 (full trust)
    history: str = ""  # Brief history of the relationship
    notes: list[str] = field(default_factory=list)  # Dynamic notes


@dataclass
class Persona:
    """A coherent identity an AI agent can embody.

    Persona is the user-facing identity concept. It can optionally
    be backed by a full AgentIdentity for cognitive architecture.

    For simple use cases:
        persona = Persona(id="assistant", name="Assistant", description="A helpful assistant")

    For complex cognitive agents:
        persona = Persona(
            id="guide",
            name="Guide",
            identity=my_agent_identity,  # Full cognitive architecture
        )
    """

    # Required: unique identifier
    id: str

    # Required: display name
    name: str

    # Brief description of who this persona is
    description: str = ""

    # Simplified traits (for quick configuration)
    traits: PersonaTraits = field(default_factory=PersonaTraits)

    # Voice/style guidance for the LLM
    voice_notes: str = ""

    # Knowledge scope tags (for filtering memories/knowledge)
    knowledge_tags: list[str] = field(default_factory=list)

    # Relationships with other personas
    relationships: dict[str, PersonaRelationship] = field(default_factory=dict)

    # Full cognitive identity (optional, for complex agents)
    identity: AgentIdentity | None = None

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime | None = None

    def get_trait(self, name: str, default: float = 0.5) -> float:
        """Get a trait value, checking both simple traits and full identity."""
        # First check simple traits
        if self.traits.traits and name in self.traits.traits:
            return self.traits[name]

        # Then check full identity if available
        if self.identity:
            return self.identity.get_trait_value(name, default)

        return default

    def get_system_prompt_context(self) -> str:
        """Generate context for inclusion in LLM system prompts.

        This provides the persona's identity information for the LLM.
        """
        parts = [f"You are {self.name}."]

        if self.description:
            parts.append(self.description)

        if self.voice_notes:
            parts.append(f"\nVoice/Style: {self.voice_notes}")

        if self.traits.traits:
            trait_desc = ", ".join(
                f"{k}: {v:.1f}" for k, v in self.traits.items()
            )
            parts.append(f"\nPersonality traits: {trait_desc}")

        if self.identity and self.identity.backstory:
            parts.append(f"\nBackstory: {self.identity.backstory}")

        return "\n".join(parts)

    def mark_active(self) -> None:
        """Update the last_active timestamp."""
        self.last_active = datetime.now()

    def add_relationship(
        self,
        target_id: str,
        relationship_type: str,
        affinity: float = 0.5,
        trust: float = 0.5,
        history: str = "",
    ) -> PersonaRelationship:
        """Add or update a relationship with another persona."""
        rel = PersonaRelationship(
            target_id=target_id,
            relationship_type=relationship_type,
            affinity=affinity,
            trust=trust,
            history=history,
        )
        self.relationships[target_id] = rel
        return rel


class PersonaContext(Protocol):
    """Protocol for persona context passed during interactions.

    This allows PersonaManager implementations to receive context
    about the current interaction when determining which persona
    should be active.
    """

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        ...


class PersonaManager(ABC):
    """Abstract base for managing active persona(s).

    PersonaManager determines which persona is active for a given
    interaction context. Different implementations support different
    use cases:

    - SinglePersonaManager: One persistent identity (e.g., a voice assistant)
    - MultiPersonaManager: Multiple identities (e.g., PartyLore NPCs)
    """

    @abstractmethod
    async def get_active_persona(self, context: dict[str, Any]) -> Persona:
        """Get the currently active persona for the given context.

        Args:
            context: Interaction context (user_id, conversation_id, etc.)

        Returns:
            The persona that should handle this interaction.
        """
        ...

    @abstractmethod
    async def list_personas(self) -> list[Persona]:
        """List all available personas."""
        ...

    @abstractmethod
    async def get_persona(self, persona_id: str) -> Persona | None:
        """Get a specific persona by ID."""
        ...

    async def switch_to(self, persona_id: str) -> Persona | None:
        """Switch the active persona (for managers that support it).

        Default implementation returns None (not supported).
        Override in subclasses that support switching.
        """
        return None

    async def on_interaction_start(
        self, persona: Persona, context: dict[str, Any]
    ) -> None:
        """Called when an interaction starts with a persona.

        Override to add custom behavior like logging or analytics.
        """
        persona.mark_active()

    async def on_interaction_end(
        self, persona: Persona, context: dict[str, Any]
    ) -> None:
        """Called when an interaction ends.

        Override for cleanup or state persistence.
        """
        pass
