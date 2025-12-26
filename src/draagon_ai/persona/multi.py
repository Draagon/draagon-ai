"""MultiPersonaManager for applications with multiple identities.

This manages multiple personas that can be active at different times.
Perfect for RPG games with NPCs, multi-character narratives, etc.
"""

from typing import Any

from draagon_ai.persona.base import Persona, PersonaManager


class MultiPersonaManager(PersonaManager):
    """Manager for multiple personas that can switch between active states.

    This is the appropriate choice for:
    - RPG games with NPCs (PartyLore)
    - Multi-character narratives
    - Role-playing scenarios
    - Any application where different identities speak

    Example:
        blacksmith = Persona(
            id="grumak",
            name="Grumak",
            description="An aging orc blacksmith",
            traits=PersonaTraits.from_dict({"gruff": 0.8, "honest": 0.9}),
            voice_notes="Short sentences. Drops articles. Speaks slowly.",
        )

        sage = Persona(
            id="elara",
            name="Elara",
            description="An elven scholar of ancient texts",
            traits=PersonaTraits.from_dict({"mysterious": 0.9, "verbose": 0.7}),
            voice_notes="Flowery language. References obscure texts.",
        )

        manager = MultiPersonaManager(
            personas=[blacksmith, sage],
            default_id="grumak",
        )

        # Get the active (or default) persona
        persona = await manager.get_active_persona({})

        # Switch to a different persona
        await manager.switch_to("elara")
    """

    def __init__(
        self,
        personas: list[Persona] | None = None,
        default_id: str | None = None,
    ) -> None:
        """Initialize with a list of personas.

        Args:
            personas: Initial list of personas to manage.
            default_id: ID of the default persona. If not specified,
                       uses the first persona in the list.
        """
        self._personas: dict[str, Persona] = {}
        self._active_id: str | None = None
        self._default_id: str | None = default_id

        if personas:
            for persona in personas:
                self.add_persona(persona)

            if default_id is None and personas:
                self._default_id = personas[0].id
                self._active_id = personas[0].id
            elif default_id and default_id in self._personas:
                self._active_id = default_id

    def add_persona(self, persona: Persona) -> None:
        """Add a persona to the manager.

        Args:
            persona: The persona to add.
        """
        self._personas[persona.id] = persona

        # If this is the first persona, make it active
        if len(self._personas) == 1:
            self._active_id = persona.id
            if self._default_id is None:
                self._default_id = persona.id

    def remove_persona(self, persona_id: str) -> Persona | None:
        """Remove a persona from the manager.

        Args:
            persona_id: ID of the persona to remove.

        Returns:
            The removed persona, or None if not found.
        """
        persona = self._personas.pop(persona_id, None)

        # If we removed the active persona, reset to default or first available
        if persona_id == self._active_id:
            if self._default_id and self._default_id in self._personas:
                self._active_id = self._default_id
            elif self._personas:
                self._active_id = next(iter(self._personas.keys()))
            else:
                self._active_id = None

        return persona

    async def get_active_persona(self, context: dict[str, Any]) -> Persona:
        """Get the currently active persona.

        Args:
            context: Interaction context (can include hints about
                    which persona should respond).

        Returns:
            The active persona.

        Raises:
            ValueError: If no personas are available.
        """
        # Check if context suggests a specific persona
        persona_id = context.get("persona_id") or context.get("npc_id")
        if persona_id and persona_id in self._personas:
            self._active_id = persona_id
            return self._personas[persona_id]

        # Return the active persona
        if self._active_id and self._active_id in self._personas:
            return self._personas[self._active_id]

        # Fall back to default
        if self._default_id and self._default_id in self._personas:
            self._active_id = self._default_id
            return self._personas[self._default_id]

        # Last resort: any persona
        if self._personas:
            persona = next(iter(self._personas.values()))
            self._active_id = persona.id
            return persona

        raise ValueError("No personas available")

    async def list_personas(self) -> list[Persona]:
        """List all managed personas."""
        return list(self._personas.values())

    async def get_persona(self, persona_id: str) -> Persona | None:
        """Get a specific persona by ID."""
        return self._personas.get(persona_id)

    async def switch_to(self, persona_id: str) -> Persona | None:
        """Switch the active persona.

        Args:
            persona_id: ID of the persona to make active.

        Returns:
            The newly active persona, or None if not found.
        """
        persona = self._personas.get(persona_id)
        if persona:
            self._active_id = persona_id
            return persona
        return None

    @property
    def active_id(self) -> str | None:
        """Get the ID of the currently active persona."""
        return self._active_id

    @property
    def default_id(self) -> str | None:
        """Get the ID of the default persona."""
        return self._default_id

    def set_default(self, persona_id: str) -> bool:
        """Set the default persona.

        Args:
            persona_id: ID of the persona to make default.

        Returns:
            True if successful, False if persona not found.
        """
        if persona_id in self._personas:
            self._default_id = persona_id
            return True
        return False

    def get_relationship(
        self, from_id: str, to_id: str
    ) -> "PersonaRelationship | None":
        """Get the relationship between two personas.

        Args:
            from_id: ID of the persona whose perspective we want.
            to_id: ID of the target persona.

        Returns:
            The relationship, or None if not found.
        """
        from_persona = self._personas.get(from_id)
        if from_persona:
            return from_persona.relationships.get(to_id)
        return None


# Import at end to avoid circular import
from draagon_ai.persona.base import PersonaRelationship  # noqa: E402, F811
