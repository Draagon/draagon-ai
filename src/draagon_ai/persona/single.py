"""SinglePersonaManager for applications with one identity.

This is the simplest PersonaManager - there's one persona that
handles all interactions. Perfect for voice assistants like Roxy.
"""

from typing import Any

from draagon_ai.persona.base import Persona, PersonaManager


class SinglePersonaManager(PersonaManager):
    """Manager for a single persistent persona.

    This is the appropriate choice for:
    - Voice assistants (Roxy)
    - Personal AI companions
    - Branded AI interfaces
    - Any application with one consistent identity

    Example:
        roxy = Persona(
            id="roxy",
            name="Roxy",
            description="A helpful voice assistant for smart homes",
            traits=PersonaTraits.from_dict({"friendly": 0.8, "concise": 0.7}),
        )
        manager = SinglePersonaManager(roxy)

        # Always returns the same persona
        persona = await manager.get_active_persona({})
    """

    def __init__(self, persona: Persona) -> None:
        """Initialize with the single persona.

        Args:
            persona: The one persona this manager will provide.
        """
        self._persona = persona

    @property
    def persona(self) -> Persona:
        """Direct access to the managed persona."""
        return self._persona

    async def get_active_persona(self, context: dict[str, Any]) -> Persona:
        """Return the single persona (context is ignored)."""
        return self._persona

    async def list_personas(self) -> list[Persona]:
        """Return list containing just our persona."""
        return [self._persona]

    async def get_persona(self, persona_id: str) -> Persona | None:
        """Return our persona if ID matches, else None."""
        if persona_id == self._persona.id:
            return self._persona
        return None

    async def update_persona(self, persona: Persona) -> None:
        """Replace the managed persona.

        This is useful when loading from persistent storage or
        when the persona's identity evolves.
        """
        self._persona = persona
