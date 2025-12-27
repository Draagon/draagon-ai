"""Persona system for Draagon AI.

The Persona system provides a clean abstraction for AI identities:
- Persona: A coherent identity an agent can embody
- PersonaManager: Manages active persona(s) for an application
- SinglePersonaManager: For apps with one identity (e.g., a voice assistant)
- MultiPersonaManager: For apps with multiple identities (e.g., PartyLore NPCs)
"""

from draagon_ai.persona.base import (
    Persona,
    PersonaTraits,
    PersonaRelationship,
    PersonaManager,
)
from draagon_ai.persona.single import SinglePersonaManager
from draagon_ai.persona.multi import MultiPersonaManager

__all__ = [
    # Core types
    "Persona",
    "PersonaTraits",
    "PersonaRelationship",
    # Abstract base
    "PersonaManager",
    # Implementations
    "SinglePersonaManager",
    "MultiPersonaManager",
]
