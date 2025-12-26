"""Personality subsystem for Draagon AI.

This module provides:
- Archetype: Pre-built personality templates
- Identity management and serialization
"""

from draagon_ai.personality.archetypes import (
    Archetype,
    ARCHETYPES,
    get_archetype,
    list_archetypes,
    # Specific archetypes
    HELPFUL_ASSISTANT,
    PROFESSIONAL_ADVISOR,
    WISE_NARRATOR,
    GRUFF_INFORMANT,
    MYSTERIOUS_MENTOR,
    EMPATHETIC_SUPPORT,
    # The Roxy archetype (Doug's personal assistant)
    ROXY,
)

__all__ = [
    "Archetype",
    "ARCHETYPES",
    "get_archetype",
    "list_archetypes",
    "HELPFUL_ASSISTANT",
    "PROFESSIONAL_ADVISOR",
    "WISE_NARRATOR",
    "GRUFF_INFORMANT",
    "MYSTERIOUS_MENTOR",
    "EMPATHETIC_SUPPORT",
    "ROXY",
]
