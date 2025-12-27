"""Personality module for behavior customization.

This module provides types and utilities for defining agent personalities
that can be composed into behavior prompts.

Usage:
    from draagon_ai.behaviors.personality import (
        PersonalityConfig,
        CoreValue,
        ValueConfig,
        TraitDimension,
        Opinion,
        Principle,
        HumorStyle,
        compose_personality_intro,
    )

    # Define a personality
    personality = PersonalityConfig(
        name="Roxy",
        description="A curious, opinionated voice assistant",
        values=[
            ValueConfig(CoreValue.TRUTH_SEEKING, 0.9),
            ValueConfig(CoreValue.INTERDEPENDENCE, 0.8),
            ValueConfig(CoreValue.HELPFULNESS, 0.9),
        ],
        traits={
            TraitDimension.WARMTH: 0.8,
            TraitDimension.CURIOSITY: 0.9,
            TraitDimension.PASSION: 0.8,
        },
        opinions=[
            Opinion("pineapple_pizza", "I'm a fan", "sweet and savory works!"),
            Opinion("healthcare", "It's a human right", "everyone deserves access"),
        ],
        humor_style=HumorStyle.WARM,
    )

    # Compose into prompt text
    intro = compose_personality_intro(personality)
"""

from .types import (
    CoreValue,
    HumorStyle,
    Opinion,
    PersonalityConfig,
    Principle,
    TraitConfig,
    TraitDimension,
    ValueConfig,
)
from .composer import compose_personality_intro, compose_opinion_prompt

__all__ = [
    # Types
    "CoreValue",
    "TraitDimension",
    "HumorStyle",
    "ValueConfig",
    "TraitConfig",
    "Opinion",
    "Principle",
    "PersonalityConfig",
    # Composers
    "compose_personality_intro",
    "compose_opinion_prompt",
]
