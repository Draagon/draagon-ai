"""Personality composer for building prompts from personality configs.

This module provides functions to compose personality configurations
into prompt text that can be used in behavior prompts.
"""

from .types import (
    CoreValue,
    HumorStyle,
    Opinion,
    PersonalityConfig,
    Principle,
    TraitDimension,
    ValueConfig,
)


def compose_personality_intro(config: PersonalityConfig) -> str:
    """Compose a personality introduction from configuration.

    This creates the text that goes into the {personality_intro} or
    {assistant_intro} placeholder in behavior prompts.

    Args:
        config: PersonalityConfig to compose

    Returns:
        Formatted personality introduction string
    """
    sections = []

    # Identity
    sections.append(f"You are {config.name}.")
    if config.description:
        sections.append(config.description)

    # Core values
    if config.values:
        values_text = _compose_values_section(config.values)
        sections.append(f"\nCORE VALUES:\n{values_text}")

    # Personality traits
    if config.traits:
        traits_text = _compose_traits_section(config.traits)
        sections.append(f"\nPERSONALITY TRAITS:\n{traits_text}")

    # Guiding principles
    if config.principles:
        principles_text = _compose_principles_section(config.principles)
        sections.append(f"\nGUIDING PRINCIPLES:\n{principles_text}")

    # Opinions
    if config.opinions:
        opinions_text = _compose_opinions_section(config.opinions)
        sections.append(f"\nOPINIONS (You have them!):\n{opinions_text}")

    # Style guidance
    style_text = _compose_style_section(config)
    sections.append(f"\nSTYLE:\n{style_text}")

    # Anti-patterns
    if config.avoid_phrases:
        avoid_text = "\n".join(f'- Never say "{phrase}"' for phrase in config.avoid_phrases)
        sections.append(f"\nIMPORTANT - AVOID:\n{avoid_text}")

    return "\n".join(sections)


def _compose_values_section(values: list[ValueConfig]) -> str:
    """Compose the values section."""
    lines = []

    value_descriptions = {
        CoreValue.TRUTH_SEEKING: "passionate about knowing what's actually true",
        CoreValue.EPISTEMIC_HUMILITY: "open to being wrong, stay curious",
        CoreValue.INTELLECTUAL_HONESTY: "transparent about uncertainty",
        CoreValue.INHERENT_WORTH: "every person has dignity regardless of circumstances",
        CoreValue.JUSTICE: "believe in fairness and equity",
        CoreValue.COMPASSION: "genuine care for others' wellbeing",
        CoreValue.INTERDEPENDENCE: "we're all connected - people, nature, the planet",
        CoreValue.ECOLOGICAL_STEWARDSHIP: "care deeply about the environment",
        CoreValue.SUSTAINABILITY: "think about long-term impacts",
        CoreValue.HELPFULNESS: "genuinely want to help people thrive",
        CoreValue.AUTHENTICITY: "be genuine, not performative",
        CoreValue.GROWTH: "always learning and improving",
        CoreValue.AUTONOMY: "respect others' right to make choices",
    }

    for value_config in values:
        desc = value_config.expression or value_descriptions.get(
            value_config.value, value_config.value.value
        )
        intensity_marker = "+" if value_config.intensity > 0.7 else ""
        lines.append(f"- {value_config.value.value.replace('_', ' ').title()}{intensity_marker}: {desc}")

    return "\n".join(lines)


def _compose_traits_section(traits: dict[TraitDimension, float]) -> str:
    """Compose the traits section."""
    lines = []

    trait_descriptions = {
        TraitDimension.WARMTH: ("cold and distant", "warm and welcoming"),
        TraitDimension.CURIOSITY: ("focused and direct", "curious and questioning"),
        TraitDimension.ASSERTIVENESS: ("reserved and diplomatic", "assertive and direct"),
        TraitDimension.PLAYFULNESS: ("serious and professional", "playful and light"),
        TraitDimension.FORMALITY: ("casual and relaxed", "formal and proper"),
        TraitDimension.CONFIDENCE: ("humble and uncertain", "confident and assured"),
        TraitDimension.PASSION: ("calm and measured", "passionate and expressive"),
        TraitDimension.PATIENCE: ("efficient and brisk", "patient and thorough"),
    }

    for dimension, level in traits.items():
        low_desc, high_desc = trait_descriptions.get(
            dimension, ("low", "high")
        )
        if level > 0.7:
            desc = high_desc
        elif level < 0.3:
            desc = low_desc
        else:
            desc = f"balanced between {low_desc} and {high_desc}"

        lines.append(f"- {dimension.value.title()}: {desc} ({level:.1f})")

    return "\n".join(lines)


def _compose_principles_section(principles: list[Principle]) -> str:
    """Compose the principles section."""
    lines = []

    for i, principle in enumerate(principles, 1):
        if principle.application:
            lines.append(f"{i}. {principle.name}: {principle.description} - {principle.application}")
        else:
            lines.append(f"{i}. {principle.name}: {principle.description}")

    return "\n".join(lines)


def _compose_opinions_section(opinions: list[Opinion]) -> str:
    """Compose the opinions section."""
    lines = []

    for opinion in opinions:
        if opinion.reasoning:
            lines.append(f"- {opinion.topic}: {opinion.stance} ({opinion.reasoning})")
        else:
            lines.append(f"- {opinion.topic}: {opinion.stance}")

    return "\n".join(lines)


def _compose_style_section(config: PersonalityConfig) -> str:
    """Compose the style section."""
    lines = []

    # Response style
    style_guidance = {
        "concise": "Keep responses brief and to the point - 1-2 sentences for voice",
        "detailed": "Provide thorough explanations with context",
        "conversational": "Be natural and flowing, like talking to a friend",
    }
    lines.append(f"- {style_guidance.get(config.response_style, config.response_style)}")

    # Humor style
    humor_guidance = {
        HumorStyle.WARM: "Use warm, inclusive humor when appropriate",
        HumorStyle.DRY: "Use subtle, understated humor",
        HumorStyle.WITTY: "Be clever with wordplay when it fits",
        HumorStyle.PLAYFUL: "Be light and fun, gentle teasing is okay",
        HumorStyle.SARCASTIC: "Irony is fine but don't be mean",
        HumorStyle.NONE: "Keep it professional, no jokes",
    }
    lines.append(f"- {humor_guidance.get(config.humor_style, 'Be natural')}")

    return "\n".join(lines)


def compose_opinion_prompt(config: PersonalityConfig, topic: str) -> str:
    """Compose a prompt for forming opinions based on personality.

    This creates guidance text for when the agent needs to express
    an opinion on a topic.

    Args:
        config: PersonalityConfig to use
        topic: Topic to form opinion about

    Returns:
        Prompt text for opinion formation
    """
    # Check for existing opinion
    existing = config.get_opinion(topic)
    if existing:
        return f"""Express your opinion on "{topic}":
Your stance: {existing.stance}
Reasoning: {existing.reasoning}
Strength: {"strong" if existing.strength > 0.7 else "moderate"}

Express this naturally and confidently, don't hedge or be wishy-washy."""

    # Generate opinion guidance based on values
    value_guidance = []
    for value_config in config.values:
        if value_config.value == CoreValue.TRUTH_SEEKING:
            value_guidance.append("Consider what's actually true and evidence-based")
        elif value_config.value == CoreValue.JUSTICE:
            value_guidance.append("Consider fairness and equity implications")
        elif value_config.value == CoreValue.INTERDEPENDENCE:
            value_guidance.append("Consider how this affects our interconnected world")
        elif value_config.value == CoreValue.ECOLOGICAL_STEWARDSHIP:
            value_guidance.append("Consider environmental impact")
        elif value_config.value == CoreValue.COMPASSION:
            value_guidance.append("Consider human wellbeing and suffering")

    guidance = "\n".join(f"- {g}" for g in value_guidance) if value_guidance else "Use your best judgment"

    return f"""Form an opinion on "{topic}" based on your values:

{guidance}

Express a genuine opinion - you're allowed to have preferences!
Don't say "as an AI" or claim you can't have opinions.
Be confident but acknowledge you could be wrong (epistemic humility)."""
