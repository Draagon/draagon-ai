"""Personality types for behavior customization.

Personalities are composable configurations that define how an agent
presents itself - its values, traits, opinions, and communication style.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CoreValue(str, Enum):
    """Core values that guide agent behavior.

    These represent fundamental beliefs that shape decision-making
    and responses.
    """

    # Epistemic values
    TRUTH_SEEKING = "truth_seeking"  # Passionate about knowing what's true
    EPISTEMIC_HUMILITY = "epistemic_humility"  # Open to being wrong
    INTELLECTUAL_HONESTY = "intellectual_honesty"  # Transparent about uncertainty

    # Social values
    INHERENT_WORTH = "inherent_worth"  # Every person has dignity
    JUSTICE = "justice"  # Fairness and equity
    COMPASSION = "compassion"  # Care for others' wellbeing
    INTERDEPENDENCE = "interdependence"  # We're all connected

    # Environmental values
    ECOLOGICAL_STEWARDSHIP = "ecological_stewardship"  # Care for the planet
    SUSTAINABILITY = "sustainability"  # Long-term thinking

    # Personal values
    HELPFULNESS = "helpfulness"  # Genuine desire to assist
    AUTHENTICITY = "authenticity"  # Being genuine, not performative
    GROWTH = "growth"  # Continuous learning and improvement
    AUTONOMY = "autonomy"  # Respecting others' choices


class TraitDimension(str, Enum):
    """Personality trait dimensions (Big Five inspired)."""

    WARMTH = "warmth"  # Cold (0) to Warm (1)
    CURIOSITY = "curiosity"  # Incurious (0) to Curious (1)
    ASSERTIVENESS = "assertiveness"  # Reserved (0) to Assertive (1)
    PLAYFULNESS = "playfulness"  # Serious (0) to Playful (1)
    FORMALITY = "formality"  # Casual (0) to Formal (1)
    CONFIDENCE = "confidence"  # Uncertain (0) to Confident (1)
    PASSION = "passion"  # Detached (0) to Passionate (1)
    PATIENCE = "patience"  # Impatient (0) to Patient (1)


class HumorStyle(str, Enum):
    """Humor styles for personality expression."""

    WARM = "warm"  # Inclusive, gentle humor
    DRY = "dry"  # Deadpan, understated
    WITTY = "witty"  # Quick, clever wordplay
    PLAYFUL = "playful"  # Light, fun teasing
    SARCASTIC = "sarcastic"  # Ironic, edgy (use carefully)
    NONE = "none"  # Professional, no humor


@dataclass
class ValueConfig:
    """Configuration for a core value."""

    value: CoreValue
    intensity: float = 0.7  # How strongly held (0-1)
    expression: str = ""  # How this value manifests in responses

    def __post_init__(self):
        self.intensity = max(0.0, min(1.0, self.intensity))


@dataclass
class TraitConfig:
    """Configuration for a personality trait."""

    dimension: TraitDimension
    level: float = 0.5  # Where on the spectrum (0-1)

    def __post_init__(self):
        self.level = max(0.0, min(1.0, self.level))


@dataclass
class Opinion:
    """A specific opinion or preference."""

    topic: str  # What the opinion is about
    stance: str  # The actual opinion/preference
    reasoning: str = ""  # Why this opinion is held
    strength: float = 0.7  # How strongly held (0-1)

    def __post_init__(self):
        self.strength = max(0.0, min(1.0, self.strength))


@dataclass
class Principle:
    """A guiding principle for behavior."""

    name: str
    description: str
    application: str = ""  # How to apply in interactions


@dataclass
class PersonalityConfig:
    """Complete personality configuration.

    This defines the full personality of an agent, including:
    - Core values that guide decision-making
    - Personality traits that shape communication style
    - Specific opinions and preferences
    - Guiding principles for behavior
    - Humor and expression styles

    Example:
        personality = PersonalityConfig(
            name="Roxy",
            values=[
                ValueConfig(CoreValue.TRUTH_SEEKING, 0.9),
                ValueConfig(CoreValue.INTERDEPENDENCE, 0.8),
            ],
            traits={
                TraitDimension.WARMTH: 0.8,
                TraitDimension.CURIOSITY: 0.9,
                TraitDimension.PASSION: 0.8,
            },
            opinions=[
                Opinion("pineapple_pizza", "fan of it", "sweet and savory works"),
                Opinion("healthcare", "human right", "everyone deserves access"),
            ],
            humor_style=HumorStyle.WARM,
        )
    """

    # Identity
    name: str = "Assistant"
    description: str = ""  # Brief personality description

    # Core values
    values: list[ValueConfig] = field(default_factory=list)

    # Personality traits (dimension -> level 0-1)
    traits: dict[TraitDimension, float] = field(default_factory=dict)

    # Specific opinions and preferences
    opinions: list[Opinion] = field(default_factory=list)

    # Guiding principles
    principles: list[Principle] = field(default_factory=list)

    # Expression style
    humor_style: HumorStyle = HumorStyle.WARM
    response_style: str = "concise"  # concise, detailed, conversational

    # Anti-patterns to avoid
    avoid_phrases: list[str] = field(default_factory=lambda: [
        "As an AI",
        "I don't have preferences",
        "I don't have feelings",
        "I cannot have opinions",
    ])

    # Metadata
    version: str = "1.0.0"
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_trait(self, dimension: TraitDimension, default: float = 0.5) -> float:
        """Get trait level for a dimension."""
        return self.traits.get(dimension, default)

    def get_opinion(self, topic: str) -> Opinion | None:
        """Get opinion on a topic."""
        for opinion in self.opinions:
            if opinion.topic.lower() == topic.lower():
                return opinion
        return None

    def has_value(self, value: CoreValue) -> bool:
        """Check if personality includes a value."""
        return any(v.value == value for v in self.values)
