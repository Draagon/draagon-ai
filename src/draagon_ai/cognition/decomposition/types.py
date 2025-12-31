"""Type definitions for the decomposition pipeline.

These types represent the structured output of semantic decomposition,
designed to flow directly into the SemanticMemory layer.

Based on prototype work in prototypes/implicit_knowledge_graphs/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Entity Types (aligned with prototype identifiers.py)
# =============================================================================


class EntityType(str, Enum):
    """Types of semantic entities.

    INSTANCE: A specific, unique real-world entity (Doug, Apple Inc.)
    CLASS: A category or type of things (person, cat, bank)
    NAMED_CONCEPT: A proper-named category (Christmas, Agile)
    ROLE: A relational concept (CEO of Apple, Doug's wife)
    ANAPHORA: A reference requiring resolution (he, she, it)
    GENERIC: A generic reference (someone, everyone)
    """

    INSTANCE = "instance"
    CLASS = "class"
    NAMED_CONCEPT = "named_concept"
    ROLE = "role"
    ANAPHORA = "anaphora"
    GENERIC = "generic"


# =============================================================================
# Trigger Types for Presuppositions
# =============================================================================


class TriggerType(str, Enum):
    """Types of presupposition triggers."""

    FACTIVE = "factive"  # "forgot" presupposes complement was true
    ITERATIVE = "iterative"  # "again" presupposes it happened before
    CHANGE_OF_STATE = "change_of_state"  # "stopped" presupposes was doing
    DEFINITE = "definite"  # "the X" presupposes X exists
    POSSESSIVE = "possessive"  # "Doug's X" presupposes Doug has X
    TEMPORAL = "temporal"  # "before/after X" presupposes X occurred
    CLEFT = "cleft"  # "It was Doug who..." presupposes someone did it


# =============================================================================
# Tense and Aspect
# =============================================================================


class Tense(str, Enum):
    """Grammatical tense."""

    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    PAST_PERFECT = "past_perfect"
    PRESENT_PERFECT = "present_perfect"
    FUTURE_PERFECT = "future_perfect"


class Aspect(str, Enum):
    """Grammatical aspect."""

    SIMPLE = "simple"
    PROGRESSIVE = "progressive"
    PERFECT = "perfect"
    PERFECT_PROGRESSIVE = "perfect_progressive"
    STATE = "state"
    ACTIVITY = "activity"
    ACHIEVEMENT = "achievement"
    ACCOMPLISHMENT = "accomplishment"


# =============================================================================
# Modality Types
# =============================================================================


class ModalType(str, Enum):
    """Types of modality."""

    NONE = "none"
    EPISTEMIC = "epistemic"  # might, may, could (possibility)
    DEONTIC = "deontic"  # must, should, ought to (obligation)
    DYNAMIC = "dynamic"  # can, able to (ability)
    HYPOTHETICAL = "hypothetical"  # would, if...then


# =============================================================================
# Polarity
# =============================================================================


class Polarity(str, Enum):
    """Sentence polarity."""

    POSITIVE = "positive"
    NEGATIVE = "negative"


# =============================================================================
# Commonsense Relation Types (ATOMIC-style)
# =============================================================================


class CommonsenseRelation(str, Enum):
    """ATOMIC-style commonsense relation types."""

    # Agent-focused
    X_INTENT = "xIntent"  # Why X does action
    X_NEED = "xNeed"  # What X needs to do action
    X_ATTR = "xAttr"  # X's attributes
    X_EFFECT = "xEffect"  # Effect on X
    X_WANT = "xWant"  # What X wants after
    X_REACT = "xReact"  # X's emotional reaction

    # Other-focused
    O_EFFECT = "oEffect"  # Effect on others
    O_WANT = "oWant"  # What others want after
    O_REACT = "oReact"  # Others' emotional reaction

    # Object-focused
    IS_BEFORE = "isBefore"  # Preconditions
    IS_AFTER = "isAfter"  # Postconditions
    HAS_SUBEVENT = "HasSubEvent"  # Component events
    HAS_FIRST_SUBEVENT = "HasFirstSubEvent"
    HAS_LAST_SUBEVENT = "HasLastSubEvent"
    HAS_PREREQUISITE = "HasPrerequisite"
    CAUSES = "Causes"
    HINDERS = "HinderedBy"


# =============================================================================
# Extracted Data Types
# =============================================================================


@dataclass
class ExtractedEntity:
    """An entity extracted from text.

    This maps to Entity in SemanticMemory.
    """

    # Identity
    text: str  # Original text span
    canonical_name: str  # Normalized name
    entity_type: EntityType

    # Disambiguation
    synset_id: str | None = None  # WordNet synset (e.g., "bank.n.01")
    wikidata_qid: str | None = None  # Wikidata entity ID
    definition: str | None = None  # Sense definition

    # Metadata
    confidence: float = 1.0
    aliases: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    # For ROLE type
    role_relation: str | None = None
    anchor_entity: str | None = None

    # For ANAPHORA type
    resolved_to: str | None = None


@dataclass
class ExtractedFact:
    """A fact extracted from text.

    This maps to Fact in SemanticMemory.
    """

    content: str  # Natural language statement
    subject_text: str  # Subject entity text
    predicate: str  # Relationship predicate
    object_value: str  # Object value

    # Linking (set during integration)
    subject_entity_id: str | None = None

    # Metadata
    confidence: float = 1.0
    source_text: str = ""
    temporal_qualifier: str | None = None  # "currently", "used to", etc.


@dataclass
class ExtractedRelationship:
    """A relationship between entities.

    This maps to Relationship in SemanticMemory.
    """

    source_text: str  # Source entity text
    target_text: str  # Target entity text
    relationship_type: str  # Type of relationship

    # Linking (set during integration)
    source_entity_id: str | None = None
    target_entity_id: str | None = None

    # Metadata
    confidence: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticRole:
    """A semantic role (predicate-argument structure)."""

    predicate: str  # The verb/predicate
    role: str  # ARG0, ARG1, ARGM-TMP, etc.
    filler: str  # The argument text
    predicate_sense: str | None = None  # WordNet sense
    confidence: float = 1.0


@dataclass
class Presupposition:
    """A presupposition triggered by the text."""

    content: str  # What is presupposed
    trigger_type: TriggerType
    trigger_text: str  # The word/phrase that triggered it
    confidence: float = 1.0
    cancellable: bool = True  # Can be cancelled by context


@dataclass
class CommonsenseInference:
    """An ATOMIC-style commonsense inference."""

    relation: CommonsenseRelation
    head: str  # The event/state
    tail: str  # The inferred knowledge
    confidence: float = 1.0


@dataclass
class TemporalInfo:
    """Temporal information from the sentence."""

    tense: Tense
    aspect: Aspect
    reference_type: str | None = None  # "absolute", "relative", "deictic"
    reference_value: str | None = None  # "tomorrow", "2024-01-15", etc.
    confidence: float = 1.0


@dataclass
class ModalityInfo:
    """Modality information from the sentence."""

    modal_type: ModalType
    modal_marker: str | None = None  # "might", "should", etc.
    certainty: float = 1.0  # 1.0 = certain, 0.0 = uncertain


@dataclass
class NegationInfo:
    """Negation information from the sentence."""

    is_negated: bool
    negation_cue: str | None = None  # "not", "never", "no", etc.
    polarity: Polarity = Polarity.POSITIVE
    scope: str | None = None  # Text span under negation


@dataclass
class InterpretationBranch:
    """A possible interpretation of ambiguous input.

    When WSD or entity classification produces alternatives,
    we create branches for each interpretation.
    """

    interpretation: str  # Description of this interpretation
    confidence: float  # Base confidence from WSD/classification
    memory_support: float = 0.0  # Support from memory context
    final_weight: float = 0.0  # Combined weight

    # What makes this interpretation distinct
    entity_interpretations: dict[str, str] = field(default_factory=dict)
    supporting_evidence: list[str] = field(default_factory=list)


# =============================================================================
# Complete Decomposition Result
# =============================================================================


@dataclass
class DecompositionResult:
    """Complete result of decomposing a natural language input.

    This is the structured output ready for integration with SemanticMemory.
    """

    # Source
    source_text: str
    content_type: str = "prose"  # prose, code, mixed

    # Extracted knowledge
    entities: list[ExtractedEntity] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)

    # Semantic structure
    semantic_roles: list[SemanticRole] = field(default_factory=list)
    presuppositions: list[Presupposition] = field(default_factory=list)
    commonsense_inferences: list[CommonsenseInference] = field(default_factory=list)

    # Temporal/Modal
    temporal: TemporalInfo | None = None
    modality: ModalityInfo | None = None
    negation: NegationInfo | None = None

    # Ambiguity handling
    branches: list[InterpretationBranch] = field(default_factory=list)

    # Metadata
    pipeline_version: str = "1.0.0"
    processing_time_ms: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)

    def get_primary_branch(self) -> InterpretationBranch | None:
        """Get the highest-weighted interpretation branch."""
        if not self.branches:
            return None
        return max(self.branches, key=lambda b: b.final_weight)

    def get_entity_by_text(self, text: str) -> ExtractedEntity | None:
        """Find an entity by its text."""
        for entity in self.entities:
            if entity.text.lower() == text.lower():
                return entity
            if entity.canonical_name.lower() == text.lower():
                return entity
        return None
