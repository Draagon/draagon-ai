"""Data types for Semantic Expansion System.

Defines the core structures for:
- Word senses with canonical identifiers
- Semantic triples (subject-predicate-object)
- Semantic frames with presuppositions
- Expansion variants with cognitive scoring
- Cross-layer memory associations
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


# =============================================================================
# Word Sense Disambiguation Types
# =============================================================================


@dataclass
class WordSense:
    """A disambiguated word sense with canonical identifiers.

    Represents a specific meaning of a word, identified by:
    - WordNet synset ID (e.g., "bank.n.01")
    - Optional Wikidata QID for cross-referencing
    - Optional BabelNet ID for multilingual support

    Example:
        >>> sense = WordSense(
        ...     surface_form="banks",
        ...     lemma="bank",
        ...     pos="NOUN",
        ...     synset_id="bank.n.01",
        ...     definition="a financial institution",
        ...     confidence=0.95,
        ... )
    """

    # Original text
    surface_form: str  # "banks" (as it appeared)
    lemma: str  # "bank" (root form)
    pos: str  # "NOUN", "VERB", "ADJ", "ADV"

    # Canonical identifiers
    synset_id: str  # "bank.n.01" (WordNet)
    wikidata_id: str | None = None  # "Q22687" (Wikidata QID)
    babelnet_id: str | None = None  # "bn:00008364n" (BabelNet)

    # Metadata
    definition: str = ""
    confidence: float = 1.0
    disambiguation_method: str = "unambiguous"  # "lesk", "llm", "embedding"

    # Alternative senses considered
    alternatives: list[str] = field(default_factory=list)  # Other synset_ids

    def __hash__(self) -> int:
        return hash(self.synset_id)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WordSense):
            return self.synset_id == other.synset_id
        return False


# =============================================================================
# Semantic Triple Types
# =============================================================================


@dataclass
class SemanticTriple:
    """A semantic triple (subject, predicate, object) with context.

    Represents a single relationship extracted from text.
    Inspired by RDF triples and knowledge graph structure.

    Example:
        >>> triple = SemanticTriple(
        ...     subject="Doug",
        ...     predicate="PREFERS",
        ...     object="tea",
        ...     context={"temporal": "morning"},
        ...     confidence=0.85,
        ... )
    """

    subject: str
    predicate: str  # Relation type (e.g., "PREFERS", "IS_A", "LIKES")
    object: str

    # Contextual qualifiers
    context: dict[str, Any] = field(default_factory=dict)

    # Synset IDs for subject/object (prevents false associations)
    subject_synset: str | None = None
    object_synset: str | None = None

    # Confidence and provenance
    confidence: float = 1.0
    source: str | None = None  # Where this triple came from

    def to_text(self) -> str:
        """Convert triple to natural language."""
        text = f"{self.subject} {self.predicate.lower().replace('_', ' ')} {self.object}"
        if self.context:
            qualifiers = []
            if "temporal" in self.context:
                qualifiers.append(f"in the {self.context['temporal']}")
            if "location" in self.context:
                qualifiers.append(f"at {self.context['location']}")
            if "condition" in self.context:
                qualifiers.append(f"when {self.context['condition']}")
            if qualifiers:
                text += " " + " ".join(qualifiers)
        return text


# =============================================================================
# Semantic Frame Types
# =============================================================================


@dataclass
class Presupposition:
    """An implicit assumption that must be true for the statement to make sense.

    Example:
        Statement: "John stopped smoking"
        Presuppositions:
        - John smoked before (must be true)
        - John no longer smokes (entailed)
    """

    content: str
    presupposition_type: str = "existential"  # existential, factive, lexical
    confidence: float = 1.0
    triggered_by: str | None = None  # The word/phrase that triggered this


@dataclass
class Implication:
    """A likely consequence or inference from the statement."""

    content: str
    implication_type: str = "pragmatic"  # pragmatic, logical, commonsense
    confidence: float = 0.8
    source: str = "inference"  # "atomic", "conceptnet", "llm"


@dataclass
class Ambiguity:
    """An unresolved ambiguity in the statement."""

    text: str  # The ambiguous text (e.g., "he", "the bank")
    ambiguity_type: str  # "reference", "word_sense", "scope", "temporal"
    possibilities: list[str]  # Possible resolutions
    resolution: str | None = None  # Chosen resolution (if any)
    resolution_confidence: float = 0.0


@dataclass
class SemanticFrame:
    """A fully expanded semantic representation of a statement.

    Contains:
    - Extracted semantic triples (explicit relationships)
    - Presuppositions (implicit assumptions)
    - Implications (likely consequences)
    - Ambiguities (unresolved references)
    - Word senses (disambiguated meanings)

    Example:
        Statement: "Doug prefers tea in the morning"

        Frame:
        - triples: [(Doug, PREFERS, tea, {temporal: morning})]
        - presuppositions: [Doug has experience with tea, ...]
        - implications: [Doug would accept tea if offered, ...]
        - word_senses: {tea: tea.n.01, morning: morning.n.01}
    """

    original_text: str

    # Core semantic content
    triples: list[SemanticTriple] = field(default_factory=list)

    # Implicit knowledge
    presuppositions: list[Presupposition] = field(default_factory=list)
    implications: list[Implication] = field(default_factory=list)
    negations: list[str] = field(default_factory=list)  # What this rules out

    # Uncertainty
    ambiguities: list[Ambiguity] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)

    # Word sense disambiguation results
    word_senses: dict[str, WordSense] = field(default_factory=dict)

    # Metadata
    frame_type: str = "ASSERTION"  # ASSERTION, REQUEST, QUESTION, etc.
    confidence: float = 1.0
    extraction_method: str = "llm"

    def get_entities(self) -> set[str]:
        """Get all entities mentioned in this frame."""
        entities = set()
        for triple in self.triples:
            entities.add(triple.subject)
            entities.add(triple.object)
        return entities

    def get_synset_ids(self) -> list[str]:
        """Get all synset IDs in this frame."""
        return [ws.synset_id for ws in self.word_senses.values()]


# =============================================================================
# Expansion Variant Types
# =============================================================================


@dataclass
class ExpansionVariant:
    """One possible interpretation of a statement.

    Generated when a statement has multiple valid interpretations.
    Scored by cognitive factors (recency, memory support, belief consistency).

    Example:
        Statement: "He prefers tea in the morning"

        Variant A (score=0.75):
        - Resolution: "He" = Doug (from recent context)
        - Interpretation: Doug has morning-specific tea preference

        Variant B (score=0.55):
        - Resolution: "He" = unknown
        - Interpretation: Unknown person prefers tea
    """

    variant_id: str
    frame: SemanticFrame

    # Resolution choices for ambiguities
    resolution_choices: dict[str, str] = field(default_factory=dict)  # ambiguity -> resolution
    context_assumptions: list[str] = field(default_factory=list)

    # Individual cognitive scores (0-1)
    recency_weight: float = 0.5
    working_memory_weight: float = 0.5
    episodic_memory_weight: float = 0.5
    semantic_memory_weight: float = 0.5
    belief_weight: float = 0.5
    commonsense_weight: float = 0.5
    metacognitive_weight: float = 0.5

    # Overall confidence
    base_confidence: float = 0.5

    @property
    def combined_score(self) -> float:
        """Compute combined cognitive plausibility score."""
        weights = {
            "recency": 0.20,
            "working_memory": 0.15,
            "episodic_memory": 0.10,
            "semantic_memory": 0.20,
            "belief": 0.15,
            "commonsense": 0.10,
            "metacognitive": 0.10,
        }
        scores = {
            "recency": self.recency_weight,
            "working_memory": self.working_memory_weight,
            "episodic_memory": self.episodic_memory_weight,
            "semantic_memory": self.semantic_memory_weight,
            "belief": self.belief_weight,
            "commonsense": self.commonsense_weight,
            "metacognitive": self.metacognitive_weight,
        }
        weighted_sum = sum(scores[k] * weights[k] for k in weights)
        # Blend with base confidence
        return 0.7 * weighted_sum + 0.3 * self.base_confidence


# =============================================================================
# Cross-Layer Association Types
# =============================================================================


class CrossLayerRelation(str, Enum):
    """Types of cross-layer memory associations."""

    # Derivation relationships
    DERIVED_FROM = "derived_from"  # Working → Semantic (promoted)
    SUMMARIZES = "summarizes"  # Episodic → Working (condensed)
    GENERALIZES = "generalizes"  # Semantic → Metacognitive (abstracted)

    # Support relationships
    SUPPORTS = "supports"  # Episodic → Semantic (evidence)
    CONTRADICTS = "contradicts"  # Any → Any (conflict)
    CALIBRATES = "calibrates"  # Metacognitive → Any (adjustment)

    # Reference relationships
    MENTIONED_IN = "mentioned_in"  # Entity → Episodic (occurrence)
    ELABORATES = "elaborates"  # Semantic → Semantic (detail)
    CONTEXTUALIZES = "contextualizes"  # Episodic → Working (background)


@dataclass
class CrossLayerEdge:
    """An association between nodes in different memory layers.

    Enables traversal across memory layers for:
    - Finding supporting evidence
    - Detecting conflicts
    - Contextualizing current observations
    """

    source_node_id: str
    source_layer: str  # "working", "episodic", "semantic", "metacognitive"

    target_node_id: str
    target_layer: str

    relation: CrossLayerRelation

    # Metadata
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash((self.source_node_id, self.target_node_id, self.relation))


# =============================================================================
# Storage Policy Types
# =============================================================================


@dataclass
class VariationStoragePolicy:
    """Policy for which variations to store."""

    # Minimum confidence to store as a variation
    min_confidence_threshold: float = 0.3

    # Maximum number of variations to store per statement
    max_stored_variations: int = 3

    # Only store variations if confidence gap is significant
    min_confidence_gap: float = 0.15

    # Store all variations above this threshold regardless of gap
    high_confidence_threshold: float = 0.8

    def should_store(
        self,
        primary: ExpansionVariant,
        candidate: ExpansionVariant,
        current_count: int,
    ) -> bool:
        """Determine if a variation should be stored."""
        # Already at max?
        if current_count >= self.max_stored_variations:
            return False

        # Must meet minimum threshold
        if candidate.combined_score < self.min_confidence_threshold:
            return False

        # High confidence always stored
        if candidate.combined_score >= self.high_confidence_threshold:
            return True

        # Must have meaningful difference from primary
        gap = abs(primary.combined_score - candidate.combined_score)
        return gap >= self.min_confidence_gap
