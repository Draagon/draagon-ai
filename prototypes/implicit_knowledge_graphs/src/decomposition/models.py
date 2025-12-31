"""Core data structures for the decomposition pipeline.

This module defines all the data structures used to represent
decomposed implicit knowledge, including:
- Semantic roles (predicate-argument structures)
- Presuppositions and their triggers
- Commonsense inferences (ATOMIC-style)
- Temporal and aspectual information
- Modality markers
- Negation and polarity
- Weighted interpretation branches

All structures are designed to be:
1. Serializable to/from JSON for storage
2. Hashable where appropriate for set operations
3. Compatible with the evolution framework

Example:
    >>> from decomposition.models import (
    ...     Presupposition, PresuppositionTrigger, DecomposedKnowledge
    ... )
    >>>
    >>> presup = Presupposition(
    ...     content="Doug forgot before",
    ...     trigger_type=PresuppositionTrigger.ITERATIVE,
    ...     trigger_text="again",
    ...     trigger_span=(25, 30),
    ...     confidence=0.9,
    ... )
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from identifiers import UniversalSemanticIdentifier


# =============================================================================
# Enumerations
# =============================================================================


class PresuppositionTrigger(str, Enum):
    """Types of presupposition triggers.

    Based on linguistic research on presupposition projection.
    See DECOMPOSITION_THEORY.md for detailed descriptions.
    """

    DEFINITE_DESC = "definite_description"
    """Definite descriptions like 'the X' presuppose X exists."""

    FACTIVE_VERB = "factive_verb"
    """Factive verbs like 'realize', 'know' presuppose their complement is true."""

    CHANGE_OF_STATE = "change_of_state"
    """Change-of-state verbs like 'stop', 'start' presuppose prior state."""

    ITERATIVE = "iterative"
    """Iteratives like 'again', 'another' presuppose prior instance."""

    TEMPORAL_CLAUSE = "temporal_clause"
    """Temporal clauses with 'before', 'after' presuppose the event occurred."""

    CLEFT = "cleft"
    """Cleft sentences like 'It was X who...' presuppose someone did the action."""

    COMPARATIVE = "comparative"
    """Comparatives like 'more than X' presuppose X has the property."""

    IMPLICATIVE = "implicative"
    """Implicative verbs like 'manage', 'forget' have presuppositional implications."""

    COUNTERFACTUAL = "counterfactual"
    """Counterfactuals like 'if X had...' presuppose X didn't happen."""

    POSSESSIVE = "possessive"
    """Possessives like 'X's Y' presuppose X has Y."""


class CommonsenseRelation(str, Enum):
    """ATOMIC 2020 commonsense relation types.

    Organized by tiers based on extraction priority:
    - Tier 1 (always extract): xIntent, xEffect, xReact, xAttr
    - Tier 2 (conditional): xNeed, xWant, oReact, Causes
    """

    # Tier 1: Always extract
    X_INTENT = "xIntent"
    """Why PersonX did this action (motivation)."""

    X_EFFECT = "xEffect"
    """What happens to PersonX as a result."""

    X_REACT = "xReact"
    """How PersonX feels emotionally."""

    X_ATTR = "xAttr"
    """Attributes of PersonX revealed by this action."""

    # Tier 2: Conditional extraction
    X_NEED = "xNeed"
    """What PersonX needed before doing this."""

    X_WANT = "xWant"
    """What PersonX wants after doing this."""

    O_REACT = "oReact"
    """How others feel about this action."""

    CAUSES = "Causes"
    """What this event causes to happen."""

    # Tier 3: Rarely extracted (for completeness)
    O_WANT = "oWant"
    """What others want as a result."""

    O_EFFECT = "oEffect"
    """What happens to others as a result."""

    IS_BEFORE = "isBefore"
    """What typically happens before this."""

    IS_AFTER = "isAfter"
    """What typically happens after this."""

    HINDERED_BY = "HinderedBy"
    """What could prevent this action."""


class Aspect(str, Enum):
    """Vendler aspectual categories for events.

    These categories describe the internal temporal structure of events.
    """

    STATE = "state"
    """Unchanging condition with no endpoint (e.g., 'knows', 'likes')."""

    ACTIVITY = "activity"
    """Ongoing process with no inherent endpoint (e.g., 'running', 'swimming')."""

    ACCOMPLISHMENT = "accomplishment"
    """Process with an inherent endpoint (e.g., 'built a house')."""

    ACHIEVEMENT = "achievement"
    """Instantaneous change of state (e.g., 'noticed', 'found')."""

    SEMELFACTIVE = "semelfactive"
    """Single instantaneous event (e.g., 'knocked', 'blinked')."""


class Tense(str, Enum):
    """Grammatical tense categories."""

    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    UNKNOWN = "unknown"


class ModalType(str, Enum):
    """Types of linguistic modality."""

    EPISTEMIC = "epistemic"
    """Speaker's certainty about truth (might, must, probably)."""

    DEONTIC = "deontic"
    """Obligation or permission (should, must, may)."""

    DYNAMIC = "dynamic"
    """Ability or willingness (can, will)."""

    EVIDENTIAL = "evidential"
    """Source of information (apparently, reportedly)."""

    BOULETIC = "bouletic"
    """Desire or wish (want to, wish)."""

    NONE = "none"
    """No modal marking."""


class Polarity(str, Enum):
    """Polarity of a statement or triple."""

    POSITIVE = "positive"
    """Affirmed/asserted."""

    NEGATIVE = "negative"
    """Negated/denied."""

    UNCERTAIN = "uncertain"
    """Polarity is unclear or hedged."""

    DOUBLE_NEGATIVE = "double_negative"
    """Double negation (e.g., 'not impossible') - pragmatically positive."""


# =============================================================================
# Core Data Structures
# =============================================================================


@dataclass
class SemanticRole:
    """A semantic role in a predicate-argument structure.

    Represents the relationship between a predicate (verb) and its arguments
    (who did what to whom, where, when, etc.).

    Attributes:
        predicate: The predicate word (e.g., "forgot")
        predicate_sense: WordNet synset ID for the predicate sense
        role: The semantic role label (ARG0, ARG1, ARGM-LOC, etc.)
        filler: The text that fills this role
        filler_id: Universal identifier for the filler entity (if identified)
        span: Character span of the filler in the source text
        confidence: Confidence score for this role assignment
    """

    predicate: str
    predicate_sense: str | None
    role: str
    filler: str
    filler_id: str | None = None  # local_id of UniversalSemanticIdentifier
    span: tuple[int, int] | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "predicate": self.predicate,
            "predicate_sense": self.predicate_sense,
            "role": self.role,
            "filler": self.filler,
            "filler_id": self.filler_id,
            "span": list(self.span) if self.span else None,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SemanticRole:
        """Deserialize from dictionary."""
        return cls(
            predicate=data["predicate"],
            predicate_sense=data.get("predicate_sense"),
            role=data["role"],
            filler=data["filler"],
            filler_id=data.get("filler_id"),
            span=tuple(data["span"]) if data.get("span") else None,
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class Presupposition:
    """A presupposition extracted from text.

    Presuppositions are what must be true for a statement to make sense.
    For example, "Doug forgot the meeting again" presupposes:
    - A specific meeting exists (from "the meeting")
    - Doug forgot before (from "again")

    Attributes:
        content: The presupposed content as text
        trigger_type: The type of linguistic trigger
        trigger_text: The specific text that triggered this presupposition
        trigger_span: Character span of the trigger in source text
        confidence: Confidence score (0.0-1.0)
        cancellable: Whether this presupposition can be cancelled in context
        entity_ids: IDs of entities mentioned in the presupposition
    """

    content: str
    trigger_type: PresuppositionTrigger
    trigger_text: str
    trigger_span: tuple[int, int] | None = None
    confidence: float = 1.0
    cancellable: bool = True
    entity_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "content": self.content,
            "trigger_type": self.trigger_type.value,
            "trigger_text": self.trigger_text,
            "trigger_span": list(self.trigger_span) if self.trigger_span else None,
            "confidence": self.confidence,
            "cancellable": self.cancellable,
            "entity_ids": self.entity_ids,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Presupposition:
        """Deserialize from dictionary."""
        return cls(
            content=data["content"],
            trigger_type=PresuppositionTrigger(data["trigger_type"]),
            trigger_text=data["trigger_text"],
            trigger_span=tuple(data["trigger_span"]) if data.get("trigger_span") else None,
            confidence=data.get("confidence", 1.0),
            cancellable=data.get("cancellable", True),
            entity_ids=data.get("entity_ids", []),
        )


@dataclass
class CommonsenseInference:
    """A commonsense inference (ATOMIC-style).

    Represents an inference about causes, effects, mental states, etc.
    derived from commonsense reasoning about events.

    Attributes:
        relation: The ATOMIC relation type
        head: The source event/situation
        tail: The inferred content
        head_entity_ids: IDs of entities in the head
        confidence: Confidence score (0.0-1.0)
        source: Where this inference came from (llm, comet, rule)
    """

    relation: CommonsenseRelation
    head: str
    tail: str
    head_entity_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0
    source: str = "llm"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "relation": self.relation.value,
            "head": self.head,
            "tail": self.tail,
            "head_entity_ids": self.head_entity_ids,
            "confidence": self.confidence,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommonsenseInference:
        """Deserialize from dictionary."""
        return cls(
            relation=CommonsenseRelation(data["relation"]),
            head=data["head"],
            tail=data["tail"],
            head_entity_ids=data.get("head_entity_ids", []),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "llm"),
        )


@dataclass
class TemporalInfo:
    """Temporal and aspectual information about an event.

    Captures when something happened, how long it lasted, and the
    internal temporal structure of the event.

    Attributes:
        aspect: Vendler aspectual category
        tense: Grammatical tense
        reference_type: Type of temporal reference (deictic, calendar, etc.)
        reference_value: The actual temporal reference ("yesterday", "3pm")
        duration: Duration expression if present
        frequency: Frequency expression if present
        confidence: Confidence score
    """

    aspect: Aspect = Aspect.STATE
    tense: Tense = Tense.UNKNOWN
    reference_type: str | None = None
    reference_value: str | None = None
    duration: str | None = None
    frequency: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "aspect": self.aspect.value,
            "tense": self.tense.value,
            "reference_type": self.reference_type,
            "reference_value": self.reference_value,
            "duration": self.duration,
            "frequency": self.frequency,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalInfo:
        """Deserialize from dictionary."""
        return cls(
            aspect=Aspect(data.get("aspect", "state")),
            tense=Tense(data.get("tense", "unknown")),
            reference_type=data.get("reference_type"),
            reference_value=data.get("reference_value"),
            duration=data.get("duration"),
            frequency=data.get("frequency"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class ModalityInfo:
    """Modal information about certainty and obligation.

    Captures how certain the speaker is, whether something is
    obligatory/permitted, and the source of information.

    Attributes:
        modal_type: Type of modality
        modal_marker: The word/phrase marking modality
        certainty: Certainty level for epistemic modality (0.0-1.0)
        evidence_source: Source type for evidential modality
        confidence: Confidence in the modality detection
    """

    modal_type: ModalType = ModalType.NONE
    modal_marker: str | None = None
    certainty: float | None = None
    evidence_source: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "modal_type": self.modal_type.value,
            "modal_marker": self.modal_marker,
            "certainty": self.certainty,
            "evidence_source": self.evidence_source,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModalityInfo:
        """Deserialize from dictionary."""
        return cls(
            modal_type=ModalType(data.get("modal_type", "none")),
            modal_marker=data.get("modal_marker"),
            certainty=data.get("certainty"),
            evidence_source=data.get("evidence_source"),
            confidence=data.get("confidence", 1.0),
        )


@dataclass
class NegationInfo:
    """Negation and polarity information.

    Captures whether a statement is negated and what is in scope
    of the negation.

    Attributes:
        is_negated: Whether the main proposition is negated
        negation_cue: The word/phrase marking negation
        negation_scope: What text is in scope of the negation
        negation_scope_span: Character span of the negation scope
        polarity: Overall polarity of the statement
    """

    is_negated: bool = False
    negation_cue: str | None = None
    negation_scope: str | None = None
    negation_scope_span: tuple[int, int] | None = None
    polarity: Polarity = Polarity.POSITIVE

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "is_negated": self.is_negated,
            "negation_cue": self.negation_cue,
            "negation_scope": self.negation_scope,
            "negation_scope_span": list(self.negation_scope_span) if self.negation_scope_span else None,
            "polarity": self.polarity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NegationInfo:
        """Deserialize from dictionary."""
        return cls(
            is_negated=data.get("is_negated", False),
            negation_cue=data.get("negation_cue"),
            negation_scope=data.get("negation_scope"),
            negation_scope_span=tuple(data["negation_scope_span"]) if data.get("negation_scope_span") else None,
            polarity=Polarity(data.get("polarity", "positive")),
        )


@dataclass
class CrossReference:
    """A cross-reference to existing memory.

    Links decomposed knowledge to prior knowledge in memory,
    enabling context-aware interpretation.

    Attributes:
        reference_id: ID of the referenced memory item
        reference_type: Type of reference (prior_instance, resolves_to, etc.)
        confidence: Confidence in the cross-reference
        memory_layer: Which memory layer (working, episodic, semantic)
    """

    reference_id: str
    reference_type: str
    confidence: float = 1.0
    memory_layer: str = "episodic"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "reference_id": self.reference_id,
            "reference_type": self.reference_type,
            "confidence": self.confidence,
            "memory_layer": self.memory_layer,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrossReference:
        """Deserialize from dictionary."""
        return cls(
            reference_id=data["reference_id"],
            reference_type=data["reference_type"],
            confidence=data.get("confidence", 1.0),
            memory_layer=data.get("memory_layer", "episodic"),
        )


@dataclass
class WeightedBranch:
    """A weighted interpretation branch.

    When text is ambiguous, multiple interpretations are possible.
    Each branch represents one interpretation with associated confidence.

    Attributes:
        branch_id: Unique identifier for this branch
        interpretation: Human-readable description of this interpretation
        confidence: Base confidence from extraction
        memory_support: Additional confidence from memory cross-references
        final_weight: Combined weight (confidence + memory_support)
        supporting_evidence: List of evidence supporting this branch
        entity_interpretations: How entities were interpreted in this branch
    """

    branch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    interpretation: str = ""
    confidence: float = 1.0
    memory_support: float = 0.0
    final_weight: float = 1.0
    supporting_evidence: list[str] = field(default_factory=list)
    entity_interpretations: dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Compute final weight after initialization."""
        self.final_weight = self.confidence + self.memory_support

    def update_weight(self, memory_adjustment: float = 0.0) -> None:
        """Update the final weight with memory support."""
        self.memory_support = memory_adjustment
        self.final_weight = self.confidence + self.memory_support

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "branch_id": self.branch_id,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "memory_support": self.memory_support,
            "final_weight": self.final_weight,
            "supporting_evidence": self.supporting_evidence,
            "entity_interpretations": self.entity_interpretations,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WeightedBranch:
        """Deserialize from dictionary."""
        branch = cls(
            branch_id=data.get("branch_id", str(uuid.uuid4())),
            interpretation=data.get("interpretation", ""),
            confidence=data.get("confidence", 1.0),
            memory_support=data.get("memory_support", 0.0),
            supporting_evidence=data.get("supporting_evidence", []),
            entity_interpretations=data.get("entity_interpretations", {}),
        )
        branch.final_weight = data.get("final_weight", branch.confidence + branch.memory_support)
        return branch


@dataclass
class DecomposedKnowledge:
    """Complete decomposed knowledge from a text.

    This is the main output of the decomposition pipeline, containing
    all extracted implicit knowledge ready for storage.

    Attributes:
        source_text: The original input text
        source_id: Unique identifier for this decomposition
        entity_ids: IDs of entities identified in the text
        wsd_results: Word sense disambiguation results (word -> synset_id)
        entity_types: Entity type classifications (entity -> type)
        semantic_roles: Extracted predicate-argument structures
        presuppositions: Extracted presuppositions
        commonsense_inferences: Generated commonsense inferences
        temporal: Temporal/aspectual information
        modality: Modal information
        negation: Negation/polarity information
        cross_references: Links to existing memory (optional)
        branches: Weighted interpretation branches
        decomposition_timestamp: When this was decomposed
        pipeline_version: Version of the pipeline used
        config_hash: Hash of the config for reproducibility
    """

    # Source
    source_text: str
    source_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Identified entities (IDs from Phase 0)
    entity_ids: list[str] = field(default_factory=list)

    # Phase 0 results (for integrated pipeline)
    wsd_results: dict[str, str] = field(default_factory=dict)
    entity_types: dict[str, str] = field(default_factory=dict)

    # Extracted knowledge
    semantic_roles: list[SemanticRole] = field(default_factory=list)
    presuppositions: list[Presupposition] = field(default_factory=list)
    commonsense_inferences: list[CommonsenseInference] = field(default_factory=list)

    # Modifiers
    temporal: TemporalInfo | None = None
    modality: ModalityInfo | None = None
    negation: NegationInfo | None = None

    # Cross-references (optional, from memory linking)
    cross_references: list[CrossReference] | None = None

    # Weighted branches for ambiguous interpretations
    branches: list[WeightedBranch] = field(default_factory=list)

    # Metadata
    decomposition_timestamp: datetime = field(default_factory=datetime.now)
    pipeline_version: str = "1.0.0"
    config_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "source_text": self.source_text,
            "source_id": self.source_id,
            "entity_ids": self.entity_ids,
            "wsd_results": self.wsd_results,
            "entity_types": self.entity_types,
            "semantic_roles": [r.to_dict() for r in self.semantic_roles],
            "presuppositions": [p.to_dict() for p in self.presuppositions],
            "commonsense_inferences": [i.to_dict() for i in self.commonsense_inferences],
            "temporal": self.temporal.to_dict() if self.temporal else None,
            "modality": self.modality.to_dict() if self.modality else None,
            "negation": self.negation.to_dict() if self.negation else None,
            "cross_references": [c.to_dict() for c in self.cross_references] if self.cross_references else None,
            "branches": [b.to_dict() for b in self.branches],
            "decomposition_timestamp": self.decomposition_timestamp.isoformat(),
            "pipeline_version": self.pipeline_version,
            "config_hash": self.config_hash,
        }

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DecomposedKnowledge:
        """Deserialize from dictionary."""
        return cls(
            source_text=data["source_text"],
            source_id=data.get("source_id", str(uuid.uuid4())),
            entity_ids=data.get("entity_ids", []),
            wsd_results=data.get("wsd_results", {}),
            entity_types=data.get("entity_types", {}),
            semantic_roles=[SemanticRole.from_dict(r) for r in data.get("semantic_roles", [])],
            presuppositions=[Presupposition.from_dict(p) for p in data.get("presuppositions", [])],
            commonsense_inferences=[CommonsenseInference.from_dict(i) for i in data.get("commonsense_inferences", [])],
            temporal=TemporalInfo.from_dict(data["temporal"]) if data.get("temporal") else None,
            modality=ModalityInfo.from_dict(data["modality"]) if data.get("modality") else None,
            negation=NegationInfo.from_dict(data["negation"]) if data.get("negation") else None,
            cross_references=[CrossReference.from_dict(c) for c in data["cross_references"]] if data.get("cross_references") else None,
            branches=[WeightedBranch.from_dict(b) for b in data.get("branches", [])],
            decomposition_timestamp=datetime.fromisoformat(data["decomposition_timestamp"]) if data.get("decomposition_timestamp") else datetime.now(),
            pipeline_version=data.get("pipeline_version", "1.0.0"),
            config_hash=data.get("config_hash", ""),
        )

    @classmethod
    def from_json(cls, json_str: str) -> DecomposedKnowledge:
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def get_primary_branch(self) -> WeightedBranch | None:
        """Get the highest-weighted branch."""
        if not self.branches:
            return None
        return max(self.branches, key=lambda b: b.final_weight)

    def get_presuppositions_by_trigger(
        self, trigger_type: PresuppositionTrigger
    ) -> list[Presupposition]:
        """Get presuppositions filtered by trigger type."""
        return [p for p in self.presuppositions if p.trigger_type == trigger_type]

    def get_inferences_by_relation(
        self, relation: CommonsenseRelation
    ) -> list[CommonsenseInference]:
        """Get commonsense inferences filtered by relation type."""
        return [i for i in self.commonsense_inferences if i.relation == relation]

    def __repr__(self) -> str:
        """Concise string representation."""
        return (
            f"DecomposedKnowledge("
            f"text='{self.source_text[:30]}...', "
            f"entities={len(self.entity_ids)}, "
            f"roles={len(self.semantic_roles)}, "
            f"presups={len(self.presuppositions)}, "
            f"inferences={len(self.commonsense_inferences)}, "
            f"branches={len(self.branches)})"
        )
