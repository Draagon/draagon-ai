"""Tests for decomposition data models."""

import pytest
from datetime import datetime

from decomposition.models import (
    # Enums
    PresuppositionTrigger,
    CommonsenseRelation,
    Aspect,
    Tense,
    ModalType,
    Polarity,
    # Data structures
    SemanticRole,
    Presupposition,
    CommonsenseInference,
    TemporalInfo,
    ModalityInfo,
    NegationInfo,
    CrossReference,
    WeightedBranch,
    DecomposedKnowledge,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test enum definitions."""

    def test_presupposition_trigger_values(self):
        """Test that all trigger types have string values."""
        assert PresuppositionTrigger.DEFINITE_DESC.value == "definite_description"
        assert PresuppositionTrigger.ITERATIVE.value == "iterative"
        assert PresuppositionTrigger.CHANGE_OF_STATE.value == "change_of_state"

    def test_commonsense_relation_tiers(self):
        """Test ATOMIC relation tiers."""
        tier1 = {CommonsenseRelation.X_INTENT, CommonsenseRelation.X_EFFECT,
                 CommonsenseRelation.X_REACT, CommonsenseRelation.X_ATTR}
        tier2 = {CommonsenseRelation.X_NEED, CommonsenseRelation.X_WANT,
                 CommonsenseRelation.O_REACT, CommonsenseRelation.CAUSES}

        # All should have string values
        for r in tier1 | tier2:
            assert isinstance(r.value, str)

    def test_aspect_categories(self):
        """Test Vendler aspect categories."""
        aspects = [Aspect.STATE, Aspect.ACTIVITY, Aspect.ACCOMPLISHMENT,
                   Aspect.ACHIEVEMENT, Aspect.SEMELFACTIVE]
        assert len(aspects) == 5
        for a in aspects:
            assert isinstance(a.value, str)

    def test_tense_values(self):
        """Test tense enum values."""
        assert Tense.PAST.value == "past"
        assert Tense.PRESENT.value == "present"
        assert Tense.FUTURE.value == "future"
        assert Tense.UNKNOWN.value == "unknown"

    def test_modal_types(self):
        """Test modality type enum."""
        assert ModalType.EPISTEMIC.value == "epistemic"
        assert ModalType.DEONTIC.value == "deontic"
        assert ModalType.EVIDENTIAL.value == "evidential"
        assert ModalType.NONE.value == "none"

    def test_polarity_values(self):
        """Test polarity enum."""
        assert Polarity.POSITIVE.value == "positive"
        assert Polarity.NEGATIVE.value == "negative"
        assert Polarity.UNCERTAIN.value == "uncertain"


# =============================================================================
# SemanticRole Tests
# =============================================================================


class TestSemanticRole:
    """Test SemanticRole dataclass."""

    def test_creation(self):
        """Test basic creation."""
        role = SemanticRole(
            predicate="forgot",
            predicate_sense="forget.v.01",
            role="ARG0",
            filler="Doug",
            confidence=0.9,
        )
        assert role.predicate == "forgot"
        assert role.role == "ARG0"
        assert role.filler == "Doug"
        assert role.confidence == 0.9

    def test_serialization(self):
        """Test to_dict and from_dict."""
        role = SemanticRole(
            predicate="forgot",
            predicate_sense="forget.v.01",
            role="ARG0",
            filler="Doug",
            span=(0, 4),
            confidence=0.9,
        )

        d = role.to_dict()
        assert d["predicate"] == "forgot"
        assert d["span"] == [0, 4]

        restored = SemanticRole.from_dict(d)
        assert restored.predicate == role.predicate
        assert restored.span == role.span

    def test_optional_fields(self):
        """Test that optional fields default correctly."""
        role = SemanticRole(
            predicate="ran",
            predicate_sense=None,
            role="ARG0",
            filler="Doug",
        )
        assert role.filler_id is None
        assert role.span is None
        assert role.confidence == 1.0


# =============================================================================
# Presupposition Tests
# =============================================================================


class TestPresupposition:
    """Test Presupposition dataclass."""

    def test_creation(self):
        """Test basic creation."""
        presup = Presupposition(
            content="Doug forgot before",
            trigger_type=PresuppositionTrigger.ITERATIVE,
            trigger_text="again",
            confidence=0.9,
        )
        assert presup.content == "Doug forgot before"
        assert presup.trigger_type == PresuppositionTrigger.ITERATIVE
        assert presup.cancellable is True  # Default

    def test_serialization(self):
        """Test to_dict and from_dict."""
        presup = Presupposition(
            content="A meeting exists",
            trigger_type=PresuppositionTrigger.DEFINITE_DESC,
            trigger_text="the meeting",
            trigger_span=(15, 26),
            confidence=0.85,
            cancellable=True,
            entity_ids=["entity-123"],
        )

        d = presup.to_dict()
        assert d["trigger_type"] == "definite_description"
        assert d["entity_ids"] == ["entity-123"]

        restored = Presupposition.from_dict(d)
        assert restored.trigger_type == PresuppositionTrigger.DEFINITE_DESC
        assert restored.entity_ids == ["entity-123"]


# =============================================================================
# CommonsenseInference Tests
# =============================================================================


class TestCommonsenseInference:
    """Test CommonsenseInference dataclass."""

    def test_creation(self):
        """Test basic creation."""
        inference = CommonsenseInference(
            relation=CommonsenseRelation.X_INTENT,
            head="Doug bought flowers",
            tail="to show love",
            confidence=0.8,
        )
        assert inference.relation == CommonsenseRelation.X_INTENT
        assert inference.head == "Doug bought flowers"
        assert inference.source == "llm"  # Default

    def test_serialization(self):
        """Test to_dict and from_dict."""
        inference = CommonsenseInference(
            relation=CommonsenseRelation.X_EFFECT,
            head="Doug dropped the glass",
            tail="has to clean up",
            head_entity_ids=["doug-123"],
            confidence=0.75,
            source="comet",
        )

        d = inference.to_dict()
        assert d["relation"] == "xEffect"
        assert d["source"] == "comet"

        restored = CommonsenseInference.from_dict(d)
        assert restored.relation == CommonsenseRelation.X_EFFECT
        assert restored.source == "comet"


# =============================================================================
# TemporalInfo Tests
# =============================================================================


class TestTemporalInfo:
    """Test TemporalInfo dataclass."""

    def test_defaults(self):
        """Test default values."""
        temporal = TemporalInfo()
        assert temporal.aspect == Aspect.STATE
        assert temporal.tense == Tense.UNKNOWN
        assert temporal.reference_type is None

    def test_full_creation(self):
        """Test with all fields."""
        temporal = TemporalInfo(
            aspect=Aspect.ACHIEVEMENT,
            tense=Tense.PAST,
            reference_type="deictic",
            reference_value="yesterday",
            duration="for 3 hours",
            frequency="always",
        )
        assert temporal.aspect == Aspect.ACHIEVEMENT
        assert temporal.reference_value == "yesterday"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        temporal = TemporalInfo(
            aspect=Aspect.ACTIVITY,
            tense=Tense.PRESENT,
            frequency="every day",
        )

        d = temporal.to_dict()
        assert d["aspect"] == "activity"
        assert d["tense"] == "present"

        restored = TemporalInfo.from_dict(d)
        assert restored.aspect == Aspect.ACTIVITY


# =============================================================================
# ModalityInfo Tests
# =============================================================================


class TestModalityInfo:
    """Test ModalityInfo dataclass."""

    def test_defaults(self):
        """Test default values."""
        modality = ModalityInfo()
        assert modality.modal_type == ModalType.NONE
        assert modality.certainty is None

    def test_epistemic(self):
        """Test epistemic modality."""
        modality = ModalityInfo(
            modal_type=ModalType.EPISTEMIC,
            modal_marker="probably",
            certainty=0.7,
        )
        assert modality.certainty == 0.7

    def test_evidential(self):
        """Test evidential modality."""
        modality = ModalityInfo(
            modal_type=ModalType.EVIDENTIAL,
            modal_marker="apparently",
            evidence_source="reported",
        )
        assert modality.evidence_source == "reported"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        modality = ModalityInfo(
            modal_type=ModalType.DEONTIC,
            modal_marker="should",
        )

        d = modality.to_dict()
        assert d["modal_type"] == "deontic"

        restored = ModalityInfo.from_dict(d)
        assert restored.modal_type == ModalType.DEONTIC


# =============================================================================
# NegationInfo Tests
# =============================================================================


class TestNegationInfo:
    """Test NegationInfo dataclass."""

    def test_positive_default(self):
        """Test default is positive polarity."""
        negation = NegationInfo()
        assert negation.is_negated is False
        assert negation.polarity == Polarity.POSITIVE

    def test_negated(self):
        """Test negated statement."""
        negation = NegationInfo(
            is_negated=True,
            negation_cue="not",
            negation_scope="attend the meeting",
            polarity=Polarity.NEGATIVE,
        )
        assert negation.is_negated is True
        assert negation.negation_cue == "not"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        negation = NegationInfo(
            is_negated=True,
            negation_cue="never",
            negation_scope_span=(5, 20),
            polarity=Polarity.NEGATIVE,
        )

        d = negation.to_dict()
        assert d["is_negated"] is True
        assert d["negation_scope_span"] == [5, 20]

        restored = NegationInfo.from_dict(d)
        assert restored.negation_scope_span == (5, 20)


# =============================================================================
# WeightedBranch Tests
# =============================================================================


class TestWeightedBranch:
    """Test WeightedBranch dataclass."""

    def test_auto_weight_computation(self):
        """Test that final_weight is computed from confidence + memory_support."""
        branch = WeightedBranch(
            interpretation="Doug forgot",
            confidence=0.8,
            memory_support=0.1,
        )
        assert branch.final_weight == 0.9

    def test_update_weight(self):
        """Test updating weight with memory support."""
        branch = WeightedBranch(
            interpretation="Test",
            confidence=0.7,
        )
        assert branch.final_weight == 0.7

        branch.update_weight(memory_adjustment=0.2)
        assert branch.memory_support == 0.2
        assert abs(branch.final_weight - 0.9) < 1e-9  # Float precision

    def test_serialization(self):
        """Test to_dict and from_dict."""
        branch = WeightedBranch(
            interpretation="Primary interpretation",
            confidence=0.85,
            memory_support=0.05,
            supporting_evidence=["evidence1", "evidence2"],
        )

        d = branch.to_dict()
        assert d["confidence"] == 0.85
        assert len(d["supporting_evidence"]) == 2

        restored = WeightedBranch.from_dict(d)
        assert restored.confidence == 0.85
        assert restored.final_weight == 0.9


# =============================================================================
# DecomposedKnowledge Tests
# =============================================================================


class TestDecomposedKnowledge:
    """Test DecomposedKnowledge dataclass."""

    def test_basic_creation(self):
        """Test creating with minimal data."""
        decomposed = DecomposedKnowledge(
            source_text="Doug forgot the meeting again",
        )
        assert decomposed.source_text == "Doug forgot the meeting again"
        assert len(decomposed.source_id) > 0  # Auto-generated UUID
        assert decomposed.entity_ids == []
        assert decomposed.presuppositions == []

    def test_full_creation(self):
        """Test creating with all fields."""
        decomposed = DecomposedKnowledge(
            source_text="Doug forgot the meeting again",
            entity_ids=["doug-123", "meeting-456"],
            presuppositions=[
                Presupposition(
                    content="Doug forgot before",
                    trigger_type=PresuppositionTrigger.ITERATIVE,
                    trigger_text="again",
                )
            ],
            commonsense_inferences=[
                CommonsenseInference(
                    relation=CommonsenseRelation.X_REACT,
                    head="Doug forgot the meeting",
                    tail="embarrassed",
                )
            ],
            temporal=TemporalInfo(
                tense=Tense.PAST,
                aspect=Aspect.ACHIEVEMENT,
            ),
            branches=[
                WeightedBranch(
                    interpretation="Doug forgot",
                    confidence=0.9,
                )
            ],
        )

        assert len(decomposed.presuppositions) == 1
        assert len(decomposed.commonsense_inferences) == 1
        assert decomposed.temporal.tense == Tense.PAST
        assert len(decomposed.branches) == 1

    def test_serialization_roundtrip(self):
        """Test full serialization roundtrip."""
        decomposed = DecomposedKnowledge(
            source_text="Test sentence",
            entity_ids=["entity-1"],
            semantic_roles=[
                SemanticRole(
                    predicate="test",
                    predicate_sense="test.v.01",
                    role="ARG0",
                    filler="subject",
                )
            ],
            presuppositions=[
                Presupposition(
                    content="Something exists",
                    trigger_type=PresuppositionTrigger.DEFINITE_DESC,
                    trigger_text="the thing",
                )
            ],
            temporal=TemporalInfo(tense=Tense.PAST),
            modality=ModalityInfo(modal_type=ModalType.EPISTEMIC, certainty=0.8),
            negation=NegationInfo(is_negated=False),
        )

        # To dict and back
        d = decomposed.to_dict()
        restored = DecomposedKnowledge.from_dict(d)

        assert restored.source_text == decomposed.source_text
        assert len(restored.semantic_roles) == 1
        assert len(restored.presuppositions) == 1
        assert restored.temporal.tense == Tense.PAST
        assert restored.modality.certainty == 0.8

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        decomposed = DecomposedKnowledge(
            source_text="JSON test",
            branches=[
                WeightedBranch(interpretation="Branch 1", confidence=0.8),
                WeightedBranch(interpretation="Branch 2", confidence=0.6),
            ],
        )

        json_str = decomposed.to_json()
        restored = DecomposedKnowledge.from_json(json_str)

        assert restored.source_text == "JSON test"
        assert len(restored.branches) == 2

    def test_get_primary_branch(self):
        """Test getting highest-weighted branch."""
        decomposed = DecomposedKnowledge(
            source_text="Test",
            branches=[
                WeightedBranch(interpretation="Low", confidence=0.5),
                WeightedBranch(interpretation="High", confidence=0.9),
                WeightedBranch(interpretation="Medium", confidence=0.7),
            ],
        )

        primary = decomposed.get_primary_branch()
        assert primary.interpretation == "High"
        assert primary.confidence == 0.9

    def test_get_primary_branch_empty(self):
        """Test get_primary_branch with no branches."""
        decomposed = DecomposedKnowledge(source_text="Test")
        assert decomposed.get_primary_branch() is None

    def test_get_presuppositions_by_trigger(self):
        """Test filtering presuppositions by trigger type."""
        decomposed = DecomposedKnowledge(
            source_text="Test",
            presuppositions=[
                Presupposition(
                    content="P1",
                    trigger_type=PresuppositionTrigger.ITERATIVE,
                    trigger_text="again",
                ),
                Presupposition(
                    content="P2",
                    trigger_type=PresuppositionTrigger.DEFINITE_DESC,
                    trigger_text="the meeting",
                ),
                Presupposition(
                    content="P3",
                    trigger_type=PresuppositionTrigger.ITERATIVE,
                    trigger_text="another",
                ),
            ],
        )

        iteratives = decomposed.get_presuppositions_by_trigger(
            PresuppositionTrigger.ITERATIVE
        )
        assert len(iteratives) == 2

        definites = decomposed.get_presuppositions_by_trigger(
            PresuppositionTrigger.DEFINITE_DESC
        )
        assert len(definites) == 1

    def test_get_inferences_by_relation(self):
        """Test filtering inferences by relation type."""
        decomposed = DecomposedKnowledge(
            source_text="Test",
            commonsense_inferences=[
                CommonsenseInference(
                    relation=CommonsenseRelation.X_INTENT,
                    head="event",
                    tail="intent1",
                ),
                CommonsenseInference(
                    relation=CommonsenseRelation.X_EFFECT,
                    head="event",
                    tail="effect1",
                ),
                CommonsenseInference(
                    relation=CommonsenseRelation.X_INTENT,
                    head="event",
                    tail="intent2",
                ),
            ],
        )

        intents = decomposed.get_inferences_by_relation(CommonsenseRelation.X_INTENT)
        assert len(intents) == 2

        effects = decomposed.get_inferences_by_relation(CommonsenseRelation.X_EFFECT)
        assert len(effects) == 1

    def test_repr(self):
        """Test string representation."""
        decomposed = DecomposedKnowledge(
            source_text="A very long sentence that should be truncated in the repr output",
            entity_ids=["e1", "e2"],
            presuppositions=[
                Presupposition(
                    content="P1",
                    trigger_type=PresuppositionTrigger.ITERATIVE,
                    trigger_text="again",
                )
            ],
        )

        repr_str = repr(decomposed)
        assert "DecomposedKnowledge" in repr_str
        assert "entities=2" in repr_str
        assert "presups=1" in repr_str
