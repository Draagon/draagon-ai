"""Industry benchmark classification framework.

This module defines classification tiers based on published NLP benchmarks:

Tier Classifications:
    - BELOW_INDUSTRY: Below published baseline performance
    - INDUSTRY_BASELINE: Matches basic published baselines (rule-based, early ML)
    - INDUSTRY_STANDARD: Matches typical published results (standard ML/neural)
    - ABOVE_STANDARD: Exceeds typical published results
    - STATE_OF_ART: Approaches or matches SOTA (transformer-based, fine-tuned)

Each task has thresholds calibrated to published benchmarks:

    NEGATION DETECTION:
    - BioScope corpus: NegBERT achieves 95.68% F1 (abstracts), 90.95% F1 (SFU)
    - Baseline systems: ~70-80% F1
    - Reference: https://arxiv.org/abs/1911.04211

    SEMANTIC ROLE LABELING:
    - CoNLL-2005: SOTA ~89% F1 (newswire), ~82% F1 (out-of-domain)
    - CoNLL-2012: SOTA ~88% F1
    - Baseline: ~75% F1
    - Reference: https://paperswithcode.com/sota/semantic-role-labeling-predicted-predicates

    TEMPORAL EXTRACTION:
    - TempEval-3: SOTA ~90% F1 for identification, ~77-86% for normalization
    - Clinical TempEval: ~83% F1
    - Reference: https://paperswithcode.com/sota/temporal-information-extraction-on-tempeval-3

    MODALITY DETECTION:
    - LEXDEMOD (deontic): Reasonable transfer from fine-tuned transformers
    - Epistemic/Deontic disambiguation: Challenging even for SOTA
    - Reference: https://aclanthology.org/2022.emnlp-main.795/

    COMMONSENSE INFERENCE:
    - ATOMIC-2020: 91.3% human acceptance rate
    - COMET: 77.5% precision (ATOMIC), 91.7% (ConceptNet)
    - Reference: https://arxiv.org/abs/2010.05953
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol
from collections.abc import Sequence


class BenchmarkTier(Enum):
    """Classification tier relative to industry benchmarks.

    Based on published NLP benchmark performance levels.
    """

    BELOW_INDUSTRY = "below_industry"
    """Below published baseline performance. Needs significant improvement."""

    INDUSTRY_BASELINE = "industry_baseline"
    """Matches basic baselines (rule-based, early ML). Acceptable minimum."""

    INDUSTRY_STANDARD = "industry_standard"
    """Matches typical published results. Solid production-ready."""

    ABOVE_STANDARD = "above_standard"
    """Exceeds typical results. Competitive with recent work."""

    STATE_OF_ART = "state_of_art"
    """Approaches or matches SOTA. Publishable quality."""


@dataclass
class BenchmarkThresholds:
    """Threshold values for classifying performance into tiers.

    Attributes:
        baseline: Minimum to achieve INDUSTRY_BASELINE (e.g., 0.70)
        standard: Minimum to achieve INDUSTRY_STANDARD (e.g., 0.80)
        above: Minimum to achieve ABOVE_STANDARD (e.g., 0.88)
        sota: Minimum to achieve STATE_OF_ART (e.g., 0.93)
    """

    baseline: float
    standard: float
    above: float
    sota: float

    def classify(self, score: float) -> BenchmarkTier:
        """Classify a score into the appropriate tier."""
        if score >= self.sota:
            return BenchmarkTier.STATE_OF_ART
        elif score >= self.above:
            return BenchmarkTier.ABOVE_STANDARD
        elif score >= self.standard:
            return BenchmarkTier.INDUSTRY_STANDARD
        elif score >= self.baseline:
            return BenchmarkTier.INDUSTRY_BASELINE
        else:
            return BenchmarkTier.BELOW_INDUSTRY


# =============================================================================
# Task-Specific Benchmark Thresholds (calibrated to published results)
# =============================================================================

NEGATION_THRESHOLDS = BenchmarkThresholds(
    baseline=0.70,  # Basic rule-based systems
    standard=0.85,  # Standard ML approaches
    above=0.91,     # NegBERT-level on SFU corpus
    sota=0.95,      # NegBERT on BioScope abstracts
)
"""
Negation detection thresholds based on:
- BioScope corpus: NegBERT 95.68% F1 (abstracts), 91.24% F1 (full papers)
- SFU Review Corpus: NegBERT 90.95% F1
- Cross-domain rule-based: ~70-80% F1
"""

SRL_THRESHOLDS = BenchmarkThresholds(
    baseline=0.65,  # Basic pattern matching
    standard=0.78,  # Standard ML (pre-neural)
    above=0.85,     # Good neural models
    sota=0.89,      # LISA + ELMo, BERT-based
)
"""
Semantic Role Labeling thresholds based on:
- CoNLL-2005 WSJ: SOTA ~89% F1
- CoNLL-2005 Brown (OOD): SOTA ~82% F1
- CoNLL-2012: SOTA ~88% F1
- Early ML baselines: ~75% F1
"""

TEMPORAL_THRESHOLDS = BenchmarkThresholds(
    baseline=0.65,  # Basic regex/rule-based
    standard=0.80,  # HeidelTime, SUTime level
    above=0.86,     # Good neural approaches
    sota=0.90,      # SOTA extraction (identification)
)
"""
Temporal extraction thresholds based on:
- TempEval-3: SOTA ~90% F1 identification
- HeidelTime/SUTime: ~80-85% F1
- Clinical TempEval: ~83% F1
"""

MODALITY_THRESHOLDS = BenchmarkThresholds(
    baseline=0.60,  # Basic modal detection
    standard=0.75,  # Good modal detection
    above=0.82,     # Epistemic/deontic disambiguation
    sota=0.88,      # Fine-tuned transformers on specific domains
)
"""
Modality detection thresholds based on:
- Modal verb detection: Generally high (>90%)
- Epistemic vs Deontic disambiguation: Much harder (~60-75%)
- LEXDEMOD legal domain: Good transfer learning results
"""

COMMONSENSE_THRESHOLDS = BenchmarkThresholds(
    baseline=0.55,  # Basic template matching
    standard=0.70,  # Reasonable inference quality
    above=0.80,     # COMET-level quality
    sota=0.88,      # ATOMIC-2020 level (91% human acceptance)
)
"""
Commonsense inference thresholds based on:
- COMET: 77.5% precision@1 on ATOMIC
- ATOMIC-2020: 91.3% human acceptance rate
- Template-based: ~55-65% quality
"""

ASPECT_THRESHOLDS = BenchmarkThresholds(
    baseline=0.55,  # Basic stative vs dynamic
    standard=0.70,  # Vendler classification
    above=0.80,     # Context-aware classification
    sota=0.88,      # Full aspectual analysis
)
"""
Aspectual classification thresholds based on:
- Vendler classification is subjective; thresholds estimated
- Stative detection relatively easy
- Achievement vs Activity vs Accomplishment harder
"""


# =============================================================================
# Test Case Structure
# =============================================================================


@dataclass
class BenchmarkTestCase:
    """A single test case with ground truth and metadata.

    Attributes:
        id: Unique identifier for the test case
        text: Input text to analyze
        expected: Dictionary of expected results
        category: Category/phenomenon being tested
        difficulty: Estimated difficulty (easy/medium/hard)
        source: Source corpus or linguistic literature
        notes: Linguistic explanation
    """

    id: str
    text: str
    expected: dict[str, Any]
    category: str
    difficulty: str = "medium"
    source: str = "custom"
    notes: str = ""


@dataclass
class BenchmarkResult:
    """Result of evaluating a component on a test case.

    Attributes:
        test_case: The test case evaluated
        predicted: What the system predicted
        correct: Whether prediction matched expected
        partial_score: Score from 0-1 for partial matches
        details: Detailed breakdown of what matched/failed
    """

    test_case: BenchmarkTestCase
    predicted: dict[str, Any]
    correct: bool
    partial_score: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkSuiteResult:
    """Aggregated results for a benchmark suite.

    Attributes:
        task: Name of the task (e.g., "negation", "srl")
        thresholds: The threshold configuration used
        results: Individual test case results
        metrics: Computed metrics (accuracy, F1, etc.)
        tier: Classified performance tier
    """

    task: str
    thresholds: BenchmarkThresholds
    results: list[BenchmarkResult]
    metrics: dict[str, float]
    tier: BenchmarkTier

    @property
    def accuracy(self) -> float:
        """Simple accuracy metric."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.correct) / len(self.results)

    @property
    def partial_accuracy(self) -> float:
        """Accuracy using partial scores."""
        if not self.results:
            return 0.0
        return sum(r.partial_score for r in self.results) / len(self.results)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for reporting."""
        return {
            "task": self.task,
            "tier": self.tier.value,
            "tier_label": self._tier_label(),
            "accuracy": self.accuracy,
            "partial_accuracy": self.partial_accuracy,
            "metrics": self.metrics,
            "thresholds": {
                "baseline": self.thresholds.baseline,
                "standard": self.thresholds.standard,
                "above": self.thresholds.above,
                "sota": self.thresholds.sota,
            },
            "total_tests": len(self.results),
            "passed": sum(1 for r in self.results if r.correct),
            "failed": sum(1 for r in self.results if not r.correct),
        }

    def _tier_label(self) -> str:
        """Human-readable tier label."""
        labels = {
            BenchmarkTier.BELOW_INDUSTRY: "⚠️ BELOW INDUSTRY",
            BenchmarkTier.INDUSTRY_BASELINE: "📊 INDUSTRY BASELINE",
            BenchmarkTier.INDUSTRY_STANDARD: "✓ INDUSTRY STANDARD",
            BenchmarkTier.ABOVE_STANDARD: "★ ABOVE STANDARD",
            BenchmarkTier.STATE_OF_ART: "🏆 STATE OF THE ART",
        }
        return labels.get(self.tier, "UNKNOWN")


# =============================================================================
# Benchmark Test Suites
# =============================================================================


NEGATION_TEST_SUITE: list[BenchmarkTestCase] = [
    # Explicit negation - should be easy
    BenchmarkTestCase(
        id="neg-explicit-001",
        text="He did not go to the store",
        expected={"has_negation": True, "type": "explicit", "cues": ["not"]},
        category="explicit_negation",
        difficulty="easy",
        source="synthetic",
        notes="Standard 'not' negation",
    ),
    BenchmarkTestCase(
        id="neg-explicit-002",
        text="She never arrived at the party",
        expected={"has_negation": True, "type": "explicit", "cues": ["never"]},
        category="explicit_negation",
        difficulty="easy",
        source="synthetic",
    ),
    BenchmarkTestCase(
        id="neg-explicit-003",
        text="Nobody came to help",
        expected={"has_negation": True, "type": "explicit", "cues": ["nobody"]},
        category="explicit_negation",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="neg-explicit-004",
        text="There is nothing we can do",
        expected={"has_negation": True, "type": "explicit", "cues": ["nothing"]},
        category="explicit_negation",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="neg-explicit-005",
        text="I haven't seen her",
        expected={"has_negation": True, "type": "explicit", "cues": ["n't"]},
        category="explicit_negation",
        difficulty="easy",
        notes="Contracted negation",
    ),

    # Morphological negation - medium difficulty
    BenchmarkTestCase(
        id="neg-morph-001",
        text="The unhappy customer complained",
        expected={"has_negation": True, "type": "morphological", "cues": ["unhappy"]},
        category="morphological_negation",
        difficulty="medium",
        notes="un- prefix negation",
    ),
    BenchmarkTestCase(
        id="neg-morph-002",
        text="That claim is impossible to verify",
        expected={"has_negation": True, "type": "morphological", "cues": ["impossible"]},
        category="morphological_negation",
        difficulty="medium",
        notes="im- prefix negation",
    ),
    BenchmarkTestCase(
        id="neg-morph-003",
        text="His behavior was inappropriate",
        expected={"has_negation": True, "type": "morphological", "cues": ["inappropriate"]},
        category="morphological_negation",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="neg-morph-004",
        text="The project was unsuccessful",
        expected={"has_negation": True, "type": "morphological", "cues": ["unsuccessful"]},
        category="morphological_negation",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="neg-morph-005",
        text="They found the evidence to be inconclusive",
        expected={"has_negation": True, "type": "morphological", "cues": ["inconclusive"]},
        category="morphological_negation",
        difficulty="medium",
    ),

    # Implicit negation - harder
    BenchmarkTestCase(
        id="neg-implicit-001",
        text="He failed to notice the error",
        expected={"has_negation": True, "type": "implicit", "cues": ["failed"]},
        category="implicit_negation",
        difficulty="hard",
        notes="'failed to' implies negation of the complement",
    ),
    BenchmarkTestCase(
        id="neg-implicit-002",
        text="She refused to help",
        expected={"has_negation": True, "type": "implicit", "cues": ["refused"]},
        category="implicit_negation",
        difficulty="hard",
    ),
    BenchmarkTestCase(
        id="neg-implicit-003",
        text="They denied the accusation",
        expected={"has_negation": True, "type": "implicit", "cues": ["denied"]},
        category="implicit_negation",
        difficulty="hard",
    ),
    BenchmarkTestCase(
        id="neg-implicit-004",
        text="The evidence lacks credibility",
        expected={"has_negation": True, "type": "implicit", "cues": ["lacks"]},
        category="implicit_negation",
        difficulty="hard",
    ),
    BenchmarkTestCase(
        id="neg-implicit-005",
        text="I doubt that's true",
        expected={"has_negation": True, "type": "implicit", "cues": ["doubt"]},
        category="implicit_negation",
        difficulty="hard",
        notes="'doubt' implies uncertainty/negation",
    ),

    # Double negation
    BenchmarkTestCase(
        id="neg-double-001",
        text="I don't think he won't come",
        expected={"has_negation": True, "type": "double", "cues": ["don't", "won't"]},
        category="double_negation",
        difficulty="hard",
        notes="Double negative - pragmatically positive",
    ),
    BenchmarkTestCase(
        id="neg-double-002",
        text="It's not impossible",
        expected={"has_negation": True, "type": "double", "cues": ["not", "impossible"]},
        category="double_negation",
        difficulty="hard",
        notes="Litotes - rhetorical understatement",
    ),

    # Negative control cases (should NOT detect negation)
    BenchmarkTestCase(
        id="neg-control-001",
        text="The cat sat on the mat",
        expected={"has_negation": False},
        category="control_positive",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="neg-control-002",
        text="She arrived on time",
        expected={"has_negation": False},
        category="control_positive",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="neg-control-003",
        text="They were happy with the result",
        expected={"has_negation": False},
        category="control_positive",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="neg-control-004",
        text="The meeting was productive",
        expected={"has_negation": False},
        category="control_positive",
        difficulty="easy",
    ),

    # Challenging edge cases
    BenchmarkTestCase(
        id="neg-edge-001",
        text="Not only did she win, she broke the record",
        expected={"has_negation": False, "rhetorical": True},
        category="rhetorical_not",
        difficulty="hard",
        notes="'not only' is not true negation",
    ),
    BenchmarkTestCase(
        id="neg-edge-002",
        text="Noting the time, she left early",
        expected={"has_negation": False},
        category="false_positive_risk",
        difficulty="medium",
        notes="'Noting' contains 'not' but isn't negation",
    ),
]


SRL_TEST_SUITE: list[BenchmarkTestCase] = [
    # Basic transitive
    BenchmarkTestCase(
        id="srl-trans-001",
        text="John ate the apple",
        expected={
            "predicates": ["ate"],
            "roles": {"ARG0": "John", "ARG1": "the apple"},
        },
        category="simple_transitive",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="srl-trans-002",
        text="The doctor examined the patient",
        expected={
            "predicates": ["examined"],
            "roles": {"ARG0": "The doctor", "ARG1": "the patient"},
        },
        category="simple_transitive",
        difficulty="easy",
    ),

    # Ditransitive / Transfer verbs
    BenchmarkTestCase(
        id="srl-ditrans-001",
        text="Doug gave the book to Mary",
        expected={
            "predicates": ["gave"],
            "roles": {"ARG0": "Doug", "ARG1": "the book", "ARG2": "Mary"},
        },
        category="ditransitive",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="srl-ditrans-002",
        text="She sent him a letter",
        expected={
            "predicates": ["sent"],
            "roles": {"ARG0": "She", "ARG1": "a letter", "ARG2": "him"},
        },
        category="ditransitive",
        difficulty="medium",
    ),

    # Passive voice - CRITICAL TEST (identified as gap)
    BenchmarkTestCase(
        id="srl-passive-001",
        text="The cat was chased by the dog",
        expected={
            "predicates": ["chased"],  # NOT "was"
            "passive": True,
            "roles": {"ARG0": "the dog", "ARG1": "The cat"},
        },
        category="passive",
        difficulty="medium",
        notes="Passive: surface subject is patient, by-phrase is agent",
    ),
    BenchmarkTestCase(
        id="srl-passive-002",
        text="The meeting was cancelled",
        expected={
            "predicates": ["cancelled"],
            "passive": True,
            "roles": {"ARG1": "The meeting"},  # Suppressed agent
        },
        category="passive_agentless",
        difficulty="medium",
        notes="Passive without expressed agent",
    ),
    BenchmarkTestCase(
        id="srl-passive-003",
        text="The window was broken by the ball",
        expected={
            "predicates": ["broken"],
            "passive": True,
            "roles": {"ARG0": "the ball", "ARG1": "The window"},
        },
        category="passive",
        difficulty="medium",
    ),

    # Modifiers
    BenchmarkTestCase(
        id="srl-mod-001",
        text="She arrived yesterday",
        expected={
            "predicates": ["arrived"],
            "roles": {"ARG0": "She", "ARGM-TMP": "yesterday"},
        },
        category="temporal_modifier",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="srl-mod-002",
        text="He ran quickly through the park",
        expected={
            "predicates": ["ran"],
            "roles": {"ARG0": "He", "ARGM-MNR": "quickly", "ARGM-LOC": "through the park"},
        },
        category="multiple_modifiers",
        difficulty="medium",
    ),

    # Complex: Multiple predicates
    BenchmarkTestCase(
        id="srl-multi-001",
        text="John saw Mary and gave her the book",
        expected={
            "predicates": ["saw", "gave"],
            "roles_saw": {"ARG0": "John", "ARG1": "Mary"},
            "roles_gave": {"ARG0": "John", "ARG1": "the book", "ARG2": "her"},
        },
        category="coordination",
        difficulty="hard",
    ),

    # Control verbs
    BenchmarkTestCase(
        id="srl-control-001",
        text="She wanted to leave",
        expected={
            "predicates": ["wanted", "leave"],
            "control": True,
            "controller": "She",
        },
        category="subject_control",
        difficulty="hard",
    ),
    BenchmarkTestCase(
        id="srl-control-002",
        text="He persuaded her to go",
        expected={
            "predicates": ["persuaded", "go"],
            "control": True,
            "controller": "her",
        },
        category="object_control",
        difficulty="hard",
    ),
]


TEMPORAL_TEST_SUITE: list[BenchmarkTestCase] = [
    # Tense detection
    BenchmarkTestCase(
        id="temp-tense-001",
        text="She arrived yesterday",
        expected={"tense": "PAST"},
        category="tense",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="temp-tense-002",
        text="He will arrive tomorrow",
        expected={"tense": "FUTURE"},
        category="tense",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="temp-tense-003",
        text="I know the answer",
        expected={"tense": "PRESENT"},
        category="tense",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="temp-tense-004",
        text="They were sleeping",
        expected={"tense": "PAST"},
        category="tense",
        difficulty="easy",
    ),

    # Aspect classification - Vendler categories
    BenchmarkTestCase(
        id="temp-aspect-001",
        text="I know the answer",
        expected={"aspect": "STATE"},
        category="aspect_state",
        difficulty="easy",
        notes="'know' is a prototypical stative verb",
    ),
    BenchmarkTestCase(
        id="temp-aspect-002",
        text="She loves chocolate",
        expected={"aspect": "STATE"},
        category="aspect_state",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="temp-aspect-003",
        text="The cat ran across the yard",
        expected={"aspect": "ACTIVITY"},
        category="aspect_activity",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="temp-aspect-004",
        text="They were swimming in the pool",
        expected={"aspect": "ACTIVITY"},
        category="aspect_activity",
        difficulty="medium",
    ),
    # ACHIEVEMENT - identified gap: punctual, telic events
    BenchmarkTestCase(
        id="temp-aspect-005",
        text="She arrived at noon",
        expected={"aspect": "ACHIEVEMENT"},
        category="aspect_achievement",
        difficulty="hard",
        notes="'arrive' is punctual and telic - ACHIEVEMENT not ACTIVITY",
    ),
    BenchmarkTestCase(
        id="temp-aspect-006",
        text="He reached the summit",
        expected={"aspect": "ACHIEVEMENT"},
        category="aspect_achievement",
        difficulty="hard",
    ),
    BenchmarkTestCase(
        id="temp-aspect-007",
        text="The bomb exploded",
        expected={"aspect": "ACHIEVEMENT"},
        category="aspect_achievement",
        difficulty="hard",
        notes="Instantaneous change of state",
    ),
    BenchmarkTestCase(
        id="temp-aspect-008",
        text="She built a house",
        expected={"aspect": "ACCOMPLISHMENT"},
        category="aspect_accomplishment",
        difficulty="hard",
        notes="Telic with duration",
    ),
    BenchmarkTestCase(
        id="temp-aspect-009",
        text="He wrote a letter",
        expected={"aspect": "ACCOMPLISHMENT"},
        category="aspect_accomplishment",
        difficulty="hard",
    ),

    # Temporal expressions
    BenchmarkTestCase(
        id="temp-expr-001",
        text="The meeting is at 3pm on Monday",
        expected={"has_temporal": True, "expressions": ["3pm", "Monday"]},
        category="temporal_expression",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="temp-expr-002",
        text="The movie lasted for two hours",
        expected={"has_temporal": True, "duration": True},
        category="duration",
        difficulty="medium",
    ),
]


MODALITY_TEST_SUITE: list[BenchmarkTestCase] = [
    # Epistemic modality
    BenchmarkTestCase(
        id="mod-epist-001",
        text="He must have left by now",
        expected={"type": "EPISTEMIC", "certainty_high": True},
        category="epistemic_necessity",
        difficulty="medium",
        notes="'must have' is epistemic - logical inference",
    ),
    BenchmarkTestCase(
        id="mod-epist-002",
        text="She might come to the party",
        expected={"type": "EPISTEMIC", "certainty_low": True},
        category="epistemic_possibility",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="mod-epist-003",
        text="He could be at home",
        expected={"type": "EPISTEMIC", "certainty_low": True},
        category="epistemic_possibility",
        difficulty="medium",
    ),

    # Deontic modality - CRITICAL (identified gap)
    BenchmarkTestCase(
        id="mod-deontic-001",
        text="You must leave now",
        expected={"type": "DEONTIC", "force": "obligation"},
        category="deontic_obligation",
        difficulty="medium",
        notes="'must' here is DEONTIC obligation, NOT epistemic",
    ),
    BenchmarkTestCase(
        id="mod-deontic-002",
        text="You may leave early",
        expected={"type": "DEONTIC", "force": "permission"},
        category="deontic_permission",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="mod-deontic-003",
        text="You should try this",
        expected={"type": "DEONTIC", "force": "obligation"},
        category="deontic_obligation",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="mod-deontic-004",
        text="Students must submit by Friday",
        expected={"type": "DEONTIC", "force": "obligation"},
        category="deontic_obligation",
        difficulty="medium",
        notes="Institutional obligation context",
    ),

    # Ability (dynamic modality)
    BenchmarkTestCase(
        id="mod-ability-001",
        text="She can speak French fluently",
        expected={"type": "DEONTIC", "force": "ability"},
        category="dynamic_ability",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="mod-ability-002",
        text="He could swim when he was five",
        expected={"type": "DEONTIC", "force": "ability"},
        category="dynamic_ability",
        difficulty="medium",
    ),

    # Evidential markers
    BenchmarkTestCase(
        id="mod-evid-001",
        text="According to John, the meeting was cancelled",
        expected={"evidential": "reported"},
        category="evidential_reported",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="mod-evid-002",
        text="Apparently, she left early",
        expected={"evidential": "reported"},
        category="evidential_reported",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="mod-evid-003",
        text="It seems that they agree",
        expected={"evidential": "inferred"},
        category="evidential_inferred",
        difficulty="medium",
    ),

    # Certainty adverbs
    BenchmarkTestCase(
        id="mod-cert-001",
        text="She definitely knows the answer",
        expected={"certainty": 0.95},
        category="certainty_high",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="mod-cert-002",
        text="She possibly knows the answer",
        expected={"certainty": 0.40},
        category="certainty_low",
        difficulty="easy",
    ),

    # Disambiguation test cases (epistemic vs deontic)
    BenchmarkTestCase(
        id="mod-disambig-001",
        text="He must be tired after that run",
        expected={"type": "EPISTEMIC"},
        category="disambiguation",
        difficulty="hard",
        notes="'must' + stative = epistemic inference about state",
    ),
    BenchmarkTestCase(
        id="mod-disambig-002",
        text="He must finish the report",
        expected={"type": "DEONTIC"},
        category="disambiguation",
        difficulty="hard",
        notes="'must' + action verb = deontic obligation",
    ),
]


COMMONSENSE_TEST_SUITE: list[BenchmarkTestCase] = [
    # Basic emotional reactions
    BenchmarkTestCase(
        id="cs-react-001",
        text="Doug forgot the meeting",
        expected={
            "xReact": ["embarrassed", "guilty", "worried"],
            "oReact": ["frustrated", "disappointed", "annoyed"],
        },
        category="emotional_reaction",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="cs-react-002",
        text="She won the lottery",
        expected={
            "xReact": ["excited", "happy", "surprised"],
        },
        category="emotional_reaction",
        difficulty="easy",
    ),
    BenchmarkTestCase(
        id="cs-react-003",
        text="He failed the exam",
        expected={
            "xReact": ["disappointed", "sad", "upset"],
        },
        category="emotional_reaction",
        difficulty="easy",
    ),

    # Intent inference
    BenchmarkTestCase(
        id="cs-intent-001",
        text="She helped her neighbor move",
        expected={
            "xIntent": ["be helpful", "be kind", "build relationship"],
        },
        category="intent",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="cs-intent-002",
        text="He studied all night",
        expected={
            "xIntent": ["pass exam", "do well", "learn material"],
        },
        category="intent",
        difficulty="medium",
    ),

    # Effect inference
    BenchmarkTestCase(
        id="cs-effect-001",
        text="She ate too much",
        expected={
            "xEffect": ["feels sick", "stomach ache", "regrets"],
        },
        category="effect",
        difficulty="medium",
    ),
    BenchmarkTestCase(
        id="cs-effect-002",
        text="He got promoted",
        expected={
            "xEffect": ["earns more", "new responsibilities", "celebrates"],
        },
        category="effect",
        difficulty="medium",
    ),

    # Complex events
    BenchmarkTestCase(
        id="cs-complex-001",
        text="The doctor told him the test results were negative",
        expected={
            "xReact": ["relieved", "happy"],
            "context": "medical_negative_is_positive",
        },
        category="context_dependent",
        difficulty="hard",
        notes="'Negative' in medical context is typically good news",
    ),
]


# =============================================================================
# Benchmark Evaluator
# =============================================================================


class BenchmarkEvaluator(Protocol):
    """Protocol for task-specific evaluators."""

    def evaluate(
        self,
        test_cases: Sequence[BenchmarkTestCase],
    ) -> list[BenchmarkResult]:
        """Evaluate test cases and return results."""
        ...


def compute_metrics(results: list[BenchmarkResult]) -> dict[str, float]:
    """Compute standard metrics from benchmark results.

    Returns:
        Dictionary with accuracy, precision, recall, F1, partial_accuracy
    """
    if not results:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "partial_accuracy": 0.0,
        }

    correct = sum(1 for r in results if r.correct)
    partial_sum = sum(r.partial_score for r in results)

    accuracy = correct / len(results)
    partial_accuracy = partial_sum / len(results)

    # For tasks with binary classification (presence/absence)
    # Calculate precision/recall if details contain TP/FP/FN
    tp = sum(r.details.get("tp", 0) for r in results)
    fp = sum(r.details.get("fp", 0) for r in results)
    fn = sum(r.details.get("fn", 0) for r in results)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "partial_accuracy": partial_accuracy,
    }


def classify_performance(
    metrics: dict[str, float],
    thresholds: BenchmarkThresholds,
    primary_metric: str = "accuracy",
) -> BenchmarkTier:
    """Classify performance into a tier based on metrics."""
    score = metrics.get(primary_metric, metrics.get("accuracy", 0.0))
    return thresholds.classify(score)


def generate_report(suite_results: list[BenchmarkSuiteResult]) -> str:
    """Generate a human-readable benchmark report.

    Args:
        suite_results: Results from all benchmark suites

    Returns:
        Formatted string report
    """
    lines = [
        "=" * 70,
        "BENCHMARK REPORT - INDUSTRY COMPARISON",
        "=" * 70,
        "",
    ]

    for result in suite_results:
        data = result.to_dict()
        lines.extend([
            f"Task: {data['task'].upper()}",
            "-" * 40,
            f"  Classification: {data['tier_label']}",
            f"  Accuracy: {data['accuracy']:.1%}",
            f"  Partial Accuracy: {data['partial_accuracy']:.1%}",
            f"  Tests: {data['passed']}/{data['total_tests']} passed",
            "",
            f"  Industry Thresholds:",
            f"    Baseline: {data['thresholds']['baseline']:.0%}",
            f"    Standard: {data['thresholds']['standard']:.0%}",
            f"    Above:    {data['thresholds']['above']:.0%}",
            f"    SOTA:     {data['thresholds']['sota']:.0%}",
            "",
        ])

    # Summary
    tier_counts = {}
    for result in suite_results:
        tier = result.tier.value
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    lines.extend([
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        "",
    ])

    for tier_value, count in sorted(tier_counts.items()):
        tier = BenchmarkTier(tier_value)
        label = {
            BenchmarkTier.BELOW_INDUSTRY: "⚠️ Below Industry",
            BenchmarkTier.INDUSTRY_BASELINE: "📊 Industry Baseline",
            BenchmarkTier.INDUSTRY_STANDARD: "✓ Industry Standard",
            BenchmarkTier.ABOVE_STANDARD: "★ Above Standard",
            BenchmarkTier.STATE_OF_ART: "🏆 State of Art",
        }.get(tier, tier_value)
        lines.append(f"  {label}: {count} task(s)")

    lines.append("")

    return "\n".join(lines)
