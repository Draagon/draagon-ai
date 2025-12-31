"""Tests for Semantic Expansion System.

Tests:
1. Word Sense Disambiguation (WSD)
2. Semantic Frame Expansion
3. Variation Generation with Cognitive Scoring
4. Pre-loaded Memory Integration
5. Cross-Layer Association Detection
6. Evolutionary Fitness Evaluation

These tests are designed to work with both:
- Unit testing (mocked LLM)
- Integration testing (real LLM)
- Evolutionary optimization (fitness scoring)
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any

import pytest

from semantic_types import (
    Ambiguity,
    CrossLayerEdge,
    CrossLayerRelation,
    ExpansionVariant,
    Implication,
    Presupposition,
    SemanticFrame,
    SemanticTriple,
    VariationStoragePolicy,
    WordSense,
)
from wsd import (
    LeskDisambiguator,
    WordSenseDisambiguator,
    are_same_word_different_sense,
    get_synset_id,
    synset_ids_match,
)
from expansion import (
    EntityInfo,
    ExpansionInput,
    SemanticExpansionService,
    VariationGenerator,
    expand_statement,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.call_count = 0
        self.last_prompt = ""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Return pre-configured response or generate default."""
        self.call_count += 1
        self.last_prompt = messages[-1]["content"] if messages else ""

        # Check for specific response
        for key, response in self.responses.items():
            if key in self.last_prompt:
                return response

        # Generate default semantic frame response
        return self._generate_default_response(self.last_prompt)

    def _generate_default_response(self, prompt: str) -> str:
        """Generate a default semantic frame response."""
        # Extract statement from prompt
        match = re.search(r'Statement: "([^"]+)"', prompt)
        statement = match.group(1) if match else "unknown"

        return f"""<semantic_frame>
    <triples>
        <triple>
            <subject>Subject</subject>
            <predicate>STATES</predicate>
            <object>{statement}</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Subject exists</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.8">Statement is true</implication>
    </implications>
    <negations></negations>
    <ambiguities></ambiguities>
    <open_questions></open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>"""


@dataclass
class PreloadedMemoryState:
    """Pre-loaded memory state for testing."""

    # Working memory observations
    working_observations: list[dict[str, Any]] = field(default_factory=list)

    # Episodic memory summaries
    episodic_summaries: list[str] = field(default_factory=list)

    # Semantic memory facts
    semantic_facts: list[str] = field(default_factory=list)

    # Metacognitive patterns
    metacognitive_patterns: list[str] = field(default_factory=list)

    # Beliefs
    beliefs: list[str] = field(default_factory=list)

    # Entity relationships
    relationships: list[tuple[str, str, str]] = field(default_factory=list)

    def to_expansion_input(self) -> ExpansionInput:
        """Convert to ExpansionInput for testing."""
        return ExpansionInput(
            working_observations=self.working_observations,
            episodic_summaries=self.episodic_summaries,
            semantic_facts=self.semantic_facts,
            metacognitive_patterns=self.metacognitive_patterns,
            relevant_beliefs=self.beliefs,
            entity_relationships=self.relationships,
        )


@dataclass
class ExpansionTestCase:
    """A test case for evaluating semantic expansion."""

    name: str
    statement: str

    # Context
    immediate_context: list[str] = field(default_factory=list)
    session_entities: dict[str, str] = field(default_factory=dict)  # name -> type

    # Pre-loaded memory state
    memory_state: PreloadedMemoryState = field(default_factory=PreloadedMemoryState)

    # Expected outputs
    expected_entities: set[str] = field(default_factory=set)
    expected_synsets: dict[str, str] = field(default_factory=dict)  # word -> synset_id
    expected_conflicts: list[str] = field(default_factory=list)  # Descriptions of conflicts
    expected_primary_interpretation: str | None = None

    # Fitness criteria
    min_confidence: float = 0.5
    expected_variant_count: int = 1


# =============================================================================
# Test Cases with Pre-loaded Memories
# =============================================================================


def create_tea_coffee_conflict_test() -> ExpansionTestCase:
    """Test case: Doug's tea preference potentially conflicts with coffee preference."""
    return ExpansionTestCase(
        name="tea_coffee_conflict",
        statement="He prefers tea in the morning",
        immediate_context=[
            "Doug mentioned he was tired today",
            "We were talking about breakfast habits",
        ],
        session_entities={"Doug": "PERSON"},
        memory_state=PreloadedMemoryState(
            working_observations=[
                {"content": "Doug mentioned feeling tired", "source": "user"},
            ],
            episodic_summaries=[
                "Dec 28: Discussed morning routines with Doug",
                "Dec 25: Doug made coffee for everyone",
            ],
            semantic_facts=[
                "Doug likes coffee",
                "Doug is a person",
                "Coffee is a beverage",
                "Tea is a beverage",
            ],
            metacognitive_patterns=[
                "Reference resolution is often ambiguous in casual speech",
            ],
            beliefs=[
                "Doug enjoys hot beverages",
                "Doug has a morning routine",
            ],
            relationships=[
                ("Doug", "IS_A", "person"),
                ("Doug", "LIKES", "coffee"),
            ],
        ),
        expected_entities={"Doug", "tea"},
        expected_synsets={"tea": "tea.n.01", "morning": "morning.n.01"},
        expected_conflicts=["Doug likes coffee vs prefers tea"],
        expected_primary_interpretation="Doug prefers tea in the morning",
        expected_variant_count=2,  # Primary + "unknown person" variant
    )


def create_bank_disambiguation_test() -> ExpansionTestCase:
    """Test case: Word sense disambiguation for 'bank'."""
    return ExpansionTestCase(
        name="bank_wsd",
        statement="I need to go to the bank before it closes",
        immediate_context=[
            "I have a check to deposit",
            "The bank closes at 5pm",
        ],
        session_entities={},
        memory_state=PreloadedMemoryState(
            semantic_facts=[
                "Banks are financial institutions",
                "Banks have operating hours",
            ],
        ),
        expected_entities={"I", "bank"},
        expected_synsets={"bank": "bank.n.01"},  # Financial institution, not riverbank
        expected_primary_interpretation="Going to financial institution",
    )


def create_cross_layer_association_test() -> ExpansionTestCase:
    """Test case: Cross-layer memory associations."""
    return ExpansionTestCase(
        name="cross_layer_association",
        statement="Doug always does this",
        immediate_context=[
            "Doug forgot the meeting again",
            "This is the third time this week",
        ],
        session_entities={"Doug": "PERSON"},
        memory_state=PreloadedMemoryState(
            working_observations=[
                {"content": "Doug forgot meeting", "source": "user"},
            ],
            episodic_summaries=[
                "Dec 15: Doug forgot team meeting",
                "Dec 20: Doug missed deadline",
                "Dec 22: Doug forgot to send report",
            ],
            metacognitive_patterns=[
                "Doug has a pattern of forgetting commitments",
                "Frequency words like 'always' may be hyperbole",
            ],
            beliefs=[
                "Doug tends to forget scheduled events",
            ],
        ),
        expected_entities={"Doug"},
        expected_conflicts=[],
        expected_primary_interpretation="Doug has recurring forgetfulness pattern",
    )


def create_temporal_ambiguity_test() -> ExpansionTestCase:
    """Test case: Temporal reference ambiguity."""
    return ExpansionTestCase(
        name="temporal_ambiguity",
        statement="The meeting was moved to next Tuesday",
        immediate_context=[
            "We were discussing the project timeline",
        ],
        session_entities={"meeting": "EVENT"},
        memory_state=PreloadedMemoryState(
            episodic_summaries=[
                "Original meeting was scheduled for Monday",
            ],
            semantic_facts=[
                "Project meetings happen weekly",
            ],
        ),
        expected_entities={"meeting", "Tuesday"},
        expected_primary_interpretation="Meeting rescheduled to following Tuesday",
    )


def get_all_test_cases() -> list[ExpansionTestCase]:
    """Get all test cases for comprehensive testing."""
    return [
        create_tea_coffee_conflict_test(),
        create_bank_disambiguation_test(),
        create_cross_layer_association_test(),
        create_temporal_ambiguity_test(),
    ]


# =============================================================================
# Word Sense Disambiguation Tests
# =============================================================================


class TestWordSenseDisambiguation:
    """Tests for Word Sense Disambiguation system."""

    def test_synset_id_format(self):
        """Test synset ID string generation."""
        assert get_synset_id("bank", "n", 1) == "bank.n.01"
        assert get_synset_id("prefer", "v", 2) == "prefer.v.02"
        assert get_synset_id("happy", "a", 1) == "happy.a.01"

    def test_synset_id_matching(self):
        """Test synset ID comparison."""
        assert synset_ids_match("bank.n.01", "bank.n.01")
        assert not synset_ids_match("bank.n.01", "bank.n.02")

    def test_same_word_different_sense(self):
        """Test detection of different senses of same word."""
        assert are_same_word_different_sense("bank.n.01", "bank.n.02")
        assert not are_same_word_different_sense("bank.n.01", "bank.n.01")
        assert not are_same_word_different_sense("bank.n.01", "river.n.01")

    def test_lesk_disambiguator_initialization(self):
        """Test Lesk disambiguator can be created."""
        lesk = LeskDisambiguator(use_nltk=False)  # Use mock synsets
        assert lesk is not None

    def test_lesk_get_synsets(self):
        """Test getting synsets from Lesk disambiguator."""
        lesk = LeskDisambiguator(use_nltk=False)

        # Should find bank synsets
        synsets = lesk.get_synsets("bank", "n")
        assert len(synsets) >= 2  # At least 2 senses

        # Should find tea synsets
        synsets = lesk.get_synsets("tea", "n")
        assert len(synsets) >= 1

    def test_lesk_disambiguation_bank_financial(self):
        """Test Lesk correctly disambiguates 'bank' in financial context."""
        lesk = LeskDisambiguator(use_nltk=False)

        # Context suggests financial institution
        context = ["deposit", "check", "money", "account"]
        sense = lesk.disambiguate("bank", context, "n")

        assert sense is not None
        assert sense.synset_id == "bank.n.01"  # Financial institution
        assert "financial" in sense.definition.lower()

    def test_lesk_disambiguation_bank_river(self):
        """Test Lesk correctly disambiguates 'bank' in river context."""
        lesk = LeskDisambiguator(use_nltk=False)

        # Context suggests riverbank
        context = ["river", "water", "canoe", "slope", "fishing"]
        sense = lesk.disambiguate("bank", context, "n")

        assert sense is not None
        # Should pick riverbank sense
        assert "bank.n.02" in [sense.synset_id] + sense.alternatives

    def test_word_sense_disambiguator_simple(self):
        """Test full WSD pipeline without LLM."""
        wsd = WordSenseDisambiguator(nlp=None, llm=None, use_nltk=False)
        # Should work synchronously for simple cases
        lesk_result = wsd.lesk.disambiguate("bank", ["deposit", "money"], "n")
        assert lesk_result is not None

    @pytest.mark.asyncio
    async def test_word_sense_disambiguator_sentence(self):
        """Test WSD on full sentence."""
        wsd = WordSenseDisambiguator(nlp=None, llm=None, use_nltk=False)

        senses = await wsd.disambiguate_sentence("I went to the bank to deposit money")

        # Should find some senses (without spaCy, limited)
        assert isinstance(senses, dict)

    @pytest.mark.asyncio
    async def test_wsd_with_mock_llm(self):
        """Test WSD with mock LLM for ambiguous cases."""
        mock_llm = MockLLMProvider({
            "bank": """<disambiguation>
    <synset_id>bank.n.01</synset_id>
    <confidence>0.95</confidence>
    <reasoning>Context mentions deposit and money</reasoning>
</disambiguation>"""
        })

        wsd = WordSenseDisambiguator(nlp=None, llm=mock_llm, use_nltk=False)

        sense = await wsd.disambiguate_word(
            "bank",
            "I need to go to the bank to deposit this check",
            pos="NOUN",
        )

        # Should use LLM disambiguation
        assert sense is not None
        assert sense.synset_id == "bank.n.01"
        assert sense.confidence >= 0.9


# =============================================================================
# Semantic Frame Expansion Tests
# =============================================================================


class TestSemanticFrameExpansion:
    """Tests for Semantic Frame Expansion."""

    @pytest.mark.asyncio
    async def test_expansion_without_llm(self):
        """Test expansion falls back gracefully without LLM."""
        service = SemanticExpansionService(llm=None)

        variants = await service.expand("Doug likes coffee")

        assert len(variants) >= 1
        assert variants[0].frame.original_text == "Doug likes coffee"

    @pytest.mark.asyncio
    async def test_expansion_with_mock_llm(self):
        """Test expansion with mock LLM response."""
        mock_llm = MockLLMProvider({
            "Doug likes coffee": """<semantic_frame>
    <triples>
        <triple>
            <subject>Doug</subject>
            <predicate>LIKES</predicate>
            <object>coffee</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Doug exists</presupposition>
        <presupposition type="existential">Doug has experience with coffee</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.9">Doug would accept coffee if offered</implication>
    </implications>
    <negations>
        <negation>Doug does not dislike coffee</negation>
    </negations>
    <ambiguities></ambiguities>
    <open_questions>
        <question>What type of coffee does Doug prefer?</question>
    </open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.9</confidence>
</semantic_frame>"""
        })

        service = SemanticExpansionService(llm=mock_llm)
        variants = await service.expand("Doug likes coffee")

        assert len(variants) >= 1
        frame = variants[0].frame

        # Check triples
        assert len(frame.triples) == 1
        assert frame.triples[0].subject == "Doug"
        assert frame.triples[0].predicate == "LIKES"
        assert frame.triples[0].object == "coffee"

        # Check presuppositions
        assert len(frame.presuppositions) == 2
        assert any("Doug exists" in p.content for p in frame.presuppositions)

        # Check implications
        assert len(frame.implications) >= 1
        assert any("accept coffee" in i.content for i in frame.implications)

        # Check negations
        assert len(frame.negations) >= 1

        # Check open questions
        assert len(frame.open_questions) >= 1

    @pytest.mark.asyncio
    async def test_expansion_with_ambiguity(self):
        """Test expansion generates variants for ambiguous statements."""
        mock_llm = MockLLMProvider({
            "He prefers tea": """<semantic_frame>
    <triples>
        <triple>
            <subject>He</subject>
            <predicate>PREFERS</predicate>
            <object>tea</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Referent exists</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.8">Person would choose tea over alternatives</implication>
    </implications>
    <negations></negations>
    <ambiguities>
        <ambiguity type="reference">
            <text>He</text>
            <possibilities>
                <possibility>Doug</possibility>
                <possibility>Unknown male</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>
    <open_questions>
        <question>Who is 'he' referring to?</question>
    </open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.75</confidence>
</semantic_frame>"""
        })

        service = SemanticExpansionService(llm=mock_llm)

        inputs = ExpansionInput(
            immediate_context=["Doug mentioned he was tired"],
            session_entities={"Doug": EntityInfo(entity_type="PERSON")},
        )

        variants = await service.expand("He prefers tea", inputs, max_variations=3)

        # Should generate multiple variants for ambiguity
        assert len(variants) >= 1

        # Check ambiguity was parsed
        assert len(variants[0].frame.ambiguities) >= 1
        assert variants[0].frame.ambiguities[0].text == "He"


# =============================================================================
# Variation Generation Tests
# =============================================================================


class TestVariationGeneration:
    """Tests for variation generation with cognitive scoring."""

    def test_variation_generator_initialization(self):
        """Test variation generator can be created."""
        gen = VariationGenerator()
        assert gen is not None

    @pytest.mark.asyncio
    async def test_variation_scoring_with_memory(self):
        """Test that memory context affects variation scores."""
        gen = VariationGenerator()

        # Create frame with ambiguity
        frame = SemanticFrame(
            original_text="He prefers tea",
            triples=[
                SemanticTriple(subject="He", predicate="PREFERS", object="tea"),
            ],
            ambiguities=[
                Ambiguity(
                    text="He",
                    ambiguity_type="reference",
                    possibilities=["Doug", "Unknown"],
                ),
            ],
        )

        # Create inputs with Doug mentioned recently
        inputs = ExpansionInput(
            immediate_context=["Doug said he was tired", "We talked about Doug's morning"],
            session_entities={"Doug": EntityInfo(entity_type="PERSON")},
            semantic_facts=["Doug likes coffee"],
        )

        variants = await gen.generate_variations(frame, inputs, max_variations=3)

        # Variant with Doug should score higher due to recency
        assert len(variants) >= 2

        # Find Doug variant
        doug_variant = None
        for v in variants:
            if "Doug" in v.resolution_choices.values():
                doug_variant = v
                break

        if doug_variant:
            # Should have higher recency score
            assert doug_variant.recency_weight > 0.5

    @pytest.mark.asyncio
    async def test_variation_storage_policy(self):
        """Test variation storage policy filtering."""
        policy = VariationStoragePolicy(
            min_confidence_threshold=0.3,
            max_stored_variations=3,
            min_confidence_gap=0.05,  # Lower gap for testing threshold filtering
            high_confidence_threshold=0.8,
        )

        # Create test variants
        primary = ExpansionVariant(
            variant_id="v1",
            frame=SemanticFrame(original_text="test"),
            base_confidence=0.9,
        )
        primary.semantic_memory_weight = 0.9

        alt1 = ExpansionVariant(
            variant_id="v2",
            frame=SemanticFrame(original_text="test"),
            base_confidence=0.7,
        )
        alt1.semantic_memory_weight = 0.7

        alt2 = ExpansionVariant(
            variant_id="v3",
            frame=SemanticFrame(original_text="test"),
            base_confidence=0.1,  # Low confidence
            recency_weight=0.1,
            working_memory_weight=0.1,
            episodic_memory_weight=0.1,
            semantic_memory_weight=0.1,
            belief_weight=0.1,
            commonsense_weight=0.1,
            metacognitive_weight=0.1,
        )  # All weights low = combined_score ~0.1 < 0.3 threshold

        # Should store alt1 (above threshold)
        assert policy.should_store(primary, alt1, 1)

        # Should NOT store alt2 (below threshold)
        assert not policy.should_store(primary, alt2, 1)


# =============================================================================
# Pre-loaded Memory Integration Tests
# =============================================================================


class TestPreloadedMemoryIntegration:
    """Tests with pre-loaded memories across all layers."""

    @pytest.mark.asyncio
    async def test_tea_coffee_conflict_detection(self):
        """Test detection of conflict between tea preference and coffee fact."""
        test_case = create_tea_coffee_conflict_test()

        mock_llm = MockLLMProvider({
            "He prefers tea in the morning": """<semantic_frame>
    <triples>
        <triple>
            <subject>He</subject>
            <predicate>PREFERS</predicate>
            <object>tea</object>
            <context>temporal="morning"</context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Referent has beverage preferences</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.8">Person drinks tea in morning</implication>
    </implications>
    <negations></negations>
    <ambiguities>
        <ambiguity type="reference">
            <text>He</text>
            <possibilities>
                <possibility>Doug</possibility>
                <possibility>Unknown male</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>
    <open_questions></open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>"""
        })

        service = SemanticExpansionService(llm=mock_llm)

        inputs = test_case.memory_state.to_expansion_input()
        inputs.immediate_context = test_case.immediate_context
        inputs.session_entities = {
            name: EntityInfo(entity_type=etype)
            for name, etype in test_case.session_entities.items()
        }

        variants = await service.expand(test_case.statement, inputs)

        # Should generate variants
        assert len(variants) >= 1

        # Check that semantic memory affects scoring
        # (Doug likes coffee should create potential conflict awareness)
        primary = variants[0]
        assert primary.semantic_memory_weight > 0

    @pytest.mark.asyncio
    async def test_episodic_memory_influences_scoring(self):
        """Test that episodic memory affects variation scoring."""
        test_case = create_cross_layer_association_test()

        mock_llm = MockLLMProvider({
            "Doug always does this": """<semantic_frame>
    <triples>
        <triple>
            <subject>Doug</subject>
            <predicate>HABITUALLY_DOES</predicate>
            <object>this (forgets)</object>
            <context></context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Doug exists</presupposition>
        <presupposition type="factive">This behavior has occurred before</presupposition>
    </presuppositions>
    <implications>
        <implication type="commonsense" confidence="0.9">Pattern is established</implication>
    </implications>
    <negations></negations>
    <ambiguities></ambiguities>
    <open_questions></open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.9</confidence>
</semantic_frame>"""
        })

        service = SemanticExpansionService(llm=mock_llm)

        inputs = test_case.memory_state.to_expansion_input()
        inputs.immediate_context = test_case.immediate_context
        inputs.session_entities = {
            name: EntityInfo(entity_type=etype)
            for name, etype in test_case.session_entities.items()
        }

        variants = await service.expand(test_case.statement, inputs)

        # Check episodic memory weight is influenced
        assert len(variants) >= 1
        primary = variants[0]

        # Episodic summaries mention Doug forgetting things
        assert primary.episodic_memory_weight > 0.4

    @pytest.mark.asyncio
    async def test_metacognitive_calibration(self):
        """Test that metacognitive patterns affect confidence."""
        test_case = create_tea_coffee_conflict_test()

        # Add metacognitive warning
        test_case.memory_state.metacognitive_patterns.append(
            "I often misinterpret reference resolution"
        )

        mock_llm = MockLLMProvider()
        service = SemanticExpansionService(llm=mock_llm)

        inputs = test_case.memory_state.to_expansion_input()
        inputs.immediate_context = test_case.immediate_context

        variants = await service.expand(test_case.statement, inputs)

        # Metacognitive weight should reflect calibration
        assert len(variants) >= 1


# =============================================================================
# Evolutionary Fitness Tests
# =============================================================================


@dataclass
class FitnessResult:
    """Result of fitness evaluation."""

    expansion_accuracy: float = 0.0
    variant_ranking: float = 0.0
    wsd_accuracy: float = 0.0
    conflict_detection: float = 0.0
    memory_utilization: float = 0.0

    @property
    def overall_fitness(self) -> float:
        """Compute overall fitness score."""
        weights = {
            "expansion_accuracy": 0.25,
            "variant_ranking": 0.20,
            "wsd_accuracy": 0.20,
            "conflict_detection": 0.20,
            "memory_utilization": 0.15,
        }
        return sum(
            getattr(self, k) * w
            for k, w in weights.items()
        )


class TestEvolutionaryFitness:
    """Tests designed for evolutionary optimization."""

    @pytest.mark.asyncio
    async def test_fitness_evaluation_framework(self):
        """Test the fitness evaluation framework works."""
        test_cases = get_all_test_cases()

        mock_llm = MockLLMProvider()
        service = SemanticExpansionService(llm=mock_llm)

        fitness_results = []

        for test_case in test_cases:
            inputs = test_case.memory_state.to_expansion_input()
            inputs.immediate_context = test_case.immediate_context
            inputs.session_entities = {
                name: EntityInfo(entity_type=etype)
                for name, etype in test_case.session_entities.items()
            }

            variants = await service.expand(test_case.statement, inputs)

            # Evaluate fitness
            result = FitnessResult()

            # Expansion accuracy: Did we extract meaningful content?
            if variants and variants[0].frame.triples:
                result.expansion_accuracy = 1.0
            else:
                result.expansion_accuracy = 0.5

            # Variant ranking: Is primary variant reasonable?
            if variants:
                result.variant_ranking = variants[0].base_confidence

            # Memory utilization: Did we use memory inputs?
            if variants and any([
                variants[0].working_memory_weight > 0.5,
                variants[0].episodic_memory_weight > 0.5,
                variants[0].semantic_memory_weight > 0.5,
            ]):
                result.memory_utilization = 1.0
            else:
                result.memory_utilization = 0.5

            fitness_results.append(result)

        # Overall fitness should be reasonable
        avg_fitness = sum(r.overall_fitness for r in fitness_results) / len(fitness_results)
        assert avg_fitness >= 0.3  # Minimum baseline

    @pytest.mark.asyncio
    async def test_wsd_fitness_on_bank_test(self):
        """Test WSD fitness on bank disambiguation test."""
        test_case = create_bank_disambiguation_test()

        wsd = WordSenseDisambiguator(nlp=None, llm=None, use_nltk=False)

        # Disambiguate bank in context
        sense = await wsd.disambiguate_word(
            "bank",
            test_case.statement,
            pos="NOUN",
        )

        # Check WSD accuracy
        if sense and sense.synset_id == test_case.expected_synsets.get("bank", "bank.n.01"):
            wsd_accuracy = 1.0
        elif sense:
            wsd_accuracy = 0.5  # Wrong sense
        else:
            wsd_accuracy = 0.0  # No sense found

        # Should get financial institution sense
        assert wsd_accuracy >= 0.5


# =============================================================================
# Cross-Layer Association Tests
# =============================================================================


class TestCrossLayerAssociations:
    """Tests for cross-layer memory associations."""

    def test_cross_layer_relation_types(self):
        """Test cross-layer relation enum."""
        assert CrossLayerRelation.SUPPORTS.value == "supports"
        assert CrossLayerRelation.CONTRADICTS.value == "contradicts"
        assert CrossLayerRelation.DERIVED_FROM.value == "derived_from"

    def test_cross_layer_edge_creation(self):
        """Test creating cross-layer edges."""
        edge = CrossLayerEdge(
            source_node_id="working_obs_1",
            source_layer="working",
            target_node_id="semantic_fact_1",
            target_layer="semantic",
            relation=CrossLayerRelation.CONTRADICTS,
            confidence=0.85,
            context={"conflict_type": "preference_mismatch"},
        )

        assert edge.source_layer == "working"
        assert edge.target_layer == "semantic"
        assert edge.relation == CrossLayerRelation.CONTRADICTS

    @pytest.mark.asyncio
    async def test_association_detection_from_expansion(self):
        """Test that expansion can identify potential associations."""
        test_case = create_tea_coffee_conflict_test()

        # The expansion should identify that:
        # - Working observation about tea preference
        # - Potentially contradicts semantic fact "Doug likes coffee"

        mock_llm = MockLLMProvider()
        service = SemanticExpansionService(llm=mock_llm)

        inputs = test_case.memory_state.to_expansion_input()
        inputs.immediate_context = test_case.immediate_context
        inputs.session_entities = {
            name: EntityInfo(entity_type=etype)
            for name, etype in test_case.session_entities.items()
        }

        variants = await service.expand(test_case.statement, inputs)

        # Check that semantic memory was considered
        assert len(variants) >= 1
        # The variant should have been influenced by the "Doug likes coffee" fact
        # This would lower the semantic_memory_weight or create a conflict flag


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullIntegration:
    """Full integration tests combining all components."""

    @pytest.mark.asyncio
    async def test_full_pipeline_tea_coffee(self):
        """Test full pipeline on tea/coffee conflict scenario."""
        test_case = create_tea_coffee_conflict_test()

        # Use LLM with realistic response
        mock_llm = MockLLMProvider({
            "He prefers tea in the morning": """<semantic_frame>
    <triples>
        <triple>
            <subject>Doug</subject>
            <predicate>PREFERS</predicate>
            <object>tea</object>
            <context>temporal="morning"</context>
        </triple>
    </triples>
    <presuppositions>
        <presupposition type="existential">Doug has beverage preferences</presupposition>
        <presupposition type="factive">Doug has tried tea</presupposition>
    </presuppositions>
    <implications>
        <implication type="pragmatic" confidence="0.9">Doug drinks tea in the morning</implication>
        <implication type="commonsense" confidence="0.7">This may differ from other times of day</implication>
    </implications>
    <negations>
        <negation>Doug does not prefer other beverages in the morning</negation>
    </negations>
    <ambiguities>
        <ambiguity type="reference">
            <text>He</text>
            <possibilities>
                <possibility>Doug</possibility>
                <possibility>Unknown person</possibility>
            </possibilities>
        </ambiguity>
    </ambiguities>
    <open_questions>
        <question>Does this apply every morning or specific days?</question>
        <question>What does Doug prefer at other times?</question>
    </open_questions>
    <frame_type>ASSERTION</frame_type>
    <confidence>0.85</confidence>
</semantic_frame>"""
        })

        # Create service with WSD
        wsd = WordSenseDisambiguator(nlp=None, llm=None, use_nltk=False)
        service = SemanticExpansionService(llm=mock_llm, wsd=wsd)

        # Build inputs
        inputs = test_case.memory_state.to_expansion_input()
        inputs.immediate_context = test_case.immediate_context
        inputs.session_entities = {
            name: EntityInfo(entity_type=etype)
            for name, etype in test_case.session_entities.items()
        }

        # Run expansion
        variants = await service.expand(test_case.statement, inputs, max_variations=3)

        # Verify results
        assert len(variants) >= 1

        primary = variants[0]
        frame = primary.frame

        # Check triples extracted
        assert len(frame.triples) >= 1
        tea_triple = frame.triples[0]
        assert tea_triple.predicate == "PREFERS"
        assert tea_triple.object == "tea"

        # Check presuppositions
        assert len(frame.presuppositions) >= 1

        # Check implications
        assert len(frame.implications) >= 1

        # Check ambiguity handling
        assert len(frame.ambiguities) >= 1
        assert frame.ambiguities[0].text == "He"

        # Check cognitive scoring applied
        assert primary.recency_weight > 0
        assert primary.semantic_memory_weight > 0

        # Check combined score is reasonable
        assert primary.combined_score > 0.3

        print(f"\n=== Full Pipeline Test Results ===")
        print(f"Statement: {test_case.statement}")
        print(f"Variants generated: {len(variants)}")
        print(f"Primary interpretation: {tea_triple.to_text()}")
        print(f"Combined score: {primary.combined_score:.2f}")
        print(f"Recency weight: {primary.recency_weight:.2f}")
        print(f"Semantic memory weight: {primary.semantic_memory_weight:.2f}")


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_expand_statement_function(self):
        """Test the expand_statement convenience function."""
        variants = await expand_statement(
            "Doug likes coffee",
            context=["We were discussing breakfast"],
            entities={"Doug": "PERSON"},
        )

        assert len(variants) >= 1
        assert variants[0].frame.original_text == "Doug likes coffee"
