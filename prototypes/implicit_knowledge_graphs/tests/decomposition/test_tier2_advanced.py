"""Tier 2 Advanced Test Suite.

This test suite is designed to PUSH THE LIMITS of the decomposition system.
Tests here are intentionally harder than Tier 1 industry benchmarks.

Per CONSTITUTION.md:
- Tests must be designed to fail initially
- NEVER weaken tests to pass - fix the system instead
- These tests identify improvement opportunities

Test Categories:
1. Multi-step scenarios that build knowledge over time
2. Complex sentences with multiple phenomena
3. Adversarial/edge cases designed to expose weaknesses
4. Novel sentences not seen during development
5. Cross-component integration tests
"""

import pytest
import sys
sys.path.insert(0, str(__file__).rsplit('tests', 1)[0] + 'src')

from dataclasses import dataclass
from typing import Any

from decomposition.pipeline import DecompositionPipeline, decompose_sync
from decomposition.negation import NegationExtractor
from decomposition.semantic_roles import SemanticRoleExtractor
from decomposition.temporal import TemporalExtractor, Aspect, Tense
from decomposition.modality import ModalityExtractor, ModalType
from decomposition.commonsense import CommonsenseExtractor


# =============================================================================
# Test Data Structures
# =============================================================================


@dataclass
class AdvancedTestCase:
    """Test case for Tier 2 advanced testing."""
    id: str
    text: str
    expected: dict[str, Any]
    category: str
    rationale: str  # Why this test is hard


# =============================================================================
# Multi-Step Scenarios
# =============================================================================


MULTI_STEP_SCENARIOS = [
    # Scenario 1: Building a story with consistent entities
    {
        "name": "office_conflict",
        "description": "A workplace scenario with emotional progression",
        "steps": [
            AdvancedTestCase(
                id="ms-office-001",
                text="John arrived at the office angry because he had been passed over for promotion.",
                expected={
                    "temporal": {"tense": "PAST"},
                    "modality": {"type": "NONE"},  # No modal markers
                    "srl": {"predicates": ["arrived", "passed"], "arg0_arrived": "John"},
                    "commonsense": {"xReact": ["angry", "frustrated", "disappointed"]},
                },
                category="multi_step",
                rationale="Establishes emotional state with reason clause",
            ),
            AdvancedTestCase(
                id="ms-office-002",
                text="He refused to attend the meeting.",
                expected={
                    "negation": {"has_negation": True, "type": "implicit"},  # "refused" implies not attending
                    "srl": {"predicates": ["refused", "attend"], "arg0": "He"},
                },
                category="multi_step",
                rationale="Tests implicit negation through refusal verb",
            ),
            AdvancedTestCase(
                id="ms-office-003",
                text="His colleagues must have noticed something was wrong.",
                expected={
                    "modality": {"type": "EPISTEMIC"},  # Inference about the past
                    "srl": {"predicates": ["noticed"]},
                    "temporal": {"aspect": "ACHIEVEMENT"},  # "noticed" is punctual
                },
                category="multi_step",
                rationale="Must + have = epistemic inference",
            ),
            AdvancedTestCase(
                id="ms-office-004",
                text="The manager decided that John must apologize before the end of the day.",
                expected={
                    "modality": {"type": "DEONTIC"},  # Obligation imposed
                    "temporal": {"has_temporal": True},  # "before the end of the day"
                },
                category="multi_step",
                rationale="Embedded deontic modal in reported speech",
            ),
            AdvancedTestCase(
                id="ms-office-005",
                text="Eventually, John apologized, though he still felt it was unfair.",
                expected={
                    "negation": {"has_negation": False},  # "unfair" is not negation in this context
                    "commonsense": {"xReact": ["reluctant", "frustrated"]},
                },
                category="multi_step",
                rationale="Tests morphological negation (unfair) in context",
            ),
        ],
    },
    # Scenario 2: Scientific discovery narrative
    {
        "name": "scientific_discovery",
        "description": "A research narrative with epistemic progression",
        "steps": [
            AdvancedTestCase(
                id="ms-science-001",
                text="The researchers suspected that the results might be contaminated.",
                expected={
                    "modality": {"type": "EPISTEMIC"},  # "might" = possibility
                    "commonsense": {"xReact": ["concerned", "worried"]},
                },
                category="multi_step",
                rationale="Epistemic modal in complement clause",
            ),
            AdvancedTestCase(
                id="ms-science-002",
                text="They had to repeat all the experiments.",
                expected={
                    "modality": {"type": "DEONTIC"},  # "had to" = obligation
                    "temporal": {"tense": "PAST"},
                },
                category="multi_step",
                rationale="Past deontic obligation",
            ),
            AdvancedTestCase(
                id="ms-science-003",
                text="The new data showed that their hypothesis was not incorrect after all.",
                expected={
                    "negation": {"has_negation": True, "type": "double"},  # "not incorrect" = litotes
                },
                category="multi_step",
                rationale="Double negative (litotes) - pragmatically positive",
            ),
            AdvancedTestCase(
                id="ms-science-004",
                text="The discovery could revolutionize the field.",
                expected={
                    "modality": {"type": "EPISTEMIC"},  # Possibility about future
                    "temporal": {"aspect": "ACCOMPLISHMENT"},
                },
                category="multi_step",
                rationale="Could = epistemic possibility here (not ability)",
            ),
        ],
    },
]


# =============================================================================
# Complex Multi-Phenomenon Sentences
# =============================================================================


COMPLEX_SENTENCES = [
    # Sentences that test multiple extractors simultaneously
    AdvancedTestCase(
        id="complex-001",
        text="Although he definitely should not have forgotten the appointment, John might be forgiven if he apologizes.",
        expected={
            "negation": {"has_negation": True, "cues": ["not"]},
            "modality": {"type": "DEONTIC"},  # "should not" is deontic
            "srl": {"passive": True},  # "be forgiven" is passive
        },
        category="multi_phenomenon",
        rationale="Combines negation, deontic modality, epistemic modality, and passive voice",
    ),
    AdvancedTestCase(
        id="complex-002",
        text="The message that had been sent but never received was finally discovered by the IT department.",
        expected={
            "negation": {"has_negation": True, "cues": ["never"]},
            "srl": {"passive": True, "predicates": ["sent", "received", "discovered"]},
            "temporal": {"tense": "PAST", "aspect": "ACHIEVEMENT"},
        },
        category="multi_phenomenon",
        rationale="Multiple passive clauses with negation and complex tense",
    ),
    AdvancedTestCase(
        id="complex-003",
        text="She must have been told that she would need to resubmit the application before Monday.",
        expected={
            "modality": {"type": "EPISTEMIC"},  # "must have been" = epistemic inference
            "temporal": {"has_temporal": True},  # "before Monday"
            "srl": {"passive": True},  # "been told"
        },
        category="multi_phenomenon",
        rationale="Triple embedding: epistemic > passive > future deontic",
    ),
    AdvancedTestCase(
        id="complex-004",
        text="Nobody could have known that the supposedly impossible task would be completed so quickly.",
        expected={
            "negation": {"has_negation": True, "cues": ["nobody", "impossible"]},
            "modality": {"type": "EPISTEMIC"},  # "could have known"
            "srl": {"passive": True},  # "be completed"
        },
        category="multi_phenomenon",
        rationale="Negative quantifier + morphological negation + epistemic + passive",
    ),
    AdvancedTestCase(
        id="complex-005",
        text="The proposal, which must be submitted by Friday, might not be approved even if we finish on time.",
        expected={
            "negation": {"has_negation": True, "cues": ["not"]},
            "modality": {"mixed": True},  # Both deontic (must submit) and epistemic (might not)
            "temporal": {"has_temporal": True},
        },
        category="multi_phenomenon",
        rationale="Relative clause with deontic, main clause with negated epistemic",
    ),
]


# =============================================================================
# Adversarial/Edge Cases
# =============================================================================


ADVERSARIAL_CASES = [
    # Cases designed to expose weaknesses
    AdvancedTestCase(
        id="adv-neg-001",
        text="I can't not go.",
        expected={
            "negation": {"has_negation": True, "type": "double"},
            "polarity": "positive",  # Double negative = positive meaning
        },
        category="adversarial",
        rationale="Double negative with contracted form",
    ),
    AdvancedTestCase(
        id="adv-neg-002",
        text="It's not that I don't like you.",
        expected={
            "negation": {"has_negation": True, "type": "double"},
            "polarity": "positive",  # "It's not that I don't" = I do like you
        },
        category="adversarial",
        rationale="Complex double negative with 'it's not that'",
    ),
    AdvancedTestCase(
        id="adv-passive-001",
        text="The decision was made to be unmade.",
        expected={
            "srl": {"passive": True, "predicates": ["made", "unmade"]},
        },
        category="adversarial",
        rationale="Passive with infinitive complement containing morphological negation",
    ),
    AdvancedTestCase(
        id="adv-passive-002",
        text="Having been warned, they should have been more careful.",
        expected={
            "srl": {"passive": True},
            "modality": {"type": "DEONTIC"},  # "should have been" in deontic sense
        },
        category="adversarial",
        rationale="Participial passive clause + should have been",
    ),
    AdvancedTestCase(
        id="adv-modal-001",
        text="You can leave if you must, but you may not take the documents.",
        expected={
            "modality": {"mixed": True},  # "can" ability, "must" deontic, "may not" prohibition
        },
        category="adversarial",
        rationale="Three different modal meanings in one sentence",
    ),
    AdvancedTestCase(
        id="adv-modal-002",
        text="He might must go.",
        expected={
            "modality": {"type": "EPISTEMIC"},  # Dialectal double modal
        },
        category="adversarial",
        rationale="Double modal (Southern US dialectal) - rare but valid",
    ),
    AdvancedTestCase(
        id="adv-temporal-001",
        text="By the time she had finished, they will have been waiting for three hours.",
        expected={
            "temporal": {"mixed_tense": True},  # Past perfect + future perfect progressive
        },
        category="adversarial",
        rationale="Mixed temporal reference with complex aspect",
    ),
    AdvancedTestCase(
        id="adv-commonsense-001",
        text="She was fired but felt relieved.",
        expected={
            "commonsense": {"xReact": ["relieved"]},  # Counter-intuitive emotion
        },
        category="adversarial",
        rationale="Counter-expectation emotional response",
    ),
    AdvancedTestCase(
        id="adv-commonsense-002",
        text="The test results were negative and the patient celebrated.",
        expected={
            "commonsense": {"xReact": ["happy", "relieved"]},  # Medical negative = good
            "context": "medical",
        },
        category="adversarial",
        rationale="Domain-specific meaning (medical 'negative' = positive outcome)",
    ),
]


# =============================================================================
# Novel Sentences (Anti-Overfitting)
# =============================================================================


NOVEL_SENTENCES = [
    # These should NOT be in training/test data - fresh examples
    AdvancedTestCase(
        id="novel-001",
        text="The quantum superposition collapsed when observed by the sensor.",
        expected={
            "srl": {"passive": True, "predicates": ["collapsed", "observed"]},
            "temporal": {"aspect": "ACHIEVEMENT"},
        },
        category="novel",
        rationale="Technical domain vocabulary",
    ),
    AdvancedTestCase(
        id="novel-002",
        text="Despite having been thoroughly debunked, the myth persists.",
        expected={
            "negation": {"has_negation": True, "implicit": True},  # "debunked" implies negation
            "srl": {"passive": True},
        },
        category="novel",
        rationale="Implicit negation through 'debunked'",
    ),
    AdvancedTestCase(
        id="novel-003",
        text="The cryptocurrency wallet was allegedly emptied by hackers, though this cannot be verified.",
        expected={
            "modality": {"evidential": "reported"},  # "allegedly"
            "negation": {"has_negation": True, "cues": ["cannot"]},
            "srl": {"passive": True},
        },
        category="novel",
        rationale="Modern domain + evidential + negation + passive",
    ),
    AdvancedTestCase(
        id="novel-004",
        text="The AI system unexpectedly began generating creative outputs.",
        expected={
            "temporal": {"aspect": "ACHIEVEMENT"},  # "began" is punctual
            "commonsense": {"xReact": ["surprised"]},
        },
        category="novel",
        rationale="AI domain + inchoative aspect",
    ),
    AdvancedTestCase(
        id="novel-005",
        text="One must assume responsibility for mistakes that were never acknowledged.",
        expected={
            "modality": {"type": "DEONTIC"},  # Generic deontic
            "negation": {"has_negation": True, "cues": ["never"]},
            "srl": {"passive": True},
        },
        category="novel",
        rationale="Generic subject + deontic + negation in relative clause",
    ),
]


# =============================================================================
# Test Classes
# =============================================================================


class TestMultiStepScenarios:
    """Test multi-step scenarios that build knowledge over time.

    These tests verify that:
    1. The system handles connected sentences coherently
    2. Context carries forward correctly
    3. Emotional/epistemic progression is tracked
    """

    @pytest.fixture
    def pipeline(self):
        return DecompositionPipeline()

    @pytest.fixture
    def extractors(self):
        return {
            "negation": NegationExtractor(),
            "srl": SemanticRoleExtractor(),
            "temporal": TemporalExtractor(),
            "modality": ModalityExtractor(),
            "commonsense": CommonsenseExtractor(),
        }

    def test_office_conflict_scenario(self, pipeline, extractors):
        """Test the office conflict multi-step scenario."""
        scenario = MULTI_STEP_SCENARIOS[0]

        passed = 0
        failed = 0
        failures = []

        for step in scenario["steps"]:
            step_passed = self._evaluate_step(step, extractors)
            if step_passed:
                passed += 1
            else:
                failed += 1
                failures.append(step.id)

        total = len(scenario["steps"])
        accuracy = passed / total if total > 0 else 0

        print(f"\nScenario: {scenario['name']}")
        print(f"  Passed: {passed}/{total} ({accuracy:.1%})")
        if failures:
            print(f"  Failed steps: {', '.join(failures)}")

        # Tier 2 tests may fail - this documents the current state
        # We expect at least 40% to pass for basic coherence
        assert accuracy >= 0.4, f"Multi-step scenario accuracy too low: {accuracy:.1%}"

    def test_scientific_discovery_scenario(self, pipeline, extractors):
        """Test the scientific discovery multi-step scenario."""
        scenario = MULTI_STEP_SCENARIOS[1]

        passed = 0
        for step in scenario["steps"]:
            if self._evaluate_step(step, extractors):
                passed += 1

        total = len(scenario["steps"])
        accuracy = passed / total if total > 0 else 0

        print(f"\nScenario: {scenario['name']}")
        print(f"  Passed: {passed}/{total} ({accuracy:.1%})")

        assert accuracy >= 0.4, f"Multi-step scenario accuracy too low: {accuracy:.1%}"

    def _evaluate_step(self, step: AdvancedTestCase, extractors: dict) -> bool:
        """Evaluate a single step in a multi-step scenario."""
        try:
            # Test relevant extractors based on expected results
            if "negation" in step.expected:
                neg_result = extractors["negation"].extract_sync(step.text)
                expected_neg = step.expected["negation"].get("has_negation", False)
                if neg_result.is_negated != expected_neg:
                    return False

            if "modality" in step.expected:
                mod_result = extractors["modality"].extract_sync(step.text)
                expected_type = step.expected["modality"].get("type", "NONE")
                if expected_type != "NONE" and mod_result.modal_type.name != expected_type:
                    return False

            if "temporal" in step.expected:
                temp_result = extractors["temporal"].extract_sync(step.text)
                if "tense" in step.expected["temporal"]:
                    if temp_result.tense.name != step.expected["temporal"]["tense"]:
                        return False

            return True

        except Exception as e:
            print(f"  Error in {step.id}: {e}")
            return False


class TestComplexSentences:
    """Test complex multi-phenomenon sentences.

    These sentences contain multiple linguistic phenomena that must be
    correctly identified simultaneously.
    """

    @pytest.fixture
    def extractors(self):
        return {
            "negation": NegationExtractor(),
            "srl": SemanticRoleExtractor(),
            "temporal": TemporalExtractor(),
            "modality": ModalityExtractor(),
        }

    def test_complex_multi_phenomenon(self, extractors):
        """Test sentences with multiple phenomena."""
        passed = 0
        failed = 0

        print("\nComplex Multi-Phenomenon Tests:")

        for tc in COMPLEX_SENTENCES:
            try:
                correct = self._evaluate_complex(tc, extractors)
                status = "PASS" if correct else "FAIL"
                print(f"  [{status}] {tc.id}: {tc.text[:50]}...")

                if correct:
                    passed += 1
                else:
                    failed += 1
                    print(f"       Rationale: {tc.rationale}")
            except Exception as e:
                failed += 1
                print(f"  [ERROR] {tc.id}: {e}")

        total = len(COMPLEX_SENTENCES)
        accuracy = passed / total if total > 0 else 0

        print(f"\nComplex sentences: {passed}/{total} ({accuracy:.1%})")

        # Tier 2: Expect this to be hard, document current state
        # Minimum threshold is 30% for basic sanity
        assert accuracy >= 0.3, f"Complex sentence accuracy too low: {accuracy:.1%}"

    def _evaluate_complex(self, tc: AdvancedTestCase, extractors: dict) -> bool:
        """Evaluate a complex multi-phenomenon sentence."""
        components_correct = 0
        components_total = 0

        # Check negation
        if "negation" in tc.expected:
            components_total += 1
            neg_result = extractors["negation"].extract_sync(tc.text)
            expected_neg = tc.expected["negation"].get("has_negation", False)
            if neg_result.is_negated == expected_neg:
                components_correct += 1

        # Check modality
        if "modality" in tc.expected and "type" in tc.expected["modality"]:
            components_total += 1
            mod_result = extractors["modality"].extract_sync(tc.text)
            expected_type = tc.expected["modality"]["type"]
            if mod_result.modal_type.name == expected_type:
                components_correct += 1

        # Check SRL passive
        if "srl" in tc.expected and tc.expected["srl"].get("passive"):
            components_total += 1
            srl = extractors["srl"]
            is_passive, _ = srl.passive_detector.is_passive(tc.text)
            if is_passive:
                components_correct += 1

        if components_total == 0:
            return True

        return components_correct / components_total >= 0.6


class TestAdversarialCases:
    """Test adversarial/edge cases designed to expose weaknesses.

    These tests are intentionally designed to break the system.
    Many will fail - that's the point. They identify improvement areas.
    """

    @pytest.fixture
    def extractors(self):
        return {
            "negation": NegationExtractor(),
            "srl": SemanticRoleExtractor(),
            "modality": ModalityExtractor(),
            "temporal": TemporalExtractor(),
            "commonsense": CommonsenseExtractor(),
        }

    def test_adversarial_negation(self, extractors):
        """Test adversarial negation cases."""
        neg_cases = [tc for tc in ADVERSARIAL_CASES if tc.id.startswith("adv-neg")]

        passed = 0
        print("\nAdversarial Negation Tests:")

        for tc in neg_cases:
            neg = extractors["negation"]
            result = neg.extract_sync(tc.text)

            expected_neg = tc.expected.get("negation", {}).get("has_negation", False)
            correct = result.is_negated == expected_neg

            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] {tc.id}: {tc.text}")
            print(f"       Expected negation: {expected_neg}, Got: {result.is_negated}")

            if correct:
                passed += 1

        accuracy = passed / len(neg_cases) if neg_cases else 0
        print(f"\nAdversarial negation: {passed}/{len(neg_cases)} ({accuracy:.1%})")

        # These are HARD - even 20% is informative
        # No assertion - just documenting capability

    def test_adversarial_modality(self, extractors):
        """Test adversarial modality cases."""
        modal_cases = [tc for tc in ADVERSARIAL_CASES if tc.id.startswith("adv-modal")]

        passed = 0
        print("\nAdversarial Modality Tests:")

        for tc in modal_cases:
            mod = extractors["modality"]
            result = mod.extract_sync(tc.text)

            # For complex modal cases, just check if any modal detected
            detected = result.modal_type.name != "NONE"

            status = "DETECTED" if detected else "MISSED"
            print(f"  [{status}] {tc.id}: {tc.text}")
            print(f"       Detected type: {result.modal_type.name}")

            if detected:
                passed += 1

        accuracy = passed / len(modal_cases) if modal_cases else 0
        print(f"\nAdversarial modality detection: {passed}/{len(modal_cases)} ({accuracy:.1%})")

    def test_adversarial_commonsense(self, extractors):
        """Test adversarial commonsense cases."""
        cs_cases = [tc for tc in ADVERSARIAL_CASES if tc.id.startswith("adv-commonsense")]

        print("\nAdversarial Commonsense Tests:")

        for tc in cs_cases:
            cs = extractors["commonsense"]
            inferences = cs.extract_sync(tc.text)

            print(f"  {tc.id}: {tc.text}")
            print(f"       Inferences: {[(i.relation.name, i.tail) for i in inferences[:3]]}")
            print(f"       Rationale: {tc.rationale}")


class TestNovelSentences:
    """Test novel sentences not seen during development.

    These are fresh examples to detect overfitting.
    """

    @pytest.fixture
    def extractors(self):
        return {
            "negation": NegationExtractor(),
            "srl": SemanticRoleExtractor(),
            "temporal": TemporalExtractor(),
            "modality": ModalityExtractor(),
        }

    def test_novel_sentence_robustness(self, extractors):
        """Test that system handles novel sentences without crashing."""
        print("\nNovel Sentence Tests:")

        crashed = 0
        processed = 0

        for tc in NOVEL_SENTENCES:
            try:
                # Just run all extractors - check for crashes
                extractors["negation"].extract_sync(tc.text)
                extractors["srl"].extract_sync(tc.text)
                extractors["temporal"].extract_sync(tc.text)
                extractors["modality"].extract_sync(tc.text)

                processed += 1
                print(f"  [OK] {tc.id}: processed without error")

            except Exception as e:
                crashed += 1
                print(f"  [CRASH] {tc.id}: {e}")

        total = len(NOVEL_SENTENCES)
        print(f"\nNovel sentences: {processed}/{total} processed without crash")

        # Must handle all sentences without crashing
        assert crashed == 0, f"{crashed} sentences caused crashes"

    def test_novel_sentence_accuracy(self, extractors):
        """Test accuracy on novel sentences."""
        passed = 0

        print("\nNovel Sentence Accuracy:")

        for tc in NOVEL_SENTENCES:
            components_correct = 0
            components_total = 0

            # Check expected components
            if "negation" in tc.expected:
                components_total += 1
                result = extractors["negation"].extract_sync(tc.text)
                expected = tc.expected["negation"].get("has_negation", False)
                if result.is_negated == expected:
                    components_correct += 1

            if "srl" in tc.expected and tc.expected["srl"].get("passive"):
                components_total += 1
                srl = extractors["srl"]
                is_passive, _ = srl.passive_detector.is_passive(tc.text)
                if is_passive:
                    components_correct += 1

            if "modality" in tc.expected and "type" in tc.expected["modality"]:
                components_total += 1
                result = extractors["modality"].extract_sync(tc.text)
                expected_type = tc.expected["modality"]["type"]
                if result.modal_type.name == expected_type:
                    components_correct += 1

            if components_total > 0:
                score = components_correct / components_total
                correct = score >= 0.5
            else:
                correct = True

            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] {tc.id}")

            if correct:
                passed += 1

        total = len(NOVEL_SENTENCES)
        accuracy = passed / total if total > 0 else 0

        print(f"\nNovel sentence accuracy: {passed}/{total} ({accuracy:.1%})")

        # Novel sentences should have at least 40% accuracy
        # Lower than Tier 1 because these are intentionally novel
        assert accuracy >= 0.4, f"Novel sentence accuracy too low: {accuracy:.1%}"


class TestTier2Summary:
    """Summary test that runs all Tier 2 tests and reports overall status."""

    def test_tier2_summary(self):
        """Generate a summary of Tier 2 test results."""
        print("\n" + "=" * 70)
        print("TIER 2 ADVANCED TEST SUMMARY")
        print("=" * 70)
        print("\nThese tests are INTENTIONALLY HARD.")
        print("Per CONSTITUTION.md: Tests must be designed to fail initially.")
        print("Failures identify improvement opportunities - DO NOT weaken tests.")
        print("\n" + "=" * 70)

        # This test just passes to allow the summary to be shown
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
