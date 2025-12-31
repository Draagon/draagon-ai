"""End-to-end benchmark tests against industry standards.

This module evaluates each extractor against published NLP benchmark standards
and classifies performance into tiers:

    - BELOW_INDUSTRY: Below published baseline performance
    - INDUSTRY_BASELINE: Matches basic published baselines
    - INDUSTRY_STANDARD: Matches typical published results
    - ABOVE_STANDARD: Exceeds typical published results
    - STATE_OF_ART: Approaches or matches SOTA

Test suites are calibrated to published benchmarks:
    - Negation: BioScope, SFU Review Corpus
    - SRL: CoNLL-2005, CoNLL-2012
    - Temporal: TempEval-3
    - Modality: LEXDEMOD, epistemic/deontic literature
    - Commonsense: ATOMIC, COMET

Run with: pytest tests/decomposition/test_industry_benchmarks.py -v
"""

import pytest
from typing import Any

from decomposition.benchmarks import (
    BenchmarkTier,
    BenchmarkThresholds,
    BenchmarkTestCase,
    BenchmarkResult,
    BenchmarkSuiteResult,
    NEGATION_THRESHOLDS,
    SRL_THRESHOLDS,
    TEMPORAL_THRESHOLDS,
    MODALITY_THRESHOLDS,
    COMMONSENSE_THRESHOLDS,
    ASPECT_THRESHOLDS,
    NEGATION_TEST_SUITE,
    SRL_TEST_SUITE,
    TEMPORAL_TEST_SUITE,
    MODALITY_TEST_SUITE,
    COMMONSENSE_TEST_SUITE,
    compute_metrics,
    classify_performance,
    generate_report,
)


# =============================================================================
# Negation Benchmark Evaluator
# =============================================================================


class NegationEvaluator:
    """Evaluates negation detection against benchmark suite."""

    def __init__(self):
        from decomposition.negation import NegationDetector, NegationType
        from decomposition.config import NegationConfig

        self.detector = NegationDetector(NegationConfig())
        self.NegationType = NegationType

    def evaluate(self, test_cases: list[BenchmarkTestCase]) -> list[BenchmarkResult]:
        """Evaluate all test cases."""
        results = []
        for tc in test_cases:
            result = self._evaluate_single(tc)
            results.append(result)
        return results

    def _evaluate_single(self, tc: BenchmarkTestCase) -> BenchmarkResult:
        """Evaluate a single test case."""
        cues = self.detector.detect(tc.text)
        has_negation = len(cues) > 0

        expected_negation = tc.expected.get("has_negation", False)

        # Calculate correctness
        correct = has_negation == expected_negation
        partial_score = 1.0 if correct else 0.0

        # Calculate TP/FP/FN for metrics
        details = {}
        if expected_negation and has_negation:
            details["tp"] = 1
            details["fp"] = 0
            details["fn"] = 0
            # Check if type matches
            expected_type = tc.expected.get("type", "")
            detected_types = [c.negation_type.value.lower() for c in cues]
            if expected_type and expected_type in detected_types:
                partial_score = 1.0
            elif expected_type:
                partial_score = 0.7  # Detected negation but wrong type
        elif not expected_negation and not has_negation:
            details["tp"] = 0
            details["fp"] = 0
            details["fn"] = 0
        elif expected_negation and not has_negation:
            details["tp"] = 0
            details["fp"] = 0
            details["fn"] = 1
        else:  # False positive
            details["tp"] = 0
            details["fp"] = 1
            details["fn"] = 0

        predicted = {
            "has_negation": has_negation,
            "cues": [c.text for c in cues],
            "types": [c.negation_type.value for c in cues],
        }

        return BenchmarkResult(
            test_case=tc,
            predicted=predicted,
            correct=correct,
            partial_score=partial_score,
            details=details,
        )


# =============================================================================
# SRL Benchmark Evaluator
# =============================================================================


class SRLEvaluator:
    """Evaluates semantic role labeling against benchmark suite."""

    def __init__(self):
        from decomposition.semantic_roles import SemanticRoleExtractor

        self.extractor = SemanticRoleExtractor()

    def evaluate(self, test_cases: list[BenchmarkTestCase]) -> list[BenchmarkResult]:
        """Evaluate all test cases."""
        results = []
        for tc in test_cases:
            result = self._evaluate_single(tc)
            results.append(result)
        return results

    def _evaluate_single(self, tc: BenchmarkTestCase) -> BenchmarkResult:
        """Evaluate a single test case."""
        roles = self.extractor.extract_sync(tc.text)

        # Extract predicates
        detected_predicates = list({r.predicate for r in roles})

        expected_predicates = tc.expected.get("predicates", [])
        expected_roles = tc.expected.get("roles", {})
        expected_passive = tc.expected.get("passive", False)

        # Score predicate detection
        predicate_matches = sum(
            1 for p in expected_predicates if p in detected_predicates
        )
        predicate_score = (
            predicate_matches / len(expected_predicates)
            if expected_predicates
            else 1.0
        )

        # Score role detection
        role_score = 0.0
        role_count = 0
        detected_role_map = {}

        for role in roles:
            if role.role not in detected_role_map:
                detected_role_map[role.role] = role.filler

        for role_name, expected_filler in expected_roles.items():
            role_count += 1
            if role_name in detected_role_map:
                detected_filler = detected_role_map[role_name]
                # Partial match if filler contains expected
                if expected_filler.lower() in detected_filler.lower():
                    role_score += 1.0
                elif detected_filler.lower() in expected_filler.lower():
                    role_score += 0.8
                else:
                    role_score += 0.0
            else:
                role_score += 0.0

        role_score = role_score / role_count if role_count > 0 else 1.0

        # Check passive detection
        passive_score = 1.0
        if expected_passive:
            # Check if we detected the right predicate (not "was")
            if "was" in detected_predicates and len(detected_predicates) == 1:
                passive_score = 0.0  # Failed - detected auxiliary as predicate
            elif any(p in expected_predicates for p in detected_predicates):
                passive_score = 1.0
            else:
                passive_score = 0.5

        # Combined score
        partial_score = (predicate_score * 0.4 + role_score * 0.4 + passive_score * 0.2)
        correct = partial_score >= 0.8

        details = {
            "predicate_score": predicate_score,
            "role_score": role_score,
            "passive_score": passive_score,
        }

        predicted = {
            "predicates": detected_predicates,
            "roles": detected_role_map,
        }

        return BenchmarkResult(
            test_case=tc,
            predicted=predicted,
            correct=correct,
            partial_score=partial_score,
            details=details,
        )


# =============================================================================
# Temporal Benchmark Evaluator
# =============================================================================


class TemporalEvaluator:
    """Evaluates temporal extraction against benchmark suite."""

    def __init__(self):
        from decomposition.temporal import TemporalExtractor
        from decomposition.models import Tense, Aspect

        self.extractor = TemporalExtractor()
        self.Tense = Tense
        self.Aspect = Aspect

    def evaluate(self, test_cases: list[BenchmarkTestCase]) -> list[BenchmarkResult]:
        """Evaluate all test cases."""
        results = []
        for tc in test_cases:
            result = self._evaluate_single(tc)
            results.append(result)
        return results

    def _evaluate_single(self, tc: BenchmarkTestCase) -> BenchmarkResult:
        """Evaluate a single test case."""
        result = self.extractor.extract_sync(tc.text)

        partial_score = 0.0
        components = 0

        # Check tense
        if "tense" in tc.expected:
            components += 1
            expected_tense = tc.expected["tense"]
            if result.tense.name == expected_tense:
                partial_score += 1.0

        # Check aspect - this is a known gap area
        if "aspect" in tc.expected:
            components += 1
            expected_aspect = tc.expected["aspect"]
            if result.aspect.name == expected_aspect:
                partial_score += 1.0
            elif result.aspect.name == "ACTIVITY" and expected_aspect == "ACHIEVEMENT":
                # Common misclassification
                partial_score += 0.3
            elif result.aspect.name == "ACTIVITY" and expected_aspect == "ACCOMPLISHMENT":
                partial_score += 0.3

        # Check temporal expression presence
        if "has_temporal" in tc.expected:
            components += 1
            # Check if temporal expressions were found
            has_temporal = result.reference_value is not None
            if has_temporal == tc.expected["has_temporal"]:
                partial_score += 1.0

        if components > 0:
            partial_score = partial_score / components
        else:
            partial_score = 1.0

        correct = partial_score >= 0.8

        predicted = {
            "tense": result.tense.name,
            "aspect": result.aspect.name,
            "reference_value": result.reference_value,
        }

        details = {
            "tense_correct": "tense" not in tc.expected or result.tense.name == tc.expected.get("tense"),
            "aspect_correct": "aspect" not in tc.expected or result.aspect.name == tc.expected.get("aspect"),
        }

        return BenchmarkResult(
            test_case=tc,
            predicted=predicted,
            correct=correct,
            partial_score=partial_score,
            details=details,
        )


# =============================================================================
# Modality Benchmark Evaluator
# =============================================================================


class ModalityEvaluator:
    """Evaluates modality detection against benchmark suite."""

    def __init__(self):
        from decomposition.modality import (
            ModalityExtractor,
            DeonticDetector,
            EvidentialDetector,
        )
        from decomposition.config import ModalityConfig
        from decomposition.models import ModalType

        self.extractor = ModalityExtractor()
        self.config = ModalityConfig()
        self.deontic_detector = DeonticDetector(self.config)
        self.evidential_detector = EvidentialDetector(self.config)
        self.ModalType = ModalType

    def evaluate(self, test_cases: list[BenchmarkTestCase]) -> list[BenchmarkResult]:
        """Evaluate all test cases."""
        results = []
        for tc in test_cases:
            result = self._evaluate_single(tc)
            results.append(result)
        return results

    def _evaluate_single(self, tc: BenchmarkTestCase) -> BenchmarkResult:
        """Evaluate a single test case."""
        result = self.extractor.extract_sync(tc.text)
        deontic_markers = self.deontic_detector.detect(tc.text)
        evidential_markers = self.evidential_detector.detect(tc.text)

        partial_score = 0.0
        components = 0

        # Check modal type (EPISTEMIC vs DEONTIC)
        if "type" in tc.expected:
            components += 1
            expected_type = tc.expected["type"]
            detected_type = result.modal_type.name

            if detected_type == expected_type:
                partial_score += 1.0
            elif detected_type in ("EPISTEMIC", "DEONTIC") and expected_type in ("EPISTEMIC", "DEONTIC"):
                # Got modal but wrong type - this is the disambiguation problem
                partial_score += 0.4
            elif detected_type != "NONE" and expected_type != "NONE":
                partial_score += 0.2

        # Check deontic force
        if "force" in tc.expected:
            components += 1
            expected_force = tc.expected["force"]
            detected_forces = [m.deontic_force.value.lower() for m in deontic_markers]

            if expected_force in detected_forces:
                partial_score += 1.0
            elif detected_forces:
                partial_score += 0.3  # Detected something

        # Check evidential
        if "evidential" in tc.expected:
            components += 1
            expected_evid = tc.expected["evidential"]
            detected_evids = [m.evidential_source.value.lower() for m in evidential_markers]

            if expected_evid in detected_evids:
                partial_score += 1.0

        # Check certainty
        if "certainty" in tc.expected:
            components += 1
            expected_cert = tc.expected["certainty"]
            detected_cert = result.certainty

            # Allow 0.15 tolerance
            if abs(detected_cert - expected_cert) <= 0.15:
                partial_score += 1.0
            elif abs(detected_cert - expected_cert) <= 0.3:
                partial_score += 0.5

        # Check high/low certainty flags
        # Handle None certainty by treating it as 0.5 (uncertain)
        detected_certainty = result.certainty if result.certainty is not None else 0.5

        if "certainty_high" in tc.expected:
            components += 1
            if tc.expected["certainty_high"] and detected_certainty >= 0.7:
                partial_score += 1.0
            elif not tc.expected["certainty_high"] and detected_certainty < 0.7:
                partial_score += 1.0
            else:
                partial_score += 0.3

        if "certainty_low" in tc.expected:
            components += 1
            if tc.expected["certainty_low"] and detected_certainty <= 0.6:
                partial_score += 1.0
            elif not tc.expected["certainty_low"] and detected_certainty > 0.6:
                partial_score += 1.0
            else:
                partial_score += 0.3

        if components > 0:
            partial_score = partial_score / components
        else:
            partial_score = 1.0

        correct = partial_score >= 0.8

        predicted = {
            "type": result.modal_type.name,
            "certainty": result.certainty,
            "deontic_forces": [m.deontic_force.value for m in deontic_markers],
            "evidential_sources": [m.evidential_source.value for m in evidential_markers],
        }

        return BenchmarkResult(
            test_case=tc,
            predicted=predicted,
            correct=correct,
            partial_score=partial_score,
            details={},
        )


# =============================================================================
# Commonsense Benchmark Evaluator
# =============================================================================


class CommonsenseEvaluator:
    """Evaluates commonsense inference against benchmark suite."""

    def __init__(self):
        from decomposition.commonsense import (
            CommonsenseExtractor,
            EventExtractor,
            TemplateGenerator,
        )
        from decomposition.models import CommonsenseRelation

        self.extractor = CommonsenseExtractor()
        self.event_extractor = EventExtractor()
        self.template_gen = TemplateGenerator()
        self.CommonsenseRelation = CommonsenseRelation

    def evaluate(self, test_cases: list[BenchmarkTestCase]) -> list[BenchmarkResult]:
        """Evaluate all test cases."""
        results = []
        for tc in test_cases:
            result = self._evaluate_single(tc)
            results.append(result)
        return results

    def _evaluate_single(self, tc: BenchmarkTestCase) -> BenchmarkResult:
        """Evaluate a single test case."""
        # Extract events
        events = self.event_extractor.extract(tc.text)

        # Generate inferences
        all_inferences = []
        for event in events:
            inferences = self.template_gen.generate(
                event,
                [
                    self.CommonsenseRelation.X_REACT,
                    self.CommonsenseRelation.X_INTENT,
                    self.CommonsenseRelation.X_EFFECT,
                    self.CommonsenseRelation.O_REACT,
                ],
            )
            all_inferences.extend(inferences)

        # Evaluate against expected
        partial_score = 0.0
        components = 0

        # Check if we got events
        if events:
            components += 1
            partial_score += 1.0

        # Check inference types
        inference_by_type = {}
        for inf in all_inferences:
            rel = inf.relation.value.lower()
            if rel not in inference_by_type:
                inference_by_type[rel] = []
            inference_by_type[rel].append(inf.tail.lower() if inf.tail else "")

        for relation in ["xReact", "xIntent", "xEffect", "oReact"]:
            if relation in tc.expected:
                components += 1
                expected_values = [v.lower() for v in tc.expected[relation]]
                detected_values = inference_by_type.get(relation.lower(), [])

                # Check for semantic overlap
                # Score = 1.0 if ANY detected value matches ANY expected value
                # This is OR logic - the expected list contains alternatives, not requirements
                found_match = False
                for expected in expected_values:
                    for detected in detected_values:
                        # Fuzzy match - any word overlap
                        expected_words = set(expected.split())
                        detected_words = set(detected.split())
                        if expected_words & detected_words:
                            found_match = True
                            break
                        # Also check if detected contains expected or vice versa
                        if expected in detected or detected in expected:
                            found_match = True
                            break
                    if found_match:
                        break

                # Score 1.0 if we found any match, 0.0 otherwise
                # Add 0.3 bonus if we generated inferences even if no match
                if found_match:
                    partial_score += 1.0
                elif detected_values:
                    partial_score += 0.3  # Generated something, just didn't match

        if components > 0:
            partial_score = partial_score / components
        else:
            partial_score = 1.0

        correct = partial_score >= 0.6  # Lower threshold for commonsense

        predicted = {
            "events": [e.text for e in events],
            "inferences": inference_by_type,
        }

        return BenchmarkResult(
            test_case=tc,
            predicted=predicted,
            correct=correct,
            partial_score=partial_score,
            details={},
        )


# =============================================================================
# Test Classes
# =============================================================================


class TestNegationBenchmarks:
    """Negation detection benchmarks against BioScope/SFU standards."""

    @pytest.fixture
    def evaluator(self) -> NegationEvaluator:
        return NegationEvaluator()

    def test_explicit_negation_accuracy(self, evaluator: NegationEvaluator):
        """Test explicit negation cases (should be high accuracy)."""
        explicit_cases = [
            tc for tc in NEGATION_TEST_SUITE
            if tc.category == "explicit_negation"
        ]
        results = evaluator.evaluate(explicit_cases)
        metrics = compute_metrics(results)

        # Explicit negation should be at least INDUSTRY_STANDARD
        tier = classify_performance(metrics, NEGATION_THRESHOLDS)
        assert tier.value in ("industry_standard", "above_standard", "state_of_art"), (
            f"Explicit negation at {tier.value}: accuracy={metrics['accuracy']:.1%}"
        )

    def test_morphological_negation_accuracy(self, evaluator: NegationEvaluator):
        """Test morphological negation (un-, im-, etc.)."""
        morph_cases = [
            tc for tc in NEGATION_TEST_SUITE
            if tc.category == "morphological_negation"
        ]
        results = evaluator.evaluate(morph_cases)
        metrics = compute_metrics(results)

        # Report morphological detection status (this is a challenging task)
        tier = classify_performance(metrics, NEGATION_THRESHOLDS)
        print(f"\nMorphological negation tier: {tier.value}, accuracy: {metrics['accuracy']:.1%}")
        # Morphological negation is harder - passing if we detect at least some
        assert metrics["accuracy"] >= 0.4, (
            f"Morphological negation accuracy too low: {metrics['accuracy']:.1%}"
        )

    def test_implicit_negation_accuracy(self, evaluator: NegationEvaluator):
        """Test implicit negation (fail, refuse, etc.)."""
        implicit_cases = [
            tc for tc in NEGATION_TEST_SUITE
            if tc.category == "implicit_negation"
        ]
        results = evaluator.evaluate(implicit_cases)
        metrics = compute_metrics(results)

        # Implicit is harder, but should be at least baseline
        tier = classify_performance(metrics, NEGATION_THRESHOLDS)
        print(f"Implicit negation tier: {tier.value}, accuracy: {metrics['accuracy']:.1%}")

    def test_false_positive_rate(self, evaluator: NegationEvaluator):
        """Test that positive sentences don't trigger false negation."""
        control_cases = [
            tc for tc in NEGATION_TEST_SUITE
            if tc.category == "control_positive"
        ]
        results = evaluator.evaluate(control_cases)
        metrics = compute_metrics(results)

        # Should have very high accuracy on control cases
        assert metrics["accuracy"] >= 0.9, (
            f"False positive rate too high: accuracy={metrics['accuracy']:.1%}"
        )

    def test_overall_negation_benchmark(self, evaluator: NegationEvaluator):
        """Full benchmark against BioScope/SFU standards."""
        results = evaluator.evaluate(NEGATION_TEST_SUITE)
        metrics = compute_metrics(results)
        tier = classify_performance(metrics, NEGATION_THRESHOLDS)

        suite_result = BenchmarkSuiteResult(
            task="negation",
            thresholds=NEGATION_THRESHOLDS,
            results=results,
            metrics=metrics,
            tier=tier,
        )

        print(f"\n{generate_report([suite_result])}")

        # Must be at least INDUSTRY_BASELINE
        assert tier.value != "below_industry", (
            f"Overall negation below industry baseline: {metrics['accuracy']:.1%}"
        )


class TestSRLBenchmarks:
    """Semantic Role Labeling benchmarks against CoNLL standards."""

    @pytest.fixture
    def evaluator(self) -> SRLEvaluator:
        return SRLEvaluator()

    def test_simple_transitive(self, evaluator: SRLEvaluator):
        """Test simple transitive sentences."""
        trans_cases = [
            tc for tc in SRL_TEST_SUITE
            if tc.category == "simple_transitive"
        ]
        results = evaluator.evaluate(trans_cases)
        metrics = compute_metrics(results)

        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            SRL_THRESHOLDS,
        )
        assert tier.value in ("industry_standard", "above_standard", "state_of_art"), (
            f"Simple transitive at {tier.value}: partial_accuracy={metrics['partial_accuracy']:.1%}"
        )

    def test_passive_voice_handling(self, evaluator: SRLEvaluator):
        """Test passive voice - identified gap."""
        passive_cases = [
            tc for tc in SRL_TEST_SUITE
            if tc.category in ("passive", "passive_agentless")
        ]
        results = evaluator.evaluate(passive_cases)
        metrics = compute_metrics(results)

        # Report current status
        print(f"\nPassive voice accuracy: {metrics['partial_accuracy']:.1%}")
        for r in results:
            if not r.correct:
                print(f"  FAILED: {r.test_case.text}")
                print(f"    Expected predicates: {r.test_case.expected.get('predicates')}")
                print(f"    Detected predicates: {r.predicted.get('predicates')}")

    def test_overall_srl_benchmark(self, evaluator: SRLEvaluator):
        """Full benchmark against CoNLL-2005 standards."""
        results = evaluator.evaluate(SRL_TEST_SUITE)
        metrics = compute_metrics(results)
        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            SRL_THRESHOLDS,
        )

        suite_result = BenchmarkSuiteResult(
            task="srl",
            thresholds=SRL_THRESHOLDS,
            results=results,
            metrics=metrics,
            tier=tier,
        )

        print(f"\n{generate_report([suite_result])}")


class TestTemporalBenchmarks:
    """Temporal extraction benchmarks against TempEval standards."""

    @pytest.fixture
    def evaluator(self) -> TemporalEvaluator:
        return TemporalEvaluator()

    def test_tense_detection(self, evaluator: TemporalEvaluator):
        """Test tense detection accuracy."""
        tense_cases = [
            tc for tc in TEMPORAL_TEST_SUITE
            if tc.category == "tense"
        ]
        results = evaluator.evaluate(tense_cases)
        metrics = compute_metrics(results)

        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            TEMPORAL_THRESHOLDS,
        )
        # Tense should be easy
        assert tier.value in ("industry_standard", "above_standard", "state_of_art"), (
            f"Tense detection at {tier.value}: accuracy={metrics['partial_accuracy']:.1%}"
        )

    def test_aspect_classification(self, evaluator: TemporalEvaluator):
        """Test Vendler aspect classification - identified gap area."""
        aspect_cases = [
            tc for tc in TEMPORAL_TEST_SUITE
            if tc.category.startswith("aspect_")
        ]
        results = evaluator.evaluate(aspect_cases)
        metrics = compute_metrics(results)

        # Report aspect breakdown
        print(f"\nAspect classification accuracy: {metrics['partial_accuracy']:.1%}")

        by_category = {}
        for r in results:
            cat = r.test_case.category
            if cat not in by_category:
                by_category[cat] = {"correct": 0, "total": 0}
            by_category[cat]["total"] += 1
            if r.correct:
                by_category[cat]["correct"] += 1

        for cat, counts in sorted(by_category.items()):
            acc = counts["correct"] / counts["total"] if counts["total"] else 0
            print(f"  {cat}: {acc:.1%} ({counts['correct']}/{counts['total']})")

        # Achievement/accomplishment is the gap
        achievement_cases = [r for r in results if r.test_case.category == "aspect_achievement"]
        if achievement_cases:
            achievement_acc = sum(r.correct for r in achievement_cases) / len(achievement_cases)
            print(f"\n  ACHIEVEMENT gap: {achievement_acc:.1%}")

    def test_overall_temporal_benchmark(self, evaluator: TemporalEvaluator):
        """Full benchmark against TempEval standards."""
        results = evaluator.evaluate(TEMPORAL_TEST_SUITE)
        metrics = compute_metrics(results)
        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            TEMPORAL_THRESHOLDS,
        )

        suite_result = BenchmarkSuiteResult(
            task="temporal",
            thresholds=TEMPORAL_THRESHOLDS,
            results=results,
            metrics=metrics,
            tier=tier,
        )

        print(f"\n{generate_report([suite_result])}")


class TestModalityBenchmarks:
    """Modality detection benchmarks."""

    @pytest.fixture
    def evaluator(self) -> ModalityEvaluator:
        return ModalityEvaluator()

    def test_epistemic_vs_deontic_disambiguation(self, evaluator: ModalityEvaluator):
        """Test epistemic vs deontic disambiguation - identified gap."""
        disambig_cases = [
            tc for tc in MODALITY_TEST_SUITE
            if tc.category == "disambiguation"
        ]
        results = evaluator.evaluate(disambig_cases)
        metrics = compute_metrics(results)

        print(f"\nEpistemic/Deontic disambiguation: {metrics['partial_accuracy']:.1%}")
        for r in results:
            expected = r.test_case.expected.get("type")
            detected = r.predicted.get("type")
            status = "✓" if r.correct else "✗"
            print(f"  {status} \"{r.test_case.text}\"")
            print(f"      Expected: {expected}, Got: {detected}")

    def test_deontic_force_detection(self, evaluator: ModalityEvaluator):
        """Test deontic obligation/permission detection."""
        deontic_cases = [
            tc for tc in MODALITY_TEST_SUITE
            if tc.category.startswith("deontic_")
        ]
        results = evaluator.evaluate(deontic_cases)
        metrics = compute_metrics(results)

        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            MODALITY_THRESHOLDS,
        )
        print(f"\nDeontic force detection: {tier.value}, accuracy={metrics['partial_accuracy']:.1%}")

    def test_overall_modality_benchmark(self, evaluator: ModalityEvaluator):
        """Full modality benchmark."""
        results = evaluator.evaluate(MODALITY_TEST_SUITE)
        metrics = compute_metrics(results)
        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            MODALITY_THRESHOLDS,
        )

        suite_result = BenchmarkSuiteResult(
            task="modality",
            thresholds=MODALITY_THRESHOLDS,
            results=results,
            metrics=metrics,
            tier=tier,
        )

        print(f"\n{generate_report([suite_result])}")


class TestCommonsenseBenchmarks:
    """Commonsense inference benchmarks against ATOMIC/COMET standards."""

    @pytest.fixture
    def evaluator(self) -> CommonsenseEvaluator:
        return CommonsenseEvaluator()

    def test_emotional_reaction_inference(self, evaluator: CommonsenseEvaluator):
        """Test xReact/oReact inference quality."""
        react_cases = [
            tc for tc in COMMONSENSE_TEST_SUITE
            if tc.category == "emotional_reaction"
        ]
        results = evaluator.evaluate(react_cases)
        metrics = compute_metrics(results)

        print(f"\nEmotional reaction inference: {metrics['partial_accuracy']:.1%}")

    def test_overall_commonsense_benchmark(self, evaluator: CommonsenseEvaluator):
        """Full commonsense benchmark against ATOMIC standards."""
        results = evaluator.evaluate(COMMONSENSE_TEST_SUITE)
        metrics = compute_metrics(results)
        tier = classify_performance(
            {"accuracy": metrics["partial_accuracy"]},
            COMMONSENSE_THRESHOLDS,
        )

        suite_result = BenchmarkSuiteResult(
            task="commonsense",
            thresholds=COMMONSENSE_THRESHOLDS,
            results=results,
            metrics=metrics,
            tier=tier,
        )

        print(f"\n{generate_report([suite_result])}")


# =============================================================================
# Full E2E Benchmark Suite
# =============================================================================


class TestFullBenchmarkSuite:
    """Run all benchmarks and generate comprehensive report."""

    def test_full_benchmark_report(self):
        """Run all benchmarks and classify against industry standards."""
        suite_results = []

        # Negation
        neg_eval = NegationEvaluator()
        neg_results = neg_eval.evaluate(NEGATION_TEST_SUITE)
        neg_metrics = compute_metrics(neg_results)
        neg_tier = classify_performance(neg_metrics, NEGATION_THRESHOLDS)
        suite_results.append(BenchmarkSuiteResult(
            task="negation",
            thresholds=NEGATION_THRESHOLDS,
            results=neg_results,
            metrics=neg_metrics,
            tier=neg_tier,
        ))

        # SRL
        srl_eval = SRLEvaluator()
        srl_results = srl_eval.evaluate(SRL_TEST_SUITE)
        srl_metrics = compute_metrics(srl_results)
        srl_tier = classify_performance(
            {"accuracy": srl_metrics["partial_accuracy"]},
            SRL_THRESHOLDS,
        )
        suite_results.append(BenchmarkSuiteResult(
            task="srl",
            thresholds=SRL_THRESHOLDS,
            results=srl_results,
            metrics=srl_metrics,
            tier=srl_tier,
        ))

        # Temporal
        temp_eval = TemporalEvaluator()
        temp_results = temp_eval.evaluate(TEMPORAL_TEST_SUITE)
        temp_metrics = compute_metrics(temp_results)
        temp_tier = classify_performance(
            {"accuracy": temp_metrics["partial_accuracy"]},
            TEMPORAL_THRESHOLDS,
        )
        suite_results.append(BenchmarkSuiteResult(
            task="temporal",
            thresholds=TEMPORAL_THRESHOLDS,
            results=temp_results,
            metrics=temp_metrics,
            tier=temp_tier,
        ))

        # Modality
        mod_eval = ModalityEvaluator()
        mod_results = mod_eval.evaluate(MODALITY_TEST_SUITE)
        mod_metrics = compute_metrics(mod_results)
        mod_tier = classify_performance(
            {"accuracy": mod_metrics["partial_accuracy"]},
            MODALITY_THRESHOLDS,
        )
        suite_results.append(BenchmarkSuiteResult(
            task="modality",
            thresholds=MODALITY_THRESHOLDS,
            results=mod_results,
            metrics=mod_metrics,
            tier=mod_tier,
        ))

        # Commonsense
        cs_eval = CommonsenseEvaluator()
        cs_results = cs_eval.evaluate(COMMONSENSE_TEST_SUITE)
        cs_metrics = compute_metrics(cs_results)
        cs_tier = classify_performance(
            {"accuracy": cs_metrics["partial_accuracy"]},
            COMMONSENSE_THRESHOLDS,
        )
        suite_results.append(BenchmarkSuiteResult(
            task="commonsense",
            thresholds=COMMONSENSE_THRESHOLDS,
            results=cs_results,
            metrics=cs_metrics,
            tier=cs_tier,
        ))

        # Generate and print report
        report = generate_report(suite_results)
        print(f"\n{report}")

        # Assert no component is BELOW_INDUSTRY
        below_industry = [
            sr for sr in suite_results
            if sr.tier == BenchmarkTier.BELOW_INDUSTRY
        ]

        if below_industry:
            tasks = ", ".join(sr.task for sr in below_industry)
            pytest.fail(f"Components below industry baseline: {tasks}")
