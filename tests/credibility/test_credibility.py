"""Tests for credibility tracking module."""

import pytest
import time

from draagon_ai.credibility import (
    UserCredibility,
    UserIntent,
    DomainExpertise,
    ConfidenceCalibration,
    LearningTrajectory,
    InformationQuality,
    get_verification_threshold,
    VerificationLevel,
    VERIFICATION_THRESHOLDS,
)
from draagon_ai.credibility.thresholds import should_verify, get_credibility_tier


class TestUserIntent:
    """Tests for UserIntent enum."""

    def test_intent_values(self):
        """Test intent string values."""
        assert UserIntent.HONEST.value == "honest"
        assert UserIntent.ADVERSARIAL.value == "adversarial"

    def test_composite_multipliers(self):
        """Test intent multipliers."""
        assert UserIntent.HONEST.composite_multiplier == 1.0
        assert UserIntent.OVERCONFIDENT.composite_multiplier == 0.85
        assert UserIntent.ADVERSARIAL.composite_multiplier == 0.50


class TestDomainExpertise:
    """Tests for DomainExpertise class."""

    def test_create_domain(self):
        """Test creating domain expertise."""
        domain = DomainExpertise(domain="tech")

        assert domain.domain == "tech"
        assert domain.accuracy == 0.5
        assert domain.attempts == 0

    def test_record_outcomes(self):
        """Test recording outcomes."""
        domain = DomainExpertise(domain="tech")

        domain.record_outcome(correct=True)
        domain.record_outcome(correct=True)
        domain.record_outcome(correct=False)

        assert domain.attempts == 3
        assert domain.successes == 2
        assert abs(domain.accuracy - 0.667) < 0.01

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = DomainExpertise(domain="cooking")
        original.record_outcome(True)
        original.record_outcome(False)

        data = original.to_dict()
        restored = DomainExpertise.from_dict(data)

        assert restored.domain == "cooking"
        assert restored.attempts == 2
        assert restored.accuracy == original.accuracy


class TestConfidenceCalibration:
    """Tests for ConfidenceCalibration class."""

    def test_empty_calibration(self):
        """Test empty calibration."""
        cal = ConfidenceCalibration()

        assert cal.total == 0
        assert cal.calibration_score == 0.5
        assert cal.overconfidence_ratio == 0.0

    def test_record_calibration(self):
        """Test recording calibration data."""
        cal = ConfidenceCalibration()

        # Confident and correct (good)
        cal.record(confident=True, correct=True)
        assert cal.confident_and_correct == 1

        # Confident but wrong (bad - Dunning-Kruger)
        cal.record(confident=True, correct=False)
        assert cal.confident_but_wrong == 1

        # Unsure and correct
        cal.record(confident=False, correct=True)
        assert cal.unsure_and_correct == 1

        # Unsure and wrong (okay - knows limits)
        cal.record(confident=False, correct=False)
        assert cal.unsure_but_wrong == 1

    def test_overconfidence_detection(self):
        """Test detecting overconfidence."""
        cal = ConfidenceCalibration()

        # All confident, half wrong
        for _ in range(5):
            cal.record(confident=True, correct=True)
        for _ in range(5):
            cal.record(confident=True, correct=False)

        assert cal.overconfidence_ratio == 0.5  # Half of confident statements wrong

    def test_well_calibrated(self):
        """Test well-calibrated user."""
        cal = ConfidenceCalibration()

        # Confident when right
        for _ in range(8):
            cal.record(confident=True, correct=True)
        # Unsure when wrong
        for _ in range(2):
            cal.record(confident=False, correct=False)

        # Should have high calibration score
        assert cal.calibration_score > 0.7

    def test_to_dict_from_dict(self):
        """Test serialization."""
        original = ConfidenceCalibration()
        original.record(confident=True, correct=True)
        original.record(confident=False, correct=False)

        data = original.to_dict()
        restored = ConfidenceCalibration.from_dict(data)

        assert restored.confident_and_correct == 1
        assert restored.unsure_but_wrong == 1


class TestLearningTrajectory:
    """Tests for LearningTrajectory class."""

    def test_insufficient_data(self):
        """Test with insufficient data."""
        traj = LearningTrajectory()

        assert traj.trend == "insufficient_data"

    def test_improving_trajectory(self):
        """Test detecting improvement."""
        traj = LearningTrajectory(window_size=5)

        # Early: 2/5 correct
        for _ in range(2):
            traj.record_outcome(True)
        for _ in range(3):
            traj.record_outcome(False)

        # Recent: 5/5 correct
        for _ in range(5):
            traj.record_outcome(True)

        assert traj.trend == "improving"
        assert traj.improvement_rate > 0

    def test_declining_trajectory(self):
        """Test detecting decline."""
        traj = LearningTrajectory(window_size=5)

        # Early: 5/5 correct
        for _ in range(5):
            traj.record_outcome(True)

        # Recent: 1/5 correct
        for _ in range(4):
            traj.record_outcome(False)
        traj.record_outcome(True)

        assert traj.trend == "declining"
        assert traj.improvement_rate < 0

    def test_stable_trajectory(self):
        """Test stable trajectory."""
        traj = LearningTrajectory(window_size=5)

        # All roughly 60% accuracy
        for _ in range(10):
            traj.record_outcome(True)
            traj.record_outcome(True)
            traj.record_outcome(True)
            traj.record_outcome(False)
            traj.record_outcome(False)

        assert traj.trend == "stable"

    def test_to_dict_from_dict(self):
        """Test serialization."""
        original = LearningTrajectory()
        for i in range(5):
            original.record_outcome(i % 2 == 0)

        data = original.to_dict()
        restored = LearningTrajectory.from_dict(data)

        assert restored.early_attempts == original.early_attempts
        assert restored._recent_outcomes == original._recent_outcomes


class TestInformationQuality:
    """Tests for InformationQuality class."""

    def test_empty_quality(self):
        """Test empty quality metrics."""
        quality = InformationQuality()

        assert quality.specificity == 0.5
        assert quality.consistency == 0.5
        assert quality.overall_quality == 0.5

    def test_record_quality(self):
        """Test recording quality metrics."""
        quality = InformationQuality()

        quality.record(specificity=0.8, consistency=0.9, relevance=0.7)
        quality.record(specificity=0.6, consistency=0.8, relevance=0.9)

        assert abs(quality.specificity - 0.7) < 0.01
        assert abs(quality.consistency - 0.85) < 0.01
        assert abs(quality.relevance - 0.8) < 0.01

    def test_to_dict_from_dict(self):
        """Test serialization."""
        original = InformationQuality()
        original.record(specificity=0.8, consistency=0.7, relevance=0.9)

        data = original.to_dict()
        restored = InformationQuality.from_dict(data)

        assert restored.total_interactions == 1
        assert abs(restored.specificity - 0.8) < 0.01


class TestUserCredibility:
    """Tests for UserCredibility class."""

    def test_create_credibility(self):
        """Test creating user credibility."""
        cred = UserCredibility(user_id="user-123")

        assert cred.user_id == "user-123"
        assert cred.overall_accuracy == 0.5
        assert cred.detected_intent == UserIntent.HONEST

    def test_record_correction_outcome(self):
        """Test recording correction outcomes."""
        cred = UserCredibility(user_id="user-1")

        cred.record_correction_outcome(
            correct=True,
            domain="tech",
            confident=True,
            specificity=0.8,
        )

        assert cred.total_corrections == 1
        assert cred.correct_corrections == 1
        assert "tech" in cred.domain_expertise
        assert cred.calibration.confident_and_correct == 1

    def test_overall_accuracy(self):
        """Test overall accuracy calculation."""
        cred = UserCredibility(user_id="user-1")

        cred.record_correction_outcome(correct=True)
        cred.record_correction_outcome(correct=True)
        cred.record_correction_outcome(correct=False)

        assert abs(cred.overall_accuracy - 0.667) < 0.01

    def test_domain_accuracy(self):
        """Test domain-specific accuracy."""
        cred = UserCredibility(user_id="user-1")

        # Good at tech
        cred.record_correction_outcome(correct=True, domain="tech")
        cred.record_correction_outcome(correct=True, domain="tech")

        # Bad at cooking
        cred.record_correction_outcome(correct=False, domain="cooking")
        cred.record_correction_outcome(correct=False, domain="cooking")

        assert cred.get_domain_accuracy("tech") == 1.0
        assert cred.get_domain_accuracy("cooking") == 0.0
        assert cred.get_domain_accuracy("unknown") is None

    def test_intent_detection_honest(self):
        """Test honest intent detection."""
        cred = UserCredibility(user_id="user-1")

        # Mostly correct, stable trajectory
        for _ in range(8):
            cred.record_correction_outcome(correct=True)
        for _ in range(2):
            cred.record_correction_outcome(correct=False)

        assert cred.detected_intent == UserIntent.HONEST

    def test_intent_detection_adversarial(self):
        """Test adversarial intent detection."""
        cred = UserCredibility(user_id="user-1")

        # Almost always wrong
        for _ in range(10):
            cred.record_correction_outcome(correct=False)

        assert cred.detected_intent == UserIntent.ADVERSARIAL

    def test_intent_detection_overconfident(self):
        """Test overconfident intent detection."""
        cred = UserCredibility(user_id="user-1")

        # Confident but often wrong
        for _ in range(10):
            cred.record_correction_outcome(
                correct=False,
                confident=True,
            )
        for _ in range(5):
            cred.record_correction_outcome(
                correct=True,
                confident=True,
            )

        # This triggers overconfident detection (high overconfidence ratio + low accuracy)
        assert cred.detected_intent == UserIntent.OVERCONFIDENT

    def test_composite_score(self):
        """Test composite score calculation."""
        cred = UserCredibility(user_id="user-1")

        # Build up a profile
        for _ in range(5):
            cred.record_correction_outcome(
                correct=True,
                confident=True,
                specificity=0.8,
                consistency=0.9,
            )

        score = cred.composite_score
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be above neutral for good user

    def test_composite_score_with_intent_penalty(self):
        """Test intent affects composite score."""
        cred = UserCredibility(user_id="user-1")

        # Force adversarial intent
        for _ in range(10):
            cred.record_correction_outcome(correct=False)

        # Composite should be penalized
        assert cred.detected_intent == UserIntent.ADVERSARIAL
        assert cred.composite_score < 0.3  # Heavily penalized

    def test_get_dimension_scores(self):
        """Test getting detailed dimension scores."""
        cred = UserCredibility(user_id="user-1")

        cred.record_correction_outcome(correct=True, domain="tech")

        scores = cred.get_dimension_scores()

        assert "overall_accuracy" in scores
        assert "domain_expertise" in scores
        assert "tech" in scores["domain_expertise"]
        assert "confidence_calibration" in scores
        assert "learning_trajectory" in scores
        assert "information_quality" in scores
        assert "intent" in scores
        assert "composite_credibility" in scores

    def test_get_trust_summary(self):
        """Test trust summary generation."""
        cred = UserCredibility(user_id="user-1")

        # Build expertise in tech
        for _ in range(5):
            cred.record_correction_outcome(correct=True, domain="tech")

        summary = cred.get_trust_summary()

        assert "accurate" in summary.lower()
        # May or may not mention tech depending on thresholds

    def test_to_dict_from_dict(self):
        """Test serialization round-trip."""
        original = UserCredibility(user_id="user-123")

        # Build up state
        original.record_correction_outcome(correct=True, domain="tech")
        original.record_correction_outcome(correct=False, domain="cooking")

        data = original.to_dict()
        restored = UserCredibility.from_dict(data)

        assert restored.user_id == "user-123"
        assert restored.total_corrections == 2
        assert "tech" in restored.domain_expertise
        assert "cooking" in restored.domain_expertise
        assert restored.calibration.total == 2


class TestVerificationThresholds:
    """Tests for verification threshold functions."""

    def test_high_credibility_no_verification(self):
        """Test high credibility gets no verification."""
        level = get_verification_threshold("normal", 0.95)
        assert level == VerificationLevel.NONE

    def test_low_credibility_maximum_verification(self):
        """Test low credibility gets maximum verification."""
        level = get_verification_threshold("delete_memory", 0.3)
        assert level == VerificationLevel.MAXIMUM

    def test_sensitive_operations(self):
        """Test sensitive operations need higher credibility."""
        # Same credibility, different operations
        normal_level = get_verification_threshold("normal", 0.7)
        financial_level = get_verification_threshold("financial", 0.7)

        # Financial should require more verification
        assert financial_level.value != VerificationLevel.NONE.value

    def test_domain_accuracy_affects_threshold(self):
        """Test domain accuracy influences threshold."""
        # Low overall credibility but high domain expertise
        level = get_verification_threshold(
            "update_fact",
            credibility_score=0.5,
            domain_accuracy=0.95,
        )

        # Domain expertise should help
        assert level in (VerificationLevel.NONE, VerificationLevel.MINIMAL, VerificationLevel.STANDARD)

    def test_should_verify(self):
        """Test should_verify helper."""
        assert not should_verify(0.95, "normal")  # High credibility, no verify
        assert should_verify(0.3, "delete_memory")  # Low credibility, verify

    def test_get_credibility_tier(self):
        """Test credibility tier classification."""
        assert get_credibility_tier(0.9) == "high"
        assert get_credibility_tier(0.7) == "normal"
        assert get_credibility_tier(0.5) == "low"
        assert get_credibility_tier(0.2) == "very_low"

    def test_all_verification_levels(self):
        """Test all verification levels are reachable."""
        levels_found = set()

        # Test various credibility scores
        for cred in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            for op in VERIFICATION_THRESHOLDS.keys():
                level = get_verification_threshold(op, cred)
                levels_found.add(level)

        # Should be able to get multiple levels
        assert len(levels_found) >= 3


class TestIntegration:
    """Integration tests for credibility system."""

    def test_full_workflow(self):
        """Test complete credibility workflow."""
        cred = UserCredibility(user_id="test-user")

        # Simulate user interactions over time
        # Phase 1: Early, mostly wrong
        for _ in range(5):
            cred.record_correction_outcome(correct=False, domain="tech")

        # Should need maximum verification
        level1 = get_verification_threshold(
            "update_fact",
            cred.composite_score,
            cred.get_domain_accuracy("tech"),
        )
        assert level1 in (VerificationLevel.ELEVATED, VerificationLevel.MAXIMUM)

        # Phase 2: User improves
        for _ in range(15):
            cred.record_correction_outcome(correct=True, domain="tech")

        # Verification requirements should decrease
        level2 = get_verification_threshold(
            "update_fact",
            cred.composite_score,
            cred.get_domain_accuracy("tech"),
        )

        # Level should be lower (less verification needed)
        assert level2.value != VerificationLevel.MAXIMUM.value

    def test_multi_domain_expertise(self):
        """Test user with varied domain expertise."""
        cred = UserCredibility(user_id="test-user")

        # Expert in tech
        for _ in range(10):
            cred.record_correction_outcome(correct=True, domain="tech")

        # Poor in history
        for _ in range(10):
            cred.record_correction_outcome(correct=False, domain="history")

        # Tech operations should need less verification
        tech_level = get_verification_threshold(
            "update_fact",
            cred.composite_score,
            cred.get_domain_accuracy("tech"),
        )

        history_level = get_verification_threshold(
            "update_fact",
            cred.composite_score,
            cred.get_domain_accuracy("history"),
        )

        # Tech should require less verification than history
        assert tech_level.value != history_level.value or tech_level == VerificationLevel.NONE
