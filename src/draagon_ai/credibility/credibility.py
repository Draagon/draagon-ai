"""Multi-dimensional user credibility tracking.

This module provides the main UserCredibility class that combines
multiple dimensions of trust into a composite credibility score.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .models import (
    UserIntent,
    DomainExpertise,
    ConfidenceCalibration,
    LearningTrajectory,
    InformationQuality,
)


# Weights for composite score calculation
DIMENSION_WEIGHTS = {
    "accuracy": 0.35,
    "calibration": 0.20,
    "trajectory": 0.15,
    "quality": 0.15,
    "consistency": 0.15,
}


@dataclass
class UserCredibility:
    """Multi-dimensional credibility tracking for a user.

    Tracks six dimensions of credibility:
    1. Overall accuracy - Verified corrections ratio
    2. Domain expertise - Per-topic accuracy
    3. Confidence calibration - Knows what they know?
    4. Learning trajectory - Improving over time?
    5. Information quality - Specific, consistent, relevant
    6. Intent pattern - Honest mistakes vs adversarial

    Usage:
        credibility = UserCredibility(user_id="user-123")

        # Record a correction outcome
        credibility.record_correction_outcome(
            correct=True,
            domain="tech",
            confident=True,
            specificity=0.8,
        )

        # Get composite score
        score = credibility.composite_score

        # Get verification threshold
        threshold = credibility.get_verification_threshold()

    Attributes:
        user_id: User identifier
        total_corrections: Total verifiable corrections made
        correct_corrections: Number of correct corrections
        domain_expertise: Per-domain accuracy tracking
        calibration: Confidence calibration tracking
        trajectory: Learning trajectory over time
        quality: Information quality metrics
        detected_intent: Current detected intent pattern
        created_at: When tracking started
        last_updated: Last update timestamp
    """

    user_id: str
    total_corrections: int = 0
    correct_corrections: int = 0
    domain_expertise: dict[str, DomainExpertise] = field(default_factory=dict)
    calibration: ConfidenceCalibration = field(default_factory=ConfidenceCalibration)
    trajectory: LearningTrajectory = field(default_factory=LearningTrajectory)
    quality: InformationQuality = field(default_factory=InformationQuality)
    detected_intent: UserIntent = UserIntent.HONEST
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

    @property
    def overall_accuracy(self) -> float:
        """Overall correction accuracy (0.0-1.0)."""
        if self.total_corrections == 0:
            return 0.5  # Neutral starting point
        return self.correct_corrections / self.total_corrections

    @property
    def composite_score(self) -> float:
        """Calculate weighted composite credibility score.

        Combines all dimensions with intent multiplier.

        Returns:
            Credibility score from 0.0 to 1.0
        """
        # Calculate each dimension
        accuracy = self.overall_accuracy
        calibration_score = self.calibration.calibration_score
        trajectory_score = self._trajectory_to_score()
        quality_score = self.quality.overall_quality
        consistency_score = self.quality.consistency

        # Weighted combination
        composite = (
            accuracy * DIMENSION_WEIGHTS["accuracy"] +
            calibration_score * DIMENSION_WEIGHTS["calibration"] +
            trajectory_score * DIMENSION_WEIGHTS["trajectory"] +
            quality_score * DIMENSION_WEIGHTS["quality"] +
            consistency_score * DIMENSION_WEIGHTS["consistency"]
        )

        # Apply intent multiplier
        composite *= self.detected_intent.composite_multiplier

        return max(0.0, min(1.0, composite))

    def _trajectory_to_score(self) -> float:
        """Convert trajectory to a 0-1 score."""
        trend = self.trajectory.trend
        if trend == "insufficient_data":
            return 0.5  # Neutral
        elif trend == "improving":
            # Boost for improvement (0.5 -> 0.8 max)
            return 0.5 + min(0.3, self.trajectory.improvement_rate)
        elif trend == "declining":
            # Penalty for decline (0.5 -> 0.2 min)
            return max(0.2, 0.5 + self.trajectory.improvement_rate)
        return 0.5  # Stable

    def record_correction_outcome(
        self,
        correct: bool,
        domain: str | None = None,
        confident: bool = True,
        specificity: float = 0.5,
        consistency: float = 0.5,
        relevance: float = 0.5,
    ) -> None:
        """Record outcome of a user correction.

        Args:
            correct: Whether the correction was verified as correct
            domain: Domain of the correction (e.g., "tech", "cooking")
            confident: Whether user expressed confidence
            specificity: How specific the information was
            consistency: How consistent with previous statements
            relevance: How relevant to the topic
        """
        # Update overall accuracy
        self.total_corrections += 1
        if correct:
            self.correct_corrections += 1

        # Update domain expertise if domain provided
        if domain:
            if domain not in self.domain_expertise:
                self.domain_expertise[domain] = DomainExpertise(domain=domain)
            self.domain_expertise[domain].record_outcome(correct)

        # Update calibration
        self.calibration.record(confident=confident, correct=correct)

        # Update trajectory
        self.trajectory.record_outcome(correct)

        # Update quality
        self.quality.record(
            specificity=specificity,
            consistency=consistency,
            relevance=relevance,
        )

        # Re-detect intent
        self._update_intent()

        self.last_updated = time.time()

    def _update_intent(self) -> None:
        """Update detected intent based on current metrics."""
        accuracy = self.overall_accuracy
        trajectory = self.trajectory.trend
        overconfidence = self.calibration.overconfidence_ratio

        # Check for adversarial (80%+ wrong, not improving)
        if accuracy < 0.2 and trajectory != "improving":
            self.detected_intent = UserIntent.ADVERSARIAL
        # Check for unreliable (wrong + declining)
        elif accuracy < 0.5 and trajectory == "declining":
            self.detected_intent = UserIntent.UNRELIABLE
        # Check for overconfident (Dunning-Kruger)
        elif overconfidence > 0.4 and accuracy < 0.6:
            self.detected_intent = UserIntent.OVERCONFIDENT
        # Check for mistaken but trying (wrong but improving)
        elif accuracy < 0.6 and trajectory == "improving":
            self.detected_intent = UserIntent.MISTAKEN
        # Default to honest
        else:
            self.detected_intent = UserIntent.HONEST

    def get_domain_accuracy(self, domain: str) -> float | None:
        """Get accuracy for a specific domain.

        Args:
            domain: Domain to check

        Returns:
            Accuracy for domain, or None if no data
        """
        if domain in self.domain_expertise:
            return self.domain_expertise[domain].accuracy
        return None

    def get_dimension_scores(self) -> dict[str, Any]:
        """Get detailed scores for all dimensions.

        Returns:
            Dict with all dimension scores and details
        """
        return {
            "overall_accuracy": self.overall_accuracy,
            "domain_expertise": {
                domain: exp.accuracy
                for domain, exp in self.domain_expertise.items()
            },
            "confidence_calibration": {
                "score": self.calibration.calibration_score,
                "overconfidence_ratio": self.calibration.overconfidence_ratio,
            },
            "learning_trajectory": {
                "trend": self.trajectory.trend,
                "improvement_rate": self.trajectory.improvement_rate,
            },
            "information_quality": {
                "specificity": self.quality.specificity,
                "consistency": self.quality.consistency,
                "relevance": self.quality.relevance,
            },
            "intent": self.detected_intent.value,
            "composite_credibility": self.composite_score,
        }

    def get_trust_summary(self) -> str:
        """Get human-readable trust summary.

        Returns:
            Summary string describing user's credibility
        """
        parts = []

        # Overall accuracy
        acc_pct = int(self.overall_accuracy * 100)
        parts.append(f"{acc_pct}% accurate overall")

        # Calibration
        if self.calibration.total > 5:
            if self.calibration.overconfidence_ratio > 0.3:
                parts.append("tends to be overconfident")
            elif self.calibration.calibration_score > 0.7:
                parts.append("knows their limits well")

        # Trajectory
        if self.trajectory.trend == "improving":
            parts.append("getting better over time")
        elif self.trajectory.trend == "declining":
            parts.append("reliability declining")

        # Domain expertise
        expert_domains = [
            domain for domain, exp in self.domain_expertise.items()
            if exp.accuracy > 0.8 and exp.attempts >= 3
        ]
        if expert_domains:
            parts.append(f"expertise in: {', '.join(expert_domains)} ({int(self.domain_expertise[expert_domains[0]].accuracy * 100)}%)")

        return "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "total_corrections": self.total_corrections,
            "correct_corrections": self.correct_corrections,
            "domain_expertise": {
                domain: exp.to_dict()
                for domain, exp in self.domain_expertise.items()
            },
            "calibration": self.calibration.to_dict(),
            "trajectory": self.trajectory.to_dict(),
            "quality": self.quality.to_dict(),
            "detected_intent": self.detected_intent.value,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UserCredibility:
        """Create from dictionary."""
        credibility = cls(
            user_id=data["user_id"],
            total_corrections=data.get("total_corrections", 0),
            correct_corrections=data.get("correct_corrections", 0),
            detected_intent=UserIntent(data.get("detected_intent", "honest")),
            created_at=data.get("created_at", time.time()),
            last_updated=data.get("last_updated", time.time()),
        )

        # Restore domain expertise
        for domain, exp_data in data.get("domain_expertise", {}).items():
            credibility.domain_expertise[domain] = DomainExpertise.from_dict(exp_data)

        # Restore other models
        if "calibration" in data:
            credibility.calibration = ConfidenceCalibration.from_dict(data["calibration"])
        if "trajectory" in data:
            credibility.trajectory = LearningTrajectory.from_dict(data["trajectory"])
        if "quality" in data:
            credibility.quality = InformationQuality.from_dict(data["quality"])

        return credibility
