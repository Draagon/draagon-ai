"""Models for credibility tracking.

This module defines the data structures for multi-dimensional credibility
tracking, including domain expertise, confidence calibration, and learning
trajectory.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class UserIntent(str, Enum):
    """Detected intent patterns for user corrections.

    The intent affects how the system responds to user input and
    applies penalties/boosts to credibility.
    """

    HONEST = "honest"              # Mostly correct, normal mistakes
    MISTAKEN = "mistaken"          # Often wrong but improving
    OVERCONFIDENT = "overconfident"  # Sure when wrong (Dunning-Kruger)
    TESTING = "testing"            # Testing the system
    UNRELIABLE = "unreliable"      # Wrong + declining trajectory
    ADVERSARIAL = "adversarial"    # 80%+ wrong, not improving

    @property
    def composite_multiplier(self) -> float:
        """Get multiplier for composite credibility score."""
        multipliers = {
            UserIntent.HONEST: 1.0,
            UserIntent.MISTAKEN: 1.0,
            UserIntent.OVERCONFIDENT: 0.85,
            UserIntent.TESTING: 1.0,
            UserIntent.UNRELIABLE: 0.70,
            UserIntent.ADVERSARIAL: 0.50,
        }
        return multipliers.get(self, 1.0)


@dataclass
class DomainExpertise:
    """Per-domain accuracy tracking.

    Users might be experts in some areas but not others.
    This tracks accuracy per domain for adaptive trust.

    Attributes:
        domain: The domain (e.g., "tech", "history", "cooking")
        accuracy: Accuracy rate in this domain (0.0-1.0)
        attempts: Total correction attempts in domain
        successes: Successful (verified) corrections
        last_updated: When this expertise was last updated
    """

    domain: str
    accuracy: float = 0.5  # Start neutral
    attempts: int = 0
    successes: int = 0
    last_updated: float = field(default_factory=time.time)

    def record_outcome(self, correct: bool) -> None:
        """Record a correction outcome in this domain.

        Args:
            correct: Whether the correction was verified as correct
        """
        self.attempts += 1
        if correct:
            self.successes += 1
        self.accuracy = self.successes / self.attempts if self.attempts > 0 else 0.5
        self.last_updated = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "domain": self.domain,
            "accuracy": self.accuracy,
            "attempts": self.attempts,
            "successes": self.successes,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DomainExpertise:
        """Create from dictionary."""
        return cls(
            domain=data["domain"],
            accuracy=data.get("accuracy", 0.5),
            attempts=data.get("attempts", 0),
            successes=data.get("successes", 0),
            last_updated=data.get("last_updated", time.time()),
        )


@dataclass
class ConfidenceCalibration:
    """Tracks how well user knows what they know.

    Detects Dunning-Kruger effect (overconfidence when wrong)
    vs well-calibrated confidence.

    Attributes:
        confident_and_correct: User was confident and right
        confident_but_wrong: User was confident but wrong (bad)
        unsure_and_correct: User was unsure but right
        unsure_but_wrong: User was unsure and wrong (okay - knows limits)
    """

    confident_and_correct: int = 0
    confident_but_wrong: int = 0
    unsure_and_correct: int = 0
    unsure_but_wrong: int = 0

    @property
    def total(self) -> int:
        """Total interactions tracked."""
        return (
            self.confident_and_correct +
            self.confident_but_wrong +
            self.unsure_and_correct +
            self.unsure_but_wrong
        )

    @property
    def calibration_score(self) -> float:
        """Calculate calibration score (0.0-1.0).

        Higher = better calibrated (confident when right, unsure when wrong).
        """
        if self.total == 0:
            return 0.5  # Neutral

        # Good: confident+correct, unsure+wrong
        good = self.confident_and_correct + self.unsure_but_wrong
        # Bad: confident+wrong
        bad = self.confident_but_wrong

        # Score based on ratio of good to total, penalizing overconfidence
        score = good / self.total
        # Extra penalty for overconfidence
        overconfidence_penalty = bad / self.total * 0.2

        return max(0.0, min(1.0, score - overconfidence_penalty))

    @property
    def overconfidence_ratio(self) -> float:
        """Ratio of confident-but-wrong to all confident statements."""
        total_confident = self.confident_and_correct + self.confident_but_wrong
        if total_confident == 0:
            return 0.0
        return self.confident_but_wrong / total_confident

    def record(self, confident: bool, correct: bool) -> None:
        """Record a calibration data point.

        Args:
            confident: Whether user expressed confidence
            correct: Whether the claim was correct
        """
        if confident and correct:
            self.confident_and_correct += 1
        elif confident and not correct:
            self.confident_but_wrong += 1
        elif not confident and correct:
            self.unsure_and_correct += 1
        else:
            self.unsure_but_wrong += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "confident_and_correct": self.confident_and_correct,
            "confident_but_wrong": self.confident_but_wrong,
            "unsure_and_correct": self.unsure_and_correct,
            "unsure_but_wrong": self.unsure_but_wrong,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConfidenceCalibration:
        """Create from dictionary."""
        return cls(
            confident_and_correct=data.get("confident_and_correct", 0),
            confident_but_wrong=data.get("confident_but_wrong", 0),
            unsure_and_correct=data.get("unsure_and_correct", 0),
            unsure_but_wrong=data.get("unsure_but_wrong", 0),
        )


@dataclass
class LearningTrajectory:
    """Tracks whether user is improving over time.

    Users who improve get trust boosts; declining users get more scrutiny.

    Attributes:
        early_accuracy: Accuracy in first N interactions
        current_accuracy: Accuracy in last N interactions
        early_attempts: Number of early interactions recorded
        recent_attempts: Number of recent interactions recorded
        window_size: Number of interactions to consider for each period
    """

    early_accuracy: float = 0.5
    current_accuracy: float = 0.5
    early_attempts: int = 0
    recent_attempts: int = 0
    window_size: int = 10

    # Tracking recent outcomes for rolling window
    _recent_outcomes: list[bool] = field(default_factory=list)

    @property
    def trend(self) -> str:
        """Get trend direction."""
        if self.early_attempts < 5 or self.recent_attempts < 5:
            return "insufficient_data"

        diff = self.current_accuracy - self.early_accuracy
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    @property
    def improvement_rate(self) -> float:
        """Calculate improvement rate (-1.0 to 1.0)."""
        return self.current_accuracy - self.early_accuracy

    def record_outcome(self, correct: bool) -> None:
        """Record an outcome for trajectory tracking.

        Args:
            correct: Whether the interaction was correct
        """
        # Update early stats if we're still in early phase
        if self.early_attempts < self.window_size:
            self.early_attempts += 1
            if correct:
                successes = self.early_accuracy * (self.early_attempts - 1) + 1
            else:
                successes = self.early_accuracy * (self.early_attempts - 1)
            self.early_accuracy = successes / self.early_attempts

        # Always update recent (rolling window)
        self._recent_outcomes.append(correct)
        if len(self._recent_outcomes) > self.window_size:
            self._recent_outcomes.pop(0)

        self.recent_attempts = len(self._recent_outcomes)
        self.current_accuracy = sum(self._recent_outcomes) / len(self._recent_outcomes)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "early_accuracy": self.early_accuracy,
            "current_accuracy": self.current_accuracy,
            "early_attempts": self.early_attempts,
            "recent_attempts": self.recent_attempts,
            "window_size": self.window_size,
            "recent_outcomes": self._recent_outcomes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningTrajectory:
        """Create from dictionary."""
        trajectory = cls(
            early_accuracy=data.get("early_accuracy", 0.5),
            current_accuracy=data.get("current_accuracy", 0.5),
            early_attempts=data.get("early_attempts", 0),
            recent_attempts=data.get("recent_attempts", 0),
            window_size=data.get("window_size", 10),
        )
        trajectory._recent_outcomes = data.get("recent_outcomes", [])
        return trajectory


@dataclass
class InformationQuality:
    """Tracks quality of information provided by user.

    Measures specificity, consistency, and relevance of user input.

    Attributes:
        specificity_sum: Sum of specificity scores
        consistency_sum: Sum of consistency scores
        relevance_sum: Sum of relevance scores
        total_interactions: Total rated interactions
    """

    specificity_sum: float = 0.0
    consistency_sum: float = 0.0
    relevance_sum: float = 0.0
    total_interactions: int = 0

    @property
    def specificity(self) -> float:
        """Average specificity score (0.0-1.0)."""
        if self.total_interactions == 0:
            return 0.5
        return self.specificity_sum / self.total_interactions

    @property
    def consistency(self) -> float:
        """Average consistency score (0.0-1.0)."""
        if self.total_interactions == 0:
            return 0.5
        return self.consistency_sum / self.total_interactions

    @property
    def relevance(self) -> float:
        """Average relevance score (0.0-1.0)."""
        if self.total_interactions == 0:
            return 0.5
        return self.relevance_sum / self.total_interactions

    @property
    def overall_quality(self) -> float:
        """Combined quality score (0.0-1.0)."""
        return (self.specificity + self.consistency + self.relevance) / 3

    def record(
        self,
        specificity: float = 0.5,
        consistency: float = 0.5,
        relevance: float = 0.5,
    ) -> None:
        """Record quality metrics for an interaction.

        Args:
            specificity: How specific/detailed the information was
            consistency: How consistent with previous statements
            relevance: How relevant to the topic
        """
        self.specificity_sum += specificity
        self.consistency_sum += consistency
        self.relevance_sum += relevance
        self.total_interactions += 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "specificity_sum": self.specificity_sum,
            "consistency_sum": self.consistency_sum,
            "relevance_sum": self.relevance_sum,
            "total_interactions": self.total_interactions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InformationQuality:
        """Create from dictionary."""
        return cls(
            specificity_sum=data.get("specificity_sum", 0.0),
            consistency_sum=data.get("consistency_sum", 0.0),
            relevance_sum=data.get("relevance_sum", 0.0),
            total_interactions=data.get("total_interactions", 0),
        )
