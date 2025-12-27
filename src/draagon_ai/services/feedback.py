"""Feedback Collector.

This module collects and manages interaction feedback for evolution.
It tracks user interactions, corrections, and success signals to build
test cases for the evolution pipeline.

Usage:
    from draagon_ai.services import FeedbackCollector

    collector = FeedbackCollector()

    # Record interactions
    interaction_id = collector.record_interaction(
        query="What time is it?",
        response="It's 3:45 PM",
        action="get_time",
        success=True,
    )

    # Record user feedback
    collector.record_success(interaction_id, quality=0.9)
    # or
    collector.record_correction(interaction_id, "Actually, can you tell me the date?")

    # Get cases for evolution
    test_cases = collector.get_evolution_cases()
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from draagon_ai.evolution import EvolutionTestCase, InteractionFeedback

logger = logging.getLogger(__name__)


# =============================================================================
# Types
# =============================================================================


@dataclass
class InteractionRecord:
    """A recorded user interaction."""

    interaction_id: str
    query: str
    response: str
    action: str | None = None
    action_args: dict[str, Any] | None = None
    success: bool = True
    latency_ms: float = 0.0
    user_id: str = ""
    session_id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)

    # Feedback (may be added later)
    quality_score: float | None = None
    user_correction: str | None = None
    was_corrected: bool = False

    def to_evolution_test_case(self) -> EvolutionTestCase:
        """Convert to an EvolutionTestCase for the evolution pipeline."""
        return EvolutionTestCase(
            query=self.query,
            expected_action=self.action,
            expected_params=self.action_args,
            was_successful=self.success and not self.was_corrected,
            quality_score=self.quality_score or (0.8 if self.success else 0.3),
            user_correction=self.user_correction,
        )

    def to_interaction_feedback(self) -> InteractionFeedback:
        """Convert to InteractionFeedback for context evolution."""
        return InteractionFeedback(
            query=self.query,
            response=self.response,
            success=self.success and not self.was_corrected,
            quality_score=self.quality_score or (0.8 if self.success else 0.3),
            user_correction=self.user_correction,
        )


@dataclass
class FeedbackStats:
    """Statistics about collected feedback."""

    total_interactions: int = 0
    successful_interactions: int = 0
    corrected_interactions: int = 0
    avg_quality_score: float = 0.0
    avg_latency_ms: float = 0.0
    unique_users: int = 0
    unique_sessions: int = 0
    actions_distribution: dict[str, int] = field(default_factory=dict)


# =============================================================================
# Feedback Collector
# =============================================================================


class FeedbackCollector:
    """Collects and manages interaction feedback for evolution.

    This class:
    - Records user interactions
    - Tracks success/failure signals
    - Captures user corrections
    - Converts to EvolutionTestCase for evolution
    - Persists to disk for durability

    Example:
        collector = FeedbackCollector(storage_path=Path("./feedback"))

        # Record interaction
        id = collector.record_interaction(
            query="Turn on the lights",
            response="I've turned on the living room lights",
            action="home_assistant",
            success=True,
        )

        # Later, record feedback
        collector.record_success(id, quality=0.9)

        # Get cases for evolution
        cases = collector.get_evolution_cases(min_count=50)
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_interactions: int = 10000,
        min_quality_for_positive: float = 0.6,
    ):
        """Initialize the feedback collector.

        Args:
            storage_path: Path to store feedback data
            max_interactions: Maximum interactions to keep in memory
            min_quality_for_positive: Minimum quality score for positive cases
        """
        self._storage_path = storage_path
        self._max_interactions = max_interactions
        self._min_quality_for_positive = min_quality_for_positive

        # In-memory storage
        self._interactions: dict[str, InteractionRecord] = {}
        self._interaction_order: list[str] = []  # For LRU eviction

        # Quick lookup
        self._users: set[str] = set()
        self._sessions: set[str] = set()

        # Load from disk if path exists
        if storage_path and storage_path.exists():
            self._load_from_disk()

    def record_interaction(
        self,
        query: str,
        response: str,
        action: str | None = None,
        action_args: dict[str, Any] | None = None,
        success: bool = True,
        latency_ms: float = 0.0,
        user_id: str = "",
        session_id: str = "",
    ) -> str:
        """Record a user interaction.

        Args:
            query: User's query
            response: Assistant's response
            action: Action taken (if any)
            action_args: Action arguments
            success: Whether the action succeeded
            latency_ms: Response latency
            user_id: User identifier
            session_id: Session identifier

        Returns:
            Interaction ID for later reference
        """
        interaction_id = str(uuid.uuid4())[:8]

        record = InteractionRecord(
            interaction_id=interaction_id,
            query=query,
            response=response,
            action=action,
            action_args=action_args,
            success=success,
            latency_ms=latency_ms,
            user_id=user_id,
            session_id=session_id,
        )

        # Store
        self._interactions[interaction_id] = record
        self._interaction_order.append(interaction_id)

        # Track users/sessions
        if user_id:
            self._users.add(user_id)
        if session_id:
            self._sessions.add(session_id)

        # Evict old interactions if needed
        self._maybe_evict()

        return interaction_id

    def record_success(
        self,
        interaction_id: str | None = None,
        quality: float = 0.8,
    ) -> bool:
        """Record that an interaction was successful.

        Args:
            interaction_id: ID of the interaction (None = most recent)
            quality: Quality score (0-1)

        Returns:
            True if recorded successfully
        """
        if interaction_id is None:
            if not self._interaction_order:
                return False
            interaction_id = self._interaction_order[-1]

        record = self._interactions.get(interaction_id)
        if not record:
            return False

        record.quality_score = quality
        record.success = True

        return True

    def record_correction(
        self,
        interaction_id: str | None,
        correction: str,
    ) -> bool:
        """Record that the user corrected the response.

        Args:
            interaction_id: ID of the interaction (None = most recent)
            correction: What the user said instead

        Returns:
            True if recorded successfully
        """
        if interaction_id is None:
            if not self._interaction_order:
                return False
            interaction_id = self._interaction_order[-1]

        record = self._interactions.get(interaction_id)
        if not record:
            return False

        record.user_correction = correction
        record.was_corrected = True
        record.quality_score = 0.3  # Low quality if corrected

        return True

    def record_failure(
        self,
        interaction_id: str | None = None,
        reason: str = "",
    ) -> bool:
        """Record that an interaction failed.

        Args:
            interaction_id: ID of the interaction (None = most recent)
            reason: Reason for failure

        Returns:
            True if recorded successfully
        """
        if interaction_id is None:
            if not self._interaction_order:
                return False
            interaction_id = self._interaction_order[-1]

        record = self._interactions.get(interaction_id)
        if not record:
            return False

        record.success = False
        record.quality_score = 0.2
        if reason:
            record.user_correction = f"[FAILURE] {reason}"

        return True

    def get_evolution_cases(
        self,
        min_count: int = 0,
        max_count: int | None = None,
        include_failures: bool = True,
        include_corrections: bool = True,
    ) -> list[EvolutionTestCase]:
        """Get test cases for evolution.

        Args:
            min_count: Minimum cases required (returns empty if not met)
            max_count: Maximum cases to return
            include_failures: Include failed interactions
            include_corrections: Include corrected interactions

        Returns:
            List of EvolutionTestCase ready for evolution pipeline
        """
        cases = []

        for record in self._interactions.values():
            # Filter based on options
            if not include_failures and not record.success:
                continue
            if not include_corrections and record.was_corrected:
                continue

            cases.append(record.to_evolution_test_case())

        if len(cases) < min_count:
            logger.info(f"Only {len(cases)} cases, need {min_count} for evolution")
            return []

        # Sort by timestamp (most recent first)
        cases.sort(
            key=lambda c: c.quality_score if c.quality_score else 0.5,
            reverse=True,
        )

        if max_count:
            cases = cases[:max_count]

        return cases

    def get_interaction_feedback(
        self,
        min_count: int = 0,
    ) -> list[InteractionFeedback]:
        """Get interaction feedback for context evolution.

        Args:
            min_count: Minimum feedback required

        Returns:
            List of InteractionFeedback for context evolution
        """
        feedback = [
            record.to_interaction_feedback()
            for record in self._interactions.values()
        ]

        if len(feedback) < min_count:
            return []

        return feedback

    def get_stats(self) -> FeedbackStats:
        """Get statistics about collected feedback."""
        total = len(self._interactions)
        successful = sum(1 for r in self._interactions.values() if r.success)
        corrected = sum(1 for r in self._interactions.values() if r.was_corrected)

        quality_scores = [
            r.quality_score for r in self._interactions.values()
            if r.quality_score is not None
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.0

        latencies = [r.latency_ms for r in self._interactions.values()]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        actions = {}
        for r in self._interactions.values():
            if r.action:
                actions[r.action] = actions.get(r.action, 0) + 1

        return FeedbackStats(
            total_interactions=total,
            successful_interactions=successful,
            corrected_interactions=corrected,
            avg_quality_score=avg_quality,
            avg_latency_ms=avg_latency,
            unique_users=len(self._users),
            unique_sessions=len(self._sessions),
            actions_distribution=actions,
        )

    def clear(self) -> None:
        """Clear all collected feedback."""
        self._interactions.clear()
        self._interaction_order.clear()
        self._users.clear()
        self._sessions.clear()

    def save(self) -> None:
        """Save feedback to disk."""
        if not self._storage_path:
            return

        self._storage_path.mkdir(parents=True, exist_ok=True)
        data_path = self._storage_path / "feedback.json"

        data = {
            "interactions": [
                {
                    **asdict(record),
                    "timestamp": record.timestamp.isoformat(),
                }
                for record in self._interactions.values()
            ],
            "saved_at": datetime.now().isoformat(),
        }

        with open(data_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(self._interactions)} interactions to {data_path}")

    def _load_from_disk(self) -> None:
        """Load feedback from disk."""
        if not self._storage_path:
            return

        data_path = self._storage_path / "feedback.json"
        if not data_path.exists():
            return

        try:
            with open(data_path) as f:
                data = json.load(f)

            for item in data.get("interactions", []):
                record = InteractionRecord(
                    interaction_id=item["interaction_id"],
                    query=item["query"],
                    response=item["response"],
                    action=item.get("action"),
                    action_args=item.get("action_args"),
                    success=item.get("success", True),
                    latency_ms=item.get("latency_ms", 0.0),
                    user_id=item.get("user_id", ""),
                    session_id=item.get("session_id", ""),
                    timestamp=datetime.fromisoformat(item["timestamp"]),
                    quality_score=item.get("quality_score"),
                    user_correction=item.get("user_correction"),
                    was_corrected=item.get("was_corrected", False),
                )
                self._interactions[record.interaction_id] = record
                self._interaction_order.append(record.interaction_id)

                if record.user_id:
                    self._users.add(record.user_id)
                if record.session_id:
                    self._sessions.add(record.session_id)

            logger.info(f"Loaded {len(self._interactions)} interactions from {data_path}")

        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")

    def _maybe_evict(self) -> None:
        """Evict old interactions if over limit."""
        while len(self._interactions) > self._max_interactions:
            if not self._interaction_order:
                break
            oldest_id = self._interaction_order.pop(0)
            if oldest_id in self._interactions:
                del self._interactions[oldest_id]


__all__ = [
    "FeedbackCollector",
    "InteractionRecord",
    "FeedbackStats",
]
