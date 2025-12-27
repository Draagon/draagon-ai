"""Feedback Collector for Evolution.

This module collects and manages interaction feedback for evolution.
It tracks user interactions, corrections, and success signals to build
test cases for the evolution pipeline.

Usage:
    from draagon_ai.services.feedback import FeedbackCollector

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

# Re-export from roxy_feedback for backward compatibility
# The implementation is already generic, just named with "Roxy"
from .roxy_feedback import (
    RoxyFeedbackCollector as FeedbackCollector,
    InteractionRecord,
    FeedbackStats,
)

__all__ = [
    "FeedbackCollector",
    "InteractionRecord",
    "FeedbackStats",
]
