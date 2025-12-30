"""Credibility and trust tracking for multi-user systems.

This module provides a multi-dimensional credibility tracking system
that enables nuanced understanding of each user's reliability.
"""

from .models import (
    UserIntent,
    DomainExpertise,
    ConfidenceCalibration,
    LearningTrajectory,
    InformationQuality,
)
from .credibility import UserCredibility
from .thresholds import (
    get_verification_threshold,
    VerificationLevel,
    VERIFICATION_THRESHOLDS,
)

__all__ = [
    # Models
    "UserIntent",
    "DomainExpertise",
    "ConfidenceCalibration",
    "LearningTrajectory",
    "InformationQuality",
    # Credibility
    "UserCredibility",
    # Thresholds
    "get_verification_threshold",
    "VerificationLevel",
    "VERIFICATION_THRESHOLDS",
]
