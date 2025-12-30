"""Verification thresholds based on credibility.

This module provides adaptive verification thresholds that adjust
based on user credibility scores.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class VerificationLevel(str, Enum):
    """Levels of verification required for operations.

    Higher levels require more proof/confirmation before proceeding.
    """

    NONE = "none"              # Trust completely, no verification
    MINIMAL = "minimal"        # Basic sanity check
    STANDARD = "standard"      # Normal verification
    ELEVATED = "elevated"      # Extra scrutiny
    MAXIMUM = "maximum"        # Full verification required


# Default thresholds for operations
# Higher credibility = lower threshold needed
VERIFICATION_THRESHOLDS = {
    "delete_memory": {
        "description": "Deleting stored memories",
        "base_threshold": 0.7,  # Credibility needed for minimal verification
        "sensitive": True,
    },
    "update_fact": {
        "description": "Updating stored facts",
        "base_threshold": 0.5,
        "sensitive": False,
    },
    "financial": {
        "description": "Financial operations",
        "base_threshold": 0.9,
        "sensitive": True,
    },
    "security": {
        "description": "Security-related operations",
        "base_threshold": 0.85,
        "sensitive": True,
    },
    "normal": {
        "description": "Normal operations",
        "base_threshold": 0.4,
        "sensitive": False,
    },
    "preference": {
        "description": "User preferences",
        "base_threshold": 0.3,  # Trust preferences more easily
        "sensitive": False,
    },
}


def get_verification_threshold(
    operation: str,
    credibility_score: float,
    domain_accuracy: float | None = None,
) -> VerificationLevel:
    """Determine verification level needed for an operation.

    The verification level adapts based on:
    - Operation type (some require more scrutiny)
    - User's overall credibility score
    - User's accuracy in the relevant domain (if applicable)

    Args:
        operation: Operation type (key in VERIFICATION_THRESHOLDS)
        credibility_score: User's composite credibility score (0.0-1.0)
        domain_accuracy: Optional domain-specific accuracy (0.0-1.0)

    Returns:
        Required verification level

    Example:
        level = get_verification_threshold("delete_memory", 0.85)
        if level == VerificationLevel.NONE:
            # Proceed without verification
        elif level == VerificationLevel.STANDARD:
            # Ask for confirmation
    """
    # Get operation config, default to normal
    config = VERIFICATION_THRESHOLDS.get(operation, VERIFICATION_THRESHOLDS["normal"])
    base_threshold = config["base_threshold"]
    is_sensitive = config.get("sensitive", False)

    # Effective credibility considers domain expertise if available
    effective_credibility = credibility_score
    if domain_accuracy is not None:
        # Blend overall and domain-specific (domain weighs more for domain ops)
        effective_credibility = (credibility_score * 0.4 + domain_accuracy * 0.6)

    # Determine level based on how much credibility exceeds threshold
    gap = effective_credibility - base_threshold

    if gap >= 0.3:
        # Way above threshold
        return VerificationLevel.NONE
    elif gap >= 0.15:
        # Comfortably above threshold
        return VerificationLevel.MINIMAL
    elif gap >= 0:
        # At or slightly above threshold
        return VerificationLevel.STANDARD
    elif gap >= -0.15:
        # Below threshold but close
        return VerificationLevel.ELEVATED
    else:
        # Well below threshold
        return VerificationLevel.MAXIMUM


def should_verify(
    credibility_score: float,
    operation: str = "normal",
    domain_accuracy: float | None = None,
) -> bool:
    """Quick check if verification is needed.

    Args:
        credibility_score: User's composite credibility score
        operation: Operation type
        domain_accuracy: Optional domain-specific accuracy

    Returns:
        True if verification should be performed
    """
    level = get_verification_threshold(operation, credibility_score, domain_accuracy)
    return level not in (VerificationLevel.NONE, VerificationLevel.MINIMAL)


def get_credibility_tier(credibility_score: float) -> str:
    """Get human-readable credibility tier.

    Args:
        credibility_score: Composite credibility score (0.0-1.0)

    Returns:
        Tier name: "high", "normal", "low", or "very_low"
    """
    if credibility_score >= 0.85:
        return "high"
    elif credibility_score >= 0.6:
        return "normal"
    elif credibility_score >= 0.4:
        return "low"
    else:
        return "very_low"
