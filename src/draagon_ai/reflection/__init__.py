"""Reflection module for draagon-ai.

This module provides post-interaction quality evaluation and self-improvement
capabilities for AI agents.

Features:
- **ReflectionService**: Evaluates interaction quality and discovers issues
- **Issue Classification**: Categorizes issues by type (prompt, knowledge, tool, bug)
- **Severity Levels**: Critical, high, medium, low for prioritization
- **User Feedback Analysis**: Detects corrections, frustration, satisfaction
- **Improvement Planning**: Generates plans to fix discovered issues

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Reflection Pipeline                          │
    │                                                                  │
    │  Interaction → [Quality Eval] → [Issue Discovery] → [Classify] │
    │                     (1-5)         (user feedback)    (type/sev) │
    │                                                                  │
    │  Issues → [Cluster] → [Prioritize] → [Plan Fix] → [Track]      │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.reflection import (
        ReflectionService,
        ReflectionResult,
        DiscoveredIssue,
        IssueType,
        IssueSeverity,
    )

    # Create reflection service
    service = ReflectionService(llm_provider)

    # Reflect on an interaction
    result = await service.reflect(
        interaction_id="12345",
        query="Turn off the lights",
        response="I turned off the lights.",
        action="call_service",
        tool_calls=["home_assistant.call_service"],
        conversation_history=[...],
    )

    if not result.no_issues:
        for issue in result.issues:
            print(f"{issue.severity}: {issue.description}")
"""

from .models import (
    IssueType,
    IssueSeverity,
    IssueStatus,
    DiscoveredIssue,
    ReflectionResult,
    IssueCluster,
    ImprovementPlan,
    FailedImprovement,
)
from .service import (
    ReflectionService,
    ReflectionConfig,
)
from .protocols import (
    LLMProvider,
    IssueStore,
)

__all__ = [
    # Models
    "IssueType",
    "IssueSeverity",
    "IssueStatus",
    "DiscoveredIssue",
    "ReflectionResult",
    "IssueCluster",
    "ImprovementPlan",
    "FailedImprovement",
    # Service
    "ReflectionService",
    "ReflectionConfig",
    # Protocols
    "LLMProvider",
    "IssueStore",
]
