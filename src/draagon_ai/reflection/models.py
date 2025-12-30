"""Data models for the reflection system.

Defines the data structures for discovered issues, their classification,
and tracking through resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class IssueType(str, Enum):
    """Type of issue discovered during reflection."""

    PROMPT = "prompt"          # Can be fixed by changing LLM instructions
    KNOWLEDGE = "knowledge"    # Missing factual information
    TOOL = "tool"              # Need to modify or add a tool
    BUG = "bug"                # Code is broken, needs fixing
    FEATURE = "feature"        # Need new capability
    EXTERNAL = "external"      # Issue outside agent's control


class IssueSeverity(str, Enum):
    """Severity of discovered issue."""

    CRITICAL = "critical"  # Wrong/harmful action taken
    HIGH = "high"          # User had to correct or repeat
    MEDIUM = "medium"      # Suboptimal but worked
    LOW = "low"            # Minor polish issue


class IssueStatus(str, Enum):
    """Status of an issue in its lifecycle."""

    OPEN = "open"              # Newly discovered
    PENDING_FIX = "pending"    # Fix being generated
    FIXED = "fixed"            # Fix applied
    WONT_FIX = "wont_fix"      # Decided not to fix
    DUPLICATE = "duplicate"    # Duplicate of another issue


@dataclass
class ReflectionResult:
    """Result of reflecting on an interaction."""

    quality_score: int  # 1-5
    issues: list[DiscoveredIssue]
    no_issues: bool

    # Metadata
    interaction_id: str
    reflected_at: datetime = field(default_factory=datetime.now)

    # Optional analysis
    user_sentiment: str | None = None  # positive, neutral, negative, frustrated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "quality_score": self.quality_score,
            "issues": [i.to_dict() for i in self.issues],
            "no_issues": self.no_issues,
            "interaction_id": self.interaction_id,
            "reflected_at": self.reflected_at.isoformat(),
            "user_sentiment": self.user_sentiment,
        }

    @property
    def is_good(self) -> bool:
        """Check if the interaction was good quality."""
        return self.quality_score >= 4 and self.no_issues


@dataclass
class DiscoveredIssue:
    """An issue discovered during post-interaction reflection."""

    id: str
    timestamp: datetime

    # Source interaction
    interaction_id: str
    query: str
    response: str
    action_taken: str
    tool_calls: list[str]

    # Issue details from reflection
    description: str
    root_cause: str
    severity: IssueSeverity
    suggested_fix: str

    # Optional fields
    prompt_blamed: str | None = None
    conversation_context: str | None = None
    user_sentiment: str | None = None

    # Classification
    issue_type: IssueType = IssueType.PROMPT
    fixable_by_agent: bool = True

    # For code issues
    suggested_approach: str | None = None
    files_involved: list[str] = field(default_factory=list)
    code_snippet: str | None = None

    # Clustering
    cluster_id: str | None = None
    related_issue_ids: list[str] = field(default_factory=list)
    occurrence_count: int = 1

    # Resolution tracking
    status: IssueStatus = IssueStatus.OPEN
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    resolution_notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "interaction_id": self.interaction_id,
            "query": self.query,
            "response": self.response,
            "action_taken": self.action_taken,
            "tool_calls": self.tool_calls,
            "conversation_context": self.conversation_context,
            "user_sentiment": self.user_sentiment,
            "description": self.description,
            "prompt_blamed": self.prompt_blamed,
            "root_cause": self.root_cause,
            "severity": self.severity.value,
            "suggested_fix": self.suggested_fix,
            "issue_type": self.issue_type.value,
            "fixable_by_agent": self.fixable_by_agent,
            "suggested_approach": self.suggested_approach,
            "files_involved": self.files_involved,
            "code_snippet": self.code_snippet,
            "cluster_id": self.cluster_id,
            "related_issue_ids": self.related_issue_ids,
            "occurrence_count": self.occurrence_count,
            "status": self.status.value,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution_notes": self.resolution_notes,
            "record_type": "discovered_issue",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DiscoveredIssue:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            interaction_id=data["interaction_id"],
            query=data["query"],
            response=data["response"],
            action_taken=data.get("action_taken", "unknown"),
            tool_calls=data.get("tool_calls", []),
            conversation_context=data.get("conversation_context"),
            user_sentiment=data.get("user_sentiment"),
            description=data["description"],
            prompt_blamed=data.get("prompt_blamed"),
            root_cause=data["root_cause"],
            severity=IssueSeverity(data["severity"]),
            suggested_fix=data["suggested_fix"],
            issue_type=IssueType(data.get("issue_type", "prompt")),
            fixable_by_agent=data.get("fixable_by_agent", True),
            suggested_approach=data.get("suggested_approach"),
            files_involved=data.get("files_involved", []),
            code_snippet=data.get("code_snippet"),
            cluster_id=data.get("cluster_id"),
            related_issue_ids=data.get("related_issue_ids", []),
            occurrence_count=data.get("occurrence_count", 1),
            status=IssueStatus(data.get("status", "open")),
            resolved_at=datetime.fromisoformat(data["resolved_at"]) if data.get("resolved_at") else None,
            resolved_by=data.get("resolved_by"),
            resolution_notes=data.get("resolution_notes"),
        )


@dataclass
class IssueCluster:
    """A cluster of related issues."""

    id: str
    issues: list[DiscoveredIssue]
    pattern_description: str
    created_at: datetime = field(default_factory=datetime.now)

    @property
    def severity(self) -> IssueSeverity:
        """Return the highest severity in the cluster."""
        severities = [IssueSeverity.CRITICAL, IssueSeverity.HIGH, IssueSeverity.MEDIUM, IssueSeverity.LOW]
        for sev in severities:
            if any(i.severity == sev for i in self.issues):
                return sev
        return IssueSeverity.LOW

    @property
    def count(self) -> int:
        return len(self.issues)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "issues": [i.to_dict() for i in self.issues],
            "pattern_description": self.pattern_description,
            "created_at": self.created_at.isoformat(),
            "severity": self.severity.value,
            "count": self.count,
        }


@dataclass
class ImprovementPlan:
    """Plan for improving a prompt based on discovered issues."""

    prompt_name: str
    issues_addressed: list[str]  # Issue IDs
    analysis: str
    changes: list[dict[str, str]]  # {"what": ..., "why": ..., "before": ..., "after": ...}
    new_prompt: str
    confidence: float
    risk_assessment: str
    test_cases_to_verify: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_name": self.prompt_name,
            "issues_addressed": self.issues_addressed,
            "analysis": self.analysis,
            "changes": self.changes,
            "new_prompt": self.new_prompt,
            "confidence": self.confidence,
            "risk_assessment": self.risk_assessment,
            "test_cases_to_verify": self.test_cases_to_verify,
        }


@dataclass
class FailedImprovement:
    """Record of a failed improvement attempt."""

    id: str
    prompt_name: str
    timestamp: datetime

    # What was attempted
    changes_attempted: list[dict[str, str]]
    new_prompt_hash: str
    issues_targeted: list[str]

    # Why it failed
    failure_reason: str
    failure_details: list[str]
    analysis: str

    # Metadata
    attempt_number: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "prompt_name": self.prompt_name,
            "timestamp": self.timestamp.isoformat(),
            "changes_attempted": self.changes_attempted,
            "new_prompt_hash": self.new_prompt_hash,
            "issues_targeted": self.issues_targeted,
            "failure_reason": self.failure_reason,
            "failure_details": self.failure_details,
            "analysis": self.analysis,
            "attempt_number": self.attempt_number,
            "record_type": "failed_improvement",
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FailedImprovement:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            prompt_name=data["prompt_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"],
            changes_attempted=data.get("changes_attempted", []),
            new_prompt_hash=data.get("new_prompt_hash", ""),
            issues_targeted=data.get("issues_targeted", []),
            failure_reason=data.get("failure_reason", "unknown"),
            failure_details=data.get("failure_details", []),
            analysis=data.get("analysis", ""),
            attempt_number=data.get("attempt_number", 1),
        )

    def format_for_llm(self) -> str:
        """Format this failure for LLM context."""
        changes_text = "\n".join([
            f"  - {c.get('what', 'Unknown change')}: {c.get('why', 'No reason given')}"
            for c in self.changes_attempted[:3]
        ])

        return f"""### Failed Attempt #{self.attempt_number}
**Changes tried:**
{changes_text}

**Why it failed:** {self.failure_reason}
**Details:** {'; '.join(self.failure_details[:3])}

**DO NOT** repeat these same changes."""
