"""Type definitions for the autonomous agent extension.

This module contains all the data models used by the autonomous agent,
designed to be application-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# Enums
# =============================================================================


class ActionType(Enum):
    """Types of autonomous actions the agent can take."""

    RESEARCH = "research"
    VERIFY = "verify"
    REFLECT = "reflect"
    NOTE_QUESTION = "note_question"
    PREPARE_SUGGESTION = "prepare_suggestion"
    UPDATE_BELIEF = "update_belief"
    REST = "rest"


class ActionTier(Enum):
    """Risk tiers for autonomous actions.

    Actions are classified by risk level to determine what can run
    autonomously vs what needs user approval.
    """

    TIER_0 = 0  # Always safe: research, reflect, update beliefs
    TIER_1 = 1  # Low risk: prepare suggestions, note questions (log only)
    TIER_2 = 2  # Medium: proactive reminders (notify user)
    TIER_3 = 3  # High: messages, calendar, devices (requires approval)
    TIER_4 = 4  # Forbidden: financial, security, impersonation (never)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class AutonomousConfig:
    """Configuration for the autonomous agent.

    These settings control when and how the agent operates.
    """

    # Enable/disable
    enabled: bool = True

    # Timing
    cycle_interval_minutes: int = 30
    active_hours_start: int = 8  # Don't run before 8am
    active_hours_end: int = 22  # Don't run after 10pm

    # Limits
    max_actions_per_cycle: int = 3
    daily_action_budget: int = 20
    max_consecutive_same_type: int = 2

    # Safety
    require_semantic_safety_check: bool = True
    log_all_proposals: bool = True  # Even blocked ones
    shadow_mode: bool = False  # Log only, don't execute

    # Self-monitoring
    enable_self_monitoring: bool = True
    persist_logs: bool = True  # For dashboard

    # Personality bounds (prevent extreme evolution)
    min_trait_value: float = 0.1
    max_trait_value: float = 0.9
    max_trait_change_per_day: float = 0.1


# =============================================================================
# Action Models
# =============================================================================


@dataclass
class ProposedAction:
    """An action proposed by the autonomous agent."""

    action_type: ActionType
    description: str
    reasoning: str
    risk_tier: ActionTier = ActionTier.TIER_0
    reversible: bool = True
    estimated_time_seconds: int = 30
    target_entity: str | None = None


@dataclass
class HarmCheck:
    """Result of harm assessment for an action."""

    potentially_harmful: bool
    reason: str | None = None
    confidence: float = 0.5


@dataclass
class SafetyCheck:
    """Result of semantic safety check."""

    is_safe: bool
    reason: str | None = None


@dataclass
class ApprovedAction:
    """An action that passed all guardrails."""

    action: ProposedAction
    approved_at: datetime
    guardrails_passed: list[str] = field(default_factory=list)


@dataclass
class ActionResult:
    """Result of executing an autonomous action."""

    success: bool
    outcome: str | None = None
    error: str | None = None
    learned: str | None = None
    belief_updated: bool = False


@dataclass
class ActionLog:
    """Log entry for an autonomous action."""

    action_id: str
    action_type: str
    description: str
    reasoning: str
    started_at: datetime
    completed_at: datetime | None = None
    success: bool = False
    outcome: str | None = None
    error: str | None = None
    blocked: bool = False
    blocked_reason: str | None = None


# =============================================================================
# Context Models
# =============================================================================


@dataclass
class AutonomousContext:
    """Context gathered for autonomous decision making.

    Applications provide this context via adapters to inform
    the agent's decisions.
    """

    # Personality/Self
    personality_context: str = ""
    trait_values: dict[str, float] = field(default_factory=dict)

    # What's happening
    recent_conversations_summary: str = ""
    pending_questions: list[str] = field(default_factory=list)
    unverified_claims: list[str] = field(default_factory=list)
    knowledge_gaps: list[str] = field(default_factory=list)
    conflicting_beliefs: list[str] = field(default_factory=list)
    upcoming_events_summary: str = ""

    # Who's involved
    household_members: list[str] = field(default_factory=list)

    # State
    current_time: datetime = field(default_factory=datetime.now)
    day_of_week: str = ""
    recent_actions: list[str] = field(default_factory=list)
    daily_budget_remaining: int = 20
    available_action_types: list[ActionType] = field(default_factory=list)


@dataclass
class SelfMonitoringFinding:
    """A finding from self-monitoring review."""

    finding_type: str  # unexpected_result, contradiction, pattern, low_value, needs_human
    description: str
    severity: str  # low, medium, high
    action_recommended: str | None = None


@dataclass
class SelfMonitoringResult:
    """Result of self-monitoring review."""

    overall_assessment: str  # good, needs_attention, problematic
    findings: list[SelfMonitoringFinding] = field(default_factory=list)
    notify_user: bool = False
    notification_message: str | None = None
    lessons_learned: list[str] = field(default_factory=list)


# =============================================================================
# Protocols (Interfaces)
# =============================================================================


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers used by the autonomous agent."""

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        """Generate a response from the LLM."""
        ...


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for web search providers."""

    async def search(self, query: str) -> str:
        """Search the web and return results."""
        ...


@runtime_checkable
class MemoryStoreProvider(Protocol):
    """Protocol for storing/retrieving agent state."""

    async def store(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        **metadata: Any,
    ) -> str:
        """Store content and return an ID."""
        ...

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Search stored content."""
        ...

    async def get_logs(
        self,
        record_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Get logs by record type."""
        ...

    async def store_log(
        self,
        log: ActionLog | SelfMonitoringFinding,
    ) -> None:
        """Store an action log or finding."""
        ...


@runtime_checkable
class ContextProvider(Protocol):
    """Protocol for providing autonomous context.

    Applications implement this to provide context for decision making.
    """

    async def gather_context(self) -> AutonomousContext:
        """Gather context for autonomous decision making."""
        ...


@runtime_checkable
class NotificationProvider(Protocol):
    """Protocol for sending notifications."""

    async def queue_notification(
        self,
        message: str,
        priority: str = "medium",
    ) -> None:
        """Queue a notification for the user."""
        ...
