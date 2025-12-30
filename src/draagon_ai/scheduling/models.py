"""Scheduling system models and data classes."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Awaitable
import uuid


class TaskStatus(Enum):
    """Status of a scheduled task."""

    PENDING = "pending"  # Waiting for trigger
    RUNNING = "running"  # Currently executing
    COMPLETED = "completed"  # Successfully finished
    FAILED = "failed"  # Failed (exceeded retries)
    CANCELLED = "cancelled"  # User cancelled
    PAUSED = "paused"  # Temporarily paused


class TaskPriority(Enum):
    """Priority levels for task execution."""

    CRITICAL = 0  # Execute immediately (alarms, safety)
    HIGH = 1  # Execute soon (user-initiated timers)
    NORMAL = 2  # Standard priority
    LOW = 3  # Background/housekeeping
    IDLE = 4  # Only when system is idle


class TriggerType(Enum):
    """Type of trigger for a scheduled task."""

    ONCE = "once"  # One-shot timer
    CRON = "cron"  # Cron expression
    INTERVAL = "interval"  # Every N seconds/minutes/hours
    EVENT = "event"  # Triggered by event


@dataclass
class ScheduledTask:
    """A scheduled task in the system.

    This is the unified model for all types of scheduled tasks.
    Features don't need to care HOW something is scheduled, just THAT it is.
    """

    # Identity
    task_id: str = field(default_factory=lambda: f"task_{uuid.uuid4().hex[:12]}")
    name: str = ""
    description: str = ""

    # Ownership (for multi-user/multi-feature)
    owner_type: str = "system"  # "system", "user", "feature", "behavior"
    owner_id: str = "default"  # User ID, feature name, etc.

    # Trigger configuration
    trigger_type: TriggerType = TriggerType.ONCE
    trigger_config: dict = field(default_factory=dict)
    # For ONCE: {"at": datetime_iso}
    # For CRON: {"expression": "0 9 * * *", "timezone": "America/New_York"}
    # For INTERVAL: {"seconds": 3600, "start_immediately": True}
    # For EVENT: {"event_type": "conversation_end", "filter": {...}}

    # Execution
    action: str = ""  # Action identifier (e.g., "notify", "run_query")
    action_params: dict = field(default_factory=dict)
    callback: Callable[..., Awaitable[Any]] | None = field(default=None, repr=False)

    # State
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL

    # Timing
    created_at: datetime = field(default_factory=datetime.now)
    next_run: datetime | None = None
    last_run: datetime | None = None
    expires_at: datetime | None = None  # Auto-delete after this time

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_delay_seconds: float = 30.0

    # Execution tracking
    execution_count: int = 0
    max_executions: int | None = None  # None = unlimited for recurring
    last_result: Any = field(default=None, repr=False)
    last_error: str | None = None

    # Tags for filtering
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for persistence."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "owner_type": self.owner_type,
            "owner_id": self.owner_id,
            "trigger_type": self.trigger_type.value,
            "trigger_config": self.trigger_config,
            "action": self.action,
            "action_params": self.action_params,
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at.isoformat(),
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_retries": self.max_retries,
            "retry_count": self.retry_count,
            "retry_delay_seconds": self.retry_delay_seconds,
            "execution_count": self.execution_count,
            "max_executions": self.max_executions,
            "last_error": self.last_error,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ScheduledTask":
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", f"task_{uuid.uuid4().hex[:12]}"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            owner_type=data.get("owner_type", "system"),
            owner_id=data.get("owner_id", "default"),
            trigger_type=TriggerType(data.get("trigger_type", "once")),
            trigger_config=data.get("trigger_config", {}),
            action=data.get("action", ""),
            action_params=data.get("action_params", {}),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 2)),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at")
            else datetime.now(),
            next_run=datetime.fromisoformat(data["next_run"])
            if data.get("next_run")
            else None,
            last_run=datetime.fromisoformat(data["last_run"])
            if data.get("last_run")
            else None,
            expires_at=datetime.fromisoformat(data["expires_at"])
            if data.get("expires_at")
            else None,
            max_retries=data.get("max_retries", 3),
            retry_count=data.get("retry_count", 0),
            retry_delay_seconds=data.get("retry_delay_seconds", 30.0),
            execution_count=data.get("execution_count", 0),
            max_executions=data.get("max_executions"),
            last_error=data.get("last_error"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    @property
    def retry_delay(self) -> timedelta:
        """Get retry delay as timedelta."""
        return timedelta(seconds=self.retry_delay_seconds)

    @retry_delay.setter
    def retry_delay(self, value: timedelta) -> None:
        """Set retry delay from timedelta."""
        self.retry_delay_seconds = value.total_seconds()

    def is_expired(self) -> bool:
        """Check if task has expired."""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def is_due(self, as_of: datetime | None = None) -> bool:
        """Check if task is due to run."""
        if self.status != TaskStatus.PENDING:
            return False
        if self.next_run is None:
            return False
        now = as_of or datetime.now()
        return self.next_run <= now

    def should_retry(self) -> bool:
        """Check if task should be retried after failure."""
        return self.retry_count < self.max_retries

    def has_reached_max_executions(self) -> bool:
        """Check if task has reached max executions limit."""
        if self.max_executions is None:
            return False
        return self.execution_count >= self.max_executions
