"""Tests for scheduling models."""

import pytest
from datetime import datetime, timedelta

from draagon_ai.scheduling.models import (
    ScheduledTask,
    TaskStatus,
    TaskPriority,
    TriggerType,
)


class TestScheduledTask:
    """Tests for ScheduledTask dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        task = ScheduledTask()

        assert task.task_id.startswith("task_")
        assert task.name == ""
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.trigger_type == TriggerType.ONCE
        assert task.max_retries == 3
        assert task.retry_count == 0
        assert task.execution_count == 0

    def test_to_dict_and_from_dict(self):
        """Test serialization round-trip."""
        task = ScheduledTask(
            task_id="test_123",
            name="Test Timer",
            description="A test timer",
            owner_type="user",
            owner_id="doug",
            trigger_type=TriggerType.CRON,
            trigger_config={"expression": "0 9 * * *"},
            action="notify",
            action_params={"message": "Hello"},
            status=TaskStatus.PENDING,
            priority=TaskPriority.HIGH,
            next_run=datetime(2025, 1, 1, 9, 0, 0),
            max_retries=5,
            tags=["test", "timer"],
            metadata={"source": "unit_test"},
        )

        data = task.to_dict()
        restored = ScheduledTask.from_dict(data)

        assert restored.task_id == task.task_id
        assert restored.name == task.name
        assert restored.description == task.description
        assert restored.owner_type == task.owner_type
        assert restored.owner_id == task.owner_id
        assert restored.trigger_type == task.trigger_type
        assert restored.trigger_config == task.trigger_config
        assert restored.action == task.action
        assert restored.action_params == task.action_params
        assert restored.status == task.status
        assert restored.priority == task.priority
        assert restored.next_run == task.next_run
        assert restored.max_retries == task.max_retries
        assert restored.tags == task.tags
        assert restored.metadata == task.metadata

    def test_is_expired(self):
        """Test expiration check."""
        # Not expired - no expiration set
        task = ScheduledTask()
        assert not task.is_expired()

        # Not expired - future date
        task.expires_at = datetime.now() + timedelta(hours=1)
        assert not task.is_expired()

        # Expired - past date
        task.expires_at = datetime.now() - timedelta(hours=1)
        assert task.is_expired()

    def test_is_due(self):
        """Test due check."""
        # Not due - not pending
        task = ScheduledTask(status=TaskStatus.RUNNING)
        assert not task.is_due()

        # Not due - no next_run
        task = ScheduledTask(status=TaskStatus.PENDING)
        assert not task.is_due()

        # Not due - future next_run
        task = ScheduledTask(
            status=TaskStatus.PENDING,
            next_run=datetime.now() + timedelta(hours=1),
        )
        assert not task.is_due()

        # Due - past next_run
        task = ScheduledTask(
            status=TaskStatus.PENDING,
            next_run=datetime.now() - timedelta(minutes=1),
        )
        assert task.is_due()

    def test_should_retry(self):
        """Test retry logic."""
        task = ScheduledTask(max_retries=3)

        assert task.should_retry()  # retry_count = 0

        task.retry_count = 2
        assert task.should_retry()  # retry_count < max

        task.retry_count = 3
        assert not task.should_retry()  # retry_count >= max

    def test_has_reached_max_executions(self):
        """Test max executions check."""
        # No limit
        task = ScheduledTask(max_executions=None)
        task.execution_count = 1000
        assert not task.has_reached_max_executions()

        # With limit - not reached
        task = ScheduledTask(max_executions=5, execution_count=3)
        assert not task.has_reached_max_executions()

        # With limit - reached
        task = ScheduledTask(max_executions=5, execution_count=5)
        assert task.has_reached_max_executions()

    def test_retry_delay_property(self):
        """Test retry_delay property."""
        task = ScheduledTask()
        task.retry_delay = timedelta(seconds=60)
        assert task.retry_delay_seconds == 60.0
        assert task.retry_delay == timedelta(seconds=60)


class TestTaskStatus:
    """Tests for TaskStatus enum."""

    def test_all_statuses(self):
        """Test all status values exist."""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"
        assert TaskStatus.PAUSED.value == "paused"


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert TaskPriority.CRITICAL.value < TaskPriority.HIGH.value
        assert TaskPriority.HIGH.value < TaskPriority.NORMAL.value
        assert TaskPriority.NORMAL.value < TaskPriority.LOW.value
        assert TaskPriority.LOW.value < TaskPriority.IDLE.value


class TestTriggerType:
    """Tests for TriggerType enum."""

    def test_all_trigger_types(self):
        """Test all trigger types exist."""
        assert TriggerType.ONCE.value == "once"
        assert TriggerType.CRON.value == "cron"
        assert TriggerType.INTERVAL.value == "interval"
        assert TriggerType.EVENT.value == "event"
