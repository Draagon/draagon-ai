"""Tests for task persistence."""

import pytest
from datetime import datetime, timedelta

from draagon_ai.scheduling.models import (
    ScheduledTask,
    TaskStatus,
    TaskPriority,
    TriggerType,
)
from draagon_ai.scheduling.persistence import InMemoryPersistence


class TestInMemoryPersistence:
    """Tests for InMemoryPersistence."""

    @pytest.fixture
    def persistence(self):
        """Create a fresh persistence instance."""
        return InMemoryPersistence()

    @pytest.mark.asyncio
    async def test_save_and_get(self, persistence):
        """Test saving and retrieving a task."""
        task = ScheduledTask(
            task_id="test_1",
            name="Test Task",
            action="notify",
        )

        await persistence.save(task)
        retrieved = await persistence.get("test_1")

        assert retrieved is not None
        assert retrieved.task_id == "test_1"
        assert retrieved.name == "Test Task"

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, persistence):
        """Test getting a nonexistent task returns None."""
        result = await persistence.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, persistence):
        """Test deleting a task."""
        task = ScheduledTask(task_id="test_1")
        await persistence.save(task)

        result = await persistence.delete("test_1")
        assert result is True

        retrieved = await persistence.get("test_1")
        assert retrieved is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, persistence):
        """Test deleting nonexistent task returns False."""
        result = await persistence.delete("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_list_tasks_no_filter(self, persistence):
        """Test listing all tasks."""
        for i in range(5):
            await persistence.save(ScheduledTask(task_id=f"task_{i}"))

        tasks = await persistence.list_tasks()
        assert len(tasks) == 5

    @pytest.mark.asyncio
    async def test_list_tasks_by_owner(self, persistence):
        """Test filtering by owner_id."""
        await persistence.save(ScheduledTask(task_id="t1", owner_id="alice"))
        await persistence.save(ScheduledTask(task_id="t2", owner_id="bob"))
        await persistence.save(ScheduledTask(task_id="t3", owner_id="alice"))

        tasks = await persistence.list_tasks(owner_id="alice")
        assert len(tasks) == 2
        assert all(t.owner_id == "alice" for t in tasks)

    @pytest.mark.asyncio
    async def test_list_tasks_by_status(self, persistence):
        """Test filtering by status."""
        await persistence.save(
            ScheduledTask(task_id="t1", status=TaskStatus.PENDING)
        )
        await persistence.save(
            ScheduledTask(task_id="t2", status=TaskStatus.RUNNING)
        )
        await persistence.save(
            ScheduledTask(task_id="t3", status=TaskStatus.PENDING)
        )

        tasks = await persistence.list_tasks(status=TaskStatus.PENDING)
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_tasks_by_trigger_type(self, persistence):
        """Test filtering by trigger type."""
        await persistence.save(
            ScheduledTask(task_id="t1", trigger_type=TriggerType.ONCE)
        )
        await persistence.save(
            ScheduledTask(task_id="t2", trigger_type=TriggerType.CRON)
        )
        await persistence.save(
            ScheduledTask(task_id="t3", trigger_type=TriggerType.CRON)
        )

        tasks = await persistence.list_tasks(trigger_type=TriggerType.CRON)
        assert len(tasks) == 2

    @pytest.mark.asyncio
    async def test_list_tasks_by_tags(self, persistence):
        """Test filtering by tags."""
        await persistence.save(
            ScheduledTask(task_id="t1", tags=["timer", "kitchen"])
        )
        await persistence.save(
            ScheduledTask(task_id="t2", tags=["timer"])
        )
        await persistence.save(
            ScheduledTask(task_id="t3", tags=["briefing"])
        )

        # Single tag match
        tasks = await persistence.list_tasks(tags=["timer"])
        assert len(tasks) == 2

        # Multiple tag match (AND)
        tasks = await persistence.list_tasks(tags=["timer", "kitchen"])
        assert len(tasks) == 1
        assert tasks[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_list_tasks_limit(self, persistence):
        """Test limit parameter."""
        for i in range(10):
            await persistence.save(ScheduledTask(task_id=f"task_{i}"))

        tasks = await persistence.list_tasks(limit=3)
        assert len(tasks) == 3

    @pytest.mark.asyncio
    async def test_get_pending(self, persistence):
        """Test getting pending tasks."""
        await persistence.save(
            ScheduledTask(task_id="t1", status=TaskStatus.PENDING)
        )
        await persistence.save(
            ScheduledTask(task_id="t2", status=TaskStatus.RUNNING)
        )
        await persistence.save(
            ScheduledTask(task_id="t3", status=TaskStatus.PENDING)
        )

        pending = await persistence.get_pending()
        assert len(pending) == 2

    @pytest.mark.asyncio
    async def test_get_pending_by_trigger_type(self, persistence):
        """Test getting pending tasks filtered by trigger type."""
        await persistence.save(
            ScheduledTask(
                task_id="t1",
                status=TaskStatus.PENDING,
                trigger_type=TriggerType.ONCE,
            )
        )
        await persistence.save(
            ScheduledTask(
                task_id="t2",
                status=TaskStatus.PENDING,
                trigger_type=TriggerType.CRON,
            )
        )

        pending = await persistence.get_pending(trigger_type=TriggerType.ONCE)
        assert len(pending) == 1
        assert pending[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_get_due_tasks(self, persistence):
        """Test getting due tasks."""
        now = datetime.now()

        # Due task
        await persistence.save(
            ScheduledTask(
                task_id="t1",
                status=TaskStatus.PENDING,
                next_run=now - timedelta(minutes=5),
            )
        )
        # Future task
        await persistence.save(
            ScheduledTask(
                task_id="t2",
                status=TaskStatus.PENDING,
                next_run=now + timedelta(hours=1),
            )
        )
        # Due but not pending
        await persistence.save(
            ScheduledTask(
                task_id="t3",
                status=TaskStatus.RUNNING,
                next_run=now - timedelta(minutes=5),
            )
        )

        due = await persistence.get_due_tasks(before=now)
        assert len(due) == 1
        assert due[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_get_due_tasks_sorted_by_priority(self, persistence):
        """Test due tasks are sorted by priority."""
        now = datetime.now()
        past = now - timedelta(minutes=5)

        await persistence.save(
            ScheduledTask(
                task_id="low",
                status=TaskStatus.PENDING,
                priority=TaskPriority.LOW,
                next_run=past,
            )
        )
        await persistence.save(
            ScheduledTask(
                task_id="high",
                status=TaskStatus.PENDING,
                priority=TaskPriority.HIGH,
                next_run=past,
            )
        )
        await persistence.save(
            ScheduledTask(
                task_id="critical",
                status=TaskStatus.PENDING,
                priority=TaskPriority.CRITICAL,
                next_run=past,
            )
        )

        due = await persistence.get_due_tasks(before=now)
        assert len(due) == 3
        assert due[0].task_id == "critical"
        assert due[1].task_id == "high"
        assert due[2].task_id == "low"

    @pytest.mark.asyncio
    async def test_get_by_event(self, persistence):
        """Test getting tasks by event type."""
        await persistence.save(
            ScheduledTask(
                task_id="t1",
                status=TaskStatus.PENDING,
                trigger_type=TriggerType.EVENT,
                trigger_config={"event_type": "conversation_end"},
            )
        )
        await persistence.save(
            ScheduledTask(
                task_id="t2",
                status=TaskStatus.PENDING,
                trigger_type=TriggerType.EVENT,
                trigger_config={"event_type": "memory_stored"},
            )
        )
        await persistence.save(
            ScheduledTask(
                task_id="t3",
                status=TaskStatus.PENDING,
                trigger_type=TriggerType.CRON,  # Not an event trigger
            )
        )

        tasks = await persistence.get_by_event("conversation_end")
        assert len(tasks) == 1
        assert tasks[0].task_id == "t1"

    @pytest.mark.asyncio
    async def test_clear(self, persistence):
        """Test clearing all tasks."""
        for i in range(5):
            await persistence.save(ScheduledTask(task_id=f"task_{i}"))

        persistence.clear()

        tasks = await persistence.list_tasks()
        assert len(tasks) == 0
