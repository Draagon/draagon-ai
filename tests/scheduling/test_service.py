"""Tests for SchedulingService facade."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from draagon_ai.scheduling.models import (
    ScheduledTask,
    TaskStatus,
    TaskPriority,
    TriggerType,
)
from draagon_ai.scheduling.persistence import InMemoryPersistence
from draagon_ai.scheduling.service import SchedulingService


class TestSchedulingService:
    """Tests for SchedulingService."""

    @pytest.fixture
    def persistence(self):
        """Create a fresh persistence instance."""
        return InMemoryPersistence()

    @pytest.fixture
    def service(self, persistence):
        """Create a scheduling service."""
        return SchedulingService(
            persistence=persistence,
            cron_check_interval=1,  # Fast for testing
            interval_check_interval=1,
        )

    @pytest.mark.asyncio
    async def test_start_stop(self, service):
        """Test starting and stopping the service."""
        assert not service.is_running

        await service.start()
        assert service.is_running

        await service.stop()
        assert not service.is_running

    @pytest.mark.asyncio
    async def test_set_timer(self, service):
        """Test setting a one-shot timer."""
        task_id = await service.set_timer(
            name="Test Timer",
            duration=timedelta(minutes=5),
            action="notify",
            action_params={"message": "Hello"},
            owner_id="alice",
        )

        assert task_id is not None

        task = await service.get_task(task_id)
        assert task is not None
        assert task.name == "Test Timer"
        assert task.trigger_type == TriggerType.ONCE
        assert task.owner_id == "alice"

    @pytest.mark.asyncio
    async def test_set_timer_at(self, service):
        """Test setting a timer for specific time."""
        target = datetime.now() + timedelta(hours=1)

        task_id = await service.set_timer_at(
            name="Scheduled Timer",
            at=target,
            action="notify",
        )

        task = await service.get_task(task_id)
        assert task is not None
        assert abs((task.next_run - target).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_timer_fires_callback(self, service):
        """Test that timer fires and executes callback."""
        fired = asyncio.Event()
        result_holder = {}

        async def my_action(params, event_data):
            result_holder["params"] = params
            fired.set()
            return {"success": True}

        service.register_action("my_action", my_action)
        await service.start()

        try:
            await service.set_timer(
                name="Quick Timer",
                duration=timedelta(milliseconds=100),
                action="my_action",
                action_params={"key": "value"},
            )

            await asyncio.wait_for(fired.wait(), timeout=2.0)
            assert result_holder["params"] == {"key": "value"}
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_schedule_cron(self, service, persistence):
        """Test scheduling a cron task."""
        task_id = await service.schedule_cron(
            name="Daily Briefing",
            expression="0 9 * * *",
            action="briefing",
            owner_id="alice",
            timezone="America/New_York",
        )

        task = await service.get_task(task_id)
        assert task is not None
        assert task.name == "Daily Briefing"
        assert task.trigger_type == TriggerType.CRON
        assert task.trigger_config["expression"] == "0 9 * * *"
        assert task.next_run is not None

    @pytest.mark.asyncio
    async def test_schedule_interval(self, service):
        """Test scheduling an interval task."""
        task_id = await service.schedule_interval(
            name="Health Check",
            interval=timedelta(hours=1),
            action="health_check",
            start_immediately=False,
        )

        task = await service.get_task(task_id)
        assert task is not None
        assert task.trigger_type == TriggerType.INTERVAL
        assert task.trigger_config["seconds"] == 3600

    @pytest.mark.asyncio
    async def test_on_event(self, service):
        """Test scheduling an event-triggered task."""
        task_id = await service.on_event(
            name="Summarize",
            event_type="conversation_end",
            action="summarize",
        )

        task = await service.get_task(task_id)
        assert task is not None
        assert task.trigger_type == TriggerType.EVENT
        assert task.trigger_config["event_type"] == "conversation_end"

    @pytest.mark.asyncio
    async def test_emit_event_triggers_task(self, service):
        """Test that emitting an event triggers registered tasks."""
        fired = asyncio.Event()

        async def my_handler(params, event_data):
            fired.set()
            return event_data

        service.register_action("my_handler", my_handler)
        await service.start()

        try:
            await service.on_event(
                name="Test Event Handler",
                event_type="test_event",
                action="my_handler",
            )

            count = await service.emit_event("test_event", {"key": "value"})
            assert count == 1

            await asyncio.wait_for(fired.wait(), timeout=2.0)
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_cancel_timer(self, service):
        """Test cancelling a timer."""
        task_id = await service.set_timer(
            name="Cancel Me",
            duration=timedelta(minutes=5),
            action="notify",
        )

        result = await service.cancel(task_id)
        assert result is True

        task = await service.get_task(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_pause_resume_cron(self, service):
        """Test pausing and resuming a cron task."""
        task_id = await service.schedule_cron(
            name="Pausable",
            expression="0 9 * * *",
            action="notify",
        )

        # Pause
        result = await service.pause(task_id)
        assert result is True

        task = await service.get_task(task_id)
        assert task.status == TaskStatus.PAUSED

        # Resume
        result = await service.resume(task_id)
        assert result is True

        task = await service.get_task(task_id)
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_delete_task(self, service):
        """Test deleting a task."""
        task_id = await service.set_timer(
            name="Delete Me",
            duration=timedelta(minutes=5),
            action="notify",
        )

        result = await service.delete(task_id)
        assert result is True

        task = await service.get_task(task_id)
        assert task is None

    @pytest.mark.asyncio
    async def test_list_tasks(self, service):
        """Test listing tasks with filters."""
        await service.set_timer(
            name="Timer 1",
            duration=timedelta(minutes=5),
            action="notify",
            owner_id="alice",
        )
        await service.set_timer(
            name="Timer 2",
            duration=timedelta(minutes=5),
            action="notify",
            owner_id="bob",
        )
        await service.schedule_cron(
            name="Cron 1",
            expression="0 9 * * *",
            action="notify",
            owner_id="alice",
        )

        # All tasks
        all_tasks = await service.list_tasks()
        assert len(all_tasks) == 3

        # By owner
        alice_tasks = await service.list_tasks(owner_id="alice")
        assert len(alice_tasks) == 2

        # By trigger type
        timer_tasks = await service.list_tasks(trigger_type=TriggerType.ONCE)
        assert len(timer_tasks) == 2

    @pytest.mark.asyncio
    async def test_register_action(self, service):
        """Test registering a custom action."""
        async def my_action(params, event_data):
            return {"success": True}

        service.register_action("my_feature.process", my_action)

        actions = service.list_actions()
        assert "my_feature.process" in actions

    @pytest.mark.asyncio
    async def test_unregister_action(self, service):
        """Test unregistering an action."""
        async def my_action(params, event_data):
            return {}

        service.register_action("temp_action", my_action)
        assert "temp_action" in service.list_actions()

        result = service.unregister_action("temp_action")
        assert result is True
        assert "temp_action" not in service.list_actions()

    @pytest.mark.asyncio
    async def test_builtin_actions_registered(self, service):
        """Test that built-in actions are registered."""
        actions = service.list_actions()
        assert "notify" in actions
        assert "log" in actions

    @pytest.mark.asyncio
    async def test_execution_hooks(self, service):
        """Test execution hooks are called."""
        executed = []
        succeeded = []
        failed = []

        async def on_execute(task):
            executed.append(task.task_id)

        async def on_success(task, result):
            succeeded.append(task.task_id)

        async def on_failure(task, error):
            failed.append(task.task_id)

        service.set_execution_hooks(
            on_execute=on_execute,
            on_success=on_success,
            on_failure=on_failure,
        )

        await service.start()

        try:
            # Use built-in notify action which should succeed
            await service.set_timer(
                name="Hook Test",
                duration=timedelta(milliseconds=100),
                action="notify",
                action_params={"message": "test"},
            )

            # Wait for execution
            await asyncio.sleep(0.5)

            assert len(executed) == 1
            assert len(succeeded) == 1
            assert len(failed) == 0
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_get_timer_remaining(self, service):
        """Test getting remaining time for a timer."""
        task_id = await service.set_timer(
            name="Check Remaining",
            duration=timedelta(seconds=60),
            action="notify",
        )

        remaining = await service.get_timer_remaining(task_id)
        assert remaining is not None
        assert remaining > timedelta(seconds=55)

    @pytest.mark.asyncio
    async def test_list_timers(self, service):
        """Test listing active timers."""
        for i in range(3):
            await service.set_timer(
                name=f"Timer {i}",
                duration=timedelta(minutes=5),
                action="notify",
                owner_id="alice",
            )

        timers = await service.list_timers(owner_id="alice")
        assert len(timers) == 3

    @pytest.mark.asyncio
    async def test_list_event_types(self, service):
        """Test listing built-in event types."""
        event_types = service.list_event_types()

        assert "conversation_end" in event_types
        assert "memory_stored" in event_types
        assert isinstance(event_types["conversation_end"], str)

    @pytest.mark.asyncio
    async def test_properties(self, service):
        """Test service properties."""
        assert service.persistence is not None
        assert service.executor is not None
        assert service.timer_engine is not None
        assert service.cron_scheduler is not None
        assert service.interval_scheduler is not None
        assert service.event_scheduler is not None

    @pytest.mark.asyncio
    async def test_tags_and_metadata(self, service):
        """Test tasks with tags and metadata."""
        task_id = await service.set_timer(
            name="Tagged Timer",
            duration=timedelta(minutes=5),
            action="notify",
            tags=["important", "kitchen"],
            metadata={"source": "voice_command"},
        )

        task = await service.get_task(task_id)
        assert "important" in task.tags
        assert "kitchen" in task.tags
        assert task.metadata["source"] == "voice_command"

    @pytest.mark.asyncio
    async def test_priority_levels(self, service):
        """Test different priority levels."""
        high_id = await service.set_timer(
            name="High Priority",
            duration=timedelta(minutes=5),
            action="notify",
            priority=TaskPriority.HIGH,
        )

        low_id = await service.set_timer(
            name="Low Priority",
            duration=timedelta(minutes=5),
            action="notify",
            priority=TaskPriority.LOW,
        )

        high = await service.get_task(high_id)
        low = await service.get_task(low_id)

        assert high.priority == TaskPriority.HIGH
        assert low.priority == TaskPriority.LOW
        assert high.priority.value < low.priority.value
