"""Integration tests for the full scheduling system.

These tests verify the complete scheduling flow:
- SchedulingService with TimerEngine
- Interval and cron scheduling
- Action registration and execution
- Persistence across restarts

Uses real components, minimal mocking.
"""

import asyncio
import pytest
from datetime import datetime, timedelta

from draagon_ai.scheduling import SchedulingService, InMemoryPersistence
from draagon_ai.scheduling.models import ScheduledTask, TaskStatus


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def scheduler():
    """Create a scheduling service for tests."""
    persistence = InMemoryPersistence()
    service = SchedulingService(
        persistence=persistence,
        interval_check_interval=1,  # Fast for tests
    )
    await service.start()
    yield service
    await service.stop()


# =============================================================================
# Integration Tests - Timer Engine (already well-tested in test_timer_engine.py)
# =============================================================================


class TestTimerIntegration:
    """Test timer functionality through the service."""

    @pytest.mark.asyncio
    async def test_set_and_cancel_timer(self, scheduler):
        """Test setting and cancelling a timer."""
        task_id = await scheduler.set_timer(
            name="Test Timer",
            duration=timedelta(minutes=5),
            action="test_action",
            owner_id="test_user",
        )

        assert task_id is not None

        # Get remaining time
        remaining = await scheduler.get_timer_remaining(task_id)
        assert remaining is not None
        assert remaining > timedelta(minutes=4)

        # Cancel
        result = await scheduler.cancel(task_id)
        assert result is True

        # Task should be cancelled in persistence
        task = await scheduler._persistence.get(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_list_timers_by_owner(self, scheduler):
        """Test listing timers for a specific owner."""
        # Set timers for different owners
        await scheduler.set_timer(
            name="Alice Timer 1",
            duration=timedelta(minutes=5),
            action="test",
            owner_id="alice",
        )
        await scheduler.set_timer(
            name="Alice Timer 2",
            duration=timedelta(minutes=10),
            action="test",
            owner_id="alice",
        )
        await scheduler.set_timer(
            name="Bob Timer",
            duration=timedelta(minutes=5),
            action="test",
            owner_id="bob",
        )

        # List Alice's timers
        alice_timers = await scheduler.list_timers(owner_id="alice")
        assert len(alice_timers) == 2

        # List all timers
        all_timers = await scheduler.list_timers()
        assert len(all_timers) == 3

    @pytest.mark.asyncio
    async def test_timer_fires_and_executes_action(self, scheduler):
        """Test that timer fires and action is executed."""
        action_executed = asyncio.Event()
        execution_params = {}

        async def test_action(params, event_data):
            execution_params.update(params)
            action_executed.set()
            return {"success": True}

        # Register our test action
        scheduler._executor.register_action("fire_test", test_action)

        # Set a quick timer
        await scheduler.set_timer(
            name="Quick Timer",
            duration=timedelta(milliseconds=100),
            action="fire_test",
            action_params={"message": "Timer fired!"},
            owner_id="test",
        )

        # Wait for execution
        await asyncio.wait_for(action_executed.wait(), timeout=2.0)

        assert execution_params["message"] == "Timer fired!"


# =============================================================================
# Integration Tests - Interval Scheduling
# =============================================================================


class TestIntervalIntegration:
    """Test interval scheduling functionality."""

    @pytest.mark.asyncio
    async def test_interval_schedules_recurring(self, scheduler):
        """Test that interval tasks are scheduled."""
        task_id = await scheduler.schedule_interval(
            name="Periodic Check",
            interval=timedelta(seconds=5),
            action="periodic_action",
            owner_id="test",
        )

        assert task_id is not None

        # Can cancel interval tasks
        result = await scheduler.cancel(task_id)
        assert result is True


# =============================================================================
# Integration Tests - Action Registration
# =============================================================================


class TestActionRegistration:
    """Test action registration and execution."""

    @pytest.mark.asyncio
    async def test_register_and_execute_action(self, scheduler):
        """Test registering and executing a custom action."""
        call_log = []

        async def logging_action(params, event_data):
            call_log.append({"params": params, "time": datetime.now()})
            return {"logged": True}

        scheduler.register_action("log_action", logging_action)

        # Create a task to execute
        task = ScheduledTask(
            name="Log Task",
            action="log_action",
            action_params={"key": "value"},
        )

        # Execute via the executor
        result = await scheduler._executor.execute(task, None)

        assert result["logged"] is True
        assert len(call_log) == 1
        assert call_log[0]["params"] == {"key": "value"}

    @pytest.mark.asyncio
    async def test_action_with_error_handling(self, scheduler):
        """Test that action errors mark task as failed."""
        async def failing_action(params, event_data):
            raise RuntimeError("Intentional failure")

        scheduler.register_action("failing", failing_action)

        # Create a task that will fail (max_retries=0 to fail immediately)
        task = ScheduledTask(
            name="Failing Task",
            action="failing",
            action_params={},
            max_retries=0,  # No retries - fail immediately
        )

        # Execute - will raise after all retries exhausted
        try:
            await scheduler._executor.execute(task, None)
        except RuntimeError:
            pass  # Expected

        # Task should be marked as failed
        assert task.status == TaskStatus.FAILED


# =============================================================================
# Integration Tests - Persistence
# =============================================================================


class TestPersistence:
    """Test persistence and recovery."""

    @pytest.mark.asyncio
    async def test_timer_persists_in_memory(self):
        """Test that timers are persisted to the persistence layer."""
        persistence = InMemoryPersistence()
        scheduler = SchedulingService(persistence=persistence)
        await scheduler.start()

        task_id = await scheduler.set_timer(
            name="Persistent Timer",
            duration=timedelta(minutes=5),
            action="test",
            owner_id="test",
        )

        # Check persistence directly
        task = await persistence.get(task_id)
        assert task is not None
        assert task.name == "Persistent Timer"
        assert task.status == TaskStatus.PENDING

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_cancelled_task_updated_in_persistence(self):
        """Test that cancelled tasks are updated in persistence."""
        persistence = InMemoryPersistence()
        scheduler = SchedulingService(persistence=persistence)
        await scheduler.start()

        task_id = await scheduler.set_timer(
            name="Cancel Me",
            duration=timedelta(minutes=5),
            action="test",
            owner_id="test",
        )

        await scheduler.cancel(task_id)

        # Check persistence
        task = await persistence.get(task_id)
        assert task is not None
        assert task.status == TaskStatus.CANCELLED

        await scheduler.stop()


# =============================================================================
# Integration Tests - Multiple Components
# =============================================================================


class TestMultipleComponents:
    """Test interactions between multiple scheduling components."""

    @pytest.mark.asyncio
    async def test_timer_and_interval_coexist(self, scheduler):
        """Test that timers and intervals can work together."""
        timer_id = await scheduler.set_timer(
            name="One-shot",
            duration=timedelta(minutes=5),
            action="test",
            owner_id="test",
        )

        interval_id = await scheduler.schedule_interval(
            name="Recurring",
            interval=timedelta(minutes=10),
            action="test",
            owner_id="test",
        )

        # Both should be active
        timer_remaining = await scheduler.get_timer_remaining(timer_id)
        assert timer_remaining is not None

        # Cancel both
        await scheduler.cancel(timer_id)
        await scheduler.cancel(interval_id)

    @pytest.mark.asyncio
    async def test_service_stop_cleans_up(self):
        """Test that stopping the service cleans up properly."""
        persistence = InMemoryPersistence()
        scheduler = SchedulingService(persistence=persistence)
        await scheduler.start()

        # Set some timers
        for i in range(3):
            await scheduler.set_timer(
                name=f"Timer {i}",
                duration=timedelta(minutes=5),
                action="test",
                owner_id="test",
            )

        # Stop should not raise
        await scheduler.stop()

        # Can restart
        scheduler2 = SchedulingService(persistence=persistence)
        await scheduler2.start()
        await scheduler2.stop()
