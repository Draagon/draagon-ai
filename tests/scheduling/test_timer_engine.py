"""Tests for TimerEngine."""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock

from draagon_ai.scheduling.models import ScheduledTask, TaskStatus, TriggerType
from draagon_ai.scheduling.persistence import InMemoryPersistence
from draagon_ai.scheduling.timer_engine import TimerEngine


class TestTimerEngine:
    """Tests for TimerEngine."""

    @pytest.fixture
    def persistence(self):
        """Create a fresh persistence instance."""
        return InMemoryPersistence()

    @pytest.fixture
    def timer_engine(self, persistence):
        """Create a timer engine."""
        return TimerEngine(persistence)

    @pytest.mark.asyncio
    async def test_schedule_with_duration(self, timer_engine, persistence):
        """Test scheduling with duration."""
        task = ScheduledTask(name="Test Timer")

        task_id = await timer_engine.schedule(
            task, duration=timedelta(seconds=10)
        )

        assert task_id == task.task_id

        # Task should be persisted
        saved = await persistence.get(task_id)
        assert saved is not None
        assert saved.trigger_type == TriggerType.ONCE
        assert saved.next_run is not None
        assert saved.next_run > datetime.now()

    @pytest.mark.asyncio
    async def test_schedule_with_absolute_time(self, timer_engine, persistence):
        """Test scheduling with absolute time."""
        task = ScheduledTask(name="Test Timer")
        target_time = datetime.now() + timedelta(minutes=5)

        task_id = await timer_engine.schedule(task, at=target_time)

        saved = await persistence.get(task_id)
        assert saved is not None
        # Allow small delta for execution time
        assert abs((saved.next_run - target_time).total_seconds()) < 1

    @pytest.mark.asyncio
    async def test_schedule_requires_duration_or_at(self, timer_engine):
        """Test that schedule requires duration or at."""
        task = ScheduledTask(name="Test Timer")

        with pytest.raises(ValueError, match="Must provide duration or at"):
            await timer_engine.schedule(task)

    @pytest.mark.asyncio
    async def test_callback_fires(self, timer_engine):
        """Test that callback fires when timer expires."""
        fired = asyncio.Event()
        fired_task = None

        async def callback(task):
            nonlocal fired_task
            fired_task = task
            fired.set()

        timer_engine.set_callback(callback)

        task = ScheduledTask(name="Quick Timer")
        await timer_engine.schedule(task, duration=timedelta(milliseconds=100))

        # Wait for timer to fire
        await asyncio.wait_for(fired.wait(), timeout=2.0)

        assert fired_task is not None
        assert fired_task.task_id == task.task_id

    @pytest.mark.asyncio
    async def test_cancel_timer(self, timer_engine, persistence):
        """Test cancelling a timer."""
        task = ScheduledTask(name="Cancel Me")
        task_id = await timer_engine.schedule(
            task, duration=timedelta(seconds=60)
        )

        result = await timer_engine.cancel(task_id)
        assert result is True

        # Task should be marked cancelled
        saved = await persistence.get(task_id)
        assert saved.status == TaskStatus.CANCELLED

        # Timer should not be in active timers
        assert task_id not in timer_engine._timers

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, timer_engine):
        """Test cancelling nonexistent timer returns False."""
        result = await timer_engine.cancel("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_remaining(self, timer_engine):
        """Test getting remaining time."""
        task = ScheduledTask(name="Check Remaining")
        task_id = await timer_engine.schedule(
            task, duration=timedelta(seconds=60)
        )

        remaining = await timer_engine.get_remaining(task_id)
        assert remaining is not None
        assert remaining > timedelta(seconds=55)
        assert remaining <= timedelta(seconds=60)

    @pytest.mark.asyncio
    async def test_get_remaining_nonexistent(self, timer_engine):
        """Test getting remaining time for nonexistent timer."""
        remaining = await timer_engine.get_remaining("nonexistent")
        assert remaining is None

    @pytest.mark.asyncio
    async def test_list_active(self, timer_engine):
        """Test listing active timers."""
        for i in range(3):
            task = ScheduledTask(name=f"Timer {i}", owner_id="alice")
            await timer_engine.schedule(task, duration=timedelta(minutes=5))

        task = ScheduledTask(name="Bob's Timer", owner_id="bob")
        await timer_engine.schedule(task, duration=timedelta(minutes=5))

        # List all
        all_timers = await timer_engine.list_active()
        assert len(all_timers) == 4

        # List by owner
        alice_timers = await timer_engine.list_active(owner_id="alice")
        assert len(alice_timers) == 3

    @pytest.mark.asyncio
    async def test_active_count(self, timer_engine):
        """Test active timer count."""
        assert timer_engine.active_count == 0

        for i in range(3):
            task = ScheduledTask(name=f"Timer {i}")
            await timer_engine.schedule(task, duration=timedelta(minutes=5))

        assert timer_engine.active_count == 3

    @pytest.mark.asyncio
    async def test_stop(self, timer_engine):
        """Test stopping timer engine."""
        for i in range(3):
            task = ScheduledTask(name=f"Timer {i}")
            await timer_engine.schedule(task, duration=timedelta(minutes=5))

        assert timer_engine.active_count == 3

        await timer_engine.stop()

        assert timer_engine.active_count == 0

    @pytest.mark.asyncio
    async def test_restore_from_persistence(self, timer_engine, persistence):
        """Test restoring timers from persistence."""
        # Manually save some pending timers
        future_task = ScheduledTask(
            task_id="future",
            name="Future Timer",
            status=TaskStatus.PENDING,
            trigger_type=TriggerType.ONCE,
            next_run=datetime.now() + timedelta(minutes=5),
        )
        await persistence.save(future_task)

        # Clear in-memory state
        timer_engine._timers.clear()

        # Restore
        restored = await timer_engine.restore_from_persistence()
        assert restored == 1
        assert timer_engine.active_count == 1

    @pytest.mark.asyncio
    async def test_missed_timer_fires_immediately(self, timer_engine, persistence):
        """Test that missed timers fire immediately on restore."""
        fired = asyncio.Event()

        async def callback(task):
            fired.set()

        timer_engine.set_callback(callback)

        # Save a timer that should have fired already
        missed_task = ScheduledTask(
            task_id="missed",
            name="Missed Timer",
            status=TaskStatus.PENDING,
            trigger_type=TriggerType.ONCE,
            next_run=datetime.now() - timedelta(minutes=5),  # Past
        )
        await persistence.save(missed_task)

        # Restore
        await timer_engine.restore_from_persistence()

        # Should fire immediately
        await asyncio.wait_for(fired.wait(), timeout=2.0)

    @pytest.mark.asyncio
    async def test_cancelled_timer_does_not_fire(self, timer_engine, persistence):
        """Test that cancelled timers don't fire."""
        fired = asyncio.Event()

        async def callback(task):
            fired.set()

        timer_engine.set_callback(callback)

        task = ScheduledTask(name="Cancel Me")
        task_id = await timer_engine.schedule(
            task, duration=timedelta(milliseconds=100)
        )

        # Cancel before it fires
        await timer_engine.cancel(task_id)

        # Wait a bit - should not fire
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(fired.wait(), timeout=0.5)
