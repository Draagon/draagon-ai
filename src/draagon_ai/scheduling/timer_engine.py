"""High-precision one-shot timer engine using asyncio."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Awaitable

from .models import ScheduledTask, TaskStatus, TriggerType
from .persistence import TaskPersistence

logger = logging.getLogger(__name__)


class TimerEngine:
    """High-precision one-shot timer engine using asyncio.

    Uses asyncio.call_later() for sub-millisecond precision instead of polling.
    Timers fire exactly when they expire, not at the next poll interval.

    Features:
    - Precise timing with asyncio.call_later
    - Persistence for durability across restarts
    - Automatic restoration of missed timers
    - Callback-based execution
    """

    def __init__(self, persistence: TaskPersistence):
        self._persistence = persistence
        self._timers: dict[str, asyncio.TimerHandle] = {}
        self._callback: Callable[[ScheduledTask], Awaitable[None]] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_callback(
        self, callback: Callable[[ScheduledTask], Awaitable[None]]
    ) -> None:
        """Set the callback for when a timer fires.

        Args:
            callback: Async function called with the task when timer expires
        """
        self._callback = callback

    async def schedule(
        self,
        task: ScheduledTask,
        duration: timedelta | None = None,
        at: datetime | None = None,
    ) -> str:
        """Schedule a one-shot timer.

        Args:
            task: The task to execute when timer fires
            duration: Time from now (e.g., timedelta(minutes=5))
            at: Absolute time to execute

        Returns:
            task_id

        Raises:
            ValueError: If neither duration nor at is provided
        """
        if duration:
            run_at = datetime.now() + duration
        elif at:
            run_at = at
        else:
            raise ValueError("Must provide duration or at")

        # Configure task
        task.trigger_type = TriggerType.ONCE
        task.next_run = run_at
        task.trigger_config = {"at": run_at.isoformat()}

        # Persist for durability
        await self._persistence.save(task)

        # Schedule in asyncio
        self._schedule_timer(task)

        logger.debug(
            f"Timer scheduled: {task.name} ({task.task_id}) "
            f"fires at {run_at.isoformat()}"
        )

        return task.task_id

    def _schedule_timer(self, task: ScheduledTask) -> None:
        """Schedule a timer in asyncio."""
        if task.next_run is None:
            return

        delay = (task.next_run - datetime.now()).total_seconds()

        if delay <= 0:
            # Already past, fire immediately
            asyncio.create_task(self._fire(task.task_id))
            return

        # Get event loop
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()

        # Schedule precise callback
        handle = loop.call_later(
            delay,
            lambda tid=task.task_id: asyncio.create_task(self._fire(tid)),
        )
        self._timers[task.task_id] = handle

        logger.debug(f"Timer {task.task_id} scheduled to fire in {delay:.2f}s")

    async def _fire(self, task_id: str) -> None:
        """Fire a timer."""
        # Remove handle from tracking
        self._timers.pop(task_id, None)

        # Get task from persistence
        task = await self._persistence.get(task_id)
        if not task:
            logger.debug(f"Timer {task_id} not found, may have been cancelled")
            return

        if task.status == TaskStatus.CANCELLED:
            logger.debug(f"Timer {task_id} was cancelled, skipping")
            return

        logger.info(f"Timer fired: {task.name} ({task_id})")

        # Execute callback
        if self._callback:
            try:
                await self._callback(task)
            except Exception as e:
                logger.error(f"Timer callback failed for {task_id}: {e}")
                # Let executor handle retry logic

    async def cancel(self, task_id: str) -> bool:
        """Cancel a timer.

        Args:
            task_id: ID of timer to cancel

        Returns:
            True if cancelled, False if not found
        """
        # Cancel asyncio handle
        handle = self._timers.pop(task_id, None)
        if handle:
            handle.cancel()

        # Update task status
        task = await self._persistence.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            await self._persistence.save(task)
            logger.info(f"Timer cancelled: {task.name} ({task_id})")
            return True

        return False

    async def get_remaining(self, task_id: str) -> timedelta | None:
        """Get time remaining for a timer.

        Args:
            task_id: ID of timer

        Returns:
            Time remaining, or None if not found/expired
        """
        task = await self._persistence.get(task_id)
        if not task or not task.next_run:
            return None

        remaining = task.next_run - datetime.now()
        if remaining.total_seconds() <= 0:
            return timedelta(0)
        return remaining

    async def list_active(self, owner_id: str | None = None) -> list[ScheduledTask]:
        """List active timers.

        Args:
            owner_id: Optional filter by owner

        Returns:
            List of active timer tasks
        """
        tasks = await self._persistence.get_pending(trigger_type=TriggerType.ONCE)
        if owner_id:
            tasks = [t for t in tasks if t.owner_id == owner_id]
        return tasks

    async def restore_from_persistence(self) -> int:
        """Restore timers after restart.

        Loads pending one-shot timers from persistence and reschedules them.
        Timers that were missed while offline are fired immediately.

        Returns:
            Number of timers restored
        """
        restored = 0
        tasks = await self._persistence.get_pending(trigger_type=TriggerType.ONCE)

        for task in tasks:
            if task.next_run is None:
                continue

            if task.next_run > datetime.now():
                # Future timer - schedule it
                self._schedule_timer(task)
                restored += 1
                logger.debug(f"Restored timer: {task.name} ({task.task_id})")
            else:
                # Missed while offline - fire immediately
                asyncio.create_task(self._fire(task.task_id))
                restored += 1
                logger.info(
                    f"Firing missed timer: {task.name} ({task.task_id}) "
                    f"was scheduled for {task.next_run.isoformat()}"
                )

        if restored > 0:
            logger.info(f"Restored {restored} timers from persistence")

        return restored

    async def stop(self) -> None:
        """Stop all timers."""
        for task_id, handle in list(self._timers.items()):
            handle.cancel()
        self._timers.clear()
        logger.debug("Timer engine stopped")

    @property
    def active_count(self) -> int:
        """Number of active timers in memory."""
        return len(self._timers)
