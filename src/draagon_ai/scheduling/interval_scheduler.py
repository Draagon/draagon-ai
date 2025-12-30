"""Simple interval-based scheduler."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Awaitable

from .models import ScheduledTask, TaskStatus, TriggerType
from .persistence import TaskPersistence

logger = logging.getLogger(__name__)


class IntervalScheduler:
    """Simple interval-based scheduler.

    Provides a cleaner API than cron for simple "every N minutes" schedules.
    Supports dynamic intervals (not just clock-aligned) and start-immediately option.

    Features:
    - Cleaner API: every=timedelta(hours=2) vs "0 */2 * * *"
    - Dynamic intervals (not clock-aligned)
    - Start-immediately option
    - Max execution limits
    - Pause/resume support
    """

    def __init__(
        self,
        persistence: TaskPersistence,
        check_interval: int = 30,
    ):
        """Initialize interval scheduler.

        Args:
            persistence: Task persistence layer
            check_interval: Seconds between checking for due tasks (default: 30)
        """
        self._persistence = persistence
        self._check_interval = check_interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._callback: Callable[[ScheduledTask], Awaitable[None]] | None = None

    def set_callback(
        self, callback: Callable[[ScheduledTask], Awaitable[None]]
    ) -> None:
        """Set the callback for when an interval task is due.

        Args:
            callback: Async function called with the task when it's due
        """
        self._callback = callback

    async def schedule(
        self,
        task: ScheduledTask,
        interval: timedelta,
        start_immediately: bool = False,
        max_executions: int | None = None,
    ) -> str:
        """Schedule a task to run at fixed intervals.

        Args:
            task: The task to execute
            interval: Time between executions (e.g., timedelta(hours=2))
            start_immediately: Run once immediately (default: False)
            max_executions: Limit executions (None = unlimited)

        Returns:
            task_id
        """
        # Configure task
        task.trigger_type = TriggerType.INTERVAL
        task.trigger_config = {
            "seconds": interval.total_seconds(),
            "start_immediately": start_immediately,
        }
        task.max_executions = max_executions

        if start_immediately:
            task.next_run = datetime.now()
        else:
            task.next_run = datetime.now() + interval

        # Persist
        await self._persistence.save(task)

        logger.debug(
            f"Interval scheduled: {task.name} ({task.task_id}) "
            f"every {interval} next_run={task.next_run.isoformat()}"
        )

        return task.task_id

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Interval scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Interval scheduler started (check interval: {self._check_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the scheduler loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("Interval scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Interval scheduler error: {e}")
                await asyncio.sleep(5)

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them."""
        now = datetime.now()
        due_tasks = await self._persistence.get_due_tasks(
            before=now,
            trigger_type=TriggerType.INTERVAL,
        )

        for task in due_tasks:
            if task.status == TaskStatus.PAUSED:
                continue

            # Execute via callback
            if self._callback:
                try:
                    await self._callback(task)
                except Exception as e:
                    logger.error(f"Interval task {task.task_id} callback failed: {e}")

            # Update task state
            interval_seconds = task.trigger_config.get("seconds", 3600)
            task.next_run = now + timedelta(seconds=interval_seconds)
            task.last_run = now
            task.execution_count += 1

            # Check if max executions reached
            if task.has_reached_max_executions():
                task.status = TaskStatus.COMPLETED
                logger.info(f"Interval task completed (max executions): {task.name}")

            await self._persistence.save(task)

    async def pause(self, task_id: str) -> bool:
        """Pause an interval task.

        Args:
            task_id: ID of task to pause

        Returns:
            True if paused, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.INTERVAL:
            task.status = TaskStatus.PAUSED
            await self._persistence.save(task)
            logger.info(f"Interval task paused: {task.name}")
            return True
        return False

    async def resume(self, task_id: str) -> bool:
        """Resume a paused interval task.

        Args:
            task_id: ID of task to resume

        Returns:
            True if resumed, False if not found or not paused
        """
        task = await self._persistence.get(task_id)
        if task and task.status == TaskStatus.PAUSED:
            # Reset next_run to now + interval
            interval_seconds = task.trigger_config.get("seconds", 3600)
            task.next_run = datetime.now() + timedelta(seconds=interval_seconds)
            task.status = TaskStatus.PENDING
            await self._persistence.save(task)
            logger.info(f"Interval task resumed: {task.name}")
            return True
        return False

    async def update_interval(
        self,
        task_id: str,
        interval: timedelta,
    ) -> bool:
        """Update the interval for a task.

        Args:
            task_id: ID of task to update
            interval: New interval

        Returns:
            True if updated, False if not found
        """
        task = await self._persistence.get(task_id)
        if not task or task.trigger_type != TriggerType.INTERVAL:
            return False

        task.trigger_config["seconds"] = interval.total_seconds()
        task.next_run = datetime.now() + interval

        await self._persistence.save(task)
        logger.info(f"Interval task updated: {task.name} interval={interval}")
        return True

    async def cancel(self, task_id: str) -> bool:
        """Cancel an interval task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.INTERVAL:
            task.status = TaskStatus.CANCELLED
            await self._persistence.save(task)
            logger.info(f"Interval task cancelled: {task.name}")
            return True
        return False

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is running."""
        return self._running
