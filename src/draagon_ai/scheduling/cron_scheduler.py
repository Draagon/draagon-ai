"""Cron-based recurring scheduler."""

import asyncio
import logging
from datetime import datetime
from typing import Callable, Awaitable

from .models import ScheduledTask, TaskStatus, TriggerType
from .persistence import TaskPersistence

logger = logging.getLogger(__name__)


class CronScheduler:
    """Cron-based recurring scheduler.

    Uses croniter for cron expression parsing and next-run calculation.
    Runs a background loop that checks for due tasks at regular intervals.

    Features:
    - Standard 5-field cron expressions
    - Timezone support
    - Start/end date constraints
    - Max execution limits
    - Pause/resume support
    """

    def __init__(
        self,
        persistence: TaskPersistence,
        check_interval: int = 60,
    ):
        """Initialize cron scheduler.

        Args:
            persistence: Task persistence layer
            check_interval: Seconds between checking for due tasks (default: 60)
        """
        self._persistence = persistence
        self._check_interval = check_interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._callback: Callable[[ScheduledTask], Awaitable[None]] | None = None

    def set_callback(
        self, callback: Callable[[ScheduledTask], Awaitable[None]]
    ) -> None:
        """Set the callback for when a cron task is due.

        Args:
            callback: Async function called with the task when it's due
        """
        self._callback = callback

    async def schedule(
        self,
        task: ScheduledTask,
        expression: str,
        timezone: str = "UTC",
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> str:
        """Schedule a recurring task with a cron expression.

        Args:
            task: The task to execute
            expression: Cron expression (e.g., "0 9 * * 1" = 9am every Monday)
            timezone: IANA timezone (e.g., "America/New_York")
            start: When to start scheduling (default: now)
            end: When to stop scheduling (optional)

        Returns:
            task_id
        """
        try:
            from croniter import croniter
        except ImportError:
            raise ImportError("croniter is required for CronScheduler")

        # Configure task
        task.trigger_type = TriggerType.CRON
        task.trigger_config = {
            "expression": expression,
            "timezone": timezone,
            "start": start.isoformat() if start else None,
            "end": end.isoformat() if end else None,
        }

        # Calculate next run
        base = start or datetime.now()
        cron = croniter(expression, base)
        task.next_run = cron.get_next(datetime)

        if end:
            task.expires_at = end

        # Persist
        await self._persistence.save(task)

        logger.debug(
            f"Cron scheduled: {task.name} ({task.task_id}) "
            f"expression='{expression}' next_run={task.next_run.isoformat()}"
        )

        return task.task_id

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            logger.warning("Cron scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            f"Cron scheduler started (check interval: {self._check_interval}s)"
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
        logger.info("Cron scheduler stopped")

    async def _loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                await self._check_and_run()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cron scheduler error: {e}")
                await asyncio.sleep(5)  # Brief pause before retry

    async def _check_and_run(self) -> None:
        """Check for due tasks and execute them."""
        try:
            from croniter import croniter
        except ImportError:
            logger.error("croniter not installed")
            return

        now = datetime.now()
        due_tasks = await self._persistence.get_due_tasks(
            before=now,
            trigger_type=TriggerType.CRON,
        )

        for task in due_tasks:
            if task.status == TaskStatus.PAUSED:
                continue

            # Execute via callback
            if self._callback:
                try:
                    await self._callback(task)
                except Exception as e:
                    logger.error(f"Cron task {task.task_id} callback failed: {e}")
                    # Continue to update next_run regardless

            # Update task state
            config = task.trigger_config
            expression = config.get("expression", "0 0 * * *")

            cron = croniter(expression, now)
            task.next_run = cron.get_next(datetime)
            task.last_run = now
            task.execution_count += 1

            # Check if expired or max executions reached
            if task.expires_at and task.next_run > task.expires_at:
                task.status = TaskStatus.COMPLETED
                logger.info(f"Cron task completed (expired): {task.name}")
            elif task.has_reached_max_executions():
                task.status = TaskStatus.COMPLETED
                logger.info(f"Cron task completed (max executions): {task.name}")

            await self._persistence.save(task)

    async def pause(self, task_id: str) -> bool:
        """Pause a cron task.

        Args:
            task_id: ID of task to pause

        Returns:
            True if paused, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.CRON:
            task.status = TaskStatus.PAUSED
            await self._persistence.save(task)
            logger.info(f"Cron task paused: {task.name}")
            return True
        return False

    async def resume(self, task_id: str) -> bool:
        """Resume a paused cron task.

        Args:
            task_id: ID of task to resume

        Returns:
            True if resumed, False if not found or not paused
        """
        task = await self._persistence.get(task_id)
        if task and task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.PENDING
            await self._persistence.save(task)
            logger.info(f"Cron task resumed: {task.name}")
            return True
        return False

    async def update_schedule(
        self,
        task_id: str,
        expression: str | None = None,
        timezone: str | None = None,
    ) -> bool:
        """Update the schedule for a cron task.

        Args:
            task_id: ID of task to update
            expression: New cron expression (optional)
            timezone: New timezone (optional)

        Returns:
            True if updated, False if not found
        """
        try:
            from croniter import croniter
        except ImportError:
            return False

        task = await self._persistence.get(task_id)
        if not task or task.trigger_type != TriggerType.CRON:
            return False

        if expression:
            task.trigger_config["expression"] = expression
        if timezone:
            task.trigger_config["timezone"] = timezone

        # Recalculate next run
        expr = task.trigger_config.get("expression", "0 0 * * *")
        cron = croniter(expr, datetime.now())
        task.next_run = cron.get_next(datetime)

        await self._persistence.save(task)
        logger.info(f"Cron task updated: {task.name} next_run={task.next_run}")
        return True

    async def cancel(self, task_id: str) -> bool:
        """Cancel a cron task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.CRON:
            task.status = TaskStatus.CANCELLED
            await self._persistence.save(task)
            logger.info(f"Cron task cancelled: {task.name}")
            return True
        return False

    @property
    def is_running(self) -> bool:
        """Whether the scheduler loop is running."""
        return self._running
