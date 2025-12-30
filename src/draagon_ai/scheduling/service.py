"""Unified scheduling service facade."""

import logging
from datetime import datetime, timedelta
from typing import Callable, Awaitable, Any

from .models import ScheduledTask, TaskStatus, TaskPriority, TriggerType
from .persistence import TaskPersistence, InMemoryPersistence
from .timer_engine import TimerEngine
from .cron_scheduler import CronScheduler
from .interval_scheduler import IntervalScheduler
from .event_scheduler import EventScheduler
from .executor import TaskExecutor, register_builtin_actions

logger = logging.getLogger(__name__)


class SchedulingService:
    """Unified scheduling service for draagon-ai.

    Provides a single interface for all scheduling needs:
    - One-shot timers (high-precision asyncio)
    - Recurring schedules (cron expressions)
    - Interval schedules (simple every-N patterns)
    - Event-triggered tasks (reactive)

    Usage:
        # Create with persistence
        persistence = QdrantPersistence(url, collection)
        scheduler = SchedulingService(persistence)
        await scheduler.start()

        # One-shot timer
        await scheduler.set_timer(
            name="Pasta timer",
            duration=timedelta(minutes=5),
            action="notify",
            action_params={"message": "Pasta is ready!"},
            owner_id="user_123",
        )

        # Recurring cron
        await scheduler.schedule_cron(
            name="Daily briefing",
            expression="0 9 * * *",
            action="daily_briefing",
            owner_id="user_123",
        )

        # Event-triggered
        await scheduler.on_event(
            name="Summarize conversation",
            event_type="conversation_end",
            action="summarize",
        )
    """

    def __init__(
        self,
        persistence: TaskPersistence | None = None,
        max_concurrent: int = 10,
        cron_check_interval: int = 60,
        interval_check_interval: int = 30,
    ):
        """Initialize scheduling service.

        Args:
            persistence: Task persistence (default: InMemoryPersistence)
            max_concurrent: Max concurrent task executions
            cron_check_interval: Seconds between cron checks
            interval_check_interval: Seconds between interval checks
        """
        self._persistence = persistence or InMemoryPersistence()

        # Initialize components
        self._timer = TimerEngine(self._persistence)
        self._cron = CronScheduler(self._persistence, cron_check_interval)
        self._interval = IntervalScheduler(self._persistence, interval_check_interval)
        self._event = EventScheduler(self._persistence)
        self._executor = TaskExecutor(self._persistence, max_concurrent)

        # Wire up callbacks
        self._timer.set_callback(self._on_task_due)
        self._cron.set_callback(self._on_task_due)
        self._interval.set_callback(self._on_task_due)
        self._event.set_callback(self._on_event_task)

        # Register built-in actions
        register_builtin_actions(self._executor)

        self._started = False

    async def start(self) -> None:
        """Start all schedulers.

        Restores timers and event handlers from persistence,
        and starts the cron and interval scheduler loops.
        """
        if self._started:
            logger.warning("Scheduling service already started")
            return

        # Restore state from persistence
        await self._timer.restore_from_persistence()
        await self._event.restore_handlers()

        # Start scheduler loops
        await self._cron.start()
        await self._interval.start()

        self._started = True
        logger.info("Scheduling service started")

    async def stop(self) -> None:
        """Stop all schedulers."""
        await self._timer.stop()
        await self._cron.stop()
        await self._interval.stop()

        self._started = False
        logger.info("Scheduling service stopped")

    # ==================== One-Shot Timers ====================

    async def set_timer(
        self,
        name: str,
        duration: timedelta,
        action: str = "notify",
        action_params: dict | None = None,
        owner_type: str = "user",
        owner_id: str = "default",
        priority: TaskPriority = TaskPriority.HIGH,
        callback: Callable[..., Awaitable[Any]] | None = None,
        tags: list[str] | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Set a one-shot timer.

        Args:
            name: Human-readable timer name
            duration: Time from now until timer fires
            action: Action to execute (default: "notify")
            action_params: Parameters for the action
            owner_type: Owner type ("user", "feature", "system")
            owner_id: Owner identifier
            priority: Execution priority
            callback: Optional callback instead of action
            tags: Optional tags for filtering
            metadata: Optional metadata

        Returns:
            task_id
        """
        task = ScheduledTask(
            name=name,
            owner_type=owner_type,
            owner_id=owner_id,
            action=action,
            action_params=action_params or {},
            priority=priority,
            callback=callback,
            tags=tags or [],
            metadata=metadata or {},
        )

        return await self._timer.schedule(task, duration=duration)

    async def set_timer_at(
        self,
        name: str,
        at: datetime,
        action: str = "notify",
        action_params: dict | None = None,
        owner_type: str = "user",
        owner_id: str = "default",
        priority: TaskPriority = TaskPriority.HIGH,
        callback: Callable[..., Awaitable[Any]] | None = None,
    ) -> str:
        """Set a timer for a specific time.

        Args:
            name: Human-readable timer name
            at: Datetime when timer should fire
            action: Action to execute
            action_params: Parameters for the action
            owner_type: Owner type
            owner_id: Owner identifier
            priority: Execution priority
            callback: Optional callback

        Returns:
            task_id
        """
        task = ScheduledTask(
            name=name,
            owner_type=owner_type,
            owner_id=owner_id,
            action=action,
            action_params=action_params or {},
            priority=priority,
            callback=callback,
        )

        return await self._timer.schedule(task, at=at)

    async def get_timer_remaining(self, task_id: str) -> timedelta | None:
        """Get time remaining for a timer."""
        return await self._timer.get_remaining(task_id)

    async def list_timers(self, owner_id: str | None = None) -> list[ScheduledTask]:
        """List active timers."""
        return await self._timer.list_active(owner_id)

    # ==================== Cron Schedules ====================

    async def schedule_cron(
        self,
        name: str,
        expression: str,
        action: str,
        action_params: dict | None = None,
        owner_type: str = "system",
        owner_id: str = "default",
        timezone: str = "UTC",
        start: datetime | None = None,
        end: datetime | None = None,
        max_executions: int | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable[..., Awaitable[Any]] | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Schedule a recurring task with cron expression.

        Args:
            name: Human-readable name
            expression: Cron expression (e.g., "0 9 * * *" = 9am daily)
            action: Action to execute
            action_params: Parameters for the action
            owner_type: Owner type
            owner_id: Owner identifier
            timezone: IANA timezone
            start: When to start scheduling
            end: When to stop scheduling
            max_executions: Maximum number of executions
            priority: Execution priority
            callback: Optional callback
            tags: Optional tags
            description: Task description

        Returns:
            task_id
        """
        task = ScheduledTask(
            name=name,
            description=description,
            owner_type=owner_type,
            owner_id=owner_id,
            action=action,
            action_params=action_params or {},
            priority=priority,
            max_executions=max_executions,
            callback=callback,
            tags=tags or [],
        )

        return await self._cron.schedule(task, expression, timezone, start, end)

    # ==================== Interval Schedules ====================

    async def schedule_interval(
        self,
        name: str,
        interval: timedelta,
        action: str,
        action_params: dict | None = None,
        start_immediately: bool = False,
        max_executions: int | None = None,
        owner_type: str = "system",
        owner_id: str = "default",
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable[..., Awaitable[Any]] | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Schedule a task to run at fixed intervals.

        Args:
            name: Human-readable name
            interval: Time between executions
            action: Action to execute
            action_params: Parameters for the action
            start_immediately: Run once immediately
            max_executions: Maximum number of executions
            owner_type: Owner type
            owner_id: Owner identifier
            priority: Execution priority
            callback: Optional callback
            tags: Optional tags
            description: Task description

        Returns:
            task_id
        """
        task = ScheduledTask(
            name=name,
            description=description,
            owner_type=owner_type,
            owner_id=owner_id,
            action=action,
            action_params=action_params or {},
            priority=priority,
            callback=callback,
            tags=tags or [],
        )

        return await self._interval.schedule(
            task, interval, start_immediately, max_executions
        )

    # ==================== Event Triggers ====================

    async def on_event(
        self,
        name: str,
        event_type: str,
        action: str,
        action_params: dict | None = None,
        event_filter: dict[str, Any] | None = None,
        max_executions: int | None = None,
        owner_type: str = "system",
        owner_id: str = "default",
        priority: TaskPriority = TaskPriority.NORMAL,
        callback: Callable[..., Awaitable[Any]] | None = None,
        tags: list[str] | None = None,
        description: str = "",
    ) -> str:
        """Schedule a task to run when an event occurs.

        Args:
            name: Human-readable name
            event_type: Type of event to listen for
            action: Action to execute
            action_params: Parameters for the action
            event_filter: Filter dict - event must match all keys
            max_executions: Maximum number of executions
            owner_type: Owner type
            owner_id: Owner identifier
            priority: Execution priority
            callback: Optional callback
            tags: Optional tags
            description: Task description

        Returns:
            task_id
        """
        task = ScheduledTask(
            name=name,
            description=description,
            owner_type=owner_type,
            owner_id=owner_id,
            action=action,
            action_params=action_params or {},
            priority=priority,
            max_executions=max_executions,
            callback=callback,
            tags=tags or [],
        )

        return await self._event.schedule(task, event_type, event_filter, max_executions)

    async def emit_event(self, event_type: str, data: dict | None = None) -> int:
        """Emit an event to trigger registered handlers.

        Args:
            event_type: Type of event
            data: Event data

        Returns:
            Number of tasks triggered
        """
        return await self._event.emit(event_type, data)

    def list_event_types(self) -> dict[str, str]:
        """List built-in event types."""
        return self._event.list_event_types()

    # ==================== Task Management ====================

    async def get_task(self, task_id: str) -> ScheduledTask | None:
        """Get a task by ID."""
        return await self._persistence.get(task_id)

    async def list_tasks(
        self,
        owner_id: str | None = None,
        owner_type: str | None = None,
        status: TaskStatus | None = None,
        trigger_type: TriggerType | None = None,
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[ScheduledTask]:
        """List scheduled tasks with optional filtering."""
        return await self._persistence.list_tasks(
            owner_id=owner_id,
            owner_type=owner_type,
            status=status,
            trigger_type=trigger_type,
            tags=tags,
            limit=limit,
        )

    async def cancel(self, task_id: str) -> bool:
        """Cancel a scheduled task.

        Works for any trigger type.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        task = await self._persistence.get(task_id)
        if not task:
            return False

        if task.trigger_type == TriggerType.ONCE:
            return await self._timer.cancel(task_id)
        elif task.trigger_type == TriggerType.CRON:
            return await self._cron.cancel(task_id)
        elif task.trigger_type == TriggerType.INTERVAL:
            return await self._interval.cancel(task_id)
        elif task.trigger_type == TriggerType.EVENT:
            return await self._event.cancel(task_id)

        return False

    async def pause(self, task_id: str) -> bool:
        """Pause a recurring task.

        Args:
            task_id: ID of task to pause

        Returns:
            True if paused, False if not found or not pausable
        """
        task = await self._persistence.get(task_id)
        if not task:
            return False

        if task.trigger_type == TriggerType.CRON:
            return await self._cron.pause(task_id)
        elif task.trigger_type == TriggerType.INTERVAL:
            return await self._interval.pause(task_id)
        elif task.trigger_type == TriggerType.EVENT:
            return await self._event.pause(task_id)

        return False

    async def resume(self, task_id: str) -> bool:
        """Resume a paused task.

        Args:
            task_id: ID of task to resume

        Returns:
            True if resumed, False if not found or not paused
        """
        task = await self._persistence.get(task_id)
        if not task:
            return False

        if task.trigger_type == TriggerType.CRON:
            return await self._cron.resume(task_id)
        elif task.trigger_type == TriggerType.INTERVAL:
            return await self._interval.resume(task_id)
        elif task.trigger_type == TriggerType.EVENT:
            return await self._event.resume(task_id)

        return False

    async def delete(self, task_id: str) -> bool:
        """Delete a task completely.

        Args:
            task_id: ID of task to delete

        Returns:
            True if deleted
        """
        # Cancel first to clean up any in-memory state
        await self.cancel(task_id)
        return await self._persistence.delete(task_id)

    # ==================== Action Registration ====================

    def register_action(
        self,
        action_name: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register an action handler.

        Features can register their own actions:
        ```python
        scheduler.register_action("my_feature.process", my_handler)
        ```

        Handler signature: async def handler(params: dict, event_data: dict | None) -> Any

        Args:
            action_name: Unique action identifier
            handler: Async function to handle the action
        """
        self._executor.register_action(action_name, handler)

    def unregister_action(self, action_name: str) -> bool:
        """Unregister an action handler."""
        return self._executor.unregister_action(action_name)

    def list_actions(self) -> list[str]:
        """List all registered action names."""
        return self._executor.list_actions()

    # ==================== Execution Hooks ====================

    def set_execution_hooks(
        self,
        on_execute: Callable[[ScheduledTask], Awaitable[None]] | None = None,
        on_success: Callable[[ScheduledTask, Any], Awaitable[None]] | None = None,
        on_failure: Callable[[ScheduledTask, Exception], Awaitable[None]] | None = None,
    ) -> None:
        """Set execution hooks for observability.

        Args:
            on_execute: Called when task starts executing
            on_success: Called when task succeeds
            on_failure: Called when task fails
        """
        self._executor.set_hooks(on_execute, on_success, on_failure)

    # ==================== Internal Callbacks ====================

    async def _on_task_due(self, task: ScheduledTask) -> None:
        """Called when a timer/cron/interval task is due."""
        try:
            await self._executor.execute(task)
        except Exception as e:
            logger.error(f"Task execution failed: {task.task_id}: {e}")

    async def _on_event_task(
        self, task: ScheduledTask, event_data: dict[str, Any]
    ) -> None:
        """Called when an event triggers a task."""
        try:
            await self._executor.execute(task, event_data)
        except Exception as e:
            logger.error(f"Event task execution failed: {task.task_id}: {e}")

    # ==================== Properties ====================

    @property
    def is_running(self) -> bool:
        """Whether the service is running."""
        return self._started

    @property
    def persistence(self) -> TaskPersistence:
        """Get the persistence layer."""
        return self._persistence

    @property
    def executor(self) -> TaskExecutor:
        """Get the task executor."""
        return self._executor

    @property
    def timer_engine(self) -> TimerEngine:
        """Get the timer engine."""
        return self._timer

    @property
    def cron_scheduler(self) -> CronScheduler:
        """Get the cron scheduler."""
        return self._cron

    @property
    def interval_scheduler(self) -> IntervalScheduler:
        """Get the interval scheduler."""
        return self._interval

    @property
    def event_scheduler(self) -> EventScheduler:
        """Get the event scheduler."""
        return self._event
