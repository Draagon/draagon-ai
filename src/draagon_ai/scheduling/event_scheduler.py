"""Event-triggered scheduler."""

import logging
from datetime import datetime
from typing import Callable, Awaitable, Any

from .models import ScheduledTask, TaskStatus, TriggerType
from .persistence import TaskPersistence

logger = logging.getLogger(__name__)


# Built-in event types (features can define their own)
BUILTIN_EVENTS = {
    "conversation_start": "When a conversation begins",
    "conversation_end": "When a conversation ends",
    "memory_stored": "When a memory is stored",
    "memory_promoted": "When a memory is promoted to a higher layer",
    "memory_accessed": "When a memory is retrieved",
    "user_login": "When a user logs in",
    "user_logout": "When a user logs out",
    "time_morning": "At morning time (configurable)",
    "time_evening": "At evening time (configurable)",
    "idle_start": "When system becomes idle",
    "idle_end": "When system becomes active after idle",
    "error_occurred": "When an error happens",
    "task_completed": "When any scheduled task completes",
    "task_failed": "When any scheduled task fails",
}


class EventScheduler:
    """Event-triggered scheduler.

    Allows tasks to be triggered by system events rather than time.
    Useful for reactive patterns like "after conversation ends, summarize".

    Features:
    - Built-in event types for common scenarios
    - Custom event types for feature-specific triggers
    - Event filtering (only trigger on matching events)
    - Max execution limits
    - Pause/resume support
    """

    def __init__(self, persistence: TaskPersistence):
        """Initialize event scheduler.

        Args:
            persistence: Task persistence layer
        """
        self._persistence = persistence
        self._handlers: dict[str, list[str]] = {}  # event_type -> [task_ids]
        self._callback: Callable[[ScheduledTask, dict], Awaitable[None]] | None = None

    def set_callback(
        self, callback: Callable[[ScheduledTask, dict], Awaitable[None]]
    ) -> None:
        """Set the callback for when an event triggers a task.

        Args:
            callback: Async function called with (task, event_data) when triggered
        """
        self._callback = callback

    async def schedule(
        self,
        task: ScheduledTask,
        event_type: str,
        event_filter: dict[str, Any] | None = None,
        max_executions: int | None = None,
    ) -> str:
        """Schedule a task to run when an event occurs.

        Args:
            task: The task to execute
            event_type: Type of event to listen for (see BUILTIN_EVENTS)
            event_filter: Optional filter dict - event must match all keys
            max_executions: Limit executions (None = unlimited)

        Returns:
            task_id
        """
        # Configure task
        task.trigger_type = TriggerType.EVENT
        task.trigger_config = {
            "event_type": event_type,
            "filter": event_filter or {},
        }
        task.max_executions = max_executions
        task.next_run = None  # Event-triggered, no scheduled time

        # Persist
        await self._persistence.save(task)

        # Register in-memory handler
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(task.task_id)

        logger.debug(
            f"Event scheduled: {task.name} ({task.task_id}) "
            f"on event '{event_type}'"
        )

        return task.task_id

    async def emit(self, event_type: str, event_data: dict[str, Any] | None = None) -> int:
        """Emit an event, triggering registered handlers.

        Args:
            event_type: Type of event to emit
            event_data: Data associated with the event

        Returns:
            Number of tasks triggered
        """
        event_data = event_data or {}
        triggered = 0

        # Get handlers from persistence (in case of restart)
        tasks = await self._persistence.get_by_event(event_type)

        for task in tasks:
            if task.status == TaskStatus.PAUSED:
                continue

            # Check filter
            if not self._matches_filter(task, event_data):
                continue

            # Execute via callback
            if self._callback:
                try:
                    await self._callback(task, event_data)
                    triggered += 1
                except Exception as e:
                    logger.error(f"Event task {task.task_id} callback failed: {e}")
                    continue

            # Update task state
            task.last_run = datetime.now()
            task.execution_count += 1

            # Check if max executions reached
            if task.has_reached_max_executions():
                task.status = TaskStatus.COMPLETED
                # Remove from in-memory handlers
                if event_type in self._handlers:
                    try:
                        self._handlers[event_type].remove(task.task_id)
                    except ValueError:
                        pass
                logger.info(f"Event task completed (max executions): {task.name}")

            await self._persistence.save(task)

        if triggered > 0:
            logger.debug(f"Event '{event_type}' triggered {triggered} tasks")

        return triggered

    def _matches_filter(self, task: ScheduledTask, event_data: dict[str, Any]) -> bool:
        """Check if event data matches task's filter.

        Args:
            task: The task with filter config
            event_data: Data from the emitted event

        Returns:
            True if event matches filter (or no filter)
        """
        event_filter = task.trigger_config.get("filter", {})
        if not event_filter:
            return True

        for key, expected in event_filter.items():
            actual = event_data.get(key)
            if actual != expected:
                return False

        return True

    async def pause(self, task_id: str) -> bool:
        """Pause an event task.

        Args:
            task_id: ID of task to pause

        Returns:
            True if paused, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.EVENT:
            task.status = TaskStatus.PAUSED
            await self._persistence.save(task)
            logger.info(f"Event task paused: {task.name}")
            return True
        return False

    async def resume(self, task_id: str) -> bool:
        """Resume a paused event task.

        Args:
            task_id: ID of task to resume

        Returns:
            True if resumed, False if not found or not paused
        """
        task = await self._persistence.get(task_id)
        if task and task.status == TaskStatus.PAUSED:
            task.status = TaskStatus.PENDING
            await self._persistence.save(task)
            logger.info(f"Event task resumed: {task.name}")
            return True
        return False

    async def update_filter(
        self,
        task_id: str,
        event_filter: dict[str, Any],
    ) -> bool:
        """Update the filter for an event task.

        Args:
            task_id: ID of task to update
            event_filter: New filter dict

        Returns:
            True if updated, False if not found
        """
        task = await self._persistence.get(task_id)
        if not task or task.trigger_type != TriggerType.EVENT:
            return False

        task.trigger_config["filter"] = event_filter
        await self._persistence.save(task)
        logger.info(f"Event task filter updated: {task.name}")
        return True

    async def cancel(self, task_id: str) -> bool:
        """Cancel an event task.

        Args:
            task_id: ID of task to cancel

        Returns:
            True if cancelled, False if not found
        """
        task = await self._persistence.get(task_id)
        if task and task.trigger_type == TriggerType.EVENT:
            task.status = TaskStatus.CANCELLED

            # Remove from in-memory handlers
            event_type = task.trigger_config.get("event_type")
            if event_type and event_type in self._handlers:
                try:
                    self._handlers[event_type].remove(task_id)
                except ValueError:
                    pass

            await self._persistence.save(task)
            logger.info(f"Event task cancelled: {task.name}")
            return True
        return False

    async def restore_handlers(self) -> int:
        """Restore in-memory handlers from persistence.

        Call this after restart to rebuild the handler registry.

        Returns:
            Number of handlers restored
        """
        self._handlers.clear()
        restored = 0

        # Get all event-triggered tasks
        tasks = await self._persistence.list_tasks(trigger_type=TriggerType.EVENT)

        for task in tasks:
            if task.status not in (TaskStatus.PENDING, TaskStatus.PAUSED):
                continue

            event_type = task.trigger_config.get("event_type")
            if event_type:
                if event_type not in self._handlers:
                    self._handlers[event_type] = []
                self._handlers[event_type].append(task.task_id)
                restored += 1

        if restored > 0:
            logger.info(f"Restored {restored} event handlers from persistence")

        return restored

    def list_event_types(self) -> dict[str, str]:
        """List built-in event types.

        Returns:
            Dict of event_type -> description
        """
        return BUILTIN_EVENTS.copy()

    def get_handlers_for_event(self, event_type: str) -> list[str]:
        """Get task IDs registered for an event type.

        Args:
            event_type: The event type

        Returns:
            List of task IDs
        """
        return self._handlers.get(event_type, []).copy()
