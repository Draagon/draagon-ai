"""Centralized task executor with retry and error handling."""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Awaitable, Any

from .models import ScheduledTask, TaskStatus, TaskPriority
from .persistence import TaskPersistence

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Centralized task executor with retry and error handling.

    Provides:
    - Action registry for routing tasks to handlers
    - Retry logic with exponential backoff
    - Concurrency control for resource management
    - Priority-based execution ordering
    - Observability (logging, metrics hooks)

    Features:
    - Single point for retry logic
    - Concurrency limits for throttling
    - Action handlers registered by features
    - Execution tracking and history
    """

    def __init__(
        self,
        persistence: TaskPersistence,
        max_concurrent: int = 10,
    ):
        """Initialize task executor.

        Args:
            persistence: Task persistence layer
            max_concurrent: Maximum concurrent task executions
        """
        self._persistence = persistence
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Action handlers registry: action_name -> handler
        self._actions: dict[str, Callable[..., Awaitable[Any]]] = {}

        # Metrics hooks (optional)
        self._on_execute: Callable[[ScheduledTask], Awaitable[None]] | None = None
        self._on_success: Callable[[ScheduledTask, Any], Awaitable[None]] | None = None
        self._on_failure: Callable[[ScheduledTask, Exception], Awaitable[None]] | None = None

    def register_action(
        self,
        action_name: str,
        handler: Callable[..., Awaitable[Any]],
    ) -> None:
        """Register an action handler.

        Features can register their own action handlers:
        ```python
        executor.register_action("my_feature.process", my_handler)
        ```

        Handler signature: async def handler(params: dict, event_data: dict | None) -> Any

        Args:
            action_name: Unique action identifier
            handler: Async function to handle the action
        """
        if action_name in self._actions:
            logger.warning(f"Overwriting existing action handler: {action_name}")
        self._actions[action_name] = handler
        logger.debug(f"Registered action handler: {action_name}")

    def unregister_action(self, action_name: str) -> bool:
        """Unregister an action handler.

        Args:
            action_name: Action to unregister

        Returns:
            True if unregistered, False if not found
        """
        if action_name in self._actions:
            del self._actions[action_name]
            logger.debug(f"Unregistered action handler: {action_name}")
            return True
        return False

    def list_actions(self) -> list[str]:
        """List all registered action names."""
        return list(self._actions.keys())

    def set_hooks(
        self,
        on_execute: Callable[[ScheduledTask], Awaitable[None]] | None = None,
        on_success: Callable[[ScheduledTask, Any], Awaitable[None]] | None = None,
        on_failure: Callable[[ScheduledTask, Exception], Awaitable[None]] | None = None,
    ) -> None:
        """Set execution hooks for observability.

        Args:
            on_execute: Called when task starts executing
            on_success: Called when task succeeds (with result)
            on_failure: Called when task fails (with exception)
        """
        self._on_execute = on_execute
        self._on_success = on_success
        self._on_failure = on_failure

    async def execute(
        self,
        task: ScheduledTask,
        event_data: dict[str, Any] | None = None,
    ) -> Any:
        """Execute a task.

        Args:
            task: The task to execute
            event_data: Optional event data for event-triggered tasks

        Returns:
            Result from the action handler

        Raises:
            Exception: If execution fails after all retries
        """
        async with self._semaphore:
            return await self._execute_with_retry(task, event_data)

    async def _execute_with_retry(
        self,
        task: ScheduledTask,
        event_data: dict[str, Any] | None = None,
    ) -> Any:
        """Execute task with retry logic."""
        # Mark as running
        task.status = TaskStatus.RUNNING
        await self._persistence.save(task)

        # Notify hooks
        if self._on_execute:
            try:
                await self._on_execute(task)
            except Exception as e:
                logger.warning(f"on_execute hook failed: {e}")

        try:
            # Execute via callback or registered action
            if task.callback:
                result = await task.callback(task, event_data)
            elif task.action in self._actions:
                handler = self._actions[task.action]
                result = await handler(task.action_params, event_data)
            else:
                raise ValueError(f"Unknown action: {task.action}")

            # Success
            await self._handle_success(task, result)
            return result

        except Exception as e:
            # Failure
            await self._handle_failure(task, e)
            raise

    async def _handle_success(self, task: ScheduledTask, result: Any) -> None:
        """Handle successful task execution."""
        # For one-shot tasks, mark as completed
        # For recurring tasks, mark as pending (scheduler will update next_run)
        from .models import TriggerType

        if task.trigger_type == TriggerType.ONCE:
            task.status = TaskStatus.COMPLETED
        else:
            task.status = TaskStatus.PENDING

        task.last_result = result
        task.last_error = None
        task.retry_count = 0
        task.last_run = datetime.now()
        task.execution_count += 1

        await self._persistence.save(task)

        logger.info(f"Task succeeded: {task.name} ({task.task_id})")

        # Notify hooks
        if self._on_success:
            try:
                await self._on_success(task, result)
            except Exception as e:
                logger.warning(f"on_success hook failed: {e}")

    async def _handle_failure(self, task: ScheduledTask, error: Exception) -> None:
        """Handle failed task execution."""
        task.last_error = str(error)
        task.retry_count += 1
        task.last_run = datetime.now()

        logger.error(f"Task failed: {task.name} ({task.task_id}): {error}")

        if task.should_retry():
            # Schedule retry with exponential backoff
            backoff = task.retry_delay * (2 ** (task.retry_count - 1))
            # Cap at 5 minutes
            if backoff > timedelta(minutes=5):
                backoff = timedelta(minutes=5)

            task.next_run = datetime.now() + backoff
            task.status = TaskStatus.PENDING

            logger.warning(
                f"Task {task.task_id} will retry in {backoff} "
                f"(attempt {task.retry_count}/{task.max_retries})"
            )
        else:
            # Exceeded retries
            task.status = TaskStatus.FAILED
            logger.error(
                f"Task {task.task_id} failed after {task.retry_count} retries"
            )

        await self._persistence.save(task)

        # Notify hooks
        if self._on_failure:
            try:
                await self._on_failure(task, error)
            except Exception as e:
                logger.warning(f"on_failure hook failed: {e}")

    async def execute_by_priority(
        self,
        tasks: list[ScheduledTask],
        event_data: dict[str, Any] | None = None,
    ) -> list[tuple[ScheduledTask, Any | Exception]]:
        """Execute multiple tasks ordered by priority.

        Args:
            tasks: Tasks to execute
            event_data: Optional event data

        Returns:
            List of (task, result_or_exception) tuples
        """
        # Sort by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority.value)

        results = []
        for task in sorted_tasks:
            try:
                result = await self.execute(task, event_data)
                results.append((task, result))
            except Exception as e:
                results.append((task, e))

        return results

    async def execute_parallel(
        self,
        tasks: list[ScheduledTask],
        event_data: dict[str, Any] | None = None,
    ) -> list[tuple[ScheduledTask, Any | Exception]]:
        """Execute multiple tasks in parallel (respecting concurrency limit).

        Args:
            tasks: Tasks to execute
            event_data: Optional event data

        Returns:
            List of (task, result_or_exception) tuples
        """

        async def execute_one(task: ScheduledTask) -> tuple[ScheduledTask, Any | Exception]:
            try:
                result = await self.execute(task, event_data)
                return (task, result)
            except Exception as e:
                return (task, e)

        results = await asyncio.gather(
            *[execute_one(task) for task in tasks],
            return_exceptions=False,
        )
        return list(results)

    @property
    def concurrent_limit(self) -> int:
        """Get the concurrency limit."""
        return self._max_concurrent

    @property
    def available_slots(self) -> int:
        """Get number of available execution slots."""
        # Semaphore doesn't expose internal count directly
        # This is an approximation
        return self._semaphore._value  # type: ignore


# Built-in action handlers that can be registered

async def notify_action(params: dict, event_data: dict | None = None) -> dict:
    """Built-in notify action.

    Params:
        message: str - Message to send
        priority: str - Priority level (optional)
        target: str - Target user/device (optional)

    Returns:
        dict with notification details
    """
    message = params.get("message", "Notification")
    priority = params.get("priority", "normal")
    target = params.get("target")

    logger.info(f"NOTIFY [{priority}]: {message}" + (f" (target: {target})" if target else ""))

    return {
        "action": "notify",
        "message": message,
        "priority": priority,
        "target": target,
        "delivered": True,
    }


async def log_action(params: dict, event_data: dict | None = None) -> dict:
    """Built-in log action.

    Params:
        message: str - Message to log
        level: str - Log level (debug, info, warning, error)

    Returns:
        dict confirming log
    """
    message = params.get("message", "")
    level = params.get("level", "info")

    log_fn = getattr(logger, level, logger.info)
    log_fn(f"SCHEDULED LOG: {message}")

    return {"action": "log", "level": level, "message": message}


def register_builtin_actions(executor: TaskExecutor) -> None:
    """Register built-in action handlers.

    Args:
        executor: TaskExecutor to register actions on
    """
    executor.register_action("notify", notify_action)
    executor.register_action("log", log_action)
