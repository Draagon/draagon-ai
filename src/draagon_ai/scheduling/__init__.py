"""Draagon-AI Scheduling System.

A robust, async-native scheduling framework supporting:
- One-shot timers (high-precision asyncio)
- Cron schedules (recurring with croniter)
- Interval schedules (simple recurring)
- Event-triggered tasks (reactive)

Usage:
    from draagon_ai.scheduling import SchedulingService, TaskPriority

    scheduler = SchedulingService(persistence)
    await scheduler.start()

    # One-shot timer
    await scheduler.set_timer(
        name="Pasta timer",
        duration=timedelta(minutes=5),
        action="notify",
        action_params={"message": "Pasta is ready!"},
    )

    # Recurring cron
    await scheduler.schedule_cron(
        name="Daily briefing",
        expression="0 9 * * *",
        action="briefing",
    )

    # Event-triggered
    await scheduler.on_event(
        name="Summarize conversation",
        event_type="conversation_end",
        action="summarize",
    )
"""

from .models import (
    ScheduledTask,
    TaskStatus,
    TaskPriority,
    TriggerType,
)
from .persistence import TaskPersistence, InMemoryPersistence, QdrantPersistence
from .timer_engine import TimerEngine
from .cron_scheduler import CronScheduler
from .interval_scheduler import IntervalScheduler
from .event_scheduler import EventScheduler
from .executor import TaskExecutor
from .service import SchedulingService

__all__ = [
    # Models
    "ScheduledTask",
    "TaskStatus",
    "TaskPriority",
    "TriggerType",
    # Persistence
    "TaskPersistence",
    "InMemoryPersistence",
    "QdrantPersistence",
    # Engines
    "TimerEngine",
    "CronScheduler",
    "IntervalScheduler",
    "EventScheduler",
    # Execution
    "TaskExecutor",
    # Facade
    "SchedulingService",
]
