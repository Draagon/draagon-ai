"""Evolution Scheduler.

This module schedules and runs evolution cycles for behaviors.
It monitors interaction feedback and triggers evolution when conditions are met.

Usage:
    from draagon_ai.services.evolution_scheduler import EvolutionScheduler

    scheduler = EvolutionScheduler(
        evolution_service=evolution_service,
        feedback_collector=feedback_collector,
    )

    # Check and maybe evolve
    result = await scheduler.maybe_evolve()

    # Or run scheduled evolution
    await scheduler.start_background_evolution()
"""

# Re-export from roxy_evolution_scheduler for backward compatibility
from .roxy_evolution_scheduler import (
    EvolutionScheduler,
    EvolutionSchedulerConfig,
    EvolutionRun,
)

__all__ = [
    "EvolutionScheduler",
    "EvolutionSchedulerConfig",
    "EvolutionRun",
]
