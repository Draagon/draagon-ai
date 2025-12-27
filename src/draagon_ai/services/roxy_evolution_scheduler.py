"""Roxy Evolution Scheduler.

This module schedules and runs evolution cycles for Roxy's behavior.
It monitors interaction feedback and triggers evolution when conditions are met.

Usage:
    from draagon_ai.services.roxy_evolution_scheduler import EvolutionScheduler

    scheduler = EvolutionScheduler(
        evolution_service=evolution_service,
        feedback_collector=feedback_collector,
    )

    # Check and maybe evolve
    result = await scheduler.maybe_evolve()

    # Or run scheduled evolution
    await scheduler.start_background_evolution()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Awaitable

from draagon_ai.behaviors.types import Behavior

from .roxy_evolution import RoxyEvolutionService, RoxyEvolutionResult
from .roxy_feedback import RoxyFeedbackCollector, FeedbackStats

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EvolutionSchedulerConfig:
    """Configuration for the evolution scheduler."""

    # Trigger conditions
    min_interactions_for_evolution: int = 50
    min_time_between_evolutions: timedelta = timedelta(hours=1)
    max_time_without_evolution: timedelta = timedelta(days=1)

    # Quality thresholds
    min_improvement_threshold: float = 0.05
    rollback_threshold: float = -0.1  # Rollback if worse by this much

    # Behavior
    auto_apply: bool = False  # Auto-apply evolved prompts
    require_approval: bool = True  # Require human approval
    save_evolved_behavior: bool = True

    # Background scheduling
    check_interval: timedelta = timedelta(minutes=5)
    enable_background: bool = True


@dataclass
class EvolutionRun:
    """Record of an evolution run."""

    run_id: str
    started_at: datetime
    completed_at: datetime | None = None
    result: RoxyEvolutionResult | None = None
    applied: bool = False
    rolled_back: bool = False
    approval_status: str = "pending"  # pending, approved, rejected


# =============================================================================
# Evolution Scheduler
# =============================================================================


class EvolutionScheduler:
    """Schedules and runs evolution cycles.

    This class:
    - Monitors feedback collection
    - Triggers evolution when conditions are met
    - Validates evolved prompts
    - Tracks evolution history
    - Supports rollback

    Example:
        scheduler = EvolutionScheduler(
            evolution_service=evolution_service,
            feedback_collector=feedback_collector,
        )

        # Check if evolution should run
        if scheduler.should_evolve():
            result = await scheduler.evolve()
            if result.success:
                await scheduler.apply_evolution(result)

        # Or run in background
        await scheduler.start_background_evolution()
    """

    def __init__(
        self,
        evolution_service: RoxyEvolutionService,
        feedback_collector: RoxyFeedbackCollector,
        config: EvolutionSchedulerConfig | None = None,
        on_evolution_complete: Callable[[RoxyEvolutionResult], Awaitable[None]] | None = None,
    ):
        """Initialize the scheduler.

        Args:
            evolution_service: The evolution service to use
            feedback_collector: The feedback collector
            config: Optional configuration
            on_evolution_complete: Callback when evolution completes
        """
        self._evolution_service = evolution_service
        self._feedback_collector = feedback_collector
        self.config = config or EvolutionSchedulerConfig()
        self._on_complete = on_evolution_complete

        # State
        self._last_evolution: datetime | None = None
        self._evolution_runs: list[EvolutionRun] = []
        self._current_run: EvolutionRun | None = None

        # Background task
        self._background_task: asyncio.Task | None = None
        self._running = False

        # Prompt backup for rollback
        self._prompt_backup: str | None = None

    @property
    def behavior(self) -> Behavior:
        """Get the current behavior."""
        return self._evolution_service.behavior

    def should_evolve(self) -> bool:
        """Check if evolution should be triggered.

        Returns:
            True if conditions for evolution are met
        """
        stats = self._feedback_collector.get_stats()

        # Check minimum interactions
        if stats.total_interactions < self.config.min_interactions_for_evolution:
            return False

        # Check time since last evolution
        if self._last_evolution:
            time_since = datetime.now() - self._last_evolution
            if time_since < self.config.min_time_between_evolutions:
                return False

        return True

    def must_evolve(self) -> bool:
        """Check if evolution is overdue.

        Returns:
            True if too much time has passed since last evolution
        """
        if self._last_evolution is None:
            return False

        time_since = datetime.now() - self._last_evolution
        return time_since > self.config.max_time_without_evolution

    async def maybe_evolve(self) -> RoxyEvolutionResult | None:
        """Run evolution if conditions are met.

        Returns:
            Evolution result if run, None otherwise
        """
        if not self.should_evolve() and not self.must_evolve():
            return None

        return await self.evolve()

    async def evolve(self) -> RoxyEvolutionResult:
        """Run evolution cycle.

        Returns:
            Evolution result
        """
        import uuid

        # Create run record
        run = EvolutionRun(
            run_id=str(uuid.uuid4())[:8],
            started_at=datetime.now(),
        )
        self._current_run = run
        self._evolution_runs.append(run)

        logger.info(f"Starting evolution run {run.run_id}")

        # Get test cases from feedback
        test_cases = self._feedback_collector.get_evolution_cases(
            min_count=self.config.min_interactions_for_evolution // 2,
        )

        if not test_cases:
            result = RoxyEvolutionResult(
                success=False,
                rejected_reason="Not enough test cases for evolution",
            )
            run.result = result
            run.completed_at = datetime.now()
            self._current_run = None
            return result

        # Backup current prompt
        if self.behavior.prompts:
            self._prompt_backup = self.behavior.prompts.decision_prompt

        # Run evolution
        result = await self._evolution_service.evolve_decision_prompt(
            test_cases=test_cases,
            validate_improvement=True,
        )

        # Update run record
        run.result = result
        run.completed_at = datetime.now()

        if result.success:
            logger.info(
                f"Evolution {run.run_id} succeeded: "
                f"{result.original_fitness:.3f} -> {result.evolved_fitness:.3f}"
            )

            # Check improvement threshold
            if result.improvement < self.config.min_improvement_threshold:
                logger.info(
                    f"Improvement {result.improvement:.3f} below threshold "
                    f"{self.config.min_improvement_threshold}"
                )

            # Auto-apply if configured
            if self.config.auto_apply and not self.config.require_approval:
                await self.apply_evolution(result)
                run.applied = True
                run.approval_status = "auto_approved"

            self._last_evolution = datetime.now()

        else:
            logger.warning(f"Evolution {run.run_id} failed: {result.rejected_reason}")

        # Callback
        if self._on_complete:
            await self._on_complete(result)

        self._current_run = None
        return result

    async def apply_evolution(
        self,
        result: RoxyEvolutionResult | None = None,
        prompt_type: str = "decision",
    ) -> bool:
        """Apply an evolution result.

        Args:
            result: Evolution result to apply (None = most recent)
            prompt_type: Which prompt to update

        Returns:
            True if applied successfully
        """
        if result is None:
            # Get most recent successful result
            for run in reversed(self._evolution_runs):
                if run.result and run.result.success and not run.applied:
                    result = run.result
                    break

        if result is None or not result.success:
            return False

        applied = self._evolution_service.apply_evolution(result, prompt_type)

        if applied:
            # Mark run as applied
            for run in self._evolution_runs:
                if run.result is result:
                    run.applied = True
                    run.approval_status = "approved"
                    break

            # Save feedback for future
            self._feedback_collector.save()

            logger.info(f"Applied evolved {prompt_type} prompt")

        return applied

    async def rollback(self) -> bool:
        """Rollback to the previous prompt.

        Returns:
            True if rollback succeeded
        """
        if self._prompt_backup is None:
            logger.warning("No prompt backup available for rollback")
            return False

        if not self.behavior.prompts:
            return False

        self.behavior.prompts.decision_prompt = self._prompt_backup

        # Mark most recent applied run as rolled back
        for run in reversed(self._evolution_runs):
            if run.applied and not run.rolled_back:
                run.rolled_back = True
                break

        logger.info("Rolled back to previous prompt")

        return True

    async def approve_evolution(self, run_id: str) -> bool:
        """Approve and apply an evolution run.

        Args:
            run_id: ID of the run to approve

        Returns:
            True if approved and applied
        """
        for run in self._evolution_runs:
            if run.run_id == run_id:
                if run.result and run.result.success:
                    run.approval_status = "approved"
                    return await self.apply_evolution(run.result)
        return False

    async def reject_evolution(self, run_id: str) -> bool:
        """Reject an evolution run.

        Args:
            run_id: ID of the run to reject

        Returns:
            True if rejected
        """
        for run in self._evolution_runs:
            if run.run_id == run_id:
                run.approval_status = "rejected"
                return True
        return False

    # =========================================================================
    # Background Evolution
    # =========================================================================

    async def start_background_evolution(self) -> None:
        """Start background evolution monitoring."""
        if self._running:
            logger.warning("Background evolution already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._background_loop())
        logger.info("Started background evolution monitoring")

    async def stop_background_evolution(self) -> None:
        """Stop background evolution monitoring."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
            self._background_task = None
        logger.info("Stopped background evolution monitoring")

    async def _background_loop(self) -> None:
        """Background loop for evolution monitoring."""
        while self._running:
            try:
                # Check if evolution should run
                if self.should_evolve() or self.must_evolve():
                    logger.info("Background evolution triggered")
                    await self.evolve()

                # Wait for next check
                await asyncio.sleep(
                    self.config.check_interval.total_seconds()
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background evolution error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    # =========================================================================
    # History and Stats
    # =========================================================================

    def get_evolution_history(self) -> list[EvolutionRun]:
        """Get evolution run history."""
        return self._evolution_runs.copy()

    def get_pending_approvals(self) -> list[EvolutionRun]:
        """Get runs pending approval."""
        return [
            run for run in self._evolution_runs
            if run.result
            and run.result.success
            and not run.applied
            and run.approval_status == "pending"
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        total_runs = len(self._evolution_runs)
        successful = sum(
            1 for run in self._evolution_runs
            if run.result and run.result.success
        )
        applied = sum(1 for run in self._evolution_runs if run.applied)
        rolled_back = sum(1 for run in self._evolution_runs if run.rolled_back)

        improvements = [
            run.result.improvement
            for run in self._evolution_runs
            if run.result and run.result.success
        ]
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0.0

        return {
            "total_runs": total_runs,
            "successful_runs": successful,
            "applied_runs": applied,
            "rolled_back_runs": rolled_back,
            "pending_approvals": len(self.get_pending_approvals()),
            "avg_improvement": avg_improvement,
            "last_evolution": self._last_evolution.isoformat() if self._last_evolution else None,
            "is_running": self._running,
            "feedback_stats": self._feedback_collector.get_stats().__dict__,
        }
