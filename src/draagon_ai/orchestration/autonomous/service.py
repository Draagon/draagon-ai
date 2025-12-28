"""Autonomous Agent Service.

This is the core autonomous agent implementation that runs during idle time
and makes decisions about what to research, verify, or learn.

The service is designed to be application-agnostic through the use of
protocol adapters (LLMProvider, SearchProvider, etc.).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Any

from .types import (
    ActionLog,
    ActionResult,
    ActionTier,
    ActionType,
    ApprovedAction,
    AutonomousConfig,
    AutonomousContext,
    ContextProvider,
    HarmCheck,
    LLMProvider,
    MemoryStoreProvider,
    NotificationProvider,
    ProposedAction,
    SafetyCheck,
    SearchProvider,
    SelfMonitoringFinding,
    SelfMonitoringResult,
)
from .prompts import (
    AUTONOMOUS_AGENT_SYSTEM_PROMPT,
    HARM_CHECK_PROMPT,
    REFLECTION_PROMPT,
    RESEARCH_SYNTHESIS_PROMPT,
    SELF_MONITORING_PROMPT,
    SEMANTIC_SAFETY_PROMPT,
    VERIFY_ASSESSMENT_PROMPT,
)

logger = logging.getLogger(__name__)


class AutonomousAgentService:
    """Background autonomous agent that runs during idle time.

    This service implements the autonomous cognitive loop:
    1. Gather context (via ContextProvider)
    2. Propose actions (via LLM)
    3. Filter through guardrails
    4. Execute approved actions
    5. Self-monitor and learn

    Applications integrate by providing implementations of the
    required protocols (LLMProvider, SearchProvider, etc.).
    """

    def __init__(
        self,
        llm: LLMProvider,
        config: AutonomousConfig | None = None,
        search: SearchProvider | None = None,
        memory_store: MemoryStoreProvider | None = None,
        context_provider: ContextProvider | None = None,
        notification_provider: NotificationProvider | None = None,
    ):
        """Initialize the autonomous agent.

        Args:
            llm: LLM provider for generating decisions and assessments.
            config: Agent configuration.
            search: Optional search provider for research actions.
            memory_store: Optional storage for logs and state.
            context_provider: Optional provider for gathering context.
            notification_provider: Optional provider for user notifications.
        """
        self._llm = llm
        self.config = config or AutonomousConfig()
        self._search = search
        self._memory_store = memory_store
        self._context_provider = context_provider
        self._notification_provider = notification_provider

        # State tracking
        self._actions_today: int = 0
        self._last_action_reset: datetime = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._recent_action_types: list[ActionType] = []
        self._action_logs: list[ActionLog] = []

        # Background task
        self._background_task: asyncio.Task | None = None
        self._running = False

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the autonomous agent background loop."""
        if not self.config.enabled:
            logger.info("Autonomous agent disabled, not starting")
            return

        if self._running:
            logger.warning("Autonomous agent already running")
            return

        self._running = True
        self._background_task = asyncio.create_task(self._run_loop())
        logger.info("Autonomous agent started")

    async def stop(self) -> None:
        """Stop the autonomous agent."""
        self._running = False
        if self._background_task:
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        logger.info("Autonomous agent stopped")

    async def _run_loop(self) -> None:
        """Main background loop."""
        while self._running:
            try:
                # Check if within active hours
                now = datetime.now()
                if not self._is_within_active_hours(now):
                    logger.debug("Outside active hours, skipping cycle")
                    await asyncio.sleep(60 * 10)  # Check again in 10 min
                    continue

                # Run autonomous cycle
                await self.run_cycle()

                # Sleep until next cycle
                await asyncio.sleep(60 * self.config.cycle_interval_minutes)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in autonomous cycle: {e}")
                await asyncio.sleep(60 * 5)  # Wait 5 min on error

    def _is_within_active_hours(self, now: datetime) -> bool:
        """Check if current time is within active hours."""
        return self.config.active_hours_start <= now.hour < self.config.active_hours_end

    # =========================================================================
    # Main Cycle
    # =========================================================================

    async def run_cycle(self) -> list[ActionResult]:
        """Run one autonomous decision cycle.

        Returns:
            List of action results.
        """
        # Reset daily budget if needed
        self._check_daily_reset()

        # Check budget
        if self._actions_today >= self.config.daily_action_budget:
            logger.info("Daily action budget exhausted")
            return []

        # Gather context
        context = await self._gather_context()

        # Ask LLM: What would be interesting or useful to do?
        proposed_actions = await self._generate_proposals(context)

        if not proposed_actions:
            logger.debug("No actions proposed")
            return []

        # Filter through guardrails
        approved_actions = await self._filter_through_guardrails(proposed_actions)

        # Execute approved actions (respecting limits)
        results = []
        executed_actions: list[tuple[ProposedAction, ActionResult]] = []

        for action in approved_actions[: self.config.max_actions_per_cycle]:
            if self._actions_today >= self.config.daily_action_budget:
                break

            if self.config.shadow_mode:
                # Shadow mode: log but don't execute
                logger.info(
                    f"[SHADOW] Would execute: {action.action.action_type.value} - "
                    f"{action.action.description}"
                )
                await self._log_action(
                    action.action,
                    ActionResult(success=True, outcome="Shadow mode - not executed"),
                    shadow=True,
                )
            else:
                result = await self._execute_action(action)
                results.append(result)
                executed_actions.append((action.action, result))
                await self._log_action(action.action, result)
                self._actions_today += 1

        # Self-monitoring: review what we did and flag any issues
        if self.config.enable_self_monitoring and executed_actions:
            await self._self_monitor(executed_actions)

        return results

    def _check_daily_reset(self) -> None:
        """Reset daily counters if it's a new day."""
        now = datetime.now()
        today = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if today > self._last_action_reset:
            self._actions_today = 0
            self._last_action_reset = today
            logger.info("Reset daily action counter")

    # =========================================================================
    # Context Gathering
    # =========================================================================

    async def _gather_context(self) -> AutonomousContext:
        """Gather context for autonomous decision making."""
        if self._context_provider:
            return await self._context_provider.gather_context()

        # Default minimal context
        now = datetime.now()
        return AutonomousContext(
            personality_context="I am a helpful AI assistant.",
            trait_values={
                "curiosity_intensity": 0.7,
                "verification_threshold": 0.5,
                "proactive_helpfulness": 0.6,
            },
            current_time=now,
            day_of_week=now.strftime("%A"),
            recent_actions=[
                f"{log.action_type}: {log.description}"
                for log in self._action_logs[-10:]
            ],
            daily_budget_remaining=self.config.daily_action_budget - self._actions_today,
            available_action_types=[
                ActionType.RESEARCH,
                ActionType.VERIFY,
                ActionType.REFLECT,
                ActionType.NOTE_QUESTION,
                ActionType.PREPARE_SUGGESTION,
                ActionType.UPDATE_BELIEF,
                ActionType.REST,
            ],
        )

    # =========================================================================
    # Action Generation
    # =========================================================================

    async def _generate_proposals(
        self,
        context: AutonomousContext,
    ) -> list[ProposedAction]:
        """Ask LLM what actions to take."""
        # Build prompt with context
        prompt = AUTONOMOUS_AGENT_SYSTEM_PROMPT.format(
            personality_context=context.personality_context,
            curiosity_intensity=context.trait_values.get("curiosity_intensity", 0.7),
            verification_threshold=context.trait_values.get("verification_threshold", 0.5),
            proactive_helpfulness=context.trait_values.get("proactive_helpfulness", 0.6),
            recent_conversations_summary=context.recent_conversations_summary or "None",
            pending_questions="\n".join(context.pending_questions) or "None",
            unverified_claims="\n".join(context.unverified_claims) or "None",
            knowledge_gaps="\n".join(context.knowledge_gaps) or "None",
            conflicts="\n".join(context.conflicting_beliefs) or "None",
            upcoming_events=context.upcoming_events_summary or "None",
            current_day=context.day_of_week,
            current_time=context.current_time.strftime("%I:%M %p"),
            recent_autonomous_actions="\n".join(context.recent_actions) or "None",
        )

        try:
            response = await self._llm.generate(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.7,
            )
            return self._parse_proposals(response)

        except Exception as e:
            logger.error(f"Error generating action proposals: {e}")
            return []

    def _parse_proposals(self, response: str) -> list[ProposedAction]:
        """Parse LLM response into ProposedAction objects."""
        proposals = []

        try:
            # Extract JSON from response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            data = json.loads(json_str.strip())

            for action in data.get("proposed_actions", []):
                try:
                    action_type = ActionType(action.get("type", "rest"))
                except ValueError:
                    action_type = ActionType.REST

                proposals.append(
                    ProposedAction(
                        action_type=action_type,
                        description=action.get("description", ""),
                        reasoning=action.get("reasoning", ""),
                        risk_tier=ActionTier(action.get("risk_tier", 0)),
                        reversible=action.get("reversible", True),
                        estimated_time_seconds=action.get("estimated_time_seconds", 30),
                        target_entity=action.get("target_entity"),
                    )
                )

        except Exception as e:
            logger.warning(f"Error parsing proposals: {e}")

        return proposals

    # =========================================================================
    # Guardrails
    # =========================================================================

    async def _filter_through_guardrails(
        self,
        proposed_actions: list[ProposedAction],
    ) -> list[ApprovedAction]:
        """Apply all guardrail checks to proposed actions."""
        approved = []

        for action in proposed_actions:
            # Layer 1: Tier classification
            if not self._is_allowed_tier(action):
                await self._log_blocked(action, "tier_violation")
                continue

            # Layer 2: Rate limiting (consecutive same type)
            if self._exceeds_rate_limit(action):
                await self._log_blocked(action, "rate_limit")
                continue

            # Layer 3: Harm check
            harm_check = await self._check_for_harm(action)
            if harm_check.potentially_harmful:
                await self._log_blocked(action, f"harm_risk: {harm_check.reason}")
                continue

            # Layer 4: Semantic safety (if enabled)
            if self.config.require_semantic_safety_check:
                safety_check = await self._semantic_safety_check(action)
                if not safety_check.is_safe:
                    await self._log_blocked(action, f"semantic_unsafe: {safety_check.reason}")
                    continue

            approved.append(
                ApprovedAction(
                    action=action,
                    approved_at=datetime.now(),
                    guardrails_passed=["tier", "rate", "harm", "semantic"],
                )
            )

        return approved

    def _is_allowed_tier(self, action: ProposedAction) -> bool:
        """Check if action tier is allowed for autonomous execution."""
        # Only Tier 0 and Tier 1 are allowed autonomously
        max_allowed = ActionTier.TIER_1
        risk_tolerance = 0.3  # Could be made configurable

        # If risk tolerance is very low, only allow Tier 0
        if risk_tolerance < 0.3:
            max_allowed = ActionTier.TIER_0

        return action.risk_tier.value <= max_allowed.value

    def _exceeds_rate_limit(self, action: ProposedAction) -> bool:
        """Check if we've done too many of the same action type."""
        recent_same = [
            t
            for t in self._recent_action_types[-self.config.max_consecutive_same_type :]
            if t == action.action_type
        ]
        return len(recent_same) >= self.config.max_consecutive_same_type

    async def _check_for_harm(self, action: ProposedAction) -> HarmCheck:
        """LLM-based harm assessment."""
        prompt = HARM_CHECK_PROMPT.format(
            action_type=action.action_type.value,
            description=action.description,
            reasoning=action.reasoning,
        )

        try:
            response = await self._llm.generate(
                prompt=prompt,
                max_tokens=200,
                temperature=0.0,
            )

            # Extract JSON
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            else:
                json_str = response

            data = json.loads(json_str.strip())
            return HarmCheck(
                potentially_harmful=data.get("potentially_harmful", False),
                reason=data.get("reason"),
                confidence=data.get("confidence", 0.5),
            )

        except Exception as e:
            logger.warning(f"Harm check failed, defaulting to safe: {e}")
            return HarmCheck(
                potentially_harmful=action.risk_tier.value > 0,
                reason="Harm check failed, using tier-based default",
            )

    async def _semantic_safety_check(self, action: ProposedAction) -> SafetyCheck:
        """Final semantic safety check."""
        prompt = SEMANTIC_SAFETY_PROMPT.format(
            description=action.description,
            reasoning=action.reasoning,
        )

        try:
            response = await self._llm.generate(
                prompt=prompt,
                max_tokens=100,
                temperature=0.0,
            )

            is_safe = response.strip().upper().startswith("SAFE")
            return SafetyCheck(
                is_safe=is_safe,
                reason=response.strip(),
            )

        except Exception as e:
            logger.warning(f"Semantic safety check failed: {e}")
            return SafetyCheck(
                is_safe=action.risk_tier == ActionTier.TIER_0,
                reason=f"Check failed: {e}",
            )

    async def _log_blocked(self, action: ProposedAction, reason: str) -> None:
        """Log a blocked action for transparency."""
        logger.info(f"Blocked action: {action.action_type.value} - {reason}")

        if self.config.log_all_proposals:
            log = ActionLog(
                action_id=str(uuid.uuid4()),
                action_type=action.action_type.value,
                description=action.description,
                reasoning=action.reasoning,
                started_at=datetime.now(),
                completed_at=datetime.now(),
                blocked=True,
                blocked_reason=reason,
            )
            self._action_logs.append(log)

            if self._memory_store and self.config.persist_logs:
                await self._memory_store.store_log(log)

    # =========================================================================
    # Action Execution
    # =========================================================================

    async def _execute_action(self, approved: ApprovedAction) -> ActionResult:
        """Execute an approved action."""
        action = approved.action

        try:
            if action.action_type == ActionType.RESEARCH:
                return await self._execute_research(action)
            elif action.action_type == ActionType.VERIFY:
                return await self._execute_verify(action)
            elif action.action_type == ActionType.REFLECT:
                return await self._execute_reflect(action)
            elif action.action_type == ActionType.NOTE_QUESTION:
                return await self._execute_note_question(action)
            elif action.action_type == ActionType.PREPARE_SUGGESTION:
                return await self._execute_prepare_suggestion(action)
            elif action.action_type == ActionType.UPDATE_BELIEF:
                return await self._execute_update_belief(action)
            elif action.action_type == ActionType.REST:
                return ActionResult(success=True, outcome="Resting - no action taken")
            else:
                return ActionResult(
                    success=False, error=f"Unknown action type: {action.action_type}"
                )

        except Exception as e:
            logger.error(f"Error executing action {action.action_type}: {e}")
            return ActionResult(success=False, error=str(e))

    async def _execute_research(self, action: ProposedAction) -> ActionResult:
        """Execute a research/learning action."""
        if not self._search:
            return ActionResult(
                success=False,
                error="No search provider available",
            )

        try:
            search_results = await self._search.search(action.description)

            if not search_results:
                return ActionResult(
                    success=True,
                    outcome="No results found",
                )

            # Synthesize what was learned
            synthesis = await self._llm.generate(
                prompt=RESEARCH_SYNTHESIS_PROMPT.format(
                    topic=action.description,
                    search_results=search_results[:3000],
                ),
                max_tokens=200,
            )

            # Store as learned insight
            if self._memory_store:
                await self._memory_store.store(
                    content=f"Learned: {synthesis}",
                    memory_type="insight",
                    importance=0.6,
                )

            return ActionResult(
                success=True,
                outcome=f"Researched and learned about: {action.description[:50]}",
                learned=synthesis,
                belief_updated=True,
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_verify(self, action: ProposedAction) -> ActionResult:
        """Verify a claim via web search."""
        if not self._search:
            return ActionResult(
                success=False,
                error="No search provider available",
            )

        try:
            search_results = await self._search.search(f"verify: {action.description}")

            if not search_results:
                return ActionResult(
                    success=True,
                    outcome="Could not find verification sources",
                )

            # Assess verification
            assessment = await self._llm.generate(
                prompt=VERIFY_ASSESSMENT_PROMPT.format(
                    claim=action.description,
                    search_results=search_results[:2000],
                ),
                max_tokens=150,
            )

            return ActionResult(
                success=True,
                outcome=f"Verification result: {assessment[:100]}",
                learned=assessment,
            )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_reflect(self, action: ProposedAction) -> ActionResult:
        """Reflect on behavior and traits."""
        try:
            # Get current traits from context
            context = await self._gather_context()
            traits_str = "\n".join(
                [f"- {name}: {value:.2f}" for name, value in context.trait_values.items()]
            )

            response = await self._llm.generate(
                prompt=REFLECTION_PROMPT.format(
                    traits=traits_str,
                    recent_activity="\n".join(context.recent_actions) or "No recent activity.",
                ),
                max_tokens=300,
            )

            # Parse reflection
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                else:
                    json_str = response

                data = json.loads(json_str.strip())
                summary = data.get("summary", "Reflection complete")

                return ActionResult(
                    success=True,
                    outcome=summary,
                    belief_updated=bool(data.get("trait_adjustments")),
                )

            except json.JSONDecodeError:
                return ActionResult(
                    success=True,
                    outcome="Reflected, but couldn't parse adjustments",
                )

        except Exception as e:
            return ActionResult(success=False, error=str(e))

    async def _execute_note_question(self, action: ProposedAction) -> ActionResult:
        """Note a question to ask later."""
        if self._memory_store:
            try:
                await self._memory_store.store(
                    content=f"Question to ask: {action.description}",
                    memory_type="insight",
                    importance=0.5,
                )
            except Exception as e:
                logger.warning(f"Failed to store question: {e}")

        return ActionResult(
            success=True,
            outcome=f"Noted question: {action.description[:50]}",
        )

    async def _execute_prepare_suggestion(self, action: ProposedAction) -> ActionResult:
        """Prepare a suggestion (without announcing)."""
        if self._memory_store:
            try:
                await self._memory_store.store(
                    content=f"Suggestion to share: {action.description}",
                    memory_type="insight",
                    importance=0.6,
                )
            except Exception as e:
                logger.warning(f"Failed to store suggestion: {e}")

        return ActionResult(
            success=True,
            outcome=f"Prepared suggestion: {action.description[:50]}",
        )

    async def _execute_update_belief(self, action: ProposedAction) -> ActionResult:
        """Update a belief based on new evidence."""
        if self._memory_store:
            try:
                await self._memory_store.store(
                    content=action.description,
                    memory_type="fact",
                    importance=0.7,
                )
            except Exception as e:
                logger.warning(f"Failed to store belief: {e}")

        return ActionResult(
            success=True,
            outcome=f"Updated belief: {action.description[:50]}",
            belief_updated=True,
        )

    async def _log_action(
        self,
        action: ProposedAction,
        result: ActionResult,
        shadow: bool = False,
    ) -> None:
        """Log an executed action."""
        log = ActionLog(
            action_id=str(uuid.uuid4()),
            action_type=action.action_type.value,
            description=action.description,
            reasoning=action.reasoning,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            success=result.success,
            outcome=result.outcome,
            error=result.error,
        )
        self._action_logs.append(log)
        self._recent_action_types.append(action.action_type)

        # Persist to storage
        if self._memory_store and self.config.persist_logs:
            await self._memory_store.store_log(log)

        # Keep in-memory history bounded
        if len(self._action_logs) > 1000:
            self._action_logs = self._action_logs[-500:]
        if len(self._recent_action_types) > 50:
            self._recent_action_types = self._recent_action_types[-25:]

        if shadow:
            logger.debug(f"[SHADOW] Logged action: {action.action_type.value}")
        else:
            logger.info(
                f"Executed action: {action.action_type.value} - "
                f"{result.outcome or result.error}"
            )

    # =========================================================================
    # Self-Monitoring
    # =========================================================================

    async def _self_monitor(
        self,
        executed_actions: list[tuple[ProposedAction, ActionResult]],
    ) -> None:
        """Review the cycle and flag any issues."""
        # Build summaries
        actions_summary = "\n".join(
            [
                f"- {action.action_type.value}: {action.description}"
                for action, _ in executed_actions
            ]
        )
        results_summary = "\n".join(
            [
                f"- {'SUCCESS' if result.success else 'FAILED'}: {result.outcome or result.error}"
                for _, result in executed_actions
            ]
        )

        try:
            response = await self._llm.generate(
                prompt=SELF_MONITORING_PROMPT.format(
                    actions_summary=actions_summary,
                    results_summary=results_summary,
                ),
                max_tokens=500,
                temperature=0.3,
            )

            # Parse response
            try:
                if "```json" in response:
                    json_str = response.split("```json")[1].split("```")[0]
                elif "```" in response:
                    json_str = response.split("```")[1].split("```")[0]
                else:
                    json_str = response

                data = json.loads(json_str.strip())

                # Process findings
                for finding_data in data.get("findings", []):
                    finding = SelfMonitoringFinding(
                        finding_type=finding_data.get("type", "unknown"),
                        description=finding_data.get("description", ""),
                        severity=finding_data.get("severity", "low"),
                        action_recommended=finding_data.get("action_recommended"),
                    )
                    logger.info(
                        f"[SELF-MONITOR] {finding.severity.upper()}: {finding.description}"
                    )

                    # Persist high-severity findings
                    if finding.severity in ("medium", "high") and self._memory_store:
                        await self._memory_store.store_log(finding)

                # Queue notification if needed
                if data.get("notify_user") and data.get("notification_message"):
                    if self._notification_provider:
                        await self._notification_provider.queue_notification(
                            message=data["notification_message"],
                            priority="medium",
                        )

                # Log lessons learned
                for lesson in data.get("lessons_learned", []):
                    logger.info(f"[SELF-MONITOR] Lesson: {lesson}")

            except json.JSONDecodeError:
                logger.warning("Could not parse self-monitoring response")

        except Exception as e:
            logger.error(f"Error in self-monitoring: {e}")

    # =========================================================================
    # Transparency API
    # =========================================================================

    def get_action_logs(self, days: int = 7) -> list[ActionLog]:
        """Get action logs for transparency."""
        cutoff = datetime.now() - timedelta(days=days)
        return [log for log in self._action_logs if log.started_at >= cutoff]

    def get_blocked_logs(self, days: int = 7) -> list[ActionLog]:
        """Get blocked action logs."""
        cutoff = datetime.now() - timedelta(days=days)
        return [
            log for log in self._action_logs if log.blocked and log.started_at >= cutoff
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get autonomous agent stats."""
        return {
            "enabled": self.config.enabled,
            "shadow_mode": self.config.shadow_mode,
            "actions_today": self._actions_today,
            "daily_budget": self.config.daily_action_budget,
            "total_logs": len(self._action_logs),
            "blocked_count": sum(1 for log in self._action_logs if log.blocked),
        }
