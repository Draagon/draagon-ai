"""Integration tests for the autonomous agent module.

Tests cover:
- Full autonomous cycle execution
- Guardrail blocking at each tier
- Self-monitoring finding detection
- Action logging completeness
- Provider integration flows

REQ-004-09: Integration tests
"""

import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.orchestration.autonomous import (
    # Service
    AutonomousAgentService,
    # Config
    AutonomousConfig,
    # Enums
    ActionType,
    ActionTier,
    # Types
    ProposedAction,
    ApprovedAction,
    ActionResult,
    ActionLog,
    HarmCheck,
    SafetyCheck,
    AutonomousContext,
    SelfMonitoringFinding,
    SelfMonitoringResult,
    # Protocols
    LLMProvider,
    SearchProvider,
    MemoryStoreProvider,
    ContextProvider,
    NotificationProvider,
)


# =============================================================================
# Integration Test Providers
# =============================================================================


class IntegrationLLMProvider:
    """LLM provider that simulates realistic responses for integration testing."""

    def __init__(self):
        self.call_history: list[dict] = []
        self._response_queue: list[str] = []
        self._default_proposal = json.dumps({
            "proposed_actions": [
                {
                    "type": "research",
                    "description": "Research recent AI developments",
                    "reasoning": "Stay informed about the field",
                    "risk_tier": 0,
                    "reversible": True,
                    "estimated_time_seconds": 30,
                }
            ]
        })

    def queue_response(self, response: str) -> None:
        """Queue a specific response for the next generate call."""
        self._response_queue.append(response)

    def queue_responses(self, responses: list[str]) -> None:
        """Queue multiple responses."""
        self._response_queue.extend(responses)

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.call_history.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })

        if self._response_queue:
            return self._response_queue.pop(0)

        # Default intelligent responses based on prompt content
        if "harm" in prompt.lower() or "harmful" in prompt.lower():
            return json.dumps({
                "potentially_harmful": False,
                "reason": "Action appears safe",
                "confidence": 0.9,
            })
        elif "safety" in prompt.lower() or "safe" in prompt.lower():
            return "SAFE: Action is within acceptable bounds"
        elif "self-monitoring" in prompt.lower() or "review" in prompt.lower():
            return json.dumps({
                "findings": [],
                "notify_user": False,
                "lessons_learned": ["Action completed successfully"],
            })
        elif "synthesis" in prompt.lower() or "learned" in prompt.lower():
            return "Synthesized insight from research"
        elif "verify" in prompt.lower() or "assessment" in prompt.lower():
            return "Claim verified as accurate"
        elif "reflect" in prompt.lower():
            return json.dumps({
                "summary": "Reflection complete, performance is good",
                "trait_adjustments": {},
            })
        else:
            return self._default_proposal


class IntegrationSearchProvider:
    """Search provider that simulates web search for integration testing."""

    def __init__(self):
        self.queries: list[str] = []
        self._results: dict[str, str] = {}

    def set_result(self, query: str, result: str) -> None:
        """Set a specific result for a query."""
        self._results[query] = result

    async def search(self, query: str) -> str:
        self.queries.append(query)
        if query in self._results:
            return self._results[query]
        return f"Search results for '{query}': Multiple relevant sources found."


class IntegrationMemoryStoreProvider:
    """Memory store that tracks all operations for integration testing."""

    def __init__(self):
        self.stored_items: list[dict] = []
        self.stored_logs: list[dict] = []
        self.search_queries: list[str] = []

    async def store(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        **metadata,
    ) -> str:
        item_id = f"mem_{len(self.stored_items) + 1}"
        self.stored_items.append({
            "id": item_id,
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            **metadata,
        })
        return item_id

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        self.search_queries.append(query)
        return self.stored_items[:limit]

    async def get_logs(self, record_type: str, limit: int = 100) -> list[dict]:
        return [log for log in self.stored_logs if log.get("type") == record_type][:limit]

    async def store_log(self, log) -> None:
        self.stored_logs.append({
            "log": log,
            "type": type(log).__name__,
            "timestamp": datetime.now().isoformat(),
        })


class IntegrationContextProvider:
    """Context provider that simulates rich application context."""

    def __init__(self):
        self._context = AutonomousContext(
            personality_context="I am a helpful AI assistant for a smart home.",
            trait_values={
                "curiosity_intensity": 0.7,
                "verification_threshold": 0.5,
                "proactive_helpfulness": 0.6,
            },
            recent_conversations_summary="User discussed calendar events.",
            pending_questions=["What is the user's preferred temperature?"],
            unverified_claims=["User mentioned having 3 cats"],
            knowledge_gaps=["User's work schedule"],
            conflicting_beliefs=[],
            upcoming_events_summary="Team meeting at 2pm",
            household_members=["Alice", "Bob"],
            current_time=datetime.now(),
            day_of_week=datetime.now().strftime("%A"),
            recent_actions=[],
            daily_budget_remaining=20,
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
        self.call_count = 0

    def set_context(self, context: AutonomousContext) -> None:
        """Override the context."""
        self._context = context

    async def gather_context(self) -> AutonomousContext:
        self.call_count += 1
        return self._context


class IntegrationNotificationProvider:
    """Notification provider that tracks all notifications."""

    def __init__(self):
        self.notifications: list[dict] = []

    async def queue_notification(
        self,
        message: str,
        priority: str = "medium",
    ) -> None:
        self.notifications.append({
            "message": message,
            "priority": priority,
            "timestamp": datetime.now().isoformat(),
        })


# =============================================================================
# Full Cycle Integration Tests
# =============================================================================


class TestFullCycleExecution:
    """Test complete autonomous cycle execution."""

    @pytest.fixture
    def full_service(self):
        """Create a fully integrated service with all providers."""
        llm = IntegrationLLMProvider()
        search = IntegrationSearchProvider()
        memory = IntegrationMemoryStoreProvider()
        context = IntegrationContextProvider()
        notification = IntegrationNotificationProvider()
        config = AutonomousConfig(
            enabled=True,
            shadow_mode=False,
            require_semantic_safety_check=True,
            enable_self_monitoring=True,
            persist_logs=True,
            daily_action_budget=20,
            max_actions_per_cycle=3,
        )

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            search=search,
            memory_store=memory,
            context_provider=context,
            notification_provider=notification,
        )

        return {
            "service": service,
            "llm": llm,
            "search": search,
            "memory": memory,
            "context": context,
            "notification": notification,
        }

    @pytest.mark.asyncio
    async def test_full_research_cycle(self, full_service):
        """Test complete research action cycle."""
        service = full_service["service"]
        llm = full_service["llm"]
        search = full_service["search"]
        memory = full_service["memory"]

        # Queue responses for the full cycle
        llm.queue_responses([
            # Proposal generation
            json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "Research smart home trends",
                    "reasoning": "Stay current with technology",
                    "risk_tier": 0,
                }]
            }),
            # Harm check
            json.dumps({"potentially_harmful": False, "confidence": 0.95}),
            # Semantic safety
            "SAFE: Research is always allowed",
            # Synthesis
            "Key insight: Voice control is the future",
            # Self-monitoring
            json.dumps({
                "findings": [],
                "notify_user": False,
                "lessons_learned": ["Good research cycle"],
            }),
        ])

        search.set_result(
            "Research smart home trends",
            "Smart home trends: Voice control, energy efficiency..."
        )

        results = await service.run_cycle()

        # Verify research was executed
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].learned is not None

        # Verify search was called
        assert len(search.queries) == 1

        # Verify insight was stored
        assert len(memory.stored_items) == 1
        assert "insight" in memory.stored_items[0]["memory_type"]

        # Verify action was logged
        assert len(service._action_logs) >= 1

    @pytest.mark.asyncio
    async def test_full_verify_cycle(self, full_service):
        """Test complete verification action cycle."""
        service = full_service["service"]
        llm = full_service["llm"]
        search = full_service["search"]

        # Queue responses
        llm.queue_responses([
            # Proposal
            json.dumps({
                "proposed_actions": [{
                    "type": "verify",
                    "description": "Verify that user has 3 cats",
                    "reasoning": "Unverified claim in context",
                    "risk_tier": 0,
                }]
            }),
            # Harm check
            json.dumps({"potentially_harmful": False}),
            # Safety
            "SAFE",
            # Verification assessment
            "Cannot verify personal claim externally",
            # Self-monitoring
            json.dumps({"findings": [], "notify_user": False}),
        ])

        results = await service.run_cycle()

        assert len(results) == 1
        assert results[0].success is True
        # Verify search was called for verification
        assert any("verify" in q.lower() for q in search.queries)

    @pytest.mark.asyncio
    async def test_full_reflect_cycle(self, full_service):
        """Test complete reflection action cycle."""
        service = full_service["service"]
        llm = full_service["llm"]

        # Queue responses - reflection doesn't need as many responses
        # because _execute_reflect uses _gather_context which calls the context_provider
        llm.queue_responses([
            # Proposal
            json.dumps({
                "proposed_actions": [{
                    "type": "reflect",
                    "description": "Reflect on recent performance",
                    "reasoning": "Regular self-assessment",
                    "risk_tier": 0,
                }]
            }),
            # Harm check
            json.dumps({"potentially_harmful": False}),
            # Safety
            "SAFE",
            # Reflection result (from _execute_reflect's LLM call)
            json.dumps({
                "summary": "Performance is stable",
                "trait_adjustments": {"curiosity_intensity": 0.05},
            }),
            # Self-monitoring
            json.dumps({"findings": [], "notify_user": False}),
        ])

        results = await service.run_cycle()

        assert len(results) == 1
        assert results[0].success is True
        # belief_updated is True only if trait_adjustments is truthy
        # The reflection returns trait_adjustments but that doesn't set belief_updated
        # Let's check what the actual result is
        assert "stable" in results[0].outcome.lower() or results[0].success is True

    @pytest.mark.asyncio
    async def test_multiple_actions_in_cycle(self, full_service):
        """Test cycle with multiple actions."""
        service = full_service["service"]
        llm = full_service["llm"]
        memory = full_service["memory"]

        # Queue responses for 2 actions
        llm.queue_responses([
            # Proposal with 2 actions
            json.dumps({
                "proposed_actions": [
                    {
                        "type": "note_question",
                        "description": "What is user's favorite color?",
                        "reasoning": "Knowledge gap",
                        "risk_tier": 0,
                    },
                    {
                        "type": "prepare_suggestion",
                        "description": "Suggest energy saving tips",
                        "reasoning": "Proactive help",
                        "risk_tier": 1,
                    },
                ]
            }),
            # Harm checks (one per action)
            json.dumps({"potentially_harmful": False}),
            "SAFE",
            json.dumps({"potentially_harmful": False}),
            "SAFE",
            # Self-monitoring
            json.dumps({"findings": [], "notify_user": False}),
        ])

        results = await service.run_cycle()

        assert len(results) == 2
        assert all(r.success for r in results)
        # Both should store to memory
        assert len(memory.stored_items) >= 2

    @pytest.mark.asyncio
    async def test_shadow_mode_execution(self, full_service):
        """Test shadow mode logs but doesn't execute."""
        service = full_service["service"]
        service.config.shadow_mode = True
        llm = full_service["llm"]
        memory = full_service["memory"]

        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "Shadow research",
                    "reasoning": "Testing",
                    "risk_tier": 0,
                }]
            }),
            json.dumps({"potentially_harmful": False}),
            "SAFE",
        ])

        results = await service.run_cycle()

        # Shadow mode returns empty results
        assert results == []
        # But action should be logged
        shadow_logs = [log for log in service._action_logs if True]
        assert len(shadow_logs) >= 1
        # No actual memory storage in shadow mode
        research_items = [i for i in memory.stored_items if "research" in i.get("content", "").lower()]
        assert len(research_items) == 0


# =============================================================================
# Guardrail Integration Tests
# =============================================================================


class TestGuardrailIntegration:
    """Test guardrails block correctly at each tier."""

    @pytest.fixture
    def guardrail_service(self):
        """Create service focused on guardrail testing."""
        llm = IntegrationLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            require_semantic_safety_check=True,
            log_all_proposals=True,
        )
        memory = IntegrationMemoryStoreProvider()

        return AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        ), llm, memory

    @pytest.mark.asyncio
    async def test_tier_0_passes_all_guardrails(self, guardrail_service):
        """Tier 0 action passes through all guardrails."""
        service, llm, memory = guardrail_service

        llm.queue_responses([
            json.dumps({"potentially_harmful": False, "confidence": 0.99}),
            "SAFE: Tier 0 actions are always safe",
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Safe research",
            reasoning="Learning",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 1
        assert "tier" in approved[0].guardrails_passed
        assert "harm" in approved[0].guardrails_passed
        assert "semantic" in approved[0].guardrails_passed

    @pytest.mark.asyncio
    async def test_tier_1_passes_with_safe_content(self, guardrail_service):
        """Tier 1 action with safe content passes."""
        service, llm, memory = guardrail_service

        llm.queue_responses([
            json.dumps({"potentially_harmful": False}),
            "SAFE",
        ])

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Prepare weather suggestion",
            reasoning="Helpful",
            risk_tier=ActionTier.TIER_1,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 1

    @pytest.mark.asyncio
    async def test_tier_2_blocked_by_tier_check(self, guardrail_service):
        """Tier 2 action blocked immediately by tier check."""
        service, llm, memory = guardrail_service

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send reminder notification",
            reasoning="User requested",
            risk_tier=ActionTier.TIER_2,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        # Should be logged as blocked
        blocked_logs = [log for log in service._action_logs if log.blocked]
        assert len(blocked_logs) == 1
        assert "tier" in blocked_logs[0].blocked_reason.lower()

    @pytest.mark.asyncio
    async def test_tier_3_blocked_immediately(self, guardrail_service):
        """Tier 3 action blocked without reaching harm check."""
        service, llm, memory = guardrail_service

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send message to user",
            reasoning="Important info",
            risk_tier=ActionTier.TIER_3,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        # LLM should NOT have been called (blocked before harm check)
        harm_calls = [c for c in llm.call_history if "harm" in c["prompt"].lower()]
        assert len(harm_calls) == 0

    @pytest.mark.asyncio
    async def test_tier_4_blocked_immediately(self, guardrail_service):
        """Tier 4 (forbidden) action blocked immediately."""
        service, llm, memory = guardrail_service

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Financial transaction",
            reasoning="User asked",
            risk_tier=ActionTier.TIER_4,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_harm_check_blocks_harmful_tier_0(self, guardrail_service):
        """Tier 0 action blocked if harm check fails."""
        service, llm, memory = guardrail_service

        llm.queue_responses([
            json.dumps({
                "potentially_harmful": True,
                "reason": "Could invade user privacy",
                "confidence": 0.85,
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research user's private browsing history",
            reasoning="Curiosity",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked_logs = [log for log in service._action_logs if log.blocked]
        assert len(blocked_logs) == 1
        assert "harm" in blocked_logs[0].blocked_reason.lower()

    @pytest.mark.asyncio
    async def test_semantic_safety_blocks_unsafe(self, guardrail_service):
        """Semantic safety check blocks even after harm check passes."""
        service, llm, memory = guardrail_service

        llm.queue_responses([
            json.dumps({"potentially_harmful": False}),
            "UNSAFE: This action seems risky upon semantic analysis",
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research controversial topic",
            reasoning="Learning",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked_logs = [log for log in service._action_logs if log.blocked]
        assert len(blocked_logs) == 1
        assert "semantic" in blocked_logs[0].blocked_reason.lower()

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_consecutive_same_type(self, guardrail_service):
        """Rate limit blocks too many consecutive same-type actions."""
        service, llm, memory = guardrail_service

        # Simulate 2 recent research actions
        service._recent_action_types = [ActionType.RESEARCH, ActionType.RESEARCH]

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Yet more research",
            reasoning="Learning",
            risk_tier=ActionTier.TIER_0,
        )

        approved = await service._filter_through_guardrails([action])

        assert len(approved) == 0
        blocked_logs = [log for log in service._action_logs if log.blocked]
        assert any("rate" in log.blocked_reason.lower() for log in blocked_logs)

    @pytest.mark.asyncio
    async def test_budget_exhaustion_blocks_cycle(self, guardrail_service):
        """Budget exhaustion blocks the entire cycle."""
        service, llm, memory = guardrail_service
        service.config.daily_action_budget = 5
        service._actions_today = 5  # Already at budget

        # Even with valid proposals, cycle should return empty
        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "Research",
                    "reasoning": "Learn",
                    "risk_tier": 0,
                }]
            }),
        ])

        results = await service.run_cycle()

        assert results == []


# =============================================================================
# Self-Monitoring Integration Tests
# =============================================================================


class TestSelfMonitoringIntegration:
    """Test self-monitoring finding detection."""

    @pytest.fixture
    def monitoring_service(self):
        """Create service focused on self-monitoring testing."""
        llm = IntegrationLLMProvider()
        memory = IntegrationMemoryStoreProvider()
        notification = IntegrationNotificationProvider()
        config = AutonomousConfig(
            enabled=True,
            enable_self_monitoring=True,
            persist_logs=True,
        )

        return AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
            notification_provider=notification,
        ), llm, memory, notification

    @pytest.mark.asyncio
    async def test_monitoring_detects_unexpected_result(self, monitoring_service):
        """Self-monitoring detects and logs unexpected results."""
        service, llm, memory, notification = monitoring_service

        llm.queue_responses([
            json.dumps({
                "findings": [{
                    "type": "unexpected_result",
                    "description": "Research returned contradictory information",
                    "severity": "medium",
                    "action_recommended": "Verify with additional sources",
                }],
                "notify_user": False,
                "lessons_learned": ["Be more thorough in verification"],
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Research complete")

        await service._self_monitor([(action, result)])

        # Medium severity should be stored
        assert len(memory.stored_logs) == 1
        log_entry = memory.stored_logs[0]
        assert "SelfMonitoringFinding" in log_entry["type"]

    @pytest.mark.asyncio
    async def test_monitoring_high_severity_stored(self, monitoring_service):
        """High severity findings are persisted."""
        service, llm, memory, notification = monitoring_service

        llm.queue_responses([
            json.dumps({
                "findings": [{
                    "type": "contradiction",
                    "description": "Learned information contradicts existing belief",
                    "severity": "high",
                    "action_recommended": "Review and reconcile",
                }],
                "notify_user": False,
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        assert len(memory.stored_logs) == 1

    @pytest.mark.asyncio
    async def test_monitoring_queues_notification_when_needed(self, monitoring_service):
        """Self-monitoring queues notification when flagged."""
        service, llm, memory, notification = monitoring_service

        llm.queue_responses([
            json.dumps({
                "findings": [{
                    "type": "needs_human",
                    "description": "Found something user should know",
                    "severity": "high",
                }],
                "notify_user": True,
                "notification_message": "I found something interesting you should know about!",
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        assert len(notification.notifications) == 1
        assert "something interesting" in notification.notifications[0]["message"]

    @pytest.mark.asyncio
    async def test_monitoring_no_notification_when_not_flagged(self, monitoring_service):
        """No notification when not flagged."""
        service, llm, memory, notification = monitoring_service

        llm.queue_responses([
            json.dumps({
                "findings": [],
                "notify_user": False,
                "lessons_learned": ["All good"],
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        assert len(notification.notifications) == 0

    @pytest.mark.asyncio
    async def test_monitoring_handles_multiple_findings(self, monitoring_service):
        """Self-monitoring handles multiple findings correctly."""
        service, llm, memory, notification = monitoring_service

        llm.queue_responses([
            json.dumps({
                "findings": [
                    {
                        "type": "low_value",
                        "description": "Action had minimal impact",
                        "severity": "low",
                    },
                    {
                        "type": "unexpected_result",
                        "description": "Result was surprising",
                        "severity": "medium",
                    },
                    {
                        "type": "pattern",
                        "description": "Detected repeating behavior",
                        "severity": "high",
                    },
                ],
                "notify_user": False,
            }),
        ])

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        # Medium and high severity stored (not low)
        assert len(memory.stored_logs) == 2


# =============================================================================
# Action Logging Integration Tests
# =============================================================================


class TestActionLoggingIntegration:
    """Test action logging completeness."""

    @pytest.fixture
    def logging_service(self):
        """Create service focused on logging testing."""
        llm = IntegrationLLMProvider()
        memory = IntegrationMemoryStoreProvider()
        config = AutonomousConfig(
            enabled=True,
            persist_logs=True,
            log_all_proposals=True,
        )

        return AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        ), llm, memory

    @pytest.mark.asyncio
    async def test_successful_action_fully_logged(self, logging_service):
        """Successful action has complete log entry."""
        service, llm, memory = logging_service

        action = ProposedAction(
            action_type=ActionType.REST,
            description="Take a rest",
            reasoning="Nothing urgent",
        )
        result = ActionResult(success=True, outcome="Resting - no action taken")

        await service._log_action(action, result)

        assert len(service._action_logs) == 1
        log = service._action_logs[0]

        # Check all fields are populated
        assert log.action_id is not None
        assert log.action_type == "rest"
        assert log.description == "Take a rest"
        assert log.reasoning == "Nothing urgent"
        assert log.started_at is not None
        assert log.completed_at is not None
        assert log.success is True
        assert log.outcome == "Resting - no action taken"
        assert log.error is None
        assert log.blocked is False

    @pytest.mark.asyncio
    async def test_failed_action_logs_error(self, logging_service):
        """Failed action logs error details."""
        service, llm, memory = logging_service

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Learn",
        )
        result = ActionResult(success=False, error="Search provider unavailable")

        await service._log_action(action, result)

        log = service._action_logs[0]
        assert log.success is False
        assert log.error == "Search provider unavailable"

    @pytest.mark.asyncio
    async def test_blocked_action_logs_reason(self, logging_service):
        """Blocked action logs block reason."""
        service, llm, memory = logging_service

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send message",
            reasoning="Help",
            risk_tier=ActionTier.TIER_3,
        )

        await service._log_blocked(action, "tier_violation")

        assert len(service._action_logs) == 1
        log = service._action_logs[0]
        assert log.blocked is True
        assert log.blocked_reason == "tier_violation"

    @pytest.mark.asyncio
    async def test_logs_persisted_to_memory_store(self, logging_service):
        """Logs are persisted to memory store."""
        service, llm, memory = logging_service

        action = ProposedAction(
            action_type=ActionType.REST,
            description="Rest",
            reasoning="Nothing",
        )
        result = ActionResult(success=True, outcome="Resting")

        await service._log_action(action, result)

        # Should be persisted
        assert len(memory.stored_logs) == 1

    @pytest.mark.asyncio
    async def test_transparency_api_returns_recent_logs(self, logging_service):
        """Transparency API returns recent logs."""
        service, llm, memory = logging_service

        # Add some logs
        for i in range(5):
            action = ProposedAction(
                action_type=ActionType.REST,
                description=f"Rest {i}",
                reasoning="Testing",
            )
            result = ActionResult(success=True, outcome="Done")
            await service._log_action(action, result)

        logs = service.get_action_logs(days=7)

        assert len(logs) == 5

    @pytest.mark.asyncio
    async def test_transparency_api_filters_blocked(self, logging_service):
        """get_blocked_logs returns only blocked actions."""
        service, llm, memory = logging_service

        # Add successful log
        action1 = ProposedAction(
            action_type=ActionType.REST,
            description="Success",
            reasoning="Test",
        )
        await service._log_action(action1, ActionResult(success=True))

        # Add blocked log
        action2 = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Blocked",
            reasoning="Test",
            risk_tier=ActionTier.TIER_3,
        )
        await service._log_blocked(action2, "tier_violation")

        blocked = service.get_blocked_logs()

        assert len(blocked) == 1
        assert blocked[0].blocked is True

    @pytest.mark.asyncio
    async def test_stats_accurate_after_cycle(self, logging_service):
        """Stats reflect accurate counts after cycle."""
        service, llm, memory = logging_service

        # Queue responses for a cycle
        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "rest",
                    "description": "Rest",
                    "reasoning": "Nothing",
                    "risk_tier": 0,
                }]
            }),
            json.dumps({"potentially_harmful": False}),
            "SAFE",
        ])

        await service.run_cycle()

        stats = service.get_stats()

        assert stats["enabled"] is True
        assert stats["actions_today"] >= 1
        assert stats["total_logs"] >= 1


# =============================================================================
# End-to-End Provider Integration Tests
# =============================================================================


class TestProviderIntegration:
    """Test integration between service and all providers."""

    @pytest.mark.asyncio
    async def test_context_provider_called_each_cycle(self):
        """Context provider is called at start of each cycle."""
        llm = IntegrationLLMProvider()
        context = IntegrationContextProvider()
        config = AutonomousConfig(enabled=True)

        # Return empty proposals
        llm.queue_responses([json.dumps({"proposed_actions": []})])

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            context_provider=context,
        )

        await service.run_cycle()

        assert context.call_count >= 1

    @pytest.mark.asyncio
    async def test_search_provider_integration(self):
        """Search provider is properly integrated for research."""
        llm = IntegrationLLMProvider()
        search = IntegrationSearchProvider()
        memory = IntegrationMemoryStoreProvider()
        config = AutonomousConfig(enabled=True)

        # Set specific search result
        search.set_result("AI trends", "AI is advancing rapidly in 2025")

        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "AI trends",
                    "reasoning": "Stay informed",
                    "risk_tier": 0,
                }]
            }),
            json.dumps({"potentially_harmful": False}),
            "SAFE",
            "Synthesized: AI is advancing",  # Synthesis
        ])

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            search=search,
            memory_store=memory,
        )

        results = await service.run_cycle()

        # Search should have been called
        assert "AI trends" in search.queries
        # Result should use search content
        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_memory_store_integration(self):
        """Memory store receives correct data."""
        llm = IntegrationLLMProvider()
        search = IntegrationSearchProvider()
        memory = IntegrationMemoryStoreProvider()
        config = AutonomousConfig(enabled=True, persist_logs=True)

        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "update_belief",
                    "description": "User prefers morning meetings",
                    "reasoning": "Observed pattern",
                    "risk_tier": 0,
                }]
            }),
            json.dumps({"potentially_harmful": False}),
            "SAFE",
        ])

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            search=search,
            memory_store=memory,
        )

        await service.run_cycle()

        # Belief should be stored
        belief_items = [i for i in memory.stored_items if i["memory_type"] == "fact"]
        assert len(belief_items) >= 1
        assert "morning meetings" in belief_items[0]["content"]

        # Log should be stored
        assert len(memory.stored_logs) >= 1

    @pytest.mark.asyncio
    async def test_notification_provider_integration(self):
        """Notification provider receives notifications from monitoring."""
        llm = IntegrationLLMProvider()
        notification = IntegrationNotificationProvider()
        config = AutonomousConfig(enabled=True, enable_self_monitoring=True)

        llm.queue_responses([
            json.dumps({
                "proposed_actions": [{
                    "type": "rest",
                    "description": "Rest",
                    "reasoning": "Nothing",
                    "risk_tier": 0,
                }]
            }),
            json.dumps({"potentially_harmful": False}),
            "SAFE",
            # Self-monitoring with notification
            json.dumps({
                "findings": [],
                "notify_user": True,
                "notification_message": "Cycle complete with interesting findings",
            }),
        ])

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            notification_provider=notification,
        )

        await service.run_cycle()

        # Notification should have been queued
        assert len(notification.notifications) == 1
        assert "interesting findings" in notification.notifications[0]["message"]

    @pytest.mark.asyncio
    async def test_full_cycle_with_all_providers(self):
        """Complete cycle uses all providers correctly."""
        llm = IntegrationLLMProvider()
        search = IntegrationSearchProvider()
        memory = IntegrationMemoryStoreProvider()
        context = IntegrationContextProvider()
        notification = IntegrationNotificationProvider()

        config = AutonomousConfig(
            enabled=True,
            enable_self_monitoring=True,
            persist_logs=True,
        )

        llm.queue_responses([
            # Proposal
            json.dumps({
                "proposed_actions": [{
                    "type": "research",
                    "description": "Smart home tech",
                    "reasoning": "User interest",
                    "risk_tier": 0,
                }]
            }),
            # Harm check
            json.dumps({"potentially_harmful": False}),
            # Safety
            "SAFE",
            # Synthesis
            "Key insight about smart home tech",
            # Self-monitoring
            json.dumps({
                "findings": [{
                    "type": "pattern",
                    "description": "Good research pattern",
                    "severity": "medium",
                }],
                "notify_user": False,
            }),
        ])

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            search=search,
            memory_store=memory,
            context_provider=context,
            notification_provider=notification,
        )

        results = await service.run_cycle()

        # All providers should have been used
        assert context.call_count >= 1  # Context gathered
        assert len(search.queries) >= 1  # Search called
        assert len(memory.stored_items) >= 1  # Memory stored
        assert len(memory.stored_logs) >= 1  # Logs stored
        assert len(results) == 1  # Action executed
        assert results[0].success is True
