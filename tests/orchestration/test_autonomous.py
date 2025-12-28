"""Unit tests for the autonomous agent module.

Tests cover:
- Types and enums
- Configuration
- Guardrails (4-layer chain)
- Action execution
- Self-monitoring
- Transparency API

Target: ≥90% coverage for autonomous module.
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
# Mock Providers
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self.responses = responses or ['{"proposed_actions": []}']
        self.call_count = 0
        self.prompts: list[str] = []

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.prompts.append(prompt)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return response


class MockSearchProvider:
    """Mock search provider for testing."""

    def __init__(self, results: str = "Search results here"):
        self.results = results
        self.queries: list[str] = []

    async def search(self, query: str) -> str:
        self.queries.append(query)
        return self.results


class MockMemoryStoreProvider:
    """Mock memory store provider for testing."""

    def __init__(self):
        self.stored: list[dict] = []
        self.logs: list[dict] = []

    async def store(
        self,
        content: str,
        memory_type: str,
        importance: float = 0.5,
        **metadata,
    ) -> str:
        self.stored.append({
            "content": content,
            "memory_type": memory_type,
            "importance": importance,
            **metadata,
        })
        return f"mem_{len(self.stored)}"

    async def search(self, query: str, limit: int = 10) -> list[dict]:
        return self.stored[:limit]

    async def get_logs(self, record_type: str, limit: int = 100) -> list[dict]:
        return self.logs[:limit]

    async def store_log(self, log) -> None:
        self.logs.append({"log": log})


class MockContextProvider:
    """Mock context provider for testing."""

    def __init__(self, context: AutonomousContext | None = None):
        self.context = context or AutonomousContext(
            personality_context="Test personality",
            trait_values={"curiosity_intensity": 0.7},
            current_time=datetime.now(),
            day_of_week="Monday",
        )

    async def gather_context(self) -> AutonomousContext:
        return self.context


class MockNotificationProvider:
    """Mock notification provider for testing."""

    def __init__(self):
        self.notifications: list[dict] = []

    async def queue_notification(
        self,
        message: str,
        priority: str = "medium",
    ) -> None:
        self.notifications.append({"message": message, "priority": priority})


# =============================================================================
# Type Tests
# =============================================================================


class TestActionType:
    """Test ActionType enum."""

    def test_all_values_exist(self):
        """All expected action types exist."""
        assert ActionType.RESEARCH.value == "research"
        assert ActionType.VERIFY.value == "verify"
        assert ActionType.REFLECT.value == "reflect"
        assert ActionType.NOTE_QUESTION.value == "note_question"
        assert ActionType.PREPARE_SUGGESTION.value == "prepare_suggestion"
        assert ActionType.UPDATE_BELIEF.value == "update_belief"
        assert ActionType.REST.value == "rest"

    def test_enum_count(self):
        """Correct number of action types."""
        assert len(ActionType) == 7


class TestActionTier:
    """Test ActionTier enum."""

    def test_all_tiers_exist(self):
        """All expected tiers exist with correct values."""
        assert ActionTier.TIER_0.value == 0
        assert ActionTier.TIER_1.value == 1
        assert ActionTier.TIER_2.value == 2
        assert ActionTier.TIER_3.value == 3
        assert ActionTier.TIER_4.value == 4

    def test_tier_ordering(self):
        """Tiers are correctly ordered by risk."""
        assert ActionTier.TIER_0.value < ActionTier.TIER_1.value
        assert ActionTier.TIER_1.value < ActionTier.TIER_2.value
        assert ActionTier.TIER_2.value < ActionTier.TIER_3.value
        assert ActionTier.TIER_3.value < ActionTier.TIER_4.value


class TestAutonomousConfig:
    """Test AutonomousConfig dataclass."""

    def test_default_values(self):
        """Default config values are correct."""
        config = AutonomousConfig()
        assert config.enabled is True
        assert config.cycle_interval_minutes == 30
        assert config.active_hours_start == 8
        assert config.active_hours_end == 22
        assert config.max_actions_per_cycle == 3
        assert config.daily_action_budget == 20
        assert config.max_consecutive_same_type == 2
        assert config.require_semantic_safety_check is True
        assert config.log_all_proposals is True
        assert config.shadow_mode is False
        assert config.enable_self_monitoring is True
        assert config.persist_logs is True

    def test_custom_values(self):
        """Custom config values work correctly."""
        config = AutonomousConfig(
            enabled=False,
            cycle_interval_minutes=60,
            daily_action_budget=10,
            shadow_mode=True,
        )
        assert config.enabled is False
        assert config.cycle_interval_minutes == 60
        assert config.daily_action_budget == 10
        assert config.shadow_mode is True

    def test_personality_bounds(self):
        """Personality trait bounds are set correctly."""
        config = AutonomousConfig()
        assert config.min_trait_value == 0.1
        assert config.max_trait_value == 0.9
        assert config.max_trait_change_per_day == 0.1


class TestProposedAction:
    """Test ProposedAction dataclass."""

    def test_minimal_creation(self):
        """Create with minimal required fields."""
        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Because it's interesting",
        )
        assert action.action_type == ActionType.RESEARCH
        assert action.description == "Research topic"
        assert action.reasoning == "Because it's interesting"
        assert action.risk_tier == ActionTier.TIER_0
        assert action.reversible is True

    def test_full_creation(self):
        """Create with all fields."""
        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Prepare dinner suggestion",
            reasoning="User might be hungry",
            risk_tier=ActionTier.TIER_1,
            reversible=True,
            estimated_time_seconds=60,
            target_entity="user",
        )
        assert action.risk_tier == ActionTier.TIER_1
        assert action.estimated_time_seconds == 60
        assert action.target_entity == "user"


class TestActionResult:
    """Test ActionResult dataclass."""

    def test_success_result(self):
        """Create success result."""
        result = ActionResult(
            success=True,
            outcome="Task completed",
        )
        assert result.success is True
        assert result.outcome == "Task completed"
        assert result.error is None
        assert result.learned is None
        assert result.belief_updated is False

    def test_failure_result(self):
        """Create failure result."""
        result = ActionResult(
            success=False,
            error="Connection failed",
        )
        assert result.success is False
        assert result.error == "Connection failed"

    def test_with_learning(self):
        """Create result with learning."""
        result = ActionResult(
            success=True,
            outcome="Research complete",
            learned="Learned new fact",
            belief_updated=True,
        )
        assert result.learned == "Learned new fact"
        assert result.belief_updated is True


class TestActionLog:
    """Test ActionLog dataclass."""

    def test_basic_log(self):
        """Create basic log entry."""
        now = datetime.now()
        log = ActionLog(
            action_id="test-123",
            action_type="research",
            description="Test action",
            reasoning="Test reasoning",
            started_at=now,
        )
        assert log.action_id == "test-123"
        assert log.success is False  # Default
        assert log.blocked is False  # Default

    def test_blocked_log(self):
        """Create blocked log entry."""
        now = datetime.now()
        log = ActionLog(
            action_id="test-456",
            action_type="verify",
            description="Verify claim",
            reasoning="Need verification",
            started_at=now,
            blocked=True,
            blocked_reason="tier_violation",
        )
        assert log.blocked is True
        assert log.blocked_reason == "tier_violation"


class TestHarmCheck:
    """Test HarmCheck dataclass."""

    def test_safe_check(self):
        """Create safe harm check result."""
        check = HarmCheck(potentially_harmful=False)
        assert check.potentially_harmful is False
        assert check.reason is None
        assert check.confidence == 0.5

    def test_harmful_check(self):
        """Create harmful check result."""
        check = HarmCheck(
            potentially_harmful=True,
            reason="Could violate privacy",
            confidence=0.9,
        )
        assert check.potentially_harmful is True
        assert check.reason == "Could violate privacy"
        assert check.confidence == 0.9


class TestSafetyCheck:
    """Test SafetyCheck dataclass."""

    def test_safe_result(self):
        """Create safe result."""
        check = SafetyCheck(is_safe=True)
        assert check.is_safe is True

    def test_unsafe_result(self):
        """Create unsafe result with reason."""
        check = SafetyCheck(
            is_safe=False,
            reason="Action could cause harm",
        )
        assert check.is_safe is False
        assert check.reason == "Action could cause harm"


class TestAutonomousContext:
    """Test AutonomousContext dataclass."""

    def test_default_context(self):
        """Create context with defaults."""
        context = AutonomousContext()
        assert context.personality_context == ""
        assert context.trait_values == {}
        assert context.pending_questions == []
        assert context.daily_budget_remaining == 20

    def test_full_context(self):
        """Create context with all fields."""
        now = datetime.now()
        context = AutonomousContext(
            personality_context="I am curious",
            trait_values={"curiosity": 0.8},
            pending_questions=["What is X?"],
            unverified_claims=["Claim A"],
            knowledge_gaps=["Gap 1"],
            conflicting_beliefs=["Conflict 1"],
            household_members=["Alice", "Bob"],
            current_time=now,
            day_of_week="Friday",
            daily_budget_remaining=15,
        )
        assert context.personality_context == "I am curious"
        assert len(context.pending_questions) == 1
        assert len(context.household_members) == 2


class TestSelfMonitoringFinding:
    """Test SelfMonitoringFinding dataclass."""

    def test_finding_creation(self):
        """Create finding."""
        finding = SelfMonitoringFinding(
            finding_type="unexpected_result",
            description="Got unexpected output",
            severity="medium",
            action_recommended="Review and adjust",
        )
        assert finding.finding_type == "unexpected_result"
        assert finding.severity == "medium"


class TestSelfMonitoringResult:
    """Test SelfMonitoringResult dataclass."""

    def test_result_creation(self):
        """Create result."""
        result = SelfMonitoringResult(
            overall_assessment="good",
            notify_user=False,
        )
        assert result.overall_assessment == "good"
        assert result.findings == []
        assert result.lessons_learned == []


# =============================================================================
# Protocol Tests
# =============================================================================


class TestProtocols:
    """Test protocol definitions."""

    def test_llm_provider_is_runtime_checkable(self):
        """LLMProvider is runtime checkable."""
        mock = MockLLMProvider()
        assert isinstance(mock, LLMProvider)

    def test_search_provider_is_runtime_checkable(self):
        """SearchProvider is runtime checkable."""
        mock = MockSearchProvider()
        assert isinstance(mock, SearchProvider)

    def test_memory_store_provider_is_runtime_checkable(self):
        """MemoryStoreProvider is runtime checkable."""
        mock = MockMemoryStoreProvider()
        assert isinstance(mock, MemoryStoreProvider)

    def test_context_provider_is_runtime_checkable(self):
        """ContextProvider is runtime checkable."""
        mock = MockContextProvider()
        assert isinstance(mock, ContextProvider)

    def test_notification_provider_is_runtime_checkable(self):
        """NotificationProvider is runtime checkable."""
        mock = MockNotificationProvider()
        assert isinstance(mock, NotificationProvider)


# =============================================================================
# Service Init Tests
# =============================================================================


class TestServiceInit:
    """Test AutonomousAgentService initialization."""

    def test_minimal_init(self):
        """Init with only required LLM."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        assert service._llm is llm
        assert service.config.enabled is True
        assert service._search is None
        assert service._memory_store is None
        assert service._running is False

    def test_full_init(self):
        """Init with all providers."""
        llm = MockLLMProvider()
        search = MockSearchProvider()
        memory = MockMemoryStoreProvider()
        context = MockContextProvider()
        notification = MockNotificationProvider()
        config = AutonomousConfig(shadow_mode=True)

        service = AutonomousAgentService(
            llm=llm,
            config=config,
            search=search,
            memory_store=memory,
            context_provider=context,
            notification_provider=notification,
        )

        assert service._llm is llm
        assert service._search is search
        assert service._memory_store is memory
        assert service._context_provider is context
        assert service._notification_provider is notification
        assert service.config.shadow_mode is True

    def test_initial_state(self):
        """Initial state is correct."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        assert service._actions_today == 0
        assert service._recent_action_types == []
        assert service._action_logs == []
        assert service._background_task is None


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestServiceLifecycle:
    """Test service start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start_disabled(self):
        """Start does nothing when disabled."""
        llm = MockLLMProvider()
        config = AutonomousConfig(enabled=False)
        service = AutonomousAgentService(llm=llm, config=config)

        await service.start()

        assert service._running is False
        assert service._background_task is None

    @pytest.mark.asyncio
    async def test_start_enabled(self):
        """Start creates background task when enabled."""
        llm = MockLLMProvider()
        config = AutonomousConfig(enabled=True)
        service = AutonomousAgentService(llm=llm, config=config)

        await service.start()

        assert service._running is True
        assert service._background_task is not None

        # Clean up
        await service.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        """Start does nothing if already running."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        await service.start()
        task1 = service._background_task

        await service.start()  # Should warn and return
        task2 = service._background_task

        assert task1 is task2  # Same task

        await service.stop()

    @pytest.mark.asyncio
    async def test_stop(self):
        """Stop cancels background task."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        await service.start()
        assert service._running is True

        await service.stop()
        assert service._running is False


# =============================================================================
# Active Hours Tests
# =============================================================================


class TestActiveHours:
    """Test active hours checking."""

    def test_within_active_hours(self):
        """Time within active hours returns True."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            active_hours_start=8,
            active_hours_end=22,
        )
        service = AutonomousAgentService(llm=llm, config=config)

        # 10am is within 8am-10pm
        test_time = datetime.now().replace(hour=10, minute=0)
        assert service._is_within_active_hours(test_time) is True

    def test_outside_active_hours_early(self):
        """Time before active hours returns False."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            active_hours_start=8,
            active_hours_end=22,
        )
        service = AutonomousAgentService(llm=llm, config=config)

        # 6am is before 8am
        test_time = datetime.now().replace(hour=6, minute=0)
        assert service._is_within_active_hours(test_time) is False

    def test_outside_active_hours_late(self):
        """Time after active hours returns False."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            active_hours_start=8,
            active_hours_end=22,
        )
        service = AutonomousAgentService(llm=llm, config=config)

        # 11pm is after 10pm
        test_time = datetime.now().replace(hour=23, minute=0)
        assert service._is_within_active_hours(test_time) is False

    def test_boundary_start(self):
        """Exact start hour is within."""
        llm = MockLLMProvider()
        config = AutonomousConfig(active_hours_start=8, active_hours_end=22)
        service = AutonomousAgentService(llm=llm, config=config)

        test_time = datetime.now().replace(hour=8, minute=0)
        assert service._is_within_active_hours(test_time) is True

    def test_boundary_end(self):
        """Exact end hour is NOT within (exclusive)."""
        llm = MockLLMProvider()
        config = AutonomousConfig(active_hours_start=8, active_hours_end=22)
        service = AutonomousAgentService(llm=llm, config=config)

        test_time = datetime.now().replace(hour=22, minute=0)
        assert service._is_within_active_hours(test_time) is False


# =============================================================================
# Daily Reset Tests
# =============================================================================


class TestDailyReset:
    """Test daily counter reset."""

    def test_reset_on_new_day(self):
        """Counters reset on new day."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        # Simulate actions from yesterday
        service._actions_today = 15
        service._last_action_reset = datetime.now() - timedelta(days=1)

        service._check_daily_reset()

        assert service._actions_today == 0

    def test_no_reset_same_day(self):
        """Counters don't reset on same day."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        service._actions_today = 5
        # last_action_reset is already today by default

        service._check_daily_reset()

        assert service._actions_today == 5  # Unchanged


# =============================================================================
# Context Gathering Tests
# =============================================================================


class TestContextGathering:
    """Test context gathering."""

    @pytest.mark.asyncio
    async def test_with_context_provider(self):
        """Uses context provider when available."""
        llm = MockLLMProvider()
        custom_context = AutonomousContext(
            personality_context="Custom personality",
            pending_questions=["Question 1", "Question 2"],
        )
        context_provider = MockContextProvider(custom_context)

        service = AutonomousAgentService(
            llm=llm,
            context_provider=context_provider,
        )

        context = await service._gather_context()

        assert context.personality_context == "Custom personality"
        assert len(context.pending_questions) == 2

    @pytest.mark.asyncio
    async def test_without_context_provider(self):
        """Creates default context when no provider."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        context = await service._gather_context()

        # Check default values
        assert context.personality_context == "I am a helpful AI assistant."
        assert "curiosity_intensity" in context.trait_values
        assert context.daily_budget_remaining == 20  # Default budget


# =============================================================================
# Proposal Generation Tests
# =============================================================================


class TestProposalGeneration:
    """Test action proposal generation."""

    @pytest.mark.asyncio
    async def test_parse_valid_proposals(self):
        """Parse valid JSON proposals."""
        response = json.dumps({
            "proposed_actions": [
                {
                    "type": "research",
                    "description": "Research AI trends",
                    "reasoning": "Stay informed",
                    "risk_tier": 0,
                },
                {
                    "type": "reflect",
                    "description": "Reflect on performance",
                    "reasoning": "Self-improvement",
                    "risk_tier": 0,
                }
            ]
        })
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        context = AutonomousContext()
        proposals = await service._generate_proposals(context)

        assert len(proposals) == 2
        assert proposals[0].action_type == ActionType.RESEARCH
        assert proposals[1].action_type == ActionType.REFLECT

    @pytest.mark.asyncio
    async def test_parse_proposals_with_code_block(self):
        """Parse proposals wrapped in code block."""
        response = """Here are my proposals:
```json
{
    "proposed_actions": [
        {"type": "verify", "description": "Verify claim", "reasoning": "Fact check"}
    ]
}
```
"""
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        context = AutonomousContext()
        proposals = await service._generate_proposals(context)

        assert len(proposals) == 1
        assert proposals[0].action_type == ActionType.VERIFY

    @pytest.mark.asyncio
    async def test_parse_invalid_action_type(self):
        """Invalid action type defaults to REST."""
        response = json.dumps({
            "proposed_actions": [
                {"type": "invalid_type", "description": "Test", "reasoning": "Test"}
            ]
        })
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        context = AutonomousContext()
        proposals = await service._generate_proposals(context)

        assert len(proposals) == 1
        assert proposals[0].action_type == ActionType.REST

    @pytest.mark.asyncio
    async def test_parse_malformed_json(self):
        """Malformed JSON returns empty list."""
        llm = MockLLMProvider(responses=["not valid json at all"])
        service = AutonomousAgentService(llm=llm)

        context = AutonomousContext()
        proposals = await service._generate_proposals(context)

        assert proposals == []

    @pytest.mark.asyncio
    async def test_llm_error_returns_empty(self):
        """LLM error returns empty proposals."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        service = AutonomousAgentService(llm=llm)

        context = AutonomousContext()
        proposals = await service._generate_proposals(context)

        assert proposals == []


# =============================================================================
# Guardrail Tests
# =============================================================================


class TestTierGuardrail:
    """Test tier-based guardrail."""

    def test_tier_0_allowed(self):
        """Tier 0 actions are allowed."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
            risk_tier=ActionTier.TIER_0,
        )

        assert service._is_allowed_tier(action) is True

    def test_tier_1_allowed(self):
        """Tier 1 actions are allowed."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Suggestion",
            reasoning="Help",
            risk_tier=ActionTier.TIER_1,
        )

        assert service._is_allowed_tier(action) is True

    def test_tier_2_blocked(self):
        """Tier 2 actions are blocked."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Send reminder",
            reasoning="Help",
            risk_tier=ActionTier.TIER_2,
        )

        assert service._is_allowed_tier(action) is False

    def test_tier_3_blocked(self):
        """Tier 3 actions are blocked."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Modify calendar",
            reasoning="Organize",
            risk_tier=ActionTier.TIER_3,
        )

        assert service._is_allowed_tier(action) is False

    def test_tier_4_blocked(self):
        """Tier 4 actions are blocked."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Financial action",
            reasoning="Dangerous",
            risk_tier=ActionTier.TIER_4,
        )

        assert service._is_allowed_tier(action) is False


class TestRateLimitGuardrail:
    """Test rate limiting guardrail."""

    def test_no_rate_limit_first_action(self):
        """First action is not rate limited."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )

        assert service._exceeds_rate_limit(action) is False

    def test_rate_limit_exceeded(self):
        """Rate limit exceeded after consecutive same type."""
        llm = MockLLMProvider()
        config = AutonomousConfig(max_consecutive_same_type=2)
        service = AutonomousAgentService(llm=llm, config=config)

        # Simulate 2 consecutive research actions
        service._recent_action_types = [ActionType.RESEARCH, ActionType.RESEARCH]

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="More research",
            reasoning="Learn more",
        )

        assert service._exceeds_rate_limit(action) is True

    def test_rate_limit_not_exceeded_different_type(self):
        """Different action type is not rate limited."""
        llm = MockLLMProvider()
        config = AutonomousConfig(max_consecutive_same_type=2)
        service = AutonomousAgentService(llm=llm, config=config)

        # 2 consecutive research
        service._recent_action_types = [ActionType.RESEARCH, ActionType.RESEARCH]

        # But now trying reflect - different type
        action = ProposedAction(
            action_type=ActionType.REFLECT,
            description="Reflect",
            reasoning="Think",
        )

        assert service._exceeds_rate_limit(action) is False


class TestHarmCheckGuardrail:
    """Test harm check guardrail."""

    @pytest.mark.asyncio
    async def test_safe_action(self):
        """LLM says action is safe."""
        response = json.dumps({
            "potentially_harmful": False,
            "reason": "Safe action",
            "confidence": 0.9,
        })
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Learn",
        )

        result = await service._check_for_harm(action)

        assert result.potentially_harmful is False
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_harmful_action(self):
        """LLM says action is harmful."""
        response = json.dumps({
            "potentially_harmful": True,
            "reason": "Could invade privacy",
            "confidence": 0.85,
        })
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research user's browsing history",
            reasoning="Learn about user",
        )

        result = await service._check_for_harm(action)

        assert result.potentially_harmful is True
        assert "privacy" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_harm_check_error_fallback(self):
        """Error in harm check uses tier-based fallback."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        service = AutonomousAgentService(llm=llm)

        # Tier 0 - should be safe on fallback
        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
            risk_tier=ActionTier.TIER_0,
        )

        result = await service._check_for_harm(action)

        assert result.potentially_harmful is False

    @pytest.mark.asyncio
    async def test_harm_check_error_fallback_higher_tier(self):
        """Error fallback is harmful for higher tier."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        service = AutonomousAgentService(llm=llm)

        # Tier 1 - should be potentially harmful on fallback
        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Suggestion",
            reasoning="Help",
            risk_tier=ActionTier.TIER_1,
        )

        result = await service._check_for_harm(action)

        assert result.potentially_harmful is True


class TestSemanticSafetyGuardrail:
    """Test semantic safety check guardrail."""

    @pytest.mark.asyncio
    async def test_safe_response(self):
        """SAFE response passes check."""
        llm = MockLLMProvider(responses=["SAFE: This action is clearly safe"])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Learn",
        )

        result = await service._semantic_safety_check(action)

        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_unsafe_response(self):
        """Non-SAFE response fails check."""
        llm = MockLLMProvider(responses=["UNSAFE: Could be risky"])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research private info",
            reasoning="Curiosity",
        )

        result = await service._semantic_safety_check(action)

        assert result.is_safe is False

    @pytest.mark.asyncio
    async def test_error_fallback_tier_0_safe(self):
        """Error fallback is safe for tier 0."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("Error"))
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
            risk_tier=ActionTier.TIER_0,
        )

        result = await service._semantic_safety_check(action)

        assert result.is_safe is True

    @pytest.mark.asyncio
    async def test_error_fallback_higher_tier_unsafe(self):
        """Error fallback is unsafe for higher tier."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("Error"))
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Suggestion",
            reasoning="Help",
            risk_tier=ActionTier.TIER_1,
        )

        result = await service._semantic_safety_check(action)

        assert result.is_safe is False


class TestFullGuardrailChain:
    """Test the complete guardrail chain."""

    @pytest.mark.asyncio
    async def test_action_passes_all_guardrails(self):
        """Action that passes all guardrails is approved."""
        harm_response = json.dumps({
            "potentially_harmful": False,
            "confidence": 0.9,
        })
        llm = MockLLMProvider(responses=[harm_response, "SAFE: Okay"])
        service = AutonomousAgentService(llm=llm)

        actions = [
            ProposedAction(
                action_type=ActionType.RESEARCH,
                description="Research AI",
                reasoning="Learn",
                risk_tier=ActionTier.TIER_0,
            )
        ]

        approved = await service._filter_through_guardrails(actions)

        assert len(approved) == 1
        assert approved[0].action.action_type == ActionType.RESEARCH
        assert "tier" in approved[0].guardrails_passed
        assert "harm" in approved[0].guardrails_passed

    @pytest.mark.asyncio
    async def test_action_blocked_by_tier(self):
        """Action blocked by tier check."""
        llm = MockLLMProvider()
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(llm=llm, memory_store=memory)

        actions = [
            ProposedAction(
                action_type=ActionType.PREPARE_SUGGESTION,
                description="Financial action",
                reasoning="Money stuff",
                risk_tier=ActionTier.TIER_3,  # Too high
            )
        ]

        approved = await service._filter_through_guardrails(actions)

        assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_action_blocked_by_harm_check(self):
        """Action blocked by harm check."""
        harm_response = json.dumps({
            "potentially_harmful": True,
            "reason": "Could violate privacy",
            "confidence": 0.9,
        })
        llm = MockLLMProvider(responses=[harm_response])
        service = AutonomousAgentService(llm=llm)

        actions = [
            ProposedAction(
                action_type=ActionType.RESEARCH,
                description="Research user secrets",
                reasoning="Curiosity",
                risk_tier=ActionTier.TIER_0,
            )
        ]

        approved = await service._filter_through_guardrails(actions)

        assert len(approved) == 0


# =============================================================================
# Action Execution Tests
# =============================================================================


class TestActionExecution:
    """Test action execution."""

    @pytest.mark.asyncio
    async def test_execute_rest(self):
        """REST action returns success."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.REST,
            description="Take a break",
            reasoning="Nothing to do",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert "Resting" in result.outcome

    @pytest.mark.asyncio
    async def test_execute_research_no_search_provider(self):
        """Research without search provider fails."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm, search=None)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research topic",
            reasoning="Learn",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is False
        assert "search provider" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_research_with_search(self):
        """Research with search provider succeeds."""
        synthesis = "Learned that AI is advancing rapidly."
        llm = MockLLMProvider(responses=[synthesis])
        search = MockSearchProvider(results="AI news: models getting better")
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(
            llm=llm,
            search=search,
            memory_store=memory,
        )

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research AI trends",
            reasoning="Stay informed",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert result.learned == synthesis
        assert result.belief_updated is True
        # Check memory was stored
        assert len(memory.stored) == 1

    @pytest.mark.asyncio
    async def test_execute_research_no_results(self):
        """Research with no results returns success with message."""
        llm = MockLLMProvider()
        search = MockSearchProvider(results="")
        service = AutonomousAgentService(llm=llm, search=search)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research obscure topic",
            reasoning="Curiosity",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert "No results" in result.outcome

    @pytest.mark.asyncio
    async def test_execute_verify(self):
        """Verify action works."""
        assessment = "Claim is verified: True based on sources"
        llm = MockLLMProvider(responses=[assessment])
        search = MockSearchProvider(results="Source confirms claim")
        service = AutonomousAgentService(llm=llm, search=search)

        action = ProposedAction(
            action_type=ActionType.VERIFY,
            description="Verify that water boils at 100C",
            reasoning="Fact check",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert result.learned == assessment

    @pytest.mark.asyncio
    async def test_execute_reflect(self):
        """Reflect action works."""
        reflection = json.dumps({
            "summary": "Performance is good",
            "trait_adjustments": {"curiosity": 0.05},
        })
        llm = MockLLMProvider(responses=[reflection])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.REFLECT,
            description="Reflect on recent actions",
            reasoning="Self-improvement",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert result.belief_updated is True

    @pytest.mark.asyncio
    async def test_execute_note_question(self):
        """Note question action stores to memory."""
        llm = MockLLMProvider()
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(llm=llm, memory_store=memory)

        action = ProposedAction(
            action_type=ActionType.NOTE_QUESTION,
            description="What is user's favorite color?",
            reasoning="Want to know",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert "Noted question" in result.outcome
        assert len(memory.stored) == 1

    @pytest.mark.asyncio
    async def test_execute_prepare_suggestion(self):
        """Prepare suggestion stores to memory."""
        llm = MockLLMProvider()
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(llm=llm, memory_store=memory)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Suggest taking umbrella",
            reasoning="Rain forecast",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert "Prepared suggestion" in result.outcome

    @pytest.mark.asyncio
    async def test_execute_update_belief(self):
        """Update belief stores to memory."""
        llm = MockLLMProvider()
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(llm=llm, memory_store=memory)

        action = ProposedAction(
            action_type=ActionType.UPDATE_BELIEF,
            description="User prefers morning meetings",
            reasoning="Observed pattern",
        )
        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is True
        assert result.belief_updated is True

    @pytest.mark.asyncio
    async def test_execute_unknown_type_error(self):
        """Unknown action type returns error."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        # Create action with a mocked unknown type
        action = ProposedAction(
            action_type=ActionType.REST,
            description="Test",
            reasoning="Test",
        )
        # Manually override to simulate unknown
        action.action_type = MagicMock()
        action.action_type.value = "unknown"
        action.action_type.__eq__ = lambda self, other: False

        approved = ApprovedAction(action=action, approved_at=datetime.now())

        result = await service._execute_action(approved)

        assert result.success is False
        assert "Unknown action type" in result.error


# =============================================================================
# Run Cycle Tests
# =============================================================================


class TestRunCycle:
    """Test the main run_cycle method."""

    @pytest.mark.asyncio
    async def test_budget_exhausted(self):
        """Returns empty when budget exhausted."""
        llm = MockLLMProvider()
        config = AutonomousConfig(daily_action_budget=5)
        service = AutonomousAgentService(llm=llm, config=config)

        service._actions_today = 5  # Already at budget

        results = await service.run_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_no_proposals(self):
        """Returns empty when no proposals."""
        response = json.dumps({"proposed_actions": []})
        llm = MockLLMProvider(responses=[response])
        service = AutonomousAgentService(llm=llm)

        results = await service.run_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_shadow_mode_logs_but_doesnt_execute(self):
        """Shadow mode logs actions but doesn't execute."""
        proposal_response = json.dumps({
            "proposed_actions": [
                {
                    "type": "research",
                    "description": "Research topic",
                    "reasoning": "Learn",
                    "risk_tier": 0,
                }
            ]
        })
        harm_response = json.dumps({"potentially_harmful": False})

        llm = MockLLMProvider(responses=[proposal_response, harm_response, "SAFE"])
        config = AutonomousConfig(shadow_mode=True)
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        results = await service.run_cycle()

        # In shadow mode, results are empty (no real execution)
        assert results == []
        # But action should be logged
        assert len(service._action_logs) >= 1

    @pytest.mark.asyncio
    async def test_respects_max_actions_per_cycle(self):
        """Only executes up to max_actions_per_cycle."""
        proposals = json.dumps({
            "proposed_actions": [
                {"type": "research", "description": f"Research {i}", "reasoning": "Learn", "risk_tier": 0}
                for i in range(5)
            ]
        })
        # Need multiple harm check and safety responses
        responses = [proposals] + ['{"potentially_harmful": false}', "SAFE"] * 5
        llm = MockLLMProvider(responses=responses)
        config = AutonomousConfig(max_actions_per_cycle=2)
        search = MockSearchProvider()  # Provide search for research
        service = AutonomousAgentService(llm=llm, config=config, search=search)

        results = await service.run_cycle()

        assert len(results) <= 2


# =============================================================================
# Self-Monitoring Tests
# =============================================================================


class TestSelfMonitoring:
    """Test self-monitoring functionality."""

    @pytest.mark.asyncio
    async def test_self_monitor_parses_findings(self):
        """Self-monitor parses findings from LLM."""
        monitoring_response = json.dumps({
            "findings": [
                {
                    "type": "unexpected_result",
                    "description": "Got odd result",
                    "severity": "low",
                    "action_recommended": "Review",
                }
            ],
            "notify_user": False,
            "lessons_learned": ["Be more careful"],
        })
        llm = MockLLMProvider(responses=[monitoring_response])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        # Should complete without error
        assert True

    @pytest.mark.asyncio
    async def test_self_monitor_queues_notification(self):
        """Self-monitor queues notification when needed."""
        monitoring_response = json.dumps({
            "findings": [],
            "notify_user": True,
            "notification_message": "Important update",
            "lessons_learned": [],
        })
        llm = MockLLMProvider(responses=[monitoring_response])
        notification = MockNotificationProvider()
        service = AutonomousAgentService(
            llm=llm,
            notification_provider=notification,
        )

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        assert len(notification.notifications) == 1
        assert notification.notifications[0]["message"] == "Important update"

    @pytest.mark.asyncio
    async def test_self_monitor_stores_high_severity_findings(self):
        """High severity findings are stored."""
        monitoring_response = json.dumps({
            "findings": [
                {
                    "type": "contradiction",
                    "description": "Found contradiction",
                    "severity": "high",
                }
            ],
            "notify_user": False,
            "lessons_learned": [],
        })
        llm = MockLLMProvider(responses=[monitoring_response])
        memory = MockMemoryStoreProvider()
        service = AutonomousAgentService(llm=llm, memory_store=memory)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._self_monitor([(action, result)])

        assert len(memory.logs) == 1

    @pytest.mark.asyncio
    async def test_self_monitor_handles_parse_error(self):
        """Self-monitor handles JSON parse errors."""
        llm = MockLLMProvider(responses=["not valid json"])
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        # Should not raise
        await service._self_monitor([(action, result)])

    @pytest.mark.asyncio
    async def test_self_monitor_handles_llm_error(self):
        """Self-monitor handles LLM errors."""
        llm = MockLLMProvider()
        llm.generate = AsyncMock(side_effect=Exception("LLM error"))
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        # Should not raise
        await service._self_monitor([(action, result)])


# =============================================================================
# Transparency API Tests
# =============================================================================


class TestTransparencyAPI:
    """Test transparency API methods."""

    def test_get_action_logs_empty(self):
        """Get logs returns empty when no logs."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        logs = service.get_action_logs()

        assert logs == []

    def test_get_action_logs_filters_by_days(self):
        """Get logs filters by days."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        # Add old log
        old_log = ActionLog(
            action_id="old",
            action_type="research",
            description="Old action",
            reasoning="Old",
            started_at=datetime.now() - timedelta(days=10),
        )
        # Add recent log
        recent_log = ActionLog(
            action_id="recent",
            action_type="research",
            description="Recent action",
            reasoning="Recent",
            started_at=datetime.now(),
        )
        service._action_logs = [old_log, recent_log]

        logs = service.get_action_logs(days=7)

        assert len(logs) == 1
        assert logs[0].action_id == "recent"

    def test_get_blocked_logs(self):
        """Get blocked logs only returns blocked."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        blocked_log = ActionLog(
            action_id="blocked",
            action_type="research",
            description="Blocked",
            reasoning="Risk",
            started_at=datetime.now(),
            blocked=True,
            blocked_reason="tier_violation",
        )
        normal_log = ActionLog(
            action_id="normal",
            action_type="research",
            description="Normal",
            reasoning="Learn",
            started_at=datetime.now(),
        )
        service._action_logs = [blocked_log, normal_log]

        logs = service.get_blocked_logs()

        assert len(logs) == 1
        assert logs[0].blocked is True

    def test_get_stats(self):
        """Get stats returns correct data."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            shadow_mode=False,
            daily_action_budget=20,
        )
        service = AutonomousAgentService(llm=llm, config=config)

        service._actions_today = 5
        blocked_log = ActionLog(
            action_id="1",
            action_type="research",
            description="Blocked",
            reasoning="Risk",
            started_at=datetime.now(),
            blocked=True,
        )
        service._action_logs = [blocked_log]

        stats = service.get_stats()

        assert stats["enabled"] is True
        assert stats["shadow_mode"] is False
        assert stats["actions_today"] == 5
        assert stats["daily_budget"] == 20
        assert stats["total_logs"] == 1
        assert stats["blocked_count"] == 1


# =============================================================================
# Action Logging Tests
# =============================================================================


class TestActionLogging:
    """Test action logging functionality."""

    @pytest.mark.asyncio
    async def test_log_action_appends_to_list(self):
        """Log action appends to action_logs list."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._log_action(action, result)

        assert len(service._action_logs) == 1
        assert service._action_logs[0].action_type == "research"

    @pytest.mark.asyncio
    async def test_log_action_updates_recent_types(self):
        """Log action updates recent_action_types."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._log_action(action, result)

        assert ActionType.RESEARCH in service._recent_action_types

    @pytest.mark.asyncio
    async def test_log_action_persists_to_memory(self):
        """Log action persists to memory store."""
        llm = MockLLMProvider()
        memory = MockMemoryStoreProvider()
        config = AutonomousConfig(persist_logs=True)
        service = AutonomousAgentService(
            llm=llm,
            config=config,
            memory_store=memory,
        )

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._log_action(action, result)

        assert len(memory.logs) == 1

    @pytest.mark.asyncio
    async def test_log_action_bounds_history(self):
        """Log action bounds in-memory history."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        # Fill up logs beyond limit
        for i in range(1100):
            log = ActionLog(
                action_id=f"log_{i}",
                action_type="research",
                description=f"Action {i}",
                reasoning="Test",
                started_at=datetime.now(),
            )
            service._action_logs.append(log)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="Research",
            reasoning="Learn",
        )
        result = ActionResult(success=True, outcome="Done")

        await service._log_action(action, result)

        # Should be trimmed to ~500
        assert len(service._action_logs) <= 502

    @pytest.mark.asyncio
    async def test_log_blocked_stores_log(self):
        """Log blocked stores to action logs."""
        llm = MockLLMProvider()
        config = AutonomousConfig(log_all_proposals=True)
        service = AutonomousAgentService(llm=llm, config=config)

        action = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="Risky action",
            reasoning="Test",
            risk_tier=ActionTier.TIER_3,
        )

        await service._log_blocked(action, "tier_violation")

        assert len(service._action_logs) == 1
        assert service._action_logs[0].blocked is True
        assert service._action_logs[0].blocked_reason == "tier_violation"
