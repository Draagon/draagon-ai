"""Tests for the autonomous agent extension."""

import pytest
from datetime import datetime

from draagon_ai_ext_autonomous import (
    AutonomousExtension,
    AutonomousAgentService,
    AutonomousConfig,
    AutonomousContext,
    ActionType,
    ActionTier,
    ProposedAction,
)


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = '{"proposed_actions": []}'):
        self.response = response
        self.calls: list[dict] = []

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 1000,
        temperature: float = 0.7,
    ) -> str:
        self.calls.append({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        })
        return self.response


class MockSearchProvider:
    """Mock search provider for testing."""

    def __init__(self, results: str = "Test search results"):
        self.results = results

    async def search(self, query: str) -> str:
        return self.results


class TestAutonomousExtension:
    """Tests for AutonomousExtension class."""

    def test_info(self):
        """Extension info should have required fields."""
        ext = AutonomousExtension()
        info = ext.info

        assert info.name == "autonomous"
        assert info.version == "0.1.0"
        assert "autonomous_agent" in info.provides_services
        assert "autonomous" in info.provides_prompt_domains

    def test_initialize(self):
        """Extension should initialize with config."""
        ext = AutonomousExtension()
        ext.initialize({
            "enabled": True,
            "cycle_minutes": 15,
            "daily_budget": 10,
            "shadow_mode": True,
        })

        config = ext._config
        assert config.enabled is True
        assert config.cycle_interval_minutes == 15
        assert config.daily_action_budget == 10
        assert config.shadow_mode is True

    def test_create_service(self):
        """Extension should create service with providers."""
        ext = AutonomousExtension()
        ext.initialize({})

        llm = MockLLMProvider()
        service = ext.create_service(llm=llm)

        assert isinstance(service, AutonomousAgentService)
        assert service._llm == llm

    def test_get_prompt_domains(self):
        """Extension should provide prompt domains."""
        ext = AutonomousExtension()
        ext.initialize({})

        domains = ext.get_prompt_domains()
        assert "autonomous" in domains
        assert "AUTONOMOUS_AGENT_SYSTEM_PROMPT" in domains["autonomous"]
        assert "HARM_CHECK_PROMPT" in domains["autonomous"]


class TestAutonomousAgentService:
    """Tests for AutonomousAgentService class."""

    def test_init(self):
        """Service should initialize with config."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            daily_action_budget=5,
        )

        service = AutonomousAgentService(llm=llm, config=config)

        assert service.config.daily_action_budget == 5
        assert service._actions_today == 0

    def test_is_within_active_hours(self):
        """Service should check active hours correctly."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            active_hours_start=8,
            active_hours_end=22,
        )
        service = AutonomousAgentService(llm=llm, config=config)

        # Test within hours
        assert service._is_within_active_hours(
            datetime(2024, 1, 1, 12, 0, 0)
        ) is True

        # Test before hours
        assert service._is_within_active_hours(
            datetime(2024, 1, 1, 6, 0, 0)
        ) is False

        # Test after hours
        assert service._is_within_active_hours(
            datetime(2024, 1, 1, 23, 0, 0)
        ) is False

    def test_is_allowed_tier(self):
        """Service should check action tiers correctly."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        # Tier 0 should be allowed
        action0 = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="test",
            reasoning="test",
            risk_tier=ActionTier.TIER_0,
        )
        assert service._is_allowed_tier(action0) is True

        # Tier 1 should be allowed
        action1 = ProposedAction(
            action_type=ActionType.PREPARE_SUGGESTION,
            description="test",
            reasoning="test",
            risk_tier=ActionTier.TIER_1,
        )
        assert service._is_allowed_tier(action1) is True

        # Tier 2 should NOT be allowed
        action2 = ProposedAction(
            action_type=ActionType.REST,
            description="test",
            reasoning="test",
            risk_tier=ActionTier.TIER_2,
        )
        assert service._is_allowed_tier(action2) is False

    def test_exceeds_rate_limit(self):
        """Service should detect rate limit violations."""
        llm = MockLLMProvider()
        config = AutonomousConfig(max_consecutive_same_type=2)
        service = AutonomousAgentService(llm=llm, config=config)

        # Add two research actions
        service._recent_action_types = [ActionType.RESEARCH, ActionType.RESEARCH]

        # Third research should exceed limit
        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="test",
            reasoning="test",
        )
        assert service._exceeds_rate_limit(action) is True

        # Different action type should not exceed
        action2 = ProposedAction(
            action_type=ActionType.VERIFY,
            description="test",
            reasoning="test",
        )
        assert service._exceeds_rate_limit(action2) is False

    def test_get_stats(self):
        """Service should return correct stats."""
        llm = MockLLMProvider()
        config = AutonomousConfig(
            enabled=True,
            shadow_mode=False,
            daily_action_budget=20,
        )
        service = AutonomousAgentService(llm=llm, config=config)
        service._actions_today = 5

        stats = service.get_stats()

        assert stats["enabled"] is True
        assert stats["shadow_mode"] is False
        assert stats["actions_today"] == 5
        assert stats["daily_budget"] == 20

    @pytest.mark.asyncio
    async def test_gather_context_default(self):
        """Service should provide default context."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        context = await service._gather_context()

        assert isinstance(context, AutonomousContext)
        assert context.personality_context != ""
        assert "curiosity_intensity" in context.trait_values

    @pytest.mark.asyncio
    async def test_parse_proposals(self):
        """Service should parse LLM proposals correctly."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        response = '''```json
{
  "reasoning": "Test reasoning",
  "proposed_actions": [
    {
      "type": "research",
      "description": "Learn about AI",
      "reasoning": "Interesting topic",
      "risk_tier": 0,
      "reversible": true
    }
  ]
}
```'''

        proposals = service._parse_proposals(response)

        assert len(proposals) == 1
        assert proposals[0].action_type == ActionType.RESEARCH
        assert proposals[0].description == "Learn about AI"
        assert proposals[0].risk_tier == ActionTier.TIER_0

    @pytest.mark.asyncio
    async def test_execute_research_no_search(self):
        """Research without search provider should fail gracefully."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm, search=None)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="test topic",
            reasoning="test",
        )

        result = await service._execute_research(action)

        assert result.success is False
        assert "No search provider" in result.error

    @pytest.mark.asyncio
    async def test_execute_research_with_search(self):
        """Research with search provider should synthesize results."""
        llm = MockLLMProvider(response="This is what I learned about the topic.")
        search = MockSearchProvider(results="Search results about the topic...")
        service = AutonomousAgentService(llm=llm, search=search)

        action = ProposedAction(
            action_type=ActionType.RESEARCH,
            description="test topic",
            reasoning="test",
        )

        result = await service._execute_research(action)

        assert result.success is True
        assert result.learned is not None

    @pytest.mark.asyncio
    async def test_execute_rest(self):
        """Rest action should succeed without doing anything."""
        llm = MockLLMProvider()
        service = AutonomousAgentService(llm=llm)

        from draagon_ai_ext_autonomous.types import ApprovedAction

        action = ProposedAction(
            action_type=ActionType.REST,
            description="Taking a break",
            reasoning="Nothing useful to do",
        )
        approved = ApprovedAction(
            action=action,
            approved_at=datetime.now(),
        )

        result = await service._execute_action(approved)

        assert result.success is True
        assert "Resting" in result.outcome


class TestActionTypes:
    """Tests for action type enums."""

    def test_action_type_values(self):
        """ActionType should have expected values."""
        assert ActionType.RESEARCH.value == "research"
        assert ActionType.VERIFY.value == "verify"
        assert ActionType.REFLECT.value == "reflect"
        assert ActionType.REST.value == "rest"

    def test_action_tier_ordering(self):
        """ActionTier should have correct ordering."""
        assert ActionTier.TIER_0.value < ActionTier.TIER_1.value
        assert ActionTier.TIER_1.value < ActionTier.TIER_2.value
        assert ActionTier.TIER_2.value < ActionTier.TIER_3.value
        assert ActionTier.TIER_3.value < ActionTier.TIER_4.value
