"""Tests for ArchitectAgent integration."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from draagon_ai.orchestration.architect_agent import (
    ArchitectAgent,
    ArchitectResult,
    create_architect_agent,
)
from draagon_ai.behaviors.types import Behavior, BehaviorStatus, BehaviorTier


class TestArchitectResult:
    """Tests for ArchitectResult dataclass."""

    def test_minimal_result(self):
        """Test creating a minimal result."""
        result = ArchitectResult(
            behavior=None,
            success=True,
            message="Done",
            phases_completed=[],
        )
        assert result.success is True
        assert result.message == "Done"
        assert result.behavior is None
        assert result.test_pass_rate == 0.0
        assert result.evolution_applied is False

    def test_full_result(self):
        """Test creating a full result with all fields."""
        behavior = Behavior(
            behavior_id="test",
            name="Test Behavior",
            description="A test behavior",
            version="1.0.0",
            tier=BehaviorTier.APPLICATION,
            status=BehaviorStatus.ACTIVE,
        )
        result = ArchitectResult(
            success=True,
            message="Created behavior",
            behavior=behavior,
            phases_completed=["research", "design", "build"],
            test_pass_rate=0.95,
            evolution_applied=True,
        )
        assert result.behavior is not None
        assert result.behavior.behavior_id == "test"
        assert result.test_pass_rate == 0.95
        assert result.evolution_applied is True
        assert len(result.phases_completed) == 3


class TestArchitectAgentInit:
    """Tests for ArchitectAgent initialization."""

    def test_init_with_llm(self):
        """Test initialization with an LLM."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)
        assert agent._architect_service is not None

    def test_init_with_registry(self):
        """Test initialization with behavior registry."""
        mock_llm = MagicMock()
        mock_registry = MagicMock()
        agent = ArchitectAgent(llm=mock_llm, behavior_registry=mock_registry)
        assert agent._architect_service._registry == mock_registry

    def test_init_with_web_search(self):
        """Test initialization with web search."""
        mock_llm = MagicMock()
        mock_search = MagicMock()
        agent = ArchitectAgent(llm=mock_llm, web_search=mock_search)
        assert agent._architect_service._web_search == mock_search


class TestArchitectAgentResearch:
    """Tests for ArchitectAgent.research method."""

    @pytest.mark.asyncio
    async def test_research_returns_dict(self):
        """Test that research returns a dict with expected keys."""
        mock_llm = MagicMock()

        # Mock the research_domain method
        with patch.object(
            ArchitectAgent, '_architect_service', create=True
        ) as mock_service:
            mock_result = MagicMock()
            mock_result.domain = "smart home"
            mock_result.core_tasks = ["control lights"]
            mock_result.suggested_actions = [{"name": "turn_on"}]
            mock_result.suggested_triggers = ["voice command"]
            mock_result.constraints = ["safety first"]
            mock_result.domain_knowledge = "Smart homes use IoT"
            mock_result.sources = ["wikipedia"]

            mock_service.research_domain = AsyncMock(return_value=mock_result)

            agent = ArchitectAgent(llm=mock_llm)
            # Inject the mock service
            agent._architect_service = mock_service

            result = await agent.research("smart home management")

            assert "domain" in result
            assert "core_tasks" in result
            assert "suggested_actions" in result
            assert result["domain"] == "smart home"
            assert "research" in agent._phases_completed


class TestArchitectAgentBuild:
    """Tests for ArchitectAgent.build method."""

    @pytest.mark.asyncio
    async def test_build_requires_design(self):
        """Test that build raises error without design."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        with pytest.raises(ValueError, match="No design available"):
            await agent.build()


class TestArchitectAgentTest:
    """Tests for ArchitectAgent.test method."""

    @pytest.mark.asyncio
    async def test_test_requires_behavior(self):
        """Test that test raises error without behavior."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        with pytest.raises(ValueError, match="No behavior available"):
            await agent.test()


class TestArchitectAgentRegister:
    """Tests for ArchitectAgent.register method."""

    @pytest.mark.asyncio
    async def test_register_requires_behavior(self):
        """Test that register raises error without behavior."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        with pytest.raises(ValueError, match="No behavior available"):
            await agent.register()


class TestArchitectAgentCreateBehavior:
    """Tests for ArchitectAgent.create_behavior."""

    @pytest.mark.asyncio
    async def test_create_behavior_returns_result(self):
        """Test that create_behavior returns an ArchitectResult."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        # Mock the service method to raise an exception (simpler test)
        with patch.object(
            agent._architect_service, 'create_behavior',
            AsyncMock(side_effect=Exception("LLM not configured"))
        ):
            result = await agent.create_behavior("test behavior")

            assert isinstance(result, ArchitectResult)
            assert result.success is False
            assert "LLM not configured" in result.message

    @pytest.mark.asyncio
    async def test_create_behavior_success(self):
        """Test successful behavior creation."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        # Create a mock behavior
        mock_behavior = MagicMock()
        mock_behavior.name = "Test Behavior"
        mock_behavior.actions = [MagicMock(), MagicMock()]
        mock_behavior.test_results = MagicMock()
        mock_behavior.test_results.pass_rate = 0.9

        with patch.object(
            agent._architect_service, 'create_behavior',
            AsyncMock(return_value=mock_behavior)
        ):
            result = await agent.create_behavior("test behavior")

            assert isinstance(result, ArchitectResult)
            assert result.success is True
            assert result.behavior is not None
            assert "Test Behavior" in result.message
            assert result.test_pass_rate == 0.9


class TestArchitectAgentReset:
    """Tests for ArchitectAgent.reset method."""

    def test_reset_clears_state(self):
        """Test that reset clears all creation state."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)

        # Set some state
        agent._current_design = {"behavior_id": "test"}
        agent._current_behavior = MagicMock()
        agent._phases_completed = ["research", "design"]

        agent.reset()

        assert agent._current_design is None
        assert agent._current_behavior is None
        assert agent._phases_completed == []


class TestArchitectAgentGetCurrentBehavior:
    """Tests for ArchitectAgent.get_current_behavior method."""

    def test_get_current_behavior_returns_none_initially(self):
        """Test that get_current_behavior returns None initially."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)
        assert agent.get_current_behavior() is None

    def test_get_current_behavior_returns_behavior(self):
        """Test that get_current_behavior returns the current behavior."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)
        mock_behavior = MagicMock()
        agent._current_behavior = mock_behavior
        assert agent.get_current_behavior() == mock_behavior


class TestArchitectServiceProperty:
    """Tests for ArchitectAgent.architect_service property."""

    def test_architect_service_property(self):
        """Test that architect_service property returns the service."""
        mock_llm = MagicMock()
        agent = ArchitectAgent(llm=mock_llm)
        assert agent.architect_service is not None
        assert agent.architect_service == agent._architect_service


class TestCreateArchitectAgent:
    """Tests for create_architect_agent factory function."""

    @pytest.mark.asyncio
    async def test_create_with_llm(self):
        """Test factory with LLM."""
        mock_llm = MagicMock()
        agent = await create_architect_agent(llm=mock_llm)
        assert isinstance(agent, ArchitectAgent)

    @pytest.mark.asyncio
    async def test_create_with_registry(self):
        """Test factory with registry."""
        mock_llm = MagicMock()
        mock_registry = MagicMock()
        agent = await create_architect_agent(llm=mock_llm, registry=mock_registry)
        assert isinstance(agent, ArchitectAgent)
        assert agent._architect_service._registry == mock_registry

    @pytest.mark.asyncio
    async def test_create_with_web_search(self):
        """Test factory with web search."""
        mock_llm = MagicMock()
        mock_search = MagicMock()
        agent = await create_architect_agent(llm=mock_llm, web_search=mock_search)
        assert isinstance(agent, ArchitectAgent)
        assert agent._architect_service._web_search == mock_search
