"""Tests for Agent and MultiAgent classes.

REQ-002-08: Unit tests (≥90% coverage)

These tests cover the Agent class which is the main entry point for using
draagon-ai orchestration.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.orchestration.agent import Agent, AgentConfig, MultiAgent
from draagon_ai.orchestration.loop import AgentContext, AgentResponse
from draagon_ai.behaviors import Behavior, Action


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock(return_value="Test response")
    return llm


@pytest.fixture
def mock_memory():
    """Create a mock memory provider."""
    memory = MagicMock()
    memory.search = AsyncMock(return_value=[])
    memory.store = AsyncMock()
    return memory


@pytest.fixture
def mock_tools():
    """Create a mock tool provider."""
    tools = MagicMock()
    tools.list_tools = MagicMock(return_value=["get_time", "search_web"])
    tools.get_tool_description = MagicMock(return_value="A test tool")
    tools.execute = AsyncMock(return_value="Tool result")
    return tools


@pytest.fixture
def sample_behavior():
    """Create a sample behavior for testing."""
    return Behavior(
        behavior_id="test_assistant",
        name="Test Assistant",
        description="A test assistant behavior",
        actions=[
            Action(name="answer", description="Respond to the user"),
            Action(name="get_time", description="Get current time"),
        ],
    )


@pytest.fixture
def agent_config():
    """Create a sample agent config."""
    return AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        personality_intro="I am a helpful test assistant.",
        default_model_tier="local",
        enable_learning=True,
        enable_proactive=False,
        metadata={"version": "1.0"},
    )


@pytest.fixture
def agent(agent_config, sample_behavior, mock_llm, mock_memory, mock_tools):
    """Create a test agent."""
    return Agent(
        config=agent_config,
        behavior=sample_behavior,
        llm=mock_llm,
        memory=mock_memory,
        tools=mock_tools,
    )


# =============================================================================
# AgentConfig Tests
# =============================================================================


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""

    def test_default_values(self):
        """Test AgentConfig has correct defaults."""
        config = AgentConfig(agent_id="test")

        assert config.agent_id == "test"
        assert config.name == ""
        assert config.personality_intro == ""
        assert config.default_model_tier == "local"
        assert config.enable_learning is True
        assert config.enable_proactive is False
        assert config.metadata == {}

    def test_custom_values(self):
        """Test AgentConfig with custom values."""
        config = AgentConfig(
            agent_id="custom",
            name="Custom Agent",
            personality_intro="I am custom.",
            default_model_tier="complex",
            enable_learning=False,
            enable_proactive=True,
            metadata={"key": "value"},
        )

        assert config.agent_id == "custom"
        assert config.name == "Custom Agent"
        assert config.personality_intro == "I am custom."
        assert config.default_model_tier == "complex"
        assert config.enable_learning is False
        assert config.enable_proactive is True
        assert config.metadata == {"key": "value"}


# =============================================================================
# Agent Initialization Tests
# =============================================================================


class TestAgentInit:
    """Tests for Agent initialization."""

    def test_minimal_init(self, mock_llm, sample_behavior):
        """Test agent can be created with minimal args."""
        config = AgentConfig(agent_id="minimal")
        agent = Agent(
            config=config,
            behavior=sample_behavior,
            llm=mock_llm,
        )

        assert agent.config == config
        assert agent.behavior == sample_behavior
        assert agent.llm == mock_llm
        assert agent.memory is None
        assert agent.tools is None

    def test_full_init(self, agent_config, sample_behavior, mock_llm, mock_memory, mock_tools):
        """Test agent with all components."""
        agent = Agent(
            config=agent_config,
            behavior=sample_behavior,
            llm=mock_llm,
            memory=mock_memory,
            tools=mock_tools,
        )

        assert agent.config == agent_config
        assert agent.behavior == sample_behavior
        assert agent.memory == mock_memory
        assert agent.tools == mock_tools
        assert agent._decision_engine is not None
        assert agent._action_executor is not None
        assert agent._loop is not None

    def test_init_without_tools(self, agent_config, sample_behavior, mock_llm, mock_memory):
        """Test agent without tools has no action executor."""
        agent = Agent(
            config=agent_config,
            behavior=sample_behavior,
            llm=mock_llm,
            memory=mock_memory,
            tools=None,
        )

        assert agent._action_executor is None

    def test_agent_id_property(self, agent):
        """Test agent_id property returns config agent_id."""
        assert agent.agent_id == "test-agent"

    def test_name_property_with_name(self, agent):
        """Test name property returns config name."""
        assert agent.name == "Test Agent"

    def test_name_property_fallback(self, mock_llm, sample_behavior):
        """Test name property falls back to agent_id."""
        config = AgentConfig(agent_id="fallback-agent", name="")
        agent = Agent(config=config, behavior=sample_behavior, llm=mock_llm)

        assert agent.name == "fallback-agent"


# =============================================================================
# Agent Process Tests
# =============================================================================


class TestAgentProcess:
    """Tests for Agent.process() method."""

    @pytest.mark.asyncio
    async def test_process_simple_query(self, agent):
        """Test processing a simple query."""
        # Mock the loop's process method
        mock_response = AgentResponse(
            response="It is 3:00 PM",
            action_taken="get_time",
        )
        agent._loop.process = AsyncMock(return_value=mock_response)

        response = await agent.process(
            query="What time is it?",
            user_id="test_user",
        )

        assert response.response == "It is 3:00 PM"
        assert response.action_taken == "get_time"
        agent._loop.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_with_session_id(self, agent):
        """Test processing with explicit session_id."""
        mock_response = AgentResponse(response="Hello!")
        agent._loop.process = AsyncMock(return_value=mock_response)

        await agent.process(
            query="Hello",
            user_id="user1",
            session_id="session123",
        )

        # Session should be stored with the session_id as key
        assert "session123" in agent._sessions

    @pytest.mark.asyncio
    async def test_process_with_area_id(self, agent):
        """Test processing with area_id."""
        mock_response = AgentResponse(response="Lights on!")
        agent._loop.process = AsyncMock(return_value=mock_response)

        await agent.process(
            query="Turn on the lights",
            user_id="user1",
            area_id="bedroom",
        )

        # Context should have area_id set
        session_key = "user1_default"
        assert agent._sessions[session_key].area_id == "bedroom"

    @pytest.mark.asyncio
    async def test_process_with_debug(self, agent):
        """Test processing with debug flag."""
        mock_response = AgentResponse(response="Debug mode!")
        agent._loop.process = AsyncMock(return_value=mock_response)

        await agent.process(
            query="Test",
            user_id="user1",
            debug=True,
        )

        # Context should have debug set
        session_key = "user1_default"
        assert agent._sessions[session_key].debug is True

    @pytest.mark.asyncio
    async def test_process_updates_history(self, agent):
        """Test that process updates conversation history."""
        mock_response = AgentResponse(response="Hello there!")
        agent._loop.process = AsyncMock(return_value=mock_response)

        await agent.process(query="Hello", user_id="user1")

        session_key = "user1_default"
        assert len(agent._sessions[session_key].conversation_history) == 1
        assert agent._sessions[session_key].conversation_history[0]["user"] == "Hello"
        assert agent._sessions[session_key].conversation_history[0]["assistant"] == "Hello there!"

    @pytest.mark.asyncio
    async def test_process_stores_pending_details(self, agent):
        """Test that full_response is stored in pending_details."""
        mock_response = AgentResponse(
            response="Short answer.",
            full_response="This is the full detailed answer with more information.",
        )
        agent._loop.process = AsyncMock(return_value=mock_response)

        await agent.process(query="Tell me about X", user_id="user1")

        session_key = "user1_default"
        assert agent._sessions[session_key].pending_details == "This is the full detailed answer with more information."

    @pytest.mark.asyncio
    async def test_process_multiple_queries_same_session(self, agent):
        """Test multiple queries in the same session."""
        agent._loop.process = AsyncMock(side_effect=[
            AgentResponse(response="First response"),
            AgentResponse(response="Second response"),
        ])

        await agent.process(query="First", user_id="user1")
        await agent.process(query="Second", user_id="user1")

        session_key = "user1_default"
        assert len(agent._sessions[session_key].conversation_history) == 2

    @pytest.mark.asyncio
    async def test_process_history_limit(self, agent):
        """Test conversation history is limited to max_history."""
        agent._loop.process = AsyncMock(return_value=AgentResponse(response="Reply"))

        # Send 15 messages (should be limited to 10)
        for i in range(15):
            await agent.process(query=f"Message {i}", user_id="user1")

        session_key = "user1_default"
        assert len(agent._sessions[session_key].conversation_history) == 10
        # Should keep the last 10 messages
        assert agent._sessions[session_key].conversation_history[0]["user"] == "Message 5"


# =============================================================================
# Agent Behavior Tests
# =============================================================================


class TestAgentBehavior:
    """Tests for behavior management."""

    @pytest.mark.asyncio
    async def test_set_behavior(self, agent):
        """Test setting a new behavior."""
        new_behavior = Behavior(
            behavior_id="new_behavior",
            name="New Behavior",
            description="A new behavior",
            actions=[],
        )

        await agent.set_behavior(new_behavior)

        assert agent.behavior == new_behavior

    @pytest.mark.asyncio
    async def test_get_behavior(self, agent, sample_behavior):
        """Test getting current behavior."""
        behavior = await agent.get_behavior()

        assert behavior == sample_behavior


# =============================================================================
# Agent Session Tests
# =============================================================================


class TestAgentSession:
    """Tests for session management."""

    @pytest.mark.asyncio
    async def test_get_session_exists(self, agent):
        """Test getting an existing session."""
        # Create a session by processing
        agent._loop.process = AsyncMock(return_value=AgentResponse(response="Hi"))
        await agent.process(query="Hello", user_id="user1", session_id="my_session")

        session = agent.get_session("my_session")

        assert session is not None
        assert session.user_id == "user1"
        assert session.session_id == "my_session"

    def test_get_session_not_exists(self, agent):
        """Test getting a non-existent session."""
        session = agent.get_session("nonexistent")

        assert session is None

    @pytest.mark.asyncio
    async def test_clear_session_exists(self, agent):
        """Test clearing an existing session."""
        # Create a session
        agent._loop.process = AsyncMock(return_value=AgentResponse(
            response="Hi",
            full_response="Full response",
        ))
        await agent.process(query="Hello", user_id="user1", session_id="clear_me")

        # Verify it has history and pending_details
        session = agent.get_session("clear_me")
        assert len(session.conversation_history) == 1
        assert session.pending_details == "Full response"

        # Clear it
        result = agent.clear_session("clear_me")

        assert result is True
        assert len(session.conversation_history) == 0
        assert session.pending_details is None

    def test_clear_session_not_exists(self, agent):
        """Test clearing a non-existent session."""
        result = agent.clear_session("nonexistent")

        assert result is False


# =============================================================================
# Agent Context Tests
# =============================================================================


class TestAgentContext:
    """Tests for context management."""

    @pytest.mark.asyncio
    async def test_get_or_create_context_new(self, agent):
        """Test creating a new context."""
        agent._loop.process = AsyncMock(return_value=AgentResponse(response="Hi"))
        await agent.process(query="Hello", user_id="new_user", session_id="new_session")

        # Context should be created
        assert "new_session" in agent._sessions
        ctx = agent._sessions["new_session"]
        assert ctx.user_id == "new_user"
        assert ctx.session_id == "new_session"

    @pytest.mark.asyncio
    async def test_get_or_create_context_existing(self, agent):
        """Test getting an existing context with updated fields."""
        agent._loop.process = AsyncMock(return_value=AgentResponse(response="Hi"))

        # First call creates context
        await agent.process(query="Hello", user_id="user1", session_id="my_session", area_id="kitchen")

        # Second call should update mutable fields
        await agent.process(query="Hi again", user_id="user1", session_id="my_session", area_id="bedroom", debug=True)

        ctx = agent._sessions["my_session"]
        assert ctx.area_id == "bedroom"  # Updated
        assert ctx.debug is True  # Updated


# =============================================================================
# MultiAgent Tests
# =============================================================================


class TestMultiAgentInit:
    """Tests for MultiAgent initialization."""

    def test_init(self, mock_llm):
        """Test MultiAgent initialization."""
        registry = MagicMock()

        multi = MultiAgent(llm=mock_llm, registry=registry)

        assert multi.llm == mock_llm
        assert multi.registry == registry
        assert multi._agents == {}
        assert multi._default_agent is None

    def test_init_with_memory_and_tools(self, mock_llm, mock_memory, mock_tools):
        """Test MultiAgent with memory and tools."""
        registry = MagicMock()

        multi = MultiAgent(
            llm=mock_llm,
            registry=registry,
            memory=mock_memory,
            tools=mock_tools,
        )

        assert multi.memory == mock_memory
        assert multi.tools == mock_tools


class TestMultiAgentAddAgent:
    """Tests for MultiAgent.add_agent()."""

    def test_add_agent(self, mock_llm, sample_behavior):
        """Test adding an agent."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        agent = multi.add_agent("assistant", sample_behavior)

        assert isinstance(agent, Agent)
        assert agent.agent_id == "assistant"
        assert "assistant" in multi._agents
        # First agent becomes default
        assert multi._default_agent == "assistant"

    def test_add_agent_with_personality(self, mock_llm, sample_behavior):
        """Test adding agent with personality intro."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        agent = multi.add_agent(
            "assistant",
            sample_behavior,
            personality_intro="I am helpful.",
        )

        assert agent.config.personality_intro == "I am helpful."

    def test_add_agent_set_default(self, mock_llm, sample_behavior):
        """Test adding agent as default."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        # Add first agent (auto-default)
        multi.add_agent("first", sample_behavior)
        # Add second agent as explicit default
        multi.add_agent("second", sample_behavior, set_default=True)

        assert multi._default_agent == "second"

    def test_add_multiple_agents(self, mock_llm, sample_behavior):
        """Test adding multiple agents."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        multi.add_agent("agent1", sample_behavior)
        multi.add_agent("agent2", sample_behavior)
        multi.add_agent("agent3", sample_behavior)

        assert len(multi._agents) == 3
        # First agent should still be default
        assert multi._default_agent == "agent1"


class TestMultiAgentGetAgent:
    """Tests for MultiAgent.get_agent()."""

    def test_get_agent_exists(self, mock_llm, sample_behavior):
        """Test getting an existing agent."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("test", sample_behavior)

        agent = multi.get_agent("test")

        assert agent is not None
        assert agent.agent_id == "test"

    def test_get_agent_not_exists(self, mock_llm):
        """Test getting a non-existent agent."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        agent = multi.get_agent("nonexistent")

        assert agent is None


class TestMultiAgentProcess:
    """Tests for MultiAgent.process()."""

    @pytest.mark.asyncio
    async def test_process_with_agent_id(self, mock_llm, sample_behavior):
        """Test processing with specific agent_id."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("assistant", sample_behavior)

        # Mock the agent's process method
        multi._agents["assistant"]._loop.process = AsyncMock(
            return_value=AgentResponse(response="Hello!")
        )

        response = await multi.process(
            query="Hello",
            user_id="user1",
            agent_id="assistant",
        )

        assert response.response == "Hello!"

    @pytest.mark.asyncio
    async def test_process_with_default_agent(self, mock_llm, sample_behavior):
        """Test processing with default agent."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("default_agent", sample_behavior)

        # Mock the agent's process method
        multi._agents["default_agent"]._loop.process = AsyncMock(
            return_value=AgentResponse(response="Default response")
        )

        response = await multi.process(query="Hello", user_id="user1")

        assert response.response == "Default response"

    @pytest.mark.asyncio
    async def test_process_unknown_agent(self, mock_llm, sample_behavior):
        """Test processing with unknown agent_id raises error."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("assistant", sample_behavior)

        with pytest.raises(ValueError, match="Unknown agent"):
            await multi.process(query="Hello", user_id="user1", agent_id="unknown")

    @pytest.mark.asyncio
    async def test_process_no_agents(self, mock_llm):
        """Test processing with no agents registered raises error."""
        registry = MagicMock()
        multi = MultiAgent(llm=mock_llm, registry=registry)

        with pytest.raises(ValueError, match="No agents registered"):
            await multi.process(query="Hello", user_id="user1")


class TestMultiAgentSelectAgent:
    """Tests for MultiAgent.select_agent()."""

    @pytest.mark.asyncio
    async def test_select_agent_matching_trigger(self, mock_llm, sample_behavior):
        """Test selecting agent based on trigger."""
        registry = MagicMock()
        registry.find_by_trigger = MagicMock(return_value=[sample_behavior])

        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("assistant", sample_behavior)

        agent = await multi.select_agent("Hello", {})

        assert agent is not None
        assert agent.agent_id == "assistant"

    @pytest.mark.asyncio
    async def test_select_agent_no_match_returns_default(self, mock_llm, sample_behavior):
        """Test select_agent returns default when no trigger matches."""
        registry = MagicMock()
        registry.find_by_trigger = MagicMock(return_value=[])

        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("default", sample_behavior)

        agent = await multi.select_agent("Unmatched query", {})

        assert agent is not None
        assert agent.agent_id == "default"

    @pytest.mark.asyncio
    async def test_select_agent_no_match_no_default(self, mock_llm):
        """Test select_agent returns None when no match and no default."""
        registry = MagicMock()
        registry.find_by_trigger = MagicMock(return_value=[])

        multi = MultiAgent(llm=mock_llm, registry=registry)
        # Don't add any agents

        agent = await multi.select_agent("Query", {})

        assert agent is None

    @pytest.mark.asyncio
    async def test_select_agent_matching_behavior_not_registered(self, mock_llm, sample_behavior):
        """Test select_agent when matching behavior's agent is not registered."""
        # Create a different behavior that triggers
        other_behavior = Behavior(
            behavior_id="other",
            name="Other",
            description="Other",
            actions=[],
        )

        registry = MagicMock()
        registry.find_by_trigger = MagicMock(return_value=[other_behavior])

        multi = MultiAgent(llm=mock_llm, registry=registry)
        multi.add_agent("assistant", sample_behavior)  # Different behavior

        agent = await multi.select_agent("Query", {})

        # Should return default since matching behavior's agent not found
        assert agent is not None
        assert agent.agent_id == "assistant"
