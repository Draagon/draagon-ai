"""Tests for RoxyOrchestrationAdapter.

REQ-002-06: Roxy adapter for orchestration

Tests cover:
- Adapter initialization
- Tool registration
- Process method with same API as Roxy
- Response conversion
- Debug info with thought traces
- Error handling
- Context management
"""

import pytest
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from draagon_ai.adapters.roxy_orchestration import (
    RoxyOrchestrationAdapter,
    RoxyToolDefinition,
    RoxyResponse,
    ToolCallInfo,
    DebugInfo,
    create_roxy_orchestration_adapter,
)
from draagon_ai.orchestration import (
    AgentResponse,
    LoopMode,
    ReActStep,
    StepType,
    ToolParameter,
    DecisionResult,
)
from draagon_ai.behaviors import VOICE_ASSISTANT_TEMPLATE


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = AsyncMock()
    llm.chat = AsyncMock(return_value=MagicMock(
        content="Test response",
        role="assistant",
    ))
    return llm


@pytest.fixture
def mock_memory():
    """Create a mock memory provider."""
    memory = AsyncMock()
    memory.search = AsyncMock(return_value=[])
    return memory


@pytest.fixture
def sample_tool_handler():
    """Create a sample async tool handler."""
    async def handler(args: dict, **kwargs) -> dict:
        return {"result": "success", "args": args}
    return handler


@pytest.fixture
def sample_tools(sample_tool_handler):
    """Create sample tool definitions."""
    return [
        RoxyToolDefinition(
            name="get_time",
            description="Get current time",
            handler=sample_tool_handler,
        ),
        RoxyToolDefinition(
            name="get_weather",
            description="Get weather for location",
            handler=sample_tool_handler,
            parameters=[
                ToolParameter(
                    name="location",
                    type="string",
                    description="Location name",
                    required=False,
                )
            ],
        ),
        RoxyToolDefinition(
            name="execute_command",
            description="Execute shell command",
            handler=sample_tool_handler,
            parameters=[
                ToolParameter(
                    name="command",
                    type="string",
                    description="Command to run",
                    required=True,
                )
            ],
            requires_confirmation=True,
            timeout_ms=60000,
        ),
    ]


@pytest.fixture
def adapter(mock_llm, mock_memory):
    """Create a RoxyOrchestrationAdapter."""
    return RoxyOrchestrationAdapter(
        llm=mock_llm,
        memory=mock_memory,
        agent_id="test_roxy",
        agent_name="Test Roxy",
        personality_intro="You are a test assistant.",
    )


# =============================================================================
# Tests: Initialization
# =============================================================================


class TestAdapterInit:
    """Tests for adapter initialization."""

    def test_default_init(self, mock_llm):
        """Test adapter with minimal arguments."""
        adapter = RoxyOrchestrationAdapter(llm=mock_llm)

        assert adapter.agent_id == "roxy"
        assert adapter.agent_name == "Roxy"
        assert adapter.loop_mode == LoopMode.AUTO
        assert len(adapter.tool_registry) == 0

    def test_custom_init(self, mock_llm, mock_memory):
        """Test adapter with custom configuration."""
        adapter = RoxyOrchestrationAdapter(
            llm=mock_llm,
            memory=mock_memory,
            agent_id="custom_agent",
            agent_name="Custom Agent",
            personality_intro="Custom personality",
            default_model_tier="complex",
            loop_mode=LoopMode.REACT,
        )

        assert adapter.agent_id == "custom_agent"
        assert adapter.agent_name == "Custom Agent"
        assert adapter.loop_mode == LoopMode.REACT
        assert adapter._agent_config.default_model_tier == "complex"

    def test_init_with_behavior(self, mock_llm):
        """Test adapter with custom behavior."""
        adapter = RoxyOrchestrationAdapter(
            llm=mock_llm,
            behavior=VOICE_ASSISTANT_TEMPLATE,
        )

        assert adapter.behavior == VOICE_ASSISTANT_TEMPLATE


# =============================================================================
# Tests: Tool Registration
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration."""

    def test_register_single_tool(self, adapter, sample_tool_handler):
        """Test registering a single tool."""
        tool_def = RoxyToolDefinition(
            name="test_tool",
            description="A test tool",
            handler=sample_tool_handler,
        )

        adapter.register_tool(tool_def)

        assert "test_tool" in adapter.get_registered_tools()
        assert len(adapter.tool_registry) == 1

    def test_register_multiple_tools(self, adapter, sample_tools):
        """Test registering multiple tools."""
        adapter.register_tools(sample_tools)

        registered = adapter.get_registered_tools()
        assert len(registered) == 3
        assert "get_time" in registered
        assert "get_weather" in registered
        assert "execute_command" in registered

    def test_tool_with_parameters(self, adapter, sample_tool_handler):
        """Test registering tool with parameters."""
        tool_def = RoxyToolDefinition(
            name="search_web",
            description="Search the web",
            handler=sample_tool_handler,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
                ToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum results",
                    required=False,
                    default=10,
                ),
            ],
        )

        adapter.register_tool(tool_def)

        schemas = adapter.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["name"] == "search_web"
        assert "query" in schemas[0]["parameters"]
        assert "max_results" in schemas[0]["parameters"]

    def test_tool_with_confirmation(self, adapter, sample_tool_handler):
        """Test registering tool that requires confirmation."""
        tool_def = RoxyToolDefinition(
            name="delete_file",
            description="Delete a file",
            handler=sample_tool_handler,
            requires_confirmation=True,
        )

        adapter.register_tool(tool_def)

        tool = adapter.tool_registry.get_tool("delete_file")
        assert tool is not None
        assert tool.requires_confirmation is True

    def test_tool_with_custom_timeout(self, adapter, sample_tool_handler):
        """Test registering tool with custom timeout."""
        tool_def = RoxyToolDefinition(
            name="slow_operation",
            description="A slow operation",
            handler=sample_tool_handler,
            timeout_ms=120000,  # 2 minutes
        )

        adapter.register_tool(tool_def)

        tool = adapter.tool_registry.get_tool("slow_operation")
        assert tool is not None
        assert tool.timeout_ms == 120000

    def test_get_tool_schemas(self, adapter, sample_tools):
        """Test getting tool schemas for LLM."""
        adapter.register_tools(sample_tools)

        schemas = adapter.get_tool_schemas()

        assert len(schemas) == 3
        for schema in schemas:
            assert "name" in schema
            assert "description" in schema
            assert "parameters" in schema


# =============================================================================
# Tests: Process Method
# =============================================================================


class TestProcess:
    """Tests for the process method."""

    @pytest.mark.asyncio
    async def test_process_basic(self, adapter, sample_tools):
        """Test basic process call."""
        adapter.register_tools(sample_tools)

        # Mock the agent's process method
        mock_response = AgentResponse(
            response="It is 10:30 AM.",
            success=True,
        )

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="What time is it?",
                user_id="doug",
                conversation_id="conv_123",
            )

        assert isinstance(response, RoxyResponse)
        assert response.response == "It is 10:30 AM."
        assert response.success is True
        assert response.conversation_id == "conv_123"

    @pytest.mark.asyncio
    async def test_process_with_debug(self, adapter, sample_tools):
        """Test process with debug enabled."""
        adapter.register_tools(sample_tools)

        # Create decision with confidence
        mock_decision = DecisionResult(
            action="get_time",
            args={},
            confidence=0.95,
            reasoning="User wants the current time.",
        )

        mock_response = AgentResponse(
            response="It is 10:30 AM.",
            success=True,
            iterations_used=1,
            loop_mode=LoopMode.SIMPLE,
            decision=mock_decision,
            react_steps=[
                ReActStep(
                    type=StepType.THOUGHT,
                    content="User wants the current time.",
                    duration_ms=50.0,
                ),
                ReActStep(
                    type=StepType.ACTION,
                    content="get_time",
                    action_name="get_time",
                    action_args={},
                    duration_ms=20.0,
                ),
                ReActStep(
                    type=StepType.OBSERVATION,
                    content="10:30 AM",
                    duration_ms=5.0,
                ),
                ReActStep(
                    type=StepType.FINAL_ANSWER,
                    content="It is 10:30 AM.",
                    duration_ms=30.0,
                ),
            ],
        )

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="What time is it?",
                user_id="doug",
                debug=True,
            )

        assert response.debug is not None
        assert response.debug.confidence == 0.95
        assert response.debug.loop_mode == "simple"
        assert response.debug.iterations_used == 1
        assert response.debug.thoughts is not None
        assert len(response.debug.thoughts) == 1
        assert "current time" in response.debug.thoughts[0]
        assert response.debug.react_steps is not None
        assert len(response.debug.react_steps) == 4

    @pytest.mark.asyncio
    async def test_process_with_area_id(self, adapter, sample_tools):
        """Test process with area_id for room-aware commands."""
        adapter.register_tools(sample_tools)

        mock_response = AgentResponse(
            response="Turning off the lights.",
            success=True,
        )

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="Turn off the lights",
                user_id="doug",
                area_id="master_bedroom",
                debug=True,
            )

        assert response.debug is not None
        assert response.debug.area_id == "master_bedroom"

    @pytest.mark.asyncio
    async def test_process_multi_step(self, adapter, sample_tools):
        """Test process with multi-step reasoning."""
        adapter.register_tools(sample_tools)

        mock_response = AgentResponse(
            response="Found 3 events. Added concert to your calendar.",
            success=True,
            iterations_used=2,
            loop_mode=LoopMode.REACT,
            react_steps=[
                ReActStep(
                    type=StepType.THOUGHT,
                    content="User wants to search and add event.",
                ),
                ReActStep(
                    type=StepType.ACTION,
                    content="search_calendar",
                    action_name="search_calendar",
                    action_args={"query": "concert"},
                ),
                ReActStep(
                    type=StepType.OBSERVATION,
                    content="[{\"title\": \"Concert\", \"date\": \"2025-12-28\"}]",
                ),
                ReActStep(
                    type=StepType.THOUGHT,
                    content="Found the concert. Now adding to calendar.",
                ),
                ReActStep(
                    type=StepType.ACTION,
                    content="create_event",
                    action_name="create_event",
                    action_args={"summary": "Concert"},
                ),
                ReActStep(
                    type=StepType.OBSERVATION,
                    content="{\"id\": \"evt_123\", \"created\": true}",
                ),
                ReActStep(
                    type=StepType.FINAL_ANSWER,
                    content="Found 3 events. Added concert to your calendar.",
                ),
            ],
        )

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="Find the concert and add it to my calendar",
                user_id="doug",
                debug=True,
            )

        assert response.debug is not None
        assert response.debug.multi_step_reasoning is True
        assert response.debug.iterations_used == 2
        assert response.debug.loop_mode == "react"
        assert len(response.debug.thoughts) == 2

    @pytest.mark.asyncio
    async def test_process_error_handling(self, adapter, sample_tools):
        """Test process error handling."""
        adapter.register_tools(sample_tools)

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(side_effect=Exception("Test error"))
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="What time is it?",
                user_id="doug",
                debug=True,
            )

        assert response.success is False
        assert "error" in response.response.lower()
        assert response.debug is not None
        assert len(response.debug.errors) == 1
        assert "Test error" in response.debug.errors[0]["error"]


# =============================================================================
# Tests: Response Conversion
# =============================================================================


class TestResponseConversion:
    """Tests for response conversion."""

    def test_extract_tool_calls_empty(self, adapter):
        """Test extracting tool calls from empty response."""
        response = AgentResponse(response="Hello", react_steps=[])

        tool_calls = adapter._extract_tool_calls(response)

        assert tool_calls == []

    def test_extract_tool_calls_with_action(self, adapter):
        """Test extracting tool calls with action step."""
        response = AgentResponse(
            response="The time is 10:30.",
            react_steps=[
                ReActStep(
                    type=StepType.ACTION,
                    content="get_time",
                    action_name="get_time",
                    action_args={},
                    duration_ms=50.0,
                ),
                ReActStep(
                    type=StepType.OBSERVATION,
                    content="10:30 AM",
                ),
            ],
        )

        tool_calls = adapter._extract_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool == "get_time"
        assert tool_calls[0].args == {}
        assert tool_calls[0].result == "10:30 AM"
        assert tool_calls[0].elapsed_ms == 50

    def test_extract_tool_calls_with_error(self, adapter):
        """Test extracting tool calls with error observation."""
        response = AgentResponse(
            response="Command failed.",
            react_steps=[
                ReActStep(
                    type=StepType.ACTION,
                    content="execute_command",
                    action_name="execute_command",
                    action_args={"command": "ls"},
                ),
                ReActStep(
                    type=StepType.OBSERVATION,
                    content={"error": "Permission denied"},
                ),
            ],
        )

        tool_calls = adapter._extract_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].tool == "execute_command"
        assert tool_calls[0].error == "Permission denied"

    def test_extract_thoughts(self, adapter):
        """Test extracting thoughts from ReAct steps."""
        response = AgentResponse(
            response="Done.",
            react_steps=[
                ReActStep(type=StepType.THOUGHT, content="First thought"),
                ReActStep(type=StepType.ACTION, content="some_action"),
                ReActStep(type=StepType.THOUGHT, content="Second thought"),
            ],
        )

        thoughts = adapter._extract_thoughts(response)

        assert len(thoughts) == 2
        assert thoughts[0] == "First thought"
        assert thoughts[1] == "Second thought"

    def test_extract_thoughts_dict_content(self, adapter):
        """Test extracting thoughts from dict content."""
        response = AgentResponse(
            response="Done.",
            react_steps=[
                ReActStep(
                    type=StepType.THOUGHT,
                    content={"reasoning": "Complex reasoning here"},
                ),
            ],
        )

        thoughts = adapter._extract_thoughts(response)

        assert len(thoughts) == 1
        assert thoughts[0] == "Complex reasoning here"

    def test_convert_react_steps(self, adapter):
        """Test converting ReAct steps to dictionaries."""
        timestamp = datetime.now()
        steps = [
            ReActStep(
                type=StepType.THOUGHT,
                content="Thinking...",
                duration_ms=50.0,
                timestamp=timestamp,
            ),
            ReActStep(
                type=StepType.ACTION,
                content="test",
                duration_ms=100.0,
            ),
        ]

        converted = adapter._convert_react_steps(steps)

        assert len(converted) == 2
        assert converted[0]["type"] == "thought"
        assert converted[0]["content"] == "Thinking..."
        assert converted[0]["duration_ms"] == 50.0
        assert converted[0]["timestamp"] == timestamp.isoformat()
        assert converted[1]["type"] == "action"


# =============================================================================
# Tests: Context Management
# =============================================================================


class TestContextManagement:
    """Tests for context management."""

    def test_get_or_create_context_new(self, adapter):
        """Test creating new context."""
        context = adapter._get_or_create_context(
            conversation_id="conv_123",
            user_id="doug",
            area_id="kitchen",
            debug=True,
        )

        assert context.session_id == "conv_123"
        assert context.user_id == "doug"
        assert context.area_id == "kitchen"
        assert context.debug is True

    def test_get_or_create_context_existing(self, adapter):
        """Test getting existing context."""
        # Create first
        adapter._get_or_create_context(
            conversation_id="conv_123",
            user_id="doug",
            area_id="kitchen",
            debug=False,
        )

        # Get again with updated fields
        context = adapter._get_or_create_context(
            conversation_id="conv_123",
            user_id="doug",
            area_id="bedroom",
            debug=True,
        )

        # Should update mutable fields
        assert context.area_id == "bedroom"
        assert context.debug is True

    def test_clear_conversation(self, adapter):
        """Test clearing conversation history."""
        # Create context
        context = adapter._get_or_create_context(
            conversation_id="conv_123",
            user_id="doug",
            area_id=None,
            debug=False,
        )
        context.conversation_history.append({"user": "hello", "assistant": "hi"})
        context.pending_details = "Some details"

        # Clear it
        result = adapter.clear_conversation("conv_123")

        assert result is True
        assert len(context.conversation_history) == 0
        assert context.pending_details is None

    def test_clear_conversation_not_found(self, adapter):
        """Test clearing non-existent conversation."""
        result = adapter.clear_conversation("nonexistent")

        assert result is False


# =============================================================================
# Tests: Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for create_roxy_orchestration_adapter factory."""

    def test_create_basic(self, mock_llm):
        """Test creating adapter with factory function."""
        adapter = create_roxy_orchestration_adapter(llm=mock_llm)

        assert adapter.agent_id == "roxy"
        assert adapter.agent_name == "Roxy"
        assert adapter.loop_mode == LoopMode.AUTO

    def test_create_with_tools(self, mock_llm, sample_tools):
        """Test creating adapter with tools."""
        adapter = create_roxy_orchestration_adapter(
            llm=mock_llm,
            tools=sample_tools,
        )

        assert len(adapter.get_registered_tools()) == 3

    def test_create_with_custom_config(self, mock_llm, mock_memory):
        """Test creating adapter with custom configuration."""
        adapter = create_roxy_orchestration_adapter(
            llm=mock_llm,
            memory=mock_memory,
            agent_id="custom",
            agent_name="Custom Agent",
            personality_intro="Custom personality",
            loop_mode=LoopMode.REACT,
        )

        assert adapter.agent_id == "custom"
        assert adapter.agent_name == "Custom Agent"
        assert adapter.loop_mode == LoopMode.REACT


# =============================================================================
# Tests: Debug Info
# =============================================================================


class TestDebugInfo:
    """Tests for debug info generation."""

    @pytest.mark.asyncio
    async def test_debug_info_latency(self, adapter, sample_tools):
        """Test that latency is captured in debug info."""
        adapter.register_tools(sample_tools)

        mock_response = AgentResponse(response="Done.", success=True)

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="test",
                user_id="doug",
                debug=True,
            )

        assert response.debug is not None
        assert response.debug.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_debug_info_router_used(self, adapter, sample_tools):
        """Test that router_used is always true (we always use agent loop)."""
        adapter.register_tools(sample_tools)

        mock_response = AgentResponse(response="Done.", success=True)

        with patch.object(adapter, '_ensure_agent') as mock_ensure:
            mock_agent = MagicMock()
            mock_agent.process = AsyncMock(return_value=mock_response)
            mock_ensure.return_value = mock_agent

            response = await adapter.process(
                query="test",
                user_id="doug",
                debug=True,
            )

        assert response.debug is not None
        assert response.debug.router_used is True


# =============================================================================
# Tests: Agent Property
# =============================================================================


class TestAgentProperty:
    """Tests for agent property access."""

    def test_agent_none_before_process(self, adapter):
        """Test that agent is None before process is called."""
        assert adapter.agent is None

    def test_agent_created_after_ensure(self, adapter, sample_tools):
        """Test that agent is created after _ensure_agent."""
        adapter.register_tools(sample_tools)

        # Force agent creation
        agent = adapter._ensure_agent()

        assert adapter.agent is not None
        assert adapter.agent is agent


# =============================================================================
# Tests: ToolCallInfo
# =============================================================================


class TestToolCallInfo:
    """Tests for ToolCallInfo dataclass."""

    def test_tool_call_info_minimal(self):
        """Test ToolCallInfo with minimal fields."""
        info = ToolCallInfo(tool="test")

        assert info.tool == "test"
        assert info.args is None
        assert info.result is None
        assert info.elapsed_ms is None
        assert info.error is None

    def test_tool_call_info_full(self):
        """Test ToolCallInfo with all fields."""
        info = ToolCallInfo(
            tool="search_web",
            args={"query": "python"},
            result=["result1", "result2"],
            elapsed_ms=150,
            error=None,
        )

        assert info.tool == "search_web"
        assert info.args == {"query": "python"}
        assert info.result == ["result1", "result2"]
        assert info.elapsed_ms == 150


# =============================================================================
# Tests: RoxyToolDefinition
# =============================================================================


class TestRoxyToolDefinition:
    """Tests for RoxyToolDefinition dataclass."""

    def test_tool_definition_minimal(self, sample_tool_handler):
        """Test RoxyToolDefinition with minimal fields."""
        tool = RoxyToolDefinition(
            name="simple_tool",
            description="A simple tool",
            handler=sample_tool_handler,
        )

        assert tool.name == "simple_tool"
        assert tool.description == "A simple tool"
        assert tool.handler is sample_tool_handler
        assert tool.parameters == []
        assert tool.timeout_ms == 30000
        assert tool.requires_confirmation is False

    def test_tool_definition_full(self, sample_tool_handler):
        """Test RoxyToolDefinition with all fields."""
        params = [
            ToolParameter(name="param1", type="string", description="First param"),
            ToolParameter(name="param2", type="integer", description="Second param"),
        ]

        tool = RoxyToolDefinition(
            name="complex_tool",
            description="A complex tool",
            handler=sample_tool_handler,
            parameters=params,
            timeout_ms=60000,
            requires_confirmation=True,
        )

        assert tool.name == "complex_tool"
        assert len(tool.parameters) == 2
        assert tool.timeout_ms == 60000
        assert tool.requires_confirmation is True


# =============================================================================
# Tests: RoxyResponse
# =============================================================================


class TestRoxyResponse:
    """Tests for RoxyResponse dataclass."""

    def test_roxy_response_minimal(self):
        """Test RoxyResponse with minimal fields."""
        response = RoxyResponse(response="Hello")

        assert response.response == "Hello"
        assert response.tool_calls == []
        assert response.iterations == 1
        assert response.success is True
        assert response.conversation_id == "default"
        assert response.debug is None

    def test_roxy_response_full(self):
        """Test RoxyResponse with all fields."""
        tool_calls = [ToolCallInfo(tool="test")]
        debug = DebugInfo(latency_ms=100)

        response = RoxyResponse(
            response="Done.",
            tool_calls=tool_calls,
            iterations=3,
            success=True,
            conversation_id="conv_123",
            debug=debug,
        )

        assert response.response == "Done."
        assert len(response.tool_calls) == 1
        assert response.iterations == 3
        assert response.conversation_id == "conv_123"
        assert response.debug is not None
        assert response.debug.latency_ms == 100
