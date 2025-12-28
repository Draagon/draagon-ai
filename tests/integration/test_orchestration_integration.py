"""Integration tests for the orchestration module (REQ-002-09).

Tests the complete orchestration flow with real components:
1. Single-step tool execution
2. Multi-step ReAct reasoning
3. Error recovery and continuation
4. Timeout handling
5. Context propagation across steps

These tests use real tool execution (not mocked) to verify
end-to-end behavior of the orchestration system.
"""

import asyncio
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.orchestration import (
    Agent,
    AgentConfig,
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
    DecisionEngine,
    DecisionResult,
    ActionExecutor,
    ActionResult,
    ToolRegistry,
    Tool,
    ToolParameter,
    LoopMode,
    StepType,
)
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts, ActionParameter


pytestmark = [
    pytest.mark.integration,
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tool_registry():
    """Create a tool registry with real tool handlers.

    Tool handlers receive (args: dict, context: dict | None) as positional args.
    """
    registry = ToolRegistry()

    # Simple time tool - always succeeds
    async def get_time_handler(args: dict, context: dict | None = None) -> dict:
        return {"time": datetime.now().strftime("%I:%M %p")}

    registry.register(Tool(
        name="get_time",
        description="Get the current time",
        handler=get_time_handler,
    ))

    # Weather tool with location parameter
    async def get_weather_handler(args: dict, context: dict | None = None) -> dict:
        location = args.get("location", "home")
        return {
            "weather": "Sunny",
            "temperature": "72°F",
            "location": location,
        }

    registry.register(Tool(
        name="get_weather",
        description="Get weather for a location",
        handler=get_weather_handler,
        parameters=[
            ToolParameter(
                name="location",
                type="string",
                description="Location to get weather for",
                required=False,
                default="home",
            ),
        ],
    ))

    # Search tool that returns multiple results
    async def search_handler(args: dict, context: dict | None = None) -> list[dict]:
        query = args.get("query", "")
        return [
            {"title": f"Result 1 for {query}", "snippet": "First result"},
            {"title": f"Result 2 for {query}", "snippet": "Second result"},
            {"title": f"Result 3 for {query}", "snippet": "Third result"},
        ]

    registry.register(Tool(
        name="search",
        description="Search for information",
        handler=search_handler,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Search query",
                required=True,
            ),
        ],
    ))

    # Calculator tool
    async def calculate_handler(args: dict, context: dict | None = None) -> dict:
        expression = args.get("expression", "0")
        try:
            # Safe eval for simple math
            result = eval(expression, {"__builtins__": {}}, {})
            return {"result": result, "expression": expression}
        except Exception as e:
            return {"error": str(e), "expression": expression}

    registry.register(Tool(
        name="calculate",
        description="Evaluate a mathematical expression",
        handler=calculate_handler,
        parameters=[
            ToolParameter(
                name="expression",
                type="string",
                description="Mathematical expression to evaluate",
                required=True,
            ),
        ],
    ))

    # Slow tool for timeout testing
    async def slow_handler(args: dict, context: dict | None = None) -> dict:
        delay = args.get("delay", 5.0)
        await asyncio.sleep(delay)
        return {"completed": True, "delay": delay}

    registry.register(Tool(
        name="slow_operation",
        description="A slow operation for testing timeouts",
        handler=slow_handler,
        timeout_ms=100,  # Very short timeout for testing
        parameters=[
            ToolParameter(
                name="delay",
                type="number",
                description="Delay in seconds",
                required=False,
                default=5.0,
            ),
        ],
    ))

    # Error-prone tool for error recovery testing
    async def error_prone_handler(args: dict, context: dict | None = None) -> dict:
        should_fail = args.get("should_fail", False)
        if should_fail:
            raise ValueError("Simulated tool failure")
        return {"status": "success"}

    registry.register(Tool(
        name="error_prone",
        description="A tool that can fail on command",
        handler=error_prone_handler,
        parameters=[
            ToolParameter(
                name="should_fail",
                type="boolean",
                description="Whether to simulate failure",
                required=False,
                default=False,
            ),
        ],
    ))

    # Counter tool for context propagation testing
    call_count = {"value": 0}

    async def counter_handler(args: dict, context: dict | None = None) -> dict:
        call_count["value"] += 1
        return {"count": call_count["value"]}

    registry.register(Tool(
        name="increment_counter",
        description="Increment and return a counter",
        handler=counter_handler,
    ))

    return registry


@pytest.fixture
def behavior():
    """Create a behavior with all test actions."""
    return Behavior(
        behavior_id="test_integration",
        name="Integration Test Behavior",
        description="Behavior for integration testing",
        actions=[
            Action(name="answer", description="Provide a direct answer"),
            Action(name="get_time", description="Get current time"),
            Action(name="get_weather", description="Get weather"),
            Action(name="search", description="Search for info"),
            Action(name="calculate", description="Calculate expression"),
            Action(name="slow_operation", description="Slow operation"),
            Action(name="error_prone", description="Error prone operation"),
            Action(name="increment_counter", description="Increment counter"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide what action to take for: {question}",
            synthesis_prompt="Synthesize response from: {tool_results}",
        ),
    )


@pytest.fixture
def mock_llm():
    """Create a mock LLM that can be programmed for specific responses."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def agent_context():
    """Create an agent context for testing."""
    return AgentContext(
        user_id="integration_test_user",
        session_id="integration_test_session",
        conversation_id="integration_test_conv",
        debug=True,
    )


# =============================================================================
# Test Classes
# =============================================================================


class TestSingleStepExecution:
    """Tests for single-step tool execution (REQ-002-09 Scenario 1)."""

    @pytest.mark.anyio
    async def test_simple_tool_execution(self, tool_registry, behavior, mock_llm, agent_context):
        """Test executing a single tool and getting a result."""
        # Configure mock LLM to select get_time action (must use <response> wrapper)
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <reasoning>User wants to know the time</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        # Create components
        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        # Process query
        response = await loop.process(
            query="What time is it?",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 1
        assert response.tool_results[0].action_name == "get_time"
        assert "time" in str(response.tool_results[0].result)

    @pytest.mark.anyio
    async def test_tool_with_parameters(self, tool_registry, behavior, mock_llm, agent_context):
        """Test executing a tool with parameters."""
        # Use JSON format since it supports args parsing
        mock_response = MagicMock()
        mock_response.content = '{"action": "get_weather", "args": {"location": "New York"}, "reasoning": "User wants weather for NYC"}'
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What's the weather in New York?",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) >= 1
        result = response.tool_results[0].result
        assert result["location"] == "New York"
        assert "weather" in result

    @pytest.mark.anyio
    async def test_direct_answer_no_tool(self, tool_registry, behavior, mock_llm, agent_context):
        """Test direct answer without tool execution."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>answer</action>
        <answer>Hello! How can I help you today?</answer>
        <reasoning>Simple greeting, no tool needed</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Hello!",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "Hello! How can I help you today?"
        assert len(response.tool_results) == 0

    @pytest.mark.anyio
    async def test_tool_returns_list(self, tool_registry, behavior, mock_llm, agent_context):
        """Test tool that returns a list of results."""
        # Use JSON format since it supports args parsing
        mock_response = MagicMock()
        mock_response.content = '{"action": "search", "args": {"query": "python tutorials"}, "reasoning": "User wants to search"}'
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Search for python tutorials",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        result = response.tool_results[0].result
        assert isinstance(result, list)
        assert len(result) == 3


class TestMultiStepReActReasoning:
    """Tests for multi-step ReAct reasoning (REQ-002-09 Scenario 2)."""

    @pytest.mark.anyio
    async def test_two_step_reasoning(self, tool_registry, behavior, mock_llm, agent_context):
        """Test a query that requires two steps to complete."""
        # First call: search for info
        # Second call: provide final answer
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # First iteration: search
                response.content = """
                <action>search</action>
                <args>{"query": "weather forecast"}</args>
                <reasoning>I need to search for weather information first</reasoning>
                """
            else:
                # Second iteration: final answer
                response.content = """
                <action>answer</action>
                <answer>Based on my search, the weather looks good!</answer>
                <reasoning>I have enough information to answer</reasoning>
                """
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Search for the weather and tell me if it's good",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.loop_mode == LoopMode.REACT
        assert len(response.tool_results) >= 1
        assert response.iterations_used >= 2

    @pytest.mark.anyio
    async def test_three_step_calculation(self, tool_registry, behavior, mock_llm, agent_context):
        """Test a query requiring multiple calculations."""
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = """
                <action>calculate</action>
                <args>{"expression": "10 + 5"}</args>
                <reasoning>First calculate 10 + 5</reasoning>
                """
            elif call_count[0] == 2:
                response.content = """
                <action>calculate</action>
                <args>{"expression": "15 * 2"}</args>
                <reasoning>Now multiply the result by 2</reasoning>
                """
            else:
                response.content = """
                <action>answer</action>
                <answer>The result of (10 + 5) * 2 is 30</answer>
                <reasoning>I have the final answer</reasoning>
                """
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Calculate (10 + 5) * 2 step by step",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2  # Two calculate actions
        assert call_count[0] == 3

    @pytest.mark.anyio
    async def test_react_steps_recorded(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that ReAct steps are properly recorded."""
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = """
                <action>get_time</action>
                <args>{}</args>
                <reasoning>Getting the current time</reasoning>
                """
            else:
                response.content = """
                <action>answer</action>
                <answer>Done!</answer>
                <reasoning>Task complete</reasoning>
                """
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="Get time",
            behavior=behavior,
            context=agent_context,
        )

        assert len(response.react_steps) >= 2

        # Check step types present
        step_types = [step.type for step in response.react_steps]
        assert StepType.ACTION in step_types
        assert StepType.OBSERVATION in step_types

    @pytest.mark.anyio
    async def test_max_iterations_enforced(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that max iterations limit is enforced."""
        # Always return an action, never a final answer
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <args>{}</args>
        <reasoning>Keep checking time</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=3),
        )

        response = await loop.process(
            query="Keep looping",
            behavior=behavior,
            context=agent_context,
        )

        # Should have stopped at max iterations
        assert response.iterations_used <= 3


class TestErrorRecoveryAndContinuation:
    """Tests for error recovery and continuation (REQ-002-09 Scenario 3)."""

    @pytest.mark.anyio
    async def test_tool_error_captured_not_thrown(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that tool errors are captured, not thrown."""
        # Use JSON format since it supports args parsing
        mock_response = MagicMock()
        mock_response.content = '{"action": "error_prone", "args": {"should_fail": true}, "reasoning": "Testing error handling"}'
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        # Should not raise, error should be captured
        response = await loop.process(
            query="Run error prone tool",
            behavior=behavior,
            context=agent_context,
        )

        # Error should be in the tool result
        assert response.tool_results[0].success is False
        assert response.tool_results[0].error is not None

    @pytest.mark.anyio
    async def test_react_continues_after_error(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that ReAct continues after a tool error."""
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # First: try error-prone tool (use JSON for args)
                response.content = '{"action": "error_prone", "args": {"should_fail": true}, "reasoning": "Trying the risky operation"}'
            elif call_count[0] == 2:
                # Second: try a working tool after seeing error
                response.content = """<response>
                <action>get_time</action>
                <reasoning>Error occurred, trying fallback</reasoning>
                </response>"""
            else:
                # Third: final answer
                response.content = """<response>
                <action>answer</action>
                <answer>Recovered from error and got the time</answer>
                <reasoning>Task complete after recovery</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Try risky operation then fallback",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert call_count[0] == 3

        # Should have results from both tools
        assert len(response.tool_results) == 2
        assert response.tool_results[0].success is False  # Error
        assert response.tool_results[1].success is True   # Recovery

    @pytest.mark.anyio
    async def test_unknown_action_handled(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that unknown actions are handled gracefully."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>nonexistent_tool</action>
        <args>{}</args>
        <reasoning>Trying unknown tool</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm, fallback_to_answer=True)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        # Should not crash
        response = await loop.process(
            query="Use unknown tool",
            behavior=behavior,
            context=agent_context,
        )

        # Should have some response (fallback or error)
        assert response is not None


class TestTimeoutHandling:
    """Tests for timeout handling (REQ-002-09 Scenario 4)."""

    @pytest.mark.anyio
    async def test_tool_timeout(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that tool timeouts are handled."""
        # Use JSON format since it supports args parsing
        mock_response = MagicMock()
        mock_response.content = '{"action": "slow_operation", "args": {"delay": 10.0}, "reasoning": "Running slow operation"}'
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Run slow operation",
            behavior=behavior,
            context=agent_context,
        )

        # Tool should have timed out
        assert response.tool_results[0].timed_out is True
        assert response.tool_results[0].success is False

    @pytest.mark.anyio
    async def test_timeout_override(self, tool_registry, behavior, mock_llm, agent_context):
        """Test timeout override via action executor."""
        # Add a tool with longer default timeout
        async def medium_handler(args: dict, context: dict | None = None):
            await asyncio.sleep(0.05)  # 50ms
            return {"done": True}

        tool_registry.register(Tool(
            name="medium_operation",
            description="Medium speed operation",
            handler=medium_handler,
            timeout_ms=200,  # 200ms timeout
        ))

        # Add action to behavior
        behavior.actions.append(Action(name="medium_operation", description="Medium op"))

        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>medium_operation</action>
        <args>{}</args>
        <reasoning>Running medium operation</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Run medium operation",
            behavior=behavior,
            context=agent_context,
        )

        # Should complete successfully within timeout
        assert response.tool_results[0].success is True
        assert response.tool_results[0].timed_out is False


class TestContextPropagation:
    """Tests for context propagation across steps (REQ-002-09 Scenario 5)."""

    @pytest.mark.anyio
    async def test_observations_accumulate(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that observations accumulate across ReAct iterations."""
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = """
                <action>increment_counter</action>
                <args>{}</args>
                <reasoning>First increment</reasoning>
                """
            elif call_count[0] == 2:
                response.content = """
                <action>increment_counter</action>
                <args>{}</args>
                <reasoning>Second increment</reasoning>
                """
            else:
                response.content = """
                <action>answer</action>
                <answer>Counter incremented twice</answer>
                <reasoning>Done incrementing</reasoning>
                """
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="Increment counter twice",
            behavior=behavior,
            context=agent_context,
        )

        # Both tool results should be recorded
        assert len(response.tool_results) == 2

        # Counter should have been incremented in sequence
        assert response.tool_results[0].result["count"] == 1
        assert response.tool_results[1].result["count"] == 2

    @pytest.mark.anyio
    async def test_context_user_id_passed(self, tool_registry, behavior, mock_llm):
        """Test that user_id from context is available during execution."""
        user_id_seen = [None]

        async def context_aware_handler(args: dict, context: dict | None = None) -> dict:
            # The executor passes context info
            if context:
                user_id_seen[0] = context.get("user_id")
            return {"seen_user": user_id_seen[0]}

        tool_registry.register(Tool(
            name="context_tool",
            description="Tool that checks context",
            handler=context_aware_handler,
        ))

        behavior.actions.append(Action(name="context_tool", description="Check context"))

        context = AgentContext(
            user_id="test_user_123",
            session_id="test_session",
            conversation_id="test_conv",
        )

        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>context_tool</action>
        <args>{}</args>
        <reasoning>Checking context</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Check context",
            behavior=behavior,
            context=context,
        )

        assert response.success is True
        assert user_id_seen[0] == "test_user_123"

    @pytest.mark.anyio
    async def test_debug_info_includes_context(self, tool_registry, behavior, mock_llm):
        """Test that debug info includes context information."""
        context = AgentContext(
            user_id="debug_user",
            session_id="debug_session",
            conversation_id="debug_conv",
            debug=True,
        )

        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <args>{}</args>
        <reasoning>Getting time for debug test</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Get time",
            behavior=behavior,
            context=context,
        )

        # Debug info should be populated
        assert "decision" in response.debug_info
        assert response.debug_info["decision"]["action"] == "get_time"


class TestModeSelection:
    """Tests for mode selection and auto-detection."""

    @pytest.mark.anyio
    async def test_simple_mode_single_iteration(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that simple mode only runs one iteration."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <args>{}</args>
        <reasoning>Getting time</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Get time",
            behavior=behavior,
            context=agent_context,
        )

        assert response.loop_mode == LoopMode.SIMPLE
        # Simple mode should not have multiple iterations
        assert len(response.react_steps) == 0

    @pytest.mark.anyio
    async def test_react_mode_allows_multiple_iterations(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that ReAct mode allows multiple iterations."""
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] < 3:
                response.content = """
                <action>get_time</action>
                <args>{}</args>
                <reasoning>Still working</reasoning>
                """
            else:
                response.content = """
                <action>answer</action>
                <answer>Done after 3 iterations</answer>
                <reasoning>Finished</reasoning>
                """
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=10),
        )

        response = await loop.process(
            query="Multi-step task",
            behavior=behavior,
            context=agent_context,
        )

        assert response.loop_mode == LoopMode.REACT
        assert response.iterations_used == 3

    @pytest.mark.anyio
    async def test_mode_override(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that mode can be overridden per-request."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>answer</action>
        <answer>Direct answer</answer>
        <reasoning>Simple response</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        # Create loop with REACT default
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        # Override to SIMPLE for this request
        response = await loop.process(
            query="Simple query",
            behavior=behavior,
            context=agent_context,
            mode_override=LoopMode.SIMPLE,
        )

        assert response.loop_mode == LoopMode.SIMPLE


class TestToolMetrics:
    """Tests for tool execution metrics collection."""

    @pytest.mark.anyio
    async def test_metrics_collected(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that tool execution metrics are collected."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <args>{}</args>
        <reasoning>Getting time</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        await loop.process(
            query="Get time",
            behavior=behavior,
            context=agent_context,
        )

        # Check metrics were collected
        metrics = tool_registry.get_metrics("get_time")
        assert metrics is not None
        assert metrics.invocation_count >= 1
        assert metrics.success_count >= 1

    @pytest.mark.anyio
    async def test_latency_recorded(self, tool_registry, behavior, mock_llm, agent_context):
        """Test that response latency is recorded."""
        mock_response = MagicMock()
        mock_response.content = """<response>
        <action>get_time</action>
        <args>{}</args>
        <reasoning>Getting time</reasoning>
        </response>"""
        mock_llm.chat.return_value = mock_response

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="Get time",
            behavior=behavior,
            context=agent_context,
        )

        # Latency should be recorded
        assert response.latency_ms > 0

        # Tool result should have latency recorded
        assert response.tool_results[0].latency_ms is not None
        assert response.tool_results[0].latency_ms >= 0
