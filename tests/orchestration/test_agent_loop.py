"""Tests for AgentLoop with ReAct support (REQ-002-01).

Tests the ReAct pattern implementation:
- Loop modes (SIMPLE, REACT, AUTO)
- Thought traces (THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER)
- Max iterations and timeout handling
- Context observation accumulation
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.orchestration import (
    AgentLoop,
    AgentContext,
    AgentResponse,
    AgentLoopConfig,
    LoopMode,
    ReActStep,
    StepType,
    DecisionEngine,
    DecisionResult,
    ActionExecutor,
    ActionResult,
)
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def mock_memory():
    """Create a mock memory provider."""
    memory = MagicMock()
    memory.search = AsyncMock(return_value=[])
    memory.store = AsyncMock()
    return memory


@pytest.fixture
def mock_decision_engine():
    """Create a mock decision engine."""
    engine = MagicMock(spec=DecisionEngine)
    engine.decide = AsyncMock()
    return engine


@pytest.fixture
def mock_action_executor():
    """Create a mock action executor."""
    executor = MagicMock(spec=ActionExecutor)
    executor.execute = AsyncMock()
    return executor


@pytest.fixture
def simple_behavior():
    """Create a simple behavior for testing."""
    return Behavior(
        behavior_id="test",
        name="Test Behavior",
        description="For testing",
        actions=[
            Action(name="answer", description="Provide an answer"),
            Action(name="search", description="Search for information"),
            Action(name="get_time", description="Get current time"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide action for: {question}",
            synthesis_prompt="Synthesize: {tool_results}",
        ),
    )


@pytest.fixture
def agent_context():
    """Create an agent context for testing."""
    return AgentContext(
        user_id="test_user",
        session_id="test_session",
        conversation_id="test_conv",
        debug=True,
    )


class TestAgentLoopConfig:
    """Tests for AgentLoopConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AgentLoopConfig()

        assert config.mode == LoopMode.AUTO
        assert config.max_iterations == 10
        assert config.iteration_timeout_seconds == 30.0
        assert config.complexity_threshold == 0.33  # Lowered for sensor queries
        assert config.log_thought_traces is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentLoopConfig(
            mode=LoopMode.REACT,
            max_iterations=5,
            iteration_timeout_seconds=15.0,
            complexity_threshold=0.5,
            log_thought_traces=False,
        )

        assert config.mode == LoopMode.REACT
        assert config.max_iterations == 5
        assert config.iteration_timeout_seconds == 15.0
        assert config.complexity_threshold == 0.5
        assert config.log_thought_traces is False

    def test_complexity_keywords(self):
        """Test default complexity keywords."""
        config = AgentLoopConfig()

        assert "and then" in config.complexity_keywords
        assert "after that" in config.complexity_keywords
        assert "compare" in config.complexity_keywords


class TestReActStep:
    """Tests for ReActStep dataclass."""

    def test_thought_step(self):
        """Test creating a THOUGHT step."""
        step = ReActStep(
            type=StepType.THOUGHT,
            content="I need to search for calendar events",
            duration_ms=150.0,
        )

        assert step.type == StepType.THOUGHT
        assert step.content == "I need to search for calendar events"
        assert step.duration_ms == 150.0
        assert step.action_name is None

    def test_action_step(self):
        """Test creating an ACTION step."""
        step = ReActStep(
            type=StepType.ACTION,
            content="search_calendar({'days': 7})",
            action_name="search_calendar",
            action_args={"days": 7},
        )

        assert step.type == StepType.ACTION
        assert step.action_name == "search_calendar"
        assert step.action_args == {"days": 7}

    def test_observation_step(self):
        """Test creating an OBSERVATION step."""
        step = ReActStep(
            type=StepType.OBSERVATION,
            content="Found 3 events",
            duration_ms=200.0,
            observation_success=True,
        )

        assert step.type == StepType.OBSERVATION
        assert step.observation_success is True
        assert step.observation_error is None

    def test_observation_error_step(self):
        """Test creating an OBSERVATION step with error."""
        step = ReActStep(
            type=StepType.OBSERVATION,
            content="Error: Service unavailable",
            observation_success=False,
            observation_error="Service unavailable",
        )

        assert step.observation_success is False
        assert step.observation_error == "Service unavailable"


class TestAgentContext:
    """Tests for AgentContext observation handling."""

    def test_add_observation(self):
        """Test adding observations to context."""
        ctx = AgentContext(user_id="test")

        ctx.add_observation("First result")
        ctx.add_observation("Second result")

        assert len(ctx.observations) == 2
        assert "First result" in ctx.observations

    def test_clear_observations(self):
        """Test clearing observations."""
        ctx = AgentContext(user_id="test")
        ctx.add_observation("Some observation")

        ctx.clear_observations()

        assert len(ctx.observations) == 0

    def test_get_observations_text(self):
        """Test formatting observations as text."""
        ctx = AgentContext(user_id="test")
        ctx.add_observation("Found 3 events")
        ctx.add_observation("Event details retrieved")

        text = ctx.get_observations_text()

        assert "Observation 1: Found 3 events" in text
        assert "Observation 2: Event details retrieved" in text

    def test_get_observations_text_empty(self):
        """Test observations text when empty."""
        ctx = AgentContext(user_id="test")

        text = ctx.get_observations_text()

        assert text == ""


class TestAgentResponse:
    """Tests for AgentResponse with ReAct steps."""

    def test_add_react_step(self):
        """Test adding ReAct steps to response."""
        response = AgentResponse(response="")

        step = response.add_react_step(
            step_type=StepType.THOUGHT,
            content="Thinking about the query",
            duration_ms=100.0,
        )

        assert len(response.react_steps) == 1
        assert step.type == StepType.THOUGHT
        assert response.react_steps[0] == step

    def test_add_action_step(self):
        """Test adding ACTION step with details."""
        response = AgentResponse(response="")

        step = response.add_react_step(
            step_type=StepType.ACTION,
            content="get_time({})",
            action_name="get_time",
            action_args={},
        )

        assert step.action_name == "get_time"
        assert step.action_args == {}

    def test_get_thought_trace(self):
        """Test getting thought trace as dict list."""
        response = AgentResponse(response="")
        response.add_react_step(StepType.THOUGHT, "Thinking")
        response.add_react_step(
            StepType.ACTION, "get_time()", action_name="get_time"
        )
        response.add_react_step(StepType.OBSERVATION, "10:30 AM", success=True)

        trace = response.get_thought_trace()

        assert len(trace) == 3
        assert trace[0]["step"] == 1
        assert trace[0]["type"] == "thought"
        assert trace[1]["type"] == "action"
        assert trace[1]["action_name"] == "get_time"
        assert trace[2]["type"] == "observation"


class TestAgentLoopSimpleMode:
    """Tests for AgentLoop in SIMPLE mode."""

    @pytest.mark.anyio
    async def test_simple_mode_direct_answer(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode with direct answer."""
        # Setup mock decision
        mock_decision_engine.decide.return_value = DecisionResult(
            action="answer",
            answer="It is 10:30 AM",
            reasoning="User asked for time, I can answer directly",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "It is 10:30 AM"
        assert response.loop_mode == LoopMode.SIMPLE
        assert len(response.react_steps) == 0  # No steps in simple mode

    @pytest.mark.anyio
    async def test_simple_mode_with_action(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode with action execution."""
        # Setup mocks
        mock_decision_engine.decide.return_value = DecisionResult(
            action="get_time",
            args={},
            reasoning="Need to get current time",
        )
        mock_action_executor.execute.return_value = ActionResult(
            action_name="get_time",
            success=True,
            result={"time": "10:30 AM"},
            formatted_result="The time is 10:30 AM",
            direct_answer="It is 10:30 AM",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "It is 10:30 AM"
        assert len(response.tool_results) == 1


class TestAgentLoopReActMode:
    """Tests for AgentLoop in ReAct mode."""

    @pytest.mark.anyio
    async def test_react_mode_single_iteration(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test ReAct mode with single iteration to final answer."""
        # First call returns a direct answer
        mock_decision_engine.decide.return_value = DecisionResult(
            action="answer",
            answer="It is 10:30 AM",
            reasoning="I can answer this directly without any tools",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "It is 10:30 AM"
        assert response.loop_mode == LoopMode.REACT
        assert response.iterations_used == 1
        # Should have THOUGHT and FINAL_ANSWER steps
        assert len(response.react_steps) == 2
        assert response.react_steps[0].type == StepType.THOUGHT
        assert response.react_steps[1].type == StepType.FINAL_ANSWER

    @pytest.mark.anyio
    async def test_react_mode_multi_step(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test ReAct mode with multiple steps."""
        # First iteration: search action
        # Second iteration: final answer
        call_count = 0

        async def decide_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DecisionResult(
                    action="search",
                    args={"query": "calendar events"},
                    reasoning="I need to search for calendar events first",
                )
            else:
                return DecisionResult(
                    action="answer",
                    answer="You have 3 events this week",
                    reasoning="Based on the search results, I can now answer",
                )

        mock_decision_engine.decide.side_effect = decide_side_effect
        mock_action_executor.execute.return_value = ActionResult(
            action_name="search",
            success=True,
            result={"events": [1, 2, 3]},
            formatted_result="Found 3 events",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="What events do I have this week?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "You have 3 events this week"
        assert response.iterations_used == 2
        # Steps: THOUGHT -> ACTION -> OBSERVATION -> THOUGHT -> FINAL_ANSWER
        assert len(response.react_steps) == 5
        assert response.react_steps[0].type == StepType.THOUGHT
        assert response.react_steps[1].type == StepType.ACTION
        assert response.react_steps[2].type == StepType.OBSERVATION
        assert response.react_steps[3].type == StepType.THOUGHT
        assert response.react_steps[4].type == StepType.FINAL_ANSWER

    @pytest.mark.anyio
    async def test_react_mode_max_iterations(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test ReAct mode hitting max iterations."""
        # Always return an action, never answer
        mock_decision_engine.decide.return_value = DecisionResult(
            action="search",
            args={"query": "more info"},
            reasoning="I need more information",
        )
        mock_action_executor.execute.return_value = ActionResult(
            action_name="search",
            success=True,
            result={},
            formatted_result="No results",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=3),
        )

        response = await loop.process(
            query="Find something",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is False
        assert response.iterations_used == 3
        assert "couldn't complete" in response.response.lower()
        assert response.debug_info.get("max_iterations_reached") is True

    @pytest.mark.anyio
    async def test_react_mode_action_error_recovery(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test ReAct mode recovers from action errors."""
        call_count = 0

        async def decide_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return DecisionResult(
                    action="search",
                    args={},
                    reasoning="Trying to search",
                )
            else:
                return DecisionResult(
                    action="answer",
                    answer="I found the information after retry",
                    reasoning="Got it on second try",
                )

        async def execute_side_effect(*args, **kwargs):
            nonlocal call_count
            if call_count == 1:
                return ActionResult(
                    action_name="search",
                    success=False,
                    result=None,
                    formatted_result="",
                    error="Service unavailable",
                )
            return ActionResult(
                action_name="search",
                success=True,
                result={"data": "found"},
                formatted_result="Found data",
            )

        mock_decision_engine.decide.side_effect = decide_side_effect
        mock_action_executor.execute.side_effect = execute_side_effect

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="Find information",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        # Should have recovered after the first error
        assert response.iterations_used == 3

    @pytest.mark.anyio
    async def test_react_mode_context_observations(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test that observations accumulate in context."""
        call_count = 0

        async def decide_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return DecisionResult(
                    action="search",
                    args={},
                    reasoning="Searching",
                )
            return DecisionResult(
                action="answer",
                answer="Done",
                reasoning="Complete",
            )

        mock_decision_engine.decide.side_effect = decide_side_effect
        mock_action_executor.execute.return_value = ActionResult(
            action_name="search",
            success=True,
            result={},
            formatted_result="Search result",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        await loop.process(
            query="Search twice",
            behavior=simple_behavior,
            context=agent_context,
        )

        # Context should have accumulated observations
        assert len(agent_context.observations) == 2


class TestAgentLoopAutoMode:
    """Tests for AgentLoop AUTO mode detection."""

    def test_detect_simple_query(self, mock_llm):
        """Test that simple queries use SIMPLE mode."""
        loop = AgentLoop(
            llm=mock_llm,
            config=AgentLoopConfig(mode=LoopMode.AUTO),
        )

        # Simple query without complexity keywords
        mode = loop._detect_complexity("What time is it?")
        assert mode == LoopMode.SIMPLE

    def test_detect_complex_query_and_then(self, mock_llm):
        """Test that 'and then' triggers REACT mode."""
        loop = AgentLoop(
            llm=mock_llm,
            config=AgentLoopConfig(mode=LoopMode.AUTO, complexity_threshold=0.3),
        )

        mode = loop._detect_complexity("Search for events and then add one")
        assert mode == LoopMode.REACT

    def test_detect_complex_query_multiple_keywords(self, mock_llm):
        """Test that multiple complexity keywords trigger REACT mode."""
        loop = AgentLoop(
            llm=mock_llm,
            config=AgentLoopConfig(mode=LoopMode.AUTO),
        )

        mode = loop._detect_complexity(
            "First check the calendar, then compare with the weather, and finally decide"
        )
        assert mode == LoopMode.REACT

    @pytest.mark.anyio
    async def test_mode_override(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test that mode_override bypasses AUTO detection."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="answer",
            answer="Done",
            reasoning="Direct answer",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(mode=LoopMode.AUTO),
        )

        # Force REACT mode even for simple query
        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
            mode_override=LoopMode.REACT,
        )

        assert response.loop_mode == LoopMode.REACT


class TestAgentLoopDebug:
    """Tests for debug output in AgentLoop."""

    @pytest.mark.anyio
    async def test_debug_thought_trace(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test that thought trace is in debug output."""
        call_count = 0

        async def decide_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return DecisionResult(
                    action="search",
                    args={},
                    reasoning="Need to search first",
                )
            return DecisionResult(
                action="answer",
                answer="Found it",
                reasoning="Got the answer",
            )

        mock_decision_engine.decide.side_effect = decide_side_effect
        mock_action_executor.execute.return_value = ActionResult(
            action_name="search",
            success=True,
            result={},
            formatted_result="Results",
        )

        agent_context.debug = True

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="Find something",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert "thought_trace" in response.debug_info
        trace = response.debug_info["thought_trace"]
        assert len(trace) > 0
        assert trace[0]["type"] == "thought"


class TestAgentLoopMemoryUpdate:
    """Tests for memory updates in ReAct mode."""

    @pytest.mark.anyio
    async def test_memory_update_on_final_answer(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test that memory updates are processed on final answer."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="answer",
            answer="Stored the birthday",
            reasoning="User told me their birthday",
            memory_update={
                "content": "User's birthday is March 15",
                "type": "fact",
                "entities": ["birthday"],
                "confidence": 0.9,
            },
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(mode=LoopMode.REACT),
        )

        response = await loop.process(
            query="My birthday is March 15",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert "User's birthday is March 15" in response.memories_stored
        mock_memory.store.assert_called_once()


# =============================================================================
# Additional Tests for Loop Coverage (REQ-002-08)
# =============================================================================


class TestAgentLoopGatherContext:
    """Tests for _gather_context method."""

    @pytest.mark.anyio
    async def test_gather_context_no_memory(
        self,
        mock_llm,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test _gather_context returns message when no memory provider."""
        loop = AgentLoop(
            llm=mock_llm,
            memory=None,  # No memory provider
            decision_engine=mock_decision_engine,
        )

        context_str, count = await loop._gather_context(
            query="Test query",
            context=agent_context,
        )

        assert context_str == "No context available."
        assert count == 0

    @pytest.mark.anyio
    async def test_gather_context_with_results(
        self,
        mock_llm,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test _gather_context formats memory results."""
        # Create mock memory with results
        mock_memory = MagicMock()
        mock_result = MagicMock()
        mock_result.memory_type = "fact"
        mock_result.content = "User likes pizza"
        mock_memory.search = AsyncMock(return_value=[mock_result])

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
        )

        context_str, count = await loop._gather_context(
            query="What do I like?",
            context=agent_context,
        )

        assert "[fact] User likes pizza" in context_str
        assert count == 1

    @pytest.mark.anyio
    async def test_gather_context_no_results(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test _gather_context with no memory results."""
        mock_memory.search.return_value = []

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
        )

        context_str, count = await loop._gather_context(
            query="Unknown topic",
            context=agent_context,
        )

        assert context_str == "No relevant memories found."
        assert count == 0

    @pytest.mark.anyio
    async def test_gather_context_exception_handling(
        self,
        mock_llm,
        mock_decision_engine,
        agent_context,
    ):
        """Test _gather_context handles exceptions."""
        mock_memory = MagicMock()
        mock_memory.search = AsyncMock(side_effect=Exception("Memory error"))

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
        )

        context_str, count = await loop._gather_context(
            query="Test",
            context=agent_context,
        )

        assert "Error gathering context" in context_str
        assert count == 0


class TestAgentLoopFormatHistory:
    """Tests for _format_history method."""

    def test_format_history_empty(self, mock_llm, mock_decision_engine):
        """Test _format_history with empty history."""
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        result = loop._format_history([])

        assert result == "No previous conversation."

    def test_format_history_with_turns(self, mock_llm, mock_decision_engine):
        """Test _format_history formats conversation turns."""
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        history = [
            {"user": "Hello", "assistant": "Hi there!"},
            {"user": "What time is it?", "assistant": "It's 3 PM."},
        ]

        result = loop._format_history(history)

        assert "User: Hello" in result
        assert "Assistant: Hi there!" in result
        assert "User: What time is it?" in result
        assert "Assistant: It's 3 PM." in result

    def test_format_history_truncates_to_five(self, mock_llm, mock_decision_engine):
        """Test _format_history only uses last 5 turns."""
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        history = [
            {"user": f"Message {i}", "assistant": f"Response {i}"}
            for i in range(10)
        ]

        result = loop._format_history(history)

        # Should only have last 5 (indices 5-9)
        assert "Message 4" not in result
        assert "Message 5" in result
        assert "Message 9" in result


class TestAgentLoopSynthesizeResponse:
    """Tests for _synthesize_response method."""

    @pytest.mark.anyio
    async def test_synthesize_no_prompts(
        self,
        mock_llm,
        mock_decision_engine,
        agent_context,
    ):
        """Test synthesis with behavior that has no prompts."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[Action(name="search", description="Search")],
            prompts=None,  # No prompts
        )

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        result = ActionResult(
            action_name="search",
            success=True,
            result={"data": "test"},
            formatted_result="Found: test data",
        )

        response = await loop._synthesize_response(
            query="Search for test",
            behavior=behavior,
            tool_results=[result],
            assistant_intro="I am helpful",
            context=agent_context,
        )

        assert response == "Found: test data"

    @pytest.mark.anyio
    async def test_synthesize_no_results(
        self,
        mock_llm,
        mock_decision_engine,
        agent_context,
    ):
        """Test synthesis with no tool results and no prompts."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[],
            prompts=None,
        )

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        response = await loop._synthesize_response(
            query="Do something",
            behavior=behavior,
            tool_results=[],
            assistant_intro="I am helpful",
            context=agent_context,
        )

        assert response == "I completed your request."

    @pytest.mark.anyio
    async def test_synthesize_with_prompts_and_llm(
        self,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test synthesis uses LLM when prompts available."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Synthesized response from LLM"
        mock_llm.chat = AsyncMock(return_value=mock_response)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
        )

        result = ActionResult(
            action_name="get_time",
            success=True,
            result={"time": "3:00 PM"},
            formatted_result="The time is 3:00 PM",
        )

        response = await loop._synthesize_response(
            query="What time is it?",
            behavior=simple_behavior,
            tool_results=[result],
            assistant_intro="I am Roxy",
            context=agent_context,
        )

        assert response == "Synthesized response from LLM"
        mock_llm.chat.assert_called_once()


class TestAgentLoopAutoMode:
    """Tests for AUTO mode complexity detection."""

    def test_detect_complexity_simple_query(self, mock_llm, mock_decision_engine):
        """Test simple queries get SIMPLE mode."""
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(mode=LoopMode.AUTO),
        )

        mode = loop._detect_complexity("What time is it?")

        assert mode == LoopMode.SIMPLE

    def test_detect_complexity_complex_query(self, mock_llm, mock_decision_engine):
        """Test complex queries get REACT mode."""
        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(
                mode=LoopMode.AUTO,
                complexity_keywords=["compare", "search", "then"],
            ),
        )

        mode = loop._detect_complexity("Search for events and then compare prices")

        assert mode == LoopMode.REACT


class TestAgentLoopSimpleModeEdgeCases:
    """Tests for edge cases in simple mode."""

    @pytest.mark.anyio
    async def test_simple_mode_memory_update_with_action(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test memory update happens after action execution."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="get_time",
            args={},
            reasoning="Getting time",
            memory_update={
                "content": "User prefers 12h time",
                "type": "preference",
                "confidence": 0.8,
            },
        )
        mock_action_executor.execute.return_value = ActionResult(
            action_name="get_time",
            success=True,
            result={"time": "3:00 PM"},
            direct_answer="It's 3:00 PM",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert "User prefers 12h time" in response.memories_stored
        mock_memory.store.assert_called_once()

    @pytest.mark.anyio
    async def test_simple_mode_no_executor(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode without action executor uses decision answer."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="get_time",
            args={},
            answer="The time is approximately 3 PM",
            reasoning="Answering time query",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=None,  # No executor
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert response.response == "The time is approximately 3 PM"

    @pytest.mark.anyio
    async def test_simple_mode_additional_actions(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode executes additional actions."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="get_time",
            args={},
            additional_actions=["search"],
            reasoning="Getting time and searching",
        )
        mock_action_executor.execute.return_value = ActionResult(
            action_name="get_time",
            success=True,
            result={"time": "3:00 PM"},
            direct_answer="It's 3:00 PM",
        )

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        # Should have called execute twice (main + additional)
        assert mock_action_executor.execute.call_count == 2

    @pytest.mark.anyio
    async def test_simple_mode_exception_handling(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode handles exceptions gracefully."""
        mock_decision_engine.decide.side_effect = Exception("Decision failed")

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What time is it?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is False
        assert "error" in response.response.lower()

    @pytest.mark.anyio
    async def test_simple_mode_action_needs_synthesis(
        self,
        mock_llm,
        mock_memory,
        mock_decision_engine,
        mock_action_executor,
        simple_behavior,
        agent_context,
    ):
        """Test simple mode synthesizes when no direct answer."""
        mock_decision_engine.decide.return_value = DecisionResult(
            action="search",
            args={"query": "weather"},
            reasoning="Searching weather",
        )
        mock_action_executor.execute.return_value = ActionResult(
            action_name="search",
            success=True,
            result={"data": "Sunny, 72F"},
            direct_answer=None,  # No direct answer - needs synthesis
        )
        mock_response = MagicMock()
        mock_response.content = "It's a beautiful sunny day!"
        mock_llm.chat = AsyncMock(return_value=mock_response)

        loop = AgentLoop(
            llm=mock_llm,
            memory=mock_memory,
            decision_engine=mock_decision_engine,
            action_executor=mock_action_executor,
            config=AgentLoopConfig(mode=LoopMode.SIMPLE),
        )

        response = await loop.process(
            query="What's the weather?",
            behavior=simple_behavior,
            context=agent_context,
        )

        assert response.success is True
        assert "sunny" in response.response.lower()
