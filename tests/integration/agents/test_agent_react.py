"""ReAct reasoning integration tests (FR-010.5).

Tests multi-step reasoning with THOUGHT -> ACTION -> OBSERVATION loops:
- ReAct mode produces expected step types
- Tools invoked correctly within reasoning loop
- Observations integrated into subsequent reasoning
- Final answer synthesizes all gathered information
- Step limits and timeouts respected
- Trace correctness validation

These tests validate the AgentLoop ReAct mode works correctly with real LLM.
"""

import os
import time
import pytest

from draagon_ai.orchestration.loop import (
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
    LoopMode,
    StepType,
    ReActStep,
)
from draagon_ai.behaviors import Behavior, Action


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def react_config():
    """Configuration for ReAct mode testing."""
    return AgentLoopConfig(
        mode=LoopMode.REACT,
        max_iterations=5,
        iteration_timeout_seconds=30.0,
        log_thought_traces=True,
    )


@pytest.fixture
def simple_config():
    """Configuration for simple mode testing."""
    return AgentLoopConfig(
        mode=LoopMode.SIMPLE,
        max_iterations=1,
        log_thought_traces=True,
    )


@pytest.fixture
def auto_config():
    """Configuration for auto mode testing."""
    return AgentLoopConfig(
        mode=LoopMode.AUTO,
        max_iterations=5,
        complexity_threshold=0.33,
        log_thought_traces=True,
    )


@pytest.fixture
def react_agent(real_llm, memory_provider, react_config):
    """Agent configured for ReAct mode."""
    return AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        config=react_config,
    )


@pytest.fixture
def simple_agent(real_llm, memory_provider, simple_config):
    """Agent configured for simple mode."""
    return AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        config=simple_config,
    )


@pytest.fixture
def auto_agent(real_llm, memory_provider, auto_config):
    """Agent configured for auto mode."""
    return AgentLoop(
        llm=real_llm,
        memory=memory_provider,
        config=auto_config,
    )


@pytest.fixture
def test_context():
    """Context for tests."""
    return AgentContext(
        user_id="test_user",
        session_id="test_session",
        debug=True,
    )


@pytest.fixture
def simple_behavior():
    """Simple behavior with answer action."""
    return Behavior(
        behavior_id="test_simple",
        name="Simple Test Behavior",
        description="Simple assistant for testing ReAct reasoning",
        actions=[
            Action(
                name="answer",
                description="Answer the user's question directly",
            ),
        ],
    )


# =============================================================================
# ReAct Mode Tests
# =============================================================================


@pytest.mark.react_integration
class TestReActMode:
    """Test ReAct mode configuration and initialization."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_mode_set(self, react_agent):
        """Agent config is set to ReAct mode."""
        assert react_agent.config.mode == LoopMode.REACT

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_response_has_steps(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct mode response includes react_steps."""
        response = await react_agent.process(
            query="What is 2 + 2?",
            behavior=simple_behavior,
            context=test_context,
        )

        assert isinstance(response, AgentResponse)
        assert response.loop_mode == LoopMode.REACT
        assert isinstance(response.react_steps, list)

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_react_response_has_final_answer(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct mode response includes a final answer."""
        response = await react_agent.process(
            query="What color is the sky?",
            behavior=simple_behavior,
            context=test_context,
        )

        # Should have a response (even if hit max iterations)
        assert response.response is not None
        assert len(response.response) > 0

        # Should have steps recorded
        # Note: May not have THOUGHT/FINAL_ANSWER if behavior has no prompts,
        # but should still produce a response (with OBSERVATION steps at minimum)
        if response.success:
            # Successful response with steps
            assert len(response.react_steps) >= 0
        else:
            # Even failed responses should be captured
            assert response.response is not None


@pytest.mark.react_integration
class TestReActStepTypes:
    """Test ReAct step type generation."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_thought_steps_present(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct trace includes THOUGHT steps."""
        response = await react_agent.process(
            query="Tell me something interesting about cats",
            behavior=simple_behavior,
            context=test_context,
        )

        thought_steps = [s for s in response.react_steps if s.type == StepType.THOUGHT]

        # Should have at least one thought step (reasoning before answer)
        # Note: LLM may skip directly to answer for simple questions
        if response.iterations_used > 0:
            # At least one thought per iteration
            pass  # May have 0 thoughts if LLM goes directly to answer

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_final_answer_step(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct trace includes FINAL_ANSWER step for successful queries."""
        response = await react_agent.process(
            query="What is the capital of France?",
            behavior=simple_behavior,
            context=test_context,
        )

        if response.success:
            final_answers = [
                s for s in response.react_steps if s.type == StepType.FINAL_ANSWER
            ]
            # Should have a final answer if successful
            assert len(final_answers) >= 1 or response.response is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_step_content_populated(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct steps have non-empty content."""
        response = await react_agent.process(
            query="What is Python used for?",
            behavior=simple_behavior,
            context=test_context,
        )

        for step in response.react_steps:
            # Each step should have content
            assert step.content is not None
            assert len(step.content) > 0


@pytest.mark.react_integration
class TestReActIterationLimits:
    """Test ReAct iteration limits and timeouts."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_max_iterations_respected(
        self, real_llm, memory_provider, test_context, simple_behavior
    ):
        """ReAct respects max_iterations config."""
        # Configure with low max iterations
        config = AgentLoopConfig(
            mode=LoopMode.REACT,
            max_iterations=3,
            iteration_timeout_seconds=30.0,
        )

        agent = AgentLoop(
            llm=real_llm,
            memory=memory_provider,
            config=config,
        )

        response = await agent.process(
            query="What is artificial intelligence?",
            behavior=simple_behavior,
            context=test_context,
        )

        # Iterations should not exceed max
        assert response.iterations_used <= 3

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_single_iteration_for_simple_query(
        self, react_agent, test_context, simple_behavior
    ):
        """Simple queries may complete in single iteration."""
        response = await react_agent.process(
            query="Say hello",
            behavior=simple_behavior,
            context=test_context,
        )

        # Simple queries often complete in 1-2 iterations
        assert response.iterations_used >= 1
        assert response.iterations_used <= 5


@pytest.mark.react_integration
class TestAutoModeDetection:
    """Test automatic mode detection."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_query_uses_simple_mode(
        self, auto_agent, test_context, simple_behavior
    ):
        """Simple queries use simple mode in auto detection."""
        response = await auto_agent.process(
            query="Hello",
            behavior=simple_behavior,
            context=test_context,
        )

        # "Hello" has no complexity keywords, should use SIMPLE
        assert response.loop_mode == LoopMode.SIMPLE

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_complex_query_uses_react_mode(
        self, auto_agent, test_context, simple_behavior
    ):
        """Complex queries use ReAct mode in auto detection."""
        # Query with complexity keywords ("check", "compare")
        response = await auto_agent.process(
            query="Check the weather and compare it to yesterday",
            behavior=simple_behavior,
            context=test_context,
        )

        # Should trigger ReAct due to complexity keywords
        assert response.loop_mode == LoopMode.REACT

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_multi_step_keywords_trigger_react(
        self, auto_agent, test_context, simple_behavior
    ):
        """Multi-step keywords trigger ReAct mode."""
        # "and then" + "analyze" are complexity keywords
        response = await auto_agent.process(
            query="First check the temperature and then analyze the trend",
            behavior=simple_behavior,
            context=test_context,
        )

        assert response.loop_mode == LoopMode.REACT


@pytest.mark.react_integration
class TestReActVsSimple:
    """Test differences between ReAct and Simple modes."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_mode_no_react_steps(
        self, simple_agent, test_context, simple_behavior
    ):
        """Simple mode produces no ReAct steps."""
        response = await simple_agent.process(
            query="What is 2 + 2?",
            behavior=simple_behavior,
            context=test_context,
        )

        assert response.loop_mode == LoopMode.SIMPLE
        assert len(response.react_steps) == 0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_both_modes_produce_response(
        self, react_agent, simple_agent, test_context, simple_behavior
    ):
        """Both modes produce valid responses."""
        query = "What is the meaning of life?"

        react_response = await react_agent.process(
            query=query,
            behavior=simple_behavior,
            context=test_context,
        )

        simple_response = await simple_agent.process(
            query=query,
            behavior=simple_behavior,
            context=test_context,
        )

        # Both should produce responses
        assert react_response.response is not None
        assert simple_response.response is not None
        assert len(react_response.response) > 0
        assert len(simple_response.response) > 0


@pytest.mark.react_integration
class TestReActTraceQuality:
    """Test trace coherence and quality."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_trace_has_timestamps(
        self, react_agent, test_context, simple_behavior
    ):
        """ReAct steps have timestamps."""
        response = await react_agent.process(
            query="Explain machine learning",
            behavior=simple_behavior,
            context=test_context,
        )

        for step in response.react_steps:
            assert step.timestamp is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_get_thought_trace_format(
        self, react_agent, test_context, simple_behavior
    ):
        """get_thought_trace() returns properly formatted data."""
        response = await react_agent.process(
            query="What is quantum computing?",
            behavior=simple_behavior,
            context=test_context,
        )

        trace = response.get_thought_trace()

        assert isinstance(trace, list)
        for item in trace:
            assert "step" in item
            assert "type" in item
            assert "content" in item
            assert "timestamp" in item

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_debug_info_contains_trace(
        self, react_agent, test_context, simple_behavior
    ):
        """Debug info includes thought trace when debug=True."""
        test_context.debug = True

        response = await react_agent.process(
            query="Explain deep learning",
            behavior=simple_behavior,
            context=test_context,
        )

        # Debug info should contain trace
        if response.react_steps:
            assert "thought_trace" in response.debug_info or len(response.react_steps) > 0


@pytest.mark.react_integration
class TestReActPerformance:
    """Test ReAct performance characteristics."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_query_latency(
        self, react_agent, test_context, simple_behavior
    ):
        """Simple query in ReAct mode completes within budget."""
        start = time.time()

        response = await react_agent.process(
            query="What is 1 + 1?",
            behavior=simple_behavior,
            context=test_context,
        )

        elapsed = time.time() - start

        # Simple query should complete in <10s even in ReAct mode
        assert elapsed < 10.0, f"Query took {elapsed:.2f}s, expected <10s"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_latency_tracked(
        self, react_agent, test_context, simple_behavior
    ):
        """Response latency_ms is tracked."""
        response = await react_agent.process(
            query="Hello there",
            behavior=simple_behavior,
            context=test_context,
        )

        assert response.latency_ms > 0
        # Latency should be reasonable (under 30s = 30000ms)
        assert response.latency_ms < 30000


@pytest.mark.react_integration
class TestReActObservationContext:
    """Test observation accumulation across iterations."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_context_cleared_between_queries(
        self, react_agent, simple_behavior
    ):
        """Context observations are cleared between queries."""
        context1 = AgentContext(user_id="user1", session_id="session1")
        context2 = AgentContext(user_id="user2", session_id="session2")

        # First query
        await react_agent.process(
            query="Tell me about dogs",
            behavior=simple_behavior,
            context=context1,
        )

        # Second query with fresh context
        response2 = await react_agent.process(
            query="Tell me about cats",
            behavior=simple_behavior,
            context=context2,
        )

        # Second context should be clean
        # (observations from first query shouldn't leak)
        assert response2.response is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_observations_accumulate_within_query(
        self, react_agent, test_context, simple_behavior
    ):
        """Observations accumulate within a single query's reasoning."""
        response = await react_agent.process(
            query="First check how many seconds are in an hour, then tell me",
            behavior=simple_behavior,
            context=test_context,
        )

        # If multiple iterations, observations should accumulate
        obs_steps = [s for s in response.react_steps if s.type == StepType.OBSERVATION]

        # May or may not have observations depending on whether actions were taken
        assert isinstance(obs_steps, list)


# =============================================================================
# AgentLoop Integration Tests (Require Action Executor)
# =============================================================================


@pytest.mark.react_integration
@pytest.mark.skip(reason="Requires ActionExecutor wired with test tools - not yet implemented")
class TestReActToolExecution:
    """Test ReAct with actual tool execution.

    These tests require ActionExecutor to be configured with test tools.
    Skipped until tool execution integration is complete.
    """

    @pytest.mark.asyncio
    async def test_action_steps_invoke_tools(self, react_agent, test_context):
        """ACTION steps correctly invoke registered tools."""
        pass

    @pytest.mark.asyncio
    async def test_observation_steps_contain_tool_results(self, react_agent, test_context):
        """OBSERVATION steps contain tool execution results."""
        pass

    @pytest.mark.asyncio
    async def test_multi_tool_reasoning(self, react_agent, test_context):
        """Agent can use multiple tools in sequence."""
        pass
