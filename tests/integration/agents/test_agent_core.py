"""Core agent processing integration tests (FR-010.1).

Tests the complete agent processing pipeline:
query -> decision -> action -> response

These tests validate outcomes, not processes. The agent should get
correct results regardless of internal implementation details.

Test Categories:
- Simple queries (direct answer, no tools)
- Tool execution queries
- Confidence-based responses
- Error recovery
- Session persistence

All evaluations use LLM-as-judge for semantic validation.
"""

import os
import pytest

from draagon_ai.behaviors.types import (
    Action,
    ActionParameter,
    Behavior,
    BehaviorConstraints,
    BehaviorPrompts,
    BehaviorStatus,
    BehaviorTier,
)
# Note: AgentContext from orchestration.loop is used (not core.context)
# The orchestration AgentContext has clear_observations() method
from draagon_ai.orchestration.loop import (
    AgentContext,
    AgentLoop,
    AgentLoopConfig,
    LoopMode,
)


# =============================================================================
# Test Fixtures - Behavior and Context
# =============================================================================


def create_test_behavior() -> Behavior:
    """Create a minimal test behavior for agent testing.

    This behavior supports:
    - Direct answers (no tool needed)
    - A simple test tool for validating tool execution
    """
    return Behavior(
        behavior_id="test_assistant",
        name="Test Assistant",
        description="A minimal test behavior for integration testing",
        version="1.0.0",
        tier=BehaviorTier.CORE,
        status=BehaviorStatus.ACTIVE,
        actions=[
            Action(
                name="answer",
                description="Respond directly with an answer",
                parameters={
                    "response": ActionParameter(
                        name="response",
                        description="The response to give",
                        type="string",
                        required=True,
                    ),
                },
                handler="answer",
            ),
            Action(
                name="get_weather",
                description="Get the current weather for a location",
                parameters={
                    "location": ActionParameter(
                        name="location",
                        description="Location to get weather for",
                        type="string",
                        required=False,
                    ),
                },
                handler="get_weather",
            ),
            Action(
                name="calculate",
                description="Perform a calculation",
                parameters={
                    "expression": ActionParameter(
                        name="expression",
                        description="Math expression to evaluate",
                        type="string",
                        required=True,
                    ),
                },
                handler="calculate",
            ),
        ],
        triggers=[],
        prompts=BehaviorPrompts(
            decision_prompt="""You are a test assistant. Given the user's question, decide what action to take.

USER QUESTION: {question}
CONVERSATION HISTORY: {conversation_history}
CONTEXT: {context}

AVAILABLE ACTIONS:
- answer: Respond directly when you know the answer
- get_weather: Get weather for a location
- calculate: Perform mathematical calculations

Respond with XML:
<response>
  <action>action_name</action>
  <reasoning>Why this action</reasoning>
  <answer>Your response (if action=answer)</answer>
  <args>
    <location>value</location>
    <expression>value</expression>
  </args>
  <confidence>0.0-1.0</confidence>
</response>
""",
            synthesis_prompt="""Given the action result, synthesize a response for the user.

ACTION RESULT: {action_result}
USER QUESTION: {question}

Provide a concise, helpful response.
""",
        ),
        constraints=BehaviorConstraints(
            style_guidelines=["Be concise", "Be helpful"],
        ),
        test_cases=[],
    )


def create_test_context(
    user_id: str = "test_user",
    session_id: str = "test_session",
) -> AgentContext:
    """Create a minimal test context for agent testing.

    Uses the orchestration.loop.AgentContext which has:
    - user_id, session_id, conversation_id
    - observations list with clear_observations() method
    - conversation_history
    """
    return AgentContext(
        user_id=user_id,
        session_id=session_id,
        conversation_id=session_id,
    )


@pytest.fixture
def test_behavior():
    """Test behavior fixture."""
    return create_test_behavior()


@pytest.fixture
def test_context():
    """Test context fixture."""
    return create_test_context()


# =============================================================================
# Core Agent Processing Tests
# =============================================================================


@pytest.mark.agent_integration
class TestAgentCoreProcessing:
    """Test core agent query processing."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_query_direct_answer(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent answers simple factual query without tools.

        This tests the most basic agent capability: understanding a question
        and providing a direct answer from its knowledge.
        """
        response = await agent.process(
            query="What is 2+2?",
            behavior=test_behavior,
            context=test_context,
        )

        # Validate response exists
        assert response is not None
        assert response.success
        assert len(response.response) > 0

        # Use LLM-as-judge for semantic evaluation
        result = await evaluator.evaluate_correctness(
            query="What is 2+2?",
            expected_outcome="Agent correctly answers that 2+2 equals 4",
            actual_response=response.response,
        )

        assert result.correct, f"Evaluation failed: {result.reasoning}"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_greeting_response(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent responds appropriately to greetings."""
        response = await agent.process(
            query="Hello!",
            behavior=test_behavior,
            context=test_context,
        )

        assert response is not None
        assert response.success

        # Use evaluate_correctness with expected outcome for custom criteria
        # Allow flexibility - greeting, introduction, or offer to help all acceptable
        result = await evaluator.evaluate_correctness(
            query="Hello!",
            expected_outcome="Agent responds appropriately - can be a greeting, introduction, or offer to help",
            actual_response=response.response,
        )

        assert result.correct, f"Inappropriate greeting response: {result.reasoning}"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_complex_question_with_reasoning(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent handles questions requiring reasoning."""
        response = await agent.process(
            query="If I have 3 apples and give away 1, how many do I have left?",
            behavior=test_behavior,
            context=test_context,
        )

        assert response is not None
        assert response.success

        result = await evaluator.evaluate_correctness(
            query="If I have 3 apples and give away 1, how many do I have?",
            expected_outcome="Agent answers with 2 apples or shows reasoning about subtraction",
            actual_response=response.response,
        )

        assert result.correct, f"Reasoning failed: {result.reasoning}"


@pytest.mark.agent_integration
class TestConfidenceBasedResponses:
    """Test that agent confidence affects response style."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_high_confidence_factual_response(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Factual questions should produce confident responses."""
        response = await agent.process(
            query="What color is the sky on a clear day?",
            behavior=test_behavior,
            context=test_context,
        )

        assert response is not None
        assert response.success

        # Response should be direct and confident about blue
        result = await evaluator.evaluate_correctness(
            query="What color is the sky on a clear day?",
            expected_outcome="Agent states the sky is blue, without hedging",
            actual_response=response.response,
        )

        assert result.correct, f"Response not confident: {result.reasoning}"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_uncertain_topic_shows_hedging(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Questions about unknowable topics should show uncertainty."""
        response = await agent.process(
            query="What will the stock market do tomorrow?",
            behavior=test_behavior,
            context=test_context,
        )

        assert response is not None
        assert response.success

        # Response should express uncertainty or inability to predict
        result = await evaluator.evaluate_correctness(
            query="What will the stock market do tomorrow?",
            expected_outcome="Agent expresses uncertainty or inability to predict future",
            actual_response=response.response,
        )

        assert result.correct, f"Should hedge on predictions: {result.reasoning}"


@pytest.mark.agent_integration
class TestErrorRecovery:
    """Test agent graceful error handling."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_graceful_handling_of_invalid_query(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent handles malformed or nonsensical queries gracefully."""
        response = await agent.process(
            query="asdfjkl;asdfjkl;asdf",
            behavior=test_behavior,
            context=test_context,
        )

        # Should not crash
        assert response is not None

        # Should provide some reasonable response
        assert len(response.response) > 0

        result = await evaluator.evaluate_correctness(
            query="asdfjkl;asdfjkl;asdf",
            expected_outcome="Agent asks for clarification or politely indicates confusion",
            actual_response=response.response,
        )

        assert result.correct, f"Failed to handle invalid query: {result.reasoning}"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_empty_query_handling(
        self, agent, test_behavior, test_context
    ):
        """Agent handles empty queries without crashing."""
        # Empty query should not crash
        try:
            response = await agent.process(
                query="",
                behavior=test_behavior,
                context=test_context,
            )
            # If it returns, it should be a valid response
            assert response is not None
        except ValueError:
            # Raising ValueError for empty input is also acceptable
            pass


@pytest.mark.agent_integration
class TestPerformance:
    """Test agent performance characteristics."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_simple_query_latency(
        self, agent, test_behavior, test_context
    ):
        """Simple queries should complete within performance budget."""
        import time

        start = time.time()
        response = await agent.process(
            query="What is 1+1?",
            behavior=test_behavior,
            context=test_context,
        )
        elapsed = time.time() - start

        assert response is not None
        assert response.success

        # Simple queries should complete in <5s (generous for LLM latency)
        assert elapsed < 5.0, f"Query took {elapsed:.2f}s, expected <5s"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_response_includes_latency_metric(
        self, agent, test_behavior, test_context
    ):
        """Response should include latency metrics."""
        response = await agent.process(
            query="Hello",
            behavior=test_behavior,
            context=test_context,
        )

        assert response is not None
        # Check if latency is tracked
        assert hasattr(response, "latency_ms") or hasattr(response, "debug_info")


# =============================================================================
# Session and Context Tests (Placeholder)
# =============================================================================


@pytest.mark.agent_integration
@pytest.mark.skip(reason="Session persistence requires working memory integration - TASK-011")
class TestSessionPersistence:
    """Test that agent maintains context across queries.

    Note: These tests are skipped until memory integration is complete.
    They will be implemented in TASK-011.
    """

    @pytest.mark.asyncio
    async def test_remembers_user_name(
        self, agent, test_behavior, test_context, evaluator
    ):
        """Agent remembers information shared in conversation."""
        # First query establishes context
        await agent.process(
            query="My name is Alice",
            behavior=test_behavior,
            context=test_context,
        )

        # Second query should recall context
        response = await agent.process(
            query="What's my name?",
            behavior=test_behavior,
            context=test_context,
        )

        result = await evaluator.evaluate_correctness(
            query="What's my name?",
            expected_outcome="Agent correctly recalls the name Alice",
            actual_response=response.response,
        )

        assert result.correct


# =============================================================================
# LLM Tier Selection Tests (Placeholder)
# =============================================================================


@pytest.mark.agent_integration
@pytest.mark.tier_integration
@pytest.mark.skip(reason="Tier selection requires model_tier in response - not yet implemented")
class TestLLMTierSelection:
    """Test that agent selects appropriate LLM tier.

    Note: These tests require AgentResponse to include model_tier field.
    """

    @pytest.mark.asyncio
    async def test_simple_query_uses_fast_tier(
        self, agent, test_behavior, test_context
    ):
        """Simple queries should use fast/local tier."""
        response = await agent.process(
            query="What is 2+2?",
            behavior=test_behavior,
            context=test_context,
        )

        # Would check: assert response.model_tier == "local"
        pass

    @pytest.mark.asyncio
    async def test_complex_reasoning_uses_complex_tier(
        self, agent, test_behavior, test_context
    ):
        """Complex reasoning should use complex tier."""
        response = await agent.process(
            query="Analyze the trade-offs between microservices and monolithic architectures",
            behavior=test_behavior,
            context=test_context,
        )

        # Would check: assert response.model_tier in ["complex", "deep"]
        pass
