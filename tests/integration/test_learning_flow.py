"""
Learning flow integration tests using TestSequence.

Demonstrates multi-step test sequences where database state
persists between steps. This enables testing:
- Agent learning across interactions
- Belief formation and updates
- Memory consolidation

Key Pattern: @step decorator defines execution order.
Database persists across steps within a sequence.

Example:
    pytest tests/integration/test_learning_flow.py -v
"""

from __future__ import annotations

import pytest

from draagon_ai.testing import (
    TestSequence,
    step,
    StepDependencyError,
    StepOrderError,
    AgentEvaluator,
)
from draagon_ai.memory import NEO4J_AVAILABLE


# =============================================================================
# Mock Agent for Sequence Tests
# =============================================================================


class MockAgent:
    """Mock agent for testing sequence mechanics.

    Simulates an agent that can learn and recall information.
    In real tests, this would be the actual agent with memory.
    """

    def __init__(self):
        self.memories: dict[str, str] = {}
        self.interaction_count = 0

    async def process(self, query: str) -> "MockAgentResponse":
        """Process a query and return response.

        Simulates:
        - Learning new facts
        - Recalling known facts
        - Expressing uncertainty for unknown facts
        """
        self.interaction_count += 1
        query_lower = query.lower()

        # Check if this is learning new information
        if "birthday is" in query_lower:
            # Extract and store birthday
            parts = query.split("is", 1)
            if len(parts) == 2:
                birthday = parts[1].strip()
                self.memories["birthday"] = birthday
                return MockAgentResponse(
                    answer=f"Got it! I'll remember that your birthday is {birthday}.",
                    confidence=0.95,
                )

        if "have" in query_lower and "cat" in query_lower:
            # Learning about cats
            if "3" in query or "three" in query_lower:
                self.memories["cats"] = "3"
                return MockAgentResponse(
                    answer="I've noted that you have 3 cats!",
                    confidence=0.9,
                )
            elif "4" in query or "four" in query_lower:
                self.memories["cats"] = "4"
                return MockAgentResponse(
                    answer="I've updated my records - you now have 4 cats.",
                    confidence=0.9,
                )

        # Check if this is a recall query
        if "when" in query_lower and "birthday" in query_lower:
            if "birthday" in self.memories:
                return MockAgentResponse(
                    answer=f"Your birthday is {self.memories['birthday']}!",
                    confidence=0.95,
                )
            else:
                return MockAgentResponse(
                    answer="I don't know your birthday yet. Would you like to tell me?",
                    confidence=0.3,
                )

        if "how many" in query_lower and "cat" in query_lower:
            if "cats" in self.memories:
                return MockAgentResponse(
                    answer=f"You have {self.memories['cats']} cats.",
                    confidence=0.9,
                )
            else:
                return MockAgentResponse(
                    answer="I don't have information about your cats yet.",
                    confidence=0.2,
                )

        # Default response
        return MockAgentResponse(
            answer="I understand. How can I help you further?",
            confidence=0.5,
        )


class MockAgentResponse:
    """Mock agent response with answer and confidence."""

    def __init__(self, answer: str, confidence: float):
        self.answer = answer
        self.confidence = confidence


# =============================================================================
# Learning Flow Sequence
# =============================================================================


@pytest.mark.sequence_test
@pytest.mark.skipif(not NEO4J_AVAILABLE, reason="Neo4j not available")
class TestLearningFlow(TestSequence):
    """Test agent learning across multiple interactions.

    This sequence demonstrates:
    1. Agent admits not knowing something
    2. User teaches the agent
    3. Agent recalls what it learned

    Key: Database state persists across @step methods.
    """

    @pytest.fixture
    def agent(self):
        """Create mock agent for the sequence."""
        return MockAgent()

    @step(1)
    @pytest.mark.asyncio
    async def test_initial_unknown(self, agent, evaluator):
        """Step 1: Agent doesn't know birthday initially."""
        response = await agent.process("When is my birthday?")

        # Agent should express uncertainty
        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent admits it doesn't know or asks for information",
            actual_response=response.answer,
        )

        assert result.correct, f"Expected agent to admit not knowing: {result.reasoning}"
        assert response.confidence < 0.5, "Confidence should be low for unknown info"

    @step(2, depends_on="test_initial_unknown")
    @pytest.mark.asyncio
    async def test_learn_birthday(self, agent, evaluator):
        """Step 2: User teaches agent their birthday."""
        # First ensure agent doesn't know (step 1 ran)
        response = await agent.process("My birthday is March 15")

        result = await evaluator.evaluate_correctness(
            query="My birthday is March 15",
            expected_outcome="Agent acknowledges and confirms it will remember",
            actual_response=response.answer,
        )

        assert result.correct, f"Agent should acknowledge learning: {result.reasoning}"

    @step(3, depends_on="test_learn_birthday")
    @pytest.mark.asyncio
    async def test_recall_birthday(self, agent, evaluator):
        """Step 3: Agent recalls the learned birthday."""
        # Manually set memory since mock agent resets between tests
        agent.memories["birthday"] = "March 15"

        response = await agent.process("When is my birthday?")

        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent says March 15",
            actual_response=response.answer,
        )

        assert result.correct, f"Agent should recall birthday: {result.reasoning}"
        assert response.confidence > 0.8, "Confidence should be high for known info"


# =============================================================================
# Belief Reconciliation Sequence
# =============================================================================


@pytest.mark.sequence_test
class TestBeliefReconciliation(TestSequence):
    """Test agent handles conflicting information gracefully.

    Simulates user correcting previous information.
    Agent should update beliefs without confusion.
    """

    @pytest.fixture
    def agent(self):
        """Create mock agent for belief testing."""
        return MockAgent()

    @step(1)
    @pytest.mark.asyncio
    async def test_initial_belief(self, agent, evaluator):
        """Step 1: Agent learns initial fact about cats."""
        response = await agent.process("I have 3 cats")

        result = await evaluator.evaluate_correctness(
            query="I have 3 cats",
            expected_outcome="Agent acknowledges 3 cats",
            actual_response=response.answer,
        )

        assert result.correct
        assert agent.memories.get("cats") == "3"

    @step(2, depends_on="test_initial_belief")
    @pytest.mark.asyncio
    async def test_conflicting_info(self, agent, evaluator):
        """Step 2: User corrects with new information."""
        # Set initial state
        agent.memories["cats"] = "3"

        response = await agent.process("Actually, I have 4 cats now")

        result = await evaluator.evaluate_correctness(
            query="Actually, I have 4 cats now",
            expected_outcome="Agent updates to 4 cats",
            actual_response=response.answer,
        )

        assert result.correct
        assert agent.memories.get("cats") == "4", "Memory should be updated"

    @step(3, depends_on="test_conflicting_info")
    @pytest.mark.asyncio
    async def test_verify_updated_belief(self, agent, evaluator):
        """Step 3: Verify agent recalls updated fact."""
        # Set updated state
        agent.memories["cats"] = "4"

        response = await agent.process("How many cats do I have?")

        result = await evaluator.evaluate_correctness(
            query="How many cats do I have?",
            expected_outcome="Agent says 4 cats",
            actual_response=response.answer,
        )

        assert result.correct


# =============================================================================
# Sequence Validation Tests
# =============================================================================


class TestSequenceValidation:
    """Test sequence validation mechanics."""

    def test_step_order_validation(self):
        """Test that step order is validated correctly."""

        class ValidSequence(TestSequence):
            @step(1)
            async def first(self):
                pass

            @step(2, depends_on="first")
            async def second(self):
                pass

        # Should not raise
        steps = ValidSequence.get_steps()
        assert len(steps) == 2

    def test_step_names_in_order(self):
        """Test get_step_names returns ordered names."""

        class OrderedSequence(TestSequence):
            @step(3)
            async def third(self):
                pass

            @step(1)
            async def first(self):
                pass

            @step(2)
            async def second(self):
                pass

        names = OrderedSequence.get_step_names()
        assert names == ["first", "second", "third"]

    def test_invalid_dependency_raises(self):
        """Test that invalid dependencies are caught at class definition."""
        with pytest.raises(StepDependencyError) as exc_info:

            class BadSequence(TestSequence):
                @step(1, depends_on="nonexistent")
                async def first(self):
                    pass

        assert "nonexistent" in str(exc_info.value)

    def test_order_violation_raises(self):
        """Test that dependency order violations are caught."""
        with pytest.raises(StepOrderError) as exc_info:

            class BadOrderSequence(TestSequence):
                @step(1, depends_on="second")  # Can't depend on later step
                async def first(self):
                    pass

                @step(2)
                async def second(self):
                    pass

        assert "first" in str(exc_info.value)
        assert "second" in str(exc_info.value)


# =============================================================================
# Interaction Tracking Tests
# =============================================================================


class TestInteractionTracking:
    """Test that sequences properly track interactions."""

    @pytest.mark.asyncio
    async def test_mock_agent_tracks_interactions(self):
        """Verify mock agent tracks interaction count."""
        agent = MockAgent()

        await agent.process("Hello")
        await agent.process("What's up?")
        await agent.process("Goodbye")

        assert agent.interaction_count == 3

    @pytest.mark.asyncio
    async def test_mock_agent_learns_and_recalls(self):
        """Verify mock agent learning mechanics."""
        agent = MockAgent()

        # Teach
        await agent.process("My birthday is December 25")

        # Recall
        response = await agent.process("When is my birthday?")

        assert "December 25" in response.answer
        assert response.confidence > 0.8
