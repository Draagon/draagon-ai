"""
Multi-agent coordination integration tests.

Demonstrates testing shared working memory and agent coordination
using the FR-009 testing framework.

These tests verify:
- Shared working memory observation flow
- Role-based context filtering
- Attention weighting and decay
- Belief candidate identification

Example:
    pytest tests/integration/test_multi_agent.py -v
"""

from __future__ import annotations

import pytest

from draagon_ai.testing import (
    SeedFactory,
    SeedItem,
    SeedSet,
    AgentEvaluator,
    AppProfile,
    ToolSet,
    RESEARCHER_PROFILE,
    ASSISTANT_PROFILE,
)


# =============================================================================
# Shared Working Memory Tests (Using Mock)
# =============================================================================


class MockSharedWorkingMemory:
    """Mock shared working memory for testing coordination patterns.

    Simulates the SharedWorkingMemory interface for tests that
    don't need the full orchestration infrastructure.
    """

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.observations: list[dict] = []

    async def add_observation(
        self,
        content: str,
        source_agent_id: str,
        attention_weight: float = 0.5,
        is_belief_candidate: bool = False,
        belief_type: str | None = None,
    ) -> dict:
        """Add an observation to shared memory."""
        obs = {
            "content": content,
            "source_agent_id": source_agent_id,
            "attention_weight": attention_weight,
            "is_belief_candidate": is_belief_candidate,
            "belief_type": belief_type,
        }
        self.observations.append(obs)
        return obs

    async def get_context_for_agent(
        self,
        agent_id: str,
        role: str = "EXECUTOR",
        max_items: int = 7,
    ) -> list[dict]:
        """Get filtered context for an agent based on role."""
        if role == "CRITIC":
            # Critics only see belief candidates
            return [o for o in self.observations if o["is_belief_candidate"]][:max_items]
        elif role == "EXECUTOR":
            # Executors see SKILL and FACT types
            return [
                o for o in self.observations
                if o["belief_type"] in ("SKILL", "FACT", None)
            ][:max_items]
        else:
            # Researchers see all
            return self.observations[:max_items]

    async def apply_attention_decay(self, decay_factor: float = 0.9):
        """Apply attention decay to all observations."""
        for obs in self.observations:
            obs["attention_weight"] *= decay_factor

    async def get_belief_candidates(self) -> list[dict]:
        """Get observations marked as belief candidates."""
        return [o for o in self.observations if o["is_belief_candidate"]]


# =============================================================================
# Mock Agent Factory
# =============================================================================


class MockAgent:
    """Mock agent for multi-agent testing."""

    def __init__(self, profile: AppProfile):
        self.profile = profile
        self.name = profile.name

    async def process(self, query: str) -> dict:
        """Process a query and return mock response."""
        return {
            "answer": f"[{self.name}] Response to: {query[:50]}...",
            "agent": self.name,
        }


class MockAgentFactory:
    """Mock agent factory for creating test agents."""

    async def create(self, profile: AppProfile) -> MockAgent:
        """Create a mock agent from profile."""
        return MockAgent(profile)


# =============================================================================
# Multi-Agent Coordination Tests
# =============================================================================


@pytest.mark.integration
class TestMultiAgentCoordination:
    """Test multi-agent coordination patterns."""

    @pytest.fixture
    def shared_memory(self):
        """Create shared working memory for test."""
        return MockSharedWorkingMemory(task_id="test_coordination")

    @pytest.fixture
    def agent_factory(self):
        """Create agent factory for test."""
        return MockAgentFactory()

    @pytest.mark.asyncio
    async def test_observation_sharing(self, shared_memory):
        """Test that observations are shared between agents."""
        # Researcher adds observation
        await shared_memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="researcher",
            attention_weight=0.9,
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Verify observation is stored
        assert len(shared_memory.observations) == 1
        assert shared_memory.observations[0]["content"] == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_role_based_filtering(self, shared_memory):
        """Test that roles see appropriate context."""
        # Add various observations
        await shared_memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="researcher",
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )
        await shared_memory.add_observation(
            content="API endpoint: /users/preferences",
            source_agent_id="researcher",
            is_belief_candidate=False,
            belief_type="FACT",
        )
        await shared_memory.add_observation(
            content="How to update preferences: call PUT /preferences",
            source_agent_id="researcher",
            is_belief_candidate=False,
            belief_type="SKILL",
        )

        # Critic sees only belief candidates
        critic_context = await shared_memory.get_context_for_agent(
            agent_id="critic",
            role="CRITIC",
        )
        assert len(critic_context) == 1
        assert "dark mode" in critic_context[0]["content"]

        # Executor sees SKILL and FACT
        executor_context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role="EXECUTOR",
        )
        assert len(executor_context) == 2

        # Researcher sees all
        researcher_context = await shared_memory.get_context_for_agent(
            agent_id="researcher",
            role="RESEARCHER",
        )
        assert len(researcher_context) == 3

    @pytest.mark.asyncio
    async def test_attention_decay(self, shared_memory):
        """Test attention weight decay over time."""
        await shared_memory.add_observation(
            content="Important observation",
            source_agent_id="agent1",
            attention_weight=1.0,
        )

        # Apply decay
        await shared_memory.apply_attention_decay(decay_factor=0.9)

        assert shared_memory.observations[0]["attention_weight"] == pytest.approx(0.9)

        # Apply again
        await shared_memory.apply_attention_decay(decay_factor=0.9)

        assert shared_memory.observations[0]["attention_weight"] == pytest.approx(0.81)

    @pytest.mark.asyncio
    async def test_belief_candidates(self, shared_memory):
        """Test belief candidate identification."""
        await shared_memory.add_observation(
            content="User has 3 cats",
            source_agent_id="listener",
            is_belief_candidate=True,
            belief_type="FACT",
        )
        await shared_memory.add_observation(
            content="Processing user request",
            source_agent_id="executor",
            is_belief_candidate=False,
        )

        candidates = await shared_memory.get_belief_candidates()

        assert len(candidates) == 1
        assert candidates[0]["content"] == "User has 3 cats"

    @pytest.mark.asyncio
    async def test_max_items_limit(self, shared_memory):
        """Test Miller's Law capacity limit (7±2 items)."""
        # Add 10 observations
        for i in range(10):
            await shared_memory.add_observation(
                content=f"Observation {i}",
                source_agent_id="agent",
            )

        # Request with max_items=7
        context = await shared_memory.get_context_for_agent(
            agent_id="test",
            role="RESEARCHER",
            max_items=7,
        )

        assert len(context) == 7

    @pytest.mark.asyncio
    async def test_agent_creation_with_profiles(self, agent_factory):
        """Test creating agents from profiles."""
        researcher = await agent_factory.create(RESEARCHER_PROFILE)
        assistant = await agent_factory.create(ASSISTANT_PROFILE)

        assert researcher.name == "researcher"
        assert assistant.name == "assistant"

        # Both can process queries
        r_response = await researcher.process("Research this topic")
        a_response = await assistant.process("Help me with this")

        assert "researcher" in r_response["answer"]
        assert "assistant" in a_response["answer"]


# =============================================================================
# Profile Configuration Tests
# =============================================================================


class TestAppProfileConfigurations:
    """Test app profile configurations for multi-agent scenarios."""

    def test_researcher_profile_has_full_tools(self):
        """Researcher profile should have full tool access."""
        assert RESEARCHER_PROFILE.tool_set == ToolSet.FULL
        assert RESEARCHER_PROFILE.llm_model_tier == "advanced"

    def test_assistant_profile_has_basic_tools(self):
        """Assistant profile should have basic tool access."""
        assert ASSISTANT_PROFILE.tool_set == ToolSet.BASIC

    def test_custom_profile_creation(self):
        """Test creating custom agent profile."""
        custom = AppProfile(
            name="coordinator",
            personality="You are a coordination agent managing other agents.",
            tool_set=ToolSet.FULL,
            memory_config={"shared_memory_enabled": True},
            llm_model_tier="advanced",
        )

        assert custom.name == "coordinator"
        assert custom.memory_config["shared_memory_enabled"] is True

    def test_profile_immutability(self):
        """Test that profiles are effectively immutable."""
        from dataclasses import replace

        # Create modified copy
        modified = replace(
            RESEARCHER_PROFILE,
            name="modified_researcher",
        )

        # Original unchanged
        assert RESEARCHER_PROFILE.name == "researcher"
        assert modified.name == "modified_researcher"


# =============================================================================
# Coordination Scenario Tests
# =============================================================================


class TestCoordinationScenarios:
    """Test realistic multi-agent coordination scenarios."""

    @pytest.mark.asyncio
    async def test_research_and_execute_workflow(self):
        """Test researcher gathers info, executor acts on it."""
        shared = MockSharedWorkingMemory(task_id="research_execute")

        # Researcher phase: gather information
        await shared.add_observation(
            content="User's home automation system runs on Home Assistant",
            source_agent_id="researcher",
            is_belief_candidate=True,
            belief_type="FACT",
        )
        await shared.add_observation(
            content="To control lights: use light.turn_on/off service",
            source_agent_id="researcher",
            is_belief_candidate=False,
            belief_type="SKILL",
        )

        # Executor gets context
        exec_context = await shared.get_context_for_agent(
            agent_id="executor",
            role="EXECUTOR",
        )

        # Executor should see the FACT and SKILL
        assert len(exec_context) == 2
        assert any("light.turn_on" in o["content"] for o in exec_context)

    @pytest.mark.asyncio
    async def test_critic_reviews_beliefs(self):
        """Test critic reviews and validates beliefs."""
        shared = MockSharedWorkingMemory(task_id="critic_review")

        # Add observations with varying confidence
        await shared.add_observation(
            content="User has 3 cats",
            source_agent_id="listener",
            is_belief_candidate=True,
            belief_type="FACT",
            attention_weight=0.9,
        )
        await shared.add_observation(
            content="User might have a dog too",
            source_agent_id="listener",
            is_belief_candidate=True,
            belief_type="FACT",
            attention_weight=0.4,
        )

        # Critic sees belief candidates
        critic_context = await shared.get_context_for_agent(
            agent_id="critic",
            role="CRITIC",
        )

        assert len(critic_context) == 2

        # Could filter by attention weight for higher confidence beliefs
        high_confidence = [o for o in critic_context if o["attention_weight"] > 0.5]
        assert len(high_confidence) == 1
        assert "3 cats" in high_confidence[0]["content"]
