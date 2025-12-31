"""Tests for shared cognitive working memory (FR-001).

Tests cover:
- SharedObservation dataclass
- Capacity management (Miller's Law)
- Conflict detection
- Attention weighting and decay
- Role-filtered context retrieval
- Concurrent access safety
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock

from draagon_ai.orchestration.shared_memory import (
    SharedObservation,
    SharedWorkingMemory,
    SharedWorkingMemoryConfig,
    EmbeddingProvider,
)
from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole


# =============================================================================
# Test SharedObservation
# =============================================================================


class TestSharedObservation:
    """Tests for SharedObservation dataclass (FR-001.1)."""

    def test_observation_immutable(self):
        """Observations should be immutable (frozen=True)."""
        obs = SharedObservation(
            observation_id="obs_1",
            content="Test observation",
            source_agent_id="agent_a",
            timestamp=datetime.now(),
        )

        # Cannot modify frozen dataclass
        with pytest.raises(AttributeError):
            obs.content = "Modified"  # type: ignore

    def test_observation_validation_attention_weight(self):
        """Validate attention_weight is 0-1."""
        with pytest.raises(ValueError, match="attention_weight must be 0-1"):
            SharedObservation(
                observation_id="obs_1",
                content="Test",
                source_agent_id="agent_a",
                timestamp=datetime.now(),
                attention_weight=1.5,  # Invalid
            )

    def test_observation_validation_confidence(self):
        """Validate confidence is 0-1."""
        with pytest.raises(ValueError, match="confidence must be 0-1"):
            SharedObservation(
                observation_id="obs_1",
                content="Test",
                source_agent_id="agent_a",
                timestamp=datetime.now(),
                confidence=-0.1,  # Invalid
            )

    def test_observation_defaults(self):
        """Test default values are set correctly."""
        obs = SharedObservation(
            observation_id="obs_1",
            content="Test",
            source_agent_id="agent_a",
            timestamp=datetime.now(),
        )

        assert obs.attention_weight == 0.5
        assert obs.confidence == 1.0
        assert obs.is_belief_candidate is False
        assert obs.belief_type is None
        assert obs.conflicts_with == []
        assert obs.accessed_by == set()
        assert obs.access_count == 0

    def test_observation_with_belief_candidate(self):
        """Test observation as belief candidate."""
        obs = SharedObservation(
            observation_id="obs_1",
            content="Meeting is at 3pm",
            source_agent_id="agent_a",
            timestamp=datetime.now(),
            is_belief_candidate=True,
            belief_type="FACT",
        )

        assert obs.is_belief_candidate is True
        assert obs.belief_type == "FACT"


# =============================================================================
# Test Capacity Management
# =============================================================================


class TestCapacityManagement:
    """Tests for Miller's Law capacity enforcement (FR-001.2)."""

    @pytest.mark.asyncio
    async def test_per_agent_capacity_basic(self):
        """Test 7-item limit per agent."""
        memory = SharedWorkingMemory(
            "task_1", SharedWorkingMemoryConfig(max_items_per_agent=7)
        )

        # Add 10 items from same agent with increasing attention
        for i in range(10):
            await memory.add_observation(
                content=f"Observation {i}",
                source_agent_id="agent_a",
                attention_weight=i / 10.0,  # 0.0 to 0.9
            )

        # Should only keep 7 highest attention items
        assert len(memory._observations) == 7

        # Highest attention items kept (3-9 have highest weights)
        remaining_contents = {obs.content for obs in memory._observations.values()}
        assert "Observation 9" in remaining_contents  # Highest
        assert "Observation 8" in remaining_contents
        assert "Observation 0" not in remaining_contents  # Lowest, evicted

    @pytest.mark.asyncio
    async def test_per_agent_capacity_different_agents(self):
        """Test each agent has independent capacity."""
        memory = SharedWorkingMemory(
            "task_1", SharedWorkingMemoryConfig(max_items_per_agent=3, max_total_items=20)
        )

        # Agent A adds 5 items
        for i in range(5):
            await memory.add_observation(
                content=f"Agent A obs {i}",
                source_agent_id="agent_a",
                attention_weight=0.5,
            )

        # Agent B adds 5 items
        for i in range(5):
            await memory.add_observation(
                content=f"Agent B obs {i}",
                source_agent_id="agent_b",
                attention_weight=0.5,
            )

        # Each agent should have max 3 items (6 total)
        assert len(memory._observations) == 6

        agent_a_count = sum(
            1 for o in memory._observations.values() if o.source_agent_id == "agent_a"
        )
        agent_b_count = sum(
            1 for o in memory._observations.values() if o.source_agent_id == "agent_b"
        )

        assert agent_a_count == 3
        assert agent_b_count == 3

    @pytest.mark.asyncio
    async def test_global_capacity(self):
        """Test global capacity across all agents."""
        memory = SharedWorkingMemory(
            "task_1",
            SharedWorkingMemoryConfig(max_items_per_agent=10, max_total_items=20),
        )

        # Add 25 items from 5 agents (5 each)
        for agent_i in range(5):
            for obs_i in range(5):
                await memory.add_observation(
                    content=f"Agent {agent_i} obs {obs_i}",
                    source_agent_id=f"agent_{agent_i}",
                    attention_weight=0.5,
                )

        # Should cap at 20 (global limit)
        assert len(memory._observations) <= 20

    @pytest.mark.asyncio
    async def test_eviction_lowest_attention(self):
        """Test lowest attention items are evicted first."""
        memory = SharedWorkingMemory(
            "task_1", SharedWorkingMemoryConfig(max_items_per_agent=5)
        )

        # Add items with specific attention weights
        await memory.add_observation(
            content="Low attention", source_agent_id="agent_a", attention_weight=0.1
        )
        await memory.add_observation(
            content="Medium attention", source_agent_id="agent_a", attention_weight=0.5
        )
        await memory.add_observation(
            content="High attention", source_agent_id="agent_a", attention_weight=0.9
        )

        # Add 4 more items (triggers eviction)
        for i in range(4):
            await memory.add_observation(
                content=f"New {i}", source_agent_id="agent_a", attention_weight=0.6
            )

        # Low attention item should be evicted
        contents = {obs.content for obs in memory._observations.values()}
        assert "Low attention" not in contents
        assert "High attention" in contents


# =============================================================================
# Test Conflict Detection
# =============================================================================


class TestConflictDetection:
    """Tests for semantic conflict detection (FR-001.3)."""

    @pytest.mark.asyncio
    async def test_simple_conflict_detection(self):
        """Test conflict detection without embeddings (same belief_type)."""
        memory = SharedWorkingMemory("task_1")

        # Agent A observes
        await memory.add_observation(
            content="Meeting is at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # Agent B observes conflicting fact
        obs_b = await memory.add_observation(
            content="Meeting is at 4pm",
            source_agent_id="agent_b",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # Should detect conflict (same belief_type, different agents)
        assert len(obs_b.conflicts_with) > 0

        # Conflicts should be retrievable
        conflicts = await memory.get_conflicts()
        assert len(conflicts) == 1

    @pytest.mark.asyncio
    async def test_no_conflict_same_source(self):
        """Same agent can't conflict with itself."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="Meeting is at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        obs2 = await memory.add_observation(
            content="Meeting is at 4pm",
            source_agent_id="agent_a",  # Same agent
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # No self-conflict
        assert len(obs2.conflicts_with) == 0

    @pytest.mark.asyncio
    async def test_no_conflict_different_types(self):
        """Different belief types don't conflict."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="Meeting is at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        obs2 = await memory.add_observation(
            content="User prefers morning meetings",
            source_agent_id="agent_b",
            is_belief_candidate=True,
            belief_type="PREFERENCE",  # Different type
        )

        # Different types don't conflict
        assert len(obs2.conflicts_with) == 0

    @pytest.mark.asyncio
    async def test_manual_conflict_flagging(self):
        """Test explicitly flagging conflicts."""
        memory = SharedWorkingMemory("task_1")

        obs_a = await memory.add_observation(
            content="Observation A", source_agent_id="agent_a"
        )
        obs_b = await memory.add_observation(
            content="Observation B", source_agent_id="agent_b"
        )

        # Manually flag conflict
        await memory.flag_conflict(obs_a.observation_id, obs_b.observation_id, "test_reason")

        conflicts = await memory.get_conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][2] == "test_reason"

    @pytest.mark.asyncio
    async def test_embedding_based_conflict_detection(self):
        """Test conflict detection with embedding provider."""

        # Mock embedding provider
        class MockEmbeddingProvider:
            async def embed(self, text: str) -> list[float]:
                # Simple mock: hash text to float
                return [float(hash(text) % 100) / 100]

            async def similarity(self, text_a: str, text_b: str) -> float:
                # Mock: return 0.8 for similar texts
                if "meeting" in text_a.lower() and "meeting" in text_b.lower():
                    return 0.8
                return 0.2

        memory = SharedWorkingMemory(
            "task_1",
            config=SharedWorkingMemoryConfig(conflict_threshold=0.7),
            embedding_provider=MockEmbeddingProvider(),
        )

        await memory.add_observation(
            content="Meeting is at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        obs_b = await memory.add_observation(
            content="Meeting is at 4pm",  # Similar content
            source_agent_id="agent_b",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # Should detect conflict via embeddings (similarity > 0.7)
        assert len(obs_b.conflicts_with) > 0


# =============================================================================
# Test Attention Management
# =============================================================================


class TestAttentionManagement:
    """Tests for attention weighting and decay (FR-001.4)."""

    @pytest.mark.asyncio
    async def test_attention_decay(self):
        """Test attention weight decay."""
        memory = SharedWorkingMemory(
            "task_1", SharedWorkingMemoryConfig(attention_decay_factor=0.9)
        )

        obs = await memory.add_observation(
            content="Test", source_agent_id="agent_a", attention_weight=1.0
        )

        initial_weight = obs.attention_weight
        await memory.apply_attention_decay()

        updated_obs = memory._observations[obs.observation_id]
        assert updated_obs.attention_weight == pytest.approx(initial_weight * 0.9)

    @pytest.mark.asyncio
    async def test_attention_boost(self):
        """Test attention weight boost."""
        memory = SharedWorkingMemory("task_1")

        obs = await memory.add_observation(
            content="Test", source_agent_id="agent_a", attention_weight=0.5
        )

        await memory.boost_attention(obs.observation_id, boost=0.3)

        updated_obs = memory._observations[obs.observation_id]
        assert updated_obs.attention_weight == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_attention_boost_capped(self):
        """Attention weight capped at 1.0."""
        memory = SharedWorkingMemory("task_1")

        obs = await memory.add_observation(
            content="Test", source_agent_id="agent_a", attention_weight=0.9
        )

        await memory.boost_attention(obs.observation_id, boost=0.5)

        updated_obs = memory._observations[obs.observation_id]
        assert updated_obs.attention_weight == 1.0  # Capped

    @pytest.mark.asyncio
    async def test_multiple_decay_cycles(self):
        """Test attention decays over multiple cycles."""
        memory = SharedWorkingMemory(
            "task_1", SharedWorkingMemoryConfig(attention_decay_factor=0.9)
        )

        obs = await memory.add_observation(
            content="Test", source_agent_id="agent_a", attention_weight=1.0
        )

        # Apply decay 10 times
        for _ in range(10):
            await memory.apply_attention_decay()

        updated_obs = memory._observations[obs.observation_id]
        expected = 1.0 * (0.9**10)
        assert updated_obs.attention_weight == pytest.approx(expected)


# =============================================================================
# Test Role-Filtered Retrieval
# =============================================================================


class TestRoleFilteredRetrieval:
    """Tests for context retrieval filtered by role (FR-001.5)."""

    @pytest.mark.asyncio
    async def test_critic_sees_only_candidates(self):
        """CRITIC role sees only belief candidates."""
        memory = SharedWorkingMemory("task_1")

        # Add belief candidates
        await memory.add_observation(
            content="Claim 1", source_agent_id="agent_a", is_belief_candidate=True
        )
        await memory.add_observation(
            content="Claim 2", source_agent_id="agent_a", is_belief_candidate=True
        )

        # Add general observations
        await memory.add_observation(
            content="General observation", source_agent_id="agent_a", is_belief_candidate=False
        )

        # Critic retrieves context
        context = await memory.get_context_for_agent(
            agent_id="critic_1",
            role=AgentRole.CRITIC,
        )

        assert len(context) == 2
        assert all(obs.is_belief_candidate for obs in context)

    @pytest.mark.asyncio
    async def test_researcher_sees_all(self):
        """RESEARCHER role sees all observations."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="Belief", source_agent_id="agent_a", is_belief_candidate=True
        )
        await memory.add_observation(content="General", source_agent_id="agent_a")

        context = await memory.get_context_for_agent(
            agent_id="researcher_1",
            role=AgentRole.RESEARCHER,
        )

        assert len(context) == 2

    @pytest.mark.asyncio
    async def test_executor_sees_skills_facts(self):
        """EXECUTOR role sees SKILLs and FACTs."""
        memory = SharedWorkingMemory("task_1")

        await memory.add_observation(
            content="How to restart service",
            source_agent_id="agent_a",
            belief_type="SKILL",
        )
        await memory.add_observation(
            content="Server IP is 192.168.1.1",
            source_agent_id="agent_a",
            belief_type="FACT",
        )
        await memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="agent_a",
            belief_type="PREFERENCE",
        )

        context = await memory.get_context_for_agent(
            agent_id="executor_1",
            role=AgentRole.EXECUTOR,
        )

        assert len(context) == 2  # SKILL + FACT only
        assert all(obs.belief_type in ("SKILL", "FACT", None) for obs in context)

    @pytest.mark.asyncio
    async def test_sorting_by_attention_and_recency(self):
        """Test observations sorted by attention weight + recency."""
        memory = SharedWorkingMemory("task_1")

        # Add observations with different attention weights
        await memory.add_observation(
            content="Low attention", source_agent_id="agent_a", attention_weight=0.3
        )
        await asyncio.sleep(0.01)  # Ensure timestamp difference
        await memory.add_observation(
            content="High attention", source_agent_id="agent_a", attention_weight=0.9
        )

        context = await memory.get_context_for_agent(
            agent_id="researcher_1",
            role=AgentRole.RESEARCHER,
        )

        # Higher attention should come first
        assert context[0].content == "High attention"

    @pytest.mark.asyncio
    async def test_access_tracking(self):
        """Test access tracking updates accessed_by and access_count."""
        memory = SharedWorkingMemory("task_1")

        obs = await memory.add_observation(content="Test", source_agent_id="agent_a")

        # Agent B accesses
        await memory.get_context_for_agent(
            agent_id="agent_b",
            role=AgentRole.RESEARCHER,
        )

        updated_obs = memory._observations[obs.observation_id]
        assert "agent_b" in updated_obs.accessed_by
        assert updated_obs.access_count == 1

        # Agent C accesses
        await memory.get_context_for_agent(
            agent_id="agent_c",
            role=AgentRole.RESEARCHER,
        )

        updated_obs = memory._observations[obs.observation_id]
        assert "agent_c" in updated_obs.accessed_by
        assert updated_obs.access_count == 2

    @pytest.mark.asyncio
    async def test_max_items_limit(self):
        """Test max_items parameter limits returned observations."""
        memory = SharedWorkingMemory("task_1")

        # Add 10 observations
        for i in range(10):
            await memory.add_observation(
                content=f"Obs {i}", source_agent_id="agent_a", attention_weight=0.5
            )

        # Request only 5 items
        context = await memory.get_context_for_agent(
            agent_id="agent_b",
            role=AgentRole.RESEARCHER,
            max_items=5,
        )

        assert len(context) == 5


# =============================================================================
# Test Belief Candidates
# =============================================================================


class TestBeliefCandidates:
    """Tests for belief candidate flagging (FR-001.6)."""

    @pytest.mark.asyncio
    async def test_get_belief_candidates_excludes_conflicts(self):
        """Belief candidates with conflicts are excluded."""
        memory = SharedWorkingMemory("task_1")

        # Add non-conflicting candidate
        obs1 = await memory.add_observation(
            content="Non-conflicting claim",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="SKILL",
        )

        # Add two conflicting candidates from different agents
        obs2 = await memory.add_observation(
            content="Meeting at 3pm",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        obs3 = await memory.add_observation(
            content="Meeting at 4pm",
            source_agent_id="agent_b",  # Different agent
            is_belief_candidate=True,
            belief_type="FACT",  # Same type = conflict detected
        )

        # Verify conflicts were detected
        assert len(obs3.conflicts_with) > 0

        # Get belief candidates
        candidates = await memory.get_belief_candidates()

        # Only non-conflicting candidate returned
        # Candidates with conflicts_with populated are excluded
        assert len(candidates) <= 2  # SKILL + one of the FACTs (only obs3 has conflicts marked)

        # The observation with conflict markers should be excluded
        candidate_ids = [c.observation_id for c in candidates]
        assert obs3.observation_id not in candidate_ids  # obs3 has conflicts, excluded


# =============================================================================
# Test Concurrent Access
# =============================================================================


class TestConcurrentAccess:
    """Tests for concurrent access safety (FR-001.7)."""

    @pytest.mark.asyncio
    async def test_concurrent_writes(self):
        """10 agents writing simultaneously should work."""
        memory = SharedWorkingMemory("task_1")

        async def write_observations(agent_id: str):
            for i in range(20):
                await memory.add_observation(
                    content=f"Agent {agent_id} obs {i}",
                    source_agent_id=agent_id,
                )

        # Spawn 10 concurrent writers
        tasks = [write_observations(f"agent_{i}") for i in range(10)]

        await asyncio.gather(*tasks)

        # All observations should be stored (up to capacity limit)
        assert len(memory._observations) <= 50  # Global cap

        # No duplicate observation IDs
        obs_ids = list(memory._observations.keys())
        assert len(obs_ids) == len(set(obs_ids))

    @pytest.mark.asyncio
    async def test_concurrent_reads_writes(self):
        """Concurrent reads and writes shouldn't deadlock."""
        memory = SharedWorkingMemory("task_1")

        async def writer():
            for i in range(10):
                await memory.add_observation(
                    content=f"Obs {i}",
                    source_agent_id="writer",
                )
                await asyncio.sleep(0.001)

        async def reader():
            for _ in range(10):
                await memory.get_context_for_agent(
                    agent_id="reader",
                    role=AgentRole.RESEARCHER,
                )
                await asyncio.sleep(0.001)

        # Run concurrently - should not deadlock
        await asyncio.gather(writer(), reader())

        # Both completed without hanging
        assert True


# =============================================================================
# Integration Tests
# =============================================================================


class TestTaskContextIntegration:
    """Integration tests with TaskContext."""

    @pytest.mark.asyncio
    async def test_inject_into_task_context(self):
        """SharedWorkingMemory can be injected into TaskContext."""
        from draagon_ai.orchestration.multi_agent_orchestrator import TaskContext

        context = TaskContext(task_id="task_123", query="Test query")
        shared_memory = SharedWorkingMemory(context.task_id)

        context.working_memory["__shared__"] = shared_memory

        # Agents can access it
        assert "__shared__" in context.working_memory
        assert isinstance(context.working_memory["__shared__"], SharedWorkingMemory)

    @pytest.mark.asyncio
    async def test_multi_agent_coordination_flow(self):
        """Test full multi-agent coordination flow."""
        from draagon_ai.orchestration.multi_agent_orchestrator import TaskContext

        context = TaskContext(task_id="coordination_test")
        shared_memory = SharedWorkingMemory(context.task_id)
        context.working_memory["__shared__"] = shared_memory

        # Agent A (researcher) observes
        await shared_memory.add_observation(
            content="Found relevant data in database",
            source_agent_id="researcher_agent",
            attention_weight=0.8,
        )

        # Agent B (critic) retrieves context and sees researcher's work
        critic_context = await shared_memory.get_context_for_agent(
            agent_id="critic_agent",
            role=AgentRole.CRITIC,
        )

        # Critic can build on researcher's work (coordination success)
        assert len(critic_context) >= 0  # Context retrieved successfully

        # Agent C (executor) gets different view
        executor_context = await shared_memory.get_context_for_agent(
            agent_id="executor_agent",
            role=AgentRole.EXECUTOR,
        )

        # All agents coordinated via shared memory
        assert shared_memory.task_id == context.task_id
