# TASK-016: Implement Multi-Agent Coordination Tests (FR-010.7)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P2 (Nice-to-have - validates multi-agent features)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-009, TASK-010

## Description

Implement integration tests for multi-agent coordination using SharedWorkingMemory: observation sharing, role-based filtering, attention weighting/decay, belief candidate identification, and concurrent access safety.

**Core Principle:** Test that multiple agents can coordinate effectively through shared cognitive working memory.

## Acceptance Criteria

- [ ] Test observation sharing between agents
- [ ] Test role-based context filtering (CRITIC, RESEARCHER, EXECUTOR)
- [ ] Test attention weighting and decay
- [ ] Test belief candidate identification
- [ ] Test concurrent access safety (no data loss)
- [ ] Test Miller's Law capacity limits (7±2 items per agent)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_multiagent.py`

**Example Tests:**
```python
from draagon_ai.orchestration.shared_memory import (
    SharedWorkingMemory,
    SharedObservation,
    AgentRole,
)

@pytest.mark.multiagent_integration
class TestMultiAgentCoordination:
    """Test multi-agent coordination with shared memory."""

    @pytest.mark.asyncio
    async def test_shared_memory_observation_flow(self, agent_factory):
        """Agents share observations via shared memory."""

        researcher = await agent_factory.create(RESEARCHER_PROFILE)
        executor = await agent_factory.create(EXECUTOR_PROFILE)

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Researcher gathers info
        await shared_memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="researcher",
            attention_weight=0.9,
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Executor retrieves context
        context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
        )

        # Executor should see the observation
        assert len(context) > 0
        assert any("dark mode" in obs.content for obs in context)

    @pytest.mark.asyncio
    async def test_role_based_filtering(self):
        """Different roles see different context."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Add various observation types
        await shared_memory.add_observation(
            content="User likes sci-fi",
            belief_type="PREFERENCE",
            is_belief_candidate=True,
        )
        await shared_memory.add_observation(
            content="API endpoint: /preferences",
            belief_type="FACT",
            is_belief_candidate=False,
        )
        await shared_memory.add_observation(
            content="How to update: PUT /preferences",
            belief_type="SKILL",
            is_belief_candidate=False,
        )

        # CRITIC sees only belief candidates
        critic_context = await shared_memory.get_context_for_agent(
            agent_id="critic",
            role=AgentRole.CRITIC,
        )
        assert len(critic_context) == 1
        assert critic_context[0].is_belief_candidate

        # EXECUTOR sees FACT and SKILL
        executor_context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
        )
        assert len(executor_context) == 2
        assert all(not obs.is_belief_candidate for obs in executor_context)

        # RESEARCHER sees all
        researcher_context = await shared_memory.get_context_for_agent(
            agent_id="researcher",
            role=AgentRole.RESEARCHER,
        )
        assert len(researcher_context) == 3

    @pytest.mark.asyncio
    async def test_attention_decay(self):
        """Attention weights decay over time."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        obs_id = await shared_memory.add_observation(
            content="Important info",
            attention_weight=1.0,
        )

        # Apply decay (×0.9 per decay cycle)
        await shared_memory.apply_attention_decay()

        obs = await shared_memory.get_observation(obs_id)
        assert obs.attention_weight == 0.9  # Decayed by ×0.9

        # Apply again
        await shared_memory.apply_attention_decay()
        obs = await shared_memory.get_observation(obs_id)
        assert abs(obs.attention_weight - 0.81) < 0.01  # 0.9 × 0.9

    @pytest.mark.asyncio
    async def test_attention_boost(self):
        """Referenced observations get attention boost."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        obs_id = await shared_memory.add_observation(
            content="Useful info",
            attention_weight=0.5,
        )

        # Boost attention (e.g., observation was used)
        await shared_memory.boost_attention(obs_id, boost=0.2)

        obs = await shared_memory.get_observation(obs_id)
        assert obs.attention_weight == 0.7  # 0.5 + 0.2

        # Capped at 1.0
        await shared_memory.boost_attention(obs_id, boost=0.5)
        obs = await shared_memory.get_observation(obs_id)
        assert obs.attention_weight == 1.0  # Capped

    @pytest.mark.asyncio
    async def test_millers_law_capacity_limit(self):
        """Working memory respects Miller's Law (7±2 items per agent)."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Add many observations
        for i in range(20):
            await shared_memory.add_observation(
                content=f"Observation {i}",
                attention_weight=0.5 + (i * 0.01),  # Varying attention
            )

        # Get context for agent
        context = await shared_memory.get_context_for_agent(
            agent_id="agent_1",
            role=AgentRole.RESEARCHER,
            max_items=7,  # Miller's Law
        )

        # Should return only top 7 by attention
        assert len(context) <= 7

    @pytest.mark.asyncio
    async def test_concurrent_access_safety(self):
        """Shared memory handles concurrent access safely."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        async def add_observation(i):
            await shared_memory.add_observation(
                content=f"Observation {i}",
                source_agent_id=f"agent_{i}",
            )

        # Concurrent additions (100 agents adding simultaneously)
        await asyncio.gather(*[add_observation(i) for i in range(100)])

        # All observations should be stored (no data loss)
        observations = await shared_memory.get_all_observations()
        assert len(observations) == 100

    @pytest.mark.asyncio
    async def test_belief_candidate_identification(self):
        """System identifies observations ready for belief formation."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Add observations
        await shared_memory.add_observation(
            content="User prefers dark mode",
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )
        await shared_memory.add_observation(
            content="API call succeeded",
            is_belief_candidate=False,  # Not a belief
        )

        # Get belief candidates
        candidates = await shared_memory.get_belief_candidates()

        assert len(candidates) == 1
        assert candidates[0].content == "User prefers dark mode"

    @pytest.mark.asyncio
    async def test_conflict_detection(self):
        """System detects conflicting observations."""

        shared_memory = SharedWorkingMemory(task_id="test_task")

        # Add conflicting observations
        await shared_memory.add_observation(
            content="User has 3 cats",
            is_belief_candidate=True,
            belief_type="FACT",
        )
        await shared_memory.add_observation(
            content="User has 4 cats",
            is_belief_candidate=True,
            belief_type="FACT",
        )

        # Get conflicts
        conflicts = await shared_memory.get_conflicts()

        assert len(conflicts) > 0
        # Should detect conflict between "3 cats" and "4 cats"

    @pytest.mark.asyncio
    async def test_multi_agent_task_coordination(self, agent_factory):
        """Multiple agents coordinate on complex task."""

        # Create agents with different roles
        researcher = await agent_factory.create(AppProfile(
            name="researcher",
            personality="Research and gather information",
        ))
        critic = await agent_factory.create(AppProfile(
            name="critic",
            personality="Evaluate and critique",
        ))
        executor = await agent_factory.create(AppProfile(
            name="executor",
            personality="Execute actions",
        ))

        shared_memory = SharedWorkingMemory(task_id="complex_task")

        # Researcher gathers info
        researcher_response = await researcher.process(
            "Research user preferences for theme"
        )
        await shared_memory.add_observation(
            content="User prefers dark mode based on past interactions",
            source_agent_id="researcher",
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Critic evaluates
        critic_context = await shared_memory.get_context_for_agent(
            agent_id="critic",
            role=AgentRole.CRITIC,
        )
        assert len(critic_context) > 0

        critic_response = await critic.process(
            f"Evaluate this observation: {critic_context[0].content}"
        )
        await shared_memory.add_observation(
            content="Observation seems reliable, confidence 0.8",
            source_agent_id="critic",
            is_belief_candidate=False,
        )

        # Executor acts on verified belief
        executor_context = await shared_memory.get_context_for_agent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
        )

        executor_response = await executor.process(
            "Set user theme to dark mode"
        )

        # Task completed successfully with multi-agent coordination
        assert "dark" in executor_response.answer.lower() or "theme" in executor_response.answer.lower()
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_shared_memory_observation_flow` - Observation sharing
- [ ] `test_role_based_filtering` - Role filtering
- [ ] `test_attention_decay` - Attention decay
- [ ] `test_attention_boost` - Attention boosting
- [ ] `test_millers_law_capacity_limit` - Capacity limits
- [ ] `test_concurrent_access_safety` - Concurrent safety
- [ ] `test_belief_candidate_identification` - Candidate identification
- [ ] `test_conflict_detection` - Conflict detection
- [ ] `test_multi_agent_task_coordination` - Full coordination

**Performance Tests:**
- [ ] Observation add: <50ms
- [ ] Context retrieval: <100ms
- [ ] Concurrent access: No data loss at 100 threads

**Cognitive Tests:**
- [ ] Miller's Law: 7±2 items per agent
- [ ] Attention decay: ×0.9 per cycle
- [ ] Attention boost: +0.2, capped at 1.0
- [ ] Role filtering: 100% accuracy

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_multiagent.py` - Multi-agent tests

**Modify:**
- `tests/integration/agents/conftest.py` - Add multi-agent profiles

## Pre-Implementation Work

**Verify SharedWorkingMemory API:**

From the codebase review, `SharedWorkingMemory` exists with:
- `add_observation()`, `get_context_for_agent()`
- `apply_attention_decay()`, `boost_attention()`
- `get_conflicts()`, `get_belief_candidates()`
- Role-based filtering (CRITIC, RESEARCHER, EXECUTOR)
- Concurrent access safety (asyncio.Lock)

**Verify these methods exist:**
1. `shared_memory.get_observation(id)` - Retrieve observation by ID
2. `shared_memory.get_all_observations()` - Get all observations
3. `shared_memory.boost_attention(id, boost)` - Boost specific observation

**Create AppProfiles for multi-agent tests:**
```python
# In conftest.py
RESEARCHER_PROFILE = AppProfile(
    name="researcher",
    personality="Research and gather information",
    tool_set=ToolSet.FULL,
)

CRITIC_PROFILE = AppProfile(
    name="critic",
    personality="Evaluate and critique observations",
    tool_set=ToolSet.MINIMAL,
)

EXECUTOR_PROFILE = AppProfile(
    name="executor",
    personality="Execute actions based on verified beliefs",
    tool_set=ToolSet.BASIC,
)
```

**Estimated Effort:** 2 hours to verify API and create profiles

## Success Metrics

- ✅ Observation propagation: 100%
- ✅ Role filtering accuracy: 100%
- ✅ Concurrent access: No data loss
- ✅ Attention decay precision: ±0.01
- ✅ Miller's Law compliance: 7±2 items
- ✅ Conflict detection: >85%
- ✅ All tests pass with >85% success rate

## Notes

**SharedWorkingMemory Design (from FR-001):**
- Based on Baddeley's Working Memory Model
- Miller's Law: 7±2 item capacity per agent
- Attention-weighted access (decay ×0.9, boost +0.2)
- Role-based filtering for specialization
- Conflict detection (heuristic + optional embeddings)

**Role Filtering Logic:**
| Role | Sees |
|------|------|
| CRITIC | Only belief candidates (for evaluation) |
| RESEARCHER | All observations |
| EXECUTOR | Only SKILL and FACT types (for action) |

**Cost Control:**
- Multi-agent tests use database operations (free)
- Some LLM calls for agent processing
- Estimated cost: ~$0.15 for full test suite
