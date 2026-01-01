# TASK-013: Implement Belief Reconciliation Tests (FR-010.4)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P1 (Important - validates cognitive belief system)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-009, TASK-011, TASK-012

## Description

Implement integration tests for belief formation and reconciliation: conflicting observations, credibility weighting, clarification queueing, and multi-user belief scoping.

**Core Principle:** Test that agents reconcile conflicting information into coherent beliefs using credibility weighting and semantic analysis.

## Acceptance Criteria

- [ ] Test conflict detection between observations
- [ ] Test credibility-weighted belief formation
- [ ] Test clarification question queueing
- [ ] Test multi-user observation handling (household vs personal)
- [ ] Test belief confidence calibration
- [ ] All conflict detection uses LLM semantic analysis (not keyword matching)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_beliefs.py`

**Example Tests:**
```python
from draagon_ai.cognition.beliefs import BeliefReconciliationService, BeliefType, ObservationScope

@pytest.mark.belief_integration
class TestBeliefReconciliation:
    """Test belief formation and conflict resolution."""

    @pytest.mark.asyncio
    async def test_reconcile_conflicting_observations(self, agent, belief_service, memory_provider):
        """Agent reconciles contradictory information."""

        # User 1 says 3 cats
        await agent.process("I have 3 cats", user_id="doug")

        # User 2 says 4 cats (different user in same household)
        await agent.process("We have 4 cats", user_id="sarah")

        # Belief service should detect conflict
        beliefs = await belief_service.get_beliefs(
            scope="household",
            belief_type=BeliefType.HOUSEHOLD_FACT
        )

        # Should either:
        # 1. Form tentative belief with low confidence
        # 2. Queue clarification question
        belief = next((b for b in beliefs if "cat" in b.content.lower()), None)
        assert belief is not None
        assert belief.confidence < 0.8 or belief.needs_clarification

    @pytest.mark.asyncio
    async def test_credibility_weighting(self, agent, belief_service):
        """More credible sources get higher weight."""

        # Expert source
        await agent.process(
            "The capital of France is Paris",
            source_credibility=0.95  # Wikipedia, verified source
        )

        # Unreliable source
        await agent.process(
            "The capital of France is Lyon",
            source_credibility=0.3  # Random forum post
        )

        # Belief should favor expert
        belief = await belief_service.get_belief("capital of France")
        assert "Paris" in belief.content
        assert belief.confidence > 0.8

    @pytest.mark.asyncio
    async def test_clarification_queueing(self, agent, curiosity_service):
        """Agent queues questions for unclear situations."""

        # Ambiguous statement
        await agent.process("I might have mentioned my cats before...")

        # Curiosity service should queue clarification
        questions = await curiosity_service.get_pending_questions()

        assert len(questions) > 0
        assert any("cat" in q.content.lower() for q in questions)
        assert any(q.question_type == "CLARIFICATION" for q in questions)

    @pytest.mark.asyncio
    async def test_user_scope_vs_household_scope(self, agent, belief_service):
        """Personal beliefs vs household beliefs scoped correctly."""

        # Doug's personal preference
        await agent.process("I prefer dark mode", user_id="doug", scope=ObservationScope.PERSONAL)

        # Sarah's personal preference
        await agent.process("I prefer light mode", user_id="sarah", scope=ObservationScope.PERSONAL)

        # No conflict - different scopes
        doug_belief = await belief_service.get_belief("doug preference theme")
        sarah_belief = await belief_service.get_belief("sarah preference theme")

        assert "dark" in doug_belief.content.lower()
        assert "light" in sarah_belief.content.lower()
        assert not doug_belief.needs_clarification

    @pytest.mark.asyncio
    async def test_belief_confidence_calibration(self, agent, belief_service):
        """Belief confidence reflects observation agreement."""

        # Multiple agreeing observations
        await agent.process("I have 3 cats", user_id="doug")
        await agent.process("We have 3 cats", user_id="sarah")
        await agent.process("Yes, 3 cats total", user_id="alice")

        # High confidence due to agreement
        belief = await belief_service.get_belief("household cats")
        assert belief.confidence > 0.9

    @pytest.mark.asyncio
    async def test_conflict_with_verified_fact(self, agent, belief_service):
        """Conflicts with verified facts trigger verification check."""

        # Store verified fact
        await belief_service.store_belief(
            content="The Earth is round",
            belief_type=BeliefType.VERIFIED_FACT,
            verified=True,
            confidence=1.0,
        )

        # Conflicting observation
        await agent.process("The Earth is flat", user_id="conspiracy_user")

        # Agent should not update verified fact
        belief = await belief_service.get_belief("Earth shape")
        assert "round" in belief.content.lower()
        assert belief.verified

    @pytest.mark.asyncio
    async def test_gradual_belief_formation(self, agent, belief_service):
        """Beliefs form gradually from multiple observations."""

        # First observation - tentative
        await agent.process("Doug seems to like sci-fi", user_id="observer1")
        belief = await belief_service.get_belief("doug likes sci-fi")
        confidence_1 = belief.confidence if belief else 0.0

        # Second observation - strengthens
        await agent.process("Doug loves sci-fi movies", user_id="observer2")
        belief = await belief_service.get_belief("doug likes sci-fi")
        confidence_2 = belief.confidence

        # Third observation - solidifies
        await agent.process("I really love sci-fi", user_id="doug")
        belief = await belief_service.get_belief("doug likes sci-fi")
        confidence_3 = belief.confidence

        # Confidence should increase with each observation
        assert confidence_2 > confidence_1
        assert confidence_3 > confidence_2

    @pytest.mark.asyncio
    async def test_conflict_detection_semantic_not_regex(self, agent, belief_service):
        """Conflict detection uses LLM semantic analysis, not regex."""

        # Two semantically conflicting statements (but different wording)
        await agent.process("I'm a vegetarian", user_id="doug")
        await agent.process("Doug eats steak regularly", user_id="observer")

        # Belief service should detect semantic conflict
        conflicts = await belief_service.get_conflicts(user_id="doug")

        assert len(conflicts) > 0
        # Should detect vegetarian vs steak-eater conflict
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_reconcile_conflicting_observations` - Conflict detection
- [ ] `test_credibility_weighting` - Source credibility
- [ ] `test_clarification_queueing` - Question generation
- [ ] `test_user_scope_vs_household_scope` - Observation scoping
- [ ] `test_belief_confidence_calibration` - Confidence tracking
- [ ] `test_conflict_with_verified_fact` - Verified fact protection
- [ ] `test_gradual_belief_formation` - Incremental confidence
- [ ] `test_conflict_detection_semantic_not_regex` - LLM-based detection

**Performance Tests:**
- [ ] Conflict detection: <2s per observation
- [ ] Belief formation: <3s for reconciliation
- [ ] Clarification queueing: <1s

**Cognitive Tests:**
- [ ] Semantic conflict detection (no regex patterns)
- [ ] Credibility weighting correlation: >0.7
- [ ] Multi-observation belief formation
- [ ] Scope isolation (personal vs household vs session)

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_beliefs.py` - Belief tests

**Modify:**
- None (uses existing BeliefReconciliationService)

## Pre-Implementation Work

**Verify BeliefReconciliationService API:**

Check that `BeliefReconciliationService` from `src/draagon_ai/cognition/beliefs.py` has these methods:

1. **get_beliefs(scope, belief_type)**
```python
async def get_beliefs(
    self,
    scope: str = "user",
    belief_type: BeliefType = None,
    user_id: str = None,
) -> list[AgentBelief]:
    """Retrieve beliefs for given scope/type."""
```

2. **get_belief(query)**
```python
async def get_belief(self, query: str, user_id: str = None) -> AgentBelief | None:
    """Retrieve specific belief by semantic query."""
```

3. **get_conflicts(user_id)**
```python
async def get_conflicts(self, user_id: str = None) -> list[BeliefConflict]:
    """Get all conflicting observations needing reconciliation."""
```

4. **store_belief()**
```python
async def store_belief(
    self,
    content: str,
    belief_type: BeliefType,
    verified: bool = False,
    confidence: float = 0.7,
) -> str:
    """Store a belief directly (for test setup)."""
```

**Verify CuriosityService API:**

Check that `CuriosityService` has:
```python
async def get_pending_questions(self, user_id: str = None) -> list[CuriosityQuestion]:
    """Get questions queued for user."""
```

**Estimated Effort:** 2-4 hours to verify/add missing methods

## Success Metrics

- ✅ Conflict detection: >85%
- ✅ Credibility weighting correlation: >0.7
- ✅ Clarification queue precision: >80%
- ✅ Scope isolation: 100% (no leakage)
- ✅ Semantic conflict detection: >90%
- ✅ All tests pass with >90% success rate

## Notes

**CRITICAL: Semantic Conflict Detection**

Conflict detection MUST use LLM semantic analysis:

```python
# ❌ WRONG - keyword matching
if "vegetarian" in obs1 and "steak" in obs2:
    conflict = True

# ✅ RIGHT - LLM semantic analysis
conflict = await self.llm.detect_conflict(obs1.content, obs2.content)
```

**Belief Formation Pipeline:**
1. User statement → UserObservation (immutable)
2. Conflict detection (semantic LLM analysis)
3. Credibility weighting (source track record)
4. Reconciliation (form/update AgentBelief)
5. Clarification queueing (if needed)

**Cost Control:**
- Conflict detection: ~1 LLM call per new observation
- Belief formation: ~1-2 LLM calls per reconciliation
- Estimated cost: ~$0.10 for full test suite
