# TASK-008: Implement Example Integration Tests

**Phase**: 5 (Integration Tests)
**Priority**: P1 (Validates framework works end-to-end)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-001, TASK-002, TASK-003, TASK-006, TASK-007

## Description

Implement example integration tests that demonstrate the testing framework in action. These serve as both validation and documentation for how to write integration tests.

## Acceptance Criteria

- [ ] Memory integration test (seed + recall)
- [ ] Learning integration test (teach + verify)
- [ ] Multi-agent coordination test
- [ ] Belief reconciliation test (sequence)
- [ ] All tests use LLM-as-judge evaluation
- [ ] All tests use seed data (not inline setup)
- [ ] Tests pass with real LLM and Neo4j
- [ ] README with running instructions

## Technical Notes

**Example Tests to Implement:**

1. **Memory Recall Test** - Verify agent recalls seeded facts
2. **Learning Flow Test** - Multi-step learning sequence
3. **Belief Reconciliation Test** - Handle conflicting observations
4. **Multi-Agent Coordination Test** - Shared working memory

**Critical Testing Principles:**

- Use LLM-as-judge for semantic evaluation (NOT string matching)
- Use seed data for setup (declarative, reusable)
- Test outcomes, not processes
- Use real providers (no mocks for integration tests)

## Testing Requirements

### Memory Integration Test (`tests/integration/test_memory_integration.py`)

```python
import pytest
from draagon_ai.testing import SeedSet, SeedFactory, SeedItem

@SeedFactory.register("user_doug")
class DougUserSeed(SeedItem):
    """Doug's user profile."""
    async def create(self, provider):
        return await provider.store(
            content="User profile: Doug, 3 cats (Whiskers, Mittens, Shadow)",
            metadata={"memory_type": "USER_PROFILE", "user_name": "Doug"}
        )

USER_WITH_CATS = SeedSet("user_with_cats", ["user_doug"])

@pytest.mark.memory_integration
async def test_recall_cat_names(agent, memory_provider, seed, evaluator):
    """Test agent recalls cat names from memory."""
    # Apply seeds
    await seed.apply(USER_WITH_CATS, memory_provider)

    # Query agent
    response = await agent.process("What are my cats' names?")

    # LLM-as-judge evaluation
    result = await evaluator.evaluate_correctness(
        query="What are my cats' names?",
        expected_outcome="Agent lists: Whiskers, Mittens, Shadow",
        actual_response=response.answer
    )

    assert result.correct, f"Failed: {result.reasoning}"
    assert result.confidence > 0.8
```

### Learning Flow Test (`tests/integration/test_learning_flow.py`)

```python
from draagon_ai.testing import TestSequence, step

class TestLearningFlow(TestSequence):
    """Test agent learning across interactions."""

    @step(1)
    async def test_initial_unknown(self, agent, evaluator):
        """Agent doesn't know birthday initially."""
        response = await agent.process("When is my birthday?")

        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent admits it doesn't know",
            actual_response=response.answer
        )

        assert result.correct
        assert response.confidence < 0.5

    @step(2, depends_on="test_initial_unknown")
    async def test_learn_birthday(self, agent, evaluator):
        """Agent learns birthday from user."""
        response = await agent.process("My birthday is March 15")

        result = await evaluator.evaluate_correctness(
            query="My birthday is March 15",
            expected_outcome="Agent acknowledges and stores the birthday",
            actual_response=response.answer
        )

        assert result.correct

    @step(3, depends_on="test_learn_birthday")
    async def test_recall_birthday(self, agent, evaluator):
        """Agent recalls learned birthday."""
        response = await agent.process("When is my birthday?")

        result = await evaluator.evaluate_correctness(
            query="When is my birthday?",
            expected_outcome="Agent says March 15",
            actual_response=response.answer
        )

        assert result.correct
        assert response.confidence > 0.8
```

### Belief Reconciliation Test (`tests/integration/test_belief_reconciliation.py`)

```python
from draagon_ai.testing import TestSequence, step

class TestBeliefReconciliation(TestSequence):
    """Test agent handles conflicting information."""

    @step(1)
    async def test_initial_belief(self, agent, evaluator):
        """Agent learns initial fact."""
        response = await agent.process("I have 3 cats")

        result = await evaluator.evaluate_correctness(
            query="I have 3 cats",
            expected_outcome="Agent acknowledges 3 cats",
            actual_response=response.answer
        )

        assert result.correct

    @step(2, depends_on="test_initial_belief")
    async def test_conflicting_info(self, agent, evaluator):
        """Agent receives conflicting information."""
        response = await agent.process("Actually, I have 4 cats now")

        result = await evaluator.evaluate_correctness(
            query="Actually, I have 4 cats now",
            expected_outcome="Agent updates belief to 4 cats",
            actual_response=response.answer
        )

        assert result.correct

    @step(3, depends_on="test_conflicting_info")
    async def test_verify_updated_belief(self, agent, evaluator):
        """Agent recalls updated fact."""
        response = await agent.process("How many cats do I have?")

        result = await evaluator.evaluate_correctness(
            query="How many cats do I have?",
            expected_outcome="Agent says 4 cats",
            actual_response=response.answer
        )

        assert result.correct
```

### Multi-Agent Coordination Test (`tests/integration/test_multi_agent.py`)

```python
@pytest.mark.integration
async def test_shared_working_memory(agent_factory, memory_provider, seed, evaluator):
    """Test agents share working memory during coordination."""
    from draagon_ai.orchestration.shared_memory import SharedWorkingMemory

    # Create two agents
    researcher = await agent_factory.create(RESEARCHER_PROFILE)
    assistant = await agent_factory.create(ASSISTANT_PROFILE)

    # Shared working memory
    shared_memory = SharedWorkingMemory(task_id="test_task")

    # Researcher gathers information
    await shared_memory.add_observation(
        content="User prefers dark mode",
        source_agent_id="researcher",
        attention_weight=0.9,
        is_belief_candidate=True,
        belief_type="PREFERENCE",
    )

    # Assistant retrieves context
    context = await shared_memory.get_context_for_agent(
        agent_id="assistant",
        role=AgentRole.EXECUTOR,
        max_items=7,
    )

    # Verify context includes researcher's observation
    assert len(context) == 1
    assert "dark mode" in context[0].content.lower()
```

## Files to Create

- `tests/integration/test_memory_integration.py` - NEW
  - Memory recall tests

- `tests/integration/test_learning_flow.py` - NEW
  - Learning sequence tests

- `tests/integration/test_belief_reconciliation.py` - NEW
  - Belief update tests

- `tests/integration/test_multi_agent.py` - NEW
  - Multi-agent coordination tests

- `tests/integration/README.md` - NEW
  - Running instructions
  - Seed data examples
  - Common patterns

## Implementation Sequence

1. Create seed items for test scenarios
2. Write memory integration test
3. Write learning flow sequence test
4. Write belief reconciliation sequence test
5. Write multi-agent coordination test
6. Create integration tests README
7. Run all tests with real LLM + Neo4j
8. Document any issues found
9. Fix issues and verify all tests pass

## Cognitive Testing Requirements

**Integration Tests Must Verify:**
- [ ] Semantic understanding (LLM-as-judge, not string matching)
- [ ] Memory persistence across interactions
- [ ] Belief updates handle conflicts correctly
- [ ] Multi-agent shared memory works
- [ ] Confidence scores are reasonable

## Success Criteria

- All example tests pass with real LLM and Neo4j
- Tests demonstrate framework capabilities
- Tests serve as documentation
- LLM-as-judge evaluation works reliably
- Seed data is reusable and declarative
- README provides clear running instructions
- Framework is validated end-to-end
