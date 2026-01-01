# TASK-005: Implement Test Sequences

**Phase**: 2 (Test Sequences)
**Priority**: P1 (Enables multi-step tests)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-001, TASK-002

## Description

Implement test sequence infrastructure for multi-step tests that share database state. This enables testing agent learning, belief reconciliation, and other stateful behavior across multiple interactions.

## Acceptance Criteria

- [ ] `TestSequence` ABC base class
- [ ] `@step(order, depends_on)` decorator for marking steps
- [ ] Step dependency validation
- [ ] `sequence_database` fixture (class-scoped, persists during sequence)
- [ ] Pytest integration: auto-detect and order sequences
- [ ] Unit test: step ordering
- [ ] Unit test: dependency validation
- [ ] Integration test: database persists across steps

## Technical Notes

**TestSequence Pattern:**

```python
class TestSequence(ABC):
    """Base class for multi-step test sequences.

    Database persists across steps within a sequence.
    Use @step decorator to define execution order.
    """

    @classmethod
    def get_steps(cls) -> list[tuple[int, str, Callable]]:
        """Get all steps in execution order."""
        steps = []
        for name in dir(cls):
            method = getattr(cls, name)
            if callable(method) and hasattr(method, "_step_order"):
                steps.append((method._step_order, name, method))
        return sorted(steps, key=lambda x: x[0])

    @classmethod
    def validate_dependencies(cls) -> None:
        """Validate step dependencies are resolvable."""
        # Check all depends_on references exist
```

**@step Decorator:**

```python
def step(order: int, depends_on: str | None = None):
    """Decorator to mark test sequence steps."""
    def decorator(func: Callable) -> Callable:
        func._step_order = order
        func._step_depends_on = depends_on

        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"Executing step {order}: {func.__name__}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator
```

**Pytest Integration:**

```python
# tests/conftest.py addition
def pytest_collection_modifyitems(items):
    """Auto-detect and order test sequences."""
    for item in items:
        if hasattr(item.cls, "get_steps"):
            item.add_marker(pytest.mark.sequence)
            item.add_marker(pytest.mark.usefixtures("sequence_database"))

@pytest.fixture(scope="class")
async def sequence_database(test_database):
    """Class-scoped database for test sequences."""
    await test_database.clear()
    yield test_database
```

## Testing Requirements

### Unit Tests (`tests/framework/test_sequences.py`)

1. **Step Ordering**
   ```python
   class TestStepOrder(TestSequence):
       @step(2)
       async def test_second(self): pass

       @step(1)
       async def test_first(self): pass

   def test_get_steps_returns_ordered():
       steps = TestStepOrder.get_steps()
       assert steps[0][1] == "test_first"
       assert steps[1][1] == "test_second"
   ```

2. **Dependency Validation**
   ```python
   class TestBadDependency(TestSequence):
       @step(1, depends_on="nonexistent")
       async def test_step(self): pass

   def test_validate_dependencies_raises():
       with pytest.raises(ValueError, match="non-existent"):
           TestBadDependency.validate_dependencies()
   ```

### Integration Test (`tests/sequences/test_learning_flow.py`)

```python
class TestLearningFlow(TestSequence):
    """Test agent learning across interactions."""

    @step(1)
    async def test_initial_unknown(self, agent):
        response = await agent.process("When is my birthday?")
        assert response.confidence < 0.5

    @step(2, depends_on="test_initial_unknown")
    async def test_learn_birthday(self, agent):
        response = await agent.process("My birthday is March 15")
        assert "march 15" in response.answer.lower()

    @step(3, depends_on="test_learn_birthday")
    async def test_recall_birthday(self, agent):
        response = await agent.process("When is my birthday?")
        assert "march 15" in response.answer.lower()
        # Database persisted - agent remembers!
```

## Files to Create

- `src/draagon_ai/testing/sequences.py` - NEW
  - `TestSequence` ABC
  - `@step` decorator

- `tests/conftest.py` - EXTEND
  - `pytest_collection_modifyitems` hook
  - `sequence_database` fixture

- `tests/framework/test_sequences.py` - NEW
  - Test step ordering
  - Test dependency validation

- `tests/sequences/test_learning_flow.py` - NEW (example)
  - Example multi-step test

## Implementation Sequence

1. Implement `@step` decorator with order and depends_on
2. Implement `TestSequence.get_steps()` classmethod
3. Implement `TestSequence.validate_dependencies()`
4. Add `sequence_database` fixture (class-scoped)
5. Add pytest hook for auto-detection
6. Write unit tests
7. Write example integration test
8. Verify database persists across steps

## Cognitive Testing Requirements

**Integration Test Must Verify:**
- [ ] Agent learns from step 2 interaction
- [ ] Memory persists between steps (not cleared)
- [ ] Agent recalls learned information in step 3
- [ ] Database state carries forward through sequence

## Success Criteria

- Steps execute in correct order
- Dependencies are validated
- Database persists across steps in sequence
- Database clears BEFORE sequence starts
- Example learning flow test passes
- Ready for belief reconciliation tests
