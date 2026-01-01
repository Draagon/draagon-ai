# Integration Tests

Integration tests for draagon-ai using the FR-009 testing framework.

## Overview

These tests demonstrate and validate the integration testing framework:

| Test File | Purpose |
|-----------|---------|
| `test_memory_integration.py` | Memory recall with seeds and LLM-as-judge |
| `test_learning_flow.py` | Multi-step learning sequences |
| `test_multi_agent.py` | Multi-agent coordination patterns |
| `test_semantic_context_integration.py` | Semantic context enrichment |

## Requirements

### Required

- **Python 3.11+**
- **Neo4j 5.15+** (for vector indexes)
- **neo4j** Python package

### Optional

- **GROQ_API_KEY** - For real LLM evaluation
- **OPENAI_API_KEY** - Alternative LLM provider

## Running Tests

### Basic (with Mock LLM)

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific test file
pytest tests/integration/test_memory_integration.py -v

# Run with markers
pytest tests/integration/ -m "memory_integration" -v
pytest tests/integration/ -m "sequence_test" -v
```

### With Real Neo4j

```bash
# Start Neo4j (Docker recommended)
docker run -d \
  --name neo4j-test \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/draagon-ai-2025 \
  neo4j:5.26

# Run tests
pytest tests/integration/ -v
```

### Environment Variables

```bash
# Neo4j configuration
export NEO4J_TEST_URI="bolt://localhost:7687"
export NEO4J_TEST_USER="neo4j"
export NEO4J_TEST_PASSWORD="draagon-ai-2025"
export NEO4J_TEST_DATABASE="neo4j"

# LLM providers (optional)
export GROQ_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

## Writing Integration Tests

### 1. Define Seed Items

Seed items provide declarative test data that uses the **real MemoryProvider API**:

```python
from draagon_ai.testing import SeedFactory, SeedItem
from draagon_ai.memory import MemoryType, MemoryScope

@SeedFactory.register("user_doug")
class DougUserSeed(SeedItem):
    """Doug's user profile."""

    async def create(self, provider, **deps) -> str:
        memory = await provider.store(
            content="Doug has 3 cats: Whiskers, Mittens, Shadow",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
        )
        return memory.id
```

### 2. Create Seed Sets

Group seeds into reusable scenarios:

```python
from draagon_ai.testing import SeedSet

USER_WITH_CATS = SeedSet(
    name="user_with_cats",
    seed_ids=["user_doug"],
    description="User profile with cat information",
)
```

### 3. Write Tests with LLM-as-Judge

Use semantic evaluation instead of string matching:

```python
@pytest.mark.memory_integration
async def test_recall_cats(seed, memory_provider, evaluator):
    # Apply seeds
    await seed.apply(USER_WITH_CATS, memory_provider)

    # Query (using mock agent here)
    response = await agent.process("What are my cats' names?")

    # LLM-as-judge evaluation
    result = await evaluator.evaluate_correctness(
        query="What are my cats' names?",
        expected_outcome="Lists: Whiskers, Mittens, Shadow",
        actual_response=response.answer,
    )

    assert result.correct, f"Failed: {result.reasoning}"
```

### 4. Multi-Step Sequences

For tests where state persists between steps:

```python
from draagon_ai.testing import TestSequence, step

class TestLearningFlow(TestSequence):

    @step(1)
    async def test_initial_unknown(self, agent, evaluator):
        response = await agent.process("When is my birthday?")
        assert response.confidence < 0.5

    @step(2, depends_on="test_initial_unknown")
    async def test_learn_birthday(self, agent, evaluator):
        response = await agent.process("My birthday is March 15")
        # Agent learns...

    @step(3, depends_on="test_learn_birthday")
    async def test_recall_birthday(self, agent, evaluator):
        response = await agent.process("When is my birthday?")
        assert "March 15" in response.answer
```

## Test Markers

| Marker | Description |
|--------|-------------|
| `@pytest.mark.integration` | All integration tests |
| `@pytest.mark.memory_integration` | Memory-specific tests |
| `@pytest.mark.sequence_test` | Multi-step sequence tests |
| `@pytest.mark.learning_integration` | Learning flow tests |
| `@pytest.mark.smoke` | Critical tests that must pass |

## Key Principles

### 1. Test Outcomes, Not Processes

```python
# GOOD - Test the outcome
result = await evaluator.evaluate_correctness(
    query="What are my cats?",
    expected_outcome="Lists cat names",
    actual_response=response.answer,
)
assert result.correct

# BAD - Test implementation details
assert "Whiskers" in response.answer  # Fragile string matching
```

### 2. Seeds Use Real Provider API

Seeds receive the **actual MemoryProvider**, not a wrapper. This ensures tests validate production interfaces:

```python
async def create(self, provider, **deps):
    # This is the REAL provider.store() method
    return await provider.store(
        content="...",
        memory_type=MemoryType.FACT,
        ...
    )
```

### 3. LLM-as-Judge for Semantic Evaluation

Use the AgentEvaluator for flexible, semantic comparisons:

```python
# Handles paraphrasing, different orderings, synonyms
result = await evaluator.evaluate_correctness(
    query="How many pets?",
    expected_outcome="Mentions 3 cats",
    actual_response="You've got three cats!",  # Different words, same meaning
)
assert result.correct  # True!
```

### 4. Instance-Based Seed Registry

Each test gets its own SeedFactory instance for parallel safety:

```python
@pytest.fixture
def seed_factory():
    return SeedFactory()  # Fresh instance per test

@pytest.fixture
def seed(seed_factory):
    return SeedApplicator(factory=seed_factory)
```

## Debugging

### View Neo4j Data

After a test fails, data is preserved for debugging:

```
http://localhost:7474
Database: neo4j
Username: neo4j
Password: draagon-ai-2025
```

### Check Test Database

```python
# In test
count = await clean_database.node_count()
print(f"Database has {count} nodes")
```

### View Evaluation Reasoning

```python
result = await evaluator.evaluate_correctness(...)

if not result.correct:
    print(f"Reasoning: {result.reasoning}")
    print(f"Confidence: {result.confidence}")
    print(f"Raw response: {result.raw_response}")
```

## Fixtures Reference

| Fixture | Scope | Description |
|---------|-------|-------------|
| `test_database` | session | Neo4j lifecycle manager |
| `clean_database` | function | Clears data before test |
| `seed_factory` | function | Fresh SeedFactory |
| `seed` | function | SeedApplicator |
| `evaluator` | function | AgentEvaluator with mock LLM |
| `llm_provider` | session | LLM provider (mock by default) |
| `mock_llm` | session | MockLLMProvider |

## Seed Registration Patterns

### Global Registration (Simple Cases)

Use `@SeedFactory.register()` decorator for seeds in standalone test files:

```python
@SeedFactory.register("my_seed")
class MySeed(SeedItem):
    async def create(self, provider, **deps):
        return await provider.store(...)
```

**Pros:** Simple, declarative, seeds available globally
**Cons:** Can conflict if other tests clear the global registry

### Instance Registration (Test Isolation)

Use fixture-based registration when running alongside tests that clear the global registry:

```python
@pytest.fixture
def factory_with_seeds():
    factory = SeedFactory()
    factory.register_instance("my_seed", MySeed())
    factory.register_instance("other_seed", OtherSeed())
    return factory

def test_something(factory_with_seeds):
    seeds = factory_with_seeds.list_seeds()
    assert "my_seed" in seeds
```

**Pros:** Isolated, no conflicts with other tests
**Cons:** Slightly more verbose

### When to Use Each

| Scenario | Recommendation |
|----------|----------------|
| Standalone integration test file | Global `@SeedFactory.register()` |
| Tests run with unit tests that clear registry | Instance `register_instance()` |
| Parallel test execution | Instance (each test gets own factory) |
| Shared seeds across test files | Global, but be aware of cleanup |

## Known Issues

### Event Loop Teardown Error

You may see this error at the end of test session:

```
RuntimeError: Event loop is closed
```

**What's happening:** The Neo4j async driver tries to close its socket connection after pytest-asyncio has already closed the event loop. This is a timing issue between pytest-asyncio fixture teardown and the driver's cleanup.

**Impact:** None - all tests pass correctly. The error occurs during cleanup after tests complete.

**Status:** Known interaction between pytest-asyncio and neo4j async driver. Not a test failure.

**Workaround:** Ignore this specific teardown error. If you need cleaner output, you can suppress it with:

```python
# In conftest.py
import warnings
warnings.filterwarnings("ignore", message="Event loop is closed")
```
