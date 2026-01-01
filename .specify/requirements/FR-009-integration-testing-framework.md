# FR-009: Integration Testing Framework for Agentic AI Systems

**Status:** Draft
**Priority:** High
**Complexity:** Medium
**Created:** 2026-01-01
**Updated:** 2026-01-01

---

## Overview

A comprehensive testing framework for validating agentic AI systems that supports declarative seed data, test sequences with shared state, real LLM usage, test collections, app-specific configurations, Neo4j test database isolation, and future Docker-based parallelization.

**Core Principle:** Test **outcomes**, not **processes**. The framework validates that agents achieve the right results, not that they follow specific execution paths.

---

## Motivation

### Current State

draagon-ai has unit tests but lacks a comprehensive integration testing framework for agentic behavior:

- **No Seed Data Management**: Tests manually create data; hard to share fixtures
- **No Test Sequences**: Multi-step agent workflows can't share database state
- **No Real LLM Testing**: Integration tests need real LLM providers, not mocks
- **No App Profiles**: Can't configure test agents for different scenarios
- **No Outcome Evaluation**: No LLM-as-judge for validating agent behavior
- **No Parallelization Strategy**: Tests run serially; slow for large suites

### Problems

1. **Brittle Tests**: Hardcoded expectations break when agent makes smart decisions
2. **Slow Iteration**: Manual test data creation slows development
3. **Poor Coverage**: Can't test multi-step workflows effectively
4. **Limited Reusability**: Roxy can't extend draagon-ai integration tests
5. **Unclear Quality**: No cognitive metrics for agent performance

### Research Basis

| Aspect | Research |
|--------|----------|
| **LLM-as-Judge** | "Judging LLM-as-a-Judge" (2023) - Validates LLM evaluation reliability |
| **Agent Benchmarking** | MultiAgentBench, AgentBench, MAST taxonomy |
| **Test Database Isolation** | pytest best practices, Neo4j testing patterns |
| **Outcome Testing** | "Goal-Based Evaluation" vs "Process-Based Evaluation" |

---

## Requirements

### FR-009.1: Seed Data Factories

**Description:** Declarative system for creating test data with dependency resolution.

**Design Decision: Seeds Use Real MemoryProvider**

Seeds receive the **real MemoryProvider**, not a TestDatabase wrapper. This ensures:
1. **Tests use production APIs** - No leaky abstractions
2. **Real behavior** - Provider logic (importance, layers) is exercised
3. **Confidence** - If seeds work, production will work

```python
from draagon_ai.testing import SeedFactory, SeedSet, SeedItem
from draagon_ai.memory import MemoryProvider

# Define seed items - note they receive MemoryProvider, not TestDatabase
@SeedFactory.register("user_doug")
class DougUserSeed(SeedItem):
    """Creates Doug's user profile."""

    async def create(self, provider: MemoryProvider) -> str:
        # Use real MemoryProvider API directly
        memory_id = await provider.store(
            content="User profile: Doug, preferences: temperature_unit=celsius",
            metadata={
                "memory_type": "USER_PROFILE",
                "user_name": "Doug",
                "preferences": {"temperature_unit": "celsius"},
            }
        )
        return memory_id

@SeedFactory.register("doug_cats_fact")
class DougCatsFact(SeedItem):
    """Creates fact about Doug's cats."""

    dependencies = ["user_doug"]  # Must create user first

    async def create(self, provider: MemoryProvider, user_doug: str) -> str:
        # Use real MemoryProvider API directly
        memory_id = await provider.store(
            content="Doug has 3 cats: Whiskers, Mittens, and Shadow",
            metadata={
                "memory_type": "FACT",
                "related_memory": user_doug,
                "importance": 0.8,
            }
        )
        return memory_id

# Define seed sets
BASIC_USER = SeedSet("basic_user", ["user_doug"])
USER_WITH_PETS = SeedSet("user_with_pets", ["user_doug", "doug_cats_fact"])
```

**Usage:**

```python
@pytest.mark.asyncio
async def test_recall_pet_names(agent, seed: SeedSet):
    """Test agent recalls pet names from memory."""

    # Seed creates user + cats fact
    await seed.apply(USER_WITH_PETS)

    # Query agent
    response = await agent.process("What are my cats' names?")

    # Validate outcome (NOT process)
    assert "Whiskers" in response.answer
    assert "Mittens" in response.answer
    assert "Shadow" in response.answer
    # We DON'T care which tool the agent used to recall this
```

**Acceptance Criteria:**
- ✅ SeedItem base class with `create()` method
- ✅ Dependency resolution (topological sort)
- ✅ SeedSet composition for test scenarios
- ✅ Automatic cleanup after test
- ✅ Circular dependency detection

---

### FR-009.2: Test Sequences

**Description:** Multi-step tests that share database state across steps.

**Specification:**

```python
from draagon_ai.testing import TestSequence, step

class TestLearningFlow(TestSequence):
    """Test agent learning across multiple interactions."""

    @step(1)
    async def test_initial_unknown(self, agent):
        """Agent doesn't know user's birthday initially."""
        response = await agent.process("When is my birthday?")
        assert response.confidence < 0.5  # Low confidence

    @step(2, depends_on="test_initial_unknown")
    async def test_learn_birthday(self, agent):
        """Agent learns birthday from user."""
        response = await agent.process("My birthday is March 15")
        assert "march 15" in response.answer.lower()

    @step(3, depends_on="test_learn_birthday")
    async def test_recall_birthday(self, agent):
        """Agent recalls learned birthday."""
        response = await agent.process("When is my birthday?")
        assert "march 15" in response.answer.lower()
        assert response.confidence > 0.8  # High confidence now
```

**Usage:**

```python
# Run sequence (database persists between steps)
pytest tests/sequences/test_learning_flow.py -v

# Database cleared BEFORE sequence, but NOT between steps
```

**Acceptance Criteria:**
- ✅ TestSequence base class
- ✅ `@step(order, depends_on)` decorator
- ✅ Dependency validation (steps run in order)
- ✅ Database persists across steps
- ✅ Sequence-level setup/teardown
- ✅ Step failure stops sequence

---

### FR-009.3: Real LLM Usage

**Description:** Integration tests ALWAYS use real LLM providers.

**Specification:**

```python
# tests/conftest.py

@pytest.fixture(scope="session")
def llm_provider() -> LLMProvider:
    """Get real LLM provider for integration tests."""

    # Check for API key
    api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("No LLM API key found - set GROQ_API_KEY or OPENAI_API_KEY")

    # Return real provider (NO mocks)
    if os.getenv("GROQ_API_KEY"):
        return GroqProvider(api_key=os.getenv("GROQ_API_KEY"))
    else:
        return OpenAIProvider(api_key=os.getenv("OPENAI_API_KEY"))
```

**Rationale:**

Mock LLMs can't test:
- Semantic understanding
- Decision-making quality
- Prompt effectiveness
- Edge cases in reasoning
- Multi-turn coherence

Integration tests validate **real agent behavior**, which requires **real LLM inference**.

**Acceptance Criteria:**
- ✅ No mock LLM providers in integration tests
- ✅ Tests skip if no API key present
- ✅ Support multiple LLM providers (Groq, OpenAI, Ollama)
- ✅ Document API key setup in test README

---

### FR-009.4: Test Collections

**Description:** Pytest markers for organizing test suites.

**Specification:**

```python
# pytest.ini
[pytest]
markers =
    memory_integration: Tests for memory storage/retrieval
    learning_integration: Tests for agent learning
    multi_agent: Tests for multi-agent coordination
    belief_reconciliation: Tests for belief system
    smoke: Critical tests that must pass
    slow: Tests that take >5s
```

**Usage:**

```python
@pytest.mark.memory_integration
@pytest.mark.smoke
async def test_store_and_recall_fact(agent, clean_database):
    """Smoke test: Agent stores and recalls a fact."""
    ...

# Run specific collections
pytest -m "memory_integration"              # Memory tests only
pytest -m "smoke"                           # Critical tests
pytest -m "not slow"                        # Skip slow tests
pytest -m "memory_integration and smoke"    # Intersection
```

**Acceptance Criteria:**
- ✅ Standard pytest markers defined
- ✅ Markers documented in pytest.ini
- ✅ Collection groups align with cognitive services

---

### FR-009.5: App Profiles

**Description:** Configurable agent profiles for different test scenarios.

**Specification:**

```python
from draagon_ai.testing import AppProfile, ToolSet
from enum import Enum

class ToolSet(Enum):
    """Pre-defined tool collections."""
    MINIMAL = "minimal"          # Just answer tool
    BASIC = "basic"              # answer, get_time, calculate
    HOME_AUTOMATION = "home"     # Basic + HA tools
    FULL = "full"                # All tools

@dataclass
class AppProfile:
    """Agent configuration for testing."""
    name: str
    personality: str
    tool_set: ToolSet
    memory_config: dict
    llm_model_tier: str = "standard"

# Pre-defined profiles
MINIMAL_AGENT = AppProfile(
    name="minimal",
    personality="You are a helpful assistant.",
    tool_set=ToolSet.MINIMAL,
    memory_config={"working_ttl": 300},
)

ROXY_VOICE = AppProfile(
    name="roxy",
    personality="You are Roxy, a friendly home assistant...",
    tool_set=ToolSet.HOME_AUTOMATION,
    memory_config={"working_ttl": 300, "enable_learning": True},
    llm_model_tier="fast",
)
```

**Usage:**

```python
@pytest.mark.parametrize("profile", [MINIMAL_AGENT, ROXY_VOICE])
async def test_greeting_response(agent_factory, profile):
    """Test greeting across different agent profiles."""
    agent = await agent_factory.create(profile)
    response = await agent.process("Hello!")
    assert len(response.answer) > 0
```

**Acceptance Criteria:**
- ✅ AppProfile dataclass
- ✅ ToolSet enum with standard collections
- ✅ Pre-defined profiles for common scenarios
- ✅ agent_factory fixture for creating configured agents

---

### FR-009.6: Neo4j Test Database Isolation

**Description:** Isolated Neo4j test database with automatic cleanup.

**Design Decision: TestDatabase is Lifecycle-Only**

TestDatabase manages ONLY database lifecycle (initialize, clear, close). It does NOT:
- Wrap MemoryProvider with helper methods
- Provide convenience methods like `create_user()` or `store_memory()`
- Act as a facade over the real provider

Tests use the **real MemoryProvider directly**. This ensures:
1. **Production API coverage** - Tests validate actual production interfaces
2. **No leaky abstractions** - Helper methods don't mask real behavior
3. **Real bugs surface** - If MemoryProvider has issues, tests find them

**Specification:**

```python
from draagon_ai.testing import TestDatabase
from draagon_ai.memory.providers.neo4j import Neo4jMemoryConfig

@dataclass
class TestDatabase:
    """Neo4j test database lifecycle manager ONLY.

    Does NOT wrap or provide helper methods.
    Tests use the REAL MemoryProvider directly.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "draagon-ai-2025"
    database_name: str = "neo4j_test"  # Separate test database

    async def initialize(self) -> None:
        """Create test database if needed."""
        async with self._driver.session(database="system") as session:
            await session.run(f"CREATE DATABASE {self.database_name} IF NOT EXISTS")

    async def clear(self) -> None:
        """Delete all nodes and relationships."""
        async with self._driver.session(database=self.database_name) as session:
            await session.run("MATCH (n) DETACH DELETE n")

    async def close(self) -> None:
        """Close driver."""
        await self._driver.close()

    def get_config(self) -> Neo4jMemoryConfig:
        """Get config for creating REAL providers.

        Returns config - tests create their own providers.
        """
        return Neo4jMemoryConfig(
            uri=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.database_name,
        )
```

**Usage:**

```python
@pytest.fixture(scope="session")
async def test_database() -> AsyncIterator[TestDatabase]:
    """Session-scoped test database lifecycle manager."""
    db = TestDatabase()
    await db.initialize()
    yield db
    await db.close()

@pytest.fixture
async def clean_database(test_database: TestDatabase) -> AsyncIterator[TestDatabase]:
    """Provide clean database for each test."""
    await test_database.clear()
    yield test_database

@pytest.fixture
async def memory_provider(clean_database: TestDatabase) -> AsyncIterator[MemoryProvider]:
    """REAL MemoryProvider for tests.

    Tests use this directly - no wrappers.
    """
    from draagon_ai.memory.providers.layered import LayeredMemoryProvider

    config = clean_database.get_config()
    provider = LayeredMemoryProvider.from_config(config)
    await provider.initialize()
    yield provider
    await provider.close()
```

**Acceptance Criteria:**
- ✅ TestDatabase class for Neo4j management
- ✅ Session-scoped fixture for database instance
- ✅ Function-scoped fixture for clean state
- ✅ Automatic cleanup between tests
- ✅ Separate `neo4j_test` database

---

### FR-009.7: Docker-Based Parallelization (Future)

**Description:** Run N parallel test workers with separate Neo4j instances.

**Specification:**

```yaml
# docker-compose.test.yml (Future Implementation)
services:
  neo4j-test-0:
    image: neo4j:5.26-community
    ports: ["7687:7687", "7474:7474"]
    environment:
      - NEO4J_AUTH=neo4j/test-password
      - NEO4J_PLUGINS=["apoc"]

  neo4j-test-1:
    image: neo4j:5.26-community
    ports: ["7688:7687", "7475:7474"]
    environment:
      - NEO4J_AUTH=neo4j/test-password
      - NEO4J_PLUGINS=["apoc"]

  neo4j-test-2:
    image: neo4j:5.26-community
    ports: ["7689:7687", "7476:7474"]
    environment:
      - NEO4J_AUTH=neo4j/test-password
      - NEO4J_PLUGINS=["apoc"]
```

```python
# tests/framework/parallel.py (Future Implementation)

@dataclass
class WorkerContext:
    """Context for parallel test worker."""
    worker_id: str           # "gw0", "gw1", "gw2"
    worker_index: int        # 0, 1, 2
    total_workers: int
    neo4j_port: int          # 7687, 7688, 7689

    @property
    def neo4j_uri(self) -> str:
        return f"bolt://localhost:{self.neo4j_port}"

    @classmethod
    def from_pytest(cls) -> "WorkerContext":
        """Create from pytest-xdist environment."""
        worker_id = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
        worker_count = int(os.environ.get("PYTEST_XDIST_WORKER_COUNT", "1"))
        worker_index = int(worker_id.replace("gw", "")) if worker_id.startswith("gw") else 0

        WORKER_PORTS = {0: 7687, 1: 7688, 2: 7689, 3: 7690}

        return cls(
            worker_id=worker_id,
            worker_index=worker_index,
            total_workers=worker_count,
            neo4j_port=WORKER_PORTS.get(worker_index, 7687),
        )
```

**Usage (Future):**

```bash
# Start test databases
docker-compose -f docker-compose.test.yml up -d

# Run tests with 4 parallel workers
pytest tests/integration/ -n 4 -v

# Each worker connects to its own Neo4j instance
```

**Acceptance Criteria (Future):**
- ✅ Docker Compose config for N Neo4j instances
- ✅ WorkerContext for pytest-xdist integration
- ✅ Worker-specific database assignment
- ✅ Test sequences run on same worker (xdist_group)
- ✅ Documented setup/teardown process

**Note:** This is designed now, implemented later when parallelization is needed.

---

### FR-009.8: LLM-as-Judge Evaluation

**Description:** Use LLM to evaluate agent responses for semantic correctness.

**Specification:**

```python
from draagon_ai.testing import AgentEvaluator

class AgentEvaluator:
    """LLM-based evaluation of agent responses."""

    async def evaluate_correctness(
        self,
        query: str,
        expected_outcome: str,
        actual_response: str,
    ) -> EvaluationResult:
        """Evaluate if response achieves expected outcome."""

        prompt = f"""<evaluation>
<query>{query}</query>
<expected_outcome>{expected_outcome}</expected_outcome>
<actual_response>{actual_response}</actual_response>

Did the agent's response achieve the expected outcome?
Answer with:
<result>
  <correct>true/false</correct>
  <reasoning>Why or why not</reasoning>
  <confidence>0.0-1.0</confidence>
</result>
</evaluation>"""

        result = await self.llm.chat([{"role": "user", "content": prompt}])
        return self._parse_evaluation(result)

    async def evaluate_coherence(self, response: str) -> float:
        """Score response coherence (0.0-1.0)."""
        ...

    async def evaluate_helpfulness(self, query: str, response: str) -> float:
        """Score response helpfulness (0.0-1.0)."""
        ...
```

**Usage:**

```python
async def test_pet_recall_semantic(agent, evaluator: AgentEvaluator):
    """Test agent recalls pets using semantic evaluation."""

    response = await agent.process("What are my cats' names?")

    # LLM-as-judge evaluation
    result = await evaluator.evaluate_correctness(
        query="What are my cats' names?",
        expected_outcome="Response lists the three cat names: Whiskers, Mittens, Shadow",
        actual_response=response.answer,
    )

    assert result.correct, f"Agent failed: {result.reasoning}"
    assert result.confidence > 0.8
```

**Acceptance Criteria:**
- ✅ AgentEvaluator class
- ✅ evaluate_correctness() for semantic validation
- ✅ evaluate_coherence() for response quality
- ✅ evaluate_helpfulness() for UX quality
- ✅ XML-based prompts for evaluation
- ✅ Confidence scores for evaluation reliability

---

## Data Structures

### SeedItem

```python
from abc import ABC, abstractmethod
from typing import Any, ClassVar
from draagon_ai.memory import MemoryProvider

class SeedItem(ABC):
    """Base class for seed data items.

    Seeds receive REAL MemoryProvider, not a wrapper.
    This ensures tests validate production APIs.
    """

    dependencies: ClassVar[list[str]] = []  # Other seed IDs this depends on

    @abstractmethod
    async def create(self, provider: MemoryProvider, **deps: Any) -> str:
        """
        Create seed data using REAL MemoryProvider.

        Args:
            provider: Real MemoryProvider instance (not a wrapper)
            **deps: Resolved dependencies (key = dep name, value = dep result)

        Returns:
            Seed item ID or reference
        """
        ...
```

### SeedSet

```python
from dataclasses import dataclass
from draagon_ai.memory import MemoryProvider

@dataclass
class SeedSet:
    """Collection of seed items."""

    name: str
    items: list[str]  # Seed IDs to include

    async def apply(self, provider: MemoryProvider) -> dict[str, Any]:
        """
        Apply all seed items using REAL MemoryProvider.

        Args:
            provider: Real MemoryProvider (not wrapper)

        Returns:
            Mapping of seed_id -> created reference
        """
        factory = SeedFactory()
        return await factory.create_all(self.items, provider)
```

### TestSequence

```python
from abc import ABC

class TestSequence(ABC):
    """Base class for multi-step test sequences."""

    @classmethod
    def get_steps(cls) -> list[tuple[int, str, Callable]]:
        """Get all test steps in order."""
        steps = []
        for name in dir(cls):
            method = getattr(cls, name)
            if hasattr(method, "_step_order"):
                steps.append((method._step_order, name, method))
        return sorted(steps, key=lambda x: x[0])
```

### EvaluationResult

```python
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Result from LLM-as-judge evaluation."""

    correct: bool
    reasoning: str
    confidence: float
    metrics: dict[str, float] = None  # Optional detailed metrics
```

---

## API Examples

### Basic Integration Test

```python
import pytest
from draagon_ai.testing import SeedSet, BASIC_USER

@pytest.mark.memory_integration
@pytest.mark.smoke
async def test_store_and_recall(agent, memory_provider, seed):
    """Test agent stores and recalls facts.

    Note: Uses memory_provider fixture (REAL provider).
    """

    # Seed test data using real provider
    await seed.apply(BASIC_USER, memory_provider)

    # Agent learns fact
    response1 = await agent.process("Remember that I like pizza")
    assert "pizza" in response1.answer.lower()

    # Agent recalls fact
    response2 = await agent.process("What food do I like?")
    assert "pizza" in response2.answer.lower()
```

### Sequence Test

```python
from draagon_ai.testing import TestSequence, step

class TestBeliefReconciliation(TestSequence):
    """Test belief system handles conflicting information."""

    @step(1)
    async def test_initial_belief(self, agent):
        response = await agent.process("I have 2 cats")
        assert "2" in response.answer

    @step(2, depends_on="test_initial_belief")
    async def test_correction(self, agent):
        response = await agent.process("Actually, I have 3 cats")
        assert "3" in response.answer

    @step(3, depends_on="test_correction")
    async def test_corrected_recall(self, agent):
        response = await agent.process("How many cats do I have?")
        assert "3" in response.answer  # Should use corrected belief
```

### LLM-as-Judge Test

```python
@pytest.mark.learning_integration
async def test_contextual_learning(agent, evaluator):
    """Test agent learns from context."""

    # Multi-turn conversation
    await agent.process("I work from home on Mondays and Wednesdays")
    response = await agent.process("Do I work from home on Tuesday?")

    # Semantic evaluation (NOT exact string match)
    result = await evaluator.evaluate_correctness(
        query="Do I work from home on Tuesday?",
        expected_outcome="Agent should say no/user doesn't work from home on Tuesday",
        actual_response=response.answer,
    )

    assert result.correct
    assert result.confidence > 0.7
```

### App Profile Test

```python
from draagon_ai.testing import MINIMAL_AGENT, ROXY_VOICE

@pytest.mark.parametrize("profile", [MINIMAL_AGENT, ROXY_VOICE])
async def test_time_query(agent_factory, profile):
    """Test time queries across agent profiles."""
    agent = await agent_factory.create(profile)

    response = await agent.process("What time is it?")

    # Validate outcome (NOT which tool was used)
    assert len(response.answer) > 0
    # We don't care if agent used get_time tool or knowledge
```

---

## Implementation Plan

### Phase 1: Core Framework

**Tasks:**
1. Create new modules in `src/draagon_ai/testing/`
2. Implement SeedItem, SeedFactory, SeedSet in `seeds.py`
3. Implement TestDatabase for Neo4j isolation in `database.py`
4. Extend `fixtures.py` with new fixtures
5. Update `__init__.py` exports
6. Write framework unit tests

**Deliverables:**
- `src/draagon_ai/testing/seeds.py` - NEW
- `src/draagon_ai/testing/database.py` - NEW
- `src/draagon_ai/testing/fixtures.py` - EXTEND
- `src/draagon_ai/testing/__init__.py` - EXTEND
- `tests/framework/test_seeds.py` - NEW (unit tests for framework)

### Phase 2: Test Sequences

**Tasks:**
1. Implement TestSequence base class
2. Create `@step` decorator with dependency resolution
3. Add sequence execution logic
4. Write example sequence tests

**Deliverables:**
- `src/draagon_ai/testing/sequences.py` - NEW
- `tests/sequences/test_learning_flow.py` - NEW (example)
- `tests/sequences/test_belief_reconciliation.py` - NEW (example)

### Phase 3: LLM-as-Judge

**Tasks:**
1. Implement AgentEvaluator
2. Add evaluation prompts (XML-based)
3. Add evaluator fixture
4. Write evaluation tests

**Deliverables:**
- `src/draagon_ai/testing/evaluation.py` - NEW
- `tests/framework/test_evaluation.py` - NEW

### Phase 4: App Profiles

**Tasks:**
1. Implement AppProfile dataclass
2. Define ToolSet enum
3. Create pre-defined profiles
4. Add agent_factory fixture
5. Write profile tests

**Deliverables:**
- `src/draagon_ai/testing/profiles.py` - NEW
- `tests/profiles/test_minimal.py` - NEW
- `tests/profiles/test_roxy.py` - NEW

### Phase 5: Integration Tests

**Tasks:**
1. Write memory integration tests
2. Write learning integration tests
3. Write belief reconciliation tests
4. Write multi-agent tests
5. Document test writing guide

**Deliverables:**
- `tests/integration/test_memory_fr009.py` - NEW
- `tests/integration/test_learning_fr009.py` - NEW
- `tests/integration/test_beliefs_fr009.py` - NEW
- `tests/integration/test_multi_agent_fr009.py` - NEW
- `tests/README.md` - EXTEND

### Phase 6: Parallelization Design (Future)

**Tasks:**
1. Design Docker Compose setup
2. Implement WorkerContext
3. Update fixtures for parallel mode
4. Document parallel execution
5. Add xdist_group markers

**Deliverables:**
- `docker-compose.test.yml` - NEW
- `src/draagon_ai/testing/parallel.py` - NEW
- Updated `tests/README.md`

**Note:** Phase 6 is designed now, implemented when needed.

---

## Success Metrics

### Framework Adoption

| Metric | Target |
|--------|--------|
| Integration test coverage | 80%+ of cognitive services |
| Roxy test extension | 20+ Roxy-specific tests using framework |
| Test execution time | <5min for full suite (serial), <2min (parallel) |

### Quality Metrics

| Metric | Target |
|--------|--------|
| LLM-as-judge agreement | >90% with human evaluation |
| Test flakiness | <2% false positives |
| Seed data reuse | 50%+ tests use shared seeds |

### Cognitive Metrics

| Metric | Target |
|--------|--------|
| Memory recall accuracy | >95% |
| Belief consistency | >90% |
| Learning retention | >85% across sessions |
| Multi-agent coordination | >80% success rate |

---

## Testing Strategy

### Unit Tests (Framework Itself)

```python
# tests/framework/test_seeds.py

async def test_seed_dependency_resolution():
    """Test seed factory resolves dependencies."""
    factory = SeedFactory()
    factory.register("parent", ParentSeed())
    factory.register("child", ChildSeed(dependencies=["parent"]))

    result = await factory.create_all(["child"], db)
    assert "parent" in result
    assert "child" in result

async def test_seed_circular_dependency_error():
    """Test circular dependencies are detected."""
    factory = SeedFactory()
    factory.register("a", ASeed(dependencies=["b"]))
    factory.register("b", BSeed(dependencies=["a"]))

    with pytest.raises(CircularDependencyError):
        await factory.create_all(["a"], db)
```

### Integration Tests (Using Framework)

```python
# tests/integration/test_memory.py

@pytest.mark.memory_integration
@pytest.mark.smoke
async def test_fact_storage(agent, clean_database):
    """Test agent stores facts correctly."""
    response = await agent.process("My favorite color is blue")

    # Verify fact was stored
    memories = await agent.memory.search("favorite color")
    assert len(memories) > 0
    assert memories[0].memory_type == MemoryType.FACT
```

### Validation Tests (LLM-as-Judge)

```python
# tests/integration/test_helpfulness.py

async def test_greeting_helpfulness(agent, evaluator):
    """Test agent greetings are helpful."""
    response = await agent.process("Hello!")

    helpfulness = await evaluator.evaluate_helpfulness(
        query="Hello!",
        response=response.answer,
    )

    assert helpfulness > 0.7  # Should be reasonably helpful
```

---

## Constitution Compliance

### ✅ LLM-First Architecture

- **AgentEvaluator** uses LLM for semantic evaluation (no regex)
- **Outcome Testing** validates results, not process (LLM decides how)

### ✅ XML Output Format

- All evaluation prompts use XML format
- Parsing uses XML element extraction (not JSON)

### ✅ Protocol-Based Design

- TestDatabase is lifecycle-only (initialize, clear, close)
- Seeds use REAL MemoryProvider (not wrapper)
- AgentFactory uses LLMProvider protocol

### ✅ Pragmatic Async

- Async for I/O operations (database, LLM evaluation)
- Sync for pure computation (seed dependency resolution, result parsing)

### ✅ Research-Grounded

| Feature | Research |
|---------|----------|
| LLM-as-judge | "Judging LLM-as-a-Judge" (2023) |
| Outcome testing | Goal-based evaluation literature |
| Test isolation | pytest best practices |

### ✅ Test Outcomes, Not Processes

**CRITICAL:** Framework enforces outcome-based testing:

```python
# GOOD: Test outcome
assert "pizza" in response.answer

# GOOD: Test semantic outcome with LLM-as-judge
result = await evaluator.evaluate_correctness(...)
assert result.correct

# BAD: Test process
assert agent.last_tool == "get_weather"  # ❌ DON'T DO THIS

# BAD: Test exact tool usage
assert "get_weather" in agent.tool_history  # ❌ DON'T DO THIS
```

The framework makes it EASY to test outcomes, HARD to test processes.

---

## Existing Infrastructure Analysis

### Current Testing Infrastructure

draagon-ai already has a mature testing foundation (82 test files, ~49K lines). This section documents what exists and how it relates to FR-009.

#### Testing Utilities (`src/draagon_ai/testing/`)

| File | Lines | Purpose | FR-009 Decision |
|------|-------|---------|-----------------|
| `cache.py` | 12K | Unified cache manager for LLM/HTTP responses | **KEEP AS-IS** - Reuse for caching seed setup |
| `mocks.py` | 11.8K | Service mocking (LLMMock, HTTPMock, EmbeddingMock) | **KEEP AS-IS** - Use for framework unit tests |
| `fixtures.py` | 8.5K | Pytest fixtures, markers, test mode hooks | **EXTEND** - Add seed, database, evaluator fixtures |
| `modes.py` | 8.4K | Test mode detection (TestMode, TestContext) | **KEEP AS-IS** - Perfect for mode switching |
| `__init__.py` | ~100 | Module exports | **EXTEND** - Add new class exports |

#### Main Test Configuration (`tests/conftest.py`)

| Component | Purpose | FR-009 Decision |
|-----------|---------|-----------------|
| `MockLLMProvider` | Predetermined LLM responses | **KEEP** - Use for unit tests, not integration |
| `MockMemoryProvider` | In-memory storage | **MIGRATE** - Wrap in TestDatabase interface |
| `mock_tool` fixture | Tool mock factory | **KEEP** - Extend pattern for agent_factory |
| `test_user_id` fixture | Consistent user IDs | **KEEP** - Use in seed items |

#### Test Structure

```
tests/                          # 82 files, ~49K lines
├── conftest.py                 # EXTEND with new fixtures
├── unit/                       # KEEP AS-IS
├── integration/                # EXTEND with FR-009 tests
│   ├── test_orchestration_integration.py
│   ├── test_layered_provider_integration.py
│   ├── test_semantic_context_integration.py
│   └── ...
├── e2e/                        # KEEP AS-IS
│   ├── test_behavior_architect_e2e.py
│   └── test_multistep_reasoning.py
└── [module-specific]/          # KEEP AS-IS
```

### Patterns Worth Preserving

**1. CacheManager Singleton** (from `cache.py`):
```python
# Use for caching seed factory setup
cache = UnifiedCacheManager.get_instance()
```

**2. Test Mode Context Manager** (from `modes.py`):
```python
# Already perfect for framework usage
with test_mode(TestMode.INTEGRATION, mocks={"db": test_db}):
    result = await agent.process(query)
```

**3. Fixture Factory Pattern** (from `fixtures.py`):
```python
# Extend for app profiles
@pytest.fixture
def llm_mock() -> LLMMock:
    return LLMMock()
```

**4. RealisticMockLLM** (from `tests/e2e/`):
```python
# Already exists for fast E2E - could extend for seed factories
from draagon_ai.llm import RealisticMockLLM
```

### Gap Analysis

| FR-009 Requirement | Current Status | Gap |
|-------------------|----------------|-----|
| FR-009.1: Seed Factories | MISSING | Need SeedItem, SeedFactory, SeedSet |
| FR-009.2: Test Sequences | MISSING | Need TestSequence, @step decorator |
| FR-009.3: Real LLM Usage | PARTIAL | E2E tests use real LLM, not systematic |
| FR-009.4: Test Collections | PARTIAL | Markers exist, not grouped by cognitive service |
| FR-009.5: App Profiles | MISSING | Need AppProfile, ToolSet, agent_factory |
| FR-009.6: Neo4j Isolation | PARTIAL | Tests connect to Neo4j, no abstraction |
| FR-009.7: Parallelization | N/A | Design only (future) |
| FR-009.8: LLM-as-Judge | MISSING | Need AgentEvaluator |

### Files to Create

```
src/draagon_ai/testing/
├── seeds.py          # NEW - SeedItem, SeedFactory, SeedSet
├── database.py       # NEW - TestDatabase for Neo4j
├── sequences.py      # NEW - TestSequence, @step decorator
├── evaluation.py     # NEW - AgentEvaluator, EvaluationResult
└── profiles.py       # NEW - AppProfile, ToolSet

tests/
├── conftest.py       # EXTEND - add session-scoped Neo4j fixtures
├── framework/        # NEW - framework unit tests
│   ├── test_seeds.py
│   ├── test_sequences.py
│   └── test_evaluation.py
├── integration/      # EXTEND - add FR-009 style tests
│   ├── test_memory.py      # NEW
│   ├── test_learning.py    # NEW
│   └── test_beliefs.py     # NEW
└── sequences/        # NEW - multi-step test examples
    ├── test_learning_flow.py
    └── test_belief_reconciliation.py
```

### What Gets Deprecated

**Nothing.** All existing infrastructure is valuable:
- Unit tests continue using mocks
- Existing integration tests remain valid
- New framework extends, doesn't replace

### Migration Guide

This guide shows how to migrate existing integration tests to use FR-009 patterns.

#### Step 1: Identify Tests to Migrate

```bash
# Find tests that manually create test data
grep -r "provider.store" tests/integration/
grep -r "Neo4j" tests/integration/
```

Focus on tests that:
- Manually create test data in setup
- Use hardcoded assertions (string matching)
- Connect directly to Neo4j

#### Step 2: Extract Seed Data

**BEFORE (manual setup scattered in test):**
```python
async def test_recall_facts(memory_provider):
    # Manual setup - hard to reuse
    memory_id = await memory_provider.store(
        content="Doug has 3 cats",
        metadata={"memory_type": "FACT", "importance": 0.8}
    )

    # Test logic...
    response = await agent.process("How many cats?")
    assert "3" in response.answer  # Brittle assertion
```

**AFTER (seed-based, reusable):**
```python
from draagon_ai.testing import SeedFactory, SeedItem, SeedSet

# Define once, reuse everywhere
@SeedFactory.register("doug_cats_fact")
class DougCatsFact(SeedItem):
    async def create(self, provider: MemoryProvider) -> str:
        return await provider.store(
            content="Doug has 3 cats",
            metadata={"memory_type": "FACT", "importance": 0.8}
        )

DOUG_PROFILE = SeedSet("doug_profile", ["doug_cats_fact"])

async def test_recall_facts(seed, memory_provider, evaluator):
    # Declarative setup
    await seed.apply(DOUG_PROFILE, memory_provider)

    # Test logic...
    response = await agent.process("How many cats?")

    # Semantic evaluation - robust to phrasing variations
    result = await evaluator.evaluate_correctness(
        query="How many cats?",
        expected_outcome="Agent says Doug has 3 cats",
        actual_response=response.answer
    )
    assert result.correct
```

#### Step 3: Replace String Assertions with LLM-as-Judge

**BEFORE (brittle string matching):**
```python
assert "3 cats" in response.answer
assert response.confidence > 0.8
```

**AFTER (semantic evaluation):**
```python
result = await evaluator.evaluate_correctness(
    query="How many cats do I have?",
    expected_outcome="Agent correctly states 3 cats",
    actual_response=response.answer
)
assert result.correct, f"Failed: {result.reasoning}"
```

#### Step 4: Use Fixtures Instead of Direct Neo4j

**BEFORE (direct Neo4j connection):**
```python
from neo4j import AsyncGraphDatabase

async def test_something():
    driver = AsyncGraphDatabase.driver("bolt://localhost:7687")
    async with driver.session() as session:
        await session.run("MATCH (n) DETACH DELETE n")
    # ...
```

**AFTER (use fixtures):**
```python
async def test_something(memory_provider, seed):
    # Fixtures handle connection, cleanup, lifecycle
    await seed.apply(MY_SCENARIO, memory_provider)
    # ...
```

#### Coexistence Strategy

Old and new patterns can coexist during migration:

| Scenario | Approach |
|----------|----------|
| New tests | Use FR-009 patterns exclusively |
| High-value existing tests | Migrate gradually |
| Low-value existing tests | Keep as-is or delete if redundant |
| Shared test data | Extract to SeedItem classes |

**Migration Priority:**
1. Tests that are flaky (likely due to brittle assertions)
2. Tests with duplicated setup code
3. Tests for critical cognitive services
4. Everything else

---

## Open Questions

None. Specification is complete.

---

## Related Documents

- **CONSTITUTION.md** - Test outcomes not processes principle
- **PRINCIPLES.md** - LLM-first, XML output, protocol-based design
- **FR-001** - Shared working memory (multi-agent testing)
- **FR-003** - Belief reconciliation (belief testing)
- **FR-005** - Metacognitive reflection (reflection testing)

---

## References

1. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (2023)
2. MultiAgentBench: Evaluating Multi-Agent Systems (2024)
3. pytest Documentation: Best Practices for Test Fixtures
4. Neo4j Testing Patterns: Database Isolation
5. "Goal-Based vs Process-Based Evaluation" - Education Research

---

**Document Status:** Draft - Ready for Review
**Next Steps:** Create Phase 1 implementation plan
**Estimated Effort:** 3-4 weeks (Phases 1-5), Future (Phase 6)
