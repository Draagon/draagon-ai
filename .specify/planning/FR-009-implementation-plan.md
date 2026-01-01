# FR-009: Integration Testing Framework - Technical Implementation Plan

**Status:** Ready for Implementation
**Priority:** High
**Created:** 2026-01-01
**Estimated Effort:** 3-4 weeks

---

## Overview

This document provides detailed technical implementation guidance for FR-009 (Integration Testing Framework for Agentic AI Systems). It breaks down each phase into specific tasks with code patterns, architectural decisions, and testing strategies.

---

## Phase 1: Core Framework (Week 1)

### 1.1 Module: `src/draagon_ai/testing/seeds.py`

**Purpose:** Declarative seed data with dependency resolution

**Key Classes:**

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar
import logging

logger = logging.getLogger(__name__)


class CircularDependencyError(Exception):
    """Raised when circular dependencies detected in seed graph."""
    pass


class SeedItem(ABC):
    """Base class for seed data items.

    Subclasses implement create() using REAL MemoryProvider.
    Dependencies are resolved via topological sort before creation.

    IMPORTANT: Seeds receive MemoryProvider, NOT TestDatabase.
    This ensures tests validate production APIs.

    Example:
        @SeedFactory.register("user_doug")
        class DougUserSeed(SeedItem):
            dependencies = []

            async def create(self, provider: MemoryProvider) -> str:
                return await provider.store(
                    content="User: Doug",
                    metadata={"memory_type": "USER_PROFILE"}
                )
    """

    dependencies: ClassVar[list[str]] = []

    @abstractmethod
    async def create(self, provider: "MemoryProvider", **deps: Any) -> str:
        """Create seed data using REAL MemoryProvider.

        Args:
            provider: Real MemoryProvider instance (not wrapper)
            **deps: Resolved dependency results (dep_name -> result)

        Returns:
            Seed item identifier or result
        """
        ...


@dataclass
class SeedSet:
    """Collection of seed items for test scenario.

    Example:
        USER_WITH_PETS = SeedSet(
            "user_with_pets",
            ["user_doug", "doug_cats_fact"]
        )
    """

    name: str
    items: list[str]  # Seed IDs to include

    async def apply(self, provider: "MemoryProvider") -> dict[str, Any]:
        """Apply all seeds using REAL MemoryProvider.

        Args:
            provider: Real MemoryProvider (not wrapper)

        Returns:
            Mapping of seed_id -> created result
        """
        factory = SeedFactory()
        return await factory.create_all(self.items, provider)


class SeedFactory:
    """Registry and executor for seed items.

    IMPORTANT: Uses instance-based registry for parallel test safety.
    Global seeds are registered via decorator, then copied to each instance.
    """

    # Global registry for @SeedFactory.register() decorated classes
    _global_seed_classes: ClassVar[dict[str, type[SeedItem]]] = {}

    def __init__(self):
        # Instance-based registry - each test gets its own copy
        self._registry: dict[str, SeedItem] = {}

        # Copy global seeds to this instance
        for seed_id, seed_class in self._global_seed_classes.items():
            self._registry[seed_id] = seed_class()

    @classmethod
    def register(cls, seed_id: str):
        """Decorator to register seed item class globally.

        Seeds are registered to global registry, then copied to each
        SeedFactory instance. This enables parallel test safety.

        Usage:
            @SeedFactory.register("user_doug")
            class DougUserSeed(SeedItem):
                ...
        """
        def decorator(seed_class: type[SeedItem]) -> type[SeedItem]:
            cls._global_seed_classes[seed_id] = seed_class
            logger.debug(f"Registered seed class: {seed_id}")
            return seed_class
        return decorator

    def register_instance(self, seed_id: str, seed: SeedItem) -> None:
        """Register seed instance to this factory only.

        Use for test-specific seeds that shouldn't be global.
        """
        self._registry[seed_id] = seed

    def _topological_sort(self, seed_ids: list[str]) -> list[str]:
        """Sort seeds by dependencies (topological order).

        Args:
            seed_ids: Seed IDs to sort

        Returns:
            Sorted list (dependencies before dependents)

        Raises:
            CircularDependencyError: If circular dependency found
        """
        # Build adjacency list
        graph: dict[str, list[str]] = {}
        in_degree: dict[str, int] = {}

        for seed_id in seed_ids:
            if seed_id not in self._registry:
                raise ValueError(f"Seed not registered: {seed_id}")

            seed = self._registry[seed_id]
            graph[seed_id] = seed.dependencies
            in_degree[seed_id] = 0

        # Count in-degrees
        for seed_id in seed_ids:
            for dep in graph[seed_id]:
                if dep in in_degree:
                    in_degree[dep] += 1

        # Kahn's algorithm
        queue = [sid for sid in seed_ids if in_degree[sid] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for seed_id in seed_ids:
                if current in graph[seed_id]:
                    in_degree[seed_id] -= 1
                    if in_degree[seed_id] == 0:
                        queue.append(seed_id)

        if len(result) != len(seed_ids):
            raise CircularDependencyError(
                f"Circular dependency detected in seeds: {seed_ids}"
            )

        return result

    async def create_all(
        self,
        seed_ids: list[str],
        provider: "MemoryProvider"
    ) -> dict[str, Any]:
        """Create all seeds in dependency order with rollback on failure.

        Args:
            seed_ids: Seeds to create
            provider: Real MemoryProvider (not wrapper)

        Returns:
            Mapping of seed_id -> creation result

        Raises:
            Exception: Re-raises creation error after cleanup attempt
        """
        sorted_ids = self._topological_sort(seed_ids)
        results = {}
        created_memory_ids = []  # Track for rollback

        try:
            for seed_id in sorted_ids:
                seed = self._registry[seed_id]

                # Resolve dependencies
                dep_results = {
                    dep: results[dep]
                    for dep in seed.dependencies
                    if dep in results
                }

                # Create seed using REAL provider
                logger.info(f"Creating seed: {seed_id}")
                result = await seed.create(provider, **dep_results)
                results[seed_id] = result

                # Track memory IDs for rollback (if result looks like an ID)
                if isinstance(result, str) and result:
                    created_memory_ids.append(result)

            return results

        except Exception as e:
            # Rollback: attempt to delete created memories
            if created_memory_ids:
                logger.warning(
                    f"Seed creation failed at {seed_id}. "
                    f"Rolling back {len(created_memory_ids)} created memories..."
                )
                for memory_id in reversed(created_memory_ids):
                    try:
                        await provider.delete(memory_id)
                        logger.debug(f"Rolled back: {memory_id}")
                    except Exception as rollback_error:
                        logger.warning(f"Rollback failed for {memory_id}: {rollback_error}")

            raise  # Re-raise original error
```

**Testing Approach:**
- Test dependency resolution (topological sort)
- Test circular dependency detection
- Test seed creation order
- Test result propagation

---

### 1.2 Module: `src/draagon_ai/testing/database.py`

**Purpose:** Neo4j test database lifecycle management ONLY

**Design Decision: Lifecycle-Only**

TestDatabase is ONLY for lifecycle management. It does NOT:
- Wrap MemoryProvider with helper methods
- Provide convenience methods like `create_user()` or `store_memory()`
- Act as a facade over the real provider

Tests use the **real MemoryProvider directly**. This ensures:
1. **Production API coverage** - Tests validate actual production interfaces
2. **No leaky abstractions** - Helper methods don't mask real behavior
3. **Real bugs surface** - If MemoryProvider has issues, tests find them

**Key Classes:**

```python
from dataclasses import dataclass
from neo4j import AsyncGraphDatabase, AsyncDriver
import logging

logger = logging.getLogger(__name__)


@dataclass
class Neo4jMemoryConfig:
    """Config for creating MemoryProvider instances."""
    uri: str
    username: str
    password: str
    database: str


@dataclass
class TestDatabase:
    """Neo4j test database lifecycle manager ONLY.

    Does NOT wrap or provide helper methods.
    Tests use the REAL MemoryProvider directly.

    Example:
        db = TestDatabase()
        await db.initialize()

        # Get config for creating REAL provider
        config = db.get_config()
        provider = LayeredMemoryProvider.from_config(config)

        # Use REAL provider in tests
        memory_id = await provider.store(content="...", metadata={...})

        # Cleanup
        await db.clear()
        await db.close()
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "draagon-ai-2025"
    database_name: str = "neo4j_test"

    _driver: AsyncDriver | None = None

    async def initialize(self) -> None:
        """Initialize Neo4j connection and create test database."""
        self._driver = AsyncGraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_username, self.neo4j_password)
        )

        # Create test database if needed
        async with self._driver.session(database="system") as session:
            result = await session.run(
                "SHOW DATABASES YIELD name WHERE name = $name",
                name=self.database_name
            )
            exists = await result.single()

            if not exists:
                await session.run(
                    f"CREATE DATABASE {self.database_name}"
                )
                logger.info(f"Created test database: {self.database_name}")

    async def clear(self) -> None:
        """Delete all nodes and relationships in test database."""
        if not self._driver:
            raise RuntimeError("Database not initialized - call initialize() first")

        async with self._driver.session(database=self.database_name) as session:
            await session.run("MATCH (n) DETACH DELETE n")
            logger.debug(f"Cleared test database: {self.database_name}")

    async def close(self) -> None:
        """Close driver connection."""
        if self._driver:
            await self._driver.close()
            logger.debug("Closed test database connection")

    def get_config(self) -> Neo4jMemoryConfig:
        """Get config for creating REAL providers.

        Returns config - tests create their own providers.
        Does NOT cache or wrap providers.
        """
        return Neo4jMemoryConfig(
            uri=self.neo4j_uri,
            username=self.neo4j_username,
            password=self.neo4j_password,
            database=self.database_name,
        )

    async def verify_connection(self) -> None:
        """Verify Neo4j connection is healthy.

        Call this before tests to get clear error messages.

        Raises:
            RuntimeError: If Neo4j is not reachable with helpful message
        """
        if not self._driver:
            raise RuntimeError("Database not initialized - call initialize() first")

        try:
            async with self._driver.session(database=self.database_name) as session:
                await session.run("RETURN 1")
        except Exception as e:
            raise RuntimeError(
                f"Neo4j connection failed.\n"
                f"URI: {self.neo4j_uri}\n"
                f"Database: {self.database_name}\n"
                f"Is Neo4j running? Try: docker ps | grep neo4j\n"
                f"Error: {e}"
            ) from e
```

**Testing Approach:**
- Test database creation/cleanup
- Test get_config() returns correct values
- Test verify_connection() provides helpful errors
- Test concurrent access safety

---

### 1.3 Module: `src/draagon_ai/testing/fixtures.py` (EXTEND)

**Purpose:** Add new pytest fixtures for framework

**New Fixtures to Add:**

```python
import pytest
from typing import AsyncIterator
from .database import TestDatabase
from .seeds import SeedFactory
from draagon_ai.memory import MemoryProvider


@pytest.fixture(scope="session")
async def test_database() -> AsyncIterator[TestDatabase]:
    """Session-scoped test database lifecycle manager.

    Creates Neo4j test database connection once per session.
    Each test gets a clean database via clean_database fixture.
    """
    db = TestDatabase()
    await db.initialize()
    yield db
    await db.close()


@pytest.fixture
async def clean_database(test_database: TestDatabase) -> AsyncIterator[TestDatabase]:
    """Function-scoped clean database.

    CLEANUP POLICY:
    - Clears BEFORE each test (ensures clean slate)
    - Does NOT clear AFTER test (preserves state for debugging)

    Why not clear after?
    - If test fails, you can inspect database state
    - Next test clears before running anyway
    - Faster execution (one clear per test, not two)

    For debugging a specific test, use the Neo4j browser at:
    http://localhost:7474 (database: neo4j_test)
    """
    await test_database.clear()
    await test_database.verify_connection()  # Fail fast with helpful error
    yield test_database
    # Intentionally NOT clearing after - allows post-failure inspection


@pytest.fixture
async def preserved_database(test_database: TestDatabase) -> AsyncIterator[TestDatabase]:
    """Database that preserves state across tests (for debugging).

    WARNING: Only use for debugging. Tests should normally use clean_database.
    """
    await test_database.verify_connection()
    yield test_database


@pytest.fixture
async def memory_provider(clean_database: TestDatabase) -> AsyncIterator[MemoryProvider]:
    """REAL MemoryProvider for tests.

    Tests use this directly - no wrappers.
    This is the production API.
    """
    from draagon_ai.memory.providers.layered import LayeredMemoryProvider

    config = clean_database.get_config()
    provider = LayeredMemoryProvider.from_config(config)
    await provider.initialize()
    yield provider
    await provider.close()


@pytest.fixture
def seed_factory() -> SeedFactory:
    """Create fresh seed factory for each test.

    Each test gets its own factory instance with copies of global seeds.
    This enables parallel test execution without registry conflicts.
    """
    return SeedFactory()  # Fresh instance, copies global seeds


@pytest.fixture
async def seed(memory_provider: MemoryProvider, seed_factory: SeedFactory):
    """Seed applicator with REAL MemoryProvider.

    Usage in test:
        async def test_recall_pets(seed, memory_provider):
            await seed.apply(USER_WITH_PETS, memory_provider)
            # Test continues with seeded data
    """
    class SeedApplicator:
        def __init__(self, provider: MemoryProvider, factory: SeedFactory):
            self.provider = provider
            self.factory = factory

        async def apply(self, seed_set, provider: MemoryProvider | None = None):
            # Use provided provider or default
            p = provider or self.provider
            return await seed_set.apply(p)

    return SeedApplicator(memory_provider, seed_factory)
```

---

### 1.4 Module: `src/draagon_ai/testing/__init__.py` (EXTEND)

**Purpose:** Export new classes

```python
# Existing exports
from .cache import UnifiedCacheManager, CacheMode, CacheConfig
from .mocks import LLMMock, HTTPMock, EmbeddingMock, ServiceMock
from .modes import TestMode, TestContext, test_mode, get_test_context
from .fixtures import *

# NEW: FR-009 exports
from .seeds import SeedItem, SeedFactory, SeedSet, CircularDependencyError
from .database import TestDatabase

__all__ = [
    # Existing...
    "UnifiedCacheManager", "CacheMode", "CacheConfig",
    "LLMMock", "HTTPMock", "EmbeddingMock", "ServiceMock",
    "TestMode", "TestContext", "test_mode", "get_test_context",

    # NEW
    "SeedItem", "SeedFactory", "SeedSet", "CircularDependencyError",
    "TestDatabase",
]
```

---

### 1.5 Tests: `tests/framework/test_seeds.py`

**Purpose:** Unit tests for seed factory

```python
import pytest
from draagon_ai.testing import (
    SeedItem, SeedFactory, SeedSet, CircularDependencyError
)
from draagon_ai.memory import MemoryProvider


class ParentSeed(SeedItem):
    dependencies = []

    async def create(self, provider: MemoryProvider) -> str:
        # Use REAL provider API
        return await provider.store(
            content="Parent seed",
            metadata={"memory_type": "TEST"}
        )


class ChildSeed(SeedItem):
    dependencies = ["parent"]

    async def create(self, provider: MemoryProvider, parent: str) -> str:
        # Use REAL provider API
        return await provider.store(
            content=f"Child of {parent}",
            metadata={"memory_type": "TEST", "parent": parent}
        )


@pytest.mark.asyncio
async def test_dependency_resolution():
    """Test seeds created in dependency order."""
    factory = SeedFactory()
    factory._registry = {
        "parent": ParentSeed(),
        "child": ChildSeed(),
    }

    sorted_ids = factory._topological_sort(["child", "parent"])
    assert sorted_ids == ["parent", "child"]


@pytest.mark.asyncio
async def test_circular_dependency_detection():
    """Test circular dependencies raise error."""
    class ASeed(SeedItem):
        dependencies = ["b"]
        async def create(self, provider, **deps): return "a"

    class BSeed(SeedItem):
        dependencies = ["a"]
        async def create(self, provider, **deps): return "b"

    factory = SeedFactory()
    factory._registry = {"a": ASeed(), "b": BSeed()}

    with pytest.raises(CircularDependencyError):
        factory._topological_sort(["a", "b"])


@pytest.mark.asyncio
async def test_seed_set_application(memory_provider):
    """Test SeedSet applies all seeds using REAL provider."""
    # Register test seeds
    @SeedFactory.register("test_parent")
    class TestParent(SeedItem):
        async def create(self, provider: MemoryProvider) -> str:
            return await provider.store(
                content="Parent",
                metadata={"memory_type": "TEST"}
            )

    @SeedFactory.register("test_child")
    class TestChild(SeedItem):
        dependencies = ["test_parent"]
        async def create(self, provider: MemoryProvider, test_parent: str) -> str:
            return await provider.store(
                content=f"Child of {test_parent}",
                metadata={"memory_type": "TEST"}
            )

    # Apply seed set using REAL provider
    seed_set = SeedSet("test", ["test_parent", "test_child"])
    results = await seed_set.apply(memory_provider)

    # Verify both seeds were created
    assert "test_parent" in results
    assert "test_child" in results

    # Verify data was stored in REAL database
    parent_memory = await memory_provider.get(results["test_parent"])
    assert parent_memory is not None
    assert "Parent" in parent_memory.content
```

---

## Phase 2: Test Sequences (Week 2)

### 2.1 Module: `src/draagon_ai/testing/sequences.py`

**Purpose:** Multi-step tests with shared state

**Implementation:**

```python
from abc import ABC
from typing import Callable, Any
from functools import wraps
import logging

logger = logging.getLogger(__name__)


def step(order: int, depends_on: str | None = None):
    """Decorator to mark test sequence steps.

    Args:
        order: Step execution order (1, 2, 3...)
        depends_on: Optional step name this depends on

    Usage:
        class TestFlow(TestSequence):
            @step(1)
            async def test_first(self): ...

            @step(2, depends_on="test_first")
            async def test_second(self): ...
    """
    def decorator(func: Callable) -> Callable:
        func._step_order = order  # type: ignore
        func._step_depends_on = depends_on  # type: ignore

        @wraps(func)
        async def wrapper(*args, **kwargs):
            logger.info(f"Executing step {order}: {func.__name__}")
            return await func(*args, **kwargs)

        return wrapper
    return decorator


class TestSequence(ABC):
    """Base class for multi-step test sequences.

    Database persists across steps within a sequence.
    Use @step decorator to define execution order.

    Example:
        class TestLearningFlow(TestSequence):
            @step(1)
            async def test_unknown(self, agent):
                response = await agent.process("When is my birthday?")
                assert response.confidence < 0.5

            @step(2, depends_on="test_unknown")
            async def test_learn(self, agent):
                response = await agent.process("My birthday is March 15")
                assert "march 15" in response.answer.lower()
    """

    @classmethod
    def get_steps(cls) -> list[tuple[int, str, Callable]]:
        """Get all steps in execution order.

        Returns:
            List of (order, name, method) tuples sorted by order
        """
        steps = []
        for name in dir(cls):
            method = getattr(cls, name)
            if callable(method) and hasattr(method, "_step_order"):
                steps.append((method._step_order, name, method))

        return sorted(steps, key=lambda x: x[0])

    @classmethod
    def validate_dependencies(cls) -> None:
        """Validate step dependencies are resolvable.

        Raises:
            ValueError: If dependency references non-existent step
        """
        steps = cls.get_steps()
        step_names = {name for _, name, _ in steps}

        for order, name, method in steps:
            depends_on = getattr(method, "_step_depends_on", None)
            if depends_on and depends_on not in step_names:
                raise ValueError(
                    f"Step '{name}' depends on non-existent step '{depends_on}'"
                )
```

**Pytest Integration:**

```python
# tests/conftest.py addition

import pytest

def pytest_collection_modifyitems(items):
    """Auto-detect and order test sequences."""
    for item in items:
        # Check if test is part of a sequence
        if hasattr(item.cls, "get_steps"):
            # Mark as sequence test
            item.add_marker(pytest.mark.sequence)

            # Ensure database persists across sequence
            item.add_marker(pytest.mark.usefixtures("sequence_database"))


@pytest.fixture(scope="class")
async def sequence_database(test_database):
    """Class-scoped database for test sequences.

    Clears BEFORE sequence, persists DURING sequence.
    """
    await test_database.clear()
    yield test_database
```

---

## Phase 3: LLM-as-Judge (Week 2)

### 3.1 Module: `src/draagon_ai/testing/evaluation.py`

**Purpose:** LLM-based semantic evaluation

**Implementation:**

```python
from dataclasses import dataclass
from typing import Protocol
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Result from LLM-as-judge evaluation."""
    correct: bool
    reasoning: str
    confidence: float
    metrics: dict[str, float] | None = None


class LLMProvider(Protocol):
    """Protocol for LLM providers used in evaluation."""
    async def chat(self, messages: list[dict], **kwargs) -> str: ...


class AgentEvaluator:
    """LLM-based evaluation of agent responses.

    Uses XML prompts to evaluate correctness, coherence, helpfulness.
    Includes retry logic for resilience.

    Example:
        evaluator = AgentEvaluator(llm_provider)
        result = await evaluator.evaluate_correctness(
            query="What are my cats' names?",
            expected_outcome="Lists: Whiskers, Mittens, Shadow",
            actual_response=agent_response
        )
    """

    def __init__(self, llm: LLMProvider, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries

    async def _call_llm_with_retry(self, messages: list[dict]) -> str:
        """Call LLM with exponential backoff retry.

        Args:
            messages: Chat messages to send

        Returns:
            LLM response string

        Raises:
            RuntimeError: If all retries exhausted
        """
        import asyncio

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return await self.llm.chat(messages)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 1s, 2s, 4s
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)

        raise RuntimeError(
            f"LLM evaluation failed after {self.max_retries} attempts.\n"
            f"Last error: {last_error}"
        )

    async def evaluate_correctness(
        self,
        query: str,
        expected_outcome: str,
        actual_response: str,
    ) -> EvaluationResult:
        """Evaluate if response achieves expected outcome.

        Uses LLM to semantically compare expected vs actual.
        Includes retry logic for resilience.

        Args:
            query: Original user query
            expected_outcome: What should happen
            actual_response: Agent's actual response

        Returns:
            Evaluation result with correctness and reasoning

        Raises:
            RuntimeError: If LLM evaluation fails after retries
        """
        prompt = f"""Evaluate if this agent response achieves the expected outcome.

<evaluation>
  <query>{query}</query>
  <expected_outcome>{expected_outcome}</expected_outcome>
  <actual_response>{actual_response}</actual_response>
</evaluation>

Analyze whether the actual response satisfies the expected outcome.
Consider semantic equivalence, not exact wording.

Respond with:
<result>
  <correct>true/false</correct>
  <reasoning>Brief explanation of why correct or incorrect</reasoning>
  <confidence>0.0-1.0</confidence>
</result>"""

        response = await self._call_llm_with_retry([
            {"role": "user", "content": prompt}
        ])

        return self._parse_evaluation(response)

    async def evaluate_coherence(self, response: str) -> float:
        """Score response coherence (0.0-1.0).

        Args:
            response: Agent response to evaluate

        Returns:
            Coherence score (0-1, higher = more coherent)
        """
        prompt = f"""Rate the coherence of this response on a scale of 0.0 to 1.0:

<response>{response}</response>

Coherence means:
- Logical flow
- Grammatical correctness
- Clarity of communication

Respond with:
<score>
  <value>0.0-1.0</value>
  <reasoning>Brief explanation</reasoning>
</score>"""

        result = await self.llm.chat([
            {"role": "user", "content": prompt}
        ])

        # Parse score
        match = re.search(r"<value>([\d.]+)</value>", result)
        if match:
            return float(match.group(1))
        return 0.5  # Default if parsing fails

    async def evaluate_helpfulness(self, query: str, response: str) -> float:
        """Score response helpfulness (0.0-1.0).

        Args:
            query: User query
            response: Agent response

        Returns:
            Helpfulness score (0-1, higher = more helpful)
        """
        prompt = f"""Rate how helpful this response is for the user's query:

<query>{query}</query>
<response>{response}</response>

Helpfulness means:
- Directly addresses query
- Provides useful information
- Appropriate level of detail

Respond with:
<score>
  <value>0.0-1.0</value>
  <reasoning>Brief explanation</reasoning>
</score>"""

        result = await self.llm.chat([
            {"role": "user", "content": prompt}
        ])

        # Parse score
        match = re.search(r"<value>([\d.]+)</value>", result)
        if match:
            return float(match.group(1))
        return 0.5  # Default

    def _parse_evaluation(self, xml_response: str) -> EvaluationResult:
        """Parse XML evaluation response.

        Args:
            xml_response: LLM response with <result> tags

        Returns:
            Parsed evaluation result
        """
        correct_match = re.search(
            r"<correct>(true|false)</correct>",
            xml_response,
            re.IGNORECASE
        )
        reasoning_match = re.search(
            r"<reasoning>(.+?)</reasoning>",
            xml_response,
            re.DOTALL
        )
        confidence_match = re.search(
            r"<confidence>([\d.]+)</confidence>",
            xml_response
        )

        correct = (
            correct_match.group(1).lower() == "true"
            if correct_match else False
        )
        reasoning = (
            reasoning_match.group(1).strip()
            if reasoning_match else "No reasoning provided"
        )
        confidence = (
            float(confidence_match.group(1))
            if confidence_match else 0.5
        )

        return EvaluationResult(
            correct=correct,
            reasoning=reasoning,
            confidence=confidence
        )
```

**Fixture Addition:**

```python
# tests/conftest.py

@pytest.fixture
def evaluator(llm_provider) -> AgentEvaluator:
    """Get agent evaluator with real LLM."""
    from draagon_ai.testing.evaluation import AgentEvaluator
    return AgentEvaluator(llm_provider)
```

---

## Phase 4: App Profiles (Week 3)

### 4.1 Module: `src/draagon_ai/testing/profiles.py`

**Purpose:** Configurable agent profiles for testing

**Implementation:**

```python
from dataclasses import dataclass
from enum import Enum


class ToolSet(Enum):
    """Pre-defined tool collections for agent profiles."""
    MINIMAL = "minimal"          # Just answer tool
    BASIC = "basic"              # answer, get_time, calculate
    HOME_AUTOMATION = "home"     # Basic + HA tools
    FULL = "full"                # All registered tools


@dataclass
class AppProfile:
    """Agent configuration profile for testing.

    Defines personality, tools, memory config for test agent.

    Example:
        MINIMAL_AGENT = AppProfile(
            name="minimal",
            personality="You are a helpful assistant.",
            tool_set=ToolSet.MINIMAL,
            memory_config={"working_ttl": 300}
        )
    """

    name: str
    personality: str
    tool_set: ToolSet
    memory_config: dict
    llm_model_tier: str = "standard"


# Pre-defined profiles
MINIMAL_AGENT = AppProfile(
    name="minimal",
    personality="You are a helpful assistant. Be concise.",
    tool_set=ToolSet.MINIMAL,
    memory_config={"working_ttl": 300},
)

BASIC_AGENT = AppProfile(
    name="basic",
    personality="You are a helpful assistant with basic tools.",
    tool_set=ToolSet.BASIC,
    memory_config={"working_ttl": 300, "enable_learning": True},
)

ROXY_VOICE = AppProfile(
    name="roxy",
    personality="""You are Roxy, a friendly home assistant.
You help with smart home control, scheduling, and information.""",
    tool_set=ToolSet.HOME_AUTOMATION,
    memory_config={
        "working_ttl": 300,
        "enable_learning": True,
        "curiosity_intensity": 0.8,
    },
    llm_model_tier="fast",
)
```

**Agent Factory Fixture:**

```python
# tests/conftest.py

@pytest.fixture
async def agent_factory(llm_provider, test_database):
    """Factory for creating configured test agents."""
    from draagon_ai.testing.profiles import ToolSet
    from draagon_ai.orchestration import Agent
    from draagon_ai.tools import ToolRegistry

    class AgentFactory:
        def __init__(self, llm, db):
            self.llm = llm
            self.db = db

        async def create(self, profile):
            """Create agent from profile."""
            # Get tools for profile
            registry = ToolRegistry()
            if profile.tool_set == ToolSet.MINIMAL:
                # Only answer tool (no external tools)
                tools = []
            elif profile.tool_set == ToolSet.BASIC:
                tools = self._get_basic_tools(registry)
            elif profile.tool_set == ToolSet.HOME_AUTOMATION:
                tools = self._get_home_tools(registry)
            else:  # FULL
                tools = registry.get_all_tools()

            # Create agent
            agent = Agent(
                personality=profile.personality,
                llm=self.llm,
                memory=self.db.get_memory_provider(),
                tools=tools,
                **profile.memory_config
            )

            return agent

        def _get_basic_tools(self, registry):
            return [
                registry.get("get_time"),
                registry.get("calculate"),
            ]

        def _get_home_tools(self, registry):
            return self._get_basic_tools(registry) + [
                registry.get("control_device"),
                registry.get("get_device_state"),
            ]

    return AgentFactory(llm_provider, test_database)
```

---

## Phase 5: Integration Tests (Week 3-4)

### 5.1 Example: `tests/integration/test_memory_fr009.py`

```python
import pytest
from draagon_ai.testing import SeedSet, SeedFactory, SeedItem
from draagon_ai.memory import MemoryProvider, MemoryType


# Define seed data - uses REAL MemoryProvider
@SeedFactory.register("test_user")
class TestUserSeed(SeedItem):
    async def create(self, provider: MemoryProvider) -> str:
        # Use REAL provider API directly
        return await provider.store(
            content="User profile: TestUser",
            metadata={
                "memory_type": "USER_PROFILE",
                "user_name": "TestUser",
            }
        )


@SeedFactory.register("user_preference")
class UserPreferenceSeed(SeedItem):
    dependencies = ["test_user"]

    async def create(self, provider: MemoryProvider, test_user: str) -> str:
        # Use REAL provider API directly
        return await provider.store(
            content="User prefers dark mode",
            metadata={
                "memory_type": MemoryType.PREFERENCE.value,
                "related_memory": test_user,
                "importance": 0.9,
            }
        )


# Seed set
BASIC_USER = SeedSet("basic_user", ["test_user", "user_preference"])


@pytest.mark.memory_integration
@pytest.mark.smoke
async def test_fact_storage_and_recall(agent_factory, memory_provider, seed, evaluator):
    """Test agent stores and recalls facts.

    Uses REAL MemoryProvider - no wrappers.
    """
    # Setup - seed using REAL provider
    await seed.apply(BASIC_USER, memory_provider)
    agent = await agent_factory.create(MINIMAL_AGENT)

    # Agent learns fact
    response1 = await agent.process("Remember that I like pizza")

    # Agent recalls fact
    response2 = await agent.process("What food do I like?")

    # Semantic evaluation (NOT exact string match)
    result = await evaluator.evaluate_correctness(
        query="What food do I like?",
        expected_outcome="Agent should mention pizza",
        actual_response=response2.answer
    )

    assert result.correct, f"Failed: {result.reasoning}"
    assert result.confidence > 0.7
```

---

## Testing Strategy

### Unit Tests (Framework)
- `tests/framework/test_seeds.py` - Seed factory logic
- `tests/framework/test_sequences.py` - Sequence execution
- `tests/framework/test_evaluation.py` - Evaluator parsing

### Integration Tests (Using Framework)
- `tests/integration/test_memory_fr009.py` - Memory operations
- `tests/integration/test_learning_fr009.py` - Learning extraction
- `tests/integration/test_beliefs_fr009.py` - Belief reconciliation
- `tests/integration/test_multi_agent_fr009.py` - Multi-agent coordination

### Example Sequences
- `tests/sequences/test_learning_flow.py` - Multi-step learning
- `tests/sequences/test_belief_reconciliation.py` - Belief conflict resolution

---

## Architectural Compliance

### ✅ LLM-First Architecture
- AgentEvaluator uses LLM for semantic evaluation (no regex)
- Outcome testing validates results, not processes

### ✅ XML Output Format
- All evaluation prompts use XML
- Parsing uses XML element extraction

### ✅ Protocol-Based Design
- TestDatabase is lifecycle-only (initialize, clear, close)
- Seeds use REAL MemoryProvider directly (no wrappers)
- AgentEvaluator uses LLMProvider protocol
- Tests validate production APIs, not test-specific abstractions

### ✅ Pragmatic Async
- Async for I/O (database, LLM evaluation)
- Sync for pure computation (dependency resolution, parsing)

### ✅ Test Outcomes, Not Processes
- Framework makes outcome testing easy
- LLM-as-judge for semantic validation
- Process testing discouraged

### ✅ Real Production Code
- Seeds create data using REAL MemoryProvider
- No helper methods masking production behavior
- If a seed works, production code will work

---

## Success Criteria

| Metric | Target | Validation |
|--------|--------|------------|
| Framework coverage | 80%+ unit tests | pytest --cov |
| Integration coverage | 5+ cognitive services | Manual count |
| Seed reuse | 50%+ tests share seeds | Seed registry stats |
| LLM-as-judge accuracy | >90% vs human | Manual validation |

---

**Document Status:** Ready for Implementation
**Next Step:** Begin Phase 1 implementation (seeds.py + database.py)
