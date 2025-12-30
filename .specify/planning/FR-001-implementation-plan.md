# Implementation Plan: FR-001 Shared Cognitive Working Memory

**Feature:** Shared Cognitive Working Memory for Multi-Agent Coordination
**Specification:** `.specify/requirements/FR-001-shared-cognitive-working-memory.md`
**Status:** Ready for Implementation
**Estimated Duration:** 10 days
**Dependencies:** None (Foundation)

---

## Executive Summary

Implement a psychologically-grounded shared working memory system that allows multiple agents to coordinate through attention-weighted observations. This is the foundation for cognitive swarm architecture, providing conflict detection, capacity management (Miller's Law: 7±2), and role-based context filtering.

**Key Innovation:** Unlike simple `dict` context sharing in current `TaskContext.working_memory`, this provides genuine cognitive capabilities: attention decay, semantic conflict detection, and heterogeneous agent-specific views.

---

## Implementation Strategy

### Phase 1: Core Data Structures (Days 1-2)
**Goal:** Implement immutable observation storage with all metadata

### Phase 2: Capacity & Eviction (Days 3-4)
**Goal:** Miller's Law enforcement with attention-based eviction

### Phase 3: Conflict Detection (Days 5-6)
**Goal:** Semantic conflict detection (placeholder → embeddings)

### Phase 4: Attention & Access (Day 7)
**Goal:** Decay, boost, access tracking

### Phase 5: Role Filtering (Day 8)
**Goal:** Context retrieval filtered by AgentRole

### Phase 6: Testing & Integration (Days 9-10)
**Goal:** Comprehensive test suite + TaskContext integration

---

## Module Structure

### New Module: `src/draagon_ai/orchestration/shared_memory.py`

**Rationale:** Place in `orchestration/` because it's tightly coupled with multi-agent orchestration. Not in `memory/` because that's for 4-layer persistent memory (episodic, semantic, metacognitive). This is task-scoped working memory.

```python
# src/draagon_ai/orchestration/shared_memory.py

"""Shared cognitive working memory for multi-agent coordination.

Provides:
- SharedObservation: Immutable observation with attribution
- SharedWorkingMemory: Task-scoped memory with Miller's Law capacity
- Conflict detection via semantic similarity
- Role-based context filtering
- Attention weighting and decay

Usage:
    shared_memory = SharedWorkingMemory(task_id="task_123")

    # Agent A adds observation
    obs = await shared_memory.add_observation(
        content="Meeting is at 3pm",
        source_agent_id="agent_a",
        attention_weight=0.8,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    # Agent B reads context (filtered by role)
    context = await shared_memory.get_context_for_agent(
        agent_id="agent_b",
        role=AgentRole.CRITIC,
        max_items=7,  # Miller's Law
    )
"""
```

**File Breakdown:**
- ~500 lines total
- 3 classes: `SharedObservation`, `SharedWorkingMemoryConfig`, `SharedWorkingMemory`
- 15+ methods across capacity, conflict, attention, retrieval
- Full type hints with `@dataclass` for data structures

---

## Data Structures

### 1. SharedObservation (Immutable)

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

@dataclass(frozen=True)  # Immutable
class SharedObservation:
    """An observation in shared working memory.

    Observations are immutable once created. They represent what an agent
    observed or concluded during task execution.

    Attributes:
        observation_id: Unique identifier (UUID)
        content: The observation content
        source_agent_id: Which agent made this observation
        timestamp: When observation was created
        attention_weight: Current attention (0-1), decays over time
        confidence: Agent's confidence in this observation (0-1)
        is_belief_candidate: Should this become a belief?
        belief_type: Type if belief candidate (FACT, SKILL, PREFERENCE, etc.)
        conflicts_with: List of observation IDs that conflict
        accessed_by: Set of agent IDs that have read this
        access_count: Number of times accessed
    """

    observation_id: str
    content: str
    source_agent_id: str
    timestamp: datetime

    # Cognitive properties
    attention_weight: float = 0.5  # 0-1
    confidence: float = 1.0  # 0-1

    # Belief tracking
    is_belief_candidate: bool = False
    belief_type: str | None = None  # "FACT", "SKILL", "PREFERENCE", etc.

    # Conflict tracking
    conflicts_with: list[str] = field(default_factory=list)

    # Access tracking (mutable fields require special handling)
    accessed_by: set[str] = field(default_factory=set)
    access_count: int = 0

    def __post_init__(self):
        """Validate observation after creation."""
        if not 0 <= self.attention_weight <= 1:
            raise ValueError(f"attention_weight must be 0-1, got {self.attention_weight}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")
```

**Note on Immutability:** `frozen=True` makes the dataclass immutable, but `accessed_by` (set) and `access_count` need updates. We'll use `object.__setattr__()` in SharedWorkingMemory methods to update these fields.

### 2. SharedWorkingMemoryConfig

```python
@dataclass
class SharedWorkingMemoryConfig:
    """Configuration for shared working memory.

    Defaults based on cognitive psychology research:
    - max_items_per_agent: 7 (Miller's Law: 7±2)
    - max_total_items: 50 (room for ~7 agents)
    - attention_decay_factor: 0.9 (10% decay per sync)
    - conflict_threshold: 0.7 (semantic similarity for conflicts)
    - sync_interval_iterations: 3 (barrier sync frequency)
    """

    # Capacity constraints
    max_items_per_agent: int = 7  # Miller's Law: 7±2
    max_total_items: int = 50  # Global capacity

    # Attention management
    attention_decay_factor: float = 0.9  # Multiply by this on decay

    # Conflict detection
    conflict_threshold: float = 0.7  # Semantic similarity threshold

    # Synchronization
    sync_interval_iterations: int = 3  # Periodic sync frequency
```

### 3. SharedWorkingMemory (Main Class)

```python
from typing import Protocol
import asyncio
import uuid

class EmbeddingProvider(Protocol):
    """Protocol for embedding providers (optional).

    Used for semantic similarity in conflict detection.
    If not provided, uses simple heuristic (same belief_type).
    """
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity (0-1) between texts."""
        ...


class SharedWorkingMemory:
    """Task-scoped working memory for multi-agent coordination.

    Provides:
    - Observation storage with source attribution
    - Miller's Law capacity management (7±2 per agent)
    - Semantic conflict detection
    - Attention weighting and decay
    - Role-based context filtering
    - Concurrent access safety

    Example:
        memory = SharedWorkingMemory("task_123")

        # Agent A observes
        await memory.add_observation(
            content="User prefers coffee in the morning",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Agent B gets context (sees what A observed)
        context = await memory.get_context_for_agent(
            agent_id="agent_b",
            role=AgentRole.RESEARCHER,
        )
    """

    def __init__(
        self,
        task_id: str,
        config: SharedWorkingMemoryConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.task_id = task_id
        self.config = config or SharedWorkingMemoryConfig()
        self.embedding_provider = embedding_provider

        # Storage
        self._observations: dict[str, SharedObservation] = {}
        self._conflicts: list[tuple[str, str, str]] = []  # (obs_a, obs_b, reason)
        self._agent_views: dict[str, list[str]] = {}  # agent_id -> [obs_ids]

        # Concurrency control
        self._global_lock = asyncio.Lock()
        self._observation_locks: dict[str, asyncio.Lock] = {}

    # Methods defined below...
```

---

## Method Implementations

### Core Operations

```python
async def add_observation(
    self,
    content: str,
    source_agent_id: str,
    *,
    attention_weight: float = 0.5,
    confidence: float = 1.0,
    is_belief_candidate: bool = False,
    belief_type: str | None = None,
) -> SharedObservation:
    """Add observation with automatic conflict detection.

    Steps:
    1. Create observation with UUID
    2. Acquire global lock
    3. Check for conflicts with existing observations
    4. Ensure capacity (evict if needed)
    5. Store observation
    6. Update agent view
    7. Release lock

    Returns:
        The created observation
    """
    async with self._global_lock:
        observation = SharedObservation(
            observation_id=str(uuid.uuid4()),
            content=content,
            source_agent_id=source_agent_id,
            timestamp=datetime.now(),
            attention_weight=attention_weight,
            confidence=confidence,
            is_belief_candidate=is_belief_candidate,
            belief_type=belief_type,
        )

        # Detect conflicts
        if is_belief_candidate and belief_type:
            conflicts = await self._detect_conflicts(observation)
            if conflicts:
                # Create new observation with conflicts
                # (can't mutate frozen dataclass)
                observation = SharedObservation(
                    **{**observation.__dict__, "conflicts_with": conflicts}
                )
                for conflict_id in conflicts:
                    self._conflicts.append(
                        (observation.observation_id, conflict_id, "semantic_conflict")
                    )

        # Ensure capacity
        await self._ensure_capacity(source_agent_id)

        # Store
        self._observations[observation.observation_id] = observation
        self._observation_locks[observation.observation_id] = asyncio.Lock()

        # Update agent view
        if source_agent_id not in self._agent_views:
            self._agent_views[source_agent_id] = []
        self._agent_views[source_agent_id].append(observation.observation_id)

        return observation
```

### Capacity Management

```python
async def _ensure_capacity(self, source_agent_id: str) -> None:
    """Evict lowest-attention items if over capacity.

    Two-level capacity enforcement:
    1. Per-agent: max 7 items (configurable)
    2. Global: max 50 items (configurable)

    Eviction strategy:
    - Lowest attention weight evicted first
    - Preserves per-agent fairness
    """
    # Per-agent capacity
    agent_obs = self._agent_views.get(source_agent_id, [])
    while len(agent_obs) >= self.config.max_items_per_agent:
        # Find lowest attention item from this agent
        lowest_id = min(
            agent_obs,
            key=lambda oid: self._observations[oid].attention_weight
            if oid in self._observations else 0
        )
        agent_obs.remove(lowest_id)
        if lowest_id in self._observations:
            del self._observations[lowest_id]
            if lowest_id in self._observation_locks:
                del self._observation_locks[lowest_id]

    # Global capacity
    while len(self._observations) >= self.config.max_total_items:
        # Find lowest attention item overall
        lowest = min(
            self._observations.values(),
            key=lambda o: o.attention_weight
        )
        del self._observations[lowest.observation_id]

        # Remove from agent views
        for agent_id, obs_ids in self._agent_views.items():
            if lowest.observation_id in obs_ids:
                obs_ids.remove(lowest.observation_id)
                break
```

### Conflict Detection

```python
async def _detect_conflicts(
    self,
    new_observation: SharedObservation,
) -> list[str]:
    """Detect semantic conflicts with existing observations.

    Phase 1 (Simple Heuristic):
    - Conflict if same belief_type from different agents
    - Placeholder for Phase 2 embedding-based detection

    Phase 2 (Embeddings):
    - Compute semantic similarity via embeddings
    - Conflict if similarity > threshold AND content differs

    Returns:
        List of observation IDs that conflict
    """
    conflicts = []

    if not (new_observation.is_belief_candidate and new_observation.belief_type):
        return conflicts

    for obs_id, obs in self._observations.items():
        # Skip same source
        if obs.source_agent_id == new_observation.source_agent_id:
            continue

        # Check if same belief type
        if obs.is_belief_candidate and obs.belief_type == new_observation.belief_type:
            # Phase 1: Simple heuristic (same type = potential conflict)
            if self.embedding_provider is None:
                conflicts.append(obs_id)
            else:
                # Phase 2: Embedding-based semantic similarity
                similarity = await self.embedding_provider.similarity(
                    new_observation.content,
                    obs.content,
                )
                if similarity > self.config.conflict_threshold:
                    conflicts.append(obs_id)

    return conflicts


async def flag_conflict(
    self,
    observation_a_id: str,
    observation_b_id: str,
    conflict_reason: str,
) -> None:
    """Explicitly flag a conflict for reconciliation.

    Called by orchestrator when it detects a conflict that
    automatic detection missed.
    """
    self._conflicts.append((observation_a_id, observation_b_id, conflict_reason))

    # Update conflicts_with for both observations
    # (requires recreating frozen dataclasses)
    if observation_a_id in self._observations:
        obs_a = self._observations[observation_a_id]
        conflicts_a = list(obs_a.conflicts_with)
        if observation_b_id not in conflicts_a:
            conflicts_a.append(observation_b_id)
            self._observations[observation_a_id] = SharedObservation(
                **{**obs_a.__dict__, "conflicts_with": conflicts_a}
            )

    if observation_b_id in self._observations:
        obs_b = self._observations[observation_b_id]
        conflicts_b = list(obs_b.conflicts_with)
        if observation_a_id not in conflicts_b:
            conflicts_b.append(observation_a_id)
            self._observations[observation_b_id] = SharedObservation(
                **{**obs_b.__dict__, "conflicts_with": conflicts_b}
            )


async def get_conflicts(self) -> list[tuple[SharedObservation, SharedObservation, str]]:
    """Get all unresolved conflicts for reconciliation."""
    result = []
    for obs_a_id, obs_b_id, reason in self._conflicts:
        obs_a = self._observations.get(obs_a_id)
        obs_b = self._observations.get(obs_b_id)
        if obs_a and obs_b:
            result.append((obs_a, obs_b, reason))
    return result
```

### Attention Management

```python
async def apply_attention_decay(self) -> None:
    """Decay attention weights (called periodically).

    Multiplies all attention weights by decay factor (default 0.9).
    """
    for obs_id, obs in self._observations.items():
        new_weight = obs.attention_weight * self.config.attention_decay_factor
        # Update immutable observation
        self._observations[obs_id] = SharedObservation(
            **{**obs.__dict__, "attention_weight": new_weight}
        )


async def boost_attention(
    self,
    observation_id: str,
    boost: float = 0.2,
) -> None:
    """Boost attention for a specific observation.

    Called when an observation becomes relevant again.
    """
    if observation_id in self._observations:
        obs = self._observations[observation_id]
        new_weight = min(1.0, obs.attention_weight + boost)
        self._observations[observation_id] = SharedObservation(
            **{**obs.__dict__, "attention_weight": new_weight}
        )
```

### Context Retrieval

```python
async def get_context_for_agent(
    self,
    agent_id: str,
    role: AgentRole,
    max_items: int | None = None,
) -> list[SharedObservation]:
    """Get relevant context filtered by agent role.

    Filtering by role:
    - CRITIC: Only belief candidates
    - RESEARCHER: All observations
    - EXECUTOR: Only SKILL and FACT types
    - Other roles: All observations

    Sorting:
    - Primary: attention_weight (descending)
    - Secondary: timestamp (descending - recent first)

    Access tracking:
    - Updates accessed_by set
    - Increments access_count
    """
    max_items = max_items or self.config.max_items_per_agent

    # Get all observations
    all_obs = list(self._observations.values())

    # Filter by role
    relevant = self._filter_by_role(all_obs, role)

    # Sort by attention weight + recency
    sorted_obs = sorted(
        relevant,
        key=lambda o: (o.attention_weight, o.timestamp.timestamp()),
        reverse=True,
    )

    # Take top N
    result = sorted_obs[:max_items]

    # Track access (mutable fields on frozen dataclass)
    for obs in result:
        # Create new observation with updated access tracking
        accessed_by = set(obs.accessed_by)
        accessed_by.add(agent_id)
        access_count = obs.access_count + 1

        self._observations[obs.observation_id] = SharedObservation(
            **{
                **obs.__dict__,
                "accessed_by": accessed_by,
                "access_count": access_count,
            }
        )

    return result


def _filter_by_role(
    self,
    observations: list[SharedObservation],
    role: AgentRole,
) -> list[SharedObservation]:
    """Filter observations by role relevance."""
    if role == AgentRole.CRITIC:
        # Critics see claims and assertions
        return [o for o in observations if o.is_belief_candidate]
    elif role == AgentRole.RESEARCHER:
        # Researchers see everything
        return observations
    elif role == AgentRole.EXECUTOR:
        # Executors see action-related observations
        return [
            o for o in observations
            if o.belief_type in ("SKILL", "FACT", None)
        ]
    else:
        return observations
```

### Belief Candidates

```python
async def get_belief_candidates(self) -> list[SharedObservation]:
    """Get observations that should become beliefs.

    Returns only non-conflicting candidates.
    Candidates with conflicts excluded until reconciled.
    """
    return [
        obs for obs in self._observations.values()
        if obs.is_belief_candidate and not obs.conflicts_with
    ]
```

---

## Integration with Existing Code

### 1. Update `TaskContext` in `multi_agent_orchestrator.py`

```python
# BEFORE (current code):
@dataclass
class TaskContext:
    # ...
    working_memory: dict[str, Any] = field(default_factory=dict)  # Simple dict


# AFTER (with shared memory):
@dataclass
class TaskContext:
    # ...
    working_memory: dict[str, Any] = field(default_factory=dict)
    # Note: SharedWorkingMemory injected via working_memory["__shared__"] key
    # This preserves backward compatibility while adding new capability
```

**No breaking changes needed.** Simply inject `SharedWorkingMemory` into `working_memory["__shared__"]` when creating context for parallel agents.

### 2. Update `ParallelCognitiveOrchestrator` (FR-002)

```python
# In FR-002 implementation:
async def orchestrate_parallel(
    self,
    agents: list[AgentSpec],
    context: TaskContext,
    agent_executor: AgentExecutor,
) -> OrchestratorResult:
    # Initialize shared memory for this task
    from .shared_memory import SharedWorkingMemory

    shared_memory = SharedWorkingMemory(context.task_id)
    context.working_memory["__shared__"] = shared_memory

    # Rest of parallel orchestration...
```

### 3. Agent Loop Integration (Optional for FR-001)

Agents can optionally write to shared memory after actions:

```python
# In agent loop, after action execution:
if "__shared__" in context.working_memory:
    shared_memory = context.working_memory["__shared__"]
    await shared_memory.add_observation(
        content=action_result.result,
        source_agent_id=agent_id,
        confidence=0.9 if action_result.success else 0.5,
    )
```

---

## Testing Strategy

### Unit Tests (`tests/orchestration/test_shared_working_memory.py`)

```python
import pytest
import asyncio
from datetime import datetime

from draagon_ai.orchestration.shared_memory import (
    SharedObservation,
    SharedWorkingMemory,
    SharedWorkingMemoryConfig,
)
from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole


class TestSharedObservation:
    """Test SharedObservation dataclass."""

    def test_observation_immutable(self):
        """Observations should be immutable (frozen)."""
        obs = SharedObservation(
            observation_id="obs_1",
            content="Test",
            source_agent_id="agent_a",
            timestamp=datetime.now(),
        )

        with pytest.raises(AttributeError):
            obs.content = "Modified"  # Should fail

    def test_observation_validation(self):
        """Validate attention_weight and confidence are 0-1."""
        with pytest.raises(ValueError):
            SharedObservation(
                observation_id="obs_1",
                content="Test",
                source_agent_id="agent_a",
                timestamp=datetime.now(),
                attention_weight=1.5,  # Invalid
            )


class TestCapacityManagement:
    """Test Miller's Law capacity enforcement."""

    @pytest.mark.asyncio
    async def test_per_agent_capacity(self):
        """Test 7-item limit per agent."""
        memory = SharedWorkingMemory(
            "task_1",
            SharedWorkingMemoryConfig(max_items_per_agent=7)
        )

        # Add 10 items from same agent
        for i in range(10):
            await memory.add_observation(
                content=f"Observation {i}",
                source_agent_id="agent_a",
                attention_weight=i / 10.0,  # Increasing attention
            )

        # Should only keep 7 highest attention items
        assert len(memory._observations) == 7

        # Highest attention items kept (7, 8, 9 have highest weights)
        remaining_contents = {obs.content for obs in memory._observations.values()}
        assert "Observation 9" in remaining_contents
        assert "Observation 0" not in remaining_contents  # Lowest attention evicted

    @pytest.mark.asyncio
    async def test_global_capacity(self):
        """Test global capacity across all agents."""
        memory = SharedWorkingMemory(
            "task_1",
            SharedWorkingMemoryConfig(max_items_per_agent=10, max_total_items=50)
        )

        # Add 60 items from 10 agents (6 each)
        for agent_i in range(10):
            for obs_i in range(6):
                await memory.add_observation(
                    content=f"Agent {agent_i} obs {obs_i}",
                    source_agent_id=f"agent_{agent_i}",
                    attention_weight=0.5,
                )

        # Should cap at 50
        assert len(memory._observations) <= 50


class TestConflictDetection:
    """Test semantic conflict detection."""

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

        # Should detect conflict
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


class TestAttentionManagement:
    """Test attention weighting and decay."""

    @pytest.mark.asyncio
    async def test_attention_decay(self):
        """Test attention weight decay."""
        memory = SharedWorkingMemory(
            "task_1",
            SharedWorkingMemoryConfig(attention_decay_factor=0.9)
        )

        obs = await memory.add_observation(
            content="Test",
            source_agent_id="agent_a",
            attention_weight=1.0,
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
            content="Test",
            source_agent_id="agent_a",
            attention_weight=0.5,
        )

        await memory.boost_attention(obs.observation_id, boost=0.3)

        updated_obs = memory._observations[obs.observation_id]
        assert updated_obs.attention_weight == 0.8

    @pytest.mark.asyncio
    async def test_attention_boost_capped(self):
        """Attention weight capped at 1.0."""
        memory = SharedWorkingMemory("task_1")

        obs = await memory.add_observation(
            content="Test",
            source_agent_id="agent_a",
            attention_weight=0.9,
        )

        await memory.boost_attention(obs.observation_id, boost=0.5)

        updated_obs = memory._observations[obs.observation_id]
        assert updated_obs.attention_weight == 1.0  # Capped


class TestRoleFilteredRetrieval:
    """Test context retrieval filtered by role."""

    @pytest.mark.asyncio
    async def test_critic_sees_only_candidates(self):
        """CRITIC role sees only belief candidates."""
        memory = SharedWorkingMemory("task_1")

        # Add belief candidates
        await memory.add_observation(
            content="Claim 1",
            source_agent_id="agent_a",
            is_belief_candidate=True,
        )
        await memory.add_observation(
            content="Claim 2",
            source_agent_id="agent_a",
            is_belief_candidate=True,
        )

        # Add general observations
        await memory.add_observation(
            content="General observation",
            source_agent_id="agent_a",
            is_belief_candidate=False,
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
            content="Belief",
            source_agent_id="agent_a",
            is_belief_candidate=True,
        )
        await memory.add_observation(
            content="General",
            source_agent_id="agent_a",
        )

        context = await memory.get_context_for_agent(
            agent_id="researcher_1",
            role=AgentRole.RESEARCHER,
        )

        assert len(context) == 2


class TestConcurrentAccess:
    """Test thread safety and concurrent access."""

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
        tasks = [
            write_observations(f"agent_{i}")
            for i in range(10)
        ]

        await asyncio.gather(*tasks)

        # All observations should be stored (up to capacity limit)
        assert len(memory._observations) <= 50  # Global cap

        # No duplicate observation IDs
        obs_ids = list(memory._observations.keys())
        assert len(obs_ids) == len(set(obs_ids))

    @pytest.mark.asyncio
    async def test_no_deadlocks(self):
        """Concurrent reads and writes shouldn't deadlock."""
        memory = SharedWorkingMemory("task_1")

        async def writer():
            for i in range(10):
                await memory.add_observation(
                    content=f"Obs {i}",
                    source_agent_id="writer",
                )
                await asyncio.sleep(0.01)

        async def reader():
            for _ in range(10):
                await memory.get_context_for_agent(
                    agent_id="reader",
                    role=AgentRole.RESEARCHER,
                )
                await asyncio.sleep(0.01)

        # Run concurrently - should not deadlock
        await asyncio.gather(writer(), reader())
```

### Integration Tests

```python
# tests/integration/test_shared_memory_integration.py

class TestTaskContextIntegration:
    """Test integration with TaskContext."""

    @pytest.mark.asyncio
    async def test_inject_into_task_context(self):
        """SharedWorkingMemory can be injected into TaskContext."""
        from draagon_ai.orchestration.multi_agent_orchestrator import TaskContext
        from draagon_ai.orchestration.shared_memory import SharedWorkingMemory

        context = TaskContext(task_id="task_123", query="Test query")
        shared_memory = SharedWorkingMemory(context.task_id)

        context.working_memory["__shared__"] = shared_memory

        # Agents can access it
        assert "__shared__" in context.working_memory
        assert isinstance(context.working_memory["__shared__"], SharedWorkingMemory)
```

### Stress Tests

```python
# tests/stress/test_shared_memory_stress.py

class TestSharedMemoryStress:
    """Stress tests for high load scenarios."""

    @pytest.mark.asyncio
    async def test_high_concurrency(self):
        """20 agents, 50 observations each."""
        memory = SharedWorkingMemory(
            "stress_test",
            SharedWorkingMemoryConfig(max_total_items=500),
        )

        async def agent_task(agent_id: str):
            for i in range(50):
                await memory.add_observation(
                    content=f"Agent {agent_id} observation {i}",
                    source_agent_id=agent_id,
                    attention_weight=0.5,
                )

        tasks = [agent_task(f"agent_{i}") for i in range(20)]
        await asyncio.gather(*tasks)

        # Should respect capacity
        assert len(memory._observations) <= 500

    @pytest.mark.asyncio
    async def test_rapid_decay_cycles(self):
        """1000 attention decay cycles."""
        memory = SharedWorkingMemory("stress_test")

        # Add observations
        for i in range(10):
            await memory.add_observation(
                content=f"Obs {i}",
                source_agent_id="agent_a",
                attention_weight=1.0,
            )

        # Decay 1000 times
        for _ in range(1000):
            await memory.apply_attention_decay()

        # All attention weights should be very low but >= 0
        for obs in memory._observations.values():
            assert 0 <= obs.attention_weight < 0.01
```

---

## Embedding Provider Integration (Phase 2)

For Phase 1, use simple heuristic. For Phase 2, add embedding support:

```python
# Example embedding provider implementation

from sentence_transformers import SentenceTransformer
import numpy as np

class SentenceTransformerEmbeddingProvider:
    """Embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # sentence-transformers is synchronous
        embedding = self.model.encode(text)
        return embedding.tolist()

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute cosine similarity."""
        emb_a = await self.embed(text_a)
        emb_b = await self.embed(text_b)

        # Cosine similarity
        dot_product = np.dot(emb_a, emb_b)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)

        return dot_product / (norm_a * norm_b)


# Usage:
embedding_provider = SentenceTransformerEmbeddingProvider()
memory = SharedWorkingMemory(
    "task_123",
    embedding_provider=embedding_provider,
)
```

---

## Performance Considerations

### Latency Targets

| Operation | Target | Strategy |
|-----------|--------|----------|
| `add_observation()` | <10ms | O(1) dict insert, O(n) conflict check (n = observations) |
| `get_context_for_agent()` | <5ms | O(n log n) sort, cache if needed |
| `apply_attention_decay()` | <20ms | O(n) update all observations |
| `get_conflicts()` | <5ms | O(c) where c = conflict count |

### Memory Usage

- Each observation: ~200 bytes (with all metadata)
- 50 observations: ~10 KB
- 10 agents × 50 observations: ~100 KB (well within limits)

### Optimization Opportunities (Future)

1. **Lazy Conflict Detection**: Only check on `get_conflicts()`, not on add
2. **Attention Index**: Maintain sorted index by attention weight for fast eviction
3. **Embedding Cache**: Cache embeddings to avoid recomputation
4. **Batch Decay**: Decay in batches rather than one-by-one

---

## Risk Mitigation

### Risk 1: Immutable Dataclass Updates

**Problem:** `frozen=True` dataclasses can't be mutated, but we need to update `accessed_by` and `access_count`.

**Solution:** Replace entire observation in dict with new instance:
```python
new_obs = SharedObservation(**{**old_obs.__dict__, "accessed_by": new_set})
self._observations[obs_id] = new_obs
```

**Alternative:** Make observation mutable and use locks for updates. (Less clean but more efficient)

### Risk 2: Embedding Provider Performance

**Problem:** Embedding computation could be slow (100ms+ per call).

**Mitigation:**
- Phase 1: Use simple heuristic (no embeddings)
- Phase 2: Add embedding cache
- Use async/await to prevent blocking
- Consider batch embedding for multiple observations

### Risk 3: Concurrent Access Deadlocks

**Problem:** Multiple locks could deadlock.

**Mitigation:**
- Use global lock for structure changes
- Keep critical sections small
- Test extensively with concurrent stress tests
- Avoid nested lock acquisition

---

## Success Criteria

### Functional

- ✅ All 7 functional requirements (FR-001.1 through FR-001.7) pass tests
- ✅ 100% test coverage for core methods
- ✅ Integration with TaskContext works without breaking changes
- ✅ Conflict detection works (Phase 1: heuristic, Phase 2: embeddings)

### Performance

- ✅ <10ms to add observation (average)
- ✅ <5ms to get context for agent
- ✅ Handles 10+ concurrent agents without race conditions
- ✅ Miller's Law capacity enforced correctly

### Cognitive

- ✅ Attention decay produces intuitively correct results
- ✅ Role filtering provides relevant context to each agent type
- ✅ Conflict detection catches obvious contradictions (>90% rate with embeddings)

---

## Implementation Checklist

### Day 1-2: Core Data Structures
- [ ] Create `shared_memory.py` in `orchestration/`
- [ ] Implement `SharedObservation` dataclass with validation
- [ ] Implement `SharedWorkingMemoryConfig` dataclass
- [ ] Implement `SharedWorkingMemory` skeleton
- [ ] Add `EmbeddingProvider` protocol
- [ ] Write docstrings for all classes
- [ ] Unit tests for dataclasses

### Day 3-4: Capacity & Eviction
- [ ] Implement `_ensure_capacity()` per-agent logic
- [ ] Implement `_ensure_capacity()` global logic
- [ ] Implement eviction algorithm (lowest attention)
- [ ] Unit tests for capacity management
- [ ] Test per-agent fairness

### Day 5-6: Conflict Detection
- [ ] Implement `_detect_conflicts()` Phase 1 (heuristic)
- [ ] Implement `flag_conflict()` explicit flagging
- [ ] Implement `get_conflicts()` retrieval
- [ ] Add embedding provider integration (optional)
- [ ] Unit tests for conflict detection
- [ ] Test with and without embeddings

### Day 7: Attention & Access
- [ ] Implement `apply_attention_decay()`
- [ ] Implement `boost_attention()`
- [ ] Implement access tracking in `get_context_for_agent()`
- [ ] Unit tests for attention management

### Day 8: Role Filtering
- [ ] Implement `_filter_by_role()` for all AgentRole types
- [ ] Implement sorting (attention + recency)
- [ ] Implement `get_context_for_agent()` full logic
- [ ] Implement `get_belief_candidates()`
- [ ] Unit tests for role filtering

### Day 9-10: Testing & Integration
- [ ] Write integration tests with TaskContext
- [ ] Write stress tests (20 agents, 1000 decay cycles)
- [ ] Performance benchmarking
- [ ] Documentation updates
- [ ] Code review and refinement

---

## Documentation Updates

### 1. Update `CLAUDE.md`

Add section on Shared Cognitive Working Memory:

```markdown
## Shared Cognitive Working Memory

For multi-agent coordination, use `SharedWorkingMemory`:

```python
from draagon_ai.orchestration.shared_memory import SharedWorkingMemory

# Create for task
shared_memory = SharedWorkingMemory(task_id="task_123")

# Agent A adds observation
await shared_memory.add_observation(
    content="User prefers dark mode",
    source_agent_id="agent_a",
    is_belief_candidate=True,
    belief_type="PREFERENCE",
)

# Agent B gets context (Miller's Law: 7 items max)
context = await shared_memory.get_context_for_agent(
    agent_id="agent_b",
    role=AgentRole.RESEARCHER,
)
```

### 2. Update API Documentation

Generate API docs with:
```bash
pdoc --html src/draagon_ai/orchestration/shared_memory.py
```

---

## Constitution Compliance

### LLM-First ✅
- Conflict detection uses embeddings (Phase 2), NOT regex
- Phase 1 uses simple heuristic as placeholder

### XML Output ✅
- No LLM prompts in FR-001 (data structures only)

### Protocol-Based ✅
- `EmbeddingProvider` is a Protocol
- Uses existing `AgentRole` enum

### Async-First ✅
- All methods are `async def`
- Uses `asyncio.Lock` for concurrency

### Research-Grounded ✅
- Miller's Law (7±2 items)
- Baddeley's Working Memory Model (attention weighting)
- Cognitive psychology principles

### Test Outcomes ✅
- Tests verify correct context retrieval
- Tests verify capacity enforcement
- NOT testing specific eviction order (implementation detail)

---

**Plan Status:** Ready for Implementation
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
**Approved By:** [Awaiting approval]
