# FR-001: Shared Cognitive Working Memory

**Feature:** Task-scoped cognitive working memory for multi-agent coordination
**Status:** Planned
**Priority:** P0 (Foundation)
**Complexity:** Medium
**Phase:** Cognitive Swarm Phase 1

---

## Overview

Implement a shared cognitive working memory system that enables multiple agents to coordinate via a psychologically-grounded attention-weighted memory pool. Unlike simple dictionary-based context sharing, this provides conflict detection, attention management, and capacity constraints based on cognitive science (Miller's Law: 7±2 items per agent).

### Research Foundation

- **Miller's Law (1956)**: Working memory capacity of 7±2 items
- **Baddeley's Working Memory Model**: Attention-weighted activation
- **MultiAgentBench (ACL 2025)**: Shared context prevents coordination failures
- **Intrinsic Memory Agents**: Heterogeneous agent-specific views improve performance by 38.6%

---

## Functional Requirements

### FR-001.1: Observation Storage with Attribution

**Requirement:** Store observations with full source attribution

**Acceptance Criteria:**
- Each observation includes:
  - Unique observation ID
  - Content (string)
  - Source agent ID
  - Timestamp
  - Attention weight (0-1)
  - Confidence (0-1)
  - Belief candidate flag
  - Belief type (FACT, SKILL, PREFERENCE, etc.)
  - Conflict markers
- Observations are immutable once stored
- Concurrent writes from multiple agents are safe (no race conditions)

**Test Approach:**
- Create observation from agent A
- Verify all fields populated
- Concurrent write test: 10 agents write simultaneously, all observations stored correctly

---

### FR-001.2: Capacity Management (Miller's Law)

**Requirement:** Enforce 7±2 items per agent, with global capacity limit

**Acceptance Criteria:**
- Per-agent limit: max 7 items (configurable)
- Global limit: max 50 items (configurable)
- When capacity exceeded, lowest attention-weight items evicted
- Eviction respects per-agent fairness (no single agent dominates)

**Test Approach:**
- Add 10 items from same agent with random attention weights
- Assert only 7 remain
- Assert highest attention items retained
- Add items from 10 agents, verify global limit enforced

**Constitution Check:** ✅ Research-grounded (Miller's Law)

---

### FR-001.3: Automatic Conflict Detection

**Requirement:** Detect semantic conflicts between observations from different agents

**Acceptance Criteria:**
- When new observation added, check existing observations for conflicts
- Conflict detected when:
  - Different source agents
  - Same belief type
  - Semantic similarity above threshold (0.7)
- Conflicting observations marked with conflict IDs
- Conflict tuples stored: (obs_a, obs_b, reason)

**Test Approach:**
- Agent A observes: "Meeting at 3pm"
- Agent B observes: "Meeting at 4pm"
- Both marked as belief candidates with type FACT
- Assert conflict detected and both observations marked

**Constitution Check:** ✅ LLM-First (semantic similarity via embeddings, NOT regex)

---

### FR-001.4: Attention Weighting and Decay

**Requirement:** Manage attention via weighting and periodic decay

**Acceptance Criteria:**
- Each observation has attention weight (0-1)
- `apply_attention_decay()` multiplies all weights by decay factor (default 0.9)
- `boost_attention(obs_id, boost)` increases weight by boost amount (capped at 1.0)
- Decay called periodically (every N sync iterations)

**Test Approach:**
- Create observation with attention=1.0
- Apply decay with factor=0.9
- Assert attention=0.9
- Boost by 0.2
- Assert attention=1.0 (capped)

---

### FR-001.5: Role-Filtered Context Retrieval

**Requirement:** Retrieve relevant context filtered by agent role

**Acceptance Criteria:**
- `get_context_for_agent(agent_id, role, max_items)` returns filtered observations
- CRITIC role: sees belief candidates only
- RESEARCHER role: sees all observations
- EXECUTOR role: sees SKILLs and FACTs
- Results sorted by attention weight + recency
- Access tracking: observation's `accessed_by` set updated

**Test Approach:**
- Add 3 belief candidates, 2 general observations
- Retrieve as CRITIC role
- Assert only 3 belief candidates returned
- Retrieve as RESEARCHER
- Assert all 5 returned

**Constitution Check:** ✅ Protocol-based (AgentRole enum)

---

### FR-001.6: Belief Candidate Flagging

**Requirement:** Mark observations that should become beliefs

**Acceptance Criteria:**
- Observations can be flagged `is_belief_candidate=True`
- `get_belief_candidates()` returns non-conflicting candidates
- Candidates with conflicts excluded until reconciled

**Test Approach:**
- Add 2 belief candidates without conflicts
- Add 1 belief candidate with conflict
- `get_belief_candidates()` returns only the 2 non-conflicting

---

### FR-001.7: Concurrent Access Safety

**Requirement:** Support safe concurrent reads/writes from multiple agents

**Acceptance Criteria:**
- Per-observation locks for updates
- Global lock for structure changes
- No race conditions when 5+ agents write simultaneously
- No deadlocks

**Test Approach:**
- Spawn 10 async tasks, each adding 20 observations
- All complete successfully
- Total observations respects global capacity
- No observation IDs duplicated

**Constitution Check:** ✅ Async-first processing

---

## Data Structures

```python
@dataclass
class SharedObservation:
    observation_id: str
    content: str
    source_agent_id: str
    timestamp: datetime
    attention_weight: float = 0.5
    confidence: float = 1.0
    is_belief_candidate: bool = False
    belief_type: str | None = None
    conflicts_with: list[str] = field(default_factory=list)
    accessed_by: set[str] = field(default_factory=set)
    access_count: int = 0

@dataclass
class SharedWorkingMemoryConfig:
    max_items_per_agent: int = 7  # Miller's Law
    max_total_items: int = 50
    attention_decay_factor: float = 0.9
    conflict_threshold: float = 0.7
    sync_interval_iterations: int = 3

class SharedWorkingMemory:
    def __init__(self, task_id: str, config: SharedWorkingMemoryConfig | None = None)
    async def add_observation(...) -> SharedObservation
    async def get_context_for_agent(agent_id: str, role: AgentRole, max_items: int | None = None) -> list[SharedObservation]
    async def flag_conflict(obs_a_id: str, obs_b_id: str, reason: str) -> None
    async def get_conflicts() -> list[tuple[SharedObservation, SharedObservation, str]]
    async def get_belief_candidates() -> list[SharedObservation]
    async def apply_attention_decay() -> None
    async def boost_attention(obs_id: str, boost: float = 0.2) -> None
```

---

## Integration Points

### Upstream Dependencies
- `AgentRole` enum (from orchestration)
- `TaskContext` (inject `__shared__` key in working_memory dict)
- Embedding provider (for semantic conflict detection)

### Downstream Consumers
- `AgentLoop`: Reads context before each iteration
- `DecisionEngine`: Includes shared observations in prompt
- `ActionExecutor`: Writes observations after tool execution
- `LearningChannel`: Broadcasts significant observations

---

## Success Criteria

### Cognitive Metrics
- **Attention Accuracy**: Relevant items retain high attention weights
- **Conflict Detection Rate**: >90% of semantic conflicts detected
- **Eviction Quality**: Low-value items evicted, high-value retained

### Performance Metrics
- **Concurrent Writes**: Handle 10 agents writing simultaneously
- **Latency**: <10ms to add observation
- **Memory Efficiency**: O(1) lookup by observation ID

---

## Testing Strategy

### Unit Tests
- `test_capacity_management()`: Miller's Law enforcement
- `test_conflict_detection()`: Semantic conflicts detected
- `test_attention_decay()`: Decay math correct
- `test_role_filtered_context()`: Roles filter correctly
- `test_concurrent_access()`: No race conditions

### Integration Tests
- `test_multi_agent_coordination()`: 5 agents share observations
- `test_belief_candidate_flow()`: Observations → candidates → reconciliation

### Stress Tests
- `test_high_concurrency()`: 20 agents, 50 observations each
- `test_rapid_decay_cycles()`: 1000 decay cycles

---

## Implementation Tasks

1. **Core Data Structures** (2 days)
   - Implement `SharedObservation` dataclass
   - Implement `SharedWorkingMemoryConfig`
   - Implement `SharedWorkingMemory` class skeleton

2. **Observation Management** (2 days)
   - `add_observation()` with locking
   - Capacity enforcement logic
   - Eviction algorithm (lowest attention)

3. **Conflict Detection** (2 days)
   - Semantic similarity via embeddings
   - Conflict marking logic
   - `get_conflicts()` retrieval

4. **Attention System** (1 day)
   - Decay implementation
   - Boost implementation
   - Access tracking

5. **Role-Based Filtering** (1 day)
   - `get_context_for_agent()` with role logic
   - Sorting by attention + recency

6. **Testing** (2 days)
   - Unit tests for all methods
   - Integration tests with mock agents
   - Stress tests

**Total Estimate:** 10 days

---

## Open Questions

[NEEDS CLARIFICATION: Should conflict detection use embeddings immediately, or placeholder with TODO for Phase 2?]
**Recommendation:** Start with simple heuristic (same belief_type = potential conflict), add embeddings in Phase 2.

[NEEDS CLARIFICATION: Should eviction prioritize recency or attention weight?]
**Recommendation:** Attention weight primary, recency as tiebreaker (attention is semantic importance).

---

## Constitution Compliance

- ✅ **LLM-First**: Semantic conflict detection via embeddings (not regex)
- ✅ **Research-Grounded**: Miller's Law (7±2), Baddeley's Working Memory Model
- ✅ **Async-First**: All methods are `async`
- ✅ **Protocol-Based**: Uses `AgentRole` protocol
- ✅ **Test Outcomes**: Tests verify correct context retrieval, not specific implementation paths

---

**Document Status:** Draft
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
