# FR-002: Parallel Multi-Agent Orchestration

**Feature:** Parallel execution of multiple agents with cognitive coordination
**Status:** Planned
**Priority:** P0 (Foundation)
**Complexity:** High
**Phase:** Cognitive Swarm Phase 2

---

## Overview

Implement parallel multi-agent orchestration that executes 3-5 agents concurrently with shared cognitive working memory, periodic synchronization, and expertise-based routing. Unlike simple concurrent execution, this provides barrier synchronization, conflict resolution triggers, and transactive memory integration.

### Research Foundation

- **Anthropic Multi-Agent Research**: 90.2% improvement with parallel agents; 3-5 subagents optimal
- **MultiAgentBench (ACL 2025)**: Detailed task descriptions prevent duplication
- **MAST Framework**: Inter-agent misalignment causes 34.7% of failures
- **Why Multi-Agent Systems Fail**: Coordination breakdowns account for 22.3% of failures

---

## Functional Requirements

### FR-002.1: Parallel Agent Execution

**Requirement:** Execute multiple agents concurrently with timeout protection

**Acceptance Criteria:**
- Run 3-5 agents in parallel (configurable max)
- Each agent has access to shared working memory
- Timeout protection per agent (default 60s)
- Agent failure doesn't crash orchestration
- Early termination allowed (agent can finish before others)

**Test Approach:**
- Create 5 agents with varying execution times (10s, 20s, 30s, 40s, 50s)
- Execute in parallel
- Assert all 5 complete
- Assert total time ≈ 50s (not 150s sequential)
- Assert each agent accessed shared memory

**Constitution Check:** ✅ Async-first processing

---

### FR-002.2: Orchestration Modes

**Requirement:** Support multiple synchronization modes

**Acceptance Criteria:**
- **FORK_JOIN**: All agents start together, sync at end only
- **BARRIER_SYNC**: Periodic sync barriers every N iterations
- **STREAMING**: Continuous sync via shared memory (no explicit barriers)
- Mode configurable via `ParallelExecutionConfig`
- Barrier sync should reconcile conflicts at each barrier

**Test Approach:**
- Run same task with all 3 modes
- FORK_JOIN: verify no mid-execution reconciliation
- BARRIER_SYNC: verify reconciliation every 3 iterations
- STREAMING: verify agents read latest observations immediately

**Constitution Check:** ✅ Research-grounded (Anthropic research on sync patterns)

---

### FR-002.3: Shared Memory Integration

**Requirement:** Inject shared working memory into agent context

**Acceptance Criteria:**
- Each agent receives `TaskContext` with `working_memory['__shared__']` = SharedWorkingMemory instance
- Same SharedWorkingMemory instance shared across all agents
- Agent outputs automatically added as observations
- Observations include agent_id, confidence based on success

**Test Approach:**
- Create 3 agents
- Execute in parallel
- Agent A adds observation "Found data X"
- Agent B reads context, sees Agent A's observation
- Assert observation source_agent_id = "agent_a"

**Constitution Check:** ✅ Protocol-based (uses TaskContext protocol)

---

### FR-002.4: Conflict Reconciliation Triggering

**Requirement:** Trigger belief reconciliation when conflicts detected

**Acceptance Criteria:**
- After all agents complete, check `shared_memory.get_conflicts()`
- If conflicts exist, call `belief_service.reconcile_multi_agent()`
- Reconciliation happens before result merging
- Reconciled beliefs available to reflection service

**Test Approach:**
- Create 2 agents with conflicting outputs:
  - Agent A: "Meeting at 3pm"
  - Agent B: "Meeting at 4pm"
- Both mark as belief candidates
- Execute in parallel
- Assert reconciliation triggered
- Assert `belief_service.reconcile_called == True`

**Constitution Check:** ✅ Cognitive authenticity (belief reconciliation)

---

### FR-002.5: Expertise-Based Routing

**Requirement:** Route queries to agents with relevant expertise

**Acceptance Criteria:**
- If `TransactiveMemory` provided, call `route_query(query, agents)`
- Agents reordered by expertise score
- Top N agents selected (max_concurrent_agents)
- If no transactive memory, use agents as provided

**Test Approach:**
- Create 5 agents: weather, calendar, home, generic1, generic2
- Transactive memory has:
  - weather: expertise 0.9 in "weather"
  - calendar: expertise 0.8 in "scheduling"
- Query: "What's the weather for my meeting tomorrow?"
- Assert weather agent ranked first
- Assert calendar agent in top 3

**Constitution Check:** ✅ Research-grounded (Wegner 1987 transactive memory)

---

### FR-002.6: Barrier Synchronization

**Requirement:** For BARRIER_SYNC mode, pause and reconcile at intervals

**Acceptance Criteria:**
- Every `sync_interval_iterations` (default 3), apply attention decay
- Check for conflicts
- If >3 conflicts, log warning (could pause agents for reconciliation)
- Agents continue after barrier

**Test Approach:**
- Run 5 agents in BARRIER_SYNC mode
- Each agent produces 1 observation per iteration
- After 3 iterations, verify:
  - Attention decay applied
  - Conflicts checked
- Agents continue to completion

**Constitution Check:** ✅ Cognitive authenticity (attention decay)

---

### FR-002.7: Expertise Tracking Update

**Requirement:** Update transactive memory based on execution results

**Acceptance Criteria:**
- After all agents complete, extract topic from query
- For each agent result, call `transactive_memory.update_expertise(agent_id, topic, success)`
- Success = `agent_result.success == True`

**Test Approach:**
- Run agent_weather on "What's the weather?" query
- Agent succeeds
- Assert `transactive_memory.update_expertise("agent_weather", "weather", True)` called
- Agent's expertise in "weather" increases

---

### FR-002.8: Timeout and Failure Handling

**Requirement:** Handle agent timeouts and failures gracefully

**Acceptance Criteria:**
- If agent exceeds timeout, cancel and return `AgentResult(success=False, error="timeout")`
- If agent raises exception, catch and return `AgentResult(success=False, error=str(exc))`
- Orchestration continues with remaining agents
- Failed agents don't prevent successful agents from contributing

**Test Approach:**
- Create 3 agents: fast (5s), timeout (120s), exception (raises ValueError)
- Set timeout to 10s
- Execute in parallel
- Assert fast succeeds
- Assert timeout fails with "timed out after 10s"
- Assert exception fails with ValueError message
- Assert orchestration completes successfully with fast agent's output

**Constitution Check:** ✅ Test outcomes (system should handle failures gracefully)

---

## Data Structures

```python
class ParallelOrchestrationMode(str, Enum):
    FORK_JOIN = "fork_join"
    BARRIER_SYNC = "barrier_sync"
    STREAMING = "streaming"

@dataclass
class ParallelExecutionConfig:
    max_concurrent_agents: int = 5
    sync_mode: ParallelOrchestrationMode = ParallelOrchestrationMode.BARRIER_SYNC
    sync_interval_iterations: int = 3
    timeout_per_agent_seconds: float = 60.0
    allow_early_termination: bool = True

class ParallelCognitiveOrchestrator(MultiAgentOrchestrator):
    def __init__(
        self,
        config: ParallelExecutionConfig | None = None,
        belief_service: BeliefReconciliationService | None = None,
        transactive_memory: TransactiveMemory | None = None,
    )

    async def orchestrate_parallel(
        self,
        agents: list[AgentSpec],
        context: TaskContext,
        agent_executor: AgentExecutor,
    ) -> OrchestratorResult

    async def _run_agent_with_shared_memory(
        self,
        agent: AgentSpec,
        context: TaskContext,
        shared_memory: SharedWorkingMemory,
        agent_executor: AgentExecutor,
    ) -> AgentResult

    async def _barrier_sync_execution(...) -> list[AgentResult]
    async def _fork_join_execution(...) -> list[AgentResult]
    async def _streaming_execution(...) -> list[AgentResult]
    async def _route_by_expertise(...) -> list[AgentSpec]
    async def _update_expertise(...) -> None
    async def _reconcile_conflicts(...) -> None
```

---

## Integration Points

### Upstream Dependencies
- `MultiAgentOrchestrator` (base class)
- `SharedWorkingMemory` (FR-001)
- `BeliefReconciliationService` (FR-003, optional)
- `TransactiveMemory` (FR-004, optional)
- `AgentSpec`, `TaskContext`, `AgentResult` (existing orchestration)

### Downstream Consumers
- Main orchestration entry point
- Multi-agent task execution
- Benchmark harness (MultiAgentBench, MemoryAgentBench)

---

## Success Criteria

### Performance Metrics
- **Speedup**: 3-agent parallel execution is 2.5x faster than sequential (allowing for coordination overhead)
- **Throughput**: Handle 5 concurrent agents without significant latency increase
- **Reliability**: 99%+ success rate with proper timeout/error handling

### Cognitive Metrics
- **Coordination Score**: >80% of shared observations accessed by multiple agents
- **Conflict Rate**: Conflicts detected and flagged within 100ms of creation
- **Expertise Routing Accuracy**: 85%+ queries routed to highest-expertise agents

---

## Testing Strategy

### Unit Tests
- `test_parallel_execution()`: 5 agents run concurrently
- `test_orchestration_modes()`: All 3 modes execute correctly
- `test_shared_memory_injection()`: Agents access same SharedWorkingMemory
- `test_timeout_handling()`: Timeouts handled gracefully
- `test_failure_handling()`: Exceptions don't crash orchestration

### Integration Tests
- `test_belief_reconciliation_in_parallel()`: Conflicts trigger reconciliation
- `test_expertise_routing()`: Agents reordered by expertise
- `test_barrier_sync_coordination()`: Barriers prevent drift

### Stress Tests
- `test_max_concurrency()`: 10 agents (above optimal) still work
- `test_rapid_conflicts()`: 20 agents producing conflicting observations
- `test_long_running_tasks()`: Agents running for 5+ minutes

---

## Implementation Tasks

1. **Core Orchestrator** (3 days)
   - Extend `MultiAgentOrchestrator`
   - Implement `orchestrate_parallel()` main flow
   - Timeout handling with `asyncio.wait_for()`

2. **Shared Memory Integration** (2 days)
   - Inject `__shared__` into TaskContext
   - Capture agent outputs as observations
   - Auto-populate source_agent_id, confidence

3. **Orchestration Modes** (3 days)
   - `_fork_join_execution()`: Simple `asyncio.gather()`
   - `_barrier_sync_execution()`: Loop with `asyncio.wait()` and periodic checks
   - `_streaming_execution()`: Continuous shared memory updates

4. **Expertise Routing** (2 days)
   - `_route_by_expertise()` using transactive memory
   - `_update_expertise()` post-execution
   - Handle case where transactive_memory is None

5. **Conflict Handling** (2 days)
   - `_reconcile_conflicts()` integration
   - Trigger reconciliation at barriers and end
   - Pass conflicts to belief service

6. **Testing** (3 days)
   - Unit tests for each mode
   - Integration tests with mock agents
   - Stress tests with many agents

**Total Estimate:** 15 days

---

## Open Questions

[NEEDS CLARIFICATION: Should BARRIER_SYNC pause all agents at barriers, or just reconcile conflicts asynchronously?]
**Recommendation:** Reconcile asynchronously (don't pause agents). Pausing would block parallel benefits. Agents can continue while conflicts resolve in background.

---

## Constitution Compliance

- ✅ **LLM-First**: No regex for coordination logic
- ✅ **Research-Grounded**: Anthropic research (3-5 agents optimal), MAST framework
- ✅ **Async-First**: All execution is async
- ✅ **Protocol-Based**: Uses TaskContext, AgentSpec, AgentResult protocols
- ✅ **Cognitive Authenticity**: Belief reconciliation, expertise tracking, shared memory
- ✅ **Test Outcomes**: Tests verify correct parallel execution, not specific sync mechanisms

---

**Document Status:** Draft
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
