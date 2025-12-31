# FR-008: Multi-Agent Integration for Roxy

**Feature:** Wire ParallelCognitiveOrchestrator into Roxy's agent system
**Status:** Planned
**Priority:** P1 (Enhancement)
**Complexity:** Medium-High
**Dependencies:** FR-001 (SharedWorkingMemory), FR-002 (ParallelCognitiveOrchestrator)

---

## Overview

Integrate the implemented SharedWorkingMemory (FR-001) and ParallelCognitiveOrchestrator (FR-002) into Roxy's production agent system. This enables multi-agent collaboration patterns within Roxy's existing orchestration architecture.

### Current State

- FR-001 `SharedWorkingMemory`: Implemented, tested, exported - NOT used
- FR-002 `ParallelCognitiveOrchestrator`: Implemented, tested, exported - NOT used
- Roxy's `AgentOrchestrator`: Uses `asyncio.gather` for parallelism but no shared memory

### Target State

Roxy can leverage multi-agent patterns for improved response quality and parallel processing with proper context sharing.

---

## Use Cases

### Use Case A: Parallel Context Gathering with Shared Memory

**Goal:** Replace ad-hoc `asyncio.gather` with proper shared working memory

**Current Pattern (orchestrator.py:568):**
```python
(_, profile_result, intent_result) = await asyncio.gather(
    self.users.increment_interaction_count(user_id),
    self.users.process_profile_update(user_id, query, llm_service=self.llm),
    self._classify_intent(query, conversation, debug_info),
)
```

**Proposed Pattern:**
```python
from draagon_ai.orchestration import SharedWorkingMemory

shared = SharedWorkingMemory(task_id=conversation_id)

# Context gathering agents add observations
await shared.add_observation(
    content=f"User intent: {intent_result.intent}",
    source_agent_id="intent_classifier",
    attention_weight=0.9,
    belief_type="FACT",
)

# Downstream processing sees shared context
context = await shared.get_context_for_agent(
    agent_id="decision_engine",
    role=AgentRole.EXECUTOR,
)
```

**Value:**
- Observations are typed and weighted
- Conflict detection between parallel results
- Audit trail of what each "agent" contributed

---

### Use Case B: Critic Agent for Response Validation

**Goal:** Add a review step before sending responses to catch errors

**Architecture:**
```
User Query
    │
    ▼
┌─────────────────┐
│ Primary Agent   │──────────────────┐
│ (generates      │                  │
│  response)      │                  │
└─────────────────┘                  │
    │                                │
    ▼                                ▼
┌─────────────────┐         ┌─────────────────┐
│ Critic Agent    │◄───────►│ SharedWorking   │
│ (validates      │         │ Memory          │
│  response)      │         └─────────────────┘
└─────────────────┘
    │
    ▼
  Response (if valid) OR Retry

```

**Implementation:**
```python
from draagon_ai.orchestration import ParallelCognitiveOrchestrator, AgentRole

orchestrator = ParallelCognitiveOrchestrator(
    agents=[
        AgentConfig(id="primary", role=AgentRole.EXECUTOR),
        AgentConfig(id="critic", role=AgentRole.CRITIC),
    ],
    sync_mode=SyncMode.BARRIER_SYNC,  # Critic waits for primary
)

result = await orchestrator.process(task_context)
```

**Value:**
- Catches factual errors before user sees them
- Validates tool calls are appropriate
- Can request clarification if response is ambiguous

---

### Use Case C: Specialist Agents

**Goal:** Domain-specific agents that can work in parallel

**Specialists:**
| Agent | Domain | Triggered By |
|-------|--------|--------------|
| `calendar_agent` | Calendar/scheduling | Time-related queries |
| `ha_agent` | Home Assistant devices | Device control queries |
| `memory_agent` | Personal knowledge | "Remember", "what did I" |
| `search_agent` | Web/knowledge search | Unknown facts |

**Architecture:**
```
User Query: "Turn on the lights and add a meeting tomorrow"
    │
    ▼
┌─────────────────┐
│ Router Agent    │ (detects multi-domain)
└─────────────────┘
    │
    ├──────────────────┬──────────────────┐
    ▼                  ▼                  ▼
┌──────────┐    ┌──────────┐    ┌──────────────────┐
│ HA Agent │    │ Calendar │    │ SharedWorking    │
│          │◄──►│ Agent    │◄──►│ Memory           │
└──────────┘    └──────────┘    └──────────────────┘
    │                  │
    ▼                  ▼
  Merged Response: "Done! Lights on and meeting added"
```

**Implementation:**
```python
orchestrator = ParallelCognitiveOrchestrator(
    agents=[
        AgentConfig(id="ha", role=AgentRole.EXECUTOR, domains=["home_assistant"]),
        AgentConfig(id="calendar", role=AgentRole.EXECUTOR, domains=["calendar"]),
    ],
    sync_mode=SyncMode.WAIT_ALL,
    merge_strategy=MergeStrategy.CONCATENATE,
)
```

**Value:**
- Parallel execution for multi-domain queries
- Each specialist optimized for its domain
- Shared memory prevents conflicts (e.g., scheduling over existing meeting)

---

## Functional Requirements

### FR-008.1: Feature Flag for Multi-Agent Mode

**Requirement:** Enable multi-agent orchestration via configuration

**Acceptance Criteria:**
- `settings.enable_multi_agent: bool` flag
- When disabled, existing single-agent behavior unchanged
- When enabled, SharedWorkingMemory created per conversation
- Gradual rollout possible (per-user, percentage-based)

---

### FR-008.2: SharedWorkingMemory Integration

**Requirement:** Create SharedWorkingMemory for each conversation

**Acceptance Criteria:**
- Memory scoped to conversation_id
- Observations from context gathering stored
- Conflicts detected and logged
- Memory cleaned up on conversation expiry

---

### FR-008.3: Critic Agent (Use Case B)

**Requirement:** Optional response validation before sending

**Acceptance Criteria:**
- `settings.enable_critic_agent: bool` flag
- Critic reviews response for:
  - Factual consistency with context
  - Appropriate tool usage
  - Tone and safety
- If critic rejects, primary agent retries (max 2 retries)
- Latency budget: Critic adds max 500ms

---

### FR-008.4: Specialist Agent Router (Use Case C)

**Requirement:** Route multi-domain queries to specialist agents

**Acceptance Criteria:**
- Router detects multi-domain queries (e.g., "lights AND calendar")
- Spawns appropriate specialist agents
- Specialists run in parallel with WAIT_ALL sync
- Results merged into single response
- Fallback to single agent if routing fails

---

## Non-Functional Requirements

### Performance

- SharedWorkingMemory overhead: < 5ms per observation
- Critic agent latency: < 500ms additional
- Specialist parallel execution: faster than sequential

### Reliability

- Single-agent fallback if multi-agent fails
- Graceful degradation under high load
- Circuit breaker for specialist agent failures

### Observability

- Metrics: multi_agent_enabled, critic_rejections, specialist_usage
- Logging: which agents contributed to each response
- Debug mode: full SharedWorkingMemory dump

---

## Implementation Plan

### Phase 1: Foundation (Use Case A)
1. Add feature flag `enable_shared_memory`
2. Create SharedWorkingMemory in AgentOrchestrator.__init__
3. Store observations from existing parallel operations
4. Log conflicts for analysis
5. No behavior change - observation only

### Phase 2: Critic Agent (Use Case B)
1. Add feature flag `enable_critic_agent`
2. Implement CriticAgent class
3. Wire into response generation pipeline
4. Add retry logic with backoff
5. A/B test against baseline

### Phase 3: Specialist Agents (Use Case C)
1. Add router detection for multi-domain queries
2. Implement specialist agent configs
3. Wire ParallelCognitiveOrchestrator
4. Implement result merging
5. Performance optimization

---

## Success Metrics

| Metric | Baseline | Target |
|--------|----------|--------|
| Response accuracy | Current | +5% with critic |
| Multi-domain latency | 2x sequential | < 1.3x sequential |
| Error rate | Current | -20% with critic |
| User satisfaction | Current | +10% |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Increased latency | Strict latency budgets, parallel execution |
| Complexity increase | Feature flags, gradual rollout |
| Agent conflicts | SharedWorkingMemory conflict detection |
| Cost increase (more LLM calls) | Critic uses fast model, specialists share context |

---

## Related Requirements

- FR-001: SharedWorkingMemory (implemented)
- FR-002: ParallelCognitiveOrchestrator (implemented)
- FR-003: Belief Reconciliation (planned)
- FR-004: Transactive Memory (planned)
