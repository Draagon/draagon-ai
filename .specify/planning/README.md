# Implementation Planning Index

**Project:** draagon-ai Cognitive Swarm Architecture
**Status:** Phase 1 Planning Complete
**Last Updated:** 2025-12-30

---

## Overview

This directory contains detailed technical implementation plans for draagon-ai features. Each plan breaks down a functional requirement into module structure, data structures, method implementations, testing strategy, and integration points.

---

## Available Plans

| Plan ID | Feature | Status | Estimate | Dependencies |
|---------|---------|--------|----------|--------------|
| [FR-001](./FR-001-implementation-plan.md) | Shared Cognitive Working Memory | Ready | 10 days | None |

---

## Plan Structure

Each implementation plan includes:

### 1. Executive Summary
- High-level overview
- Key innovations
- Implementation strategy

### 2. Module Structure
- File locations (`src/draagon_ai/...`)
- Class breakdown
- Protocol definitions

### 3. Data Structures
- Dataclasses with full type hints
- Immutability considerations
- Validation logic

### 4. Method Implementations
- Core operations with code examples
- Algorithm descriptions
- Edge case handling

### 5. Integration Points
- Existing code modifications
- Backward compatibility
- Injection patterns

### 6. Testing Strategy
- Unit tests (per-method)
- Integration tests (with existing systems)
- Stress tests (concurrent access, high load)

### 7. Constitution Compliance
- LLM-First validation
- XML Output validation
- Protocol-Based validation
- Async-First validation
- Research-Grounded validation

### 8. Implementation Checklist
- Day-by-day breakdown
- Deliverables per phase
- Verification criteria

---

## Planning Methodology

### Step 1: Read FR Specification
- Understand functional requirements
- Note acceptance criteria
- Identify open questions

### Step 2: Review Architecture
- Check existing module structure
- Review related protocols
- Identify integration points

### Step 3: Design Implementation
- Module placement
- Data structure design
- Method signatures
- Testing approach

### Step 4: Validate Against Constitution
- LLM-First: No semantic regex
- XML Output: All LLM prompts use XML
- Protocol-Based: New integrations via Protocols
- Async-First: Non-blocking operations
- Research-Grounded: Cite papers
- Test Outcomes: Validate results, not processes

### Step 5: Create Detailed Plan
- Code examples for core methods
- Complete test suite outline
- Integration strategy
- Risk mitigation

---

## FR-001: Shared Cognitive Working Memory

**Location:** `src/draagon_ai/orchestration/shared_memory.py`

**Summary:** Task-scoped cognitive working memory enabling multi-agent coordination through attention-weighted observations, Miller's Law capacity management (7±2 items/agent), semantic conflict detection, and role-based context filtering.

**Key Classes:**
- `SharedObservation`: Immutable observation with attribution
- `SharedWorkingMemoryConfig`: Configuration (capacity, decay, threshold)
- `SharedWorkingMemory`: Main class with 15+ methods

**Key Methods:**
- `add_observation()`: Store with conflict detection
- `get_context_for_agent()`: Retrieve filtered by role
- `apply_attention_decay()`: Periodic attention decay
- `get_conflicts()`: Retrieve for reconciliation
- `get_belief_candidates()`: Non-conflicting candidates

**Integration:**
- Inject into `TaskContext.working_memory["__shared__"]`
- Used by `ParallelCognitiveOrchestrator` (FR-002)
- No breaking changes to existing code

**Testing:**
- 25+ unit tests across 6 test classes
- 3 integration tests
- 2 stress tests (concurrency, decay cycles)

**Constitution Compliance:** ✅ 100%
- LLM-First: Embeddings for conflict detection (Phase 2)
- Async-First: All methods `async def`
- Protocol-Based: `EmbeddingProvider` protocol
- Research-Grounded: Miller's Law, Baddeley's Working Memory
- Test Outcomes: Validates correct behavior, not implementation

**Estimate:** 10 days
- Days 1-2: Data structures
- Days 3-4: Capacity & eviction
- Days 5-6: Conflict detection
- Day 7: Attention & access
- Day 8: Role filtering
- Days 9-10: Testing & integration

---

## Next Plans (Planned)

### FR-002: Parallel Multi-Agent Orchestration (15 days)
- **Dependencies:** FR-001
- **Key Classes:** `ParallelCognitiveOrchestrator`, `ParallelExecutionConfig`
- **Integration:** Extends `MultiAgentOrchestrator`, uses `SharedWorkingMemory`

### FR-003: Multi-Agent Belief Reconciliation (13 days)
- **Dependencies:** FR-001, FR-002
- **Key Classes:** `MultiAgentBeliefReconciliation`, `ReconciliationResult`
- **Integration:** Called by orchestrator on conflicts

### FR-004: Transactive Memory System (10 days)
- **Dependencies:** FR-002
- **Key Classes:** `TransactiveMemory`, `ExpertiseEntry`
- **Integration:** Query routing in orchestrator

### FR-005: Metacognitive Reflection Service (11 days)
- **Dependencies:** FR-002, FR-003, FR-004
- **Key Classes:** `MetacognitiveReflectionService`, `ReflectionResult`
- **Integration:** Post-task analysis

---

## Timeline

### Phase 1: Foundation (Weeks 1-6, 38 days)
```
Week 1-2:  FR-001 (10 days) - Shared Working Memory
Week 3-5:  FR-002 (15 days) - Parallel Orchestration
Week 6-7:  FR-003 (13 days) - Belief Reconciliation
```

### Phase 2: Enhancement (Weeks 7-10, 21 days)
```
Week 7-8:  FR-004 (10 days) - Transactive Memory
Week 9-10: FR-005 (11 days) - Metacognitive Reflection
```

**Total:** 59 days sequential, ~38 days on critical path (FR-001 → FR-002 → FR-003)

---

## Constitution Validation Process

Every implementation plan must pass:

1. **LLM-First Check**
   - ✅ No semantic regex patterns
   - ✅ LLM handles all semantic understanding
   - ✅ Exceptions documented (security, TTS, entity IDs)

2. **XML Output Check**
   - ✅ All LLM prompts return XML
   - ✅ No JSON output from LLMs

3. **Protocol-Based Check**
   - ✅ New integrations via Python Protocols
   - ✅ No hard dependencies on implementations

4. **Async-First Check**
   - ✅ All I/O and LLM calls are `async`
   - ✅ Background tasks for non-critical ops

5. **Research-Grounded Check**
   - ✅ Cites peer-reviewed papers
   - ✅ Based on cognitive science or multi-agent research

6. **Test Outcomes Check**
   - ✅ Tests validate results (correct answer)
   - ✅ Tests don't enforce specific implementation paths

---

## Development Workflow

1. **Read Plan**: Review implementation plan thoroughly
2. **Setup Branch**: Create feature branch (`feature/FR-001-shared-memory`)
3. **Implement Phase 1**: Follow day-by-day checklist
4. **Test Continuously**: Run tests after each method
5. **Integration**: Wire into existing code
6. **Review**: Self-review against constitution
7. **Commit**: Commit with descriptive message
8. **Next Phase**: Move to next day's tasks

---

**Document Status:** Active
**Maintainer:** draagon-ai team
**Last Review:** 2025-12-30
