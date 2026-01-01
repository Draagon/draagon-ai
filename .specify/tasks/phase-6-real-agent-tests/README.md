# Phase 6: Real Agent Integration Tests (FR-010)

**Phase Goal:** Implement comprehensive end-to-end integration tests for real agent behavior using actual LLM providers, Neo4j memory, and cognitive services.

**Total Effort:** ~14 days
**Priority:** High
**Status:** Ready to start

---

## Overview

This phase extends the FR-009 testing framework from mock agents to real production components. We test actual agent behavior with real LLM calls, real database operations, and real cognitive services.

**Core Principle:** Test real agent behavior with real components. Validate that the cognitive architecture (memory, learning, beliefs, reasoning) works correctly when integrated, not just in isolation.

---

## Task Breakdown

### TASK-009: Real Agent Test Fixtures (1 day) - P0
**Status:** Pending
**Files:** `tests/integration/agents/conftest.py`

Create test fixtures for real agent integration tests:
- `embedding_provider` - Real embedding provider (session-scoped)
- `real_llm` - Groq or OpenAI based on env vars (session-scoped)
- `memory_provider` - Neo4jMemoryProvider with semantic decomposition (function-scoped)
- `tool_registry` - Clean ToolRegistry per test (function-scoped)
- `agent` - Fully configured AgentLoop (function-scoped)
- `evaluator` - LLM-as-judge evaluator (function-scoped)
- `advance_time` - Test utility for TTL testing

**Pre-work:**
- None - can start immediately
- Requires: Neo4j running, GROQ_API_KEY or OPENAI_API_KEY

---

### TASK-010: Core Agent Processing Tests (FR-010.1) (2 days) - P0
**Status:** Pending
**Dependencies:** TASK-009
**Files:** `tests/integration/agents/test_agent_core.py`

Test core agent query → decision → action → response flow:
- Simple query → direct answer (no tools)
- Query requiring tool → tool execution → answer
- Confidence-based responses (low confidence = hedging)
- Error recovery (graceful degradation)
- Session persistence across queries
- LLM tier selection (simple → local, complex → complex)

**Pre-work:**
- Add `model_tier` to AgentResponse if missing (~1 hour)

---

### TASK-011: Memory Integration Tests (FR-010.2) (3 days) - P0
**Status:** Pending
**Dependencies:** TASK-009, TASK-010
**Files:** `tests/integration/agents/test_agent_memory.py`

Test 4-layer cognitive memory architecture:
- Memory storage and recall across sessions
- Semantic search relevance
- Memory reinforcement (boost on success, demote on failure)
- Layer promotion (working → episodic → semantic → metacognitive)
- Layer demotion (failed memories move down)
- TTL enforcement and expiration
- Importance-based ranking

**Pre-work:**
- Wire memory usage tracking to AgentLoop (~4 hours)
- Add `used_memory_ids` to AgentResponse
- Track memory access in DecisionEngine

---

### TASK-012: Learning Integration Tests (FR-010.3) (2 days) - P1
**Status:** Pending
**Dependencies:** TASK-009, TASK-011
**Files:** `tests/integration/agents/test_agent_learning.py`

Test agent learning from interactions:
- Autonomous skill extraction from successful executions
- Fact learning from user statements
- Correction acceptance and belief updates
- Skill verification (demote broken skills)
- Multi-user knowledge scoping
- Fact vs skill classification

**Pre-work:**
- Verify LearningService integration (~6 hours)
- Wire post-response learning hooks
- Implement correction detection (LLM-based, NOT regex)

---

### TASK-013: Belief Reconciliation Tests (FR-010.4) (2 days) - P1
**Status:** Pending
**Dependencies:** TASK-009, TASK-011, TASK-012
**Files:** `tests/integration/agents/test_agent_beliefs.py`

Test belief formation and reconciliation:
- Conflict detection between observations
- Credibility-weighted belief formation
- Clarification question queueing
- Multi-user observation handling (household vs personal)
- Belief confidence calibration
- Gradual belief formation from multiple observations

**Pre-work:**
- Verify BeliefReconciliationService API (~2 hours)
- Verify CuriosityService API for clarification queueing

---

### TASK-014: ReAct Reasoning Tests (FR-010.5) (1.5 days) - P2
**Status:** Pending
**Dependencies:** TASK-009, TASK-010
**Files:** `tests/integration/agents/test_agent_react.py`

Test multi-step ReAct reasoning:
- THOUGHT → ACTION → OBSERVATION traces
- Tools invoked within reasoning loop
- Observations integrated into reasoning
- Final answer synthesis
- Max steps limit enforcement
- ReAct vs Simple mode comparison

**Pre-work:**
- Verify ReAct implementation (~2 hours)
- Add `react_trace` to AgentResponse if missing

---

### TASK-015: Tool Execution Tests (FR-010.6) (1.5 days) - P1
**Status:** Pending
**Dependencies:** TASK-009, TASK-010
**Files:** `tests/integration/agents/test_agent_tools.py`

Test tool system lifecycle:
- Tool discovery from `@tool` decorator
- Parameter validation before execution
- Timeout enforcement (terminate long-running tools)
- Error handling and graceful degradation
- Metrics collection (invocation count, success rate, latency)
- Tool selection based on description

**Pre-work:**
- Verify ToolRegistry metrics API (~1 hour)

---

### TASK-016: Multi-Agent Coordination Tests (FR-010.7) (2 days) - P2
**Status:** Pending
**Dependencies:** TASK-009, TASK-010
**Files:** `tests/integration/agents/test_agent_multiagent.py`

Test multi-agent coordination via SharedWorkingMemory:
- Observation sharing between agents
- Role-based context filtering (CRITIC, RESEARCHER, EXECUTOR)
- Attention weighting and decay
- Belief candidate identification
- Concurrent access safety (no data loss)
- Miller's Law capacity limits (7±2 items)

**Pre-work:**
- Verify SharedWorkingMemory API (~2 hours)
- Create multi-agent AppProfiles

---

### TASK-017: CI/CD Integration (1 day) - P3 DEFERRED
**Status:** Deferred (per FR-011 design decision)
**Dependencies:** TASK-009 through TASK-016, FR-011
**Files:** `.github/workflows/agent-integration.yml`

Integrate tests into GitHub Actions CI/CD:
- Create workflow with Neo4j service
- Configure LLM API secrets
- Add cost monitoring
- Enable manual trigger only

**When to Enable:**
- All tasks complete and stable
- Local tests >95% pass rate
- Cost per run validated (<$1)

---

## Implementation Sequence

### Week 1: Foundation (Days 1-5)
**Day 1:**
- ✅ TASK-009: Real agent fixtures (setup infrastructure)

**Days 2-3:**
- ✅ TASK-010: Core agent tests (validate basic wiring)

**Days 4-5:**
- ⏳ TASK-011: Memory tests (start - most complex)

### Week 2: Cognitive Features (Days 6-10)
**Days 6-7:**
- ✅ TASK-011: Memory tests (complete)
- ⏳ TASK-012: Learning tests (start)

**Days 8-9:**
- ✅ TASK-012: Learning tests (complete)
- ✅ TASK-013: Belief reconciliation tests

**Day 10:**
- ✅ TASK-015: Tool execution tests

### Week 3: Advanced Features (Days 11-14)
**Days 11-12:**
- ✅ TASK-014: ReAct reasoning tests

**Days 13-14:**
- ✅ TASK-016: Multi-agent coordination tests
- 📝 Documentation and cleanup

**Future:**
- ⏸️ TASK-017: CI/CD integration (when ready)

---

## Pre-Implementation Checklist

Before starting Phase 6, ensure:

- [ ] FR-009 complete (TASK-001 through TASK-008 done) ✅
- [ ] Neo4j 5.26+ running locally (bolt://localhost:7687)
- [ ] GROQ_API_KEY or OPENAI_API_KEY environment variable set
- [ ] Embedding provider configured
- [ ] All FR-009 tests passing (161 tests)

---

## Success Criteria

**Quantitative:**
- Test coverage: >80% of agent features
- Pass rate: >95% on stable tests
- Flakiness: <5% flake rate
- Performance: All tests within latency requirements (see FR-010)
- LLM costs: <$1 per full test suite run

**Qualitative:**
- Cognitive validation: Memory, learning, beliefs work in practice
- Integration confidence: Full pipeline tested end-to-end
- Regression detection: Catches integration bugs before production
- Documentation: Tests serve as usage examples

---

## Cost Estimates

| Task | LLM Calls | Estimated Cost |
|------|-----------|----------------|
| TASK-010 (Core Agent) | ~20-30 | $0.02-0.05 |
| TASK-011 (Memory) | ~10-20 | $0.02-0.05 |
| TASK-012 (Learning) | ~30-40 | $0.05-0.10 |
| TASK-013 (Beliefs) | ~30-40 | $0.05-0.10 |
| TASK-014 (ReAct) | ~50-70 | $0.10-0.20 |
| TASK-015 (Tools) | ~10-20 | $0.02-0.05 |
| TASK-016 (Multi-Agent) | ~40-60 | $0.10-0.15 |
| **TOTAL** | ~190-280 | **$0.36-0.70** |

**Note:** Costs based on Groq pricing (~$0.0002/call). Actual costs may vary.

---

## Performance Targets

| Test Category | Max Latency | Success Rate |
|---------------|-------------|--------------|
| Core Agent | 2s (simple), 5s (tool) | >95% |
| Memory | 1s (store/retrieve) | >99% |
| Learning | 3s (extraction) | >70% |
| Beliefs | 2s (reconciliation) | >85% |
| ReAct | 10s (multi-step) | >80% |
| Tools | 500ms (execution) | >90% |
| Multi-Agent | 5s (coordination) | >85% |

---

## Related Specifications

- **FR-009**: Integration Testing Framework (provides infrastructure)
- **FR-010**: Real Agent Integration Tests (this phase)
- **FR-011**: CI/CD Infrastructure (future automation)

---

## Notes

**Focus on Outcomes:**
- Test outcomes, not processes (per constitution)
- Use LLM-as-judge for semantic validation (NO string matching)
- Validate results, not implementation details

**Cost Control:**
- Session-scoped `real_llm` reduces API costs
- Function-scoped `memory_provider` ensures clean state
- Consider `USE_MOCK_LLM` env var for free local testing

**Flakiness Handling:**
- LLM responses are non-deterministic
- Use LLM-as-judge for robust semantic validation
- Retry flaky tests up to 2 times with pytest-rerunfailures

---

**Ready to Start:** TASK-009 (Real Agent Fixtures)
**Next:** Set up test infrastructure, then proceed with core agent tests
