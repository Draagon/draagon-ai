# draagon-ai Architecture Migration - Delivery Checklist

**Created:** 2025-12-27
**Status:** Planning
**Reference:** [ARCHITECTURE_REVISED.md](../ARCHITECTURE_REVISED.md)

---

## Overview

This checklist tracks the migration of generic assistant functionality from roxy-voice-assistant into draagon-ai core, and the refactoring of Roxy to use these core systems.

### Success Criteria
- [ ] All generic assistant functionality lives in draagon-ai
- [ ] Roxy is a thin implementation layer (adapters + personality + channel)
- [ ] Full test coverage (unit + integration + E2E)
- [ ] Memory MCP Server enables cross-app knowledge sharing
- [ ] ReAct loop enables multi-step reasoning

---

## Phase 1: Memory System Migration

**Requirement Doc:** [REQ-001-MEMORY.md](./REQ-001-MEMORY.md)
**Priority:** High (Foundation for everything else)
**Estimated Effort:** Large

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 1.1 | Qdrant backend for TemporalCognitiveGraph | [x] | QdrantGraphStore with bi-temporal persistence, 15 tests passing |
| 1.2 | LayeredMemoryProvider with 4-layer support | [ ] | |
| 1.3 | Memory promotion service integration | [ ] | |
| 1.4 | Scope-based access control | [ ] | |
| 1.5 | Roxy adapter using LayeredMemoryProvider | [ ] | |
| 1.6 | Migration script for existing memories | [ ] | |
| 1.7 | Unit tests (≥90% coverage) | [ ] | |
| 1.8 | Integration tests with Qdrant | [ ] | |
| 1.9 | Performance benchmarks | [ ] | |

### Review Gate
```
Review Prompt: "Review the memory system implementation against REQ-001-MEMORY.md.
Verify: (1) All 4 layers are functional, (2) Promotion works correctly,
(3) Scopes enforce access control, (4) Tests cover edge cases,
(5) Performance meets benchmarks. List any gaps or issues."
```

---

## Phase 2: Orchestrator Migration

**Requirement Doc:** [REQ-002-ORCHESTRATOR.md](./REQ-002-ORCHESTRATOR.md)
**Priority:** High (Core agent loop)
**Estimated Effort:** Large

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 2.1 | AgentLoop with ReAct support | [ ] | |
| 2.2 | DecisionEngine integration | [ ] | |
| 2.3 | ActionExecutor with tool registry | [ ] | |
| 2.4 | Configurable loop modes (simple/ReAct) | [ ] | |
| 2.5 | Thought trace logging | [ ] | |
| 2.6 | Roxy adapter for orchestration | [ ] | |
| 2.7 | Remove duplicate Roxy orchestrator | [ ] | |
| 2.8 | Unit tests (≥90% coverage) | [ ] | |
| 2.9 | Integration tests | [ ] | |
| 2.10 | Multi-step reasoning E2E tests | [ ] | |

### Review Gate
```
Review Prompt: "Review the orchestrator implementation against REQ-002-ORCHESTRATOR.md.
Verify: (1) ReAct loop produces correct thought traces, (2) Simple mode still works,
(3) Tool execution is reliable, (4) Error handling is robust,
(5) Tests cover multi-step scenarios. List any gaps or issues."
```

---

## Phase 3: Cognitive Services Consolidation

**Requirement Doc:** [REQ-003-COGNITION.md](./REQ-003-COGNITION.md)
**Priority:** Medium (Enhances intelligence)
**Estimated Effort:** Medium

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 3.1 | Belief reconciliation using core service | [ ] | |
| 3.2 | Curiosity engine using core service | [ ] | |
| 3.3 | Opinion formation using core service | [ ] | |
| 3.4 | Learning service using core service | [ ] | |
| 3.5 | Identity manager integration | [ ] | |
| 3.6 | Remove duplicate Roxy cognitive services | [ ] | |
| 3.7 | Adapter layer for Roxy-specific needs | [ ] | |
| 3.8 | Unit tests (≥90% coverage) | [ ] | |
| 3.9 | Integration tests | [ ] | |

### Review Gate
```
Review Prompt: "Review the cognitive services consolidation against REQ-003-COGNITION.md.
Verify: (1) All services use draagon-ai core, (2) Roxy adapters are thin,
(3) No duplicate logic remains, (4) Tests cover belief conflicts,
(5) Personality traits are preserved. List any gaps or issues."
```

---

## Phase 4: Autonomous Agent Core Integration

**Requirement Doc:** [REQ-004-AUTONOMOUS.md](./REQ-004-AUTONOMOUS.md)
**Priority:** Medium (Background intelligence)
**Estimated Effort:** Medium

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 4.1 | Move autonomous agent to draagon-ai core | [ ] | |
| 4.2 | Protocol-based dependency injection | [ ] | |
| 4.3 | Guardrail system with tiers | [ ] | |
| 4.4 | Self-monitoring capability | [ ] | |
| 4.5 | Action logging and dashboard | [ ] | |
| 4.6 | Roxy adapter implementation | [ ] | |
| 4.7 | Remove extension version | [ ] | |
| 4.8 | Unit tests (≥90% coverage) | [ ] | |
| 4.9 | Integration tests | [ ] | |
| 4.10 | Safety E2E tests | [ ] | |

### Review Gate
```
Review Prompt: "Review the autonomous agent implementation against REQ-004-AUTONOMOUS.md.
Verify: (1) Guardrails prevent harmful actions, (2) Self-monitoring detects issues,
(3) Tier system enforces proper approvals, (4) Logging is comprehensive,
(5) Tests cover adversarial scenarios. List any gaps or issues."
```

---

## Phase 5: Memory MCP Server

**Requirement Doc:** [REQ-005-MCP.md](./REQ-005-MCP.md)
**Priority:** Medium (Cross-app integration)
**Estimated Effort:** Medium

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 5.1 | MCP server scaffolding | [ ] | |
| 5.2 | memory.store tool | [ ] | |
| 5.3 | memory.search tool | [ ] | |
| 5.4 | memory.list tool | [ ] | |
| 5.5 | beliefs.reconcile tool | [ ] | |
| 5.6 | Scope-based access control | [ ] | |
| 5.7 | Authentication/authorization | [ ] | |
| 5.8 | Claude Code integration test | [ ] | |
| 5.9 | Unit tests (≥90% coverage) | [ ] | |
| 5.10 | Integration tests | [ ] | |

### Review Gate
```
Review Prompt: "Review the MCP server implementation against REQ-005-MCP.md.
Verify: (1) All tools work with Claude Code, (2) Scopes prevent unauthorized access,
(3) Concurrent access is handled, (4) Error responses follow MCP spec,
(5) Tests cover authentication edge cases. List any gaps or issues."
```

---

## Phase 6: Roxy Thin Client Refactor

**Requirement Doc:** [REQ-006-ROXY.md](./REQ-006-ROXY.md)
**Priority:** High (Final integration)
**Estimated Effort:** Large

### Deliverables

| # | Item | Status | Notes |
|---|------|--------|-------|
| 6.1 | All adapters using draagon-ai | [ ] | |
| 6.2 | Personality config (YAML) | [ ] | |
| 6.3 | Voice/TTS channel handling | [ ] | |
| 6.4 | Home Assistant integration | [ ] | |
| 6.5 | Wyoming protocol support | [ ] | |
| 6.6 | Remove all duplicated code | [ ] | |
| 6.7 | FastAPI shell only | [ ] | |
| 6.8 | Full regression test suite | [ ] | |
| 6.9 | Performance comparison | [ ] | |
| 6.10 | Documentation update | [ ] | |

### Review Gate
```
Review Prompt: "Review the Roxy refactor against REQ-006-ROXY.md.
Verify: (1) Roxy is truly thin (adapters + personality + channel only),
(2) All functionality comes from draagon-ai, (3) Voice pipeline works,
(4) No regression in user experience, (5) Codebase is significantly smaller.
List any gaps or issues."
```

---

## Testing Requirements

### Unit Tests
- Minimum 90% code coverage for new code
- All public APIs have tests
- Edge cases and error conditions covered
- Mocks used appropriately for external dependencies

### Integration Tests
- Cross-component interactions tested
- Database interactions tested with real Qdrant
- LLM interactions tested with mocked responses
- Timeout and error handling verified

### E2E Tests
- Full user scenarios tested
- Voice pipeline tested end-to-end
- Multi-turn conversations tested
- Performance benchmarks established

### Test Documentation
Each requirement doc includes:
- Test case table with inputs/outputs
- Test data requirements
- Mock strategy
- CI/CD integration notes

---

## Definition of "Full Implementation"

A deliverable is considered **fully implemented** when:

1. **Functional Completeness**
   - All specified behaviors work correctly
   - Error handling covers known failure modes
   - Edge cases are handled gracefully

2. **Test Coverage**
   - Unit test coverage ≥ 90%
   - Integration tests pass
   - E2E tests pass (where applicable)

3. **Documentation**
   - Public APIs documented with docstrings
   - Usage examples provided
   - Architecture decisions recorded

4. **Performance**
   - Meets specified latency requirements
   - No memory leaks
   - Scales to expected load

5. **Code Quality**
   - Passes linting (ruff)
   - Type hints complete
   - No TODO/FIXME in critical paths

---

## God-Level Review Process

After completing each phase, use this prompt for comprehensive review:

```
ULTRA-THOROUGH REVIEW PROMPT:

You are reviewing a production-critical system. Be extremely rigorous.

## Context
I have just completed [PHASE NAME] implementation. The requirement document is [REQ-XXX.md].

## Review Tasks

1. **Requirement Verification**
   - Read every line of the requirement doc
   - For each requirement, verify it is implemented
   - For each acceptance criterion, verify it passes
   - List any gaps with severity (Critical/High/Medium/Low)

2. **Code Quality Assessment**
   - Check for code smells, anti-patterns
   - Verify error handling is comprehensive
   - Check for potential race conditions
   - Verify logging is adequate
   - Check for security vulnerabilities

3. **Test Coverage Analysis**
   - Run coverage report and analyze gaps
   - Identify untested edge cases
   - Check test quality (not just quantity)
   - Verify mocks are appropriate

4. **Integration Verification**
   - Check component boundaries are clean
   - Verify dependency injection is correct
   - Check for tight coupling
   - Verify adapters are truly thin

5. **Performance Assessment**
   - Check for N+1 queries
   - Verify caching is appropriate
   - Check for blocking operations
   - Verify timeout handling

6. **Documentation Check**
   - All public APIs documented
   - Architecture decisions recorded
   - Migration notes complete
   - Examples provided

## Output Format
Provide a structured report with:
- PASS/FAIL for each deliverable
- Specific issues found with code references
- Recommended fixes
- Overall readiness assessment (Ready/Needs Work/Major Issues)
```

---

## Progress Tracking

| Phase | Started | Completed | Reviewed | Notes |
|-------|---------|-----------|----------|-------|
| 1. Memory | [x] | [ ] | [ ] | 1.1 done |
| 2. Orchestrator | [ ] | [ ] | [ ] | |
| 3. Cognition | [ ] | [ ] | [ ] | |
| 4. Autonomous | [ ] | [ ] | [ ] | |
| 5. MCP Server | [ ] | [ ] | [ ] | |
| 6. Roxy Refactor | [ ] | [ ] | [ ] | |

---

## Change Log

| Date | Phase | Change | Author |
|------|-------|--------|--------|
| 2025-12-27 | All | Initial checklist created | Claude |
| 2025-12-27 | 1 | REQ-001-01 complete: QdrantGraphStore implemented | Claude |

