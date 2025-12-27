# DO NEXT

**Just say: "do next" or "continue requirements"**

I'll read this file, do the current task, update status, and move to the next step.

---

## CURRENT TASK

```
PHASE: 1 - Memory System
REQ: REQ-001-02
NAME: LayeredMemoryProvider with 4-layer support
STEP: IMPLEMENT
```

---

## STATUS KEY

- `[ ]` = Not started
- `[>]` = In progress (current)
- `[x]` = Complete
- `[!]` = Blocked/Issue

---

## PHASE 1: Memory System (REQ-001)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | Qdrant Backend for TemporalCognitiveGraph | [x] | [x] | [x] | [x] |
| 02 | LayeredMemoryProvider with 4-layer support | [>] | [ ] | [ ] | [ ] |
| 03 | Memory promotion service integration | [ ] | [ ] | [ ] | [ ] |
| 04 | Scope-based access control | [ ] | [ ] | [ ] | [ ] |
| 05 | Roxy adapter using LayeredMemoryProvider | [ ] | [ ] | [ ] | [ ] |
| 06 | Migration script for existing memories | [ ] | [ ] | [ ] | [ ] |
| 07 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 08 | Integration tests with Qdrant | [ ] | [ ] | [ ] | [ ] |
| 09 | Performance benchmarks | [ ] | [ ] | [ ] | [ ] |

**Phase 1 Status:** IN PROGRESS (1/9 complete)

---

## PHASE 2: Orchestrator (REQ-002)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | AgentLoop with ReAct support | [ ] | [ ] | [ ] | [ ] |
| 02 | DecisionEngine integration | [ ] | [ ] | [ ] | [ ] |
| 03 | ActionExecutor with tool registry | [ ] | [ ] | [ ] | [ ] |
| 04 | Configurable loop modes (simple/ReAct) | [ ] | [ ] | [ ] | [ ] |
| 05 | Thought trace logging | [ ] | [ ] | [ ] | [ ] |
| 06 | Roxy adapter for orchestration | [ ] | [ ] | [ ] | [ ] |
| 07 | Remove duplicate Roxy orchestrator | [ ] | [ ] | [ ] | [ ] |
| 08 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 09 | Integration tests | [ ] | [ ] | [ ] | [ ] |
| 10 | Multi-step reasoning E2E tests | [ ] | [ ] | [ ] | [ ] |

**Phase 2 Status:** NOT STARTED

---

## PHASE 3: Cognitive Services (REQ-003)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | Belief reconciliation using core service | [ ] | [ ] | [ ] | [ ] |
| 02 | Curiosity engine using core service | [ ] | [ ] | [ ] | [ ] |
| 03 | Opinion formation using core service | [ ] | [ ] | [ ] | [ ] |
| 04 | Learning service using core service | [ ] | [ ] | [ ] | [ ] |
| 05 | Identity manager integration | [ ] | [ ] | [ ] | [ ] |
| 06 | Remove duplicate Roxy cognitive services | [ ] | [ ] | [ ] | [ ] |
| 07 | Adapter layer for Roxy-specific needs | [ ] | [ ] | [ ] | [ ] |
| 08 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 09 | Integration tests | [ ] | [ ] | [ ] | [ ] |

**Phase 3 Status:** NOT STARTED

---

## PHASE 4: Autonomous Agent (REQ-004)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | Move autonomous agent to draagon-ai core | [ ] | [ ] | [ ] | [ ] |
| 02 | Protocol-based dependency injection | [ ] | [ ] | [ ] | [ ] |
| 03 | Guardrail system with tiers | [ ] | [ ] | [ ] | [ ] |
| 04 | Self-monitoring capability | [ ] | [ ] | [ ] | [ ] |
| 05 | Action logging and dashboard | [ ] | [ ] | [ ] | [ ] |
| 06 | Roxy adapter implementation | [ ] | [ ] | [ ] | [ ] |
| 07 | Remove extension version | [ ] | [ ] | [ ] | [ ] |
| 08 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 09 | Integration tests | [ ] | [ ] | [ ] | [ ] |
| 10 | Safety E2E tests | [ ] | [ ] | [ ] | [ ] |

**Phase 4 Status:** NOT STARTED

---

## PHASE 5: Memory MCP Server (REQ-005)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | MCP server scaffolding | [ ] | [ ] | [ ] | [ ] |
| 02 | memory.store tool | [ ] | [ ] | [ ] | [ ] |
| 03 | memory.search tool | [ ] | [ ] | [ ] | [ ] |
| 04 | memory.list tool | [ ] | [ ] | [ ] | [ ] |
| 05 | beliefs.reconcile tool | [ ] | [ ] | [ ] | [ ] |
| 06 | Scope-based access control | [ ] | [ ] | [ ] | [ ] |
| 07 | Authentication/authorization | [ ] | [ ] | [ ] | [ ] |
| 08 | Claude Code integration test | [ ] | [ ] | [ ] | [ ] |
| 09 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 10 | Integration tests | [ ] | [ ] | [ ] | [ ] |

**Phase 5 Status:** NOT STARTED

---

## PHASE 6: Roxy Thin Client (REQ-006)

| # | Requirement | Implement | Test | Review | Complete |
|---|-------------|-----------|------|--------|----------|
| 01 | All adapters using draagon-ai | [ ] | [ ] | [ ] | [ ] |
| 02 | Personality config (YAML) | [ ] | [ ] | [ ] | [ ] |
| 03 | Voice/TTS channel handling | [ ] | [ ] | [ ] | [ ] |
| 04 | Home Assistant integration | [ ] | [ ] | [ ] | [ ] |
| 05 | Wyoming protocol support | [ ] | [ ] | [ ] | [ ] |
| 06 | Remove all duplicated code | [ ] | [ ] | [ ] | [ ] |
| 07 | FastAPI shell only | [ ] | [ ] | [ ] | [ ] |
| 08 | Full regression test suite | [ ] | [ ] | [ ] | [ ] |
| 09 | Performance comparison | [ ] | [ ] | [ ] | [ ] |
| 10 | Documentation update | [ ] | [ ] | [ ] | [ ] |

**Phase 6 Status:** NOT STARTED

---

## CURRENT WORK LOG

### REQ-001-02: LayeredMemoryProvider with 4-layer support

**Status:** IMPLEMENT step starting

**Work Done:**
- (starting)

---

### REQ-001-01: Qdrant Backend for TemporalCognitiveGraph ✅ COMPLETED

**Work Done:**
- Created `QdrantGraphStore` class extending `TemporalCognitiveGraph`
- Implemented node persistence with bi-temporal timestamps (event_time, ingestion_time, valid_from, valid_until)
- Implemented edge persistence with full metadata
- Implemented `load_from_qdrant()` for graph loading on startup
- All node/edge operations persist incrementally to Qdrant
- Search uses Qdrant's native vector search with scope and type filtering
- Added type-weighted importance scoring for search results
- Lazy loading: nodes/edges loaded from Qdrant on first access if not in memory

**Test Results:** ✅ 15/15 PASSED

**Review Results:** ✅ READY (3 issues found and fixed)

**Files Changed:**
- `src/draagon_ai/memory/providers/qdrant_graph.py` (NEW - ~700 lines)
- `src/draagon_ai/memory/providers/__init__.py` (updated exports)
- `tests/memory/test_qdrant_graph_store.py` (NEW - ~380 lines)

---

## HOW THIS WORKS

When you say "do next" or "continue requirements", I will:

1. Read this file
2. Look at CURRENT TASK
3. Do the current STEP:
   - **IMPLEMENT**: Write the code, update "Work Done" and "Files Changed"
   - **TEST**: Run tests, report results
   - **REVIEW**: Do god-level review, list issues
   - **FIX**: Fix issues found in review
   - **COMPLETE**: Update status, move to next requirement
4. Update this file with progress
5. Advance STEP (or move to next requirement if done)

You can also say:
- "skip this one" - Mark blocked, move to next
- "just review" - Jump to review step
- "status" - I'll summarize where we are
- "phase review" - Full review of current phase

---

## REFERENCE

Full requirement docs:
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-001-MEMORY.md`
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-002-ORCHESTRATOR.md`
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-003-COGNITION.md`
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-004-AUTONOMOUS.md`
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-005-MCP.md`
- `/home/doug/Development/draagon-ai/docs/requirements/REQ-006-ROXY.md`

