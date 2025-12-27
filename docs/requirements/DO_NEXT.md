# DO NEXT

**Just say: "do next" or "continue requirements"**

I'll read this file, do the current task, update status, and move to the next step.

---

## CURRENT TASK

```
PHASE: 1 - Memory System
REQ: REQ-001-06
NAME: Migration script for existing memories
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
| 02 | LayeredMemoryProvider with 4-layer support | [x] | [x] | [x] | [x] |
| 03 | Memory promotion service integration | [x] | [x] | [x] | [x] |
| 04 | Scope-based access control | [x] | [x] | [x] | [x] |
| 05 | Roxy adapter using LayeredMemoryProvider | [x] | [x] | [x] | [x] |
| 06 | Migration script for existing memories | [ ] | [ ] | [ ] | [ ] |
| 07 | Unit tests (≥90% coverage) | [ ] | [ ] | [ ] | [ ] |
| 08 | Integration tests with Qdrant | [ ] | [ ] | [ ] | [ ] |
| 09 | Performance benchmarks | [ ] | [ ] | [ ] | [ ] |

**Phase 1 Status:** IN PROGRESS (5/9 complete)

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

### REQ-001-05: Roxy adapter using LayeredMemoryProvider

**Status:** ✅ COMPLETED

**Work Done:**
- Created `RoxyLayeredAdapter` class in `src/draagon_ai/adapters/roxy.py`
- Implements same interface as Roxy's MemoryService (drop-in replacement)
- Complete type/scope mappings:
  - `ROXY_TYPE_MAPPING` - RoxyMemoryType → MemoryType
  - `DRAAGON_TYPE_MAPPING` - MemoryType → RoxyMemoryType
  - `ROXY_SCOPE_MAPPING` - RoxyMemoryScope → MemoryScope
  - `DRAAGON_SCOPE_MAPPING` - MemoryScope → RoxyMemoryScope
- Implemented core methods:
  - `store()` - Store memories via LayeredMemoryProvider
  - `search()` - Search with type filtering
  - `get_by_id()` - Get memory by ID
  - `update_memory()` - Update memory fields
  - `delete()` - Delete memory
  - `reinforce_memory()` - Boost memory importance
  - `search_with_self_rag()` - Simplified self-RAG search
- Exposed promotion/consolidation methods:
  - `promote_all()` - Run full promotion cycle
  - `consolidate()` - Run decay + cleanup + promotion
  - `get_promotion_stats()` - Get promotion statistics
- Updated `src/draagon_ai/adapters/__init__.py` to export adapter

**Test Results:** ✅ 329/329 PASSED
- 29 new adapter tests across 7 test classes
- 300 existing memory tests continue passing

**Files Changed:**
- `src/draagon_ai/adapters/roxy.py` (NEW - ~520 lines)
- `src/draagon_ai/adapters/__init__.py` (UPDATED)
- `tests/adapters/test_roxy_adapter.py` (NEW - ~560 lines)

**Review Results:** ✅ READY
- All acceptance criteria met
- Complete type mappings in both directions
- Backward compatible with Roxy's MemoryService interface
- Promotion/consolidation exposed for background jobs

---

### REQ-001-04: Scope-based access control

**Status:** ✅ COMPLETED

**Work Done:**
- Added scope hierarchy constants (`SCOPE_HIERARCHY`, `SCOPE_TYPE_MAPPING`)
- Implemented helper functions:
  - `get_scope_level()` - Returns hierarchy level (0=WORLD, 4=SESSION)
  - `get_accessible_scopes()` - Returns all scopes readable from a given scope
  - `can_scope_read()` - Check if query scope can read target scope
  - `can_scope_write()` - Check if query scope can write to target scope
- Added scope enforcement config options to `LayeredMemoryConfig`:
  - `enforce_scope_permissions` - Enable/disable scope checking
  - `default_agent_id` - Default agent for scope inference
  - `default_context_id` - Default context for scope inference
- Added scope validation to `store()` method (raises `PermissionError`)
- Added scope filtering to `search()` method with `caller_scope` parameter
- Updated `__all__` exports with new functions

**Test Results:** ✅ 300/300 PASSED (full memory test suite)
- 23 new scope access control tests across 4 test classes:
  - `TestScopeHierarchyHelpers` (7 tests)
  - `TestScopeEnforcementStore` (8 tests)
  - `TestScopeEnforcementSearch` (6 tests)
  - `TestScopeTypeMappingIntegration` (2 tests)

**Files Changed:**
- `src/draagon_ai/memory/providers/layered.py` (UPDATED - added ~80 lines)
- `tests/memory/test_layered_provider.py` (UPDATED - added ~200 lines)

**Review Results:** ✅ READY
- All core acceptance criteria met (T01-T03)
- Scope hierarchy correctly implemented
- Write permission enforcement working
- Search filtering by accessible scopes working
- Note: Full cross-agent isolation (T04) deferred to integration tests (REQ-001-08)

---

### REQ-001-03: Memory promotion service integration

**Status:** ✅ COMPLETED

**Work Done:**
- Integrated existing `MemoryPromotion` and `MemoryConsolidator` services into `LayeredMemoryProvider`
- Added promotion config settings to `LayeredMemoryConfig`:
  - Working → Episodic thresholds (importance, access, min_age)
  - Episodic → Semantic thresholds
  - Semantic → Metacognitive thresholds
  - Batch size and max_per_cycle limits
- Added `get_promotion_config()` helper method
- Created promotion and consolidator instances in `_setup_layers()`
- Added convenience methods:
  - `promote_all()` → Returns PromotionStats
  - `promote_working_to_episodic()` → Returns count
  - `promote_episodic_to_semantic()` → Returns count
  - `promote_semantic_to_metacognitive()` → Returns count
  - `consolidate()` → Returns full stats dict (decay + cleanup + promotion)
  - `get_promotion_stats()` → Returns current promotion service stats
- Added `promotion` and `consolidator` properties
- Updated exports in `__all__`

**Test Results:** ✅ 277/277 PASSED (full memory test suite)
- 18 new promotion integration tests
- All existing tests continue passing

**Files Changed:**
- `src/draagon_ai/memory/providers/layered.py` (UPDATED - added ~120 lines)
- `tests/memory/test_layered_provider.py` (UPDATED - added ~150 lines)

**Review Results:** ✅ READY
- All acceptance criteria met
- Only cosmetic issues found (docstring enhancement, type ignore comments)
- Background scheduler intentionally out of scope (library exposes methods; scheduler is deployment concern)

---

### REQ-001-02: LayeredMemoryProvider with 4-layer support

**Status:** ✅ COMPLETED

**Work Done:**
- Extended `LayeredMemoryConfig` with Qdrant settings (url, api_key, collections, embedding_dimension)
- Added per-layer TTL settings with getter methods (working_ttl, episodic_ttl, semantic_ttl, metacognitive_ttl)
- Updated `LayeredMemoryProvider` with `embedding_provider` parameter
- Added async `initialize()` method for Qdrant setup (creates QdrantGraphStore when configured)
- Added `_setup_layers()` method that configures all 4 layers with proper TTLs
- Added `close()` method for cleanup
- Added `_ensure_initialized()` guard to prevent use before initialization
- Added `is_initialized` and `uses_qdrant` properties
- In-memory mode auto-initializes; Qdrant mode requires explicit `initialize()` call
- Fixed all MemoryProvider ABC signature mismatches
- Fixed layer API method calls (add_fact, add_skill, add_insight, add_event)
- Implemented cross-layer search (all 4 layers)

**Test Results:** ✅ 259/259 PASSED (full memory test suite)
- 29 LayeredMemoryProvider tests (all passing)
- All other memory tests passing (230 additional tests)

**Files Changed:**
- `src/draagon_ai/memory/providers/layered.py` (UPDATED - ~200 lines)
- `tests/memory/test_layered_provider.py` (NEW - ~365 lines)

**Review Results:** ✅ READY - All issues fixed

| Issue | Severity | Resolution |
|-------|----------|------------|
| 1 | CRITICAL | ✅ Fixed `store()` signature with keyword-only marker |
| 2 | CRITICAL | ✅ Fixed `search()` to use `memory_types`/`scopes` lists |
| 3 | CRITICAL | ✅ Fixed `update()` to return `Memory \| None` |
| 4 | CRITICAL | ✅ Fixed `get()` and `delete()` signatures |
| 5 | HIGH | ✅ Fixed layer method names (add_fact, add_skill, add_insight) |
| 6 | HIGH | ✅ Implemented cross-layer search (all 4 layers) |
| 7 | MEDIUM | N/A - ABC has default implementation |

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

