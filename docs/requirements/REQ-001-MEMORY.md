# REQ-001: Memory System Migration

**Priority:** High
**Estimated Effort:** Large (2-3 weeks)
**Dependencies:** None (Foundation)
**Blocks:** REQ-002, REQ-003, REQ-004, REQ-005, REQ-006

---

## 1. Overview

### 1.1 Current State
- **Roxy** uses a simple `MemoryService` with flat Qdrant storage
- **draagon-ai** has a sophisticated 4-layer memory system that is NOT being used:
  - `WorkingMemory` - Session-scoped, limited capacity (7±2 items)
  - `EpisodicMemory` - Autobiographical experiences with chronological linking
  - `SemanticMemory` - Factual knowledge, entities, relationships
  - `MetacognitiveMemory` - Skills, strategies, insights
- Roxy's memory adapter (`RoxyMemoryAdapter`) implements `MemoryProvider` but doesn't use layers

### 1.2 Target State
- Roxy uses `LayeredMemoryProvider` from draagon-ai
- All 4 memory layers are functional with Qdrant backend
- Memory promotion automatically moves items between layers
- Scopes control access (WORLD/CONTEXT/AGENT/USER/SESSION)
- Existing Roxy memories are migrated to new structure

### 1.3 Success Metrics
- Memory retrieval quality improves (measured by benchmark suite)
- Memory promotion runs successfully in background
- Scope-based queries return correct results
- No data loss during migration

---

## 2. Detailed Requirements

### 2.1 Qdrant Backend for TemporalCognitiveGraph

**ID:** REQ-001-01
**Priority:** Critical

#### Description
Implement a Qdrant-backed storage for `TemporalCognitiveGraph`. Currently the graph is in-memory only.

#### Acceptance Criteria
- [x] `QdrantGraphStore` class implements graph persistence
- [x] Nodes are stored with bi-temporal timestamps (valid_time, transaction_time)
- [x] Edges are stored with relationship metadata
- [x] Graph can be loaded from Qdrant on startup
- [x] Graph changes are persisted incrementally (not full rewrite)
- [x] Vector embeddings are stored for semantic search

#### Implementation Notes (2025-12-27)
- Implemented as `QdrantGraphStore` extending `TemporalCognitiveGraph`
- Bi-temporal: event_time, ingestion_time, valid_from, valid_until
- Edges stored with placeholder 1-dim vector (Qdrant requires vectors)
- Lazy loading: nodes/edges loaded from Qdrant on first access if not in memory
- Type-weighted importance scoring for search results
- 15 unit tests passing with mocked Qdrant client

#### Technical Notes
```python
# Target API
class QdrantGraphStore:
    async def save_node(self, node: TemporalNode) -> str: ...
    async def save_edge(self, edge: TemporalEdge) -> str: ...
    async def load_graph(self, scope: HierarchicalScope) -> TemporalCognitiveGraph: ...
    async def search_nodes(self, query: str, node_types: list[NodeType], limit: int) -> list[GraphSearchResult]: ...
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Save fact node | Node persisted, ID returned | Unit |
| T02 | Load graph with 1000 nodes | Graph loaded in <2s | Integration |
| T03 | Search nodes by embedding | Top-k results by similarity | Integration |
| T04 | Bi-temporal query | Only valid-time-matching nodes | Unit |

---

### 2.2 LayeredMemoryProvider Implementation

**ID:** REQ-001-02
**Priority:** Critical

#### Description
Ensure `LayeredMemoryProvider` works end-to-end with all 4 layers using Qdrant backend.

#### Acceptance Criteria
- [ ] All 4 layer types can store and retrieve memories
- [ ] Layer-appropriate TTLs are enforced
- [ ] Cross-layer queries work correctly
- [ ] Provider implements full `MemoryProvider` protocol
- [ ] Configuration via `LayeredMemoryConfig`

#### Technical Notes
```python
# Configuration
config = LayeredMemoryConfig(
    working_capacity=7,
    working_ttl_seconds=300,
    episodic_ttl_days=14,
    semantic_ttl_days=180,
    metacognitive_ttl_days=365,
    qdrant_url="http://192.168.168.216:6333",
    collection_name="draagon_memories",
)

provider = LayeredMemoryProvider(config, embedding_provider)
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Store to working memory | Item in working layer | Unit |
| T02 | Store skill | Item in metacognitive layer | Unit |
| T03 | Search across layers | Results from all matching layers | Integration |
| T04 | TTL expiration | Expired items not returned | Integration |

---

### 2.3 Memory Promotion Service

**ID:** REQ-001-03
**Priority:** High

#### Description
Implement automatic memory promotion between layers based on access patterns, importance, and consolidation rules.

#### Acceptance Criteria
- [ ] Working → Episodic promotion on session end
- [ ] Episodic → Semantic promotion for repeated patterns
- [ ] Semantic → Metacognitive promotion for skills/strategies
- [ ] Promotion runs as background job
- [ ] Promotion stats are logged and trackable
- [ ] Manual promotion trigger available

#### Promotion Rules
```
Working Memory Items:
- After session ends → Promote important items to Episodic
- Threshold: importance > 0.5 OR access_count > 3

Episodic Memories:
- Pattern detected across 3+ episodes → Extract to Semantic
- Time threshold: 7 days without similar episode

Semantic Facts:
- Becomes procedural knowledge → Promote to Metacognitive
- Threshold: used in 5+ successful tool executions
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Session end with 5 items | 3 promoted (importance > 0.5) | Integration |
| T02 | Same fact in 3 episodes | Extracted to semantic | Integration |
| T03 | Skill used 5 times | Promoted to metacognitive | Integration |
| T04 | Promotion stats | Accurate counts returned | Unit |

---

### 2.4 Hierarchical Scope Access Control

**ID:** REQ-001-04
**Priority:** High

#### Description
Implement scope-based access control so queries only return appropriately scoped memories.

#### Scope Hierarchy
```
WORLD - Global facts (capitals, dates) - all agents can read
  └─ CONTEXT - Shared within context (household) - context members read/write
      └─ AGENT - Agent's private memories - only this agent
          └─ USER - Per-user within agent - only this user
              └─ SESSION - Current conversation only - ephemeral
```

#### Acceptance Criteria
- [ ] Queries respect scope hierarchy
- [ ] Lower scopes can read higher scopes (SESSION can read WORLD)
- [ ] Higher scopes cannot read lower scopes (WORLD cannot read USER)
- [ ] Write permissions enforced per scope
- [ ] Scope validation on all memory operations

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | USER query | Returns USER + AGENT + CONTEXT + WORLD | Unit |
| T02 | WORLD query | Returns only WORLD | Unit |
| T03 | Write to higher scope | Permission denied | Unit |
| T04 | Cross-agent query | Only shared scopes visible | Integration |

---

### 2.5 Roxy Adapter for LayeredMemoryProvider

**ID:** REQ-001-05
**Priority:** High

#### Description
Create/update Roxy adapter to use `LayeredMemoryProvider` instead of direct Qdrant access.

#### Acceptance Criteria
- [ ] `RoxyMemoryAdapter` uses `LayeredMemoryProvider`
- [ ] All existing Roxy memory operations continue working
- [ ] Memory types map correctly to layers
- [ ] Scopes map correctly to hierarchy
- [ ] Backward compatible with existing code

#### Mapping
```python
# Roxy MemoryType → draagon-ai Layer
FACT → SemanticMemory (as Fact node)
SKILL → MetacognitiveMemory (as Skill node)
INSIGHT → MetacognitiveMemory (as Insight node)
PREFERENCE → SemanticMemory (as Entity property)
EPISODIC → EpisodicMemory (as Event node)
INSTRUCTION → MetacognitiveMemory (as Strategy node)
KNOWLEDGE → SemanticMemory (as Fact node)

# Roxy Scope → draagon-ai Scope
"private" → USER or AGENT
"shared" → CONTEXT
"system" → WORLD
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Store via Roxy API | Appears in correct layer | Integration |
| T02 | Search via Roxy API | Returns from all layers | Integration |
| T03 | Existing tests pass | No regression | Regression |

---

### 2.6 Migration Script

**ID:** REQ-001-06
**Priority:** Medium

#### Description
Script to migrate existing Roxy memories from flat structure to layered structure.

#### Acceptance Criteria
- [ ] Reads all memories from current collection
- [ ] Classifies each into appropriate layer
- [ ] Creates proper node types (Fact, Skill, Entity, etc.)
- [ ] Preserves all metadata (importance, entities, timestamps)
- [ ] Supports dry-run mode
- [ ] Supports rollback
- [ ] Progress reporting

#### Migration Steps
```
1. Backup current collection
2. Read memories in batches (1000 at a time)
3. Classify each memory by type
4. Create appropriate node in new structure
5. Verify migration completeness
6. Switch Roxy to new provider
7. Archive old collection
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | 100 test memories | All migrated correctly | Integration |
| T02 | Dry run | No changes, report generated | Integration |
| T03 | Rollback | Original state restored | Integration |

---

### 2.7 Unit Tests

**ID:** REQ-001-07
**Priority:** High

#### Coverage Requirements
- Minimum 90% line coverage
- 100% coverage of public APIs
- All error paths tested

#### Test Files
```
tests/
  unit/
    memory/
      test_qdrant_graph_store.py
      test_layered_provider.py
      test_promotion.py
      test_scopes.py
      test_working_memory.py
      test_episodic_memory.py
      test_semantic_memory.py
      test_metacognitive_memory.py
```

#### Mock Strategy
- Mock Qdrant client for unit tests
- Mock embedding provider
- Use fixtures for common node/edge types

---

### 2.8 Integration Tests

**ID:** REQ-001-08
**Priority:** High

#### Test Environment
- Real Qdrant instance (test collection)
- Real embedding provider (Ollama)
- Isolated from production data

#### Test Files
```
tests/
  integration/
    memory/
      test_qdrant_integration.py
      test_full_memory_flow.py
      test_promotion_cycle.py
      test_scope_isolation.py
```

#### Test Scenarios
1. Full lifecycle: create → search → update → promote → search again
2. Multi-user isolation
3. Concurrent access handling
4. Large-scale queries (1000+ memories)

---

### 2.9 Performance Benchmarks

**ID:** REQ-001-09
**Priority:** Medium

#### Benchmark Targets
| Operation | Target | Max Acceptable |
|-----------|--------|----------------|
| Store single memory | <50ms | 100ms |
| Search (top-5) | <100ms | 200ms |
| Promotion cycle (100 items) | <5s | 10s |
| Load graph (1000 nodes) | <2s | 5s |

#### Benchmark Script
```python
# scripts/benchmark_memory.py
async def run_benchmarks():
    results = {
        "store_single": await benchmark_store(),
        "search_top_5": await benchmark_search(),
        "promotion_100": await benchmark_promotion(),
        "load_graph_1000": await benchmark_load(),
    }
    report_results(results)
```

---

## 3. Implementation Plan

### 3.1 Sequence
1. Implement `QdrantGraphStore` (REQ-001-01)
2. Wire up `LayeredMemoryProvider` (REQ-001-02)
3. Implement promotion service (REQ-001-03)
4. Add scope access control (REQ-001-04)
5. Update Roxy adapter (REQ-001-05)
6. Write migration script (REQ-001-06)
7. Complete unit tests (REQ-001-07)
8. Complete integration tests (REQ-001-08)
9. Run benchmarks (REQ-001-09)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| Data loss during migration | Backup + rollback capability |
| Performance regression | Benchmark before/after |
| Breaking existing functionality | Full regression test suite |
| Qdrant schema changes | Versioned schemas |

---

## 4. Review Checklist

Use this checklist for the god-level review:

### Functional Completeness
- [ ] All 4 memory layers work independently
- [ ] Cross-layer search returns correct results
- [ ] Promotion moves items correctly
- [ ] Scopes enforce access control
- [ ] Roxy adapter maintains compatibility

### Code Quality
- [ ] No hardcoded values (all configurable)
- [ ] Error handling for all Qdrant operations
- [ ] Logging at appropriate levels
- [ ] Type hints complete
- [ ] Docstrings on all public methods

### Test Coverage
- [ ] Unit tests ≥ 90%
- [ ] All layers have dedicated tests
- [ ] Edge cases covered (empty results, timeouts, etc.)
- [ ] Mocks are appropriate

### Performance
- [ ] Benchmarks meet targets
- [ ] No N+1 queries
- [ ] Batch operations used where appropriate
- [ ] Caching considered

### Documentation
- [ ] API docs complete
- [ ] Migration guide written
- [ ] Architecture diagrams updated

---

## 5. God-Level Review Prompt

```
MEMORY SYSTEM REVIEW - REQ-001

Context: Memory system migration from flat Roxy storage to draagon-ai's
4-layer cognitive memory with Qdrant backend.

Review the implementation against these specific criteria:

1. LAYER FUNCTIONALITY
   - Can I store and retrieve from each layer independently?
   - Are TTLs enforced correctly for each layer?
   - Does working memory respect capacity limits (7±2)?
   - Are episodic memories chronologically linked?
   - Are semantic entities properly deduplicated?
   - Are metacognitive skills tracked with effectiveness?

2. PROMOTION LOGIC
   - Does working → episodic promotion trigger on session end?
   - Does episodic → semantic detect patterns across 3+ episodes?
   - Does semantic → metacognitive happen for repeated skills?
   - Are promotion stats accurate?
   - Does manual promotion work?

3. SCOPE ENFORCEMENT
   - Can USER scope read WORLD but not vice versa?
   - Are write permissions correctly restricted?
   - Does cross-agent isolation work?
   - Are scope violations logged?

4. QDRANT INTEGRATION
   - Is bi-temporal indexing working?
   - Are embeddings stored correctly?
   - Is batch loading efficient?
   - Are concurrent writes handled?

5. ROXY COMPATIBILITY
   - Do all existing Roxy tests pass?
   - Is the adapter truly thin?
   - Are memory types mapped correctly?
   - Is there any duplicated logic?

6. MIGRATION SAFETY
   - Does dry-run work correctly?
   - Can we rollback if needed?
   - Is progress accurately reported?
   - Are all memories accounted for?

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

