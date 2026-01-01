# Reasoning Loop Audit

**Audit Date:** 2025-12-31
**Auditor:** Claude (Opus 4.5)
**Scope:** 9-step probabilistic graph reasoning pipeline

---

## Executive Summary

The reasoning loop implements a 9-step pipeline for probabilistic graph reasoning. This audit evaluates each step for **completeness**, **quality**, **testability**, and **gaps** relative to the design specification in `PROBABILISTIC_GRAPH_REASONING.md`.

### Overall Assessment

| Aspect | Rating | Notes |
|--------|--------|-------|
| **Completeness** | 60% | Steps 1-3, 5, 7-8 implemented; 4, 6, 9 incomplete |
| **Quality** | 75% | Good patterns, needs hardening |
| **Testability** | 80% | Good test coverage, some gaps |
| **Production Readiness** | 50% | MVP-level, needs Phase 2 work |

---

## Step-by-Step Audit

### Step 1: Semantic Graph Extraction

**Location:** `loop.py:350-410` (`_simple_extraction`)

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ⚠️ Partial | Uses simplified LLM extraction, not Phase 0/1 pipeline |
| Quality | ⚠️ Medium | XML parsing robust but misses semantic depth |
| Testability | ✅ Good | Mockable LLM, tested via `test_basic_processing` |
| Error Handling | ✅ Good | Falls back to minimal graph on failure |

**Issues Identified:**

1. **TODO Not Resolved:** Line 248 says "TODO: Integrate full IntegratedPipeline when available"
   - The `IntegratedPipeline` exists in prototypes but isn't wired in
   - Missing: WSD, coreference, temporal, presuppositions, commonsense

2. **Shallow Extraction:**
   - Current: Extracts entities and basic relationships
   - Design: Should extract all Phase 1 attributes (negation, modality, sentiment, etc.)
   - Impact: `classify_phase1_content()` returns mostly `ENTITY` types

3. **Config Flag Unused:**
   - `config.use_phase01_extraction` exists but isn't checked

**Recommendations:**
- [ ] Wire in `IntegratedPipeline` from prototypes
- [ ] Add extraction depth configuration (simple/full)
- [ ] Add extraction quality metrics to result

---

### Step 2: Recency Context Management

**Location:** `context.py:23-86` (`RecencyWindow`)

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ✅ Full | Implements sliding window with decay |
| Quality | ✅ Good | Clean design, configurable |
| Testability | ✅ Good | Unit tested in `test_reasoning_loop.py` |
| Error Handling | ✅ Good | Handles empty window gracefully |

**Issues Identified:**

1. **No Graph Deduplication:**
   - Same entities across messages are treated as separate
   - Could merge overlapping nodes for efficiency

2. **Time Decay Not Configurable at Runtime:**
   - `time_decay=0.9` is fixed in dataclass

3. **Missing Timestamp Storage:**
   - Stores datetime but doesn't persist across sessions
   - Recency window resets on restart

**Recommendations:**
- [ ] Add entity deduplication across graphs
- [ ] Make time_decay configurable via ReasoningConfig
- [ ] Consider session persistence for recency window

---

### Step 3: Probabilistic Expansion

**Location:** `expander.py:179-358` (`ProbabilisticExpander`)

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ✅ Full | Generates branches with probabilities |
| Quality | ✅ Good | XML prompt, normalization, sorting |
| Testability | ✅ Good | Mock-friendly, unit tested |
| Error Handling | ✅ Good | Falls back to single branch |

**Issues Identified:**

1. **Graph Building Limited:**
   - `InterpretationBranch._build_graph()` only handles subject/predicate/object
   - Missing: temporal, location, modifiers, nested structures

2. **Synset Not Validated:**
   - Accepts any `synset="..."` string from LLM
   - No validation against WordNet

3. **Probability Parsing Fragile:**
   - Regex `probability="([\d.]+)"` could match malformed values
   - No range validation before normalization

4. **No Caching:**
   - Same messages re-expand each time
   - Could cache expansion results for common patterns

**Recommendations:**
- [ ] Extend `_build_graph` for richer structures
- [ ] Add synset validation against NLTK WordNet
- [ ] Add probability bounds checking (0.0-1.0)
- [ ] Add expansion result caching

---

### Step 4: Recursive Deepening

**Location:** `loop.py:267` (skipped)

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ❌ Not Implemented | Comment: "Skip recursive deepening for MVP" |
| Quality | N/A | |
| Testability | N/A | |

**Design Specification:**
```python
async def deepen_branch(branch, depth=2):
    result = await pipeline.process(branch.interpretation)
    deeper_graph = builder.build(result)
    branch.graph.merge(deeper_graph)
```

**Gap Analysis:**
- Purpose: Add semantic depth to high-probability branches
- Impact: Branches have shallow graphs (just S-P-O)
- Priority: Medium (would improve context retrieval quality)

**Recommendations:**
- [ ] Implement `deepen_branch()` method
- [ ] Add depth limit configuration
- [ ] Only deepen branches above probability threshold

---

### Step 5: Context Retrieval from Neo4j

**Location:** `context.py:122-264` (`ContextRetriever`)

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ✅ Full | BFS traversal with depth limit |
| Quality | ⚠️ Medium | Loads full graph then filters (inefficient) |
| Testability | ✅ Good | Neo4j integration tests exist |
| Error Handling | ✅ Good | Returns empty context if no store |

**Issues Identified:**

1. **Inefficient Traversal:**
   - Line 203: `full_graph = self.store.load(instance_id)` loads entire instance
   - Then does in-memory BFS
   - Design: Should use native Neo4j `MATCH path = ...` traversal

2. **No Memory Layer Awareness:**
   - Retrieves all nodes regardless of expiration
   - Should filter by `memory_expires_at` and layer

3. **Anchor Matching by ID Only:**
   - `anchor_ids = [n.node_id for n in anchor_nodes]`
   - If new message creates nodes with different IDs, won't match stored nodes
   - Should match by `canonical_name` or synset

4. **Missing Relevance Scoring:**
   - All retrieved nodes treated equally
   - Design mentions "Score and prune based on relevance to query"

**Recommendations:**
- [ ] Add Neo4j native traversal query
- [ ] Filter by memory layer and expiration
- [ ] Match anchors by name/synset, not just ID
- [ ] Add relevance scoring (TF-IDF, embedding similarity)

---

### Step 6: Best Interpretation Selection (Beam Search ReAct)

**Location:** `loop.py:289-291`

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ❌ Minimal | Just picks highest probability branch |
| Quality | ⚠️ Low | No actual beam search or ReAct |
| Testability | N/A | Not implemented |

**Current Implementation:**
```python
# Step 6: Select best interpretation (simple for MVP - just pick highest prob)
# TODO: Implement beam search ReAct
result.best_interpretation = expansion.top_branch
```

**Design Specification:**
- `BeamSearchReAct` class with parallel beam exploration
- Scoring based on action success, observation coherence
- Adaptive pruning strategies
- This is described as "KEY INNOVATION" in design doc

**Gap Analysis:**
- Most critical missing piece
- Currently: Static probability selection
- Should: Dynamic exploration with feedback

**Recommendations:**
- [ ] Create `beam_search.py` with `BeamSearchReAct` class
- [ ] Implement `BeamState` dataclass
- [ ] Add scoring logic based on ReAct outcomes
- [ ] Implement adaptive pruning (Option C from design)

---

### Step 7: Memory Reinforcement

**Location:** `loop.py:293-311`

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ✅ Implemented | Reinforces context nodes |
| Quality | ⚠️ Medium | Reinforces all context, not just used |
| Testability | ✅ Good | Integration tests pass |
| Error Handling | ✅ Good | Try/except with logging |

**Issues Identified:**

1. **Reinforces All Context Nodes:**
   ```python
   context_node_ids = [n.node_id for n in result.retrieved_context.subgraph.iter_nodes()]
   ```
   - Reinforces everything retrieved, not just what was useful
   - Design: "Boost retrieved context that was actually used"

2. **No Negative Reinforcement:**
   - Only boosts, never penalizes
   - Failed interpretations should have lower reinforcement

3. **Fixed Reinforcement Amount:**
   - `amount=self.config.reinforcement_amount` (0.1)
   - Should vary by how useful the node was

4. **Missing Edge Reinforcement:**
   - Only reinforces nodes
   - Design: "Boost edges used in winning interpretation"

**Recommendations:**
- [ ] Track which context nodes were actually used in response
- [ ] Add negative reinforcement for failed paths
- [ ] Variable reinforcement based on usage importance
- [ ] Add edge reinforcement

---

### Step 8: Graph Persistence

**Location:** `loop.py:313-340`

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ✅ Full | Memory-aware saving with TTL |
| Quality | ✅ Good | Content type classification, layer support |
| Testability | ✅ Good | 56 memory tests pass |
| Error Handling | ✅ Good | Try/except with fallback |

**Issues Identified:**

1. **No Confidence Threshold:**
   - Persists all nodes regardless of confidence
   - Design: "Persist if: High confidence (winning beam score > threshold)"

2. **No Novelty Check:**
   - May store duplicate information
   - Design: "Novel information (not already in graph)"

3. **Always Uses Default Layer:**
   - `default_layer=self.config.default_memory_layer` (WORKING)
   - Higher-confidence facts could start in higher layer

4. **Missing Selective Persistence:**
   - `SelectivePersistence` class from design not implemented
   - Should extract facts from successful trajectories

**Recommendations:**
- [ ] Add confidence threshold for persistence
- [ ] Implement novelty detection (check existing graph)
- [ ] Dynamic layer assignment based on confidence
- [ ] Implement `SelectivePersistence` class

---

### Step 9: Response Generation

**Location:** Not implemented

| Criterion | Status | Notes |
|-----------|--------|-------|
| Completeness | ❌ Not Implemented | No response synthesis |
| Quality | N/A | |
| Testability | N/A | |

**Design Specification:**
```python
class ResponseSynthesizer:
    async def synthesize(self, beam_result, original_message) -> str:
        if beam_result.convergence > 0.8:
            return self._confident_response(beam_result.best_beam)
        elif beam_result.best_beam.score > 0.7:
            return self._explain_interpretation(beam_result.best_beam)
        else:
            return self._clarification_question(beam_result)
```

**Gap Analysis:**
- The pipeline processes but doesn't generate output
- `ReasoningResult.best_interpretation` is set but not used to generate text
- Caller must interpret the result themselves

**Recommendations:**
- [ ] Create `synthesizer.py` with `ResponseSynthesizer` class
- [ ] Add convergence measurement
- [ ] Implement clarification question generation
- [ ] Add to `ReasoningResult`: `response_text: str`

---

## Cross-Cutting Concerns

### Error Handling

| Component | Error Strategy | Quality |
|-----------|---------------|---------|
| Graph Extraction | Fallback to minimal graph | ✅ Good |
| Expansion | Fallback to single branch | ✅ Good |
| Context Retrieval | Return empty context | ✅ Good |
| Reinforcement | Log warning, continue | ✅ Good |
| Persistence | Log warning, continue | ✅ Good |
| Neo4j Connection | Lazy init with None fallback | ✅ Good |

**Overall:** Error handling is consistent and graceful.

### Performance

| Concern | Status | Notes |
|---------|--------|-------|
| Neo4j Lazy Init | ✅ | Only connects when needed |
| Context Caching | ❌ | Full graph loaded each time |
| Expansion Caching | ❌ | No result caching |
| Parallel Processing | ❌ | Sequential steps, no async gather |

**Recommendations:**
- [ ] Add LRU cache for context retrieval
- [ ] Cache expansion results for similar messages
- [ ] Use `asyncio.gather` for parallel steps

### Observability

| Metric | Tracked | Location |
|--------|---------|----------|
| Extraction Time | ✅ | `result.extraction_time_ms` |
| Expansion Time | ✅ | `result.expansion_time_ms` |
| Retrieval Time | ✅ | `result.retrieval_time_ms` |
| Total Time | ✅ | `result.total_time_ms` |
| Nodes Added | ✅ | `result.nodes_added` |
| Nodes Reinforced | ✅ | `result.nodes_reinforced` |
| Content Types | ✅ | `result.content_types_used` |

**Good:** Comprehensive timing and counts.
**Missing:**
- [ ] LLM call counts
- [ ] Cache hit/miss rates
- [ ] Branch selection reasoning

---

## Test Coverage Analysis

### Existing Tests

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_reasoning_loop.py` | 15 | Loop, Expander, Context |
| `test_memory.py` | 56 | Memory layers, TTL, reinforcement |

### Coverage Gaps

| Component | Missing Tests |
|-----------|--------------|
| Phase 0/1 Integration | No tests for full pipeline integration |
| Beam Search ReAct | Not implemented, no tests |
| Response Synthesis | Not implemented, no tests |
| Edge Reinforcement | Not implemented, no tests |
| Anchor Matching | No tests for name-based matching |

---

## Recommendations Summary

### High Priority (Blocking Production)

1. **Implement Beam Search ReAct (Step 6)**
   - Currently just picks highest probability
   - Core innovation from design doc is missing

2. **Wire in Phase 0/1 Pipeline (Step 1)**
   - Shallow extraction limits downstream quality
   - `IntegratedPipeline` exists in prototypes

3. **Fix Context Anchor Matching (Step 5)**
   - Current ID-based matching won't find stored nodes
   - Need name/synset-based matching

### Medium Priority (Quality Improvement)

4. **Implement Recursive Deepening (Step 4)**
   - Add semantic depth to promising branches

5. **Add Response Synthesis (Step 9)**
   - Pipeline processes but doesn't respond

6. **Improve Reinforcement Logic (Step 7)**
   - Track actual usage, not just retrieval
   - Add edge reinforcement

### Low Priority (Optimization)

7. **Add Neo4j Native Traversal**
   - Current in-memory BFS is inefficient

8. **Add Caching**
   - Context and expansion caching

9. **Add Confidence Threshold**
   - Only persist high-confidence nodes

---

## Conclusion

The reasoning loop provides a solid foundation with good patterns for error handling, testing, and observability. The main gaps are:

1. **Beam Search ReAct** - The "KEY INNOVATION" is not implemented
2. **Phase 0/1 Integration** - Using simplified extraction
3. **Response Synthesis** - No output generation

The memory system (Step 7-8) is well-implemented with the new temporal memory model. Context retrieval (Step 5) works but has efficiency and matching issues.

**Recommended Next Phase:**
1. Implement BeamSearchReAct class
2. Wire in IntegratedPipeline for Phase 0/1
3. Add ResponseSynthesizer
4. Fix anchor matching in context retrieval

This would bring the implementation to ~85% completeness relative to the design specification.

---

## Appendix A: Existing Components Being Reinvented

**CRITICAL:** The reasoning loop is recreating functionality that already exists in draagon-ai. This is a significant waste of effort and creates maintenance burden.

### Components Already Built (Should Reuse)

| What We Built | What Already Exists | Location |
|---------------|---------------------|----------|
| `VolatileWorkingMemory` in memory.py | `SharedWorkingMemory` | `orchestration/shared_memory.py` |
| Simple ReAct selection in loop.py | Full ReAct loop with THOUGHT→ACTION→OBSERVATION | `orchestration/loop.py` |
| No parallel execution | `ParallelMultiAgentOrchestrator` with barrier sync | `orchestration/parallel_orchestrator.py` |
| Simple extraction | `IntegratedPipeline` (Phase 0+1) | `cognition/decomposition/extractors/integrated_pipeline.py` |
| No graph building | `GraphBuilder` | `cognition/decomposition/graph/builder.py` |

### Detailed Comparison

#### 1. Working Memory Duplication

**New (reasoning/memory.py):**
```python
class VolatileWorkingMemory:
    """Volatile working memory shared across ReAct swarm."""
    def __init__(self, task_id: str, max_items: int = 9):  # Miller's Law 7±2
```

**Existing (orchestration/shared_memory.py):**
```python
class SharedWorkingMemory:
    """Task-scoped working memory that enables multiple agents to coordinate."""
    # Also has:
    # - attention_weight and decay
    # - role-based filtering (CRITIC, RESEARCHER, EXECUTOR)
    # - conflict detection
    # - belief candidate tracking
```

**Recommendation:** Delete `VolatileWorkingMemory`, use `SharedWorkingMemory` directly.

#### 2. ReAct Loop Duplication

**New (reasoning/loop.py Step 6):**
```python
# Step 6: Select best interpretation (simple for MVP - just pick highest prob)
# TODO: Implement beam search ReAct
result.best_interpretation = expansion.top_branch
```

**Existing (orchestration/loop.py):**
```python
class LoopMode(Enum):
    SIMPLE = "simple"  # Single-step
    REACT = "react"    # Multi-step: THOUGHT → ACTION → OBSERVATION loop
    AUTO = "auto"      # Auto-detect complexity

class ReActStep:
    type: StepType  # THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER
    content: str
    action_name: str | None
    action_args: dict
```

**Recommendation:** Wire the reasoning loop INTO the existing ReAct loop, not parallel to it.

#### 3. Parallel/Swarm Duplication

**New:** No parallel execution, sequential steps

**Existing (orchestration/parallel_orchestrator.py):**
```python
class ParallelMultiAgentOrchestrator:
    """True parallel execution with synchronization."""
    # Features:
    # - DependencyResolver (topological sort for execution waves)
    # - SyncMode: FORK_JOIN, BARRIER_SYNC, STREAMING
    # - Streaming pub/sub observation broadcasting
    # - Miller's Law (7±2) capacity enforcement
```

**Recommendation:** Use `ParallelMultiAgentOrchestrator` for beam search parallel exploration.

#### 4. Extraction Pipeline

**New (reasoning/loop.py):**
```python
async def _simple_extraction(self, message: str) -> SemanticGraph:
    # Just extracts entities and basic relationships via LLM
```

**Existing (cognition/decomposition/extractors/integrated_pipeline.py):**
```python
class IntegratedPipeline:
    # Full Phase 0 + Phase 1:
    # - Content analysis (PROSE, CODE, DATA, CONFIG, MIXED)
    # - WSD with synset resolution
    # - Entity type classification
    # - Semantic decomposition (10 attribute types)
    # - Document chunking for large content
```

**Recommendation:** Wire `IntegratedPipeline` into Step 1.

---

## Appendix B: Missing Document Loaders

**Finding:** No dedicated markdown/document loader exists for ingesting files into the semantic graph.

**What Exists:**
- `ContentAnalyzer` - Analyzes content type (prose, code, data, config)
- `PromptLoader` - Loads prompts from domain dictionaries into Qdrant
- No file-based document ingestion

**What's Needed:**
```python
class DocumentLoader:
    """Load documents into semantic knowledge graph."""

    async def load_file(self, path: Path) -> SemanticGraph:
        """Load a file, run through Phase 0/1, store in Neo4j."""

    async def load_directory(self, path: Path, pattern: str = "**/*.md") -> list[SemanticGraph]:
        """Recursively load documents matching pattern."""

    async def load_with_chunking(self, content: str, chunk_size: int = 4000) -> SemanticGraph:
        """Chunk large content and process each chunk."""
```

---

## Appendix C: Dual Pipeline Architecture

**User's Vision:** Run BOTH pipelines - "normal" way AND semantic approach - in parallel.

### Proposed Architecture

```
User Message
     │
     ├──────────────────────────────────────┐
     │                                       │
     ▼                                       ▼
┌─────────────┐                    ┌──────────────────┐
│ Normal Path │                    │ Semantic Path    │
│ (Current)   │                    │ (Graph Reasoning)│
│             │                    │                  │
│ • Qdrant    │                    │ • Phase 0/1      │
│   vector    │                    │ • Neo4j graph    │
│   search    │                    │ • Probabilistic  │
│ • Single    │                    │   expansion      │
│   context   │                    │ • Multi-hop      │
│             │                    │   retrieval      │
└──────┬──────┘                    └────────┬─────────┘
       │                                     │
       ▼                                     ▼
┌──────────────────────────────────────────────────────┐
│                  Context Merger                       │
│                                                       │
│ • Deduplicate by content hash                        │
│ • Merge overlapping entities                          │
│ • Weight by confidence & recency                      │
│ • Resolve conflicts                                   │
└──────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────┐
│              Unified ReAct Agent                      │
│                                                       │
│ • Has access to both context sources                  │
│ • Uses existing orchestration/loop.py                 │
│ • Parallel beam exploration via ParallelOrchestrator  │
└──────────────────────────────────────────────────────┘
       │
       ▼
   Response
```

### Key Component: Context Deduplication Cache

```python
class ContextDeduplicationCache:
    """Prevents re-processing identical text snippets."""

    def __init__(self, ttl_seconds: int = 300):  # 5 minute cache
        self._cache: dict[str, tuple[datetime, Any]] = {}

    def content_hash(self, text: str) -> str:
        """Create hash of normalized text."""
        normalized = " ".join(text.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    async def get_or_process(
        self,
        text: str,
        processor: Callable[[str], Awaitable[Any]],
    ) -> Any:
        """Return cached result or process and cache."""
        h = self.content_hash(text)

        if h in self._cache:
            ts, result = self._cache[h]
            if (datetime.now() - ts).total_seconds() < self.ttl_seconds:
                return result

        result = await processor(text)
        self._cache[h] = (datetime.now(), result)
        return result
```

### Integration Points

1. **Normal Path**: Uses existing `QdrantMemoryProvider` for vector search
2. **Semantic Path**: Uses new `ReasoningLoop` → Neo4j
3. **Merger**: New `ContextMerger` class that:
   - Hashes context snippets to detect duplicates
   - Tracks which path each piece came from
   - Weights results (semantic path may have higher confidence for multi-hop)
4. **ReAct**: Existing `orchestration/loop.py` with `LoopMode.REACT`

### Benefits of Dual Pipeline

| Aspect | Normal Path | Semantic Path | Combined |
|--------|-------------|---------------|----------|
| Speed | ✅ Fast (vector only) | ⚠️ Slower (graph traversal) | Configurable timeout |
| Depth | ⚠️ Single-hop | ✅ Multi-hop reasoning | Best of both |
| Explainability | ⚠️ Opaque embeddings | ✅ Traceable graph paths | Auditable |
| Cost | ✅ Low (no LLM calls for retrieval) | ⚠️ Higher (LLM for expansion) | Tunable |

### Implementation Priority

1. **Phase 1**: Wire existing components together
   - Connect `IntegratedPipeline` to `ReasoningLoop`
   - Use `SharedWorkingMemory` instead of `VolatileWorkingMemory`
   - Use `ParallelMultiAgentOrchestrator` for beam search

2. **Phase 2**: Add document loader
   - `DocumentLoader` class for file ingestion
   - Chunking support for large documents
   - Neo4j storage integration

3. **Phase 3**: Dual pipeline
   - `ContextMerger` for combining results
   - `ContextDeduplicationCache` for efficiency
   - Configuration to enable/disable each path
