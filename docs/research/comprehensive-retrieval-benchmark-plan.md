# Comprehensive Retrieval Benchmark Plan

**Date:** 2026-01-02
**Status:** Planning

## The Three Retrieval Approaches

| Approach | Description | Best For | Limitations |
|----------|-------------|----------|-------------|
| **Raw Context** | Load full files into LLM context | Small files, simple queries | Scales poorly, token limits |
| **Vector/RAG** | Embed & search chunks | Large docs, similarity search | No relationships, chunk boundaries |
| **Semantic Graph** | Knowledge graph (Neo4j) | Multi-hop, relationships | Requires ingestion, extraction quality |

## What We Have Already

### 1. RetrievalBenchmark Framework (`retrieval_benchmark.py`)
- `SemanticWebProcessor` - queries knowledge graph
- `RawContextProcessor` - injects file content
- `RAGProcessor` - vector similarity search (partial)
- `BenchmarkEvaluator` - LLM-as-judge (correctness, completeness, relevance)
- Winner determination, scale testing, token tracking

### 2. HybridRetrievalOrchestrator (`hybrid_retrieval.py`)
- `QueryAnalyzer` - LLM classifies query type, routes to strategies
- `LocalRetrievalAgent`, `GraphRetrievalAgent`, `VectorRetrievalAgent`
- Query expansion with confidence scores
- Weighted RRF merging

### 3. Embedding Strategy Tests (just completed)
- 4 strategies: RAW, HyDE, Query2Doc, Grounded
- Tiered test cases (Basic → Breaking)
- Holdout set for overfitting detection

### 4. Semantic Expansion Prototype (`prototypes/semantic_expansion/`)
- Frame semantics extraction
- Presupposition detection
- Implication generation
- Word sense disambiguation

## The Plan

### Phase 1: Unify the Benchmarks

Create a single benchmark that tests all 3 approaches with the SAME test cases:

```
Test Case → [Raw Context] → Answer₁
         → [RAG + Best Strategy] → Answer₂
         → [Semantic Graph] → Answer₃
         → LLM-as-Judge evaluates all 3
```

### Phase 2: Query Analysis for Strategy Selection

The LLM should decide which approach(es) to use based on query characteristics:

| Query Characteristic | Best Approach |
|---------------------|---------------|
| "What is X?" (entity lookup) | Semantic Graph |
| "Find similar to..." | Vector/RAG |
| Full file content needed | Raw Context |
| Multi-hop ("Doug's team's auth") | Semantic Graph |
| Keyword search | RAG (with HyDE) |
| Cross-project patterns | Semantic Graph + Vector |

### Phase 3: Scale Testing

Test each approach at different scales:

| Scale | Files | Token Estimate | Expected Winner |
|-------|-------|----------------|-----------------|
| Small | 1-3 files, <5KB each | ~2K tokens | Raw Context (simplest) |
| Medium | 5-10 files, ~50KB total | ~15K tokens | Tie |
| Large | 20+ files, 200KB+ | 50K+ tokens | Semantic Graph |
| Needle | 1 large file, fact buried | 20K+ tokens | Semantic Graph/RAG |

### Phase 4: Query Expansion Decision

When should we expand queries?

| Query Type | Expand? | Strategy |
|------------|---------|----------|
| Ambiguous ("other teams") | Yes | Use graph context |
| Direct ("Engineering team auth") | No | Query directly |
| Vague ("security stuff") | Yes | HyDE expansion |
| Multi-hop ("Doug's team") | Maybe | Graph traversal instead |

## Test Case Categories

### Category 1: Direct Lookup (Semantic Graph should win)
```
Query: "What authentication does the Engineering team use?"
Expected: OAuth2, JWT, PKCE
Why Semantic wins: Direct entity → relationship query
```

### Category 2: Similarity Search (RAG should win)
```
Query: "Find code similar to our retry logic"
Expected: Matching code patterns
Why RAG wins: Vector similarity excels here
```

### Category 3: Full Context Needed (Raw Context should win)
```
Query: "Summarize this file"
Expected: Comprehensive summary
Why Raw wins: Needs complete context
```

### Category 4: Needle in Haystack (Graph/RAG win)
```
Query: "What port does the API use?"
Expected: Specific port number
Why Graph/RAG win: Find specific fact in large context
```

### Category 5: Multi-hop Reasoning (Semantic Graph should win)
```
Query: "What database does Doug's team's service use?"
Expected: Doug → Engineering → CustomerService → PostgreSQL
Why Semantic wins: Relationship traversal
```

### Category 6: Cross-Project Discovery (Semantic Graph wins)
```
Query: "Which other projects handle authentication similarly?"
Expected: List of projects with similar patterns
Why Semantic wins: Cross-document entity linking
```

## Implementation Steps

### Step 1: Create Unified Test Harness
```python
class UnifiedRetrievalBenchmark:
    """Compare all 3 approaches on same test cases."""

    async def run_test_case(self, tc: TestCase) -> dict[str, Result]:
        results = {}

        # Run all 3 approaches in parallel
        results["raw_context"] = await self.raw_processor.process(tc)
        results["rag"] = await self.rag_processor.process(tc)
        results["semantic"] = await self.semantic_processor.process(tc)

        # Evaluate all 3
        for name, result in results.items():
            result.evaluation = await self.evaluator.evaluate(tc, result)

        return results
```

### Step 2: Implement LLM Strategy Selector
```python
class StrategySelector:
    """LLM decides which approach(es) to use."""

    async def select(self, query: str, context: dict) -> list[Strategy]:
        analysis = await self.query_analyzer.analyze(query)

        if analysis.query_type == QueryType.ENTITY_LOOKUP:
            return [Strategy.SEMANTIC]
        elif analysis.query_type == QueryType.SIMILARITY:
            return [Strategy.RAG]
        elif analysis.needs_full_context:
            return [Strategy.RAW]
        else:
            return [Strategy.SEMANTIC, Strategy.RAG]  # Combine
```

### Step 3: Integrate Query Expansion Decision
```python
class ExpansionDecider:
    """Decide if query needs expansion before retrieval."""

    async def should_expand(self, query: str) -> bool:
        # Check for ambiguous terms
        ambiguous_patterns = [
            "other", "similar", "like", "same",
            "those", "they", "it", "the service"
        ]

        # Check if entities are resolvable
        entities = await self.graph.find_entities(query)

        return has_ambiguous_terms or not entities
```

### Step 4: Build Scale Test Suite
```python
def create_scale_tests(project_root: Path) -> list[TestCase]:
    """Create test cases at different scales."""

    return [
        # Small scale (Raw Context wins)
        TestCase(
            files=["README.md"],
            query="What is this project about?",
            expected_winner="raw_context",
        ),

        # Large scale (Semantic wins)
        TestCase(
            files=glob("src/**/*.py"),  # All Python files
            query="What authentication pattern do we use?",
            expected_winner="semantic",
        ),

        # Needle in haystack
        TestCase(
            files=["large_config.yaml"],
            query="What is the API timeout value?",
            expected_winner="rag",  # or semantic
        ),
    ]
```

## Metrics to Track

| Metric | Description | Goal |
|--------|-------------|------|
| Recall | % of expected content found | > 80% |
| Precision | % of results that are relevant | > 70% |
| Latency | Time to get answer | < 3s |
| Token Usage | Input + output tokens | Minimize |
| Correctness | LLM judge score | > 0.8 |
| Winner Accuracy | Did we pick the right strategy? | > 85% |

## Anti-Overfitting Measures

1. **Holdout Set**: 30% of test cases never used for tuning
2. **Cross-Validation**: Rotate which cases are holdout
3. **Scale Diversity**: Tests at all scales
4. **Query Type Diversity**: All failure modes represented
5. **Adversarial Cases**: Queries designed to trip up each approach

## Semantic Expansion Integration (Completed)

### Integration Architecture

The semantic expansion prototype provides a **query preprocessing step** that can enhance any retrieval approach:

```
Query → [Semantic Expansion] → Expanded Query → [Retrieval] → Results
                │
                └→ Entity Resolution
                └→ Word Sense Disambiguation
                └→ Pronoun Resolution
                └→ Query Expansion
```

### Integration Points

1. **SemanticExpansionPreprocessor** (`benchmark_semantic_expansion_integration.py`)
   - Wraps prototype's `TwoPassSemanticOrchestrator`
   - Provides `expand_query()` method for retrieval pipelines
   - Returns `ExpansionResult` with expanded query, entities, terms

2. **Usage Pattern**
```python
expander = SemanticExpansionPreprocessor(llm)
expansion = await expander.expand_query(query, context)
# Use expansion.expanded_query for retrieval
```

3. **Prototype Components Used**
   - `WordSenseDisambiguator` from `wsd.py`
   - `SemanticExpansionService` from `expansion.py`
   - LLM-based entity resolution and expansion

### Benchmark Results

Initial benchmark (9 test cases) showed 100% baseline recall - test cases too easy.
Need harder test cases where:
- Raw query misses content
- Expansion helps find correct content
- WSD prevents wrong matches (e.g., bank financial vs river)

### Key Files

- `tests/integration/agents/benchmark_semantic_expansion_integration.py` - Integration benchmark
- `prototypes/semantic_expansion/src/integration.py` - Source orchestrator

## Next Steps

1. [x] Build unified benchmark harness
2. [x] Implement strategy selector with LLM
3. [ ] Create scale test suite (small/medium/large/needle)
4. [ ] Add harder test cases where expansion actually matters
5. [ ] Run baseline comparison (no strategy selection)
6. [ ] Add strategy selection, compare to optimal
7. [ ] Identify failure modes, add to Tier 4
8. [ ] Document findings and recommendations
