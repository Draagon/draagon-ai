# Hybrid Parallel Retrieval: Full Specification

**Date:** 2025-01-01
**Status:** Implementation Spec
**Related:** FR-010, Multi-Agent Orchestration, Retrieval Benchmark

---

## 1. Concept Overview

### The Core Idea

Run **multiple retrieval strategies in parallel** using independent agents, then **merge and synthesize** their results. Each strategy has different strengths:

| Strategy | Strengths | Weaknesses |
|----------|-----------|------------|
| **Local Context** | Fast, always current, team-curated | Limited to one project |
| **Semantic Graph** | Relationships, multi-hop, cross-project entities | Needs upfront indexing, can be stale |
| **Vector RAG** | Similarity, fuzzy matching, examples | Loses structure, noisy at scale |

By running them in parallel, we get:
- **Speed**: No sequential bottleneck
- **Coverage**: Each catches what others miss
- **Confidence**: Agreement = higher confidence

### The Insight: Graph Scopes Vector

The semantic graph doesn't just provide results—it **scopes** the vector search:

```
Without Graph Scoping:
  "customer table patterns" → searches ALL 500 projects → 80% noise

With Graph Scoping:
  Graph: "Which projects have Customer entities?" → [billing, auth, crm]
  Vector: Search only billing/auth/crm → 95% relevant
```

---

## 2. Architecture

### 2.1 Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HybridRetrievalOrchestrator                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  Query Analyzer  │  ← Classifies query, determines which agents to run   │
│  └────────┬─────────┘                                                       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    Parallel Agent Execution                         │    │
│  │                    (via MultiAgentOrchestrator)                     │    │
│  │                                                                     │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                │    │
│  │  │ LocalAgent  │  │ GraphAgent  │  │ VectorAgent │                │    │
│  │  │             │  │             │  │             │                │    │
│  │  │ Searches:   │  │ Queries:    │  │ Searches:   │                │    │
│  │  │ - CLAUDE.md │  │ - Neo4j     │  │ - Qdrant    │                │    │
│  │  │ - Local idx │  │ - Entities  │  │ - Embeddings│                │    │
│  │  │             │  │ - Relations │  │             │                │    │
│  │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                │    │
│  │         │                │                │                        │    │
│  │         └────────────────┼────────────────┘                        │    │
│  │                          ▼                                          │    │
│  │              ┌───────────────────────┐                             │    │
│  │              │  SharedWorkingMemory  │                             │    │
│  │              │  (Observations pool)  │                             │    │
│  │              └───────────────────────┘                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────┐                                                       │
│  │  Result Merger   │  ← Deduplicates, ranks, handles conflicts            │
│  └────────┬─────────┘                                                       │
│           │                                                                  │
│           ▼                                                                  │
│  ┌──────────────────┐                                                       │
│  │ Synthesis Agent  │  ← Generates final answer from merged context        │
│  └──────────────────┘                                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
1. QUERY ARRIVES
   "What patterns do other teams use for customer authentication?"

2. QUERY ANALYSIS
   - Detected entities: ["customer", "authentication"]
   - Query type: cross-project pattern search
   - Recommended paths: [LOCAL, GRAPH, VECTOR]

3. PARALLEL RETRIEVAL (via MultiAgentOrchestrator)

   LocalAgent:                    GraphAgent:                   VectorAgent:
   ├─ Search CLAUDE.md           ├─ Find Customer entities    ├─ Embed query
   ├─ Search local patterns      ├─ Find Auth entities        ├─ Search top-K
   └─ Return: 2 observations     ├─ Find projects with both   └─ Return: 5 chunks
                                 ├─ Get relationship paths
                                 └─ Return: 4 observations

   All observations → SharedWorkingMemory

4. RESULT MERGING
   - Deduplicate similar observations
   - Rank by confidence + source agreement
   - Resolve conflicts (if any)
   - Graph observations can SCOPE remaining vector search

5. SYNTHESIS
   - Build prompt with ranked observations
   - Generate coherent answer
   - Cite sources
```

### 2.3 Graph-Scoped Vector Search

A key optimization: Graph results inform Vector search scope.

```python
# Phase 1: Graph identifies relevant scope
graph_results = await graph_agent.find_relevant_entities(query)
# Returns: ["billing-service", "auth-service", "customer-portal"]

# Phase 2: Vector searches ONLY within that scope
vector_filter = {"project": {"$in": graph_results.projects}}
vector_results = await vector_agent.search(query, filter=vector_filter)
# Now searching 3 projects instead of 500

# Phase 3: Merge and synthesize
```

This can run as:
- **Parallel + Re-search**: Graph and Vector run parallel, then Vector re-searches with scope
- **Sequential**: Graph first, scoped Vector second (slower but more precise)
- **Full Parallel**: Both independent, merge handles scope implicitly

---

## 3. Query Classification

### 3.1 Query Types and Routing

```python
class QueryType(Enum):
    ENTITY_LOOKUP = "entity_lookup"        # "What is CustomerService?"
    RELATIONSHIP = "relationship"           # "How does X connect to Y?"
    SIMILARITY = "similarity"               # "Find code similar to this"
    PATTERN = "pattern"                     # "How do teams handle X?"
    CROSS_PROJECT = "cross_project"         # "What other projects use X?"
    LOCAL_ONLY = "local_only"               # "What does our CLAUDE.md say?"
    MULTI_HOP = "multi_hop"                 # "What calls the service that uses Customer?"
```

### 3.2 Routing Matrix

| Query Type | Local | Graph | Vector | Notes |
|------------|-------|-------|--------|-------|
| ENTITY_LOOKUP | ⚡ | ✅ | ❌ | Graph is authoritative |
| RELATIONSHIP | ❌ | ✅ | ❌ | Graph excels |
| SIMILARITY | ❌ | ❌ | ✅ | Vector's strength |
| PATTERN | ⚡ | ✅ | ✅ | Both needed |
| CROSS_PROJECT | ❌ | ✅ | ✅ | Graph scopes Vector |
| LOCAL_ONLY | ✅ | ❌ | ❌ | Fast path |
| MULTI_HOP | ❌ | ✅ | ⚡ | Graph primary, Vector for context |

Legend: ✅ Primary, ⚡ Secondary/Optional, ❌ Skip

### 3.3 Classification Heuristics

```python
def classify_query(query: str, context: dict) -> QueryClassification:
    """Classify query to determine retrieval paths."""

    classification = QueryClassification(query=query)

    # Entity detection (capitalized words, known entity types)
    entities = extract_entity_mentions(query)
    if entities:
        classification.detected_entities = entities
        classification.add_path(RetrievalPath.GRAPH)

    # Relationship indicators
    relationship_patterns = [
        r"how does .+ connect to",
        r"what calls",
        r"what uses",
        r"depends on",
        r"relationship between",
    ]
    if any(re.search(p, query.lower()) for p in relationship_patterns):
        classification.query_type = QueryType.RELATIONSHIP
        classification.add_path(RetrievalPath.GRAPH)

    # Similarity indicators
    similarity_patterns = [
        r"similar to",
        r"like this",
        r"examples of",
        r"find .+ that",
    ]
    if any(re.search(p, query.lower()) for p in similarity_patterns):
        classification.query_type = QueryType.SIMILARITY
        classification.add_path(RetrievalPath.VECTOR)

    # Cross-project indicators
    cross_project_patterns = [
        r"other (teams?|projects?|services?)",
        r"across .+ (org|company|enterprise)",
        r"how do .+ handle",
        r"patterns? for",
    ]
    if any(re.search(p, query.lower()) for p in cross_project_patterns):
        classification.query_type = QueryType.CROSS_PROJECT
        classification.add_path(RetrievalPath.GRAPH)
        classification.add_path(RetrievalPath.VECTOR)
        classification.graph_scopes_vector = True

    # Always include local as fast fallback
    classification.add_path(RetrievalPath.LOCAL)

    return classification
```

---

## 4. Agent Specifications

### 4.1 LocalAgent

**Purpose:** Fast retrieval from current project context.

**Sources:**
- CLAUDE.md and related files
- Local index (.draagon/ cache)
- In-memory recent context

**Output:**
```python
@dataclass
class LocalObservation:
    content: str
    source_file: str
    section: str | None
    relevance_score: float
    observation_type: Literal["instruction", "pattern", "fact", "context"]
```

### 4.2 GraphAgent

**Purpose:** Structured entity and relationship retrieval.

**Capabilities:**
- Entity lookup by name
- Relationship traversal (1-N hops)
- Cross-project entity resolution
- Pattern detection

**Output:**
```python
@dataclass
class GraphObservation:
    content: str
    entities: list[Entity]
    relationships: list[Relationship]
    source_projects: list[str]
    hop_depth: int
    confidence: float
    observation_type: Literal["entity", "relationship", "pattern", "cross_ref"]
```

### 4.3 VectorAgent

**Purpose:** Semantic similarity search across documents.

**Capabilities:**
- Embedding-based search
- Optional scope filtering (from Graph)
- Chunk retrieval with context

**Output:**
```python
@dataclass
class VectorObservation:
    content: str
    source_file: str
    source_project: str
    similarity_score: float
    chunk_context: str | None  # surrounding text
    observation_type: Literal["example", "documentation", "code", "discussion"]
```

---

## 5. Result Merging

### 5.1 Observation Normalization

All observations are normalized to a common format:

```python
@dataclass
class NormalizedObservation:
    content: str
    source: str                      # "local:CLAUDE.md", "graph:neo4j", "vector:qdrant"
    source_project: str | None
    confidence: float                # 0.0 - 1.0
    observation_type: str
    entities_mentioned: list[str]
    metadata: dict[str, Any]

    # For deduplication
    content_hash: str
    semantic_embedding: list[float] | None
```

### 5.2 Deduplication Strategy

```python
async def deduplicate_observations(
    observations: list[NormalizedObservation],
    similarity_threshold: float = 0.85,
) -> list[NormalizedObservation]:
    """Deduplicate similar observations, keeping highest confidence."""

    unique = []
    for obs in sorted(observations, key=lambda o: -o.confidence):
        is_duplicate = False
        for existing in unique:
            # Check semantic similarity
            similarity = cosine_similarity(obs.semantic_embedding, existing.semantic_embedding)
            if similarity >= similarity_threshold:
                # Merge metadata, boost confidence if multiple sources agree
                existing.confidence = min(1.0, existing.confidence + 0.1)
                existing.metadata["corroborated_by"] = existing.metadata.get("corroborated_by", [])
                existing.metadata["corroborated_by"].append(obs.source)
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(obs)

    return unique
```

### 5.3 Ranking Algorithm

```python
def rank_observations(observations: list[NormalizedObservation]) -> list[NormalizedObservation]:
    """Rank observations by relevance and confidence."""

    def score(obs: NormalizedObservation) -> float:
        base_score = obs.confidence

        # Boost for source agreement
        corroboration_count = len(obs.metadata.get("corroborated_by", []))
        agreement_boost = 0.1 * corroboration_count

        # Boost for graph-sourced entity matches
        if obs.source.startswith("graph:") and obs.entities_mentioned:
            entity_boost = 0.15
        else:
            entity_boost = 0

        # Boost for local context (team-curated)
        if obs.source.startswith("local:"):
            local_boost = 0.1
        else:
            local_boost = 0

        # Penalty for very long chunks (likely less focused)
        length_penalty = max(0, (len(obs.content) - 500) / 5000) * 0.1

        return base_score + agreement_boost + entity_boost + local_boost - length_penalty

    return sorted(observations, key=score, reverse=True)
```

### 5.4 Conflict Resolution

When observations contradict:

```python
@dataclass
class Conflict:
    observations: list[NormalizedObservation]
    conflict_type: Literal["factual", "temporal", "scope"]
    resolution_strategy: Literal["latest_wins", "highest_confidence", "ask_user", "include_both"]

async def resolve_conflicts(observations: list[NormalizedObservation]) -> list[NormalizedObservation]:
    """Detect and resolve conflicting observations."""

    conflicts = detect_conflicts(observations)

    for conflict in conflicts:
        if conflict.conflict_type == "temporal":
            # Prefer more recent information
            winner = max(conflict.observations, key=lambda o: o.metadata.get("last_updated", 0))
            for loser in conflict.observations:
                if loser != winner:
                    loser.confidence *= 0.5
                    loser.metadata["superseded_by"] = winner.source

        elif conflict.conflict_type == "factual":
            # Prefer graph over vector, vector over local for facts
            source_priority = {"graph": 3, "vector": 2, "local": 1}
            winner = max(conflict.observations,
                        key=lambda o: source_priority.get(o.source.split(":")[0], 0))
            # Don't discard others, but mark the conflict
            for obs in conflict.observations:
                obs.metadata["conflict_detected"] = True
                obs.metadata["preferred_source"] = winner.source

    return observations
```

---

## 6. Synthesis

### 6.1 Context Building

```python
def build_synthesis_context(
    query: str,
    ranked_observations: list[NormalizedObservation],
    max_context_tokens: int = 8000,
) -> str:
    """Build context for synthesis, respecting token limits."""

    context_parts = []
    current_tokens = 0

    # Group by source type for clarity
    by_source = group_by(ranked_observations, lambda o: o.source.split(":")[0])

    for source_type in ["local", "graph", "vector"]:
        observations = by_source.get(source_type, [])
        if not observations:
            continue

        source_header = {
            "local": "FROM YOUR PROJECT:",
            "graph": "FROM ENTERPRISE KNOWLEDGE GRAPH:",
            "vector": "FROM SIMILAR DOCUMENTS:",
        }[source_type]

        context_parts.append(f"\n{source_header}\n")

        for obs in observations:
            obs_text = format_observation(obs)
            obs_tokens = estimate_tokens(obs_text)

            if current_tokens + obs_tokens > max_context_tokens:
                break

            context_parts.append(obs_text)
            current_tokens += obs_tokens

    return "\n".join(context_parts)
```

### 6.2 Synthesis Prompt

```python
SYNTHESIS_PROMPT = """You are synthesizing information from multiple retrieval sources to answer a query.

QUERY: {query}

RETRIEVED CONTEXT:
{context}

INSTRUCTIONS:
1. Synthesize a coherent answer from the retrieved information
2. Cite sources when making specific claims: [local:file.md], [graph:EntityName], [vector:project/file]
3. Note if sources agree or conflict
4. If information is incomplete, say what's missing
5. Prioritize local project context, then graph relationships, then similar examples

Answer:"""
```

---

## 7. Implementation Components

### 7.1 Core Classes

```python
# Main orchestrator
class HybridRetrievalOrchestrator:
    async def retrieve(self, query: str, context: dict) -> HybridResult
    async def retrieve_with_scope(self, query: str, scope: Scope) -> HybridResult

# Query analysis
class QueryAnalyzer:
    def classify(self, query: str) -> QueryClassification
    def extract_entities(self, query: str) -> list[str]
    def determine_paths(self, classification: QueryClassification) -> list[RetrievalPath]

# Individual agents
class LocalRetrievalAgent:
    async def retrieve(self, query: str, context: TaskContext) -> list[LocalObservation]

class GraphRetrievalAgent:
    async def retrieve(self, query: str, context: TaskContext) -> list[GraphObservation]
    async def find_scope(self, entities: list[str]) -> Scope

class VectorRetrievalAgent:
    async def retrieve(self, query: str, context: TaskContext, scope: Scope | None) -> list[VectorObservation]

# Result processing
class ResultMerger:
    async def merge(self, observations: list[Observation]) -> list[NormalizedObservation]
    async def deduplicate(self, observations: list[NormalizedObservation]) -> list[NormalizedObservation]
    async def rank(self, observations: list[NormalizedObservation]) -> list[NormalizedObservation]

class SynthesisAgent:
    async def synthesize(self, query: str, observations: list[NormalizedObservation]) -> str
```

### 7.2 Integration with Existing Infrastructure

```python
# Uses existing MultiAgentOrchestrator
orchestrator = MultiAgentOrchestrator(
    mode=OrchestrationMode.PARALLEL,
    agents=[
        AgentSpec(id="local", role=AgentRole.RESEARCHER, ...),
        AgentSpec(id="graph", role=AgentRole.RESEARCHER, ...),
        AgentSpec(id="vector", role=AgentRole.RESEARCHER, ...),
    ],
)

# Uses existing SharedWorkingMemory
shared_memory = SharedWorkingMemory(task_id="query_123")

# Agents add observations to shared memory
await shared_memory.add_observation(
    content="Customer entity found in billing-service",
    source_agent_id="graph",
    attention_weight=0.9,
    is_belief_candidate=True,
)
```

---

## 8. Configuration

```python
@dataclass
class HybridRetrievalConfig:
    # Agent enablement
    enable_local: bool = True
    enable_graph: bool = True
    enable_vector: bool = True

    # Parallelism
    parallel_execution: bool = True
    graph_scopes_vector: bool = True  # Use graph results to scope vector search

    # Limits
    max_observations_per_agent: int = 10
    max_total_observations: int = 25
    max_context_tokens: int = 8000

    # Deduplication
    similarity_threshold: float = 0.85

    # Timeouts
    agent_timeout_ms: int = 5000
    total_timeout_ms: int = 10000

    # Fallback behavior
    fallback_on_timeout: bool = True
    fallback_order: list[str] = field(default_factory=lambda: ["local", "graph", "vector"])
```

---

## 9. Metrics and Observability

```python
@dataclass
class HybridRetrievalMetrics:
    # Timing
    total_time_ms: float
    local_time_ms: float
    graph_time_ms: float
    vector_time_ms: float
    merge_time_ms: float
    synthesis_time_ms: float

    # Observations
    local_observations: int
    graph_observations: int
    vector_observations: int
    after_dedup: int

    # Quality
    source_agreement: float  # % of observations corroborated
    conflicts_detected: int
    conflicts_resolved: int

    # Coverage
    query_types_matched: list[QueryType]
    entities_found: list[str]
    projects_searched: list[str]
```

---

## 10. Example Scenarios

### Scenario A: Database Schema Patterns

**Query:** "What patterns do other teams use for customer tables?"

**Flow:**
1. Classify: CROSS_PROJECT + PATTERN
2. Paths: [LOCAL, GRAPH, VECTOR], graph_scopes_vector=True
3. Local: Finds nothing specific
4. Graph: Finds Customer entities in [billing, auth, crm], returns schemas
5. Vector (scoped to billing/auth/crm): Finds design docs, migration comments
6. Merge: 12 observations → 8 after dedup, ranked
7. Synthesize: "Three patterns observed: billing uses UUID keys, auth uses email as natural key, crm uses composite. All include audit columns. See [graph:billing/Customer], [vector:auth/docs/schema.md]"

### Scenario B: API Integration

**Query:** "How does OrderService connect to PaymentGateway?"

**Flow:**
1. Classify: RELATIONSHIP
2. Paths: [GRAPH] (vector skipped - structural query)
3. Graph: Traverses OrderService → CALLS → PaymentGateway, finds intermediaries
4. Merge: 4 graph observations
5. Synthesize: "OrderService calls PaymentGateway via the PaymentAdapter. Flow: OrderService → PaymentAdapter.processPayment() → PaymentGateway.charge(). Retry logic in adapter. [graph:OrderService→PaymentAdapter→PaymentGateway]"

### Scenario C: Code Examples

**Query:** "Find code similar to this error handling pattern"

**Flow:**
1. Classify: SIMILARITY
2. Paths: [VECTOR] (graph skipped - similarity query)
3. Vector: Embeds pattern, searches, returns top 10 similar chunks
4. Merge: Dedup similar matches
5. Synthesize: "Found 6 similar patterns. Most common in error-handling/, also in api/middleware/. Example from billing-service/handlers/payment.py uses same try/catch structure. [vector:billing-service/handlers/payment.py:45-60]"
