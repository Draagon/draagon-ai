# Semantic Query Expansion with Knowledge Graph Context

**Research Date:** 2025-01-02
**Status:** Research Complete - Ready for Test Implementation

---

## 1. Problem Statement

Given an ambiguous query like:

> "How do other teams handle authentication?"

The term "other teams" is ambiguous. To resolve it properly, we need to:
1. Look up semantic knowledge about "teams" in the knowledge graph
2. Discover the user's team membership
3. Identify all teams in the company
4. Expand "other teams" to specific team names
5. Generate prioritized query variations for parallel processing

---

## 2. Research Findings

### 2.1 Key Techniques from Literature

| Technique | Source | Description |
|-----------|--------|-------------|
| **HyDE** | [Precise Zero-Shot Dense Retrieval](https://arxiv.org/abs/2212.10496) | Generate hypothetical documents, then embed and search |
| **RAG-Fusion** | [RAG-Fusion Paper](https://arxiv.org/abs/2402.03367) | Generate multiple query reformulations, retrieve in parallel, merge with Reciprocal Rank Fusion |
| **ThinkQE** | [Query Expansion Survey](https://arxiv.org/html/2509.07794v1) | Iterate: retrieve → expand → filter loop |
| **PBR** | [Personalize Before Retrieve](https://arxiv.org/abs/2510.08935) | Use user history + graph-based PageRank for personalized expansion |
| **Omni-RAG** | [LLM-Assisted Query Understanding](https://arxiv.org/html/2506.21384v1) | Rewrite → Decompose → Retrieve per intent → Aggregate → Rerank |
| **Entity Resolution** | [Entity Resolved KGs](https://neo4j.com/blog/developer/entity-resolved-knowledge-graphs/) | Two-pass disambiguation with vector embeddings |

### 2.2 Key Insights

1. **Multi-hypothesis is better than single**: Generate multiple interpretations, retrieve in parallel, merge results
2. **Confidence scoring is essential**: Each expansion should have a probability/confidence weight
3. **Graph context improves disambiguation**: Using structured knowledge about entities reduces ambiguity
4. **Reciprocal Rank Fusion works**: Merging ranked results from multiple queries consistently outperforms single-query
5. **Personalization matters**: User context (team membership, history) dramatically improves relevance

### 2.3 Processing Order Question

Two viable approaches:

| Approach | Flow | Pros | Cons |
|----------|------|------|------|
| **A: Expand First** | Query → LLM Expand → Multiple Variations → Each through Phase 0/1 | Preserves original ambiguity for graph resolution | Expensive (N times semantic processing) |
| **B: Decompose First** | Query → Phase 0/1 → Graph Lookup → LLM Expand on Graph Results | Uses semantic decomposition once | May lose nuance in original phrasing |
| **C: Hybrid** | Query → Phase 0/1 on key terms → Graph Lookup → LLM Expand → Embed variations (skip full Phase 0/1) | Best of both: semantic for entities, LLM for expansion | More complex |

**Recommendation: Approach C (Hybrid)**
- Extract key ambiguous terms (e.g., "other teams") through Phase 0
- Look up semantic context from graph
- LLM expands to concrete variations using graph context
- Variations are embedded directly (no full Phase 0/1 - already resolved)

---

## 3. Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Query Expansion Pipeline                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐                                               │
│  │ Original Query   │ "How do other teams handle authentication?"   │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Phase 0: Extract │ Entities: [teams], Relationships: [handle]    │
│  │ Key Terms        │ Ambiguous: [other teams] → needs resolution   │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Graph Lookup     │ "teams" → HAS_SENSE → team.n.01              │
│  │                  │ team.n.01 → INSTANCE_OF → [Engineering,       │
│  │                  │                           Platform, Data,      │
│  │                  │                           Mobile, QA]          │
│  │                  │ user:Doug → MEMBER_OF → Engineering           │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ LLM Expansion (with graph context)                           │   │
│  │                                                               │   │
│  │ Context:                                                      │   │
│  │ - User is on Engineering team                                 │   │
│  │ - Company teams: Platform, Data, Mobile, QA (excluding user)│   │
│  │ - "other teams" likely means "teams other than mine"         │   │
│  │                                                               │   │
│  │ Generate prioritized expansions:                              │   │
│  │                                                               │   │
│  │ 1. "How does Platform team handle authentication?" (0.85)     │   │
│  │ 2. "How does Data team handle authentication?" (0.82)         │   │
│  │ 3. "How does Mobile team handle authentication?" (0.80)       │   │
│  │ 4. "How does QA team handle authentication?" (0.78)           │   │
│  │ 5. "What authentication patterns exist across teams?" (0.70)  │   │
│  │                                                               │   │
│  └────────┬─────────────────────────────────────────────────────┘   │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Filter by        │ Threshold: 0.75                               │
│  │ Confidence       │ Keep: [1, 2, 3, 4] → 4 parallel queries       │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Parallel         │ Query 1 → [results₁, confidence₁]             │
│  │ Retrieval        │ Query 2 → [results₂, confidence₂]             │
│  │                  │ Query 3 → [results₃, confidence₃]             │
│  │                  │ Query 4 → [results₄, confidence₄]             │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Merge with       │ RRF score = Σ(1 / (k + rank_i)) × conf_i     │
│  │ Weighted RRF     │                                               │
│  └────────┬─────────┘                                               │
│           │                                                          │
│           ▼                                                          │
│  ┌──────────────────┐                                               │
│  │ Synthesize       │ Answer with provenance from multiple teams    │
│  └──────────────────┘                                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Test Plan

### 4.1 Test Data Schema

Pre-load the following knowledge graph structure:

```
COMPANY: Acme Corp
├── TEAM: Engineering (user: Doug is member)
│   ├── uses: JWT authentication
│   ├── pattern: OAuth2 with PKCE
│   └── doc: "Engineering uses JWT tokens with 1hr expiry"
│
├── TEAM: Platform
│   ├── uses: API keys + JWT
│   ├── pattern: Service mesh mTLS
│   └── doc: "Platform team uses Istio service mesh with mTLS"
│
├── TEAM: Data
│   ├── uses: IAM roles
│   ├── pattern: AWS IAM + service accounts
│   └── doc: "Data team uses AWS IAM roles for service auth"
│
├── TEAM: Mobile
│   ├── uses: Biometric + OAuth
│   ├── pattern: Native biometric with refresh tokens
│   └── doc: "Mobile uses native biometric auth with 30-day refresh"
│
└── TEAM: QA
    ├── uses: Test tokens
    ├── pattern: Mock auth for testing
    └── doc: "QA uses mock authentication with bypass tokens"
```

### 4.2 Test Cases

#### Test 1: Basic "Other Teams" Expansion

**Input:**
```
Query: "How do other teams handle authentication?"
User Context: { user: "Doug", team: "Engineering" }
```

**Expected Expansions:**
| Rank | Expanded Query | Confidence | Reasoning |
|------|---------------|------------|-----------|
| 1 | "How does Platform team handle authentication?" | 0.85 | Platform most relevant for infra |
| 2 | "How does Data team handle authentication?" | 0.82 | Data has distinct patterns |
| 3 | "How does Mobile team handle authentication?" | 0.80 | Mobile has unique mobile auth |
| 4 | "How does QA team handle authentication?" | 0.78 | QA may have test-specific patterns |

**Expected Final Answer:**
- Should mention Platform's mTLS
- Should mention Data's IAM roles
- Should mention Mobile's biometric
- Should NOT mention Engineering (user's own team)

---

#### Test 2: No User Context - Multiple Interpretations

**Input:**
```
Query: "How do other teams handle authentication?"
User Context: { } (no team membership known)
```

**Expected Expansions:**
| Rank | Expanded Query | Confidence | Reasoning |
|------|---------------|------------|-----------|
| 1 | "What authentication patterns exist across all teams?" | 0.75 | Generic cross-team |
| 2 | "How does each team handle authentication?" | 0.72 | Enumerate all |
| 3 | "Compare authentication approaches by team" | 0.68 | Comparative |

**Note:** Without user context, system can't determine "other" relative to what.

---

#### Test 3: Entity Resolution - "The API"

**Input:**
```
Query: "How does the API handle rate limiting?"
Graph Context:
  - "API" → [Payment API, User API, Internal API, Gateway API]
  - user:Doug → last_worked_on → Payment API
```

**Expected Expansions:**
| Rank | Expanded Query | Confidence | Reasoning |
|------|---------------|------------|-----------|
| 1 | "How does Payment API handle rate limiting?" | 0.90 | User's recent context |
| 2 | "How does Gateway API handle rate limiting?" | 0.75 | Gateway most relevant for rate limiting |
| 3 | "How does User API handle rate limiting?" | 0.60 | Lower relevance |

---

#### Test 4: Semantic Hierarchy - "Databases"

**Input:**
```
Query: "What databases do we use?"
Graph Context:
  - "database" → HAS_SENSE → database.n.01
  - database.n.01 → INSTANCE_OF → [PostgreSQL, MongoDB, Redis, Elasticsearch]
  - PostgreSQL → used_by → [Engineering, Data]
  - MongoDB → used_by → [Mobile]
  - Redis → used_for → caching
  - Elasticsearch → used_for → search
```

**Expected Expansions:**
| Rank | Expanded Query | Confidence | Reasoning |
|------|---------------|------------|-----------|
| 1 | "What relational databases (PostgreSQL) do we use?" | 0.85 | Most common |
| 2 | "What document databases (MongoDB) do we use?" | 0.80 | Different category |
| 3 | "What cache databases (Redis) do we use?" | 0.75 | Common infrastructure |
| 4 | "What search databases (Elasticsearch) do we use?" | 0.70 | Specialized use |

---

### 4.3 Processing Pipeline Variations to Test

| Variation | Description | Expected Behavior |
|-----------|-------------|-------------------|
| **V1: LLM-First** | Query → LLM Expand → Phase 0/1 each | Expensive but thorough |
| **V2: Decompose-First** | Query → Phase 0/1 → Graph → LLM Expand | Efficient, single decomposition |
| **V3: Hybrid** | Query → Phase 0 (entities only) → Graph → LLM Expand | Best balance |
| **V4: Parallel Decompose** | Query → LLM Expand → Parallel Phase 0/1 | Maximum parallelism |

---

### 4.4 Metrics to Measure

| Metric | Description | Target |
|--------|-------------|--------|
| **Expansion Accuracy** | Do expansions correctly interpret ambiguity? | > 80% |
| **Recall** | Are all relevant team patterns found? | > 90% |
| **Precision** | Are irrelevant results excluded? | > 85% |
| **Latency** | Total time for expansion + retrieval | < 3s |
| **Confidence Calibration** | Does high confidence correlate with accuracy? | r² > 0.7 |

---

### 4.5 Edge Cases

| Case | Input | Expected Behavior |
|------|-------|-------------------|
| **No graph matches** | "How do alien teams handle X?" | Fallback to literal query |
| **Single interpretation** | "How does Engineering handle X?" | No expansion needed |
| **Conflicting context** | User in Engineering, asks about Engineering | Clarify or expand to sub-teams |
| **Empty graph** | New deployment, no team data | Warn user, search literally |
| **Too many expansions** | "What do teams use?" → 50 variations | Cap at N, prioritize by confidence |

---

## 5. Implementation Order

1. **Phase 1: Query Expansion Core**
   - Implement `QueryExpander` class
   - Add LLM prompt for expansion with graph context
   - Test with mocked graph context

2. **Phase 2: Graph Integration**
   - Connect to real Neo4j graph
   - Implement entity resolution lookup
   - Test with seeded team data

3. **Phase 3: Parallel Retrieval**
   - Implement weighted parallel retrieval
   - Add Reciprocal Rank Fusion for merging
   - Benchmark latency

4. **Phase 4: Confidence Calibration**
   - Track expansion accuracy over time
   - Tune confidence thresholds
   - Add user feedback loop

---

## 6. Sources

- [Query Expansion Survey (2025)](https://arxiv.org/html/2509.07794v1) - Comprehensive overview of PLM/LLM-based QE
- [Personalize Before Retrieve](https://arxiv.org/abs/2510.08935) - User-centric personalized expansion
- [RAG-Fusion](https://arxiv.org/abs/2402.03367) - Multi-query + RRF approach
- [Omni-RAG Query Understanding](https://arxiv.org/html/2506.21384v1) - Decomposition before retrieval
- [HyDE](https://arxiv.org/abs/2212.10496) - Hypothetical document embeddings
- [Entity Resolved Knowledge Graphs](https://neo4j.com/blog/developer/entity-resolved-knowledge-graphs/) - Neo4j ERKG tutorial
- [Haystack Query Expansion](https://haystack.deepset.ai/blog/query-expansion) - Practical implementation
