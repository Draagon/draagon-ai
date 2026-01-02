# Hybrid Retrieval Architecture: Enterprise Code Intelligence

**Date:** 2025-01-01
**Status:** Design Proposal
**Related:** FR-010, Retrieval Benchmark, Multi-Agent Orchestration

---

## Problem Statement

When working on large coding projects within an enterprise:

1. **Local context is insufficient** - A single project's CLAUDE.md doesn't know about patterns used across the organization
2. **Vector RAG loses structure** - "Find similar code" misses logical relationships like "this table relates to that service"
3. **Full semantic graphs are expensive** - Building graphs for everything is slow and storage-intensive
4. **Real-time assistance is needed** - When creating a new database table, show what other teams have done for that entity

### The Vision

> When a developer creates a new `customers` table, the system should surface:
> - Other projects that have `customer` entities
> - Common field patterns (email, phone, address structures)
> - Related services that consume customer data
> - Data governance rules about customer PII
> - Historical decisions about customer data modeling

---

## Proposed Architecture

### Layer 1: Local Project Context (Fast, Always Available)

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Project Files                       │
├─────────────────────────────────────────────────────────────┤
│  CLAUDE.md          │ Project instructions, patterns         │
│  .draagon/          │ Local semantic cache                   │
│  src/**/*.py        │ Source code (indexed locally)          │
│  docs/**/*.md       │ Documentation                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Local Index    │
                    │  (SQLite/JSON)  │
                    └─────────────────┘
```

**Characteristics:**
- Maintained by team (CLAUDE.md)
- Fast local access
- No network dependency
- Refreshed on git pull/commit hooks

---

### Layer 2: Enterprise Semantic Graph (Deep, Relationship-Aware)

```
┌─────────────────────────────────────────────────────────────┐
│                 Enterprise Knowledge Graph                   │
│                        (Neo4j)                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [Project A] ──uses──> [Customer Entity]                   │
│        │                      │                              │
│        │                      ├──has_field──> [email]       │
│        │                      ├──has_field──> [phone]       │
│        │                      └──stored_in──> [customers]   │
│        │                                          │          │
│   [Project B] ──uses──────────────────────────────┘          │
│        │                                                     │
│        └──calls──> [CustomerService API]                    │
│                           │                                  │
│   [Project C] ──calls─────┘                                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What gets extracted into the graph:**
- **Entities:** Classes, tables, services, APIs, config keys
- **Relationships:** uses, calls, stored_in, depends_on, owned_by
- **Metadata:** Team ownership, data classification, last updated
- **Patterns:** Common field structures, naming conventions

**NOT in the graph (too granular):**
- Individual lines of code
- Variable names within functions
- Comments and docstrings (→ Vector RAG)

---

### Layer 3: Enterprise Vector Index (Broad, Similarity-Based)

```
┌─────────────────────────────────────────────────────────────┐
│                 Enterprise Vector Store                      │
│                      (Qdrant)                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Chunks from all projects:                                   │
│  - README sections                                           │
│  - Code docstrings                                           │
│  - API documentation                                         │
│  - Commit messages (summarized)                              │
│  - Slack/docs discussions (if integrated)                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**What gets embedded:**
- Documentation chunks
- Function/class docstrings
- README sections
- High-level summaries

**NOT embedded (use graph instead):**
- Entity relationships
- Structural connections
- Type hierarchies

---

## Hybrid Query Flow

### Parallel Agent Architecture

```
                           User Query
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Query Classifier  │
                    │   (Fast heuristics) │
                    └─────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
     ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
     │ Local Agent │  │ Graph Agent │  │ Vector Agent│
     │             │  │             │  │             │
     │ Search:     │  │ Query:      │  │ Search:     │
     │ - CLAUDE.md │  │ - Neo4j     │  │ - Qdrant    │
     │ - Local src │  │ - Entities  │  │ - Docs      │
     │ - .draagon/ │  │ - Relations │  │ - Examples  │
     └─────────────┘  └─────────────┘  └─────────────┘
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    ┌─────────────────────┐
                    │   Result Merger     │
                    │   (Dedupe, Rank)    │
                    └─────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Synthesis Agent   │
                    │   (Generate Answer) │
                    └─────────────────────┘
```

### Query Classification Heuristics

```python
def classify_query(query: str) -> set[RetrievalPath]:
    """Determine which retrieval paths to use."""
    paths = set()

    # Always check local context first (fast)
    paths.add(RetrievalPath.LOCAL)

    # Entity/relationship queries → Graph
    if contains_entity_reference(query):  # "Customer", "OrderService"
        paths.add(RetrievalPath.GRAPH)

    if is_relationship_query(query):  # "how does X connect to Y"
        paths.add(RetrievalPath.GRAPH)

    # Similarity/example queries → Vector
    if is_similarity_query(query):  # "examples of", "similar to"
        paths.add(RetrievalPath.VECTOR)

    if is_pattern_query(query):  # "how do other teams handle"
        paths.add(RetrievalPath.GRAPH)
        paths.add(RetrievalPath.VECTOR)

    # Cross-project queries → Both
    if mentions_other_projects(query):
        paths.add(RetrievalPath.GRAPH)
        paths.add(RetrievalPath.VECTOR)

    return paths
```

---

## Use Case: Real-Time Database Schema Assistance

### Scenario

Developer types in their IDE:
```sql
CREATE TABLE customers (
    id UUID PRIMARY KEY,
    email VARCHAR(255),
    |  -- cursor here
```

### System Response

**Graph Query:** "Find all entities named 'customer' or 'Customer' across enterprise"

**Results from Graph:**
```
Found 7 projects with Customer entities:

1. billing-service (Team: Payments)
   - customers table: id, email, stripe_customer_id, created_at
   - Customer class: includes address, phone fields

2. auth-service (Team: Identity)
   - users table (represents customers): id, email, password_hash
   - Note: "customer" in auth is called "user"

3. analytics-platform (Team: Data)
   - customer_dim table: customer_id, first_seen, segment
   - Joins with: orders_fact, sessions_fact
```

**Vector Query:** "customer table schema patterns best practices"

**Results from Vector:**
```
From internal docs:
- "Customer PII must include data_classification column"
- "Use UUID for customer IDs (not auto-increment)"
- "Include audit columns: created_at, updated_at, created_by"

From code examples:
- billing-service/models/customer.py: "Address is separate table, FK relationship"
```

**Synthesized Suggestion:**
```sql
CREATE TABLE customers (
    id UUID PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    -- Common fields from other projects:
    phone VARCHAR(50),
    stripe_customer_id VARCHAR(255),  -- if using Stripe
    -- Required by data governance:
    data_classification VARCHAR(50) DEFAULT 'PII',
    -- Audit columns (standard):
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Consider: Address as separate table (see billing-service pattern)
-- See also: analytics-platform for reporting dimensions
```

---

## When to Skip Each Retrieval Path

### Skip Graph When:
- Query is purely similarity-based ("find code that looks like this")
- No entity names detected
- Time-sensitive query (graph may be stale)
- Simple documentation lookup

### Skip Vector When:
- Query is structural ("what calls CustomerService?")
- Exact entity lookup ("what is the Customer table schema?")
- Relationship traversal ("how does Order connect to Payment?")
- Query mentions specific project/team names

### Skip Both (Local Only) When:
- Query is about current project's patterns
- CLAUDE.md has explicit instructions
- Fast response needed, enterprise context not critical

---

## Implementation Phases

### Phase 1: Local + Graph (Current Sprint)
- ✅ DocumentIngestionOrchestrator (extracts to graph)
- ✅ RetrievalBenchmark (compare approaches)
- 🔄 SemanticWebProcessor (query graph)
- ⬜ Cross-project entity resolution

### Phase 2: Add Vector RAG
- ⬜ Qdrant integration
- ⬜ Embedding pipeline for docs/code
- ⬜ RAGProcessor enhancement
- ⬜ Parallel agent orchestration

### Phase 3: Hybrid Intelligence
- ⬜ Query classifier
- ⬜ Result merger with deduplication
- ⬜ Confidence-weighted synthesis
- ⬜ Real-time IDE integration

### Phase 4: Enterprise Scale
- ⬜ Incremental graph updates (git hooks)
- ⬜ Multi-tenant graph partitioning
- ⬜ Access control integration
- ⬜ Staleness detection and refresh

---

## Key Insight: Semantic Graph as "Index" for Vector RAG

The semantic graph can **guide** vector search:

```
1. Query: "How do other teams handle customer authentication?"

2. Graph Query:
   - Find entities: Customer, Authentication, Auth
   - Find projects using both
   - Get: auth-service, identity-platform, customer-portal

3. Vector Query (scoped):
   - Search ONLY in: auth-service/**, identity-platform/**, customer-portal/**
   - Query: "customer authentication flow"

4. Result: Targeted, relevant chunks instead of enterprise-wide noise
```

This is the **killer feature**: Graph provides the "where to look", Vector provides the "what to find".

---

## Comparison: Your Original Plan vs Semantic-Enhanced

### Original Plan (Vector-First)
```
Extract metadata → Store in Vector DB → Similarity search → Results
```
**Limitation:** Misses structural relationships, may return "similar-sounding" but irrelevant results

### Semantic-Enhanced Plan
```
Extract metadata → Build Semantic Graph → Use graph to scope Vector search → Richer results
```
**Advantage:** Graph tells you WHICH projects are relevant, Vector finds the details

### Example Difference

Query: "What database schema patterns exist for order management?"

**Vector-Only:**
- Might return:
  - A blog post about "ordering pizza" (false positive)
  - Documentation about "sort order" (false positive)
  - Actual order tables (true positive, but mixed in noise)

**Graph-Scoped Vector:**
1. Graph finds: Projects with `Order` entity → billing-service, fulfillment-api, analytics
2. Vector searches only those projects
3. Returns: Only relevant order schema patterns, no noise

---

## Technical Considerations

### Graph Schema for Code Intelligence

```cypher
// Core entities
(:Project {name, repo_url, team, last_indexed})
(:Entity {name, type, fqn})  // class, table, service, api
(:Field {name, type, nullable})
(:Pattern {name, description})  // reusable patterns

// Relationships
(:Project)-[:CONTAINS]->(:Entity)
(:Entity)-[:HAS_FIELD]->(:Field)
(:Entity)-[:USES]->(:Entity)
(:Entity)-[:CALLS]->(:Entity)  // API calls
(:Entity)-[:STORED_IN]->(:Entity)  // table relationships
(:Entity)-[:IMPLEMENTS]->(:Pattern)
(:Project)-[:OWNED_BY]->(:Team)
(:Entity)-[:CLASSIFIED_AS]->(:DataClass)  // PII, Public, etc.
```

### Vector Embedding Strategy

```python
# What to embed (chunked)
EMBED_SOURCES = [
    "README.md",           # Project overview
    "**/docs/**/*.md",     # Documentation
    "**/*_test.py",        # Test descriptions (docstrings)
    "**/api/**/*.yaml",    # OpenAPI specs
]

# What NOT to embed (use graph)
SKIP_EMBEDDING = [
    "**/*.py",             # Code structure → Graph
    "**/migrations/**",    # Schema → Graph
    "**/config/**",        # Config → Graph
]
```

---

## Success Metrics

1. **Precision at K:** % of top-K results that are relevant
2. **Recall:** % of relevant results found
3. **Latency:** P50/P95 response time
4. **Cross-project discovery:** % of queries that surface useful cross-project info
5. **Developer satisfaction:** Survey/feedback

---

## Next Steps

1. **Prototype parallel agent retrieval** using existing MultiAgentOrchestrator
2. **Build query classifier** with simple heuristics first
3. **Test with real enterprise codebase** (multiple related projects)
4. **Measure baseline** before hybrid, after hybrid
5. **Iterate on graph schema** based on query patterns
