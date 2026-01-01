# Universal Semantic Memory Architecture

**Status:** Design Document
**Last Updated:** 2025-12-31
**Purpose:** A domain-agnostic semantic graph memory layer for any AI agent application

---

## The Problem

Every AI application rebuilds the same context management from scratch:

| Application | Current Approach | Problems |
|-------------|------------------|----------|
| **Claude Code** | CLAUDE.md files loaded into context | Files grow monolithic, signal lost in noise, manual curation |
| **Party-Lore GM** | Custom XML builders, 50+ entity types, complex joins | Tightly coupled, massive prompts, hard to extend |
| **Personal Assistant** | Qdrant + raw text memories | No relationships, brute-force RAG, no reasoning about connections |
| **Enterprise (Pharma Rep)** | Custom ETL pipelines per domain | Expensive, siloed, no knowledge transfer |

**The Common Pattern:** All need to answer "What context is relevant to THIS query right now?"

---

## The Vision

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Universal Semantic Memory Layer                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                Agent Orchestration Layer                             │    │
│  │        (ReAct loops, swarms, decision engine, MCP, tools)           │    │
│  │                                                                      │    │
│  │   "I need context for this task"                                    │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                   Universal Semantic Memory                          │    │
│  │                                                                      │    │
│  │   retrieve_context(query, constraints) → ContextBundle              │    │
│  │                                                                      │    │
│  │   • Same API regardless of domain                                   │    │
│  │   • Domain concepts are just nodes/edges in the graph               │    │
│  │   • Extensions register new node types, edge types, vocabularies    │    │
│  │   • One universal interface for all applications                    │    │
│  │                                                                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Domain-Agnostic Graph                            │    │
│  │                                                                      │    │
│  │   INSTANCE nodes: Doug, Whiskers, patient_123, src/auth.py          │    │
│  │   CLASS nodes: Person, Cat, Patient, PythonFile, Medication         │    │
│  │   ATTRIBUTE nodes: age:45, color:orange, dosage:10mg                │    │
│  │   EVENT nodes: "prescribed_medication", "file_modified", "learned"  │    │
│  │   COLLECTION nodes: Doug's cats, auth-related files                 │    │
│  │                                                                      │    │
│  │   Relationships: owns, instance_of, depends_on, prescribed_to, ...  │    │
│  │                                                                      │    │
│  └──────────────────────────────┬──────────────────────────────────────┘    │
│                                 │                                            │
│                                 ▼                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                     Extension System                                 │    │
│  │                                                                      │    │
│  │   Extensions EXTEND the graph vocabulary, not the API:              │    │
│  │   • register_node_type("medication", schema={...})                  │    │
│  │   • register_edge_type("prescribed_to", constraints={...})          │    │
│  │   • register_vocabulary("pharma", synonyms=[...])                   │    │
│  │   • register_ingestion_pipeline("code", parser=CodeParser)          │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** There are no "adapters" or application-specific code paths. The semantic memory has ONE universal API. Domain-specific meaning emerges from the DATA in the graph, not from code.

---

## Core Design Principles

### 1. Domain-Agnostic Ontology

The graph stores knowledge in a way that works for ANY domain:

| Concept | Game Master | Claude Code | Personal Assistant | Pharma Rep |
|---------|-------------|-------------|-------------------|------------|
| INSTANCE | "Raven" (NPC) | "src/auth.py" (file) | "Doug" (user) | "Patient 123" |
| CLASS | "Merchant" (role) | "AuthModule" (pattern) | "Person" (type) | "Diabetic" (condition) |
| ATTRIBUTE | disposition: friendly | lines: 450 | birthday: Mar 15 | A1C: 7.2 |
| EVENT | "betrayed party" | "file modified" | "mentioned wife" | "prescribed insulin" |
| RELATIONSHIP | trusts → player | imports → module | married_to → Sarah | takes → medication |

**Key Insight:** The node types (INSTANCE, CLASS, ATTRIBUTE, EVENT, COLLECTION) and edge patterns are universal. The domain-specific meaning emerges from the data, not the schema.

### 2. Bi-Temporal Everything

Every edge has temporal validity:

```python
@dataclass
class GraphEdge:
    source_node_id: str
    target_node_id: str
    relation_type: str

    # When was this TRUE in the world?
    valid_from: datetime
    valid_to: datetime | None  # None = still true

    # When did we LEARN this?
    recorded_at: datetime

    # How confident are we?
    confidence: float  # 0.0 - 1.0
    source_observation_id: str | None
```

This enables:
- "What did we know at time T?" (historical queries)
- "What changed since last session?" (delta queries)
- "What's currently true?" (current state queries)

### 3. Belief-Aware Storage

Not all knowledge is equal:

```python
@dataclass
class BeliefMetadata:
    # Source credibility
    source_type: SourceType  # USER_STATED, INFERRED, OBSERVED, EXTERNAL_API
    source_credibility: float  # 0.0 - 1.0

    # Verification status
    verified: bool
    contradicted_by: list[str]  # Observation IDs that conflict

    # Usage tracking (reinforcement learning)
    times_used: int
    times_helpful: int
    times_wrong: int

    # Importance for retrieval
    importance: float  # Computed from type + usage + recency
```

### 4. Hierarchical Compression

The graph naturally compresses repeated patterns:

```
WITHOUT GRAPH (raw memories):
  "Doug has a cat named Whiskers"
  "Doug's cat is orange"
  "Doug's cat is a tabby"
  "Doug got Whiskers in 2020"
  "Sarah also has a cat"
  "Sarah's cat is named Mittens"
  "Mittens is black"

WITH GRAPH (structured):
  Doug (INSTANCE)
    --owns--> Whiskers (INSTANCE)
                --instance_of--> cat.n.01 (CLASS)
                --has_attr--> color: orange
                --has_attr--> breed: tabby
                --has_attr--> acquired: 2020

  Sarah (INSTANCE)
    --owns--> Mittens (INSTANCE)
                --instance_of--> cat.n.01 (CLASS)  ← SAME CLASS NODE
                --has_attr--> color: black
```

**Token savings:** Instead of 7 separate memories, we have a structured graph that:
1. Shares the `cat.n.01` CLASS node (deduplication)
2. Enables targeted retrieval ("Doug's cat" → just Whiskers subgraph)
3. Supports queries ("all cats" → traverse from cat.n.01)

---

## Universal Node Types

```python
class NodeType(str, Enum):
    # Core ontological types (domain-agnostic)
    INSTANCE = "instance"      # Specific individual: Doug, src/auth.py, Patient_123
    CLASS = "class"            # Abstract type/category: Person, PythonFile, Medication
    ATTRIBUTE = "attribute"    # Property value: age:45, color:orange, dosage:10mg
    EVENT = "event"            # Something that happened: file_modified, prescribed
    COLLECTION = "collection"  # Group of instances: Doug's cats, auth-related files

    # Belief/reasoning types
    OBSERVATION = "observation"  # Raw user statement (immutable)
    BELIEF = "belief"            # Reconciled understanding (can change)

    # Temporal types
    TIME_POINT = "time_point"    # Specific moment: 2025-12-31T10:00:00
    TIME_RANGE = "time_range"    # Duration: "December 2025", "last week"
```

---

## Universal Edge Types

```python
class EdgeRelationType(str, Enum):
    # Ontological relationships
    INSTANCE_OF = "instance_of"      # Whiskers --instance_of--> Cat
    SUBCLASS_OF = "subclass_of"      # Cat --subclass_of--> Mammal
    HAS_ATTRIBUTE = "has_attribute"  # Whiskers --has_attribute--> color:orange
    MEMBER_OF = "member_of"          # Whiskers --member_of--> Doug's cats

    # Semantic relationships (from WordNet + custom)
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    PART_OF = "part_of"              # wheel --part_of--> car
    HAS_PART = "has_part"            # car --has_part--> wheel

    # Possessive/relational
    OWNS = "owns"
    BELONGS_TO = "belongs_to"
    CREATED_BY = "created_by"
    AUTHORED_BY = "authored_by"

    # Spatial
    LOCATED_IN = "located_in"
    CONTAINS = "contains"
    NEAR = "near"

    # Temporal
    HAPPENED_AT = "happened_at"
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"

    # Causal/logical
    CAUSES = "causes"
    CAUSED_BY = "caused_by"
    DEPENDS_ON = "depends_on"
    ENABLES = "enables"
    PREVENTS = "prevents"

    # Social/relational
    KNOWS = "knows"
    WORKS_WITH = "works_with"
    RELATED_TO = "related_to"

    # Belief relationships
    SUPPORTS = "supports"            # Observation supports belief
    CONTRADICTS = "contradicts"      # Observation contradicts belief
    DERIVED_FROM = "derived_from"    # Belief derived from observations
```

**Note:** Domain-specific edge types (prescribes, imports, trusts) are registered via the Extension System, not hardcoded in the core.

---

## The Extension System

Extensions allow domains to register additional vocabulary without changing the core API:

```python
class SemanticMemoryExtension:
    """Base class for domain extensions."""

    name: str  # e.g., "pharma", "code", "rpg", "fitness"

    def register(self, memory: SemanticMemory) -> None:
        """Register this extension's vocabulary with the memory system."""
        pass


class ExtensionRegistry:
    """Manages domain extensions."""

    def register_extension(self, extension: SemanticMemoryExtension) -> None:
        """Register a domain extension."""
        extension.register(self.memory)

    def register_node_type(
        self,
        name: str,
        parent_type: NodeType = NodeType.INSTANCE,
        schema: dict | None = None,
        validation: Callable | None = None,
    ) -> None:
        """Register a custom node type.

        Example:
            register_node_type(
                "medication",
                parent_type=NodeType.CLASS,
                schema={"dosage": "string", "frequency": "string"},
            )
        """
        pass

    def register_edge_type(
        self,
        name: str,
        source_types: list[str] | None = None,
        target_types: list[str] | None = None,
        properties_schema: dict | None = None,
        inverse: str | None = None,
    ) -> None:
        """Register a custom edge type.

        Example:
            register_edge_type(
                "prescribed_to",
                source_types=["medication"],
                target_types=["patient"],
                inverse="takes",
            )
        """
        pass

    def register_vocabulary(
        self,
        domain: str,
        synonyms: dict[str, list[str]] | None = None,
        wordnet_extensions: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Register domain-specific vocabulary.

        Example:
            register_vocabulary(
                "pharma",
                synonyms={"patient": ["client", "subject"]},
                wordnet_extensions=[("metformin", "medication.n.01", "Diabetes drug")],
            )
        """
        pass

    def register_ingestion_pipeline(
        self,
        content_type: str,
        parser: ContentParser,
        decomposer: Decomposer | None = None,
    ) -> None:
        """Register a content ingestion pipeline.

        Example:
            register_ingestion_pipeline(
                "python_file",
                parser=PythonASTParser(),
                decomposer=CodeDecomposer(),
            )
        """
        pass
```

---

## The Universal API

The semantic memory exposes ONE interface that all applications use:

```python
class SemanticMemory:
    """Universal semantic memory - same API for all domains."""

    async def retrieve_context(
        self,
        query: str | list[str],
        constraints: RetrievalConstraints | None = None,
    ) -> ContextBundle:
        """Retrieve relevant context for a query.

        This is THE primary interface. All applications use this same method.

        Args:
            query: Natural language query or list of queries (multi-anchor)
            constraints: Optional retrieval constraints

        Returns:
            ContextBundle with structured context for LLM consumption
        """
        pass

    async def ingest(
        self,
        content: str,
        content_type: str = "text",
        metadata: dict | None = None,
    ) -> list[str]:
        """Ingest content into the graph.

        Returns list of created node IDs.
        """
        pass

    async def update(
        self,
        observation: str,
        source: ObservationSource,
    ) -> UpdateResult:
        """Process a new observation, updating the graph accordingly."""
        pass

    def get_extension_registry(self) -> ExtensionRegistry:
        """Get the extension registry for registering domain vocabulary."""
        pass


@dataclass
class RetrievalConstraints:
    """Constraints for context retrieval."""

    # Temporal constraints
    temporal: TemporalConstraint | None = None  # "current", "last_week", specific range

    # Confidence filtering
    confidence_threshold: float = 0.5

    # Scope filtering (e.g., user_id, project_id)
    scope: dict[str, str] | None = None

    # Size limits
    max_tokens: int = 4000
    max_nodes: int = 100
    max_depth: int = 5

    # What to include
    include_beliefs: bool = True
    include_temporal_context: bool = True
    include_conflicts: bool = True


@dataclass
class ContextBundle:
    """Universal context output - same structure for all domains."""

    # Core retrieved facts (highest relevance)
    core_facts: list[Fact]

    # Supporting context (lower relevance but useful)
    supporting: list[Fact]

    # Belief metadata (confidence, conflicts)
    beliefs: list[BeliefInfo]

    # Temporal context (what changed recently)
    temporal: TemporalContext | None

    # Conflicts that need attention
    conflicts: list[Conflict]

    # Token estimate for context sizing
    estimated_tokens: int

    def to_text(self, format: str = "markdown") -> str:
        """Render context as text for LLM consumption."""
        pass

    def to_structured(self) -> dict:
        """Return structured representation for programmatic access."""
        pass
```

---

## The Retrieval Pipeline

### Phase 1: Query Understanding

```python
async def understand_query(query: str, llm: LLMProvider) -> QueryUnderstanding:
    """Extract entities, intent, and constraints from natural language query."""

    result = await llm.chat([
        {"role": "system", "content": QUERY_UNDERSTANDING_PROMPT},
        {"role": "user", "content": query}
    ])

    return QueryUnderstanding(
        entities=[
            Entity(name="Doug", type_hint="person"),
            Entity(name="cat", type_hint="animal"),
        ],
        intent="retrieve_attribute",  # retrieve, list, count, compare, temporal
        constraints=QueryConstraints(
            temporal=None,
            confidence_threshold=0.5,
        ),
        focus="what color is the cat",
    )
```

### Phase 2: Anchor Finding

```python
async def find_anchors(
    understanding: QueryUnderstanding,
    graph: SemanticGraph,
    vector_index: VectorIndex,
) -> list[AnchorNode]:
    """Find entry points into the graph using vector similarity."""

    anchors = []

    for entity in understanding.entities:
        candidates = await vector_index.search(
            query=entity.name,
            filter={"node_type": entity.type_hint} if entity.type_hint else None,
            limit=5,
        )

        for candidate in candidates:
            node = graph.get_node(candidate.node_id)
            anchors.append(AnchorNode(
                node=node,
                similarity=candidate.score,
                relevance=compute_relevance(node, understanding),
            ))

    return sorted(anchors, key=lambda a: a.relevance, reverse=True)[:10]
```

### Phase 3: Graph Traversal

```python
async def expand_from_anchors(
    anchors: list[AnchorNode],
    graph: SemanticGraph,
    understanding: QueryUnderstanding,
    max_depth: int = 5,
    max_nodes: int = 100,
) -> Subgraph:
    """Expand from anchor nodes following relevant edges."""

    visited = set()
    subgraph = Subgraph()
    queue = PriorityQueue()

    for anchor in anchors:
        queue.put((-anchor.relevance, anchor.node.node_id, 0, []))

    while not queue.empty() and len(subgraph.nodes) < max_nodes:
        priority, node_id, depth, path = queue.get()

        if node_id in visited or depth > max_depth:
            continue
        visited.add(node_id)

        node = graph.get_node(node_id)
        subgraph.add_node(node, path=path)

        for edge in graph.get_outgoing_edges(node_id, current_only=True):
            if edge.target_node_id not in visited:
                relevance = score_edge_relevance(edge, understanding)
                queue.put((
                    -relevance,
                    edge.target_node_id,
                    depth + 1,
                    path + [edge],
                ))
                subgraph.add_edge(edge)

    return subgraph
```

### Phase 4: Belief Integration

```python
async def integrate_beliefs(
    subgraph: Subgraph,
    belief_store: BeliefStore,
    understanding: QueryUnderstanding,
) -> EnrichedSubgraph:
    """Add belief metadata, filter by confidence, resolve conflicts."""

    enriched = EnrichedSubgraph(subgraph)

    for node in subgraph.nodes:
        beliefs = await belief_store.get_beliefs_for_node(node.node_id)

        for belief in beliefs:
            if belief.confidence < understanding.constraints.confidence_threshold:
                continue

            if belief.contradicted_by:
                enriched.mark_conflict(node.node_id, belief)

            enriched.add_belief(node.node_id, belief)

    return enriched
```

### Phase 5: Context Assembly

```python
async def assemble_context(
    subgraph: EnrichedSubgraph,
    understanding: QueryUnderstanding,
    max_tokens: int = 4000,
) -> ContextBundle:
    """Format subgraph for LLM consumption."""

    prioritized = prioritize_for_context(subgraph, understanding)

    context = ContextBundle()

    for node, edges in prioritized.core_facts:
        context.add_fact(format_node_with_edges(node, edges))
        if context.estimated_tokens > max_tokens * 0.6:
            break

    for node, edges in prioritized.supporting:
        context.add_supporting(format_node_with_edges(node, edges))
        if context.estimated_tokens > max_tokens * 0.8:
            break

    for conflict in subgraph.conflicts:
        context.add_caveat(format_conflict(conflict))

    context.add_temporal(format_temporal_context(subgraph, understanding))

    return context
```

---

## Storage Strategy

### Recommendation: Neo4j with Vector Index

| Requirement | Weight | Neo4j | Qdrant + In-Memory | PostgreSQL + AGE |
|-------------|--------|-------|-------------------|------------------|
| Deep traversal (5-10 hops) | Critical | ✓✓ Native Cypher | ✗ O(n^depth) lookups | ✓ But SQL-based |
| Vector similarity | Critical | ✓ Native (5.11+) | ✓✓ Native | ✓ pgvector |
| Bi-temporal queries | High | ✓ With properties | ✓ Manual filtering | ✓ Native |
| Scale (100K+ nodes) | High | ✓✓ Designed for it | ✗ Memory limits | ✓ |
| Hybrid queries | High | ✓✓ Single Cypher | ✗ Cross-system | ✓ Complex SQL |
| Operational simplicity | Medium | 1 system | 2 systems to sync | 1 system |
| Open source | Medium | Community Edition | ✓ | ✓ |

**Decision: Neo4j Community Edition with native vector indexing**

### Schema Design

```cypher
// Node labels (map to NodeType)
(:Instance {
    node_id: String,
    canonical_name: String,
    properties: Map,
    embedding: Vector(1536),
    created_at: DateTime,
    importance: Float
})

(:Class {
    node_id: String,
    synset_id: String,
    canonical_name: String,
    definition: String,
    embedding: Vector(1536)
})

(:Attribute {
    node_id: String,
    key: String,
    value: Any,
    unit: String?
})

(:Event {
    node_id: String,
    description: String,
    occurred_at: DateTime,
    embedding: Vector(1536)
})

(:Collection {
    node_id: String,
    name: String,
    count: Integer
})

// Relationship properties (bi-temporal)
[:EDGE_TYPE {
    edge_id: String,
    valid_from: DateTime,
    valid_to: DateTime?,
    recorded_at: DateTime,
    confidence: Float,
    source_observation_id: String?
}]

// Vector index for similarity search
CREATE VECTOR INDEX node_embeddings FOR (n:Instance) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}
```

---

## Integration with Agent Orchestration

The semantic memory integrates with draagon-ai's agent orchestration layer:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          draagon-ai Integration                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Agent Orchestration                               │    │
│  │                                                                      │    │
│  │   ReAct Loop ─────────────────────────────────────────┐             │    │
│  │   Agent Swarms ───────────────────────────────────────┤             │    │
│  │   Decision Engine ────────────────────────────────────┤             │    │
│  │   MCP Tools ──────────────────────────────────────────┤             │    │
│  │                                                       │             │    │
│  │                                                       ▼             │    │
│  │                              memory.retrieve_context(query)         │    │
│  │                                                       │             │    │
│  └───────────────────────────────────────────────────────┼─────────────┘    │
│                                                          │                   │
│                                                          ▼                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Universal Semantic Memory                         │    │
│  │                                                                      │    │
│  │   Extension Registry ←── Domain extensions register vocabulary      │    │
│  │         │                                                            │    │
│  │         ▼                                                            │    │
│  │   Retrieval Pipeline ──→ ContextBundle                              │    │
│  │         │                                                            │    │
│  │         ▼                                                            │    │
│  │   Neo4j (Graph + Vectors)                                           │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    Ingestion Pipeline                                │    │
│  │                                                                      │    │
│  │   Phase 0/1 Decomposition ────┐                                     │    │
│  │   (Entity extraction, WSD)    │                                     │    │
│  │                               ▼                                     │    │
│  │   GraphBuilder ──────────→ SemanticGraph ──────→ Neo4j              │    │
│  │   (Incremental updates)                                             │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Extension Test Cases

To validate that the architecture is truly universal, we define 5 diverse extension scenarios that MUST work with the same core API:

### Test Case 1: RPG Game Master (Party-Lore)

**Domain:** Narrative RPG with NPCs, locations, plot threads, player relationships

**Extension Registration:**
```python
class RPGExtension(SemanticMemoryExtension):
    name = "rpg"

    def register(self, memory: SemanticMemory) -> None:
        reg = memory.get_extension_registry()

        # Custom node types
        reg.register_node_type("npc", parent_type=NodeType.INSTANCE,
            schema={"disposition": "string", "threat_level": "int"})
        reg.register_node_type("location", parent_type=NodeType.INSTANCE,
            schema={"danger_level": "int", "lighting": "string"})
        reg.register_node_type("plot_thread", parent_type=NodeType.INSTANCE,
            schema={"status": "string", "phase": "string"})

        # Custom edge types
        reg.register_edge_type("trusts", source_types=["npc"], target_types=["player", "npc"],
            properties_schema={"affinity": "int"}, inverse="trusted_by")
        reg.register_edge_type("involved_in", target_types=["plot_thread"])
        reg.register_edge_type("present_at", source_types=["npc", "player"], target_types=["location"])

        # Domain vocabulary
        reg.register_vocabulary("rpg", synonyms={
            "npc": ["character", "person", "figure"],
            "quest": ["mission", "task", "objective"],
        })
```

**Test Query:** "What does Raven know about the missing artifact, and can she be trusted?"

**Expected Retrieval:**
- Raven (NPC) node with disposition, threat_level
- Relationships: Raven --trusts--> players (with affinity scores)
- Plot threads Raven is involved_in
- Recent events involving Raven
- Beliefs about Raven's trustworthiness (with confidence)

---

### Test Case 2: Claude Code / Software Development

**Domain:** Codebase understanding, file relationships, architectural patterns

**Extension Registration:**
```python
class CodeExtension(SemanticMemoryExtension):
    name = "code"

    def register(self, memory: SemanticMemory) -> None:
        reg = memory.get_extension_registry()

        # Custom node types
        reg.register_node_type("file", parent_type=NodeType.INSTANCE,
            schema={"path": "string", "language": "string", "lines": "int"})
        reg.register_node_type("function", parent_type=NodeType.INSTANCE,
            schema={"signature": "string", "complexity": "int"})
        reg.register_node_type("class", parent_type=NodeType.CLASS,
            schema={"methods": "list", "inherits": "list"})
        reg.register_node_type("pattern", parent_type=NodeType.CLASS,
            schema={"description": "string", "examples": "list"})

        # Custom edge types
        reg.register_edge_type("imports", source_types=["file"], target_types=["file", "module"])
        reg.register_edge_type("defines", source_types=["file"], target_types=["function", "class"])
        reg.register_edge_type("calls", source_types=["function"], target_types=["function"])
        reg.register_edge_type("implements", target_types=["pattern"])
        reg.register_edge_type("tested_by", target_types=["file"])

        # Ingestion pipeline for code
        reg.register_ingestion_pipeline("python", parser=PythonASTParser())
        reg.register_ingestion_pipeline("typescript", parser=TypeScriptParser())
```

**Test Query:** "How does the authentication system work and what files would I need to modify to add OAuth?"

**Expected Retrieval:**
- Files related to authentication (auth.py, login.ts, etc.)
- Import/dependency graph showing relationships
- Patterns used (e.g., "middleware pattern", "token-based auth")
- Functions involved in auth flow
- Test files that cover auth functionality
- Recent changes to auth-related files (temporal context)

---

### Test Case 3: Healthcare / Pharma Field Rep

**Domain:** Patient information, medications, interactions, prescribing history

**Extension Registration:**
```python
class PharmaExtension(SemanticMemoryExtension):
    name = "pharma"

    def register(self, memory: SemanticMemory) -> None:
        reg = memory.get_extension_registry()

        # Custom node types
        reg.register_node_type("patient", parent_type=NodeType.INSTANCE,
            schema={"mrn": "string", "dob": "date", "conditions": "list"})
        reg.register_node_type("medication", parent_type=NodeType.CLASS,
            schema={"generic_name": "string", "drug_class": "string"})
        reg.register_node_type("prescription", parent_type=NodeType.EVENT,
            schema={"dosage": "string", "frequency": "string", "prescriber": "string"})
        reg.register_node_type("lab_result", parent_type=NodeType.ATTRIBUTE,
            schema={"test_name": "string", "value": "float", "unit": "string"})

        # Custom edge types
        reg.register_edge_type("prescribed_to", source_types=["medication"], target_types=["patient"],
            inverse="takes")
        reg.register_edge_type("interacts_with", source_types=["medication"], target_types=["medication"],
            properties_schema={"severity": "string", "effect": "string"})
        reg.register_edge_type("contraindicated_for", source_types=["medication"], target_types=["condition"])
        reg.register_edge_type("has_condition", source_types=["patient"], target_types=["condition"])

        # Domain vocabulary
        reg.register_vocabulary("pharma",
            synonyms={"patient": ["client", "subject", "individual"]},
            wordnet_extensions=[("metformin", "medication.n.01", "Diabetes medication")])
```

**Test Query:** "What medications is patient 12345 currently taking, and are there any interactions I should be aware of?"

**Expected Retrieval:**
- Patient 12345 with current conditions
- All active prescriptions (temporal: valid_to IS NULL)
- Medications with dosages
- Drug-drug interactions between current medications
- Recent lab results relevant to medications
- Contraindications based on patient conditions
- Beliefs with confidence (e.g., "patient reported taking OTC ibuprofen" - confidence: 0.7)

---

### Test Case 4: Personal Fitness Coach

**Domain:** Workout history, goals, nutrition, progress tracking

**Extension Registration:**
```python
class FitnessExtension(SemanticMemoryExtension):
    name = "fitness"

    def register(self, memory: SemanticMemory) -> None:
        reg = memory.get_extension_registry()

        # Custom node types
        reg.register_node_type("workout", parent_type=NodeType.EVENT,
            schema={"type": "string", "duration_minutes": "int", "intensity": "string"})
        reg.register_node_type("exercise", parent_type=NodeType.CLASS,
            schema={"muscle_groups": "list", "equipment": "list"})
        reg.register_node_type("goal", parent_type=NodeType.INSTANCE,
            schema={"target": "string", "deadline": "date", "status": "string"})
        reg.register_node_type("measurement", parent_type=NodeType.ATTRIBUTE,
            schema={"type": "string", "value": "float", "unit": "string"})
        reg.register_node_type("meal", parent_type=NodeType.EVENT,
            schema={"calories": "int", "protein": "int", "carbs": "int", "fat": "int"})

        # Custom edge types
        reg.register_edge_type("includes_exercise", source_types=["workout"], target_types=["exercise"])
        reg.register_edge_type("targets", source_types=["workout", "exercise"], target_types=["muscle_group"])
        reg.register_edge_type("progresses_toward", source_types=["workout", "measurement"], target_types=["goal"])
        reg.register_edge_type("prefers", source_types=["user"], target_types=["exercise", "workout_type"])
        reg.register_edge_type("avoids", source_types=["user"], target_types=["exercise"],
            properties_schema={"reason": "string"})

        # Domain vocabulary
        reg.register_vocabulary("fitness", synonyms={
            "workout": ["training", "session", "exercise"],
            "reps": ["repetitions"],
        })
```

**Test Query:** "How has my bench press progressed this month, and what should I do next workout?"

**Expected Retrieval:**
- User's bench press performance (measurements) over time (temporal query)
- Recent workouts that included bench press
- Goals related to bench press or chest strength
- User's preferences (prefers/avoids edges)
- Muscle groups that need attention based on recent workout distribution
- Recommendations based on progression patterns

---

### Test Case 5: Educational Tutor

**Domain:** Student knowledge, curriculum, learning progress, misconceptions

**Extension Registration:**
```python
class EducationExtension(SemanticMemoryExtension):
    name = "education"

    def register(self, memory: SemanticMemory) -> None:
        reg = memory.get_extension_registry()

        # Custom node types
        reg.register_node_type("concept", parent_type=NodeType.CLASS,
            schema={"subject": "string", "difficulty": "int", "prerequisites": "list"})
        reg.register_node_type("lesson", parent_type=NodeType.EVENT,
            schema={"topic": "string", "duration": "int", "format": "string"})
        reg.register_node_type("assessment", parent_type=NodeType.EVENT,
            schema={"type": "string", "score": "float", "max_score": "float"})
        reg.register_node_type("misconception", parent_type=NodeType.BELIEF,
            schema={"correct_understanding": "string", "common_cause": "string"})
        reg.register_node_type("learning_style", parent_type=NodeType.ATTRIBUTE,
            schema={"type": "string", "strength": "float"})

        # Custom edge types
        reg.register_edge_type("understands", source_types=["student"], target_types=["concept"],
            properties_schema={"mastery_level": "float", "last_assessed": "datetime"})
        reg.register_edge_type("struggles_with", source_types=["student"], target_types=["concept"],
            properties_schema={"attempts": "int"})
        reg.register_edge_type("prerequisite_for", source_types=["concept"], target_types=["concept"])
        reg.register_edge_type("has_misconception", source_types=["student"], target_types=["misconception"])
        reg.register_edge_type("covered_in", source_types=["concept"], target_types=["lesson"])
        reg.register_edge_type("tested_by", source_types=["concept"], target_types=["assessment"])

        # Domain vocabulary
        reg.register_vocabulary("education", synonyms={
            "student": ["learner", "pupil"],
            "understand": ["grasp", "comprehend", "master"],
        })
```

**Test Query:** "What concepts is this student ready to learn next, and what misconceptions should I address first?"

**Expected Retrieval:**
- Student's current concept mastery levels (understands edges with mastery_level)
- Concepts the student struggles with
- Active misconceptions
- Prerequisite graph to determine what's unlocked
- Recent assessments showing progress or regression
- Learning style preferences
- Temporal context: what was covered in recent lessons

---

## Validation Criteria

For each test case, the universal API must:

1. **Same Method Call:** `memory.retrieve_context(query)` works identically
2. **Extension-Only Customization:** Domain specifics come only from registered extensions
3. **Proper Traversal:** Multi-hop relationships are followed correctly (e.g., patient → prescription → medication → interactions)
4. **Temporal Awareness:** Current vs historical queries work correctly
5. **Belief Integration:** Confidence scores and conflicts are surfaced
6. **Token Efficiency:** Retrieved context fits within limits without losing critical information
7. **No Domain Leakage:** Code changes to the core memory system are not needed for any domain

---

## Next Steps

1. **Install Neo4j** - Set up Neo4j Community Edition with Docker
2. **Persistence Layer** - Implement Neo4j storage for SemanticGraph
3. **Embedding Pipeline** - Add vector embeddings to nodes during graph building
4. **Extension System** - Implement ExtensionRegistry
5. **Retrieval Pipeline** - Implement the 5-phase retrieval algorithm
6. **Test Case Implementation** - Build tests for all 5 extension scenarios
7. **MCP Integration** - Create MCP server exposing the universal API

---

## Appendix: Why This Works for All Domains

The universal design works because:

1. **Same cognitive patterns** - All domains involve entities, relationships, events, and beliefs
2. **Same retrieval needs** - "What's relevant to this query?" is domain-agnostic
3. **Same temporal concerns** - All domains care about current vs historical state
4. **Same confidence concerns** - All domains need to track belief reliability
5. **Same compression benefits** - All domains benefit from structured knowledge vs raw text

The domain-specific meaning emerges from:
- The vocabulary registered via extensions
- The node/edge types registered via extensions
- The data stored in the graph

But the core graph structure, retrieval algorithm, and belief system are universal.
