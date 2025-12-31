# Phase 2: Memory Integration & Semantic Graph Architecture

**Version:** 2.0.0
**Status:** Requirements (Deferred from Phase 1)
**Priority:** P1 - Memory-Aware Decomposition
**Depends On:** Phase 0 (WSD, Entity Classification), Phase 1 (Decomposition Pipeline)

---

## Overview

Phase 2 extends the decomposition pipeline with **memory integration**, **semantic graph storage**, and **contextual pre-expansion**. While Phase 1 extracts implicit knowledge from text in isolation, Phase 2 enables the pipeline to:

1. **Expand ambiguous input before decomposition** using semantic memory context
2. Use existing memory to inform decomposition (memory-aware WSD, anaphora resolution)
3. Link extracted knowledge to existing memory (cross-references)
4. **Store decomposed knowledge as a semantic graph** with entity-centric relationships
5. **Merge new knowledge into existing graph** (augment, not duplicate)
6. Learn and improve from multi-turn conversations (synset reinforcement)

**Deliverable:** A memory-integrated decomposition pipeline with semantic graph storage that enables graph-based retrieval and compression.

### Key Architectural Insight

The fundamental shift from Phase 1 is moving from **document-oriented storage** to **graph-oriented storage**:

```
Phase 1 (Document-oriented):
┌─────────────────────────────────────────────┐
│ DecomposedKnowledge                         │
│   source_text: "Doug has 6 cats"            │
│   entities: [Doug, cats]                    │
│   semantic_roles: [has(Doug, cats)]         │
└─────────────────────────────────────────────┘

Phase 2 (Graph-oriented):
┌───────────────────────────────────────────────────────────────┐
│                     SEMANTIC GRAPH                             │
│                                                                │
│  (Doug:Person) ──[HAS count=6]──→ (cats:Collection)           │
│                                        │                       │
│                                   [MEMBER_OF]                  │
│                                        ↓                       │
│                              (Cat₁) (Cat₂) (Cat₃) ...         │
│                                │                               │
│  Later: "Whiskers is my orange tabby"                         │
│  → Find unattributed Cat node → Add properties                │
│                                                                │
└───────────────────────────────────────────────────────────────┘
```

### Research References

This architecture is informed by:
- [Graphiti](https://github.com/getzep/graphiti) - Real-time knowledge graph memory for AI agents
- [Microsoft GraphRAG](https://microsoft.github.io/graphrag/) - Graph-based RAG for improved context retrieval
- [Neo4j LLM Knowledge Graph Builder](https://medium.com/neo4j/llm-knowledge-graph-builder-first-release-of-2025-532828c4ba76)

---

## Background

### Items Deferred from Phase 1

During Phase 1 implementation review, the following items were identified as requiring memory integration and deferred to Phase 2:

1. **Cross-Reference Linking** - Currently returns empty list (placeholder implementation)
2. **Memory-Aware Branch Weighting** - Memory support/contradiction scoring
3. **Multi-Invocation Synset Learning** - Reinforcement across conversations
4. **Anaphora Resolution** - Requires memory to resolve "he", "it", etc.
5. **Context Carryover** - Maintaining context across document chunks

### Why Phase 2?

Phase 1 focused on single-invocation extraction - processing text without external context. This was intentional to:

- Establish a working baseline before adding complexity
- Allow testing of core extraction without memory dependencies
- Keep the Phase 1 scope manageable

Phase 2 adds the "memory dimension" - connecting extractions to persistent knowledge.

---

## Architecture

### Full Phase 2 Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                 PHASE 2: CONTEXTUAL PRE-EXPANSION                    │
│                                                                      │
│  Input: "I got it!" + speaker_id + conversation_context             │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              MEMORY CONTEXT RETRIEVAL                        │    │
│  │  - Resolve speaker: "I" → Doug                               │    │
│  │  - Query semantic graph for recent topics                    │    │
│  │  - Retrieve candidate referents for "it"                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              LLM CONTEXTUAL EXPANSION                        │    │
│  │  Generate multiple plausible expansions:                     │    │
│  │  - "Doug understood the solution to the coding problem"      │    │
│  │  - "Doug received the Amazon package"                        │    │
│  │  - "Doug caught the ball Whiskers was chasing"               │    │
│  │  Each with confidence weight                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
└──────────────────────────────┼──────────────────────────────────────┘
                               │
                               ▼ (Multiple expanded interpretations)
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 0 + PHASE 1 (Existing)                      │
│  WSD → Entity Classification → Decomposition                        │
│  (Run on each expanded interpretation)                               │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: MEMORY INTEGRATION                       │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              CROSS-REFERENCE LINKING                         │    │
│  │  Link presuppositions, entities to existing graph nodes      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              MEMORY-AWARE BRANCH WEIGHTING                   │    │
│  │  Boost/penalize branches based on graph support              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              GRAPH MERGE OPERATIONS                          │    │
│  │  - Match entities to existing nodes                          │    │
│  │  - Augment nodes with new properties/relations               │    │
│  │  - Temporal versioning (valid_from, valid_to)                │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                       │
│                              ▼                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              SYNSET REINFORCEMENT                            │    │
│  │  Update confidence based on usage outcomes                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC MEMORY GRAPH                             │
│                                                                      │
│  Entities as nodes:  Doug → [has] → Cat₁, Cat₂, Cat₃...            │
│  Facts as edges:     Cat₁ → [name] → "Whiskers"                     │
│  Temporal tracking:  Each edge has (valid_from, valid_to)           │
│  Cardinality:        Doug -[has count=6]→ Cats                      │
│                                                                      │
│  Query: "How many cats?" → Graph traversal → 6                      │
│  Query: "Cat names?" → [Whiskers, Mittens, ...]                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

### REQ-2.1: Cross-Reference Linking

**Description:** Link extracted presuppositions and entities to existing memory items.

**Current State:** `_build_cross_references()` in `pipeline.py` returns empty list (placeholder).

**Required Functionality:**

```python
@dataclass
class CrossReference:
    """Link to existing memory item."""
    ref_id: str                      # UUID of this reference
    source_type: str                 # "presupposition", "entity", "inference"
    source_text: str                 # The text being linked
    target_memory_id: str            # Memory item UUID
    target_content: str              # Summary of target
    link_type: CrossReferenceType    # SUPPORTS, CONTRADICTS, EXTENDS
    confidence: float                # How confident is this link
    evidence: list[str]              # Why we think this links


class CrossReferenceType(str, Enum):
    SUPPORTS = "supports"            # Memory confirms this
    CONTRADICTS = "contradicts"      # Memory conflicts with this
    EXTENDS = "extends"              # This adds to memory
    RESOLVES = "resolves"            # This resolves anaphora
    RECALLS = "recalls"              # "again" references prior


async def build_cross_references(
    decomposed: DecomposedKnowledge,
    memory: MemoryProvider,
    config: CrossReferenceConfig,
) -> list[CrossReference]:
    """Build cross-references to existing memory."""
```

**Linking Scenarios:**

| Trigger | Memory Query | Link Type |
|---------|--------------|-----------|
| "again", "another" | Prior similar events | RECALLS |
| "the {NP}" | Prior mentions of NP | EXTENDS |
| Pronouns | Entity resolution | RESOLVES |
| Contradicting facts | Conflicting memories | CONTRADICTS |
| Supporting facts | Confirming memories | SUPPORTS |

**Acceptance Criteria:**
- [ ] Cross-references populated for presuppositions with iterative triggers
- [ ] Pronoun resolution to prior entities
- [ ] Support/contradiction detection via embedding similarity
- [ ] Confidence scoring based on similarity and recency
- [ ] Unit tests with mock memory provider

### REQ-2.2: Memory-Aware Branch Weighting

**Description:** Use memory to boost or penalize interpretation branches.

**Current State:** `memory_support` field exists but is always 0.0.

**Required Functionality:**

```python
async def compute_memory_support(
    branch: WeightedBranch,
    cross_refs: list[CrossReference],
    config: WeightingConfig,
) -> float:
    """Compute memory support score for a branch.

    Returns:
        Float in range [-penalty, +boost] based on:
        - Number of supporting cross-references
        - Number of contradicting cross-references
        - Confidence of cross-references
    """
```

**Weighting Logic:**

```python
# From WeightingConfig
memory_support_boost: float = 0.2       # Max boost from support
memory_contradiction_penalty: float = 0.3  # Max penalty from contradiction

# Calculation
support_score = sum(
    cr.confidence * config.memory_support_boost
    for cr in cross_refs
    if cr.link_type == CrossReferenceType.SUPPORTS
)
contradiction_score = sum(
    cr.confidence * config.memory_contradiction_penalty
    for cr in cross_refs
    if cr.link_type == CrossReferenceType.CONTRADICTS
)
memory_support = support_score - contradiction_score
```

**Acceptance Criteria:**
- [ ] Branches with memory support rank higher
- [ ] Branches contradicting memory rank lower
- [ ] Final weight properly combines all factors
- [ ] Cross-references included in branch evidence
- [ ] Unit tests for weighting scenarios

### REQ-2.3: Synset Reinforcement Learning

**Description:** Update synset confidence based on usage outcomes across conversations.

**Current State:** `SynsetLearningService` exists but isn't integrated into decomposition pipeline feedback loop.

**Required Functionality:**

```python
async def reinforce_wsd_results(
    decomposition_id: str,
    wsd_results: dict[str, DisambiguationResult],
    outcome: UsageOutcome,
    synset_learning: SynsetLearningService,
) -> dict[str, float]:  # Updated confidences
    """Reinforce WSD results based on usage outcome.

    Args:
        decomposition_id: ID of the decomposition
        wsd_results: The WSD results that were used
        outcome: Whether the interpretation was successful
        synset_learning: Service for persistence

    Returns:
        Dict of word -> new confidence after reinforcement
    """


class UsageOutcome(str, Enum):
    SUCCESS = "success"          # User confirmed interpretation
    FAILURE = "failure"          # User corrected interpretation
    IMPLICIT_SUCCESS = "implicit"  # No correction = assumed success
```

**Integration Points:**

1. **After User Confirmation:**
   - User explicitly confirms interpretation
   - Call `reinforce_wsd_results(outcome=SUCCESS)`

2. **After User Correction:**
   - User says "no, I meant..." or similar
   - Call `reinforce_wsd_results(outcome=FAILURE)`
   - Optionally learn the correct sense

3. **Implicit Success (Decay Strategy):**
   - After N successful uses without correction
   - Boost confidence slightly

**Acceptance Criteria:**
- [ ] Reinforcement integrated into agent feedback loop
- [ ] Success boosts confidence toward max
- [ ] Failure decreases confidence toward research threshold
- [ ] Persistence to evolving synset database
- [ ] Unit tests for reinforcement scenarios

### REQ-2.4: Anaphora Resolution

**Description:** Resolve pronouns and other anaphoric references to their antecedents.

**Current State:** Anaphora detection exists but resolution requires memory.

**Required Functionality:**

```python
@dataclass
class AnaphoraResolution:
    """Resolution of an anaphoric reference."""
    anaphor: str                     # "he", "it", "they"
    span: tuple[int, int]            # Position in text
    antecedent: str | None           # Resolved entity text
    antecedent_id: str | None        # Entity UUID
    source: str                      # "memory", "local", "inferred"
    confidence: float


async def resolve_anaphora(
    text: str,
    entities: list[UniversalSemanticIdentifier],
    memory: MemoryProvider | None,
    config: AnaphoraConfig,
) -> list[AnaphoraResolution]:
    """Resolve anaphoric references in text."""
```

**Resolution Strategy (Priority Order):**

1. **Local Resolution:** Look for antecedents in current text/sentence
2. **Memory Resolution:** Query recent conversation memory
3. **Inference Resolution:** Use LLM to infer from context

**Acceptance Criteria:**
- [ ] Pronoun detection (he, she, it, they, etc.)
- [ ] Demonstrative detection (this, that, these, those)
- [ ] Local antecedent search using syntactic heuristics
- [ ] Memory-based resolution for cross-sentence references
- [ ] Confidence scoring based on distance and agreement
- [ ] Unit tests with various anaphora types

### REQ-2.5: Chunk Context Carryover

**Description:** Maintain context when processing large documents in chunks.

**Current State:** Each chunk is processed independently. Entity references and presuppositions don't carry between chunks.

**Required Functionality:**

```python
@dataclass
class ChunkContext:
    """Context to carry between chunks."""
    entities_seen: dict[str, UniversalSemanticIdentifier]
    wsd_cache: dict[str, DisambiguationResult]
    presuppositions_active: list[Presupposition]
    last_chunk_id: str


async def process_with_context(
    chunks: list[TextChunk],
    context: ChunkContext | None,
    pipeline: IntegratedPipeline,
) -> tuple[IntegratedResult, ChunkContext]:
    """Process chunks with context carryover.

    Returns:
        Merged result and updated context for next batch
    """
```

**Carryover Items:**

| Item | Purpose |
|------|---------|
| Entities seen | Avoid re-identifying same entity |
| WSD cache | Consistent disambiguation across chunks |
| Active presuppositions | Track cross-chunk presuppositions |
| Coreference chains | Track pronoun→entity mappings |

**Acceptance Criteria:**
- [ ] Entity identifiers consistent across chunks
- [ ] WSD results cached and reused
- [ ] Presuppositions carry forward (e.g., "again" in chunk 2)
- [ ] Merge deduplication for cross-chunk items
- [ ] Unit tests with multi-chunk documents

### REQ-2.6: Memory Provider Protocol

**Description:** Define the protocol for memory providers to integrate with Phase 2.

**Required Protocol:**

```python
class MemoryProvider(Protocol):
    """Protocol for memory providers to integrate with decomposition."""

    async def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        limit: int = 10,
    ) -> list[MemoryItem]:
        """Search memory for relevant items."""
        ...

    async def get_by_id(
        self,
        memory_id: str,
    ) -> MemoryItem | None:
        """Get specific memory item by ID."""
        ...

    async def get_recent(
        self,
        limit: int = 10,
        entity_filter: list[str] | None = None,
    ) -> list[MemoryItem]:
        """Get recent memory items, optionally filtered by entities."""
        ...

    async def find_related(
        self,
        entity_id: str,
        relation_types: list[str] | None = None,
    ) -> list[MemoryItem]:
        """Find memory items related to an entity."""
        ...


@dataclass
class MemoryItem:
    """A memory item for cross-referencing."""
    memory_id: str
    content: str
    content_type: str  # "fact", "episode", "skill", etc.
    entities: list[str]  # Entity IDs mentioned
    timestamp: datetime
    confidence: float
    embedding: list[float] | None  # For similarity search
```

**Acceptance Criteria:**
- [ ] Protocol defined and documented
- [ ] Mock implementation for testing
- [ ] Qdrant-backed implementation (optional)
- [ ] Unit tests with mock provider

### REQ-2.7: Pre-Decomposition Context Expansion

**Description:** Expand ambiguous input using semantic memory context BEFORE running Phase 0/1.

**Rationale:** Short, ambiguous sentences like "I got it!" cannot be decomposed meaningfully in isolation. With memory context, we can expand them into multiple plausible full-form interpretations.

**Required Functionality:**

```python
@dataclass
class ExpandedInterpretation:
    """A contextually-expanded interpretation of ambiguous input."""
    original_text: str              # "I got it!"
    expanded_text: str              # "Doug understood the coding solution"
    confidence: float               # How likely is this interpretation
    context_sources: list[str]      # What memory items informed this
    pronoun_resolutions: dict[str, str]  # {"I": "Doug", "it": "the solution"}


class ContextualExpansionPipeline:
    """Expand ambiguous input before decomposition."""

    async def expand(
        self,
        text: str,
        speaker_id: str | None,
        memory_graph: SemanticGraph,
        conversation_context: list[str],
        config: ExpansionConfig,
    ) -> list[ExpandedInterpretation]:
        """Generate multiple plausible expansions.

        Steps:
        1. Detect pronouns and ambiguous references
        2. Resolve speaker ("I" → speaker_id entity)
        3. Query graph for recent topics, candidate referents
        4. Use LLM to generate contextually-plausible expansions
        5. Weight expansions by context relevance

        Returns:
            List of weighted expanded interpretations
        """


@dataclass
class ExpansionConfig:
    """Configuration for contextual expansion."""
    max_expansions: int = 3         # Max interpretations to generate
    min_confidence: float = 0.3     # Minimum expansion confidence
    context_window: int = 5         # Recent messages to consider
    enable_llm_expansion: bool = True
```

**Expansion Scenarios:**

| Input | Context | Expansions |
|-------|---------|------------|
| "I got it!" | Speaker=Doug, recent topic=coding | "Doug understood the solution" |
| "I got it!" | Speaker=Doug, recent topic=package | "Doug received the package" |
| "She loves them" | Speaker=Doug, recent=cats | "Doug's wife loves the cats" |

**Acceptance Criteria:**
- [ ] Pronoun detection and speaker resolution
- [ ] Memory graph query for recent topics
- [ ] LLM-based expansion with context
- [ ] Multiple weighted expansions returned
- [ ] Integration with IntegratedPipeline
- [ ] Unit tests with mock memory

### REQ-2.8: Semantic Graph Model

**Description:** Define the entity-centric graph structure for storing decomposed knowledge.

**Rationale:** Graph storage enables:
- Natural representation of relationships (Doug → [has] → Cat)
- Efficient traversal queries ("How many cats?")
- Incremental knowledge building (add properties to existing entities)
- Context compression (don't repeat entity details)

**Data Model:**

```python
@dataclass
class GraphNode:
    """A node in the semantic graph."""
    node_id: str                    # UUID
    node_type: NodeType             # ENTITY, CONCEPT, EVENT, ATTRIBUTE
    canonical_name: str             # "Doug", "Whiskers"
    entity_type: EntityType         # INSTANCE, CLASS, ANAPHORA
    properties: dict[str, Any]      # {"breed": "tabby", "color": "orange"}
    synset_id: str | None           # WordNet sense if applicable
    created_at: datetime
    updated_at: datetime
    source_ids: list[str]           # Decomposition IDs that contributed


class NodeType(str, Enum):
    ENTITY = "entity"               # Person, place, thing (Doug, Whiskers)
    CONCEPT = "concept"             # Abstract class (Cat, Meeting)
    EVENT = "event"                 # Something that happened
    ATTRIBUTE = "attribute"         # Property value ("orange", "6")


@dataclass
class GraphEdge:
    """A relationship in the semantic graph."""
    edge_id: str                    # UUID
    source_node_id: str             # From node
    target_node_id: str             # To node
    relation_type: str              # "has", "is_a", "name", "breed"
    properties: dict[str, Any]      # {"count": 6, "confidence": 0.95}
    valid_from: datetime            # When this became true
    valid_to: datetime | None       # When this stopped being true (None = current)
    source_decomposition_id: str    # Which decomposition created this
    confidence: float


@dataclass
class SemanticGraph:
    """The complete semantic memory graph."""
    nodes: dict[str, GraphNode]
    edges: dict[str, GraphEdge]

    def find_node(self, name: str, node_type: NodeType | None = None) -> GraphNode | None:
        """Find a node by canonical name."""

    def get_outgoing_edges(self, node_id: str, relation_type: str | None = None) -> list[GraphEdge]:
        """Get edges from a node."""

    def get_incoming_edges(self, node_id: str, relation_type: str | None = None) -> list[GraphEdge]:
        """Get edges to a node."""

    def count_relations(self, node_id: str, relation_type: str, target_type: str | None = None) -> int:
        """Count relations (e.g., 'How many cats does Doug have?')."""

    def traverse(self, start_node_id: str, path: list[str], max_depth: int = 3) -> list[GraphNode]:
        """Traverse graph following relation path."""
```

**Graph Query Examples:**

```python
# "How many cats does Doug have?"
doug_node = graph.find_node("Doug", NodeType.ENTITY)
count = graph.count_relations(doug_node.node_id, "has", target_type="Cat")
# Returns: 6

# "What are my cats' names?"
cat_edges = graph.get_outgoing_edges(doug_node.node_id, "has")
cat_nodes = [graph.nodes[e.target_node_id] for e in cat_edges]
names = [graph.get_property(n.node_id, "name") for n in cat_nodes]
# Returns: ["Whiskers", "Mittens", ...]

# "Tell me about Whiskers"
whiskers = graph.find_node("Whiskers", NodeType.ENTITY)
properties = whiskers.properties  # {"breed": "tabby", "color": "orange"}
relations = graph.get_outgoing_edges(whiskers.node_id)
# Returns full entity context
```

**Acceptance Criteria:**
- [ ] GraphNode and GraphEdge dataclasses defined
- [ ] SemanticGraph with query methods
- [ ] Bi-temporal edges (valid_from, valid_to)
- [ ] Node/edge serialization to JSON
- [ ] Unit tests for graph operations

### REQ-2.9: Graph Merge Operations

**Description:** Merge new decomposed knowledge into existing graph (augment, not duplicate).

**Rationale:** When we process "Whiskers is my orange tabby", we should:
- Find existing Cat node linked to Doug (if any)
- Add/update properties (name=Whiskers, color=orange, breed=tabby)
- NOT create a duplicate cat entry

**Required Functionality:**

```python
class GraphMerger:
    """Merge decomposed knowledge into semantic graph."""

    async def merge(
        self,
        decomposed: DecomposedKnowledge,
        graph: SemanticGraph,
        config: MergeConfig,
    ) -> MergeResult:
        """Merge decomposition into existing graph.

        Steps:
        1. Entity Resolution: Match entities to existing nodes
        2. Conflict Detection: Check for contradicting facts
        3. Property Merge: Add/update node properties
        4. Edge Creation: Create new relationships
        5. Temporal Update: Update validity intervals

        Returns:
            MergeResult with nodes created/updated and conflicts found
        """

    async def resolve_entity(
        self,
        entity: UniversalSemanticIdentifier,
        graph: SemanticGraph,
    ) -> GraphNode | None:
        """Find existing node for an entity.

        Uses tri-modal search (like Graphiti):
        - Semantic embedding similarity
        - Keyword/name matching
        - Graph traversal (related entities)
        """


@dataclass
class MergeResult:
    """Result of merging into graph."""
    nodes_created: list[str]        # New node IDs
    nodes_updated: list[str]        # Updated node IDs
    edges_created: list[str]        # New edge IDs
    edges_invalidated: list[str]    # Edges marked with valid_to
    conflicts: list[MergeConflict]  # Conflicts detected


@dataclass
class MergeConflict:
    """A conflict between new and existing knowledge."""
    conflict_type: ConflictType     # CONTRADICTS, SUPERSEDES, AMBIGUOUS
    existing_edge_id: str
    new_content: str
    resolution: str | None          # How it was resolved
    confidence: float


class ConflictType(str, Enum):
    CONTRADICTS = "contradicts"     # "Doug has 5 cats" vs "Doug has 6 cats"
    SUPERSEDES = "supersedes"       # New info replaces old (temporal)
    AMBIGUOUS = "ambiguous"         # Can't determine relationship


@dataclass
class MergeConfig:
    """Configuration for graph merging."""
    entity_match_threshold: float = 0.8   # Min similarity to match
    auto_resolve_supersedes: bool = True  # Auto-invalidate old edges
    conflict_strategy: str = "flag"       # "flag", "newest_wins", "ask"
```

**Merge Scenarios:**

| New Knowledge | Existing Graph | Action |
|---------------|----------------|--------|
| "Doug has 6 cats" | Doug node exists | Create 6 Cat nodes linked to Doug |
| "Whiskers is my cat" | Doug has 6 cats | Match to unattributed Cat, add name |
| "Whiskers is orange" | Whiskers node exists | Add color property |
| "Doug has 7 cats" | Doug has 6 cats (edge) | CONFLICT: invalidate old, create new |

**Acceptance Criteria:**
- [ ] Entity resolution with tri-modal search
- [ ] Property merging for existing nodes
- [ ] Edge creation with temporal tracking
- [ ] Conflict detection and handling
- [ ] Supersedes logic for updates
- [ ] Unit tests for merge scenarios

### REQ-2.10: Graph-Based Retrieval

**Description:** Retrieve context from semantic graph for LLM prompts.

**Rationale:** Graph traversal provides better context than semantic search alone:
- Structured relationships (not just similar text)
- Cardinality queries ("how many")
- Multi-hop reasoning ("Doug's cat's breed")
- Compression (return node summary, not all source texts)

**Required Functionality:**

```python
class GraphRetriever:
    """Retrieve context from semantic graph."""

    async def retrieve_for_query(
        self,
        query: str,
        graph: SemanticGraph,
        config: RetrievalConfig,
    ) -> GraphContext:
        """Retrieve relevant graph context for a query.

        Steps:
        1. Extract query entities and relations
        2. Find matching nodes in graph
        3. Traverse to related context
        4. Format for LLM consumption
        """

    async def retrieve_for_entity(
        self,
        entity_name: str,
        graph: SemanticGraph,
        depth: int = 2,
    ) -> EntityContext:
        """Get full context for an entity (properties + relations)."""

    async def answer_cardinality(
        self,
        query: str,
        graph: SemanticGraph,
    ) -> int | None:
        """Answer 'how many' questions via graph traversal."""


@dataclass
class GraphContext:
    """Context retrieved from graph."""
    primary_nodes: list[GraphNode]
    related_nodes: list[GraphNode]
    relevant_edges: list[GraphEdge]
    formatted_context: str          # LLM-ready text summary
    confidence: float


@dataclass
class EntityContext:
    """Full context for a single entity."""
    node: GraphNode
    properties: dict[str, Any]
    outgoing_relations: list[tuple[str, GraphNode]]  # (relation, target)
    incoming_relations: list[tuple[GraphNode, str]]  # (source, relation)
    temporal_history: list[GraphEdge]  # Past versions of relationships
```

**Query Examples:**

```python
# "Tell me about my cats"
context = await retriever.retrieve_for_query("my cats", graph)
# Returns: Doug's 6 cats with names, breeds, colors

# "How many cats do I have?"
count = await retriever.answer_cardinality("How many cats do I have?", graph)
# Returns: 6

# "What do you know about Doug?"
entity_ctx = await retriever.retrieve_for_entity("Doug", graph, depth=2)
# Returns: Doug + all properties + relations + related entities
```

**Acceptance Criteria:**
- [ ] Query-to-graph entity extraction
- [ ] Multi-hop traversal with depth limit
- [ ] Cardinality query handling
- [ ] Formatted context for LLM
- [ ] Integration with existing retrieval patterns
- [ ] Unit tests for retrieval scenarios

---

## Evolution Framework Requirements

### REQ-2.E1: Memory Search Tuning

**Description:** Evolvable parameters for memory search.

**Evolvable Parameters:**

```python
@dataclass
class MemorySearchConfig:
    """Evolvable config for memory search."""

    # Retrieval
    similarity_threshold: float = 0.7     # Min similarity to consider
    max_results: int = 10                 # Max items to retrieve
    recency_weight: float = 0.3           # Weight recent items higher

    # Filtering
    min_confidence: float = 0.5           # Only use confident memories
    entity_match_boost: float = 0.2       # Boost for entity overlap
```

### REQ-2.E2: Cross-Reference Metrics

**Description:** Metrics for evaluating cross-reference quality.

**Metrics:**

```python
@dataclass
class CrossReferenceMetrics:
    """Metrics for cross-reference quality."""

    # Coverage
    presups_with_refs: int       # Presuppositions with cross-refs
    entities_resolved: int       # Anaphora resolved
    total_cross_refs: int        # Total cross-references

    # Quality
    avg_link_confidence: float   # Average link confidence
    contradiction_rate: float    # % of refs that are contradictions

    # Impact
    branches_affected: int       # Branches with memory_support != 0
    weight_changes: list[float]  # Distribution of weight changes
```

---

## Test Requirements

### Test Categories

1. **Unit Tests** - Individual components (cross-ref linking, anaphora resolution)
2. **Integration Tests** - Full pipeline with mock memory
3. **Multi-Turn Tests** - Reinforcement across conversations
4. **Performance Tests** - Memory query latency impact

### Test Data Requirements

**Cross-Reference Test Cases (minimum 50):**

| Category | Count | Examples |
|----------|-------|----------|
| Iterative triggers | 15 | "again", "another time", "once more" |
| Anaphora resolution | 15 | "he said", "it broke", "they left" |
| Support/contradiction | 10 | Facts that confirm/conflict |
| Entity linking | 10 | "the meeting" → specific meeting |

### Acceptance Criteria

- [ ] All unit tests passing
- [ ] Integration tests with mock memory passing
- [ ] Cross-reference precision > 80% on test set
- [ ] Anaphora resolution accuracy > 70%
- [ ] Memory query latency < 100ms (cached)
- [ ] Reinforcement correctly updates confidences

---

## Implementation Plan

### Stage 1: Semantic Graph Model (REQ-2.8)
- [ ] Define GraphNode, GraphEdge, NodeType dataclasses
- [ ] Implement SemanticGraph with query methods
- [ ] Bi-temporal edge tracking
- [ ] Serialization to JSON
- [ ] Unit tests for graph operations

### Stage 2: Memory Provider Protocol (REQ-2.6)
- [ ] Define MemoryProvider protocol
- [ ] Create mock implementation for testing
- [ ] Graph-backed implementation
- [ ] Unit tests for protocol

### Stage 3: Graph Merge Operations (REQ-2.9)
- [ ] Entity resolution (tri-modal search)
- [ ] Property merging for existing nodes
- [ ] Edge creation with temporal tracking
- [ ] Conflict detection and handling
- [ ] Unit tests for merge scenarios

### Stage 4: Cross-Reference Linking (REQ-2.1)
- [ ] Implement iterative trigger linking ("again" → prior events)
- [ ] Implement entity mention linking (definite descriptions)
- [ ] Similarity-based support/contradiction detection
- [ ] Unit tests

### Stage 5: Memory-Aware Weighting (REQ-2.2)
- [ ] Integrate cross-references into weighting
- [ ] Compute memory_support scores
- [ ] Add cross-refs to branch evidence
- [ ] Unit tests

### Stage 6: Anaphora Resolution (REQ-2.4)
- [ ] Pronoun detection
- [ ] Local antecedent search
- [ ] Memory-based resolution
- [ ] Unit tests

### Stage 7: Pre-Decomposition Expansion (REQ-2.7)
- [ ] Pronoun detection and speaker resolution
- [ ] Memory graph query for recent topics
- [ ] LLM-based expansion with context
- [ ] Integration with IntegratedPipeline
- [ ] Unit tests with mock memory

### Stage 8: Graph-Based Retrieval (REQ-2.10)
- [ ] Query-to-graph entity extraction
- [ ] Multi-hop traversal
- [ ] Cardinality query handling
- [ ] Formatted context for LLM
- [ ] Unit tests

### Stage 9: Chunk Context Carryover (REQ-2.5)
- [ ] ChunkContext data structure
- [ ] Entity/WSD caching across chunks
- [ ] Presupposition carryover
- [ ] Merge deduplication
- [ ] Unit tests

### Stage 10: Synset Reinforcement Integration (REQ-2.3)
- [ ] Define feedback interfaces
- [ ] Integrate with decomposition pipeline
- [ ] Persistence to evolving synset DB
- [ ] Unit tests

### Stage 11: Integration Testing
- [ ] End-to-end tests with mock memory
- [ ] Multi-turn conversation tests
- [ ] Performance benchmarks
- [ ] Documentation

---

## Success Criteria

Phase 2 is complete when:

1. **Functional:**
   - Semantic graph stores entities and relationships
   - Graph merge augments existing nodes (no duplicates)
   - Pre-decomposition expansion handles ambiguous input ("I got it!")
   - Cross-references populated for relevant triggers
   - Memory affects branch weighting
   - Anaphora resolved with memory
   - Graph-based retrieval answers cardinality queries
   - Reinforcement updates synset confidence

2. **Quality:**
   - Entity resolution precision > 85%
   - Cross-reference precision > 80%
   - Anaphora resolution accuracy > 70%
   - No regression in Phase 1 metrics
   - Graph merge conflict detection > 90%

3. **Performance:**
   - Memory query latency < 100ms (cached)
   - Graph traversal < 50ms for depth=2
   - Total pipeline latency < 6s with memory

4. **Integration:**
   - Works with draagon-ai MemoryProvider
   - SemanticGraph persists across sessions
   - Feedback loop integrated with agent

---

## Dependencies

**From Phase 0/1:**
- IntegratedPipeline
- DecomposedKnowledge, CrossReference, WeightedBranch
- UniversalSemanticIdentifier, EntityType
- SynsetLearningService

**From draagon-ai:**
- MemoryProvider protocol (or compatible interface)
- LLMProvider protocol

**External:**
- Vector similarity search (Qdrant or equivalent)
- Graph database (optional: Neo4j, or in-memory graph structure)

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Memory latency | Slow decomposition | Batch queries, caching |
| False cross-refs | Bad linking | Conservative thresholds, user feedback |
| Reinforcement divergence | Bad WSD over time | Decay, bounds checking |
| Memory cold start | No cross-refs initially | Graceful degradation |
| Scope creep | Delayed delivery | Strict stage boundaries |
| Graph complexity | Hard to query/maintain | Start simple, add complexity incrementally |
| Entity resolution errors | Duplicate nodes | Tri-modal search, LLM verification |
| Expansion hallucination | Wrong context injected | Confidence thresholds, user feedback |

---

## Design Decisions

### DD-2.1: Graph-First Storage

**Decision:** Store decomposed knowledge in a semantic graph structure rather than document-oriented storage.

**Rationale:**
- Natural representation of entity relationships
- Enables graph traversal queries ("how many cats?")
- Supports incremental knowledge building
- Provides context compression (entity summary vs all source texts)

**Trade-offs:**
- More complex than document storage
- Requires entity resolution for merging
- Query patterns differ from vector search

### DD-2.2: Pre-Decomposition Expansion

**Decision:** Expand ambiguous input BEFORE running Phase 0/1, not after.

**Rationale:**
- Short sentences like "I got it!" yield no extractions without context
- Memory context enables pronoun resolution before entity classification
- Multiple expansions can be decomposed in parallel

**Trade-offs:**
- Adds latency before decomposition
- LLM expansion may hallucinate
- Requires memory to be populated first

### DD-2.3: Bi-Temporal Edge Model

**Decision:** Track both `valid_from` and `valid_to` for all relationships.

**Rationale:**
- Enables historical queries ("What did Doug have in 2024?")
- Supports conflict resolution via temporal superseding
- Preserves audit trail of knowledge changes

**References:**
- [Graphiti bi-temporal model](https://github.com/getzep/graphiti)
- [Temporal databases](https://en.wikipedia.org/wiki/Temporal_database)

---

**End of Phase 2 Requirements**
