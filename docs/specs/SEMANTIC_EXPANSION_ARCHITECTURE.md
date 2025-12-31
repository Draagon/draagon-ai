# Semantic Expansion Architecture: Deep Design

**Version:** 0.1.0-DESIGN
**Status:** Technical Design (Pre-Implementation)
**Last Updated:** 2025-12-30
**Builds On:** SEMANTIC_EXPANSION_CONCEPT.md

---

## Table of Contents

1. [Expansion Inputs and Variation Generation](#1-expansion-inputs-and-variation-generation)
2. [Graph Structure in Qdrant](#2-graph-structure-in-qdrant)
3. [Storing Memories as Semantic Graphs](#3-storing-memories-as-semantic-graphs)
4. [Word Sense Disambiguation System](#4-word-sense-disambiguation-system)
5. [Cross-Layer Memory Associations](#5-cross-layer-memory-associations)
6. [Evolutionary Design Framework](#6-evolutionary-design-framework)
7. [Integration Architecture](#7-integration-architecture)
8. [Research References](#8-research-references)

---

## 1. Expansion Inputs and Variation Generation

### 1.1 Complete Input Taxonomy

When generating semantic expansion variations, these inputs contribute to different interpretations:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEMANTIC EXPANSION INPUT SOURCES                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    A. CONTEXTUAL INPUTS                              │    │
│  │                                                                      │    │
│  │  1. Immediate Context                                                │    │
│  │     - Previous 1-3 utterances in conversation                       │    │
│  │     - Recently mentioned entities (antecedent resolution)           │    │
│  │     - Active topic/domain                                           │    │
│  │                                                                      │    │
│  │  2. Session Context                                                  │    │
│  │     - All entities mentioned this session                           │    │
│  │     - Established co-reference chains ("he" = Doug)                 │    │
│  │     - Session goals/intents if stated                               │    │
│  │                                                                      │    │
│  │  3. Temporal Context                                                 │    │
│  │     - Current time/date                                             │    │
│  │     - Time expressions in statement ("morning", "tomorrow")         │    │
│  │     - Recency of related events                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    B. COGNITIVE MEMORY INPUTS                        │    │
│  │                                                                      │    │
│  │  4. Working Memory (5 min TTL)                                       │    │
│  │     - Active observations from current task                         │    │
│  │     - Attention-weighted recent items                               │    │
│  │     - Unresolved conflicts flagged for clarification                │    │
│  │                                                                      │    │
│  │  5. Episodic Memory (2 week TTL)                                     │    │
│  │     - Recent conversation summaries                                  │    │
│  │     - Interaction patterns with entities                            │    │
│  │     - Context from similar past situations                          │    │
│  │                                                                      │    │
│  │  6. Semantic Memory (6 month TTL)                                    │    │
│  │     - Established facts about entities                              │    │
│  │     - Known preferences and habits                                  │    │
│  │     - Learned skills and procedures                                 │    │
│  │                                                                      │    │
│  │  7. Metacognitive Memory (Permanent)                                 │    │
│  │     - Self-knowledge about interpretation accuracy                  │    │
│  │     - Historical error patterns ("I often misinterpret X")          │    │
│  │     - Calibration data for confidence estimates                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    C. BELIEF SYSTEM INPUTS                           │    │
│  │                                                                      │    │
│  │  8. Established Beliefs                                              │    │
│  │     - Verified facts (high confidence)                              │    │
│  │     - Unverified claims (medium confidence)                         │    │
│  │     - Conflicting beliefs awaiting resolution                       │    │
│  │                                                                      │    │
│  │  9. Entity Relationships                                             │    │
│  │     - Known relationships (Doug → spouse → Sarah)                   │    │
│  │     - Group memberships (Doug ∈ household)                          │    │
│  │     - Authority/credibility levels                                   │    │
│  │                                                                      │    │
│  │  10. Opinions & Preferences                                          │    │
│  │      - Agent's formed opinions about topics                         │    │
│  │      - User's known preferences                                     │    │
│  │      - Historical preference patterns                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    D. LINGUISTIC INPUTS                              │    │
│  │                                                                      │    │
│  │  11. Lexical Analysis                                                │    │
│  │      - Word senses (via WordNet/BabelNet synsets)                   │    │
│  │      - Lemmatized root forms                                        │    │
│  │      - Part-of-speech tags                                          │    │
│  │                                                                      │    │
│  │  12. Syntactic Structure                                             │    │
│  │      - Dependency parse tree                                        │    │
│  │      - Semantic role labels (agent, patient, instrument)            │    │
│  │      - Clause boundaries and scope                                  │    │
│  │                                                                      │    │
│  │  13. Pragmatic Signals                                               │    │
│  │      - Speech act type (assertion, request, question)               │    │
│  │      - Hedging language ("maybe", "I think")                        │    │
│  │      - Emphasis and focus markers                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    E. COMMONSENSE KNOWLEDGE                          │    │
│  │                                                                      │    │
│  │  14. ATOMIC/COMET Inferences                                         │    │
│  │      - xIntent: Why did the subject do this?                        │    │
│  │      - xNeed: What did subject need to do this?                     │    │
│  │      - xWant: What will subject want next?                          │    │
│  │      - xEffect: What happens to subject as result?                  │    │
│  │      - xReact: How does subject feel?                               │    │
│  │      - oReact: How do others feel?                                  │    │
│  │                                                                      │    │
│  │  15. ConceptNet Relations                                            │    │
│  │      - IsA, PartOf, UsedFor, CapableOf                              │    │
│  │      - HasProperty, LocatedAt, Causes                               │    │
│  │      - Antonym, Synonym, RelatedTo                                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Variation Generation Strategy

```python
@dataclass
class ExpansionInput:
    """All inputs for generating semantic expansion variations."""

    # A. Contextual
    immediate_context: list[str]  # Recent utterances
    session_entities: dict[str, EntityInfo]  # Mentioned entities
    active_topic: str | None
    current_time: datetime

    # B. Memory Layers
    working_observations: list[SharedObservation]
    episodic_summaries: list[EpisodeSummary]
    semantic_facts: list[SemanticFact]
    metacognitive_calibration: CalibrationData

    # C. Beliefs
    relevant_beliefs: list[AgentBelief]
    entity_relationships: list[Relationship]
    known_preferences: list[Preference]

    # D. Linguistic
    word_senses: dict[str, list[WordSense]]  # word -> possible senses
    lemmas: dict[str, str]  # word -> lemma
    pos_tags: dict[str, str]  # word -> POS
    semantic_roles: list[SemanticRole]

    # E. Commonsense
    atomic_inferences: dict[str, list[str]]  # relation_type -> inferences
    conceptnet_relations: list[ConceptNetTriple]


class VariationGenerator:
    """Generate interpretation variations from expansion inputs."""

    async def generate_variations(
        self,
        statement: str,
        inputs: ExpansionInput,
        max_variations: int = 5,
    ) -> list[ExpansionVariant]:
        """
        Generate interpretation variations.

        Strategy:
        1. Identify all ambiguity points (entity references, word senses,
           scope, temporal reference)
        2. For each ambiguity, get resolution options from inputs
        3. Generate cross-product of plausible combinations
        4. Weight each combination using cognitive scoring
        5. Prune to top N variations
        """

        # Step 1: Identify ambiguity points
        ambiguities = await self._identify_ambiguities(statement, inputs)

        # Step 2: Get resolution options for each
        resolution_options = {}
        for amb in ambiguities:
            options = await self._get_resolution_options(amb, inputs)
            resolution_options[amb.id] = options

        # Step 3: Generate combinations (pruned cross-product)
        combinations = self._generate_pruned_combinations(
            resolution_options,
            max_combinations=max_variations * 3  # Generate extra, then filter
        )

        # Step 4: Score each combination
        scored_variants = []
        for combo in combinations:
            variant = await self._build_variant(statement, combo, inputs)
            variant.score = await self._compute_cognitive_score(variant, inputs)
            scored_variants.append(variant)

        # Step 5: Return top N
        scored_variants.sort(key=lambda v: v.score, reverse=True)
        return scored_variants[:max_variations]

    async def _compute_cognitive_score(
        self,
        variant: ExpansionVariant,
        inputs: ExpansionInput,
    ) -> float:
        """
        Compute cognitive plausibility score for a variant.

        Weights:
        - Recency bias: Recent context matches score higher
        - Memory support: Variants supported by memory layers score higher
        - Belief consistency: Variants consistent with beliefs score higher
        - Commonsense alignment: Variants matching ATOMIC inferences score higher
        - Metacognitive calibration: Adjust based on historical accuracy
        """

        weights = {
            "recency": 0.20,
            "working_memory": 0.15,
            "episodic_memory": 0.10,
            "semantic_memory": 0.20,
            "belief_consistency": 0.15,
            "commonsense": 0.10,
            "metacognitive": 0.10,
        }

        scores = {
            "recency": self._score_recency(variant, inputs),
            "working_memory": self._score_working_memory(variant, inputs),
            "episodic_memory": self._score_episodic_memory(variant, inputs),
            "semantic_memory": self._score_semantic_memory(variant, inputs),
            "belief_consistency": self._score_belief_consistency(variant, inputs),
            "commonsense": self._score_commonsense(variant, inputs),
            "metacognitive": self._score_metacognitive(variant, inputs),
        }

        return sum(scores[k] * weights[k] for k in weights)
```

### 1.3 Variation Threshold and Storage

```python
@dataclass
class VariationStoragePolicy:
    """Policy for which variations to store."""

    # Minimum confidence to store as a variation
    min_confidence_threshold: float = 0.3

    # Maximum number of variations to store per statement
    max_stored_variations: int = 3

    # Only store variations if confidence gap is significant
    min_confidence_gap: float = 0.15

    # Store all variations above this threshold regardless of gap
    high_confidence_threshold: float = 0.8


def should_store_variation(
    primary: ExpansionVariant,
    candidate: ExpansionVariant,
    policy: VariationStoragePolicy,
) -> bool:
    """Determine if a variation should be stored alongside the primary."""

    # Must meet minimum threshold
    if candidate.score < policy.min_confidence_threshold:
        return False

    # High confidence always stored
    if candidate.score >= policy.high_confidence_threshold:
        return True

    # Must have meaningful difference from primary
    gap = abs(primary.score - candidate.score)
    if gap < policy.min_confidence_gap:
        return False

    return True
```

---

## 2. Graph Structure in Qdrant

### 2.1 Hybrid Architecture: Vector + Graph

Based on [GraphRAG with Qdrant and Neo4j](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/) and the [HybridRAG approach](https://memgraph.com/blog/why-hybridrag), we use a **dual representation**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       HYBRID SEMANTIC STORAGE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    QDRANT (Vector Database)                          │    │
│  │                                                                      │    │
│  │  Collection: semantic_nodes                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Point {                                                     │    │    │
│  │  │    id: "node_uuid",                                          │    │    │
│  │  │    vector: [0.123, 0.456, ...],  // Embedding of node content│    │    │
│  │  │    payload: {                                                │    │    │
│  │  │      content: "Doug prefers tea in the morning",             │    │    │
│  │  │      node_type: "PREFERENCE",                                │    │    │
│  │  │      entity_id: "entity_doug",                               │    │    │
│  │  │      synset_ids: ["person.n.01", "tea.n.01", "morning.n.01"],│    │    │
│  │  │      lemmas: ["doug", "prefer", "tea", "morning"],           │    │    │
│  │  │      memory_layer: "semantic",                               │    │    │
│  │  │      confidence: 0.85,                                       │    │    │
│  │  │      variation_of: null,  // or parent node_id               │    │    │
│  │  │      variation_rank: 0,   // 0 = primary, 1+ = alternatives  │    │    │
│  │  │      created_at: "2025-12-30T10:00:00Z",                     │    │    │
│  │  │      graph_edges: [  // Embedded edge references             │    │    │
│  │  │        {"rel": "PREFERS", "target": "entity_tea",            │    │    │
│  │  │         "context": "morning"},                               │    │    │
│  │  │        {"rel": "HAS_CONDITION", "target": "time_morning"}    │    │    │
│  │  │      ]                                                       │    │    │
│  │  │    }                                                         │    │    │
│  │  │  }                                                           │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                      │    │
│  │  Collection: semantic_edges                                          │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  Point {                                                     │    │    │
│  │  │    id: "edge_uuid",                                          │    │    │
│  │  │    vector: [0.789, 0.012, ...],  // Embedding of full triple │    │    │
│  │  │    payload: {                                                │    │    │
│  │  │      source_id: "entity_doug",                               │    │    │
│  │  │      relation: "PREFERS",                                    │    │    │
│  │  │      target_id: "entity_tea",                                │    │    │
│  │  │      context: {"temporal": "morning"},                       │    │    │
│  │  │      confidence: 0.85,                                       │    │    │
│  │  │      source_node_id: "node_uuid",  // Link to source node    │    │    │
│  │  │    }                                                         │    │    │
│  │  │  }                                                           │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    IN-MEMORY GRAPH (NetworkX)                        │    │
│  │                    (Loaded on demand for traversal)                  │    │
│  │                                                                      │    │
│  │  G.nodes["entity_doug"] = {                                          │    │
│  │    "synset": "person.n.01",                                          │    │
│  │    "lemma": "doug",                                                  │    │
│  │    "aliases": ["douglas", "he", "him"],                              │    │
│  │    "qdrant_ids": ["node_1", "node_2", ...]                          │    │
│  │  }                                                                   │    │
│  │                                                                      │    │
│  │  G.edges["entity_doug", "entity_tea"] = {                           │    │
│  │    "relation": "PREFERS",                                            │    │
│  │    "context": {"temporal": "morning"},                               │    │
│  │    "confidence": 0.85,                                               │    │
│  │    "qdrant_edge_id": "edge_uuid"                                     │    │
│  │  }                                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Semantic Triple Embedding Strategy

Following [Knowledge Graph Embedding research](https://en.wikipedia.org/wiki/Knowledge_graph_embedding):

```python
class SemanticTripleEmbedder:
    """
    Embed semantic triples for vector storage.

    Uses TransE-inspired approach: embed (subject, relation, object)
    such that subject_embedding + relation_embedding ≈ object_embedding

    For Qdrant storage, we embed the FULL triple as a single vector
    for similarity search, with structured payload for graph traversal.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(embedding_model)

    async def embed_triple(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict | None = None,
    ) -> tuple[list[float], dict]:
        """
        Embed a semantic triple.

        Returns:
            (embedding_vector, structured_payload)
        """
        # Create natural language representation of triple
        triple_text = self._triple_to_text(subject, relation, object_, context)

        # Embed the full triple
        embedding = self.model.encode(triple_text).tolist()

        # Build structured payload
        payload = {
            "subject": subject,
            "relation": relation,
            "object": object_,
            "context": context or {},
            "triple_text": triple_text,
        }

        return embedding, payload

    def _triple_to_text(
        self,
        subject: str,
        relation: str,
        object_: str,
        context: dict | None,
    ) -> str:
        """Convert triple to natural language for embedding."""

        # Relation-specific templates
        templates = {
            "PREFERS": "{subject} prefers {object}",
            "LIKES": "{subject} likes {object}",
            "IS_A": "{subject} is a {object}",
            "HAS_PROPERTY": "{subject} has {object}",
            "LOCATED_AT": "{subject} is located at {object}",
            "CAUSES": "{subject} causes {object}",
            "BELIEVES": "{subject} believes that {object}",
        }

        template = templates.get(relation, "{subject} {relation} {object}")
        text = template.format(subject=subject, relation=relation.lower(), object=object_)

        # Add context if present
        if context:
            if "temporal" in context:
                text += f" in the {context['temporal']}"
            if "location" in context:
                text += f" at {context['location']}"
            if "condition" in context:
                text += f" when {context['condition']}"

        return text
```

### 2.3 Graph Query Operations

```python
class SemanticGraphQuery:
    """Query semantic graph using hybrid vector + structure approach."""

    def __init__(self, qdrant_client, graph_cache: nx.DiGraph):
        self.qdrant = qdrant_client
        self.graph = graph_cache

    async def find_related(
        self,
        query: str,
        max_hops: int = 2,
        min_similarity: float = 0.7,
    ) -> list[SemanticNode]:
        """
        Find semantically related nodes using hybrid approach.

        1. Vector search for initial matches
        2. Graph traversal for relationship expansion
        3. Re-rank by combined score
        """

        # Step 1: Vector similarity search
        vector_results = await self._vector_search(query, limit=10)

        # Step 2: Graph expansion from top matches
        expanded = set()
        for result in vector_results[:5]:
            entity_id = result.payload.get("entity_id")
            if entity_id and entity_id in self.graph:
                neighbors = self._get_neighbors(entity_id, max_hops)
                expanded.update(neighbors)

        # Step 3: Fetch expanded nodes from Qdrant
        expanded_results = await self._fetch_nodes(list(expanded))

        # Step 4: Combine and re-rank
        all_results = vector_results + expanded_results
        return self._rerank(all_results, query)

    async def find_conflicts(
        self,
        new_triple: tuple[str, str, str],
    ) -> list[Conflict]:
        """
        Find triples that conflict with a new assertion.

        Uses graph structure to find:
        1. Same subject + same relation + different object
        2. Contradictory relations (LIKES vs DISLIKES)
        3. Mutually exclusive properties
        """
        subject, relation, object_ = new_triple

        conflicts = []

        # Check existing edges from subject
        if subject in self.graph:
            for _, target, edge_data in self.graph.out_edges(subject, data=True):
                existing_rel = edge_data.get("relation")

                # Same relation, different object?
                if existing_rel == relation and target != object_:
                    conflicts.append(Conflict(
                        existing=(subject, existing_rel, target),
                        new=new_triple,
                        conflict_type="DIFFERENT_OBJECT",
                        confidence=edge_data.get("confidence", 0.5),
                    ))

                # Contradictory relation?
                if self._are_contradictory(existing_rel, relation):
                    conflicts.append(Conflict(
                        existing=(subject, existing_rel, target),
                        new=new_triple,
                        conflict_type="CONTRADICTORY_RELATION",
                        confidence=edge_data.get("confidence", 0.5),
                    ))

        return conflicts

    def _are_contradictory(self, rel_a: str, rel_b: str) -> bool:
        """Check if two relations are contradictory."""
        contradictions = {
            ("LIKES", "DISLIKES"),
            ("PREFERS", "AVOIDS"),
            ("BELIEVES", "DOUBTS"),
            ("CAN", "CANNOT"),
            ("IS_A", "IS_NOT_A"),
        }
        pair = tuple(sorted([rel_a, rel_b]))
        return pair in contradictions
```

---

## 3. Storing Memories as Semantic Graphs

### 3.1 Memory Storage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY STORAGE PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: "Doug mentioned he prefers tea in the morning"                       │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: SEMANTIC EXPANSION                                          │    │
│  │                                                                      │    │
│  │  Primary Interpretation (confidence: 0.85):                          │    │
│  │    - (Doug, PREFERS, tea, context={temporal: morning})              │    │
│  │    - (Doug, IS_A, person)                                           │    │
│  │    - (tea, IS_A, beverage)                                          │    │
│  │                                                                      │    │
│  │  Variation 1 (confidence: 0.65):                                     │    │
│  │    - (Doug, SOMETIMES_PREFERS, tea, context={temporal: morning})    │    │
│  │    - Interpretation: Occasional preference, not absolute            │    │
│  │                                                                      │    │
│  │  Variation 2 (confidence: 0.40):                                     │    │
│  │    - (Doug, PREFERS, tea)  [no temporal context]                    │    │
│  │    - Interpretation: General preference for tea                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: WORD SENSE DISAMBIGUATION                                   │    │
│  │                                                                      │    │
│  │  "Doug" → entity_doug (resolved from context)                        │    │
│  │  "prefers" → prefer.v.01 (WordNet: like better)                     │    │
│  │  "tea" → tea.n.01 (WordNet: beverage, NOT tea.n.02: meal)           │    │
│  │  "morning" → morning.n.01 (time before noon)                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: STORE IN QDRANT                                             │    │
│  │                                                                      │    │
│  │  semantic_nodes collection:                                          │    │
│  │    Node 1: Primary interpretation (variation_rank=0)                 │    │
│  │    Node 2: Variation 1 (variation_rank=1, variation_of=Node1)       │    │
│  │    Node 3: Variation 2 (variation_rank=2, variation_of=Node1)       │    │
│  │                                                                      │    │
│  │  semantic_edges collection:                                          │    │
│  │    Edge 1: (entity_doug, PREFERS, entity_tea) @ confidence=0.85     │    │
│  │    Edge 2: (entity_doug, SOMETIMES_PREFERS, entity_tea) @ conf=0.65 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: UPDATE GRAPH CACHE                                          │    │
│  │                                                                      │    │
│  │  Add nodes: entity_doug, entity_tea, time_morning                   │    │
│  │  Add edges with Qdrant IDs for later retrieval                      │    │
│  │  Link to existing graph nodes if entities already exist             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 5: CROSS-LAYER LINKING                                         │    │
│  │                                                                      │    │
│  │  Check for related nodes in other memory layers:                     │    │
│  │    - Semantic: (Doug, LIKES, coffee) - POTENTIAL CONFLICT!          │    │
│  │    - Episodic: "Yesterday's coffee conversation"                    │    │
│  │                                                                      │    │
│  │  Create cross-layer association edges                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Variation Storage Implementation

```python
@dataclass
class StoredVariation:
    """A stored interpretation variation."""

    node_id: str  # Qdrant point ID
    primary_id: str | None  # ID of primary interpretation (None if this is primary)
    variation_rank: int  # 0 = primary, 1+ = alternative
    confidence: float
    semantic_frame: SemanticFrame

    # Synset IDs for word sense disambiguation
    synset_ids: dict[str, str]  # word -> synset_id

    # Cross-references
    related_in_layers: dict[str, list[str]]  # layer_name -> [node_ids]


class SemanticMemoryStore:
    """Store memories as semantically expanded graphs with variations."""

    async def store_with_expansion(
        self,
        raw_memory: str,
        memory_layer: str,
        inputs: ExpansionInput,
    ) -> list[StoredVariation]:
        """
        Store a memory with full semantic expansion.

        Returns list of stored variations (primary + alternatives).
        """

        # Generate expansion variations
        variations = await self.expansion_service.generate_variations(
            raw_memory, inputs
        )

        stored = []
        primary_id = None

        for i, variant in enumerate(variations):
            # Skip variations below threshold
            if i > 0 and not should_store_variation(
                variations[0], variant, self.storage_policy
            ):
                continue

            # Perform word sense disambiguation
            synset_ids = await self.wsd_service.disambiguate(
                variant.frame, inputs
            )

            # Embed and store in Qdrant
            node_id = await self._store_node(
                variant=variant,
                synset_ids=synset_ids,
                memory_layer=memory_layer,
                primary_id=primary_id,
                variation_rank=i,
            )

            if i == 0:
                primary_id = node_id

            # Store edges for semantic triples
            await self._store_edges(variant.frame.triples, node_id)

            # Find and create cross-layer links
            cross_links = await self._find_cross_layer_links(
                variant, memory_layer
            )
            await self._store_cross_links(node_id, cross_links)

            stored.append(StoredVariation(
                node_id=node_id,
                primary_id=primary_id if i > 0 else None,
                variation_rank=i,
                confidence=variant.score,
                semantic_frame=variant.frame,
                synset_ids=synset_ids,
                related_in_layers=cross_links,
            ))

        return stored
```

---

## 4. Word Sense Disambiguation System

### 4.1 Identifier System Architecture

Based on [WordNet](https://wordnet.princeton.edu/) synset structure and [BabelNet](https://babelnet.org/about) multilingual linking:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WORD SENSE IDENTIFIER SYSTEM                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: "I went to the bank"                                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: TOKENIZE AND LEMMATIZE (spaCy)                             │    │
│  │                                                                      │    │
│  │  Token    │ Lemma   │ POS  │ Morphology                             │    │
│  │  ─────────┼─────────┼──────┼─────────────                            │    │
│  │  I        │ I       │ PRON │ Case=Nom|Number=Sing|Person=1          │    │
│  │  went     │ go      │ VERB │ Tense=Past|VerbForm=Fin                │    │
│  │  to       │ to      │ ADP  │                                        │    │
│  │  the      │ the     │ DET  │                                        │    │
│  │  bank     │ bank    │ NOUN │ Number=Sing                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: GET CANDIDATE SYNSETS (WordNet/BabelNet)                   │    │
│  │                                                                      │    │
│  │  "bank" (noun) candidates:                                          │    │
│  │    bank.n.01: financial institution                                 │    │
│  │    bank.n.02: sloping land beside water                            │    │
│  │    bank.n.03: supply or stock held in reserve                      │    │
│  │    bank.n.05: a building for banking                               │    │
│  │    bank.n.07: a slope in turning (aviation)                        │    │
│  │                                                                      │    │
│  │  "go" (verb) candidates:                                            │    │
│  │    go.v.01: change location; move                                   │    │
│  │    go.v.04: follow a certain course                                │    │
│  │    go.v.09: be spent                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: CONTEXTUAL DISAMBIGUATION                                   │    │
│  │                                                                      │    │
│  │  Method 1: Lesk Algorithm (gloss overlap)                           │    │
│  │    - "went to" suggests physical movement                           │    │
│  │    - "bank" with "went to" → likely bank.n.01 or bank.n.05         │    │
│  │                                                                      │    │
│  │  Method 2: LLM-based WSD (more accurate)                            │    │
│  │    Prompt: "Which meaning of 'bank' is used in 'I went to          │    │
│  │             the bank'? Options: [financial institution,            │    │
│  │             riverbank, reserve supply, ...]"                        │    │
│  │    → bank.n.01 (financial institution) with 95% confidence         │    │
│  │                                                                      │    │
│  │  Method 3: Embedding similarity                                     │    │
│  │    - Embed sentence context                                         │    │
│  │    - Compare to embeddings of each synset definition               │    │
│  │    - Highest similarity wins                                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                          │                                                   │
│                          ▼                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: ASSIGN CANONICAL IDENTIFIERS                               │    │
│  │                                                                      │    │
│  │  Final disambiguation:                                               │    │
│  │    "bank" → {                                                       │    │
│  │      synset_id: "bank.n.01",                                        │    │
│  │      wikidata_id: "Q22687",  // bank as financial institution      │    │
│  │      babelnet_id: "bn:00008364n",                                   │    │
│  │      definition: "a financial institution",                         │    │
│  │      confidence: 0.95                                               │    │
│  │    }                                                                │    │
│  │                                                                      │    │
│  │  This ID is used in all graph edges to prevent false associations  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Implementation

```python
@dataclass
class WordSense:
    """A disambiguated word sense with canonical identifiers."""

    # Original text
    surface_form: str  # "banks" (as it appeared)
    lemma: str  # "bank" (root form)
    pos: str  # "NOUN"

    # Canonical identifiers
    synset_id: str  # "bank.n.01" (WordNet)
    wikidata_id: str | None  # "Q22687" (Wikidata QID)
    babelnet_id: str | None  # "bn:00008364n" (BabelNet)

    # Metadata
    definition: str
    confidence: float
    disambiguation_method: str  # "lesk", "llm", "embedding"


class WordSenseDisambiguator:
    """
    Disambiguate word senses using multiple methods.

    Based on research from:
    - WordNet synsets (https://wordnet.princeton.edu/)
    - BabelNet multilingual WSD (https://babelnet.org/)
    - spaCy lemmatization (https://spacy.io/)
    """

    def __init__(
        self,
        nlp,  # spaCy model
        wordnet,  # NLTK WordNet
        llm: LLMProvider | None = None,
        embedding_model = None,
    ):
        self.nlp = nlp
        self.wordnet = wordnet
        self.llm = llm
        self.embedding_model = embedding_model

    async def disambiguate_sentence(
        self,
        sentence: str,
    ) -> dict[str, WordSense]:
        """
        Disambiguate all content words in a sentence.

        Returns:
            dict mapping token position to WordSense
        """

        # Step 1: Parse with spaCy
        doc = self.nlp(sentence)

        results = {}

        for token in doc:
            # Skip function words
            if token.pos_ in ("DET", "ADP", "CCONJ", "SCONJ", "PUNCT"):
                continue

            # Get candidate synsets
            candidates = self._get_synset_candidates(token.lemma_, token.pos_)

            if not candidates:
                continue

            if len(candidates) == 1:
                # Unambiguous
                sense = self._synset_to_word_sense(
                    token, candidates[0], 1.0, "unambiguous"
                )
            else:
                # Disambiguate
                sense = await self._disambiguate_token(token, candidates, doc)

            results[f"{token.i}:{token.text}"] = sense

        return results

    async def _disambiguate_token(
        self,
        token,
        candidates: list,
        doc,
    ) -> WordSense:
        """Disambiguate a single token using best available method."""

        # Try LLM-based WSD (most accurate)
        if self.llm:
            sense = await self._llm_disambiguate(token, candidates, doc)
            if sense and sense.confidence > 0.8:
                return sense

        # Try embedding similarity
        if self.embedding_model:
            sense = await self._embedding_disambiguate(token, candidates, doc)
            if sense and sense.confidence > 0.7:
                return sense

        # Fall back to Lesk algorithm
        return self._lesk_disambiguate(token, candidates, doc)

    async def _llm_disambiguate(
        self,
        token,
        candidates: list,
        doc,
    ) -> WordSense | None:
        """Use LLM for word sense disambiguation."""

        options = "\n".join([
            f"- {c.name()}: {c.definition()}"
            for c in candidates[:5]  # Limit to top 5
        ])

        prompt = f"""Determine which meaning of '{token.lemma_}' is used in this sentence:

Sentence: "{doc.text}"

Possible meanings:
{options}

Respond in XML:
<disambiguation>
    <synset_id>The WordNet synset ID (e.g., bank.n.01)</synset_id>
    <confidence>0.0-1.0</confidence>
    <reasoning>Brief explanation</reasoning>
</disambiguation>
"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )

        return self._parse_llm_response(response, token, candidates)

    def _get_synset_candidates(
        self,
        lemma: str,
        pos: str,
    ) -> list:
        """Get WordNet synset candidates for a lemma."""

        # Map spaCy POS to WordNet POS
        pos_map = {
            "NOUN": "n",
            "VERB": "v",
            "ADJ": "a",
            "ADV": "r",
        }

        wn_pos = pos_map.get(pos)
        if not wn_pos:
            return []

        return self.wordnet.synsets(lemma, pos=wn_pos)
```

### 4.3 Preventing False Associations

```python
class SenseAwareGraphQuery:
    """
    Query semantic graph with word sense awareness.

    Prevents false associations like:
    - "bank" (financial) being related to "bank" (river)
    - "lead" (metal) being related to "lead" (guide)
    """

    async def find_related_by_sense(
        self,
        query: str,
        target_synset: str,
    ) -> list[SemanticNode]:
        """
        Find nodes related to a specific word sense.

        Only returns nodes where the word sense matches.
        """

        # Filter by synset_id in Qdrant payload
        filter_condition = {
            "must": [
                {
                    "key": "synset_ids",
                    "match": {"any": [target_synset]}
                }
            ]
        }

        results = await self.qdrant.search(
            collection_name="semantic_nodes",
            query_vector=self._embed(query),
            query_filter=filter_condition,
        )

        return results

    def are_same_sense(
        self,
        sense_a: WordSense,
        sense_b: WordSense,
    ) -> bool:
        """Check if two word senses refer to the same concept."""

        # Direct synset match
        if sense_a.synset_id == sense_b.synset_id:
            return True

        # Check hypernym chain (more general concept)
        if self._share_hypernym(sense_a.synset_id, sense_b.synset_id):
            return True

        # Check Wikidata ID if available
        if sense_a.wikidata_id and sense_b.wikidata_id:
            return sense_a.wikidata_id == sense_b.wikidata_id

        return False
```

---

## 5. Cross-Layer Memory Associations

### 5.1 Association Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CROSS-LAYER MEMORY ASSOCIATIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  METACOGNITIVE (Permanent)                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  "I tend to misinterpret temporal references"                       │    │
│  │  synsets: [temporal.n.01, reference.n.01]                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                                                    │
│         │ CALIBRATES (cross-layer edge)                                     │
│         ▼                                                                    │
│  SEMANTIC (6 months)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  (Doug, LIKES, coffee) confidence: 0.9                              │    │
│  │  synsets: [person.n.01:Doug, like.v.02, coffee.n.01]                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│         │                                         │                          │
│         │ POTENTIALLY_CONFLICTS                   │ MENTIONED_IN             │
│         ▼                                         ▼                          │
│  SEMANTIC                                    EPISODIC (2 weeks)              │
│  ┌───────────────────────────┐    ┌────────────────────────────────────┐    │
│  │  (Doug, PREFERS, tea)     │    │  "Dec 28: Discussed morning        │    │
│  │  context: {morning}       │    │   beverage preferences with Doug"  │    │
│  │  synsets: same as above   │◄──►│                                    │    │
│  │  + [morning.n.01]         │    │  mentioned_entities: [Doug, tea]   │    │
│  └───────────────────────────┘    └────────────────────────────────────┘    │
│         │                                         │                          │
│         │ DERIVED_FROM                            │ SUPPORTS                 │
│         ▼                                         ▼                          │
│  WORKING (5 minutes)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Observation: "He prefers tea in the morning"                       │    │
│  │  source_agent: agent_1, confidence: 0.85                            │    │
│  │  links: [semantic_node_123, episodic_node_456]                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Association Types

```python
class CrossLayerRelation(str, Enum):
    """Types of cross-layer associations."""

    # Derivation relationships
    DERIVED_FROM = "derived_from"  # Working → Semantic (promoted)
    SUMMARIZES = "summarizes"  # Episodic → Working (condensed)
    GENERALIZES = "generalizes"  # Semantic → Metacognitive (abstracted)

    # Support relationships
    SUPPORTS = "supports"  # Episodic → Semantic (evidence)
    CONTRADICTS = "contradicts"  # Any → Any (conflict)
    CALIBRATES = "calibrates"  # Metacognitive → Any (adjustment)

    # Reference relationships
    MENTIONED_IN = "mentioned_in"  # Entity → Episodic (occurrence)
    ELABORATES = "elaborates"  # Semantic → Semantic (detail)
    CONTEXTUALIZES = "contextualizes"  # Episodic → Working (background)


@dataclass
class CrossLayerEdge:
    """An association between nodes in different memory layers."""

    source_node_id: str
    source_layer: str  # "working", "episodic", "semantic", "metacognitive"

    target_node_id: str
    target_layer: str

    relation: CrossLayerRelation

    # Metadata
    confidence: float
    created_at: datetime
    context: dict | None  # Why this association exists


class CrossLayerLinker:
    """Create and manage cross-layer associations."""

    async def find_cross_layer_links(
        self,
        new_node: SemanticNode,
        source_layer: str,
    ) -> list[CrossLayerEdge]:
        """
        Find nodes in other layers that should be linked.

        Uses:
        1. Entity matching (same entities mentioned)
        2. Synset matching (same word senses)
        3. Semantic similarity (related concepts)
        4. Temporal proximity (recent mentions)
        """

        edges = []

        # Get entities and synsets from new node
        entities = new_node.payload.get("entities", [])
        synsets = new_node.payload.get("synset_ids", [])

        # Search each other layer
        for layer in ["working", "episodic", "semantic", "metacognitive"]:
            if layer == source_layer:
                continue

            # Find by entity overlap
            entity_matches = await self._find_by_entities(entities, layer)
            for match in entity_matches:
                edges.append(CrossLayerEdge(
                    source_node_id=new_node.id,
                    source_layer=source_layer,
                    target_node_id=match.id,
                    target_layer=layer,
                    relation=self._infer_relation(source_layer, layer, "entity"),
                    confidence=match.overlap_score,
                    created_at=datetime.now(),
                    context={"link_type": "entity_overlap", "entities": entities},
                ))

            # Find by synset overlap (same word senses)
            synset_matches = await self._find_by_synsets(synsets, layer)
            for match in synset_matches:
                edges.append(CrossLayerEdge(
                    source_node_id=new_node.id,
                    source_layer=source_layer,
                    target_node_id=match.id,
                    target_layer=layer,
                    relation=self._infer_relation(source_layer, layer, "synset"),
                    confidence=match.overlap_score,
                    created_at=datetime.now(),
                    context={"link_type": "synset_overlap", "synsets": synsets},
                ))

            # Check for conflicts
            conflicts = await self._find_conflicts(new_node, layer)
            for conflict in conflicts:
                edges.append(CrossLayerEdge(
                    source_node_id=new_node.id,
                    source_layer=source_layer,
                    target_node_id=conflict.conflicting_node_id,
                    target_layer=layer,
                    relation=CrossLayerRelation.CONTRADICTS,
                    confidence=conflict.confidence,
                    created_at=datetime.now(),
                    context={"conflict_type": conflict.conflict_type},
                ))

        return edges

    def _infer_relation(
        self,
        source_layer: str,
        target_layer: str,
        link_type: str,
    ) -> CrossLayerRelation:
        """Infer the appropriate relation based on layers."""

        layer_order = ["working", "episodic", "semantic", "metacognitive"]
        source_idx = layer_order.index(source_layer)
        target_idx = layer_order.index(target_layer)

        if source_idx < target_idx:
            # Linking from transient to permanent
            return CrossLayerRelation.SUPPORTS
        elif source_idx > target_idx:
            # Linking from permanent to transient
            return CrossLayerRelation.CONTEXTUALIZES
        else:
            return CrossLayerRelation.ELABORATES
```

---

## 6. Evolutionary Design Framework

### 6.1 What Can Evolve

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EVOLUTIONARY COMPONENTS                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. EXPANSION PROMPTS                                                │    │
│  │     - The LLM prompt that extracts semantic frames                  │    │
│  │     - Can mutate: wording, examples, structure                      │    │
│  │     - Fitness: How accurately it extracts meaning                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. WEIGHTING SCHEMES                                                │    │
│  │     - How cognitive factors are weighted for variants               │    │
│  │     - Can mutate: weight values, factor combinations                │    │
│  │     - Fitness: How often top variant is correct                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. ASSOCIATION RULES                                                │    │
│  │     - When to create cross-layer links                              │    │
│  │     - Can mutate: thresholds, relation inference                    │    │
│  │     - Fitness: Retrieval precision/recall                           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. VARIATION STORAGE POLICY                                         │    │
│  │     - Which variations to store, thresholds                         │    │
│  │     - Can mutate: thresholds, max counts                            │    │
│  │     - Fitness: Storage efficiency vs. interpretation coverage       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5. WSD STRATEGIES                                                   │    │
│  │     - How word senses are disambiguated                             │    │
│  │     - Can mutate: method selection, confidence thresholds           │    │
│  │     - Fitness: Disambiguation accuracy                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Evolutionary Framework

```python
@dataclass
class EvolvableConfig:
    """A configuration that can evolve."""

    config_id: str
    generation: int
    parent_ids: list[str]  # For crossover tracking

    # The actual configuration values
    expansion_prompt: str
    weighting_scheme: dict[str, float]
    association_thresholds: dict[str, float]
    variation_policy: VariationStoragePolicy
    wsd_config: WSDConfig

    # Fitness tracking
    fitness_scores: dict[str, float]  # metric_name -> score
    evaluation_count: int


class SemanticExpansionEvolver:
    """
    Evolutionary optimization for semantic expansion.

    Inspired by PromptBreeder, applies genetic algorithms to:
    - Prompts
    - Weight schemes
    - Association rules
    """

    def __init__(
        self,
        population_size: int = 10,
        mutation_rate: float = 0.2,
        crossover_rate: float = 0.3,
        elite_count: int = 2,
    ):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count

        self.population: list[EvolvableConfig] = []
        self.generation = 0

    async def evolve_generation(
        self,
        test_cases: list[ExpansionTestCase],
    ) -> EvolvableConfig:
        """
        Run one generation of evolution.

        Returns the best config from this generation.
        """

        # Evaluate current population
        for config in self.population:
            config.fitness_scores = await self._evaluate(config, test_cases)

        # Sort by overall fitness
        self.population.sort(
            key=lambda c: self._overall_fitness(c.fitness_scores),
            reverse=True,
        )

        # Keep elites
        new_population = self.population[:self.elite_count]

        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            if random.random() < self.crossover_rate:
                # Crossover
                parent_a, parent_b = random.sample(self.population[:5], 2)
                child = self._crossover(parent_a, parent_b)
            else:
                # Mutation from top performers
                parent = random.choice(self.population[:5])
                child = self._mutate(parent)

            new_population.append(child)

        self.population = new_population
        self.generation += 1

        return self.population[0]  # Best config

    def _mutate(self, config: EvolvableConfig) -> EvolvableConfig:
        """Create mutated variant of a config."""

        new_config = deepcopy(config)
        new_config.config_id = str(uuid.uuid4())
        new_config.generation = self.generation + 1
        new_config.parent_ids = [config.config_id]
        new_config.evaluation_count = 0

        # Randomly mutate one aspect
        mutation_type = random.choice([
            "prompt", "weights", "thresholds", "policy"
        ])

        if mutation_type == "prompt":
            new_config.expansion_prompt = self._mutate_prompt(
                config.expansion_prompt
            )
        elif mutation_type == "weights":
            new_config.weighting_scheme = self._mutate_weights(
                config.weighting_scheme
            )
        elif mutation_type == "thresholds":
            new_config.association_thresholds = self._mutate_thresholds(
                config.association_thresholds
            )
        elif mutation_type == "policy":
            new_config.variation_policy = self._mutate_policy(
                config.variation_policy
            )

        return new_config

    def _crossover(
        self,
        parent_a: EvolvableConfig,
        parent_b: EvolvableConfig,
    ) -> EvolvableConfig:
        """Create child by combining two parents."""

        return EvolvableConfig(
            config_id=str(uuid.uuid4()),
            generation=self.generation + 1,
            parent_ids=[parent_a.config_id, parent_b.config_id],

            # Take prompt from one parent
            expansion_prompt=random.choice([
                parent_a.expansion_prompt,
                parent_b.expansion_prompt,
            ]),

            # Blend weights
            weighting_scheme=self._blend_weights(
                parent_a.weighting_scheme,
                parent_b.weighting_scheme,
            ),

            # Take thresholds from one parent
            association_thresholds=random.choice([
                parent_a.association_thresholds,
                parent_b.association_thresholds,
            ]),

            # Take policy from one parent
            variation_policy=random.choice([
                parent_a.variation_policy,
                parent_b.variation_policy,
            ]),

            wsd_config=random.choice([
                parent_a.wsd_config,
                parent_b.wsd_config,
            ]),

            fitness_scores={},
            evaluation_count=0,
        )

    async def _evaluate(
        self,
        config: EvolvableConfig,
        test_cases: list[ExpansionTestCase],
    ) -> dict[str, float]:
        """Evaluate a config against test cases."""

        scores = {
            "expansion_accuracy": 0.0,
            "variant_ranking": 0.0,
            "wsd_accuracy": 0.0,
            "association_precision": 0.0,
            "association_recall": 0.0,
        }

        for test in test_cases:
            result = await self._run_test(config, test)

            # Update scores
            scores["expansion_accuracy"] += result.expansion_correct
            scores["variant_ranking"] += result.correct_variant_rank
            scores["wsd_accuracy"] += result.wsd_correct
            scores["association_precision"] += result.association_precision
            scores["association_recall"] += result.association_recall

        # Normalize
        n = len(test_cases)
        return {k: v / n for k, v in scores.items()}
```

### 6.3 Test Framework for Evolution

```python
@dataclass
class ExpansionTestCase:
    """A test case for evaluating semantic expansion."""

    # Input
    statement: str
    context: ExpansionInput

    # Expected outputs
    expected_primary_interpretation: SemanticFrame
    expected_alternatives: list[SemanticFrame]
    expected_synsets: dict[str, str]
    expected_cross_layer_links: list[CrossLayerEdge]

    # Pre-loaded memory state
    preloaded_working: list[SharedObservation]
    preloaded_episodic: list[EpisodeSummary]
    preloaded_semantic: list[SemanticNode]
    preloaded_metacognitive: list[MetacognitiveEntry]


class EvolutionaryTestSuite:
    """
    Test suite designed for evolutionary evaluation.

    Includes:
    1. Semantically expanded memories across all layers
    2. Cross-layer associations
    3. Variation scenarios
    4. WSD challenges
    """

    def generate_test_cases(self) -> list[ExpansionTestCase]:
        """Generate test cases for evolution."""

        return [
            # Test 1: Simple preference with potential conflict
            ExpansionTestCase(
                statement="He prefers tea in the morning",
                context=self._build_context(
                    recent_utterances=["Doug mentioned he was tired"],
                    session_entities={"Doug": EntityInfo(type="PERSON")},
                ),
                expected_primary_interpretation=SemanticFrame(
                    triples=[
                        ("Doug", "PREFERS", "tea", {"temporal": "morning"}),
                    ],
                ),
                expected_alternatives=[
                    SemanticFrame(
                        triples=[("unknown_male", "PREFERS", "tea")],
                    ),
                ],
                expected_synsets={
                    "tea": "tea.n.01",  # beverage, not meal
                    "morning": "morning.n.01",
                },
                expected_cross_layer_links=[
                    CrossLayerEdge(
                        source_layer="working",
                        target_layer="semantic",
                        relation=CrossLayerRelation.CONTRADICTS,
                        # Links to preloaded "Doug likes coffee"
                    ),
                ],
                preloaded_semantic=[
                    SemanticNode(
                        content="Doug likes coffee",
                        triples=[("Doug", "LIKES", "coffee")],
                        synsets={"coffee": "coffee.n.01"},
                    ),
                ],
            ),

            # Test 2: Word sense disambiguation challenge
            ExpansionTestCase(
                statement="I need to go to the bank before it closes",
                context=self._build_context(
                    recent_utterances=["I need to deposit this check"],
                ),
                expected_synsets={
                    "bank": "bank.n.01",  # financial institution
                    "close": "close.v.01",  # cease operation
                },
            ),

            # Test 3: Cross-layer association test
            ExpansionTestCase(
                statement="Doug always does this",
                context=self._build_context(
                    recent_utterances=["Doug forgot the meeting again"],
                ),
                expected_cross_layer_links=[
                    # Should link to episodic memory of past forgetfulness
                    CrossLayerEdge(
                        source_layer="working",
                        target_layer="episodic",
                        relation=CrossLayerRelation.SUPPORTS,
                    ),
                    # Should link to metacognitive pattern recognition
                    CrossLayerEdge(
                        source_layer="working",
                        target_layer="metacognitive",
                        relation=CrossLayerRelation.GENERALIZES,
                    ),
                ],
                preloaded_episodic=[
                    EpisodeSummary(
                        content="Dec 15: Doug forgot team meeting",
                        entities=["Doug"],
                    ),
                    EpisodeSummary(
                        content="Dec 20: Doug missed deadline",
                        entities=["Doug"],
                    ),
                ],
                preloaded_metacognitive=[
                    MetacognitiveEntry(
                        content="Doug has a pattern of forgetting commitments",
                        pattern_type="BEHAVIORAL_PATTERN",
                    ),
                ],
            ),
        ]
```

---

## 7. Integration Architecture

### 7.1 Complete System Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTEGRATED SEMANTIC EXPANSION SYSTEM                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Statement                                                             │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. LINGUISTIC PREPROCESSING (spaCy)                                │    │
│  │     - Tokenization, POS tagging                                     │    │
│  │     - Lemmatization (running → run)                                 │    │
│  │     - Dependency parsing                                            │    │
│  │     - Named entity recognition                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. WORD SENSE DISAMBIGUATION (WordNet/BabelNet + LLM)              │    │
│  │     - Get candidate synsets                                         │    │
│  │     - Context-based disambiguation                                  │    │
│  │     - Assign canonical IDs (synset_id, wikidata_id)                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. GATHER EXPANSION INPUTS                                          │    │
│  │     - Pull from all 4 memory layers                                 │    │
│  │     - Get relevant beliefs                                          │    │
│  │     - Get ATOMIC/COMET inferences                                   │    │
│  │     - Assemble ExpansionInput object                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  4. SEMANTIC FRAME EXPANSION (LLM)                                   │    │
│  │     - Extract presuppositions                                       │    │
│  │     - Generate semantic triples                                     │    │
│  │     - Identify ambiguities                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  5. VARIATION GENERATION                                             │    │
│  │     - Generate interpretation variants                              │    │
│  │     - Score by cognitive weights                                    │    │
│  │     - Filter by storage policy                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  6. PARALLEL VARIANT EXPLORATION (if needed)                         │    │
│  │     - Spin off agents for top variants                              │    │
│  │     - Each agent explores implications                              │    │
│  │     - Reconcile results                                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  7. STORAGE (Qdrant + Graph Cache)                                   │    │
│  │     - Embed triples and nodes                                       │    │
│  │     - Store with synset IDs in payload                              │    │
│  │     - Update graph cache                                            │    │
│  │     - Create cross-layer associations                               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│        │                                                                     │
│        ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  8. CONFLICT DETECTION & BELIEF UPDATE                               │    │
│  │     - Check for conflicts with existing beliefs                     │    │
│  │     - Queue for reconciliation if needed                            │    │
│  │     - Update belief confidence                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Relationship to FR-001-005

| Current FR | Enhancement with Semantic Expansion |
|------------|-------------------------------------|
| FR-001: Shared Working Memory | Observations become expanded semantic frames with synsets |
| FR-002: Parallel Orchestration | Variant exploration uses parallel agents |
| FR-003: Belief Reconciliation | Conflicts detected via semantic triple matching |
| FR-004: Transactive Memory | Expertise includes synset-level topic knowledge |
| FR-005: Metacognitive Reflection | Learns from interpretation accuracy patterns |

---

## 8. Research References

### Word Sense Disambiguation
- [WordNet](https://wordnet.princeton.edu/) - Princeton lexical database
- [BabelNet](https://babelnet.org/about) - Multilingual semantic network with 23M synsets
- [spaCy Lemmatizer](https://spacy.io/api/lemmatizer) - Morphological analysis
- [NLP-progress WSD](http://nlpprogress.com/english/word_sense_disambiguation.html) - SOTA benchmarks

### Knowledge Graph Embeddings
- [TransE, RotatE](https://en.wikipedia.org/wiki/Knowledge_graph_embedding) - Triple embedding methods
- [GraphRAG with Qdrant](https://qdrant.tech/documentation/examples/graphrag-qdrant-neo4j/) - Hybrid approach
- [HybridRAG](https://memgraph.com/blog/why-hybridrag) - Vector + Graph combination

### Commonsense Knowledge
- [ATOMIC/COMET](https://arxiv.org/abs/2010.05953) - Commonsense inference graphs
- [ConceptNet](https://conceptnet.io/) - Semantic network
- [Time-aware COMET (2024)](https://aclanthology.org/2024.lrec-main.1405.pdf) - Temporal reasoning

### Frame Semantics
- [FrameNet](https://framenet.icsi.berkeley.edu/) - Conceptual frames
- [PropBank](https://propbank.github.io/) - Semantic roles
- [AMR](https://amr.isi.edu/) - Abstract meaning representation

---

**End of Design Document**

*This document expands on SEMANTIC_EXPANSION_CONCEPT.md with detailed architectural designs for implementation.*
