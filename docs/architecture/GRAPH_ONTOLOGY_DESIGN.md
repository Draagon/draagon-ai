# Semantic Graph Ontology Design

## Overview

This document defines the ontology design for draagon-ai's semantic knowledge graph,
optimized for LLM context retrieval (GraphRAG patterns).

## The Three-Layer Model

Following established knowledge graph patterns from Wikidata, DBpedia, and RDF/OWL standards,
our graph has three distinct layers connected by specific edge types:

### Layer Interconnection Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            CLASS LAYER                                       │
│                     (Abstract Types / Ontology)                              │
│                                                                              │
│    ┌─────────────┐         ┌─────────────┐         ┌─────────────┐         │
│    │ person.n.01 │◄──SUB───│ mammal.n.01 │───SUB──►│  cat.n.01   │         │
│    │   (CLASS)   │  CLASS  │   (CLASS)   │  CLASS  │   (CLASS)   │         │
│    └──────▲──────┘   OF    └─────────────┘   OF    └──────▲──────┘         │
│           │                                                │                 │
└───────────┼────────────────────────────────────────────────┼─────────────────┘
            │ INSTANCE_OF                                    │ INSTANCE_OF
            │                                                │
┌───────────┼────────────────────────────────────────────────┼─────────────────┐
│           │                 INSTANCE LAYER                 │                 │
│           │           (Specific Individuals)               │                 │
│    ┌──────┴──────┐                              ┌──────────┴───────┐        │
│    │    Doug     │                              │  Doug's cats     │        │
│    │ (INSTANCE)  │───────── OWNS ──────────────►│  (COLLECTION)    │        │
│    │             │                              │   count: 3       │        │
│    └─────────────┘                              └────────┬─────────┘        │
│                                                    MEMBER_OF                 │
│                                          ┌───────────┼───────────┐          │
│                                          ▼           ▼           ▼          │
│                                   ┌──────────┐┌──────────┐┌──────────┐      │
│                                   │ Whiskers ││  [cat2]  ││  [cat3]  │      │
│                                   │(INSTANCE)││(INSTANCE)││(INSTANCE)│      │
│                                   │name:known││anonymous ││anonymous │      │
│                                   └──────────┘└──────────┘└──────────┘      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Cross-Layer Edge Types

| Edge Type | From Layer | To Layer | Purpose | Example |
|-----------|------------|----------|---------|---------|
| `INSTANCE_OF` | INSTANCE | CLASS | Ontological typing | Whiskers → cat.n.01 |
| `HAS_SENSE` | INSTANCE | CLASS | Linguistic WSD | "cat" (word) → cat.n.01 |
| `SUBCLASS_OF` | CLASS | CLASS | Taxonomic hierarchy | cat.n.01 → mammal.n.01 |
| `MEMBER_OF` | INSTANCE | COLLECTION | Group membership | Whiskers → "Doug's cats" |
| `OWNS`, `KNOWS` | INSTANCE | INSTANCE | Semantic relations | Doug → Whiskers |

### Distinction: INSTANCE_OF vs HAS_SENSE

Both edges connect INSTANCE nodes to CLASS nodes, but serve different purposes:

- **INSTANCE_OF**: Ontological - "Whiskers IS A cat" (type hierarchy)
- **HAS_SENSE**: Linguistic - "The word 'cat' in this text means cat.n.01" (WSD)

In code, INSTANCE_OF is used for EntityType.INSTANCE entities (specific individuals),
while HAS_SENSE is used for EntityType.CLASS/GENERIC (abstract references).

## Detailed Layer Specifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  LAYER 1: SCHEMA (Classes/Concepts)                                         │
│  ═══════════════════════════════════                                         │
│  Abstract types representing CATEGORIES of things.                          │
│  These are WordNet synsets, taxonomic classes, and ontological concepts.    │
│                                                                              │
│  Examples:                                                                   │
│    • cat.n.01 (FelisCatus - the concept of cats)                            │
│    • person.n.01 (the concept of a human person)                            │
│    • meeting.n.01 (the concept of a meeting)                                │
│                                                                              │
│  Relationships:                                                              │
│    • SUBCLASS_OF (cat.n.01 SUBCLASS_OF mammal.n.01)                         │
│    • EQUIVALENT_TO (cat.n.01 EQUIVALENT_TO feline.n.01)                     │
│    • RELATED_CONCEPT (pet.n.01 RELATED_CONCEPT owner.n.01)                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 2: INSTANCES (Specific Entities)                                     │
│  ══════════════════════════════════════                                     │
│  Specific, identifiable things that exist in the world.                     │
│  These may be named (Doug, Whiskers) or anonymous ([Doug's cat]).           │
│                                                                              │
│  Examples:                                                                   │
│    • Doug (a specific person)                                               │
│    • Whiskers (Doug's specific cat)                                         │
│    • [anonymous cat] (an unnamed instance we know exists)                   │
│                                                                              │
│  Relationships:                                                              │
│    • INSTANCE_OF (Whiskers INSTANCE_OF cat.n.01)                            │
│    • Doug INSTANCE_OF person.n.01                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  LAYER 3: RELATIONS (Instance-to-Instance Links)                            │
│  ═══════════════════════════════════════════════                            │
│  Semantic relationships between instances.                                  │
│                                                                              │
│  Examples:                                                                   │
│    • Doug OWNS Whiskers                                                     │
│    • Doug LIVES_IN Portland                                                 │
│    • Meeting_123 ATTENDED_BY Doug                                           │
│                                                                              │
│  Properties on edges:                                                       │
│    • confidence: 0.0-1.0                                                    │
│    • valid_from/valid_to: bi-temporal tracking                              │
│    • source_decomposition_id: provenance                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## NodeType Clarification

```python
class NodeType(str, Enum):
    """
    INSTANCE: A specific, identifiable individual thing.
              - Named: "Doug", "Whiskers", "Apple Inc."
              - Anonymous: "[Doug's cat]", "[the meeting]"
              These are rdf:type targets, not sources.

    CLASS: An abstract category/type/concept.
           - WordNet synsets: cat.n.01, person.n.01
           - Taxonomic: Mammal, Animal, Pet
           These are rdfs:subClassOf participants.

    EVENT: A specific occurrence at a point/period in time.
           - "Doug's birthday party on March 15"
           - "The meeting at 3pm"
           Events are instances, but temporal in nature.

    ATTRIBUTE: A property value (reified for graph storage).
           - Used when we need to attach metadata to property values
           - Example: "orange" color with certainty 0.8

    COLLECTION: A group/set of instances.
           - "Doug's cats" (the set, not individual cats)
           - "Team Alpha members"
    """
    INSTANCE = "instance"   # Specific individual (was ENTITY)
    CLASS = "class"         # Abstract type/concept (was CONCEPT)
    EVENT = "event"         # Temporal instance
    ATTRIBUTE = "attribute" # Reified property value
    COLLECTION = "collection"  # Group of instances
```

## Example: "Doug has 3 cats"

### What We Know
- Doug exists (a person)
- Doug has cats (multiple)
- The quantity is 3
- We may not know the cats' names

### Graph Representation

```
                     ┌──────────────────────────────────┐
                     │          CLASS LAYER             │
                     │                                  │
                     │  ┌───────────┐    ┌───────────┐ │
                     │  │ cat.n.01  │    │person.n.01│ │
                     │  │ (CLASS)   │    │ (CLASS)   │ │
                     │  └─────▲─────┘    └─────▲─────┘ │
                     │        │                │       │
                     │   SUBCLASS_OF      SUBCLASS_OF  │
                     │        │                │       │
                     │  ┌───────────┐    ┌───────────┐ │
                     │  │mammal.n.01│    │living.n.01│ │
                     │  └───────────┘    └───────────┘ │
                     └────────│────────────────│───────┘
                              │                │
                         INSTANCE_OF      INSTANCE_OF
                              │                │
┌─────────────────────────────▼────────────────▼─────────────────────────────┐
│                         INSTANCE LAYER                                      │
│                                                                             │
│  ┌─────────────────┐                      ┌─────────────────┐              │
│  │  Doug           │                      │ Doug's cats     │              │
│  │  (INSTANCE)     │────── OWNS ─────────►│ (COLLECTION)    │              │
│  │                 │                      │ count: 3        │              │
│  │  name: "Doug"   │                      └────────┬────────┘              │
│  └─────────────────┘                               │                        │
│                                              MEMBER_OF                      │
│                                     ┌──────────┬───┴───┬──────────┐        │
│                                     ▼          ▼       ▼          │        │
│                              ┌──────────┐┌──────────┐┌──────────┐ │        │
│                              │ cat_1    ││ cat_2    ││ cat_3    │ │        │
│                              │(INSTANCE)││(INSTANCE)││(INSTANCE)│ │        │
│                              │          ││          ││          │ │        │
│                              │anonymous ││anonymous ││anonymous │ │        │
│                              └────┬─────┘└────┬─────┘└────┬─────┘ │        │
│                                   │           │           │       │        │
│                                   └───────────┼───────────┘       │        │
│                                         INSTANCE_OF               │        │
│                                               │                   │        │
└───────────────────────────────────────────────┼───────────────────┘        │
                                                ▼
                                          ┌───────────┐
                                          │ cat.n.01  │
                                          │ (CLASS)   │
                                          └───────────┘
```

### Later: "One of Doug's cats is named Whiskers"

We update the anonymous cat_1 to have a name:

```python
# Find anonymous cat in Doug's collection
# Update with name property
cat_1.canonical_name = "Whiskers"
cat_1.properties["name"] = "Whiskers"
```

### Even Later: "Whiskers is an orange tabby"

```python
# Add properties to the existing Whiskers node
whiskers.properties["color"] = "orange"
whiskers.properties["pattern"] = "tabby"
whiskers.properties["breed"] = "domestic shorthair"  # implied by tabby
```

## Key Design Decisions

### DD-3.1: Instances vs Classes Are Distinct Node Types

**Rationale**: Following Wikidata/RDF patterns, we separate:
- `CLASS` nodes: WordNet synsets, taxonomic types (deduplicated globally)
- `INSTANCE` nodes: Specific entities (can be anonymous)

**LLM Context Benefit**: When retrieving context, we can:
1. Get the instance and its specific properties
2. Optionally expand to class for type information
3. Control context size by depth of class hierarchy

### DD-3.2: Anonymous Instances Are First-Class Citizens

**Rationale**: When we learn "Doug has cats" without names, we still create instance nodes.

**Implementation**:
```python
# Create anonymous instance
cat_instance = GraphNode(
    node_type=NodeType.INSTANCE,
    canonical_name="[Doug's cat #1]",  # Placeholder name
    properties={"is_anonymous": True, "owner_ref": "doug_node_id"}
)
```

**LLM Context Benefit**: Preserves cardinality information even without specifics.

### DD-3.3: Collections Capture Cardinality

**Rationale**: "Doug has 3 cats" should store the count somewhere reliable.

**Implementation**:
```python
# Create collection node
collection = GraphNode(
    node_type=NodeType.COLLECTION,
    canonical_name="Doug's cats",
    properties={"count": 3, "count_confidence": 0.95}
)

# Doug OWNS the collection
graph.create_edge(doug.node_id, collection.node_id, "OWNS")

# Individual cats are MEMBER_OF collection
for cat in cats:
    graph.create_edge(cat.node_id, collection.node_id, "MEMBER_OF")
```

**LLM Context Benefit**: Cardinality queries ("How many cats does Doug have?") are instant.

### DD-3.4: Synset Deduplication Via CLASS Nodes

**Rationale**: "cat.n.01" should exist exactly once, with all instances linking to it.

**Implementation**:
```python
def find_or_create_class(synset_id: str) -> GraphNode:
    existing = graph.find_nodes_by_synset(synset_id, node_type=NodeType.CLASS)
    if existing:
        return existing[0]
    return graph.create_node(
        node_type=NodeType.CLASS,
        canonical_name=synset_id,
        synset_id=synset_id,
    )
```

**LLM Context Benefit**: Single source of truth for type information; no redundant class definitions.

## Edge Types for Three-Layer Model

```python
class SemanticEdgeType(str, Enum):
    # === Instance-to-Class Relations ===
    INSTANCE_OF = "instance_of"     # Whiskers INSTANCE_OF cat.n.01

    # === Class-to-Class Relations ===
    SUBCLASS_OF = "subclass_of"     # cat.n.01 SUBCLASS_OF mammal.n.01
    EQUIVALENT_CLASS = "equivalent_class"  # For synonymous classes
    RELATED_CLASS = "related_class"  # Looser conceptual relation

    # === Instance-to-Instance Relations ===
    OWNS = "owns"
    KNOWS = "knows"
    LIVES_IN = "lives_in"
    WORKS_AT = "works_at"
    MEMBER_OF = "member_of"
    PART_OF = "part_of"

    # === Semantic Role Relations (from Phase 1) ===
    ARG0 = "arg0"  # Agent
    ARG1 = "arg1"  # Patient/Theme
    # ... etc (from PropBank)

    # === Commonsense Relations (ATOMIC) ===
    X_INTENT = "xIntent"
    X_EFFECT = "xEffect"
    # ... etc
```

## LLM Context Retrieval Strategy

### Entity-Centric Subgraph Extraction

When the user asks about Doug:

```python
def get_context_for_entity(entity_name: str, depth: int = 2) -> Subgraph:
    """
    Depth 0: Just the entity node
    Depth 1: Entity + direct relations + related instances
    Depth 2: + class information + collection details
    Depth 3: + class hierarchy (usually too much)
    """
    entity = graph.find_node(entity_name)

    subgraph = Subgraph()
    subgraph.add_node(entity)

    if depth >= 1:
        # Get all instance-to-instance relations
        for edge in graph.get_edges(entity.node_id):
            subgraph.add_edge(edge)
            subgraph.add_node(graph.get_node(edge.target_node_id))

    if depth >= 2:
        # Get INSTANCE_OF edges to classes
        for node in subgraph.nodes:
            if node.node_type == NodeType.INSTANCE:
                class_edges = graph.get_edges(
                    node.node_id,
                    relation_type="instance_of"
                )
                for edge in class_edges:
                    subgraph.add_edge(edge)
                    subgraph.add_node(graph.get_node(edge.target_node_id))

    return subgraph
```

### Context Serialization for LLM

```python
def serialize_for_llm(subgraph: Subgraph) -> str:
    """
    Compact, semantic representation for LLM context.

    Output example:
    <knowledge>
    <entity name="Doug" type="person">
      <owns target="Whiskers" type="cat" confidence="0.95"/>
      <owns target="[cat #2]" type="cat" confidence="0.95"/>
      <lives_in target="Portland" type="city" confidence="0.88"/>
    </entity>
    <entity name="Whiskers" type="cat">
      <color>orange</color>
      <pattern>tabby</pattern>
    </entity>
    </knowledge>
    """
```

## Migration from Current NodeType

| Old NodeType | New NodeType | Notes |
|--------------|--------------|-------|
| ENTITY | INSTANCE | Specific individual things |
| CONCEPT | CLASS | WordNet synsets, taxonomic types |
| EVENT | EVENT | No change (events are temporal instances) |
| ATTRIBUTE | ATTRIBUTE | No change |
| COLLECTION | COLLECTION | No change |

## References

- [Wikidata Item Classification](https://www.wikidata.org/wiki/Wikidata:Item_classification)
- [RDF Schema Specification](https://www.w3.org/TR/rdf-schema/)
- [OWL Web Ontology Language](https://www.w3.org/TR/owl2-overview/)
- [SubgraphRAG Framework](https://www.emergentmind.com/topics/subgraphrag-framework)
- [GraphRAG Knowledge Graph Patterns](https://pub.towardsai.net/graphrag-explained-building-knowledge-grounded-llm-systems-with-neo4j-and-langchain-017a1820763e)
