# Deep Dive: Cross-Reference Linking at Decomposition Time

**Status:** Research Backlog
**Priority:** High
**Depends On:** Phase 0 (Identifiers), Phase 1 (Decomposition)

---

## Overview

When decomposing text into implicit knowledge, presuppositions and inferences often reference knowledge that should already exist in memory. This document explores how to detect these references and link them to existing memory at decomposition time.

**Core Question:** When we decompose "Doug forgot the meeting again", how do we:
1. Detect that "again" implies prior forgetting episodes exist?
2. Search memory for those prior episodes?
3. Link the new decomposition to the retrieved memories?
4. Use those links to inform branch weighting?

---

## Problem Statement

### Current Gap

The decomposition pipeline extracts presuppositions like:
- "Doug forgot before" (from "again" trigger)
- "A specific meeting exists" (from "the meeting")

But we haven't specified:
1. **When** to search memory (during decomposition? after?)
2. **What** to search for (entities? presupposition text? synsets?)
3. **How** to represent the links (edge types? confidence?)
4. **Whether** links affect storage (store linked or separate?)

### Why This Matters

Without cross-reference linking:
- Presuppositions float unconnected
- "Again" patterns aren't detected across time
- Retrieval misses related episodes
- Branch weighting can't use memory evidence

---

## Research Questions

### Q1: Linking Triggers

What patterns in decomposed knowledge should trigger memory searches?

**Candidates:**
| Pattern | Trigger Text | Search Strategy |
|---------|--------------|-----------------|
| Iterative presupposition | "again", "another", "still" | Find prior instances of same predicate |
| Definite description | "the X" | Find entity X in memory |
| Anaphora | "he", "she", "it" | Find recent entity matching type |
| Temporal reference | "before that", "last time" | Find temporally related episodes |
| Factive presupposition | "knew", "realized" | Find factual basis for knowledge |

**Questions to investigate:**
- Should ALL presuppositions trigger searches, or only specific types?
- What's the cost/benefit of eager vs lazy linking?
- Can we batch searches for efficiency?

### Q2: Search Strategy

How should we search memory for cross-references?

**Options:**

1. **Entity-based search** (like `search_by_entities`)
   - Extract entities from presupposition
   - Search memory for those entities
   - Pro: Fast, precise
   - Con: Misses semantic matches

2. **Semantic search**
   - Embed presupposition text
   - Vector similarity search
   - Pro: Finds related content
   - Con: May be too broad

3. **Synset-filtered search**
   - Search by entity + filter by synset match
   - Pro: Precise sense matching
   - Con: Requires WSD on memory

4. **Predicate-based search**
   - Extract predicate (e.g., "forgot")
   - Find prior instances of same predicate + entity
   - Pro: Finds patterns
   - Con: Needs predicate extraction

**Questions to investigate:**
- Which strategy works best for which trigger type?
- Can we combine strategies?
- How does this integrate with existing `search_by_entities`?

### Q3: Link Representation

How should cross-references be represented in the graph?

**Edge Types to Consider:**
```python
class CrossReferenceType(str, Enum):
    # Temporal links
    PRIOR_INSTANCE = "prior_instance"      # "again" → previous occurrence
    TEMPORAL_CONTEXT = "temporal_context"  # "before that" → referenced event

    # Referential links
    RESOLVES_TO = "resolves_to"            # "he" → Doug
    REFERS_TO = "refers_to"                # "the meeting" → meeting_123

    # Evidential links
    SUPPORTS = "supports"                  # Memory supports this presupposition
    CONTRADICTS = "contradicts"            # Memory contradicts this

    # Pattern links
    INSTANCE_OF_PATTERN = "instance_of"    # This is instance of recurring pattern
```

**Questions to investigate:**
- Should links be bidirectional?
- How to handle multiple potential links (ambiguity)?
- Should links have confidence scores?

### Q4: Integration with Branch Weighting

How do cross-references inform branch weights?

**Hypothesis:** Branches with strong memory support should be weighted higher.

```
Input: "He forgot the meeting again"

Branch A: "He" = Doug (recent mention)
  - Memory search finds: 3 prior "Doug forgot X" episodes
  - Memory support: HIGH
  - Branch weight boost: +0.2

Branch B: "He" = Unknown
  - No memory support
  - Branch weight: unchanged
```

**Questions to investigate:**
- How much should memory support boost confidence?
- What if memory contradicts a branch?
- Should this be evolvable?

### Q5: Storage Strategy

Should linked knowledge be stored together or separately?

**Option A: Store with links**
```
Node: "Doug forgot the meeting again"
  Links:
    - PRIOR_INSTANCE → episode_dec_15
    - PRIOR_INSTANCE → episode_dec_20
    - REFERS_TO → meeting_weekly_sync
```

**Option B: Store separately, link at retrieval**
- Store decomposition without links
- Compute links dynamically at retrieval
- Pro: Simpler storage, always fresh
- Con: Slower retrieval, repeated computation

**Option C: Hybrid**
- Store high-confidence links
- Compute uncertain links at retrieval

**Questions to investigate:**
- What's the storage/retrieval tradeoff?
- Do links go stale as memory changes?
- How to handle link invalidation?

---

## Implementation Considerations

### Integration with `search_by_entities`

The existing `search_by_entities` method (seen in `layered.py`) could be extended:

```python
async def search_for_cross_references(
    self,
    presupposition: Presupposition,
    entities: list[UniversalSemanticIdentifier],
    search_config: CrossRefSearchConfig,
) -> list[CrossReference]:
    """Search for memory cross-references for a presupposition."""

    references = []

    # Strategy 1: Entity-based
    if presupposition.trigger_type in ["DEFINITE_DESC", "ANAPHORA"]:
        entity_results = await self.search_by_entities(
            [e.canonical_name for e in entities],
            limit=search_config.entity_search_limit,
        )
        references.extend(self._process_entity_results(entity_results))

    # Strategy 2: Predicate-based (for iterative)
    if presupposition.trigger_type == "ITERATIVE":
        predicate = extract_predicate(presupposition)
        predicate_results = await self.search_by_predicate_and_entity(
            predicate=predicate,
            entity=entities[0] if entities else None,
            limit=search_config.predicate_search_limit,
        )
        references.extend(self._process_predicate_results(predicate_results))

    return references
```

### Evolvable Configuration

```python
@dataclass
class CrossRefSearchConfig:
    """Evolvable configuration for cross-reference search."""

    # Which triggers to search for
    enabled_triggers: list[str] = field(default_factory=lambda: [
        "ITERATIVE", "DEFINITE_DESC", "ANAPHORA"
    ])

    # Search limits
    entity_search_limit: int = 5
    predicate_search_limit: int = 3

    # Confidence thresholds
    min_link_confidence: float = 0.5

    # Weight adjustments
    memory_support_boost: float = 0.2
    memory_contradiction_penalty: float = 0.3
```

---

## Research Tasks

When we dive into this topic, investigate:

- [ ] Survey how existing KG systems handle cross-referencing
- [ ] Benchmark entity-based vs semantic vs synset-filtered search
- [ ] Design link representation schema
- [ ] Test impact of linking on branch weighting accuracy
- [ ] Determine optimal eager vs lazy linking strategy
- [ ] Design evolvable search configuration
- [ ] Integrate with existing `search_by_entities` pattern

---

## Related Documents

- `DECOMPOSITION_THEORY.md` - Presupposition types and triggers
- `WSD_AND_ENTITY_LINKING.md` - Entity identification
- `PHASE_0_IDENTIFIERS.md` - Identifier system requirements
- `EVOLUTIONARY_TESTING.md` - Making this evolvable

---

## Notes

This deep dive should happen after Phase 1 (basic decomposition) is working, as we need decomposed presuppositions to test linking strategies.

The `search_by_entities` method in the layered memory provider is a good starting point for implementation.

---

**Status:** Research Complete

---

## 6. Research Findings (Added 2025-12-31)

Based on recent research, here are key findings to inform implementation:

### Cross-Document Coreference Resolution

From [2024 research on cross-document coreference](https://arxiv.org/abs/2504.05767):

> "These methods employ dynamic linking mechanisms that associate entities in the knowledge graph with their corresponding textual mentions. By utilizing contextual embeddings along with graph-based inference strategies, they effectively capture the relationships and interactions among entities."

**Key Insight:** Use contextual embeddings + graph-based inference for entity linking across documents/episodes.

### LLM-Based Coreference for Knowledge Graphs

From [LINK-KG framework](https://arxiv.org/html/2510.26486):

> "LINK-KG is a modular framework that integrates a three-stage, LLM-guided coreference resolution pipeline with downstream KG extraction. At the core of this approach is a type-specific Prompt Cache, which consistently tracks and resolves references across document chunks."

**Key Insight:** Maintain a "Prompt Cache" that tracks entities across conversation chunks - similar to our working memory layer.

### Dialogue-Level Coreference (NLPCC 2024)

From [NLPCC 2024 shared task](https://dl.acm.org/doi/10.1007/978-981-97-9443-0_11):

> "Dialogue relation extraction that identifies the relations between argument pairs in dialogue text suffers much from the frequent occurrence of personal pronouns, or entity and speaker coreference."

**Key Insight:** Dialogue/conversation context requires special handling - pronouns are frequent and speaker attribution matters.

### Memory-Augmented Approaches

From [dialogue state tracking research](https://dl.acm.org/doi/abs/10.1016/j.csl.2024.101741):

> "Memory-augmented transformations further contribute to accuracy in knowledge-intensive tasks by efficiently integrating external knowledge."

**Key Insight:** Our 4-layer memory architecture is well-suited for this - use working memory for recent context, episodic for conversation history.

---

## 7. Updated Implementation Recommendations

Based on research, here's the recommended approach:

### 7.1 Three-Stage Linking Pipeline

```python
class CrossReferencePipeline:
    """Three-stage cross-reference linking (inspired by LINK-KG)."""

    async def link(
        self,
        decomposed: DecomposedKnowledge,
        memory: LayeredMemoryProvider,
        config: CrossRefConfig,
    ) -> LinkedKnowledge:
        """Link decomposed knowledge to existing memory."""

        # Stage 1: Entity Resolution
        # Resolve pronouns and definite descriptions to known entities
        entity_links = await self._resolve_entities(
            decomposed.entities,
            decomposed.anaphora,
            memory.working_memory,  # Recent context first
        )

        # Stage 2: Cross-Reference Search
        # Find related episodes and facts in memory
        cross_refs = await self._search_cross_references(
            decomposed.presuppositions,
            entity_links,
            memory,
            config,
        )

        # Stage 3: Link Scoring and Selection
        # Score links by confidence, select best
        scored_links = await self._score_and_select(
            cross_refs,
            decomposed,
            config,
        )

        return LinkedKnowledge(
            decomposed=decomposed,
            entity_links=entity_links,
            cross_references=scored_links,
        )
```

### 7.2 Entity Resolution with Memory Context

```python
async def _resolve_entities(
    self,
    entities: list[UniversalSemanticIdentifier],
    anaphora: list[Anaphora],
    working_memory: WorkingMemory,
) -> dict[str, UniversalSemanticIdentifier]:
    """Resolve anaphora using working memory context."""

    resolutions = {}

    for anaphor in anaphora:
        if anaphor.text.lower() in ["he", "him", "his"]:
            # Find most recent male entity
            candidates = working_memory.get_recent_entities(
                entity_type="PERSON",
                gender="male",
                limit=3,
            )
        elif anaphor.text.lower() in ["she", "her", "hers"]:
            candidates = working_memory.get_recent_entities(
                entity_type="PERSON",
                gender="female",
                limit=3,
            )
        elif anaphor.text.lower() in ["it", "this", "that"]:
            # Find most recent non-person entity
            candidates = working_memory.get_recent_entities(
                entity_type="THING",
                limit=3,
            )
        else:
            candidates = []

        if candidates:
            # Score by recency and context fit
            best = max(candidates, key=lambda c: c.recency_score)
            resolutions[anaphor.text] = best.identifier
            resolutions[anaphor.text + "_confidence"] = best.recency_score

    return resolutions
```

### 7.3 Trigger-Specific Search Strategies

```python
TRIGGER_SEARCH_STRATEGIES = {
    "ITERATIVE": {
        "search_type": "predicate_entity",
        "description": "Find prior instances of same action by same entity",
        "example": "'again' → find prior 'forgot' events by Doug",
    },
    "DEFINITE_DESC": {
        "search_type": "entity_lookup",
        "description": "Find the specific entity referenced",
        "example": "'the meeting' → find meeting entity",
    },
    "ANAPHORA": {
        "search_type": "recency_ranked",
        "description": "Find recent entity matching type",
        "example": "'he' → find recent male person",
    },
    "TEMPORAL": {
        "search_type": "temporal_search",
        "description": "Find events in temporal relation",
        "example": "'before that' → find prior event",
    },
    "FACTIVE": {
        "search_type": "fact_lookup",
        "description": "Find factual basis for knowledge claim",
        "example": "'knew that X' → find X in semantic memory",
    },
}
```

### 7.4 Branch Weighting Integration

```python
async def boost_branch_with_memory(
    branch: WeightedBranch,
    cross_refs: list[CrossReference],
    config: CrossRefConfig,
) -> WeightedBranch:
    """Boost branch confidence based on memory support."""

    support_score = 0.0
    contradiction_score = 0.0

    for ref in cross_refs:
        if ref.relation_type == "SUPPORTS":
            support_score += ref.confidence * config.support_weight
        elif ref.relation_type == "CONTRADICTS":
            contradiction_score += ref.confidence * config.contradiction_weight

    # Net adjustment
    adjustment = support_score - contradiction_score

    # Clamp to reasonable range
    adjustment = max(-0.3, min(0.3, adjustment))

    branch.memory_support_adjustment = adjustment
    branch.base_confidence += adjustment

    return branch
```

---

## 8. Fitness Metrics for Evolution

```python
@dataclass
class CrossRefFitness:
    """Fitness metrics for cross-reference linking."""

    # Accuracy metrics
    entity_resolution_accuracy: float  # Correct anaphora resolution
    link_precision: float              # Links that were correct
    link_recall: float                 # Correct links that were found

    # Efficiency metrics
    search_latency: float              # Average search time
    memory_queries: int                # Number of memory queries

    # Impact metrics
    branch_improvement: float          # Did linking improve branch selection?
    retrieval_improvement: float       # Did links improve downstream retrieval?
```

---

## Research References

- [Cross-Document Contextual Coreference Resolution in KGs (2024)](https://arxiv.org/abs/2504.05767)
- [LINK-KG: LLM-Driven Coreference for KG Construction](https://arxiv.org/html/2510.26486)
- [NLPCC 2024 Dialogue-Level Coreference Task](https://dl.acm.org/doi/10.1007/978-981-97-9443-0_11)
- [End-to-End Dialog Neural Coreference (2024)](https://arxiv.org/html/2504.05824v1)
- [Coreference Resolution in Conversational AI](https://spotintelligence.com/2024/01/17/coreference-resolution-nlp/)
- [ECDG-DST: Dialogue State Tracking with Memory](https://dl.acm.org/doi/abs/10.1016/j.csl.2024.101741)
