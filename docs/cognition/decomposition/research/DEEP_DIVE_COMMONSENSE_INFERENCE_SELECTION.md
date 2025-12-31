# Deep Dive: Commonsense Inference Selection Strategy

**Status:** Active Research
**Priority:** High
**Phase:** Informs Phase 1 (Decomposition)

---

## Executive Summary

ATOMIC 2020 defines 23 commonsense relation types. Extracting ALL of them for every statement would be:
- **Expensive** - Multiple LLM/COMET calls per statement
- **Noisy** - Many inferences won't be relevant for retrieval
- **Storage-heavy** - 23x more nodes per statement

This document explores **which inferences to extract, when, and how** to maximize retrieval value while minimizing cost.

**Key Finding:** Not all relation types are equally useful. Research shows 5 core relations (xIntent, xNeed, xWant, xEffect, xReact) are most commonly used and provide the highest value for understanding events.

---

## Table of Contents

1. [ATOMIC Relation Taxonomy](#1-atomic-relation-taxonomy)
2. [Relevance Analysis by Relation Type](#2-relevance-analysis-by-relation-type)
3. [Selection Strategies](#3-selection-strategies)
4. [Quality and Hallucination Concerns](#4-quality-and-hallucination-concerns)
5. [Implementation Recommendations](#5-implementation-recommendations)
6. [Evolvable Configuration](#6-evolvable-configuration)

---

## 1. ATOMIC Relation Taxonomy

### ATOMIC 2020: 23 Relation Types

Based on [ATOMIC 2020](https://arxiv.org/pdf/2010.05953), relations are organized into three categories:

#### Social Interaction Relations (9 types)

| Relation | Definition | Example: "Doug forgot the meeting" |
|----------|------------|-----------------------------------|
| **xIntent** | Why X did this | "was distracted", "didn't prioritize" |
| **xNeed** | What X needed beforehand | "to know about the meeting", "a calendar" |
| **xWant** | What X wants after | "to apologize", "to reschedule" |
| **xEffect** | What happens to X | "misses the meeting", "gets in trouble" |
| **xReact** | How X feels | "embarrassed", "guilty", "stressed" |
| **xAttr** | X's attributes | "forgetful", "disorganized", "busy" |
| **oWant** | What others want | "an explanation", "to reschedule" |
| **oEffect** | What happens to others | "meeting delayed", "wait for Doug" |
| **oReact** | How others feel | "frustrated", "annoyed", "worried" |

#### Physical Entity Relations (7 types)

| Relation | Definition | Example: "coffee" |
|----------|------------|-------------------|
| **ObjectUse** | What it's used for | "drinking", "staying awake" |
| **AtLocation** | Where typically found | "kitchen", "coffee shop" |
| **MadeOf** | Composition | "water", "coffee beans" |
| **HasProperty** | Properties | "hot", "caffeinated", "bitter" |
| **CapableOf** | What it can do | "stain clothes", "give energy" |
| **Desires** | What it "wants" | N/A for objects |
| **NotDesires** | What it avoids | N/A for objects |

#### Event-Centered Relations (7 types)

| Relation | Definition | Example: "forgetting the meeting" |
|----------|------------|----------------------------------|
| **Causes** | What this causes | "missing the meeting", "delays" |
| **IsBefore** | What typically precedes | "being busy", "not checking calendar" |
| **IsAfter** | What typically follows | "apologizing", "rescheduling" |
| **HasSubEvent** | Component events | "realizing too late", "rushing" |
| **HinderedBy** | What prevents this | "setting reminders", "good habits" |
| **xReason** | General reason | "distraction", "poor memory" |
| **isFilledBy** | Role fillers | N/A |

---

## 2. Relevance Analysis by Relation Type

### Tier 1: High Value (Extract Always)

These relations provide the most useful context for understanding and retrieval:

| Relation | Why High Value | Retrieval Benefit |
|----------|----------------|-------------------|
| **xIntent** | Explains motivation | "Why did Doug do X?" queries |
| **xEffect** | Consequences matter | "What happened after X?" queries |
| **xReact** | Emotional context | Sentiment-aware retrieval |
| **xAttr** | Character traits | Pattern detection across episodes |

**Research Support:** Studies show these 4-5 relations are used in most COMET applications for social understanding.

### Tier 2: Medium Value (Extract Conditionally)

Extract based on context or query type:

| Relation | When to Extract | Retrieval Benefit |
|----------|-----------------|-------------------|
| **xNeed** | For procedural/how-to content | "How to do X?" queries |
| **xWant** | For goal-oriented content | "What does X want?" queries |
| **oReact** | When others are mentioned | Multi-person understanding |
| **Causes/IsAfter** | For temporal reasoning | "What led to X?" queries |

### Tier 3: Low Value (Rarely Extract)

These provide marginal benefit for most use cases:

| Relation | Why Low Value | When Maybe Useful |
|----------|---------------|-------------------|
| **oWant/oEffect** | Often redundant with xEffect | Multi-agent scenarios |
| **ObjectUse/AtLocation** | LLM already knows this | Domain-specific apps |
| **MadeOf/HasProperty** | Encyclopedic, not inferential | Product knowledge bases |
| **Desires/NotDesires** | Rarely applicable | Personification contexts |
| **HinderedBy** | Counterfactual, speculative | Planning applications |
| **isFilledBy** | Technical, role-specific | Slot-filling tasks |

---

## 3. Selection Strategies

### Strategy A: Static Tier-Based Selection

Always extract Tier 1, optionally extract Tier 2, skip Tier 3.

```python
ALWAYS_EXTRACT = ["xIntent", "xEffect", "xReact", "xAttr"]
CONDITIONAL_EXTRACT = ["xNeed", "xWant", "oReact", "Causes"]
NEVER_EXTRACT = ["ObjectUse", "MadeOf", "HasProperty", "Desires", "NotDesires"]

async def select_relations_static(
    event: str,
    context: DecompositionContext,
) -> list[str]:
    """Static tier-based selection."""
    relations = ALWAYS_EXTRACT.copy()

    if context.is_procedural:
        relations.append("xNeed")
    if context.has_multiple_agents:
        relations.extend(["oReact", "oEffect"])
    if context.needs_temporal:
        relations.extend(["Causes", "IsAfter"])

    return relations
```

**Pros:** Simple, predictable, fast
**Cons:** May miss relevant inferences in edge cases

### Strategy B: Query-Driven Selection

Select relations based on anticipated query types:

```python
QUERY_TYPE_TO_RELATIONS = {
    "why": ["xIntent", "xReason", "xNeed"],
    "what_happened": ["xEffect", "oEffect", "Causes"],
    "how_feel": ["xReact", "oReact"],
    "what_next": ["xWant", "oWant", "IsAfter"],
    "who_is": ["xAttr"],
    "how_to": ["xNeed", "HasSubEvent"],
}

async def select_relations_query_driven(
    event: str,
    anticipated_queries: list[str],
) -> list[str]:
    """Select based on anticipated query types."""
    relations = set()
    for query_type in anticipated_queries:
        relations.update(QUERY_TYPE_TO_RELATIONS.get(query_type, []))
    return list(relations)
```

**Pros:** Targeted, efficient
**Cons:** Requires anticipating queries, may miss unanticipated needs

### Strategy C: LLM-Guided Selection

Ask LLM which relations are relevant for this specific event:

```python
async def select_relations_llm_guided(
    event: str,
    all_relations: list[str],
    llm: LLMProvider,
) -> list[str]:
    """Use LLM to select relevant relations."""

    prompt = f"""For this event, which commonsense relations would provide useful context?

Event: "{event}"

Available relations:
- xIntent: Why the person did this
- xEffect: What happens to them
- xReact: How they feel
- xAttr: What this says about them
- xNeed: What they needed beforehand
- xWant: What they want afterward
- oReact: How others feel
- Causes: What this leads to

Select ONLY relations that provide NON-OBVIOUS, USEFUL context.
Skip relations where the inference would be trivial or already obvious.

Respond in XML:
<relevant_relations>
    <relation>xIntent</relation>
    <relation>xEffect</relation>
    ...
</relevant_relations>
<reasoning>Brief explanation</reasoning>
"""

    response = await llm.chat([{"role": "user", "content": prompt}])
    return parse_relations(response)
```

**Pros:** Context-sensitive, can handle edge cases
**Cons:** Expensive (LLM call per event), variable quality

### Strategy D: Hybrid with Confidence Gating

Extract all Tier 1, use LLM to filter Tier 2 based on confidence:

```python
async def select_relations_hybrid(
    event: str,
    config: InferenceSelectionConfig,
    llm: LLMProvider,
) -> list[str]:
    """Hybrid: always extract Tier 1, LLM-filter Tier 2."""

    # Always include Tier 1
    relations = config.tier1_relations.copy()

    # For Tier 2, ask LLM if relevant
    tier2_check = await llm.chat([{
        "role": "user",
        "content": f"""For event "{event}", which of these are worth extracting?
        {config.tier2_relations}
        Only select if the inference would be non-obvious and useful.
        Respond with just the relation names, comma-separated."""
    }])

    relevant_tier2 = parse_comma_list(tier2_check)
    relations.extend([r for r in relevant_tier2 if r in config.tier2_relations])

    return relations
```

**Recommended approach** - balances coverage with efficiency.

---

## 4. Quality and Hallucination Concerns

### COMET Accuracy

From research:
- **ATOMIC (original):** 77.5% precision at top-1
- **ConceptNet:** 91.7% precision at top-1
- **ATOMIC 2020:** 91.3% human acceptance rate

This means ~8-23% of generated inferences may be incorrect or nonsensical.

### Mitigation Strategies

#### 1. Confidence Thresholding

COMET/LLM outputs should include confidence scores:

```python
@dataclass
class CommonsenseInference:
    relation: str
    content: str
    confidence: float
    source: str  # "COMET", "LLM", "ConceptNet"

# Filter low-confidence inferences
MIN_CONFIDENCE = 0.6
filtered = [i for i in inferences if i.confidence >= MIN_CONFIDENCE]
```

#### 2. Semantic Deduplication

COMET often generates near-duplicates:

```python
def deduplicate_inferences(
    inferences: list[CommonsenseInference],
    similarity_threshold: float = 0.85,
) -> list[CommonsenseInference]:
    """Remove semantically similar inferences."""
    unique = []
    for inf in inferences:
        if not any(
            semantic_similarity(inf.content, u.content) > similarity_threshold
            for u in unique
        ):
            unique.append(inf)
    return unique
```

#### 3. Plausibility Filtering (Amazon's COSMO approach)

From [Amazon's COSMO framework](https://www.amazon.science/blog/building-commonsense-knowledge-graphs-to-aid-product-recommendation):

> "COSMO involves a recursive procedure where an LLM generates hypotheses about commonsense implications, then applies various heuristics to winnow them down."

Filtering heuristics:
- Too similar to input → reject
- Too generic ("something happens") → reject
- Contradicts known facts → reject
- Low specificity → reduce confidence

```python
async def filter_plausibility(
    event: str,
    inferences: list[CommonsenseInference],
) -> list[CommonsenseInference]:
    """Filter implausible inferences."""
    filtered = []
    for inf in inferences:
        # Too similar to input?
        if semantic_similarity(event, inf.content) > 0.8:
            continue
        # Too generic?
        if inf.content in GENERIC_PHRASES:
            continue
        # Passes filters
        filtered.append(inf)
    return filtered
```

#### 4. Source Preference Ordering

Different sources have different reliability:

```python
SOURCE_RELIABILITY = {
    "human_annotated": 1.0,
    "ConceptNet": 0.9,
    "ATOMIC_2020": 0.85,
    "COMET": 0.75,
    "LLM_generated": 0.70,
}

def adjust_confidence_by_source(inference: CommonsenseInference) -> float:
    """Adjust confidence based on source reliability."""
    source_factor = SOURCE_RELIABILITY.get(inference.source, 0.5)
    return inference.confidence * source_factor
```

---

## 5. Implementation Recommendations

### Recommended Default Configuration

Based on the research, here's the recommended starting point:

```python
@dataclass
class InferenceSelectionConfig:
    """Configuration for commonsense inference selection."""

    # Tier 1: Always extract
    tier1_relations: list[str] = field(default_factory=lambda: [
        "xIntent",   # Why they did it
        "xEffect",   # What happens to them
        "xReact",    # How they feel
        "xAttr",     # What this says about them
    ])

    # Tier 2: Extract conditionally
    tier2_relations: list[str] = field(default_factory=lambda: [
        "xNeed",     # Prerequisites
        "xWant",     # Desires after
        "oReact",    # Others' feelings
    ])

    # Tier 3: Skip by default
    tier3_relations: list[str] = field(default_factory=lambda: [
        "oWant", "oEffect", "ObjectUse", "AtLocation",
        "MadeOf", "HasProperty", "Desires", "NotDesires",
        "HinderedBy", "isFilledBy",
    ])

    # Selection strategy
    strategy: str = "hybrid"  # "static", "query_driven", "llm_guided", "hybrid"

    # Quality filters
    min_confidence: float = 0.6
    similarity_threshold: float = 0.85
    max_inferences_per_relation: int = 3
    max_total_inferences: int = 10

    # Source preferences
    prefer_comet_over_llm: bool = True
    use_conceptnet_for_entities: bool = True
```

### Extraction Pipeline

```python
async def extract_commonsense_inferences(
    event: str,
    config: InferenceSelectionConfig,
    comet_model: COMETModel | None,
    llm: LLMProvider,
) -> list[CommonsenseInference]:
    """Extract relevant commonsense inferences for an event."""

    # Step 1: Select relevant relations
    if config.strategy == "static":
        relations = config.tier1_relations + config.tier2_relations
    elif config.strategy == "hybrid":
        relations = await select_relations_hybrid(event, config, llm)
    else:
        relations = await select_relations_llm_guided(event, ALL_RELATIONS, llm)

    # Step 2: Generate inferences
    inferences = []
    for relation in relations:
        if comet_model and config.prefer_comet_over_llm:
            # Use COMET (faster, cheaper)
            results = comet_model.generate(event, relation, num_beams=5)
            for r in results[:config.max_inferences_per_relation]:
                inferences.append(CommonsenseInference(
                    relation=relation,
                    content=r.text,
                    confidence=r.score,
                    source="COMET",
                ))
        else:
            # Use LLM
            result = await generate_inference_with_llm(event, relation, llm)
            inferences.append(result)

    # Step 3: Filter for quality
    inferences = [i for i in inferences if i.confidence >= config.min_confidence]
    inferences = deduplicate_inferences(inferences, config.similarity_threshold)
    inferences = await filter_plausibility(event, inferences)

    # Step 4: Limit total
    inferences = sorted(inferences, key=lambda i: i.confidence, reverse=True)
    return inferences[:config.max_total_inferences]
```

### Cost Analysis

| Approach | Inferences per Event | LLM Calls | Relative Cost |
|----------|---------------------|-----------|---------------|
| All 23 relations | ~50+ | 23+ | Very High |
| Tier 1 only (4) | ~12 | 4 | Low |
| Tier 1+2 (7) | ~20 | 7 | Medium |
| Hybrid (4-7) | ~15 | 5-8 | Medium |
| COMET (Tier 1) | ~12 | 0 | Very Low |

**Recommendation:** Use COMET for Tier 1 when available, LLM for Tier 2 selection.

---

## 6. Evolvable Configuration

### Parameters to Evolve

```python
@dataclass
class EvolvableInferenceConfig(InferenceSelectionConfig):
    """Inference config with evolution tracking."""

    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    generation: int = 0

    # Evolvable parameters
    tier1_relations: list[str]  # Which relations in Tier 1
    tier2_relations: list[str]  # Which relations in Tier 2
    min_confidence: float       # 0.3 - 0.9
    max_total_inferences: int   # 5 - 20

    # Weights for relation selection
    relation_weights: dict[str, float] = field(default_factory=lambda: {
        "xIntent": 1.0,
        "xEffect": 1.0,
        "xReact": 0.9,
        "xAttr": 0.8,
        "xNeed": 0.6,
        "xWant": 0.5,
        "oReact": 0.4,
    })
```

### Fitness Metrics

1. **Retrieval Contribution** - Do these inferences help retrieval?
   - Measure: Queries that hit inference nodes / total queries

2. **Noise Ratio** - Are we storing useless inferences?
   - Measure: Inferences never retrieved / total stored

3. **Coverage** - Are we missing important inferences?
   - Measure: Queries that should have hit inferences but didn't

4. **Cost** - Are we being efficient?
   - Measure: LLM calls per event, tokens used

### Evolution Strategy

```python
async def evaluate_inference_config(
    config: EvolvableInferenceConfig,
    test_events: list[str],
    test_queries: list[str],
    llm: LLMProvider,
) -> dict[str, float]:
    """Evaluate inference configuration fitness."""

    # Extract inferences for test events
    all_inferences = []
    for event in test_events:
        infs = await extract_commonsense_inferences(event, config, None, llm)
        all_inferences.extend(infs)

    # Measure retrieval contribution
    hit_count = 0
    for query in test_queries:
        results = await search_inferences(query, all_inferences)
        if results:
            hit_count += 1

    retrieval_contribution = hit_count / len(test_queries)

    # Measure noise (inferences never useful)
    useful_inferences = set()
    for query in test_queries:
        results = await search_inferences(query, all_inferences)
        useful_inferences.update(r.id for r in results)

    noise_ratio = 1 - (len(useful_inferences) / len(all_inferences))

    # Cost (LLM calls)
    cost = len(config.tier1_relations) + len(config.tier2_relations)

    return {
        "retrieval_contribution": retrieval_contribution,
        "noise_ratio": noise_ratio,
        "efficiency": 1.0 / cost,
    }
```

---

## Summary

### Key Decisions

1. **Default to Tier 1 (4 relations):** xIntent, xEffect, xReact, xAttr
2. **Conditionally add Tier 2:** Based on context or LLM guidance
3. **Skip Tier 3:** Rarely useful, high noise
4. **Use COMET when available:** Faster, cheaper than LLM
5. **Filter aggressively:** Confidence > 0.6, deduplicate, plausibility check
6. **Limit total:** Max 10 inferences per event
7. **Make it evolvable:** All thresholds and relation lists

### Next Steps

- [ ] Implement `InferenceSelectionConfig` with defaults
- [ ] Integrate COMET model (or use LLM fallback)
- [ ] Build quality filtering pipeline
- [ ] Create fitness metrics for evolution
- [ ] Test on sample events from draagon-ai use cases

---

## Research References

- [ATOMIC 2020 Paper](https://arxiv.org/pdf/2010.05953) - Relation taxonomy and evaluation
- [COMET GitHub](https://github.com/atcbosselut/comet-commonsense) - Implementation
- [Amazon COSMO](https://www.amazon.science/blog/building-commonsense-knowledge-graphs-to-aid-product-recommendation) - Plausibility filtering
- [Time-aware COMET (2024)](https://aclanthology.org/2024.lrec-main.1405/) - Temporal extensions
- [Task-Aware RAG](https://towardsdatascience.com/task-aware-rag-strategies-for-when-sentence-similarity-fails-54c44690fee3/) - Query-driven selection

---

**Status:** Research complete, ready for implementation in Phase 1
