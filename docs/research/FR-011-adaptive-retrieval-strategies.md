# FR-011: Adaptive Retrieval Strategy Learning

**Status:** Draft
**Created:** 2025-01-02
**Author:** Claude + Doug

---

## 1. Executive Summary

This document specifies an adaptive retrieval system that learns which query expansion and retrieval strategies work best for different situations. Rather than using a fixed pipeline, the system runs multiple strategies in parallel and uses reinforcement learning (multi-armed contextual bandits) to learn which strategies to prioritize based on query characteristics and historical outcomes.

### Key Innovation

Instead of picking ONE strategy, the system:
1. **Runs multiple strategies in parallel** (initially)
2. **Observes which strategies contribute to good outcomes**
3. **Learns to skip strategies that don't help** for specific query types
4. **Adapts over time** as the knowledge base and query patterns evolve

---

## 2. Problem Statement

### Current Limitations

Query expansion techniques (HyDE, Query2Doc, RAG-Fusion) each have strengths and weaknesses:

| Technique | Strengths | Weaknesses |
|-----------|-----------|------------|
| **Raw Query** | Fast, no hallucination risk | Short queries miss semantic matches |
| **HyDE** | Bridges query-document gap | Can hallucinate, loses original query |
| **Query2Doc** | Keeps original + expansion | Still can hallucinate |
| **Phase0/1 + Graph** | Grounded in real entities | Requires graph context, slower |

**No single strategy is best for all queries.**

### Research Support

- **[Adaptive-RAG](https://arxiv.org/abs/2403.14403)** (NAACL 2024): Uses classifier to select strategy based on query complexity
- **[MBA-RAG](https://arxiv.org/abs/2412.01572)** (2024): Multi-armed bandit for strategy selection, balances accuracy vs efficiency
- **[GGI-MO-MAB](https://arxiv.org/html/2412.07618v2)** (2024): Multi-objective bandit for knowledge graph RAG

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ADAPTIVE RETRIEVAL ORCHESTRATOR                      │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      QUERY ANALYSIS (Phase 0)                         │  │
│  │  Extract: entities, complexity, domain, ambiguity signals            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    STRATEGY SELECTOR (Contextual Bandit)              │  │
│  │                                                                        │  │
│  │  Context Features:                    Strategy Arms:                   │  │
│  │  - Query length                       - Arm A: Raw Query               │  │
│  │  - Entity count                       - Arm B: HyDE Expansion          │  │
│  │  - Ambiguity score                    - Arm C: Query2Doc               │  │
│  │  - Domain embedding                   - Arm D: Phase0/1 + Graph        │  │
│  │  - Historical query similarity        - Arm E: Multi-Query Fusion      │  │
│  │                                                                        │  │
│  │  Output: Probability distribution over strategies                      │  │
│  │  (Initially uniform, learns from outcomes)                            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    PARALLEL STRATEGY EXECUTION                        │  │
│  │                                                                        │  │
│  │  Strategy A ─────┐                                                    │  │
│  │  Strategy B ─────┼──► Run if P(arm) > threshold OR exploration mode   │  │
│  │  Strategy C ─────┤                                                    │  │
│  │  Strategy D ─────┘                                                    │  │
│  │                                                                        │  │
│  │  Early termination: Skip if bandit has high confidence in other arms  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    RESULT FUSION (Weighted RRF)                       │  │
│  │                                                                        │  │
│  │  Merge results from all executed strategies                           │  │
│  │  Weight by: strategy confidence × retrieval score                     │  │
│  │  Track which strategy contributed each result                         │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    ANSWER SYNTHESIS + FEEDBACK                        │  │
│  │                                                                        │  │
│  │  LLM generates answer from retrieved context                          │  │
│  │  Outcome signal: user feedback, answer quality, confidence            │  │
│  │  Update bandit: reward strategies that contributed to good outcome    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                    │                                        │
│                                    ▼                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    MEMORY & LEARNING                                  │  │
│  │                                                                        │  │
│  │  Store: (context_features, strategies_used, outcome) tuples           │  │
│  │  Learn: Update bandit parameters, prune ineffective strategies        │  │
│  │  Adapt: Periodic retraining on accumulated experience                 │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. The Four Query Expansion Strategies

### 4.1 Strategy A: Raw Query Embedding

**Description:** Embed the query directly without any expansion.

```python
async def strategy_raw(query: str) -> list[float]:
    """Direct embedding of original query."""
    return await embedder.embed(query)
```

**Best for:**
- Short, precise queries ("OAuth2 PKCE flow")
- Queries with technical terms that shouldn't be expanded
- When speed is critical

**Weaknesses:**
- Short queries may not match longer documents
- Misses semantic variations

### 4.2 Strategy B: HyDE (Hypothetical Document Embedding)

**Description:** LLM generates a hypothetical answer document, embed that.

```python
async def strategy_hyde(query: str) -> list[float]:
    """Generate hypothetical document and embed it."""
    prompt = f"""Write a detailed paragraph that answers this question:
    Question: {query}

    Write as if you are an expert explaining this topic."""

    hypothetical_doc = await llm.chat([{"role": "user", "content": prompt}])
    return await embedder.embed(hypothetical_doc)
```

**Best for:**
- Open-ended questions ("How do teams handle X?")
- When query is much shorter than target documents
- Exploratory queries

**Weaknesses:**
- Can hallucinate irrelevant content
- Loses the original query terms
- Slower (requires LLM call)

### 4.3 Strategy C: Query2Doc (Original + Expansion)

**Description:** Concatenate original query with LLM-generated expansion.

```python
async def strategy_query2doc(query: str) -> list[float]:
    """Expand query but keep original."""
    prompt = f"""Expand this query with relevant context and terminology:
    Query: {query}

    Write 2-3 sentences that elaborate on what the user is asking."""

    expansion = await llm.chat([{"role": "user", "content": prompt}])
    combined = f"{query} {expansion}"
    return await embedder.embed(combined)
```

**Best for:**
- Queries that need context but shouldn't lose original terms
- Medium-complexity questions
- When hallucination risk needs mitigation

**Weaknesses:**
- Still can hallucinate in expansion
- Longer text may dilute key terms

### 4.4 Strategy D: Phase0/1 + Graph-Grounded Expansion

**Description:** Use semantic decomposition to extract entities, look up related context from knowledge graph, then expand with grounded information.

```python
async def strategy_grounded(query: str, user_context: dict) -> list[float]:
    """Graph-grounded expansion using Phase 0/1 decomposition."""

    # Phase 0: Extract entities and structure
    decomposition = await decomposer.decompose(query)
    entities = [e.name for e in decomposition.entities]

    # Look up related entities in knowledge graph
    graph_context = await semantic_memory.find_related(entities, limit=10)

    # Phase 1: Expand with grounded context
    context_str = format_graph_context(graph_context)
    prompt = f"""Given this query and known context, write an expanded version:

    Query: {query}
    Known entities: {entities}
    Related context: {context_str}
    User context: {user_context}

    Expand the query using ONLY the known context. Do not invent entities."""

    grounded_expansion = await llm.chat([{"role": "user", "content": prompt}])
    return await embedder.embed(f"{query} {grounded_expansion}")
```

**Best for:**
- Queries with ambiguous references ("other teams", "the API")
- When knowledge graph has relevant context
- Enterprise/domain-specific queries

**Weaknesses:**
- Requires populated knowledge graph
- Slowest (graph lookup + LLM call)
- May miss novel entities not in graph

### 4.5 Strategy E: Multi-Query Fusion (RAG-Fusion)

**Description:** Generate multiple query variations, retrieve for each, merge with RRF.

```python
async def strategy_fusion(query: str, n_variations: int = 3) -> list[Observation]:
    """Generate multiple query variations and fuse results."""

    prompt = f"""Generate {n_variations} different ways to ask this question:
    Original: {query}

    Make each variation capture a different aspect or phrasing."""

    variations = parse_variations(await llm.chat([{"role": "user", "content": prompt}]))

    # Retrieve for each variation in parallel
    all_results = await asyncio.gather(*[
        retrieve_for_query(v) for v in [query] + variations
    ])

    # Merge with Reciprocal Rank Fusion
    return rrf_merge(all_results)
```

**Best for:**
- Complex queries with multiple facets
- When single phrasing might miss relevant documents
- Comprehensive research queries

**Weaknesses:**
- Most expensive (multiple retrievals)
- Can introduce noise from bad variations
- Overkill for simple queries

---

## 5. Contextual Bandit Learning

### 5.1 Problem Formulation

At each query, the system must choose which strategies to run. This is a **contextual multi-armed bandit** problem:

- **Context (x):** Features extracted from the query
- **Arms (a):** The 5 strategies (A, B, C, D, E)
- **Reward (r):** Outcome quality signal

### 5.2 Context Features

```python
@dataclass
class QueryContext:
    """Features extracted from query for bandit decision."""

    # Length features
    query_length: int              # Character count
    word_count: int                # Token count

    # Complexity features
    entity_count: int              # Named entities detected
    has_ambiguous_refs: bool       # "other", "the", etc.
    question_type: str             # factoid, how-to, comparison, etc.

    # Domain features
    domain_embedding: list[float]  # Query embedding for similarity
    detected_domain: str           # tech, business, personal, etc.

    # Historical features
    similar_query_outcomes: dict   # What worked for similar queries

    # Graph features
    graph_entities_available: int  # How many related entities in graph
```

### 5.3 Bandit Algorithm: LinUCB with Disjoint Arms

We use **LinUCB** because:
1. Handles continuous context features (embeddings)
2. Balances exploration vs exploitation
3. Proven effective for retrieval problems

```python
class LinUCBStrategySelector:
    """LinUCB contextual bandit for strategy selection."""

    def __init__(self, n_arms: int = 5, d: int = 64, alpha: float = 1.0):
        self.n_arms = n_arms
        self.d = d  # Context feature dimension
        self.alpha = alpha  # Exploration parameter

        # Per-arm parameters
        self.A = [np.eye(d) for _ in range(n_arms)]  # d×d matrices
        self.b = [np.zeros(d) for _ in range(n_arms)]  # d-vectors

    def select_arms(self, context: np.ndarray, threshold: float = 0.1) -> list[int]:
        """Select which arms to pull based on context."""
        ucb_scores = []

        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]

            # UCB = predicted reward + exploration bonus
            p = theta @ context + self.alpha * np.sqrt(context @ A_inv @ context)
            ucb_scores.append(p)

        # Select arms above threshold (relative to max)
        max_score = max(ucb_scores)
        selected = [i for i, s in enumerate(ucb_scores) if s >= max_score * threshold]

        # Always include at least one arm (exploration)
        if not selected:
            selected = [np.argmax(ucb_scores)]

        return selected

    def update(self, arm: int, context: np.ndarray, reward: float):
        """Update arm parameters based on observed reward."""
        self.A[arm] += np.outer(context, context)
        self.b[arm] += reward * context
```

### 5.4 Reward Signal

The reward combines multiple signals:

```python
def compute_reward(
    result: RetrievalResult,
    user_feedback: Optional[str],
    strategy_latency: float,
) -> float:
    """Compute reward for a strategy based on outcome."""

    # Base reward: Did retrieved content contribute to answer?
    contribution_score = result.contribution_to_answer  # 0.0 - 1.0

    # User feedback bonus
    if user_feedback == "helpful":
        feedback_bonus = 0.3
    elif user_feedback == "not_helpful":
        feedback_bonus = -0.3
    else:
        feedback_bonus = 0.0

    # Efficiency penalty (prefer faster strategies)
    latency_penalty = min(strategy_latency / 5000, 0.2)  # Cap at 0.2

    # Combine
    reward = contribution_score + feedback_bonus - latency_penalty

    return max(0.0, min(1.0, reward))  # Clamp to [0, 1]
```

### 5.5 Exploration vs Exploitation

```python
class AdaptiveExploration:
    """Adaptive exploration rate based on confidence."""

    def __init__(self, initial_rate: float = 0.3, min_rate: float = 0.05):
        self.rate = initial_rate
        self.min_rate = min_rate
        self.query_count = 0

    def should_explore(self, bandit_confidence: float) -> bool:
        """Decide whether to force exploration."""
        # Decay exploration over time
        effective_rate = max(
            self.min_rate,
            self.rate * (0.99 ** self.query_count)
        )

        # Explore more when bandit is uncertain
        if bandit_confidence < 0.5:
            effective_rate *= 2

        self.query_count += 1
        return random.random() < effective_rate

    def force_all_arms(self) -> bool:
        """Periodically force all arms for calibration."""
        return self.query_count % 100 == 0  # Every 100 queries
```

---

## 6. Learning Lifecycle

### 6.1 Cold Start (No History)

```
Query 1-100: Run ALL strategies in parallel
             Collect outcomes for each
             Build initial bandit model

Query 101+:  Use bandit to select strategies
             Still explore 10-30% of time
             Continue learning from outcomes
```

### 6.2 Warm Operation

```
1. Query arrives
2. Extract context features
3. Bandit selects strategies (typically 2-3 of 5)
4. Run selected strategies in parallel
5. Fuse results, generate answer
6. Observe outcome (implicit + explicit feedback)
7. Update bandit with (context, arm, reward) tuples
```

### 6.3 Strategy Pruning

After sufficient data, identify strategies that NEVER help:

```python
def should_prune_strategy(strategy_id: int, min_samples: int = 100) -> bool:
    """Decide if a strategy should be permanently disabled."""

    stats = get_strategy_stats(strategy_id)

    if stats.total_uses < min_samples:
        return False  # Not enough data

    # Prune if consistently worse than random
    if stats.avg_reward < 0.1 and stats.contribution_rate < 0.05:
        return True

    # Prune if always dominated by another strategy
    for other_id in range(5):
        if other_id == strategy_id:
            continue
        other_stats = get_strategy_stats(other_id)
        if stats.avg_reward < other_stats.avg_reward * 0.5:
            # This strategy is always 2x worse
            return True

    return False
```

### 6.4 Periodic Recalibration

```python
async def recalibrate_bandit(frequency: str = "weekly"):
    """Periodically recalibrate bandit on accumulated data."""

    # Get all historical (context, arm, reward) tuples
    history = await get_retrieval_history(days=30)

    # Retrain bandit from scratch
    new_bandit = LinUCBStrategySelector()
    for ctx, arm, reward in history:
        new_bandit.update(arm, ctx, reward)

    # Validate on holdout set
    holdout = history[-100:]
    validation_score = evaluate_bandit(new_bandit, holdout)

    if validation_score > current_bandit_score * 0.95:
        # New bandit is at least 95% as good
        deploy_bandit(new_bandit)
```

---

## 7. Integration with Existing Systems

### 7.1 Memory Reinforcement Connection

The adaptive retrieval system connects to the existing memory reinforcement system:

```python
# When a retrieval strategy contributes to a good answer:
for memory_id in result.memories_used:
    await memory_provider.record_usage(memory_id, "success")

# When a retrieval strategy led to a bad answer:
for memory_id in result.memories_used:
    await memory_provider.record_usage(memory_id, "failure")
```

### 7.2 Belief System Connection

Retrieval outcomes inform belief confidence:

```python
if retrieval_succeeded and answer_was_helpful:
    # Strengthen beliefs supported by retrieved evidence
    for belief in beliefs_used:
        belief.confidence = min(1.0, belief.confidence + 0.05)
```

### 7.3 Metacognitive Layer

The system's strategy preferences become metacognitive knowledge:

```python
# Store learned preferences as metacognitive memory
await memory_provider.store(
    content="For technical API queries, Strategy D (graph-grounded) works best",
    memory_type=MemoryType.INSIGHT,
    layer=MemoryLayer.METACOGNITIVE,
    metadata={"domain": "api", "best_strategy": "D", "confidence": 0.85}
)
```

---

## 8. Metrics and Monitoring

### 8.1 Per-Strategy Metrics

| Metric | Description |
|--------|-------------|
| **Selection Rate** | How often this strategy is chosen |
| **Contribution Rate** | How often results from this strategy appear in final answer |
| **Avg Latency** | Mean execution time |
| **Avg Reward** | Mean reward when used |
| **Exploration vs Exploitation Ratio** | How often chosen by exploration |

### 8.2 System Metrics

| Metric | Description |
|--------|-------------|
| **Strategy Diversity** | Entropy of strategy selection distribution |
| **Convergence Rate** | How quickly bandit stabilizes |
| **Regret** | Cumulative difference from optimal (hindsight) |
| **Pruning Rate** | Strategies disabled over time |

### 8.3 Quality Metrics

| Metric | Description |
|--------|-------------|
| **Answer Quality** | User ratings, LLM-as-judge scores |
| **Retrieval Precision** | Relevant docs / retrieved docs |
| **Retrieval Recall** | Relevant docs retrieved / total relevant |
| **Latency** | End-to-end query time |

---

## 9. Implementation Plan

### Phase 1: Implement Four Strategies (This PR)

1. Implement `StrategyA_Raw`
2. Implement `StrategyB_HyDE`
3. Implement `StrategyC_Query2Doc`
4. Implement `StrategyD_Grounded`
5. Implement parallel execution with RRF fusion
6. Test with Ollama embeddings

### Phase 2: Add Bandit Learning

1. Implement `LinUCBStrategySelector`
2. Add context feature extraction
3. Implement reward computation
4. Add outcome tracking and bandit updates
5. Test learning behavior

### Phase 3: Integration and Monitoring

1. Connect to memory reinforcement
2. Add metrics and monitoring
3. Implement strategy pruning
4. Add periodic recalibration
5. Deploy and observe in production

---

## 10. Open Questions

1. **How much exploration is enough?** Start with 20%, tune based on regret.

2. **Should we weight strategy contributions in RRF?** Yes, by bandit confidence.

3. **How to handle cold start for new domains?** Transfer learning from similar domains.

4. **When to permanently prune a strategy?** After 100+ uses with <5% contribution.

5. **How to detect distribution shift?** Monitor strategy selection entropy over time.

---

## 11. References

- [Adaptive-RAG](https://arxiv.org/abs/2403.14403) - NAACL 2024, classifier-based strategy selection
- [MBA-RAG](https://arxiv.org/abs/2412.01572) - Multi-armed bandit for RAG strategy selection
- [GGI-MO-MAB](https://arxiv.org/html/2412.07618v2) - Multi-objective bandit for KG-RAG
- [HyDE](https://arxiv.org/abs/2212.10496) - Hypothetical Document Embeddings
- [Query2Doc](https://arxiv.org/abs/2303.07678) - Query expansion with LLMs
- [LinUCB](https://arxiv.org/abs/1003.0146) - Contextual bandit algorithm
- [RAG-Fusion](https://arxiv.org/abs/2402.03367) - Multi-query retrieval fusion

---

**End of FR-011**
