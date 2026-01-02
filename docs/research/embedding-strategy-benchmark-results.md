# Embedding Strategy Benchmark Results

**Date:** 2026-01-02
**Test Framework:** `tests/integration/agents/benchmark_embedding_strategies.py`

## Executive Summary

Tested 4 embedding strategies (RAW, HyDE, Query2Doc, Grounded) across 26 test cases in 4 difficulty tiers plus a held-out set for overfitting detection. Key finding: **all strategies show signs of overfitting** with 17-23% performance gaps between training and holdout sets.

## Strategy Definitions

| Strategy | Description | Latency | Best For |
|----------|-------------|---------|----------|
| **RAW** | Direct query embedding | ~10ms | Direct matches, domain jargon |
| **HyDE** | LLM generates hypothetical answer, embed that | ~650ms | Lexical mismatch, multi-hop reasoning |
| **Query2Doc** | Original + LLM expansion concatenated | ~300ms | Entity disambiguation, balanced approach |
| **Grounded** | Uses graph context to expand query | ~150ms | When graph context is available |

## Performance by Tier

```
Strategy         Tier 1     Tier 2     Tier 3     Tier 4    Holdout
----------------------------------------------------------------------
raw              100.0%      93.3%      93.3%      87.5%      68.8%
hyde             100.0%      80.0%      93.3%      95.8%      75.0%
query2doc        100.0%     100.0%      93.3%      91.7%      75.0%
grounded         100.0%     100.0%      93.3%      85.4%      68.8%
```

### Key Observations

1. **All strategies ace Tier 1** (100%) - basic direct queries work for everyone
2. **HyDE struggles with entity confusion** (80% Tier 2) - LLM hallucinations cause wrong Atlas disambiguation
3. **HyDE excels at Tier 4** (95.8%) - hypothetical document generation helps with complex queries
4. **Major holdout gap for all** - indicates overfitting to test patterns

## Performance by Failure Mode

| Failure Mode | Best Strategy | RAW | HyDE | Query2Doc | Grounded |
|--------------|--------------|-----|------|-----------|----------|
| **None** (direct) | RAW | 100% | 100% | 100% | 100% |
| **Lexical Mismatch** | HyDE | 83% | **100%** | 90% | 85% |
| **Entity Confusion** | RAW/Q2D | **100%** | 75% | **100%** | 100% |
| **Multi-hop** | HyDE | 73% | **83%** | 83% | 73% |
| **Implicit Reference** | All tie | 100% | 100% | 100% | 100% |
| **Out-of-domain** | All tie | 100% | 100% | 100% | 100% |

### Strategy Recommendations by Query Type

| Query Type | Recommended Strategy | Reason |
|------------|---------------------|--------|
| Direct lookup | RAW | Fastest, works fine |
| Synonym/rephrasing | HyDE | Bridges vocabulary gap |
| Ambiguous entities | Query2Doc | Context helps disambiguate |
| Multi-hop reasoning | HyDE | Hypothetical doc connects facts |
| Technical jargon | RAW | Embeddings already understand |

## Test Cases That Broke All Strategies

### 1. Multi-hop with Missing Link (0% across all)
```
Query: "What auth does the team that uses PostgreSQL employ?"
Expected: ["Analytics"] (Analytics uses PostgreSQL, but no auth info exists)
```
**Why it fails:** The knowledge base has no authentication info for the Analytics team. This tests graceful failure handling.

### 2. Negation Queries (partial failures)
```
Query: "Which teams explicitly avoid using JWT tokens?"
Expected: ["Platform", "mTLS", "Mobile", "biometric"] (teams NOT using JWT)
Best: HyDE @ 100%, others ≤50%
```
**Why RAW/Grounded fail:** Embeddings can't understand negation - "avoid JWT" is semantically close to "JWT".

## Overfitting Analysis

All strategies show significant overfitting:
```
raw          train=91.7% holdout=68.8% gap=+22.9% ⚠️ OVERFITTING
hyde         train=92.9% holdout=75.0% gap=+17.9% ⚠️ OVERFITTING
query2doc    train=94.9% holdout=75.0% gap=+19.9% ⚠️ OVERFITTING
grounded     train=92.0% holdout=68.8% gap=+23.2% ⚠️ OVERFITTING
```

**Causes:**
1. Small test set (14 train, 4 holdout) - high variance
2. Holdout has harder patterns (negation, missing data, comparison)
3. Knowledge base is constrained - limits query diversity

**Mitigations (for adaptive learning system):**
1. Increase holdout set size to 30%+ of total
2. Use k-fold cross-validation instead of single split
3. Add randomized test generation
4. Include adversarial query augmentation

## Recommendations for Adaptive Strategy Selection

Based on these results, an adaptive system should:

### 1. Start with Strategy Priors
```python
initial_weights = {
    "raw": 0.25,      # Good baseline
    "hyde": 0.30,     # Best for hard cases
    "query2doc": 0.25, # Good balance
    "grounded": 0.20,  # Only when graph available
}
```

### 2. Context Features for Bandit Learning
- **Query length**: Short → RAW, Long → HyDE
- **Has negation words**: True → HyDE
- **Has entity names**: True → Query2Doc
- **Query format**: Keyword → RAW, Question → HyDE
- **Domain jargon detected**: True → RAW (embeddings understand)

### 3. Avoid Single-Strategy Reliance
Run top 2-3 strategies in parallel, merge with RRF. No single strategy wins all cases.

### 4. Watchlist Failure Modes
- **Negation**: Embeddings fundamentally struggle. May need pre-processing to detect.
- **Missing data**: No strategy can find non-existent information.
- **Entity confusion with LLM expansion**: HyDE can hallucinate wrong context.

## Next Steps

1. **Expand holdout set** - Need more diverse holdout patterns
2. **Implement LinUCB bandit** - Use query features to select strategies
3. **Add negation detection** - Pre-filter queries with negation for special handling
4. **Test with larger knowledge base** - Current KB (12 docs) may not reflect production
5. **Add latency constraints** - For real-time systems, may need to skip HyDE

## Semantic Expansion Integration (2026-01-02)

### What We Built

Created `benchmark_semantic_expansion_integration.py` that tests how semantic expansion affects retrieval quality:

1. **SemanticExpansionPreprocessor** - Integration wrapper for prototype
2. **Tiered Test Cases** (16 total):
   - Tier 1: Direct queries (baseline)
   - Tier 2: Lexical mismatch (expansion should help)
   - Tier 3: Entity disambiguation
   - Tier 4: Word sense disambiguation
   - Tier 5: Multi-hop reasoning
   - Tier 6: Pronoun resolution
   - Tier 7: Concept/synonym expansion

### Baseline Results (Without Expansion)

| Ambiguity Type | Baseline Recall | Notes |
|----------------|-----------------|-------|
| None (direct) | 100% | Works fine |
| Entity | 100% | Small KB helps |
| Word Sense | 100% | Both "bank" docs retrieved |
| Multi-hop | 100% | Related terms connect |
| Pronoun | 100% | Context in query helps |
| Concept | 100% | Domain terms work |
| **Lexical** | **55.6%** | **This is where expansion should help** |

### Key Finding: Lexical Mismatch is the Target

The lexical mismatch cases show clear degradation:
- "What security measures protect data transfer?" → 0% recall (should find mTLS)
- "How do we handle user login?" → 66.7% recall

These are the cases where semantic expansion adds the most value by bridging vocabulary gaps.

### Files Created

- `tests/integration/agents/benchmark_semantic_expansion_integration.py`
- Updated `docs/research/comprehensive-retrieval-benchmark-plan.md`

### Prototype Status

The semantic expansion prototype (`prototypes/semantic_expansion/`) is ready for integration:
- 135/135 tests passing
- Core components: WSD, SemanticExpansionService, TwoPassOrchestrator
- Integration point: `SemanticExpansionPreprocessor.expand_query()`

## Sources

- [BEIR Benchmark](https://github.com/beir-cellar/beir)
- [MTEB Benchmark](https://huggingface.co/spaces/mteb/leaderboard)
- [Hard Negative Mining for Enterprise Retrieval](https://arxiv.org/html/2505.18366)
- [RAG Failure Modes](https://snorkel.ai/blog/retrieval-augmented-generation-rag-failure-modes)
- [ViDoRe V3 Enterprise Benchmark](https://huggingface.co/blog/QuentinJG/introducing-vidore-v3)
