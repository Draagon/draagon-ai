# TASK-090: Industry Baseline Comparison

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Proves competitive positioning)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-089 (Full Benchmark)

---

## Description

Compare benchmark results against industry baselines to validate competitive positioning:
- BM25: Should beat by +20% nDCG@10
- Contriever: Should beat by +10% nDCG@10
- Published BEIR benchmarks for reference
- Generate comparison report with significance testing

This proves we're not just "working" but actually industry-leading.

**Location:** `src/draagon_ai/testing/benchmarks/industry_comparison.py`

---

## Acceptance Criteria

### Baseline Targets
- [ ] Beat BM25 by ≥20% on nDCG@10
- [ ] Beat Contriever by ≥10% on nDCG@10
- [ ] Faithfulness ≥ 0.80
- [ ] Context Recall ≥ 0.80
- [ ] Multi-hop success rate ≥ 70%

### Statistical Validation
- [ ] Improvements are statistically significant (p < 0.05)
- [ ] Effect size (Cohen's d) calculated
- [ ] Confidence intervals don't overlap with baselines

### BEIR Reference
- [ ] Compare to published BEIR scores for mxbai-embed-large
- [ ] Note any discrepancies with explanations
- [ ] Document corpus differences

### Comparison Report
- [ ] Side-by-side metric tables
- [ ] Improvement percentages with significance stars
- [ ] Visualization data for charts
- [ ] Executive summary of competitive position

---

## Technical Notes

### Industry Baselines

```python
# Published baseline scores for reference
INDUSTRY_BASELINES = {
    "BM25": {
        # BEIR average scores
        "ndcg_at_10": 0.42,
        "recall_at_10": 0.65,
        "mrr": 0.45,
        "source": "BEIR Benchmark (2021)",
    },
    "Contriever": {
        "ndcg_at_10": 0.46,
        "recall_at_10": 0.70,
        "mrr": 0.49,
        "source": "Contriever Paper (2022)",
    },
    "mxbai-embed-large": {
        "mteb_score": 64.68,
        "ndcg_at_10": 0.52,  # Estimated from MTEB
        "source": "MTEB Leaderboard (2024)",
    },
    "OpenAI ada-002": {
        "ndcg_at_10": 0.54,
        "recall_at_10": 0.75,
        "source": "Various benchmarks",
    },
}

# Our target improvements
IMPROVEMENT_TARGETS = {
    "vs_bm25": {
        "ndcg_at_10": 0.20,  # +20%
        "recall_at_10": 0.15,
    },
    "vs_contriever": {
        "ndcg_at_10": 0.10,  # +10%
        "recall_at_10": 0.10,
    },
}
```

### Comparison Framework

```python
from dataclasses import dataclass
from scipy import stats

@dataclass
class BaselineComparison:
    baseline_name: str
    metric_name: str
    baseline_score: float
    our_score: float
    improvement_absolute: float
    improvement_percent: float
    p_value: float
    effect_size: float  # Cohen's d
    is_significant: bool
    meets_target: bool
    target_improvement: float


class IndustryComparator:
    def __init__(
        self,
        our_results: HarnessResult,
        baseline_results: dict[str, HarnessResult],
    ):
        self.our_results = our_results
        self.baseline_results = baseline_results

    def compare_all(self) -> list[BaselineComparison]:
        """Compare against all baselines."""
        comparisons = []

        for baseline_name, baseline_result in self.baseline_results.items():
            for metric_name in ["ndcg_at_10", "recall_at_10", "mrr"]:
                if metric_name not in self.our_results.aggregates:
                    continue
                if metric_name not in baseline_result.aggregates:
                    continue

                comparison = self._compare_metric(
                    baseline_name,
                    metric_name,
                    baseline_result.aggregates[metric_name],
                    self.our_results.aggregates[metric_name],
                )
                comparisons.append(comparison)

        return comparisons

    def _compare_metric(
        self,
        baseline_name: str,
        metric_name: str,
        baseline_agg: AggregateResult,
        our_agg: AggregateResult,
    ) -> BaselineComparison:
        """Statistical comparison for single metric."""

        # Absolute and relative improvement
        improvement_abs = our_agg.mean - baseline_agg.mean
        improvement_pct = improvement_abs / baseline_agg.mean if baseline_agg.mean > 0 else 0

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(our_agg.values, baseline_agg.values)

        # Effect size (Cohen's d)
        pooled_std = ((our_agg.std**2 + baseline_agg.std**2) / 2) ** 0.5
        effect_size = improvement_abs / pooled_std if pooled_std > 0 else 0

        # Check target
        target_key = f"vs_{baseline_name.lower()}"
        target = IMPROVEMENT_TARGETS.get(target_key, {}).get(metric_name, 0)
        meets_target = improvement_pct >= target

        return BaselineComparison(
            baseline_name=baseline_name,
            metric_name=metric_name,
            baseline_score=baseline_agg.mean,
            our_score=our_agg.mean,
            improvement_absolute=improvement_abs,
            improvement_percent=improvement_pct,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < 0.05,
            meets_target=meets_target,
            target_improvement=target,
        )

    def generate_comparison_report(self) -> str:
        """Generate comprehensive comparison report."""
        comparisons = self.compare_all()

        return f"""# Industry Baseline Comparison

## Executive Summary

{self._generate_summary(comparisons)}

## Detailed Comparisons

{self._generate_comparison_table(comparisons)}

## Statistical Significance

{self._generate_significance_section(comparisons)}

## Target Achievement

{self._generate_target_section(comparisons)}

## Methodology Notes

- Baselines run on same corpus and queries as our system
- {self.our_results.num_runs} runs with different seeds for statistical validity
- p-values from paired t-test
- Effect size is Cohen's d (small: 0.2, medium: 0.5, large: 0.8)

## References

{self._generate_references()}
"""

    def _generate_summary(self, comparisons: list[BaselineComparison]) -> str:
        """Generate executive summary."""
        all_targets_met = all(c.meets_target for c in comparisons)
        all_significant = all(c.is_significant for c in comparisons if c.improvement_percent > 0)

        status = "✅ All targets met" if all_targets_met else "⚠️ Some targets not met"

        lines = [status, ""]

        # Group by baseline
        for baseline in ["BM25", "Contriever"]:
            baseline_comps = [c for c in comparisons if c.baseline_name.lower() == baseline.lower()]
            if baseline_comps:
                avg_improvement = sum(c.improvement_percent for c in baseline_comps) / len(baseline_comps)
                lines.append(f"- vs {baseline}: +{avg_improvement*100:.1f}% average improvement")

        return "\n".join(lines)

    def _generate_comparison_table(self, comparisons: list[BaselineComparison]) -> str:
        """Generate markdown comparison table."""
        lines = [
            "| Baseline | Metric | Baseline | Ours | Δ | % | p-value | Sig |",
            "|----------|--------|----------|------|---|---|---------|-----|",
        ]

        for c in comparisons:
            sig = "***" if c.p_value < 0.001 else "**" if c.p_value < 0.01 else "*" if c.p_value < 0.05 else ""
            lines.append(
                f"| {c.baseline_name} | {c.metric_name} | {c.baseline_score:.3f} | "
                f"{c.our_score:.3f} | {c.improvement_absolute:+.3f} | "
                f"{c.improvement_percent*100:+.1f}% | {c.p_value:.4f} | {sig} |"
            )

        return "\n".join(lines)
```

### BEIR Reference Comparison

```python
def compare_to_beir_published() -> str:
    """Compare our results to published BEIR scores."""

    # Our corpus is custom, so we note differences
    notes = """
## BEIR Reference Comparison

Our benchmark uses a custom diverse corpus (500+ documents across 8 categories)
rather than standard BEIR datasets. Direct comparison to published BEIR scores
requires caution due to:

1. **Corpus Differences**: Our corpus includes legal, narrative, conversational
   content not in typical BEIR datasets

2. **Query Types**: We include multi-hop, zero-result, and adversarial queries
   not present in standard BEIR

3. **Evaluation Method**: We use RAGAS for faithfulness evaluation in addition
   to standard retrieval metrics

### Published BEIR Scores (for reference)

| Model | BEIR Avg nDCG@10 | Our nDCG@10 | Notes |
|-------|------------------|-------------|-------|
| BM25 | 0.42 | {our_bm25} | Lexical baseline |
| Contriever | 0.46 | {our_contriever} | Unsupervised dense |
| mxbai-embed-large | 0.52* | {our_score} | *Estimated from MTEB |

*Our higher scores likely reflect:*
- Hybrid retrieval (combining dense + lexical)
- Query expansion
- Re-ranking

*Our lower scores on some queries likely reflect:*
- Challenging multi-hop queries
- Adversarial examples
- Zero-result detection (penalizes false positives)
"""
    return notes
```

---

## Testing Requirements

### Unit Tests
```python
def test_improvement_calculation():
    """Improvement percentages calculated correctly."""
    comparator = IndustryComparator(our_results, baseline_results)
    comparisons = comparator.compare_all()

    for c in comparisons:
        expected_pct = (c.our_score - c.baseline_score) / c.baseline_score
        assert abs(c.improvement_percent - expected_pct) < 0.001

def test_significance_testing():
    """Significance correctly determined."""
    # Create results where improvement is clearly significant
    our_values = [0.9, 0.91, 0.89, 0.90, 0.92]
    baseline_values = [0.7, 0.71, 0.69, 0.70, 0.72]

    # Should be significant
    t_stat, p_value = stats.ttest_rel(our_values, baseline_values)
    assert p_value < 0.01

def test_target_achievement():
    """Target achievement correctly evaluated."""
    comparison = BaselineComparison(
        baseline_name="bm25",
        metric_name="ndcg_at_10",
        baseline_score=0.42,
        our_score=0.52,
        improvement_percent=0.238,  # +23.8%
        target_improvement=0.20,
        ...
    )

    assert comparison.meets_target  # 23.8% > 20% target
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_comparison():
    """Run comparison with real benchmark results."""
    # Load real results
    result = BenchmarkResult.load(Path("benchmark_results/latest"))

    comparator = IndustryComparator(
        our_results=result.harness_result,
        baseline_results=result.baseline_results,
    )

    comparisons = comparator.compare_all()
    report = comparator.generate_comparison_report()

    # Verify targets
    bm25_ndcg = next(c for c in comparisons if c.baseline_name == "BM25" and c.metric_name == "ndcg_at_10")
    assert bm25_ndcg.meets_target, f"BM25 target not met: {bm25_ndcg.improvement_percent*100:.1f}% < 20%"
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/industry_comparison.py`
- `src/draagon_ai/testing/benchmarks/baselines.py` (add published scores)
- Add to report generation
- Add tests to `tests/benchmarks/test_comparison.py`

---

## Definition of Done

- [ ] Beat BM25 by ≥20% nDCG@10 (with statistical significance)
- [ ] Beat Contriever by ≥10% nDCG@10 (with statistical significance)
- [ ] Effect sizes calculated (Cohen's d)
- [ ] Comparison report generated
- [ ] BEIR reference notes included
- [ ] All improvements statistically significant
- [ ] Target achievement tracked
- [ ] Tests passing
