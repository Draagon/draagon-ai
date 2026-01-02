# TASK-085: Variance Analysis Tools

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Helps diagnose instability)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-083, TASK-084

---

## Description

Analyze sources of variance in benchmark results:
- Identify which queries have high variance
- Decompose variance by query type, difficulty, category
- Detect instability patterns (e.g., embedding initialization)
- Recommend actions to reduce variance

This helps understand WHY results vary and what to fix.

**Location:** `src/draagon_ai/testing/benchmarks/variance_analysis.py`

---

## Acceptance Criteria

### Variance Decomposition
- [ ] Per-query variance across runs
- [ ] Variance by query type (standard, multi-hop, adversarial)
- [ ] Variance by difficulty level
- [ ] Variance by document category

### High-Variance Query Detection
- [ ] Identify queries with CV > 0.1
- [ ] Rank queries by variance
- [ ] Export high-variance queries for investigation

### Pattern Detection
- [ ] Detect systematic patterns (e.g., first run always lower)
- [ ] Identify correlation between variance and query properties
- [ ] Flag potential causes (embedding, LLM temperature, etc.)

### Recommendations
- [ ] Suggest seed stabilization if variance > threshold
- [ ] Recommend more runs if CI is wide
- [ ] Identify queries to exclude or investigate

---

## Technical Notes

### Variance Analyzer

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np
from scipy import stats

@dataclass
class QueryVariance:
    query_id: str
    query_type: QueryType
    difficulty: QueryDifficulty
    category: DocumentCategory
    metric_values: dict[str, list[float]]  # metric -> values per run
    metric_variance: dict[str, float]
    metric_cv: dict[str, float]
    is_high_variance: bool

    def get_most_variable_metric(self) -> tuple[str, float]:
        """Return metric with highest CV."""
        return max(self.metric_cv.items(), key=lambda x: x[1])


@dataclass
class VarianceDecomposition:
    total_variance: float
    by_query_type: dict[QueryType, float]
    by_difficulty: dict[QueryDifficulty, float]
    by_category: dict[DocumentCategory, float]
    residual_variance: float  # Unexplained variance


@dataclass
class VariancePattern:
    pattern_type: str  # "first_run_bias", "trend", "random"
    description: str
    evidence: dict
    recommendation: str


class VarianceAnalyzer:
    def __init__(
        self,
        per_query_results: list[dict[str, dict[str, float]]],  # run -> query -> metrics
        queries: list[BenchmarkQuery],
        high_variance_threshold: float = 0.1,  # CV > 10%
    ):
        self.per_query_results = per_query_results
        self.queries = {q.query_id: q for q in queries}
        self.threshold = high_variance_threshold

    def analyze_per_query_variance(self) -> list[QueryVariance]:
        """Calculate variance for each query across runs."""
        query_variances = []

        for query_id in self._get_all_query_ids():
            query = self.queries.get(query_id)
            if not query:
                continue

            # Collect values per metric across runs
            metric_values = {}
            for run_results in self.per_query_results:
                if query_id in run_results:
                    for metric, value in run_results[query_id].items():
                        if metric not in metric_values:
                            metric_values[metric] = []
                        metric_values[metric].append(value)

            # Calculate variance and CV for each metric
            metric_variance = {}
            metric_cv = {}
            for metric, values in metric_values.items():
                if len(values) > 1:
                    var = np.var(values)
                    mean = np.mean(values)
                    cv = np.std(values) / mean if mean > 0 else 0
                    metric_variance[metric] = var
                    metric_cv[metric] = cv

            # Check if any metric has high variance
            is_high = any(cv > self.threshold for cv in metric_cv.values())

            query_variances.append(QueryVariance(
                query_id=query_id,
                query_type=query.query_type,
                difficulty=query.difficulty,
                category=query.target_category,
                metric_values=metric_values,
                metric_variance=metric_variance,
                metric_cv=metric_cv,
                is_high_variance=is_high,
            ))

        return query_variances

    def decompose_variance(
        self,
        metric_name: str = "faithfulness",
    ) -> VarianceDecomposition:
        """Decompose total variance by query properties."""
        query_variances = self.analyze_per_query_variance()

        # Total variance
        all_values = []
        for qv in query_variances:
            if metric_name in qv.metric_values:
                all_values.extend(qv.metric_values[metric_name])
        total_variance = np.var(all_values) if all_values else 0

        # Variance by query type
        by_type = {}
        for qt in QueryType:
            type_values = []
            for qv in query_variances:
                if qv.query_type == qt and metric_name in qv.metric_values:
                    type_values.extend(qv.metric_values[metric_name])
            if type_values:
                by_type[qt] = np.var(type_values)

        # Variance by difficulty
        by_difficulty = {}
        for diff in QueryDifficulty:
            diff_values = []
            for qv in query_variances:
                if qv.difficulty == diff and metric_name in qv.metric_values:
                    diff_values.extend(qv.metric_values[metric_name])
            if diff_values:
                by_difficulty[diff] = np.var(diff_values)

        # Variance by category
        by_category = {}
        for cat in DocumentCategory:
            cat_values = []
            for qv in query_variances:
                if qv.category == cat and metric_name in qv.metric_values:
                    cat_values.extend(qv.metric_values[metric_name])
            if cat_values:
                by_category[cat] = np.var(cat_values)

        return VarianceDecomposition(
            total_variance=total_variance,
            by_query_type=by_type,
            by_difficulty=by_difficulty,
            by_category=by_category,
            residual_variance=0,  # TODO: Calculate properly
        )

    def detect_patterns(self) -> list[VariancePattern]:
        """Detect systematic variance patterns."""
        patterns = []

        # Pattern 1: First run bias
        first_run_pattern = self._detect_first_run_bias()
        if first_run_pattern:
            patterns.append(first_run_pattern)

        # Pattern 2: Trend (increasing/decreasing performance)
        trend_pattern = self._detect_trend()
        if trend_pattern:
            patterns.append(trend_pattern)

        # Pattern 3: Difficulty correlation
        difficulty_pattern = self._detect_difficulty_correlation()
        if difficulty_pattern:
            patterns.append(difficulty_pattern)

        return patterns

    def _detect_first_run_bias(self) -> Optional[VariancePattern]:
        """Check if first run systematically differs."""
        first_run_means = []
        other_run_means = []

        for query_id in self._get_all_query_ids():
            for metric in ["faithfulness", "context_recall"]:
                values = self._get_query_metric_values(query_id, metric)
                if len(values) >= 3:
                    first_run_means.append(values[0])
                    other_run_means.append(np.mean(values[1:]))

        if not first_run_means:
            return None

        # Statistical test
        t_stat, p_value = stats.ttest_rel(first_run_means, other_run_means)

        if p_value < 0.05:
            diff = np.mean(first_run_means) - np.mean(other_run_means)
            direction = "lower" if diff < 0 else "higher"

            return VariancePattern(
                pattern_type="first_run_bias",
                description=f"First run is systematically {direction} than subsequent runs",
                evidence={"p_value": p_value, "mean_diff": diff},
                recommendation="Consider warm-up run or discarding first run results",
            )

        return None

    def get_high_variance_queries(self) -> list[QueryVariance]:
        """Return queries with high variance for investigation."""
        all_variances = self.analyze_per_query_variance()
        high_variance = [qv for qv in all_variances if qv.is_high_variance]
        return sorted(high_variance, key=lambda qv: max(qv.metric_cv.values()), reverse=True)

    def generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations."""
        recommendations = []

        # Check high variance queries
        high_var = self.get_high_variance_queries()
        if len(high_var) > len(self.queries) * 0.2:
            recommendations.append(
                f"⚠️ {len(high_var)} queries ({len(high_var)/len(self.queries):.0%}) have high variance. "
                "Consider increasing number of runs or investigating embedding stability."
            )

        # Check patterns
        patterns = self.detect_patterns()
        for pattern in patterns:
            recommendations.append(f"📊 {pattern.description}: {pattern.recommendation}")

        # Check overall variance
        decomp = self.decompose_variance()
        if decomp.total_variance > 0.01:
            recommendations.append(
                "Consider fixing random seeds for embeddings and LLM calls."
            )

        return recommendations
```

---

## Testing Requirements

### Unit Tests
```python
def test_per_query_variance():
    """Calculate variance correctly for each query."""
    analyzer = VarianceAnalyzer(
        per_query_results=[
            {"q1": {"accuracy": 0.8}},
            {"q1": {"accuracy": 0.85}},
            {"q1": {"accuracy": 0.82}},
        ],
        queries=[BenchmarkQuery(query_id="q1", ...)],
    )

    variances = analyzer.analyze_per_query_variance()
    assert len(variances) == 1
    assert variances[0].query_id == "q1"
    assert 0 < variances[0].metric_cv["accuracy"] < 0.1

def test_high_variance_detection():
    """High variance queries identified."""
    analyzer = VarianceAnalyzer(
        per_query_results=[
            {"q1": {"accuracy": 0.5}, "q2": {"accuracy": 0.9}},
            {"q1": {"accuracy": 0.9}, "q2": {"accuracy": 0.91}},
        ],
        queries=[...],
        high_variance_threshold=0.1,
    )

    high_var = analyzer.get_high_variance_queries()
    assert any(qv.query_id == "q1" for qv in high_var)
    assert not any(qv.query_id == "q2" for qv in high_var)

def test_first_run_bias_detection():
    """Detect first run bias pattern."""
    # Simulate first run being systematically lower
    results = []
    for run in range(5):
        run_results = {}
        for q in range(10):
            base = 0.85
            if run == 0:
                base -= 0.1  # First run lower
            run_results[f"q{q}"] = {"accuracy": base + np.random.normal(0, 0.01)}
        results.append(run_results)

    analyzer = VarianceAnalyzer(results, queries=[...])
    patterns = analyzer.detect_patterns()

    assert any(p.pattern_type == "first_run_bias" for p in patterns)
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/variance_analysis.py`
- Add tests to `tests/benchmarks/test_variance.py`

---

## Definition of Done

- [ ] Per-query variance calculation
- [ ] Variance decomposition by type/difficulty/category
- [ ] High-variance query identification
- [ ] First run bias detection
- [ ] Trend detection
- [ ] Actionable recommendations
- [ ] Export high-variance queries
- [ ] Unit tests passing
