# TASK-091: Smoke Test Suite (50 queries)

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Required for PR validation)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-086 (BenchmarkRunner)

---

## Description

Create a fast smoke test subset for CI/CD:
- 50 representative queries (stratified sample)
- Single run (no statistical aggregation)
- Simplified evaluation (key metrics only)
- ≤ 5 minutes runtime
- Pass/fail based on regression thresholds

This runs on every PR to catch obvious regressions quickly.

**Location:** `src/draagon_ai/testing/benchmarks/smoke.py`

---

## Acceptance Criteria

### Query Selection
- [ ] 50 queries stratified by type, difficulty, category
- [ ] Fixed query set (deterministic, not random)
- [ ] Includes at least:
  - 30 standard queries
  - 10 multi-hop queries
  - 5 zero-result queries
  - 5 adversarial queries

### Execution
- [ ] Single run (no multiple seeds)
- [ ] ≤ 5 minutes total runtime
- [ ] Uses cached corpus (no re-indexing if unchanged)
- [ ] Parallel query execution (10 concurrent)

### Evaluation
- [ ] Core metrics only: faithfulness, context_recall
- [ ] Skip expensive evaluations (answer_relevancy optional)
- [ ] Threshold-based pass/fail

### Regression Detection
- [ ] Compare to baseline scores (from last successful run)
- [ ] Fail if any metric drops > 5%
- [ ] Report which metrics regressed

---

## Technical Notes

### Query Selection

```python
def select_smoke_queries(
    full_suite: QuerySuite,
    target_count: int = 50,
) -> QuerySuite:
    """Select stratified sample for smoke testing."""

    # Target distribution
    distribution = {
        QueryType.STANDARD: 30,
        QueryType.MULTI_HOP_BRIDGE: 4,
        QueryType.MULTI_HOP_COMPARISON: 3,
        QueryType.MULTI_HOP_AGGREGATION: 3,
        QueryType.ZERO_RESULT: 5,
        QueryType.ADVERSARIAL: 5,
    }

    selected = []

    for query_type, count in distribution.items():
        type_queries = [q for q in full_suite.queries if q.query_type == query_type]

        # Further stratify by difficulty
        for difficulty in QueryDifficulty:
            diff_queries = [q for q in type_queries if q.difficulty == difficulty]
            per_diff = count // 4  # Roughly equal by difficulty

            # Deterministic selection (by query_id hash)
            sorted_queries = sorted(diff_queries, key=lambda q: q.query_id)
            selected.extend(sorted_queries[:per_diff])

    # Ensure we have exactly target_count
    selected = selected[:target_count]

    return QuerySuite(
        queries=selected,
        metadata={"smoke_test": True, "source": full_suite.metadata.get("source")},
    )


# Pre-generated smoke query IDs for determinism
SMOKE_QUERY_IDS = [
    "std_tech_001", "std_tech_002", "std_legal_001", "std_narrative_001",
    "mh_bridge_001", "mh_bridge_002", "mh_comparison_001",
    "zr_ood_001", "zr_ood_002", "zr_temporal_001",
    "adv_keyword_001", "adv_paraphrase_001",
    # ... 50 total
]
```

### Smoke Test Runner

```python
from dataclasses import dataclass
import time

@dataclass
class SmokeTestResult:
    passed: bool
    duration_seconds: float
    metrics: dict[str, float]
    regressions: list[str]
    baseline_metrics: dict[str, float]


class SmokeTestRunner:
    def __init__(
        self,
        corpus_path: Path,
        queries_path: Path,
        baseline_path: Path,
        regression_threshold: float = 0.05,  # 5%
    ):
        self.corpus_path = corpus_path
        self.queries_path = queries_path
        self.baseline_path = baseline_path
        self.regression_threshold = regression_threshold

    async def run(self) -> SmokeTestResult:
        """Run smoke tests and check for regressions."""
        start_time = time.time()

        # Load data
        corpus = DocumentCorpus.load(self.corpus_path)
        full_queries = QuerySuite.load(self.queries_path)
        smoke_queries = self._get_smoke_queries(full_queries)

        logger.info(f"Running smoke tests with {len(smoke_queries.queries)} queries")

        # Initialize components
        retriever = self._get_retriever()
        evaluator = self._get_evaluator()

        # Index corpus
        await retriever.index(corpus.documents)

        # Run queries
        results = await self._run_queries(retriever, smoke_queries, corpus)

        # Evaluate
        metrics = await self._evaluate(evaluator, results, smoke_queries)

        # Check regressions
        baseline = self._load_baseline()
        regressions = self._check_regressions(metrics, baseline)

        duration = time.time() - start_time

        return SmokeTestResult(
            passed=len(regressions) == 0,
            duration_seconds=duration,
            metrics=metrics,
            regressions=regressions,
            baseline_metrics=baseline,
        )

    async def _run_queries(
        self,
        retriever,
        queries: QuerySuite,
        corpus: DocumentCorpus,
    ) -> list[dict]:
        """Run all queries with concurrency."""
        semaphore = asyncio.Semaphore(10)

        async def run_one(query):
            async with semaphore:
                try:
                    result = await asyncio.wait_for(
                        retriever.retrieve(query.question, k=10),
                        timeout=10.0,
                    )
                    return {"query": query, "result": result, "error": None}
                except Exception as e:
                    return {"query": query, "result": None, "error": str(e)}

        tasks = [run_one(q) for q in queries.queries]
        return await asyncio.gather(*tasks)

    async def _evaluate(
        self,
        evaluator,
        results: list[dict],
        queries: QuerySuite,
    ) -> dict[str, float]:
        """Evaluate results with core metrics only."""
        successful_results = [r for r in results if r["error"] is None]

        if not successful_results:
            return {"faithfulness": 0, "context_recall": 0, "success_rate": 0}

        # Calculate context recall (fast, no LLM needed)
        recall_scores = []
        for r in successful_results:
            query = r["query"]
            retrieved = set(r["result"].doc_ids)
            ground_truth = set(query.ground_truth_document_ids)
            recall = len(retrieved & ground_truth) / len(ground_truth) if ground_truth else 1.0
            recall_scores.append(recall)

        # Faithfulness (sample 20 for speed)
        sample_size = min(20, len(successful_results))
        sample = successful_results[:sample_size]

        faithfulness_scores = []
        for r in sample:
            score = await evaluator.evaluate_faithfulness_fast(
                r["result"].answer,
                r["result"].context[:3],  # First 3 docs only
            )
            faithfulness_scores.append(score)

        return {
            "faithfulness": sum(faithfulness_scores) / len(faithfulness_scores),
            "context_recall": sum(recall_scores) / len(recall_scores),
            "success_rate": len(successful_results) / len(results),
        }

    def _check_regressions(
        self,
        current: dict[str, float],
        baseline: dict[str, float],
    ) -> list[str]:
        """Check if any metrics regressed beyond threshold."""
        regressions = []

        for metric, current_value in current.items():
            baseline_value = baseline.get(metric, 0)

            if baseline_value > 0:
                drop = (baseline_value - current_value) / baseline_value
                if drop > self.regression_threshold:
                    regressions.append(
                        f"{metric}: {current_value:.3f} vs baseline {baseline_value:.3f} "
                        f"(-{drop*100:.1f}%)"
                    )

        return regressions

    def _load_baseline(self) -> dict[str, float]:
        """Load baseline metrics from last successful run."""
        if not self.baseline_path.exists():
            logger.warning("No baseline found, using defaults")
            return {"faithfulness": 0.80, "context_recall": 0.80, "success_rate": 0.95}

        with open(self.baseline_path) as f:
            return json.load(f)

    def save_as_baseline(self, result: SmokeTestResult):
        """Save current results as new baseline (only if passed)."""
        if result.passed:
            with open(self.baseline_path, "w") as f:
                json.dump(result.metrics, f, indent=2)
            logger.info("Saved new baseline")
```

### CLI Entry Point

```python
def main():
    parser = argparse.ArgumentParser(description="Run smoke tests")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--baseline", default="smoke_baseline.json")
    parser.add_argument("--update-baseline", action="store_true")

    args = parser.parse_args()

    runner = SmokeTestRunner(
        corpus_path=Path(args.corpus),
        queries_path=Path(args.queries),
        baseline_path=Path(args.baseline),
    )

    result = asyncio.run(runner.run())

    # Output
    print(f"\n{'='*50}")
    if result.passed:
        print("✅ SMOKE TESTS PASSED")
    else:
        print("❌ SMOKE TESTS FAILED")
        print("\nRegressions:")
        for r in result.regressions:
            print(f"  - {r}")

    print(f"\nDuration: {result.duration_seconds:.1f}s")
    print(f"\nMetrics:")
    for name, value in result.metrics.items():
        print(f"  {name}: {value:.3f}")

    if args.update_baseline and result.passed:
        runner.save_as_baseline(result)

    return 0 if result.passed else 1


if __name__ == "__main__":
    exit(main())
```

---

## Testing Requirements

### Unit Tests
```python
def test_smoke_query_selection():
    """Smoke queries are stratified correctly."""
    full_suite = QuerySuite(queries=[...])  # 250 queries
    smoke = select_smoke_queries(full_suite, target_count=50)

    assert len(smoke.queries) == 50

    # Check distribution
    types = Counter(q.query_type for q in smoke.queries)
    assert types[QueryType.STANDARD] >= 25
    assert types.get(QueryType.MULTI_HOP_BRIDGE, 0) >= 2

def test_regression_detection():
    """Regressions detected correctly."""
    runner = SmokeTestRunner(...)

    current = {"faithfulness": 0.75, "context_recall": 0.85}
    baseline = {"faithfulness": 0.80, "context_recall": 0.80}

    regressions = runner._check_regressions(current, baseline)

    assert len(regressions) == 1
    assert "faithfulness" in regressions[0]

def test_no_false_positive_regression():
    """Minor fluctuations don't trigger regression."""
    runner = SmokeTestRunner(regression_threshold=0.05)

    current = {"faithfulness": 0.78}
    baseline = {"faithfulness": 0.80}  # 2.5% drop, below threshold

    regressions = runner._check_regressions(current, baseline)
    assert len(regressions) == 0
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_smoke_run_completes():
    """Smoke tests complete within time limit."""
    runner = SmokeTestRunner(...)

    start = time.time()
    result = await runner.run()
    duration = time.time() - start

    assert duration <= 300  # 5 minutes
    assert result.metrics["success_rate"] > 0.9
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/smoke.py`
- `src/draagon_ai/testing/benchmarks/__main__.py` (add smoke command)
- `data/smoke_queries.json` (pre-selected query IDs)
- Add tests to `tests/benchmarks/test_smoke.py`

---

## Definition of Done

- [ ] 50 queries selected with proper stratification
- [ ] Smoke test runs in ≤ 5 minutes
- [ ] Core metrics calculated correctly
- [ ] Regression detection working
- [ ] Baseline save/load working
- [ ] CLI entry point
- [ ] Integration test passing
- [ ] Can be run from CI/CD
