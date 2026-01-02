# TASK-083: Multiple-Run Harness

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Statistical validity requires multiple runs)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-078 (RAGAS Integration)

---

## Description

Implement harness for running benchmarks multiple times with different seeds:
- Run N iterations (default: 5) of the same benchmark
- Vary random seeds for embedding initialization, query order
- Collect per-run metrics for statistical analysis
- Detect and flag anomalous runs

This ensures results are reproducible and not due to lucky/unlucky randomness.

**Location:** `src/draagon_ai/testing/benchmarks/harness.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `MultiRunHarness` executes benchmark N times
- [ ] Each run uses different random seed
- [ ] Collects all metrics per run
- [ ] Calculates aggregate statistics

### Statistical Output
- [ ] Mean and standard deviation for each metric
- [ ] 95% confidence intervals
- [ ] Min/max values
- [ ] Coefficient of variation (CV) for stability assessment

### Anomaly Detection
- [ ] Flag runs > 2 standard deviations from mean
- [ ] Optionally exclude anomalous runs
- [ ] Report on run-to-run variance

### Reproducibility
- [ ] Each run's seed is recorded
- [ ] Any run can be replayed with same seed
- [ ] Results file includes all seeds used

---

## Technical Notes

### Harness Implementation

```python
from dataclasses import dataclass, field
from typing import Callable, Any
import random
import numpy as np
import statistics

@dataclass
class RunResult:
    run_id: int
    seed: int
    metrics: dict[str, float]
    duration_seconds: float
    is_anomalous: bool = False


@dataclass
class AggregateResult:
    metric_name: str
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    min_value: float
    max_value: float
    coefficient_of_variation: float
    values: list[float] = field(default_factory=list)

    @classmethod
    def from_values(cls, name: str, values: list[float]) -> "AggregateResult":
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0

        # 95% CI using t-distribution
        from scipy import stats
        n = len(values)
        if n > 1:
            t_value = stats.t.ppf(0.975, n - 1)
            margin = t_value * std / (n ** 0.5)
        else:
            margin = 0.0

        cv = std / mean if mean > 0 else 0.0

        return cls(
            metric_name=name,
            mean=mean,
            std=std,
            ci_95_lower=mean - margin,
            ci_95_upper=mean + margin,
            min_value=min(values),
            max_value=max(values),
            coefficient_of_variation=cv,
            values=values,
        )


@dataclass
class HarnessResult:
    num_runs: int
    run_results: list[RunResult]
    aggregates: dict[str, AggregateResult]
    total_duration_seconds: float
    anomalous_runs: list[int]

    def to_report(self) -> str:
        """Generate human-readable report."""
        lines = [
            f"Benchmark Results ({self.num_runs} runs)",
            "=" * 50,
            "",
        ]

        for name, agg in self.aggregates.items():
            lines.append(f"{name}:")
            lines.append(f"  Mean:  {agg.mean:.4f} ± {agg.std:.4f}")
            lines.append(f"  95% CI: [{agg.ci_95_lower:.4f}, {agg.ci_95_upper:.4f}]")
            lines.append(f"  Range: [{agg.min_value:.4f}, {agg.max_value:.4f}]")
            lines.append(f"  CV:    {agg.coefficient_of_variation:.2%}")
            lines.append("")

        if self.anomalous_runs:
            lines.append(f"⚠️  Anomalous runs: {self.anomalous_runs}")

        lines.append(f"Total time: {self.total_duration_seconds:.1f}s")

        return "\n".join(lines)


class MultiRunHarness:
    def __init__(
        self,
        num_runs: int = 5,
        base_seed: int = 42,
        anomaly_threshold: float = 2.0,  # Standard deviations
    ):
        self.num_runs = num_runs
        self.base_seed = base_seed
        self.anomaly_threshold = anomaly_threshold

    async def run(
        self,
        benchmark_fn: Callable[..., dict[str, float]],
        **kwargs,
    ) -> HarnessResult:
        """Run benchmark multiple times and aggregate results."""
        import time

        start_time = time.time()
        run_results = []

        for run_id in range(self.num_runs):
            seed = self.base_seed + run_id
            self._set_seeds(seed)

            run_start = time.time()
            metrics = await benchmark_fn(**kwargs)
            duration = time.time() - run_start

            run_results.append(RunResult(
                run_id=run_id,
                seed=seed,
                metrics=metrics,
                duration_seconds=duration,
            ))

        # Aggregate metrics
        aggregates = self._aggregate_metrics(run_results)

        # Detect anomalies
        anomalous_runs = self._detect_anomalies(run_results, aggregates)

        for result in run_results:
            result.is_anomalous = result.run_id in anomalous_runs

        return HarnessResult(
            num_runs=self.num_runs,
            run_results=run_results,
            aggregates=aggregates,
            total_duration_seconds=time.time() - start_time,
            anomalous_runs=anomalous_runs,
        )

    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass

    def _aggregate_metrics(
        self,
        run_results: list[RunResult],
    ) -> dict[str, AggregateResult]:
        """Calculate aggregate statistics for each metric."""
        all_metrics = {}

        for result in run_results:
            for name, value in result.metrics.items():
                if name not in all_metrics:
                    all_metrics[name] = []
                all_metrics[name].append(value)

        return {
            name: AggregateResult.from_values(name, values)
            for name, values in all_metrics.items()
        }

    def _detect_anomalies(
        self,
        run_results: list[RunResult],
        aggregates: dict[str, AggregateResult],
    ) -> list[int]:
        """Detect runs with metrics outside normal range."""
        anomalous = set()

        for name, agg in aggregates.items():
            if agg.std == 0:
                continue

            for result in run_results:
                value = result.metrics.get(name, 0)
                z_score = abs(value - agg.mean) / agg.std

                if z_score > self.anomaly_threshold:
                    anomalous.add(result.run_id)

        return sorted(anomalous)
```

### Usage Example

```python
async def run_benchmark(corpus, queries, retriever) -> dict[str, float]:
    """Single benchmark run returning metrics."""
    evaluator = RAGASEvaluator(...)
    results = await retriever.retrieve_all(queries)
    evaluation = await evaluator.evaluate_batch(results, queries)

    return {
        "faithfulness": evaluation.mean_faithfulness,
        "context_recall": evaluation.mean_recall,
        "ndcg_at_10": evaluation.ndcg_at_10,
    }

# Run 5 times with different seeds
harness = MultiRunHarness(num_runs=5, base_seed=42)
result = await harness.run(
    run_benchmark,
    corpus=corpus,
    queries=queries,
    retriever=retriever,
)

print(result.to_report())
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_multiple_runs():
    """Harness executes correct number of runs."""
    async def dummy_benchmark():
        return {"accuracy": random.random()}

    harness = MultiRunHarness(num_runs=3)
    result = await harness.run(dummy_benchmark)

    assert len(result.run_results) == 3
    assert "accuracy" in result.aggregates

@pytest.mark.asyncio
async def test_seed_reproducibility():
    """Same seed produces same results."""
    async def seeded_benchmark():
        return {"value": random.random()}

    harness = MultiRunHarness(num_runs=1, base_seed=42)
    result1 = await harness.run(seeded_benchmark)

    harness2 = MultiRunHarness(num_runs=1, base_seed=42)
    result2 = await harness2.run(seeded_benchmark)

    assert result1.run_results[0].metrics == result2.run_results[0].metrics

@pytest.mark.asyncio
async def test_anomaly_detection():
    """Anomalous runs are detected."""
    call_count = 0

    async def anomaly_benchmark():
        nonlocal call_count
        call_count += 1
        # Third run is anomalous
        if call_count == 3:
            return {"accuracy": 0.1}  # Way below others
        return {"accuracy": 0.9 + random.random() * 0.05}

    harness = MultiRunHarness(num_runs=5, anomaly_threshold=2.0)
    result = await harness.run(anomaly_benchmark)

    assert 2 in result.anomalous_runs
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_benchmark_harness():
    """Run real benchmark multiple times."""
    harness = MultiRunHarness(num_runs=3)

    result = await harness.run(
        run_real_benchmark,
        corpus=test_corpus,
        queries=test_queries,
    )

    # Should have low variance for deterministic operations
    assert result.aggregates["context_recall"].coefficient_of_variation < 0.1
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/harness.py`
- Add tests to `tests/benchmarks/test_harness.py`

---

## Definition of Done

- [ ] MultiRunHarness executes N runs
- [ ] Different seeds per run
- [ ] Mean and std calculated correctly
- [ ] 95% confidence intervals correct
- [ ] Anomaly detection working
- [ ] Seeds recorded for reproducibility
- [ ] Reproducibility test passing
- [ ] Integration with RAGAS evaluator
