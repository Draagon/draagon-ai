# TASK-084: Statistical Reporting

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Makes results interpretable)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-083 (Multiple-Run Harness)

---

## Description

Generate publication-quality statistical reports from benchmark results:
- Tables with mean ± std, confidence intervals
- Comparison tables (vs baselines)
- Significance testing results
- Export to Markdown, CSV, JSON

Reports should be ready for documentation and external review.

**Location:** `src/draagon_ai/testing/benchmarks/reporting.py`

---

## Acceptance Criteria

### Report Formats
- [ ] Markdown tables for documentation
- [ ] CSV for spreadsheet analysis
- [ ] JSON for programmatic access
- [ ] Plain text summary for terminal

### Statistical Content
- [ ] Metric means with confidence intervals
- [ ] Standard deviation and coefficient of variation
- [ ] Per-category breakdown (if applicable)
- [ ] Comparison with baselines (% improvement)

### Significance Testing
- [ ] Paired t-test for mean differences
- [ ] Wilcoxon signed-rank for non-parametric
- [ ] p-values and significance stars (* p<0.05, ** p<0.01, *** p<0.001)
- [ ] Effect size (Cohen's d)

### Visualization Helpers
- [ ] ASCII box plots for terminal
- [ ] Data formatted for matplotlib/plotly
- [ ] Trend data for historical tracking

---

## Technical Notes

### Report Generator

```python
from dataclasses import dataclass
from typing import Literal
from pathlib import Path
import json
import csv

ReportFormat = Literal["markdown", "csv", "json", "text"]


@dataclass
class ComparisonResult:
    baseline_name: str
    metric_name: str
    baseline_mean: float
    our_mean: float
    improvement_pct: float
    p_value: float
    effect_size: float
    significant: bool

    @property
    def significance_stars(self) -> str:
        if self.p_value < 0.001:
            return "***"
        elif self.p_value < 0.01:
            return "**"
        elif self.p_value < 0.05:
            return "*"
        return ""


class StatisticalReporter:
    def __init__(self, harness_result: HarnessResult):
        self.result = harness_result

    def generate_summary_table(self, format: ReportFormat = "markdown") -> str:
        """Generate summary table of all metrics."""
        if format == "markdown":
            return self._markdown_summary()
        elif format == "csv":
            return self._csv_summary()
        elif format == "json":
            return self._json_summary()
        else:
            return self._text_summary()

    def _markdown_summary(self) -> str:
        lines = [
            "| Metric | Mean | Std | 95% CI | Min | Max |",
            "|--------|------|-----|--------|-----|-----|",
        ]

        for name, agg in sorted(self.result.aggregates.items()):
            lines.append(
                f"| {name} | {agg.mean:.4f} | {agg.std:.4f} | "
                f"[{agg.ci_95_lower:.4f}, {agg.ci_95_upper:.4f}] | "
                f"{agg.min_value:.4f} | {agg.max_value:.4f} |"
            )

        return "\n".join(lines)

    def _csv_summary(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Metric", "Mean", "Std", "CI_Lower", "CI_Upper", "Min", "Max", "CV"])

        for name, agg in sorted(self.result.aggregates.items()):
            writer.writerow([
                name, agg.mean, agg.std, agg.ci_95_lower, agg.ci_95_upper,
                agg.min_value, agg.max_value, agg.coefficient_of_variation
            ])

        return output.getvalue()

    def _json_summary(self) -> str:
        data = {
            "num_runs": self.result.num_runs,
            "metrics": {
                name: {
                    "mean": agg.mean,
                    "std": agg.std,
                    "ci_95": [agg.ci_95_lower, agg.ci_95_upper],
                    "range": [agg.min_value, agg.max_value],
                    "cv": agg.coefficient_of_variation,
                    "values": agg.values,
                }
                for name, agg in self.result.aggregates.items()
            },
            "anomalous_runs": self.result.anomalous_runs,
            "duration_seconds": self.result.total_duration_seconds,
        }
        return json.dumps(data, indent=2)

    def generate_comparison_table(
        self,
        baseline_results: dict[str, HarnessResult],
    ) -> str:
        """Generate comparison table against baselines."""
        lines = [
            "| Metric | Ours | BM25 | Contriever | vs BM25 | vs Contriever |",
            "|--------|------|------|------------|---------|---------------|",
        ]

        for name, our_agg in sorted(self.result.aggregates.items()):
            row = f"| {name} | {our_agg.mean:.4f} |"

            for baseline_name in ["BM25", "Contriever"]:
                if baseline_name in baseline_results:
                    baseline_agg = baseline_results[baseline_name].aggregates.get(name)
                    if baseline_agg:
                        row += f" {baseline_agg.mean:.4f} |"
                    else:
                        row += " - |"
                else:
                    row += " - |"

            # Calculate improvements
            for baseline_name in ["BM25", "Contriever"]:
                if baseline_name in baseline_results:
                    baseline_agg = baseline_results[baseline_name].aggregates.get(name)
                    if baseline_agg and baseline_agg.mean > 0:
                        improvement = (our_agg.mean - baseline_agg.mean) / baseline_agg.mean * 100
                        comparison = self._compare_to_baseline(our_agg, baseline_agg)
                        row += f" {improvement:+.1f}%{comparison.significance_stars} |"
                    else:
                        row += " - |"
                else:
                    row += " - |"

            lines.append(row)

        return "\n".join(lines)

    def _compare_to_baseline(
        self,
        our_agg: AggregateResult,
        baseline_agg: AggregateResult,
    ) -> ComparisonResult:
        """Statistical comparison between our results and baseline."""
        from scipy import stats

        # Paired t-test
        t_stat, p_value = stats.ttest_rel(our_agg.values, baseline_agg.values)

        # Effect size (Cohen's d)
        pooled_std = ((our_agg.std ** 2 + baseline_agg.std ** 2) / 2) ** 0.5
        effect_size = (our_agg.mean - baseline_agg.mean) / pooled_std if pooled_std > 0 else 0

        improvement = (our_agg.mean - baseline_agg.mean) / baseline_agg.mean * 100

        return ComparisonResult(
            baseline_name="",
            metric_name=our_agg.metric_name,
            baseline_mean=baseline_agg.mean,
            our_mean=our_agg.mean,
            improvement_pct=improvement,
            p_value=p_value,
            effect_size=effect_size,
            significant=p_value < 0.05,
        )

    def generate_ascii_boxplot(self, metric_name: str, width: int = 50) -> str:
        """Generate ASCII box plot for terminal display."""
        agg = self.result.aggregates.get(metric_name)
        if not agg:
            return f"No data for {metric_name}"

        values = sorted(agg.values)
        q1 = np.percentile(values, 25)
        q2 = np.percentile(values, 50)
        q3 = np.percentile(values, 75)
        min_val = min(values)
        max_val = max(values)

        # Scale to width
        range_val = max_val - min_val
        if range_val == 0:
            return f"{metric_name}: {min_val:.4f} (no variance)"

        def scale(v):
            return int((v - min_val) / range_val * (width - 2))

        # Build box plot
        plot = [" "] * width
        plot[0] = "|"
        plot[-1] = "|"
        plot[scale(q1)] = "["
        plot[scale(q3)] = "]"
        plot[scale(q2)] = "|"

        for i in range(scale(q1), scale(q3) + 1):
            if plot[i] == " ":
                plot[i] = "─"

        return f"{metric_name}: {''.join(plot)} [{min_val:.3f} - {max_val:.3f}]"
```

### Report File Writer

```python
def save_reports(
    result: HarnessResult,
    output_dir: Path,
    baseline_results: dict[str, HarnessResult] = None,
):
    """Save all report formats to directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reporter = StatisticalReporter(result)

    # Summary table
    (output_dir / "summary.md").write_text(reporter.generate_summary_table("markdown"))
    (output_dir / "summary.csv").write_text(reporter.generate_summary_table("csv"))
    (output_dir / "results.json").write_text(reporter.generate_summary_table("json"))

    # Comparison table (if baselines provided)
    if baseline_results:
        comparison = reporter.generate_comparison_table(baseline_results)
        (output_dir / "comparison.md").write_text(comparison)

    # Full report with all details
    full_report = f"""# Benchmark Results

## Summary
{reporter.generate_summary_table("markdown")}

## Run Details
- Number of runs: {result.num_runs}
- Total duration: {result.total_duration_seconds:.1f}s
- Anomalous runs: {result.anomalous_runs or "None"}

## Box Plots
```
{chr(10).join(reporter.generate_ascii_boxplot(name) for name in result.aggregates)}
```
"""
    (output_dir / "report.md").write_text(full_report)
```

---

## Testing Requirements

### Unit Tests
```python
def test_markdown_table_generation():
    """Markdown table formatted correctly."""
    result = HarnessResult(
        num_runs=3,
        aggregates={
            "accuracy": AggregateResult.from_values("accuracy", [0.85, 0.87, 0.86]),
        },
        ...
    )

    reporter = StatisticalReporter(result)
    table = reporter.generate_summary_table("markdown")

    assert "| Metric |" in table
    assert "| accuracy |" in table
    assert "0.86" in table  # Mean

def test_significance_stars():
    """Significance stars assigned correctly."""
    assert ComparisonResult(p_value=0.04, ...).significance_stars == "*"
    assert ComparisonResult(p_value=0.009, ...).significance_stars == "**"
    assert ComparisonResult(p_value=0.0009, ...).significance_stars == "***"
    assert ComparisonResult(p_value=0.1, ...).significance_stars == ""

def test_csv_export():
    """CSV export is valid."""
    result = HarnessResult(...)
    reporter = StatisticalReporter(result)
    csv_data = reporter.generate_summary_table("csv")

    reader = csv.reader(io.StringIO(csv_data))
    rows = list(reader)

    assert rows[0] == ["Metric", "Mean", "Std", "CI_Lower", "CI_Upper", "Min", "Max", "CV"]
```

### Integration Test
```python
@pytest.mark.integration
async def test_full_report_generation(tmp_path):
    """Generate all report formats from real benchmark."""
    harness = MultiRunHarness(num_runs=3)
    result = await harness.run(run_benchmark, ...)

    save_reports(result, tmp_path)

    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "results.json").exists()
    assert (tmp_path / "report.md").exists()
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/reporting.py`
- Add tests to `tests/benchmarks/test_reporting.py`

---

## Definition of Done

- [ ] Markdown table generation
- [ ] CSV export
- [ ] JSON export
- [ ] Baseline comparison tables
- [ ] Significance testing with p-values
- [ ] Effect size (Cohen's d)
- [ ] Significance stars
- [ ] ASCII box plots
- [ ] Report file writer
- [ ] All formats tested
