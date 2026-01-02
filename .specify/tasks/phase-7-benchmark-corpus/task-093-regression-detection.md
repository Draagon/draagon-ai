# TASK-093: Regression Detection Setup

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Quality gate)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-091, TASK-092

---

## Description

Implement comprehensive regression detection for benchmark metrics:
- Threshold-based detection (> 5% drop fails)
- Trend analysis (3+ consecutive drops = warning)
- Statistical significance (don't fail on noise)
- Automated alerts and issue creation
- Baseline management (update on releases)

This ensures we never ship a performance regression.

**Location:** `src/draagon_ai/testing/benchmarks/regression.py`

---

## Acceptance Criteria

### Detection Methods
- [ ] Threshold detection: fail if metric drops > X%
- [ ] Trend detection: warn on 3+ consecutive drops
- [ ] Statistical detection: compare to rolling average ± 2σ
- [ ] Category-specific thresholds (stricter for core metrics)

### Thresholds
- [ ] Faithfulness: fail if < 0.80 or drops > 5%
- [ ] Context Recall: fail if < 0.80 or drops > 5%
- [ ] nDCG@10: fail if drops > 10%
- [ ] Success Rate: fail if < 0.95

### Baseline Management
- [ ] Baseline stored in `.specify/benchmarks/baselines/`
- [ ] Versioned baselines (per release)
- [ ] Auto-update on main merge (if passing)
- [ ] Manual baseline update command

### Alerting
- [ ] GitHub issue creation on regression
- [ ] Slack notification (optional)
- [ ] Email to maintainers (optional)
- [ ] Summary in PR comment

---

## Technical Notes

### Regression Detector

```python
from dataclasses import dataclass
from typing import Optional
from enum import Enum

class RegressionSeverity(str, Enum):
    INFO = "info"        # Minor fluctuation
    WARNING = "warning"  # Concerning trend
    ERROR = "error"      # Threshold breach
    CRITICAL = "critical"  # Core metric failure


@dataclass
class RegressionThreshold:
    metric: str
    min_absolute: Optional[float] = None  # Must be >= this value
    max_drop_percent: float = 0.05         # Max allowed drop from baseline
    severity: RegressionSeverity = RegressionSeverity.ERROR


# Default thresholds
DEFAULT_THRESHOLDS = [
    RegressionThreshold(
        metric="faithfulness",
        min_absolute=0.80,
        max_drop_percent=0.05,
        severity=RegressionSeverity.CRITICAL,
    ),
    RegressionThreshold(
        metric="context_recall",
        min_absolute=0.80,
        max_drop_percent=0.05,
        severity=RegressionSeverity.CRITICAL,
    ),
    RegressionThreshold(
        metric="ndcg_at_10",
        min_absolute=0.50,
        max_drop_percent=0.10,
        severity=RegressionSeverity.ERROR,
    ),
    RegressionThreshold(
        metric="success_rate",
        min_absolute=0.95,
        max_drop_percent=0.02,
        severity=RegressionSeverity.CRITICAL,
    ),
]


@dataclass
class RegressionReport:
    metric: str
    current_value: float
    baseline_value: float
    threshold: RegressionThreshold
    drop_percent: float
    is_regression: bool
    severity: RegressionSeverity
    message: str


class RegressionDetector:
    def __init__(
        self,
        thresholds: list[RegressionThreshold] = None,
        history_path: Path = None,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.history_path = history_path
        self._history: list[dict] = []

        if history_path and history_path.exists():
            self._history = json.loads(history_path.read_text())

    def check(
        self,
        current: dict[str, float],
        baseline: dict[str, float],
    ) -> list[RegressionReport]:
        """Check for regressions against baseline."""
        reports = []

        for threshold in self.thresholds:
            metric = threshold.metric
            current_val = current.get(metric)
            baseline_val = baseline.get(metric)

            if current_val is None:
                continue

            report = self._check_threshold(threshold, current_val, baseline_val)
            reports.append(report)

        return reports

    def _check_threshold(
        self,
        threshold: RegressionThreshold,
        current: float,
        baseline: Optional[float],
    ) -> RegressionReport:
        """Check single metric against threshold."""
        is_regression = False
        message = ""

        # Check absolute minimum
        if threshold.min_absolute and current < threshold.min_absolute:
            is_regression = True
            message = f"Below minimum: {current:.3f} < {threshold.min_absolute}"

        # Check percent drop
        if baseline and baseline > 0:
            drop = (baseline - current) / baseline
            if drop > threshold.max_drop_percent:
                is_regression = True
                message = f"Dropped {drop*100:.1f}% (max allowed: {threshold.max_drop_percent*100:.0f}%)"
        else:
            drop = 0

        if not is_regression:
            message = "OK"

        return RegressionReport(
            metric=threshold.metric,
            current_value=current,
            baseline_value=baseline or 0,
            threshold=threshold,
            drop_percent=drop if baseline else 0,
            is_regression=is_regression,
            severity=threshold.severity if is_regression else RegressionSeverity.INFO,
            message=message,
        )

    def check_trend(self, metric: str, window: int = 5) -> Optional[str]:
        """Check for concerning trends in history."""
        if len(self._history) < window:
            return None

        recent = [h.get("metrics", {}).get(metric, 0) for h in self._history[-window:]]

        # Check for consecutive drops
        drops = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
        if drops >= window - 1:
            return f"Warning: {metric} has dropped for {drops} consecutive runs"

        # Check for downward trend
        from scipy import stats
        x = list(range(len(recent)))
        slope, _, r_value, _, _ = stats.linregress(x, recent)

        if slope < -0.01 and abs(r_value) > 0.7:
            return f"Warning: {metric} shows downward trend (slope: {slope:.4f})"

        return None

    def check_statistical_significance(
        self,
        metric: str,
        current: float,
        window: int = 10,
        sigma: float = 2.0,
    ) -> Optional[str]:
        """Check if current value is statistically anomalous."""
        if len(self._history) < window:
            return None

        recent_values = [
            h.get("metrics", {}).get(metric, 0)
            for h in self._history[-window:]
        ]

        mean = statistics.mean(recent_values)
        std = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

        if std > 0:
            z_score = (current - mean) / std
            if z_score < -sigma:
                return f"Anomaly: {metric} is {abs(z_score):.1f}σ below rolling mean"

        return None

    def generate_report(
        self,
        regressions: list[RegressionReport],
        trends: list[str],
        anomalies: list[str],
    ) -> str:
        """Generate human-readable regression report."""
        lines = ["# Regression Analysis", ""]

        # Summary
        critical = [r for r in regressions if r.severity == RegressionSeverity.CRITICAL and r.is_regression]
        errors = [r for r in regressions if r.severity == RegressionSeverity.ERROR and r.is_regression]

        if critical:
            lines.append("## 🚨 CRITICAL REGRESSIONS")
            for r in critical:
                lines.append(f"- **{r.metric}**: {r.message}")
            lines.append("")

        if errors:
            lines.append("## ❌ Regressions")
            for r in errors:
                lines.append(f"- **{r.metric}**: {r.message}")
            lines.append("")

        if trends:
            lines.append("## ⚠️ Concerning Trends")
            for t in trends:
                lines.append(f"- {t}")
            lines.append("")

        if anomalies:
            lines.append("## 📊 Statistical Anomalies")
            for a in anomalies:
                lines.append(f"- {a}")
            lines.append("")

        # All metrics table
        lines.append("## All Metrics")
        lines.append("| Metric | Current | Baseline | Change | Status |")
        lines.append("|--------|---------|----------|--------|--------|")

        for r in regressions:
            status = "✅" if not r.is_regression else "❌" if r.severity in [RegressionSeverity.CRITICAL, RegressionSeverity.ERROR] else "⚠️"
            change = f"{-r.drop_percent*100:+.1f}%" if r.baseline_value else "N/A"
            lines.append(f"| {r.metric} | {r.current_value:.3f} | {r.baseline_value:.3f} | {change} | {status} |")

        return "\n".join(lines)
```

### Baseline Manager

```python
class BaselineManager:
    def __init__(self, baseline_dir: Path):
        self.baseline_dir = baseline_dir
        self.baseline_dir.mkdir(parents=True, exist_ok=True)

    def get_current_baseline(self) -> dict[str, float]:
        """Get current active baseline."""
        current_path = self.baseline_dir / "current.json"
        if current_path.exists():
            return json.loads(current_path.read_text())
        return {}

    def update_baseline(self, metrics: dict[str, float], version: str = None):
        """Update baseline with new metrics."""
        timestamp = datetime.now().isoformat()
        version = version or datetime.now().strftime("%Y%m%d")

        baseline = {
            "version": version,
            "timestamp": timestamp,
            "metrics": metrics,
        }

        # Save versioned
        versioned_path = self.baseline_dir / f"baseline_{version}.json"
        versioned_path.write_text(json.dumps(baseline, indent=2))

        # Update current
        current_path = self.baseline_dir / "current.json"
        current_path.write_text(json.dumps(metrics, indent=2))

        logger.info(f"Updated baseline to version {version}")

    def list_baselines(self) -> list[str]:
        """List all saved baselines."""
        return sorted([
            p.stem.replace("baseline_", "")
            for p in self.baseline_dir.glob("baseline_*.json")
        ])

    def rollback(self, version: str):
        """Rollback to a previous baseline."""
        versioned_path = self.baseline_dir / f"baseline_{version}.json"
        if not versioned_path.exists():
            raise ValueError(f"Baseline version {version} not found")

        data = json.loads(versioned_path.read_text())
        current_path = self.baseline_dir / "current.json"
        current_path.write_text(json.dumps(data["metrics"], indent=2))

        logger.info(f"Rolled back to baseline {version}")
```

### GitHub Issue Creator

```python
def create_regression_issue(
    regressions: list[RegressionReport],
    repo: str,
    token: str,
):
    """Create GitHub issue for regressions."""
    import requests

    critical = [r for r in regressions if r.is_regression and r.severity == RegressionSeverity.CRITICAL]

    if not critical:
        return

    title = f"🚨 Critical Benchmark Regression Detected"

    body = f"""## Regression Report

The following critical regressions were detected in the latest benchmark:

| Metric | Current | Baseline | Drop |
|--------|---------|----------|------|
"""

    for r in critical:
        body += f"| {r.metric} | {r.current_value:.3f} | {r.baseline_value:.3f} | {r.drop_percent*100:.1f}% |\n"

    body += """

## Action Required

1. Investigate recent changes that may have caused this regression
2. Run benchmarks locally to reproduce
3. Fix the issue or update the baseline if intentional

/cc @maintainers
"""

    response = requests.post(
        f"https://api.github.com/repos/{repo}/issues",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json",
        },
        json={
            "title": title,
            "body": body,
            "labels": ["benchmark", "regression", "priority:critical"],
        },
    )

    response.raise_for_status()
    logger.info(f"Created issue: {response.json()['html_url']}")
```

---

## Testing Requirements

### Unit Tests
```python
def test_threshold_regression_detected():
    """Regression detected when threshold exceeded."""
    detector = RegressionDetector()

    current = {"faithfulness": 0.75}
    baseline = {"faithfulness": 0.85}

    reports = detector.check(current, baseline)
    faithfulness_report = next(r for r in reports if r.metric == "faithfulness")

    assert faithfulness_report.is_regression
    assert faithfulness_report.severity == RegressionSeverity.CRITICAL

def test_no_false_positive():
    """Normal fluctuation doesn't trigger regression."""
    detector = RegressionDetector()

    current = {"faithfulness": 0.82}
    baseline = {"faithfulness": 0.84}  # 2.4% drop, below 5% threshold

    reports = detector.check(current, baseline)
    faithfulness_report = next(r for r in reports if r.metric == "faithfulness")

    assert not faithfulness_report.is_regression

def test_trend_detection():
    """Consecutive drops detected."""
    history = [
        {"metrics": {"faithfulness": 0.90}},
        {"metrics": {"faithfulness": 0.88}},
        {"metrics": {"faithfulness": 0.86}},
        {"metrics": {"faithfulness": 0.84}},
        {"metrics": {"faithfulness": 0.82}},
    ]

    detector = RegressionDetector()
    detector._history = history

    warning = detector.check_trend("faithfulness", window=5)
    assert warning is not None
    assert "consecutive" in warning or "trend" in warning
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/regression.py`
- `src/draagon_ai/testing/benchmarks/baseline.py`
- `.specify/benchmarks/baselines/` directory
- Integration with GitHub Actions
- Add tests to `tests/benchmarks/test_regression.py`

---

## Definition of Done

- [ ] Threshold-based regression detection
- [ ] Trend analysis (consecutive drops)
- [ ] Statistical anomaly detection
- [ ] Baseline management (versions, rollback)
- [ ] GitHub issue creation
- [ ] Human-readable reports
- [ ] Integration with CI/CD
- [ ] Tests passing
- [ ] Documentation complete
