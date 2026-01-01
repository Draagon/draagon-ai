# FR-011: CI/CD Infrastructure for draagon-ai

**Status:** Draft
**Priority:** High
**Complexity:** Medium
**Created:** 2026-01-01
**Updated:** 2026-01-01
**Depends On:** FR-009 (Testing Framework), FR-010 (Real Agent Tests)

> **⚠️ DESIGN DECISION: CI Disabled Initially**
>
> All automated workflows are configured for **manual trigger only** (`workflow_dispatch`).
> Automatic triggers (push, schedule) are commented out for now.
>
> **Rationale:** Focus on local testing first. Enable automation when ready.
>
> **To enable later:** Uncomment the `on: push` and `on: schedule` triggers.

---

## Overview

Continuous Integration and Continuous Deployment infrastructure for draagon-ai, including GitHub Actions workflows, test automation, quality gates, and release management. Ensures code quality, prevents regressions, and automates the release process.

**Core Principle:** Automated quality enforcement. No manual release steps. Fast feedback loops for developers.

---

## Motivation

### Current State

draagon-ai has comprehensive tests but lacks automated CI/CD:

- ❌ **No Automated Testing**: Tests run manually on developer machines
- ❌ **No Quality Gates**: PRs can merge without passing tests
- ❌ **No Release Automation**: Releases are manual and error-prone
- ❌ **No Cost Monitoring**: LLM API costs for tests not tracked
- ❌ **No Performance Tracking**: No baseline for latency/throughput
- ❌ **No Multi-Platform Testing**: Only tested on developer's OS

### Problems

1. **Regressions Escape**: Bugs reach production because tests aren't run consistently
2. **Slow Feedback**: Developers don't know if PR breaks tests until manual run
3. **Inconsistent Environment**: Works on dev machine, fails in production
4. **Cost Surprises**: LLM API costs for integration tests unknown
5. **Release Friction**: Manual releases are slow and risky

### Research Basis

| Aspect | Best Practice | Application |
|--------|---------------|-------------|
| **CI/CD** | GitHub Actions, GitLab CI | Automate test runs on every PR |
| **Test Environments** | Docker, containers | Isolate test database, LLM mocks |
| **Quality Gates** | Fail PR if tests fail | Prevent merging broken code |
| **Cost Monitoring** | Track API costs | Alert if LLM costs spike |
| **Semantic Versioning** | semver.org | Automated version bumps |

---

## Requirements

### FR-011.1: GitHub Actions Workflows

**Description:** Automated test execution on pull requests and main branch pushes.

**Workflows:**

1. **PR Validation** (`.github/workflows/pr.yml`)
   - Runs on every pull request
   - Fast feedback (<5 minutes)
   - Validates code quality and unit tests

2. **Integration Tests** (`.github/workflows/integration.yml`)
   - Runs on PR approval or main branch
   - Uses real Neo4j and mock LLM (cost control)
   - Validates end-to-end behavior

3. **Nightly Full Test** (`.github/workflows/nightly.yml`)
   - Runs daily at 2 AM UTC
   - Uses real LLM providers
   - Comprehensive validation

4. **Release** (`.github/workflows/release.yml`)
   - Triggered by version tag push
   - Builds package, runs all tests
   - Publishes to PyPI

**PR Validation Workflow:**

```yaml
# .github/workflows/pr.yml
name: PR Validation

on:
  # DISABLED: Uncomment when ready for automated CI
  # pull_request:
  #   branches: [main]
  workflow_dispatch:  # Manual trigger only for now

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Lint with ruff
        run: |
          ruff check src/ tests/

      - name: Type check with mypy
        run: |
          mypy src/draagon_ai

      - name: Format check with black
        run: |
          black --check src/ tests/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src/draagon_ai --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml

  framework-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.26
        env:
          NEO4J_AUTH: neo4j/test-password
        ports:
          - 7687:7687
        options: >-
          --health-cmd "cypher-shell -u neo4j -p test-password 'RETURN 1'"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run framework tests (mock LLM)
        env:
          NEO4J_TEST_URI: bolt://localhost:7687
          NEO4J_TEST_PASSWORD: test-password
        run: |
          pytest tests/testing/ tests/integration/test_*integration.py -v
```

**Acceptance Criteria:**
- ✅ PR validation runs on every pull request
- ✅ Quality checks (lint, type, format) enforced
- ✅ Unit tests run with coverage reporting
- ✅ Framework tests run with Neo4j
- ✅ PR blocked if any job fails
- ✅ Workflow completes in <10 minutes

---

### FR-011.2: Integration Test Workflow

**Description:** Run real agent integration tests with cost controls.

**Strategy:**
- PR validation: Use **mock LLM** (free, fast)
- Main branch: Use **cached LLM** (reduced cost)
- Nightly: Use **real LLM** (full validation)

**Integration Test Workflow:**

```yaml
# .github/workflows/integration.yml
name: Integration Tests

on:
  # DISABLED: Uncomment when ready for automated CI
  # push:
  #   branches: [main]
  workflow_dispatch:  # Manual trigger only for now

jobs:
  agent-tests:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.26
        env:
          NEO4J_AUTH: neo4j/test-password
        ports:
          - 7687:7687

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,integration]"

      - name: Run agent integration tests (mock LLM)
        env:
          NEO4J_TEST_URI: bolt://localhost:7687
          NEO4J_TEST_PASSWORD: test-password
          USE_MOCK_LLM: "true"  # Cost control
        run: |
          pytest tests/integration/agents/ -v -m "not slow"

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: test-results
          path: test-reports/

  performance-baseline:
    runs-on: ubuntu-latest
    needs: agent-tests
    steps:
      - uses: actions/checkout@v4

      - name: Run performance tests
        run: |
          pytest tests/performance/ -v --benchmark-only

      - name: Compare to baseline
        run: |
          python scripts/compare_performance.py \
            --current test-reports/benchmark.json \
            --baseline benchmarks/baseline.json \
            --threshold 0.1  # Fail if 10% slower
```

**Acceptance Criteria:**
- ✅ Integration tests run on main branch pushes
- ✅ Mock LLM used for cost control
- ✅ Performance baselines tracked
- ✅ Test results uploaded as artifacts
- ✅ Workflow completes in <15 minutes

---

### FR-011.3: Nightly Full Validation

**Description:** Comprehensive testing with real LLM providers on a daily schedule.

**Purpose:**
- Validate against real LLM behavior (catches API changes)
- Test full feature set including expensive operations
- Generate metrics for monitoring

**Nightly Workflow:**

```yaml
# .github/workflows/nightly.yml
name: Nightly Full Validation

on:
  # DISABLED: Uncomment when ready for automated CI
  # schedule:
  #   - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:  # Manual trigger only for now

jobs:
  full-test-suite:
    runs-on: ubuntu-latest
    services:
      neo4j:
        image: neo4j:5.26
        env:
          NEO4J_AUTH: neo4j/test-password
        ports:
          - 7687:7687

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev,integration]"

      - name: Run all tests with real LLM
        env:
          NEO4J_TEST_URI: bolt://localhost:7687
          NEO4J_TEST_PASSWORD: test-password
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
          USE_MOCK_LLM: "false"
        run: |
          pytest tests/ -v --slow \
            --cov=src/draagon_ai \
            --cov-report=html \
            --junit-xml=test-reports/junit.xml

      - name: Track LLM costs
        run: |
          python scripts/track_llm_costs.py \
            --report test-reports/llm-costs.json \
            --alert-threshold 5.00  # Alert if >$5

      - name: Upload coverage to Codecov
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml

      - name: Notify on failure
        if: failure()
        uses: 8398a7/action-slack@v3
        with:
          status: ${{ job.status }}
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
          text: 'Nightly tests failed! Check logs.'

  matrix-testing:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest, windows-latest]
        python-version: ['3.11', '3.12']

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Run tests
        run: |
          pip install -e ".[dev]"
          pytest tests/unit/ -v
```

**Acceptance Criteria:**
- ✅ Runs daily at scheduled time
- ✅ Uses real LLM providers
- ✅ Tests across OS/Python versions
- ✅ Tracks LLM API costs
- ✅ Notifies team on failure
- ✅ Uploads comprehensive coverage

---

### FR-011.4: Release Automation

**Description:** Automated package building and PyPI publishing on version tags.

**Release Process:**

1. Developer updates version in `pyproject.toml`
2. Commits with message: `chore: bump version to X.Y.Z`
3. Tags commit: `git tag vX.Y.Z`
4. Pushes tag: `git push --tags`
5. GitHub Action builds and publishes

**Release Workflow:**

```yaml
# .github/workflows/release.yml
name: Release

on:
  # DISABLED: Uncomment when ready for automated CI
  # push:
  #   tags:
  #     - 'v*.*.*'
  workflow_dispatch:  # Manual trigger only for now

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -e ".[dev]"

      - name: Run full test suite
        env:
          NEO4J_TEST_URI: bolt://localhost:7687
        run: |
          pytest tests/ -v

  build:
    needs: validate
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install build tools
        run: |
          pip install build twine

      - name: Build package
        run: |
          python -m build

      - name: Check package
        run: |
          twine check dist/*

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: dist
          path: dist/

  publish:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # For trusted publishing
    steps:
      - uses: actions/download-artifact@v4
        with:
          name: dist
          path: dist/

      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}

  create-release:
    needs: publish
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate changelog
        id: changelog
        run: |
          git log $(git describe --tags --abbrev=0 HEAD^)..HEAD \
            --pretty=format:"- %s" > CHANGELOG.txt

      - name: Create GitHub Release
        uses: actions/create-release@v1
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          body_path: CHANGELOG.txt
          draft: false
          prerelease: false
```

**Acceptance Criteria:**
- ✅ Triggered by version tag push
- ✅ Runs full test suite before building
- ✅ Builds Python package
- ✅ Publishes to PyPI automatically
- ✅ Creates GitHub release with changelog
- ✅ Fails gracefully if tests fail

---

### FR-011.5: Quality Gates & Branch Protection

**Description:** Enforce quality standards via GitHub branch protection rules.

**Branch Protection Rules for `main`:**

```yaml
# Settings > Branches > Branch protection rules
Branch: main

Protections:
  ✅ Require pull request reviews before merging
    - Required approving reviews: 1
    - Dismiss stale PR approvals when new commits are pushed
    - Require review from Code Owners

  ✅ Require status checks to pass before merging
    - Required checks:
      - quality (lint, type, format)
      - unit-tests
      - framework-tests
    - Require branches to be up to date before merging

  ✅ Require conversation resolution before merging

  ✅ Require signed commits

  ❌ Do not allow force pushes
  ❌ Do not allow deletions
```

**Code Owners:**

```
# .github/CODEOWNERS
* @doug

# Require architecture review for core changes
/src/draagon_ai/orchestration/ @doug
/src/draagon_ai/memory/ @doug
/src/draagon_ai/cognition/ @doug

# Require testing expertise for test infrastructure
/tests/testing/ @doug
/.github/workflows/ @doug
```

**Acceptance Criteria:**
- ✅ Main branch protected from direct pushes
- ✅ PRs require passing status checks
- ✅ PRs require code review approval
- ✅ PRs require up-to-date branch
- ✅ Force pushes disabled

---

### FR-011.6: Cost Monitoring & Alerts

**Description:** Track and alert on LLM API costs for integration tests.

**Cost Tracking Script:**

```python
# scripts/track_llm_costs.py

import json
import sys
from pathlib import Path

def estimate_costs(test_report: Path, alert_threshold: float):
    """Estimate LLM API costs from test run.

    Args:
        test_report: Path to test report with LLM call counts
        alert_threshold: Dollar amount to trigger alert
    """
    with open(test_report) as f:
        report = json.load(f)

    # Estimate costs based on call counts
    costs = {
        "groq_calls": report.get("groq_calls", 0) * 0.0001,  # $0.0001 per call
        "evaluator_calls": report.get("evaluator_calls", 0) * 0.0002,
    }

    total_cost = sum(costs.values())

    print(f"Estimated LLM costs: ${total_cost:.4f}")
    print(json.dumps(costs, indent=2))

    if total_cost > alert_threshold:
        print(f"⚠️  ALERT: Cost ${total_cost:.2f} exceeds threshold ${alert_threshold:.2f}")
        sys.exit(1)

    # Save for trending
    output = {
        "date": datetime.now().isoformat(),
        "total_cost": total_cost,
        "breakdown": costs,
    }

    history_file = Path("test-reports/cost-history.jsonl")
    with open(history_file, "a") as f:
        f.write(json.dumps(output) + "\n")
```

**Cost Alert Workflow:**

```yaml
# .github/workflows/cost-alert.yml
name: Weekly Cost Report

on:
  # DISABLED: Uncomment when ready for automated CI
  # schedule:
  #   - cron: '0 9 * * 1'  # Monday 9 AM UTC
  workflow_dispatch:  # Manual trigger only for now

jobs:
  cost-report:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Generate cost report
        run: |
          python scripts/generate_cost_report.py \
            --input test-reports/cost-history.jsonl \
            --output cost-report.md

      - name: Post to Slack
        uses: 8398a7/action-slack@v3
        with:
          status: custom
          custom_payload: |
            {
              text: 'Weekly LLM Cost Report',
              attachments: [{
                color: 'good',
                text: '${{ steps.report.outputs.summary }}'
              }]
            }
          webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

**Acceptance Criteria:**
- ✅ Tracks LLM API costs per test run
- ✅ Alerts if costs exceed threshold
- ✅ Generates weekly cost reports
- ✅ Trends tracked in JSON history

---

### FR-011.7: Performance Benchmarking

**Description:** Track performance baselines and detect regressions.

**Benchmark Tests:**

```python
# tests/performance/test_benchmarks.py

import pytest
from draagon_ai.orchestration import AgentLoop

@pytest.mark.benchmark
def test_simple_query_latency(benchmark, agent):
    """Benchmark simple query processing."""

    def process_query():
        return asyncio.run(agent.process("What is 2+2?"))

    result = benchmark(process_query)

    # Assert latency < 2s
    assert result.stats.mean < 2.0

@pytest.mark.benchmark
def test_memory_search_latency(benchmark, memory_provider):
    """Benchmark memory search."""

    def search():
        return asyncio.run(memory_provider.search("cats", limit=10))

    result = benchmark(search)

    # Assert latency < 500ms
    assert result.stats.mean < 0.5
```

**Benchmark Comparison:**

```python
# scripts/compare_performance.py

import json
import sys
from pathlib import Path

def compare_benchmarks(current: Path, baseline: Path, threshold: float):
    """Compare benchmark results to baseline.

    Args:
        current: Current benchmark results
        baseline: Baseline results to compare against
        threshold: % regression allowed (0.1 = 10%)
    """
    with open(current) as f:
        current_data = json.load(f)

    with open(baseline) as f:
        baseline_data = json.load(f)

    regressions = []

    for test_name, current_stats in current_data.items():
        if test_name not in baseline_data:
            continue

        baseline_stats = baseline_data[test_name]

        # Check if slower
        regression = (current_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]

        if regression > threshold:
            regressions.append({
                "test": test_name,
                "regression_pct": regression * 100,
                "current_mean": current_stats["mean"],
                "baseline_mean": baseline_stats["mean"],
            })

    if regressions:
        print("⚠️  Performance regressions detected:")
        for r in regressions:
            print(f"  {r['test']}: {r['regression_pct']:.1f}% slower")
        sys.exit(1)

    print("✅ Performance within acceptable range")
```

**Acceptance Criteria:**
- ✅ Benchmark tests track key operations
- ✅ Baseline results stored in repo
- ✅ CI compares to baseline
- ✅ Fails if regression > threshold
- ✅ Reports show latency trends

---

## Infrastructure Requirements

### GitHub Secrets

Configure these secrets in repository settings:

| Secret | Purpose | Where to Get |
|--------|---------|--------------|
| `GROQ_API_KEY` | Real LLM for nightly tests | https://console.groq.com |
| `OPENAI_API_KEY` | Alternative LLM provider | https://platform.openai.com |
| `PYPI_API_TOKEN` | Publish to PyPI | https://pypi.org/manage/account/token/ |
| `SLACK_WEBHOOK` | Test failure notifications | Slack App > Incoming Webhooks |
| `CODECOV_TOKEN` | Coverage reporting | https://codecov.io |

### Repository Settings

```yaml
# .github/settings.yml (via GitHub App)
repository:
  name: draagon-ai
  description: Agentic AI framework with cognitive capabilities
  homepage: https://github.com/yourusername/draagon-ai
  topics:
    - ai
    - agents
    - cognitive-architecture
    - llm
    - memory

  has_issues: true
  has_projects: true
  has_wiki: false
  has_downloads: true

  default_branch: main

  allow_squash_merge: true
  allow_merge_commit: false
  allow_rebase_merge: false

  delete_branch_on_merge: true
```

---

## Success Criteria

### Quantitative Metrics

- **PR Validation Time**: <10 minutes (fail fast)
- **Integration Test Time**: <15 minutes (on main)
- **Nightly Test Time**: <60 minutes (comprehensive)
- **Test Flakiness**: <2% flake rate
- **LLM Cost per PR**: <$0.10 (mock LLM)
- **LLM Cost per Nightly**: <$2.00 (real LLM)
- **Release Time**: <30 minutes (tag to PyPI)

### Qualitative Metrics

- **Developer Experience**: Fast feedback, clear error messages
- **Reliability**: No false positives, consistent results
- **Cost Transparency**: Clear LLM cost tracking
- **Ease of Debugging**: Test artifacts available for failed runs

---

## Implementation Phases

### Phase 1: Basic CI (1 day)
- [ ] Create PR validation workflow
- [ ] Add linting, type checking, formatting
- [ ] Run unit tests on PR
- [ ] Set up code coverage

### Phase 2: Integration Tests (2 days)
- [ ] Add Neo4j service to workflows
- [ ] Run framework tests with mock LLM
- [ ] Upload test artifacts
- [ ] Configure branch protection

### Phase 3: Nightly Testing (1 day)
- [ ] Create nightly workflow with real LLM
- [ ] Add cost tracking script
- [ ] Set up Slack notifications
- [ ] Add multi-platform matrix

### Phase 4: Release Automation (1 day)
- [ ] Create release workflow
- [ ] Set up PyPI publishing
- [ ] Add changelog generation
- [ ] Test release process

### Phase 5: Performance & Monitoring (1 day)
- [ ] Add benchmark tests
- [ ] Create baseline results
- [ ] Add performance comparison
- [ ] Set up weekly cost reports

**Total Effort:** ~6 days

---

## Constitution Compliance

✅ **LLM-First Architecture**: N/A (infrastructure, not semantic logic)
✅ **XML Output Format**: N/A (infrastructure)
✅ **Protocol-Based Design**: N/A (infrastructure)
✅ **Pragmatic Async**: Workflows run async tasks appropriately
✅ **Test Outcomes, Not Processes**: CI validates test outcomes (pass/fail)
✅ **Research-Grounded**: Follows industry best practices for CI/CD

---

## Open Questions

[NEEDS CLARIFICATION: Should we use GitHub-hosted runners or self-hosted runners?]

**Recommendation:** Start with GitHub-hosted (simpler, no maintenance). Switch to self-hosted if need GPU or heavy compute.

[NEEDS CLARIFICATION: What's the target LLM cost budget for nightly tests?]

**Recommendation:** $2/night = $60/month. Set alerts at $5/run.

---

## Related Work

- **FR-009**: Integration Testing Framework (tested by this CI/CD)
- **FR-010**: Real Agent Integration Tests (run by workflows)

---

**Status:** Ready for implementation (workflows disabled initially)
**Next Steps:** Focus on FR-010 real agent tests first. Enable CI workflows when local testing is validated.
