# TASK-092: GitHub Actions Workflow

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (CI/CD automation)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-091 (Smoke Tests)

---

## Description

Create GitHub Actions workflows for automated benchmark execution:
- Smoke tests on every PR (≤ 5 minutes)
- Nightly full benchmark (with historical tracking)
- Manual trigger for full benchmark
- Results published to `.specify/benchmarks/results/`

This automates quality gates and historical tracking.

**Location:** `.github/workflows/`

---

## Acceptance Criteria

### PR Workflow
- [ ] Triggers on pull requests to main
- [ ] Runs smoke tests (50 queries)
- [ ] Fails PR if regression detected (> 5% drop)
- [ ] Posts summary comment on PR
- [ ] ≤ 5 minutes runtime

### Nightly Workflow
- [ ] Runs at 2 AM UTC daily
- [ ] Full benchmark (250 queries, 3 runs)
- [ ] Compares to baselines
- [ ] Commits results to repo
- [ ] Creates issue if regression detected

### Manual Workflow
- [ ] Trigger via workflow_dispatch
- [ ] Configurable: runs, queries, baselines
- [ ] Full or smoke mode
- [ ] Optional baseline update

### Infrastructure
- [ ] Self-hosted runner with GPU (for embeddings)
- [ ] Ollama pre-installed
- [ ] Caching for corpus and models
- [ ] Secrets for API keys

---

## Technical Notes

### PR Smoke Test Workflow

```yaml
# .github/workflows/benchmark-smoke.yml
name: Benchmark Smoke Tests

on:
  pull_request:
    branches: [main]
    paths:
      - 'src/draagon_ai/**'
      - 'tests/**'

concurrency:
  group: benchmark-${{ github.ref }}
  cancel-in-progress: true

jobs:
  smoke-test:
    runs-on: self-hosted  # Need GPU for embeddings
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Cache dependencies
        uses: actions/cache@v4
        with:
          path: |
            ~/.cache/pip
            ~/.cache/huggingface
          key: ${{ runner.os }}-pip-${{ hashFiles('pyproject.toml') }}

      - name: Install dependencies
        run: |
          pip install -e ".[dev,benchmark]"

      - name: Cache corpus
        uses: actions/cache@v4
        with:
          path: data/benchmark_corpus.json
          key: corpus-${{ hashFiles('data/benchmark_corpus.json') }}

      - name: Check Ollama
        run: |
          ollama list | grep mxbai-embed-large || ollama pull mxbai-embed-large

      - name: Run smoke tests
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          python -m draagon_ai.testing.benchmarks smoke \
            --corpus data/benchmark_corpus.json \
            --queries data/benchmark_queries.json \
            --baseline .specify/benchmarks/smoke_baseline.json \
            --output smoke_results.json

      - name: Check for regressions
        id: check
        run: |
          if [ -f smoke_results.json ]; then
            passed=$(jq -r '.passed' smoke_results.json)
            if [ "$passed" = "false" ]; then
              echo "REGRESSION=true" >> $GITHUB_OUTPUT
              echo "::error::Smoke tests detected regression"
            fi
          fi

      - name: Comment on PR
        if: always()
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const results = JSON.parse(fs.readFileSync('smoke_results.json'));

            const status = results.passed ? '✅' : '❌';
            const body = `## Benchmark Smoke Tests ${status}

            | Metric | Score | Baseline | Change |
            |--------|-------|----------|--------|
            | Faithfulness | ${results.metrics.faithfulness.toFixed(3)} | ${results.baseline_metrics.faithfulness.toFixed(3)} | ${((results.metrics.faithfulness - results.baseline_metrics.faithfulness) * 100).toFixed(1)}% |
            | Context Recall | ${results.metrics.context_recall.toFixed(3)} | ${results.baseline_metrics.context_recall.toFixed(3)} | ${((results.metrics.context_recall - results.baseline_metrics.context_recall) * 100).toFixed(1)}% |

            Duration: ${results.duration_seconds.toFixed(1)}s

            ${results.regressions.length > 0 ? '### Regressions\\n' + results.regressions.map(r => '- ' + r).join('\\n') : ''}
            `;

            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: body
            });

      - name: Fail on regression
        if: steps.check.outputs.REGRESSION == 'true'
        run: exit 1
```

### Nightly Full Benchmark

```yaml
# .github/workflows/benchmark-nightly.yml
name: Nightly Benchmark

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:
    inputs:
      runs:
        description: 'Number of runs'
        default: '3'
      full:
        description: 'Full benchmark (true) or smoke (false)'
        default: 'true'

jobs:
  benchmark:
    runs-on: self-hosted
    timeout-minutes: 60

    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.BENCHMARK_TOKEN }}  # For committing results

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev,benchmark]"

      - name: Check Ollama
        run: ollama list | grep mxbai-embed-large || ollama pull mxbai-embed-large

      - name: Run benchmark
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          if [ "${{ github.event.inputs.full || 'true' }}" = "true" ]; then
            python -m draagon_ai.testing.benchmarks full \
              --corpus data/benchmark_corpus.json \
              --queries data/benchmark_queries.json \
              --runs ${{ github.event.inputs.runs || '3' }} \
              --output .specify/benchmarks/results/
          else
            python -m draagon_ai.testing.benchmarks smoke \
              --corpus data/benchmark_corpus.json \
              --queries data/benchmark_queries.json
          fi

      - name: Commit results
        run: |
          git config user.name "GitHub Actions"
          git config user.email "actions@github.com"
          git add .specify/benchmarks/results/
          git commit -m "chore(benchmark): nightly results $(date +%Y-%m-%d)" || true
          git push

      - name: Check for regression
        id: regression
        run: |
          if [ -f .specify/benchmarks/results/latest/regressions.txt ]; then
            echo "REGRESSION=true" >> $GITHUB_OUTPUT
          fi

      - name: Create issue on regression
        if: steps.regression.outputs.REGRESSION == 'true'
        uses: actions/github-script@v7
        with:
          script: |
            const fs = require('fs');
            const regressions = fs.readFileSync('.specify/benchmarks/results/latest/regressions.txt', 'utf8');

            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: '🚨 Benchmark Regression Detected',
              body: `The nightly benchmark detected performance regressions:\n\n\`\`\`\n${regressions}\n\`\`\`\n\nPlease investigate and resolve.`,
              labels: ['benchmark', 'regression', 'priority:high']
            });

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results
          path: .specify/benchmarks/results/latest/
```

### Manual Trigger Workflow

```yaml
# .github/workflows/benchmark-manual.yml
name: Manual Benchmark

on:
  workflow_dispatch:
    inputs:
      mode:
        description: 'Benchmark mode'
        required: true
        default: 'smoke'
        type: choice
        options:
          - smoke
          - full
      runs:
        description: 'Number of runs (full mode only)'
        default: '5'
      update_baseline:
        description: 'Update baseline after run'
        type: boolean
        default: false
      max_queries:
        description: 'Max queries (0 = all)'
        default: '0'

jobs:
  benchmark:
    runs-on: self-hosted
    timeout-minutes: 120

    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -e ".[dev,benchmark]"

      - name: Run benchmark
        env:
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          args="--corpus data/benchmark_corpus.json --queries data/benchmark_queries.json"

          if [ "${{ inputs.mode }}" = "full" ]; then
            args="$args --runs ${{ inputs.runs }}"
          fi

          if [ "${{ inputs.max_queries }}" != "0" ]; then
            args="$args --max-queries ${{ inputs.max_queries }}"
          fi

          if [ "${{ inputs.update_baseline }}" = "true" ]; then
            args="$args --update-baseline"
          fi

          python -m draagon_ai.testing.benchmarks ${{ inputs.mode }} $args

      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: manual-benchmark-${{ github.run_number }}
          path: benchmark_results/
```

### Self-Hosted Runner Setup

```bash
#!/bin/bash
# scripts/setup-benchmark-runner.sh

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull embedding model
ollama pull mxbai-embed-large

# Install Python dependencies
pip install torch sentence-transformers rank-bm25

# Verify GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create benchmark data directory
mkdir -p /opt/benchmark/data
mkdir -p /opt/benchmark/cache
```

---

## Testing Requirements

### Workflow Tests
```yaml
# Test workflow locally with act
act pull_request -W .github/workflows/benchmark-smoke.yml

# Verify workflow syntax
yamllint .github/workflows/*.yml
```

### Integration Tests
```python
def test_workflow_files_exist():
    """All workflow files present."""
    workflows = Path(".github/workflows")
    assert (workflows / "benchmark-smoke.yml").exists()
    assert (workflows / "benchmark-nightly.yml").exists()
    assert (workflows / "benchmark-manual.yml").exists()

def test_secrets_documented():
    """Required secrets are documented."""
    readme = Path(".github/workflows/README.md").read_text()
    assert "GROQ_API_KEY" in readme
    assert "BENCHMARK_TOKEN" in readme
```

---

## Files to Create/Modify

- `.github/workflows/benchmark-smoke.yml`
- `.github/workflows/benchmark-nightly.yml`
- `.github/workflows/benchmark-manual.yml`
- `.github/workflows/README.md` (documentation)
- `scripts/setup-benchmark-runner.sh`

---

## Definition of Done

- [ ] Smoke test runs on PRs
- [ ] PR comments with results
- [ ] Nightly benchmark runs
- [ ] Results committed to repo
- [ ] Regression creates issue
- [ ] Manual trigger works
- [ ] Self-hosted runner documented
- [ ] Secrets configured
- [ ] Caching working
- [ ] Workflows tested
