# TASK-017: CI/CD Integration for Real Agent Tests (Optional)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P3 (Optional - automation for future)
**Effort**: 1 day
**Status**: Deferred (CI disabled initially per FR-011)
**Dependencies**: TASK-009 through TASK-016, FR-011

## Description

Integrate real agent integration tests into GitHub Actions CI/CD workflows. This task is **DEFERRED** until local testing is validated and cost controls are in place.

**Status:** FR-011 specifies CI/CD workflows are disabled initially (manual trigger only). This task will be implemented when the team is ready to enable automated testing.

## Acceptance Criteria (When Enabled)

- [ ] Create `.github/workflows/agent-integration.yml`
- [ ] Configure Neo4j service container
- [ ] Set up LLM API secrets (GROQ_API_KEY or OPENAI_API_KEY)
- [ ] Add cost monitoring and alerts
- [ ] Configure test database isolation
- [ ] Set up test result artifacts
- [ ] Enable manual trigger (`workflow_dispatch`)
- [ ] Document how to enable automated triggers

## Technical Notes

**Workflow File:** `.github/workflows/agent-integration.yml`

**Example Workflow (DISABLED):**
```yaml
name: Agent Integration Tests

on:
  # DISABLED: Uncomment when ready for automated CI
  # push:
  #   branches: [main]
  # pull_request:
  #   branches: [main]
  workflow_dispatch:  # Manual trigger only for now

jobs:
  agent-integration:
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
          pip install -e ".[dev,integration]"

      - name: Run real agent integration tests
        env:
          NEO4J_TEST_URI: bolt://localhost:7687
          NEO4J_TEST_PASSWORD: test-password
          GROQ_API_KEY: ${{ secrets.GROQ_API_KEY }}
        run: |
          pytest tests/integration/agents/ -v -m "agent_integration" \
            --junit-xml=test-reports/agent-integration.xml \
            --cov=src/draagon_ai --cov-report=xml

      - name: Track LLM costs
        run: |
          python scripts/track_llm_costs.py \
            --report test-reports/llm-costs.json \
            --alert-threshold 1.00  # Alert if >$1

      - name: Upload test results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: agent-test-results
          path: test-reports/

      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: ./coverage.xml
```

## Pre-Implementation Work (When Ready to Enable)

**1. Set Up GitHub Secrets:**
```bash
# In repository settings > Secrets and variables > Actions
GROQ_API_KEY=your_key_here
# Or
OPENAI_API_KEY=your_key_here
```

**2. Create Cost Tracking Script:**
```python
# scripts/track_llm_costs.py
# (Already planned in FR-011)
```

**3. Validate Local Tests First:**
- All tests in TASK-009 through TASK-016 pass locally
- Cost per test run measured and within budget (<$1)
- Flakiness rate acceptable (<5%)

**Estimated Effort:** 1 day when ready to enable

## Testing Requirements (When Enabled)

**CI Tests:**
- [ ] Workflow triggers on manual dispatch
- [ ] Neo4j service container starts correctly
- [ ] Tests run with real LLM provider
- [ ] Cost tracking script executes
- [ ] Test results uploaded as artifacts
- [ ] Coverage uploaded to Codecov

**Performance:**
- [ ] Full test suite: <15 minutes
- [ ] LLM cost: <$1 per run

## Files to Create/Modify (When Enabled)

**Create:**
- `.github/workflows/agent-integration.yml` - CI workflow
- `scripts/track_llm_costs.py` - Cost monitoring

**Modify:**
- `.github/workflows/pr.yml` - Add agent integration check (optional)

## Success Metrics (When Enabled)

- ✅ Workflow runs successfully on manual trigger
- ✅ All agent integration tests pass in CI
- ✅ Cost tracking reports accurate costs
- ✅ Test results available as artifacts
- ✅ Flakiness: <5%

## Notes

**Why Deferred:**
From FR-011 design decision:
> All automated workflows are configured for **manual trigger only** (`workflow_dispatch`).
> Automatic triggers (push, schedule) are commented out for now.
>
> **Rationale:** Focus on local testing first. Enable automation when ready.

**When to Enable:**
- All tasks TASK-009 through TASK-016 completed
- Local tests stable (>95% pass rate)
- Cost per run validated (<$1)
- Team ready for automated runs

**Cost Control:**
- Use mock LLM for framework tests (free)
- Use real LLM only for agent integration tests (paid)
- Set cost alerts at $1/run threshold
- Track costs in `test-reports/llm-costs.json`

**To Enable Automated Triggers:**
```yaml
# In .github/workflows/agent-integration.yml
on:
  push:  # UNCOMMENT THIS
    branches: [main]
  pull_request:  # UNCOMMENT THIS
    branches: [main]
  workflow_dispatch:
```
