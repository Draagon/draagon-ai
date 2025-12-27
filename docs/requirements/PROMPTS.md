# Copy-Paste Prompts

Just fill in the `[brackets]` and go.

---

## 1. IMPLEMENT

```
IMPLEMENT: [REQ-XXX-YY] - [Name]

Read the full requirement at:
/home/doug/Development/draagon-ai/docs/requirements/[REQ-XXX-NAME.md]

Section: [2.X - Section Name]

Do this:
1. Read the requirement and acceptance criteria
2. Check for existing code to build on
3. Implement it with proper types, logging, error handling
4. List files changed and which acceptance criteria are met
5. Note any deviations or blockers

Don't test yet - just implement.
```

---

## 2. TEST

```
TEST: [REQ-XXX-YY]

The implementation is done. Now:

1. Create/run unit tests (target 90% coverage)
2. Create/run integration tests
3. Manually verify each acceptance criterion
4. Report: pass/fail, coverage, any failures

If tests fail, fix and re-test.
If tests pass, we'll review.
```

---

## 3. REVIEW

```
REVIEW: [REQ-XXX-YY]

Tests pass. Now do god-level review:

1. Run the God-Level Review Prompt from the requirement doc
2. Check: no hardcoded values, proper error handling, types, docs
3. Check: single responsibility, DI, thin adapters
4. Check: input validation, no secrets in code
5. Check: no N+1 queries, no redundant work

For each issue:
- SEVERITY: CRITICAL/HIGH/MEDIUM/LOW
- Location: file:line
- Problem: what's wrong
- Fix: how to fix

Verdict: READY / NEEDS_WORK
```

---

## 4. FIX

```
FIX: [REQ-XXX-YY]

Review found issues:

[paste issues here]

For each:
1. Fix it
2. Verify tests still pass
3. Report: ISSUE → FIX → VERIFIED

Then re-run REVIEW prompt.
```

---

## 5. COMPLETE

```
COMPLETE: [REQ-XXX-YY]

Implementation reviewed and approved. Update docs:

1. DELIVERY_CHECKLIST.md - mark [x], add notes, update progress table
2. REQ-XXX.md - mark acceptance criteria [x], add implementation notes
3. CLAUDE.md - update if behavior changed
4. Add changelog entry

Confirm all docs updated.
```

---

## The Flow

```
IMPLEMENT → TEST → (fail? fix, re-test) → REVIEW → (issues? FIX, re-review) → COMPLETE
```

---

## Quick Start Example

**You say:**
```
IMPLEMENT: REQ-001-01 - Qdrant Backend for TemporalCognitiveGraph

Read: /home/doug/Development/draagon-ai/docs/requirements/REQ-001-MEMORY.md
Section: 2.1

Implement with proper types, logging, error handling. List what you did.
```

**Then:**
```
TEST: REQ-001-01

Run unit tests, integration tests, verify acceptance criteria. Report results.
```

**Then:**
```
REVIEW: REQ-001-01

God-level review. List any issues with severity.
```

**If issues, then:**
```
FIX: REQ-001-01

Issues:
- MEDIUM: Missing retry logic in QdrantGraphStore.save_node
- LOW: Timeout should be configurable

Fix these, verify tests pass.
```

**Finally:**
```
COMPLETE: REQ-001-01

Update checklist, requirement doc, CLAUDE.md if needed.
```

---

## Phase Complete?

When all items in a phase are done:

```
PHASE REVIEW: [Phase N] - [Name]

All REQ-00X items complete. Final review:
1. Verify EVERY acceptance criterion met
2. Run full test suite
3. Check for orphaned code
4. Verify docs complete
5. Run benchmarks if applicable

Status: COMPLETE / INCOMPLETE
Gaps found?
Tech debt?
Ready for next phase?
```

