# Implementation Workflow & Prompts

This document provides the prompts and workflow for implementing, testing, reviewing, and completing requirements.

---

## The Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐ │
│   │  START   │───▶│IMPLEMENT │───▶│   TEST   │───▶│  REVIEW  │ │
│   │          │    │          │    │          │    │          │ │
│   └──────────┘    └──────────┘    └────┬─────┘    └────┬─────┘ │
│                        ▲               │               │        │
│                        │               │               │        │
│                        │    ┌──────────▼───────────────▼──┐    │
│                        │    │                              │    │
│                        └────│      ISSUES FOUND?           │    │
│                      (yes)  │                              │    │
│                             └──────────────┬───────────────┘    │
│                                            │ (no)               │
│                                            ▼                    │
│                                     ┌──────────┐               │
│                                     │  UPDATE  │               │
│                                     │  DOCS    │               │
│                                     └────┬─────┘               │
│                                          │                      │
│                                          ▼                      │
│                                     ┌──────────┐               │
│                                     │   DONE   │               │
│                                     └──────────┘               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prompt 1: START IMPLEMENTATION

Use this prompt to begin work on a requirement.

```
IMPLEMENT REQUIREMENT: [REQ-XXX-YY]

Read the requirement document at:
/home/doug/Development/draagon-ai/docs/requirements/[REQ-XXX-YYYYY.md]

Focus on section: [section number and name, e.g., "2.1 Qdrant Backend for TemporalCognitiveGraph"]

Before writing any code:
1. Read and understand the requirement completely
2. Read the acceptance criteria - these are your success conditions
3. Review the technical notes and code examples
4. Check if there's existing code to build on or integrate with
5. Identify any dependencies that need to be implemented first

Then implement the requirement:
1. Write the code following the patterns shown in the requirement
2. Ensure all acceptance criteria can be checked off
3. Add appropriate logging and error handling
4. Write type hints for all public APIs
5. Add docstrings explaining what each component does

After implementation, list:
- Files created/modified
- Which acceptance criteria are now met
- Any deviations from the spec (with justification)
- Any blockers or issues discovered

Do NOT run tests yet - we'll do that in the next step.
```

---

## Prompt 2: TEST IMPLEMENTATION

Use this after implementation to verify it works.

```
TEST REQUIREMENT: [REQ-XXX-YY]

The implementation for [section name] is complete. Now test it.

## Unit Tests

1. Check if unit tests exist for this requirement
2. If not, create them following the test cases in the requirement doc
3. Run the unit tests:
   ```
   pytest tests/unit/[relevant_path]/ -v
   ```
4. Report coverage for the new code

## Integration Tests

1. Check if integration tests exist
2. If not, create them based on the integration scenarios in the requirement doc
3. Run integration tests:
   ```
   pytest tests/integration/[relevant_path]/ -v
   ```

## Manual Verification

For each acceptance criterion in the requirement:
- [ ] Criterion 1: [describe how you verified it]
- [ ] Criterion 2: [describe how you verified it]
- etc.

## Test Results

Report:
1. Unit test results (pass/fail, coverage %)
2. Integration test results (pass/fail)
3. Manual verification results
4. Any test failures with details
5. Any edge cases that need more testing

If all tests pass, we'll proceed to review.
If tests fail, fix the issues and re-test.
```

---

## Prompt 3: REVIEW IMPLEMENTATION

Use this after tests pass for thorough review.

```
REVIEW REQUIREMENT: [REQ-XXX-YY]

The implementation is complete and tests pass. Now perform a thorough review.

Use the God-Level Review Prompt from the requirement document, but also check:

## Code Quality

1. **No Hardcoded Values**
   - Are all magic numbers/strings configurable?
   - Are URLs, timeouts, limits in config?

2. **Error Handling**
   - Are all external calls wrapped in try/except?
   - Are errors logged with context?
   - Do errors propagate correctly?

3. **Logging**
   - Is there appropriate debug logging?
   - Are important operations logged at info level?
   - Are errors logged with stack traces?

4. **Type Safety**
   - Are all function signatures typed?
   - Are generics used correctly?
   - Would mypy/pyright pass?

5. **Documentation**
   - Do all public functions have docstrings?
   - Are complex algorithms explained?
   - Are there usage examples?

## Architecture

1. **Single Responsibility**
   - Does each class/function do one thing?
   - Are there any god objects?

2. **Dependency Injection**
   - Are dependencies injected, not hardcoded?
   - Are protocols used for abstraction?

3. **Adapter Thinness** (if applicable)
   - Are adapters truly thin?
   - Is there logic that should be in core?

## Security

1. **Input Validation**
   - Are inputs validated?
   - Are there injection risks?

2. **Secrets**
   - Are secrets in config, not code?
   - Are they not logged?

## Performance

1. **N+1 Queries**
   - Are there loops with DB calls inside?

2. **Unnecessary Work**
   - Is there redundant computation?
   - Could caching help?

## Report Format

For each issue found:
```
ISSUE: [severity: CRITICAL/HIGH/MEDIUM/LOW]
Location: [file:line]
Problem: [description]
Fix: [suggested fix]
```

Final verdict: READY / NEEDS_WORK / MAJOR_ISSUES
```

---

## Prompt 4: FIX ISSUES

Use this when review found issues.

```
FIX REVIEW ISSUES: [REQ-XXX-YY]

The review found the following issues:

[paste issues from review]

For each issue:
1. Understand the problem
2. Implement the fix
3. Verify the fix doesn't break anything
4. Document what you changed

After fixing all issues:
1. Re-run the relevant tests
2. Confirm all tests still pass
3. List each issue and how it was resolved

Format:
```
ISSUE: [original issue]
FIX: [what you did]
VERIFIED: [how you verified it works]
```

If any issues cannot be fixed, explain why and propose alternatives.
```

---

## Prompt 5: UPDATE DOCUMENTATION

Use this after implementation is reviewed and approved.

```
UPDATE DOCS FOR: [REQ-XXX-YY]

The implementation is complete and reviewed. Now update the documentation.

## 1. Update Delivery Checklist

Edit: /home/doug/Development/draagon-ai/docs/requirements/DELIVERY_CHECKLIST.md

For this requirement:
- Mark the item as complete: `[x]`
- Add any notes about the implementation
- Update the "Progress Tracking" table if this completes a phase

## 2. Update Requirement Doc

Edit: /home/doug/Development/draagon-ai/docs/requirements/[REQ-XXX.md]

- Mark completed acceptance criteria with [x]
- Add any implementation notes or deviations
- Update test case results if applicable
- Add any lessons learned

## 3. Update CLAUDE.md (if applicable)

If this changes how Roxy works:
- Update relevant sections in /home/doug/Development/roxy-voice-assistant/CLAUDE.md
- Update architecture diagrams if structure changed
- Update code examples if APIs changed

## 4. Add Changelog Entry

In the delivery checklist, add to the Change Log:
| Date | Phase | Change | Author |
|------|-------|--------|--------|
| [today] | [phase] | [what was implemented] | Claude |

## Summary

Report what was updated and confirm all documentation is current.
```

---

## Quick Reference Card

### Starting a New Requirement
```
IMPLEMENT REQUIREMENT: REQ-001-01

Read /home/doug/Development/draagon-ai/docs/requirements/REQ-001-MEMORY.md
Focus on section 2.1: Qdrant Backend for TemporalCognitiveGraph

[paste the START IMPLEMENTATION prompt above]
```

### After Implementation
```
TEST REQUIREMENT: REQ-001-01

[paste the TEST IMPLEMENTATION prompt above]
```

### After Tests Pass
```
REVIEW REQUIREMENT: REQ-001-01

[paste the REVIEW IMPLEMENTATION prompt above]
```

### After Issues Fixed
```
UPDATE DOCS FOR: REQ-001-01

[paste the UPDATE DOCUMENTATION prompt above]
```

---

## Example Full Workflow

Here's an example of going through the full loop:

### Day 1: Start
```
You: "IMPLEMENT REQUIREMENT: REQ-001-01 - Qdrant Backend for TemporalCognitiveGraph.
      Read the requirement doc and implement it."

Claude: [reads doc, implements QdrantGraphStore, lists files and criteria met]
```

### Day 1: Test
```
You: "TEST REQUIREMENT: REQ-001-01. Run all tests and verify acceptance criteria."

Claude: [creates tests, runs them, reports 4/5 passing, 1 failing]
```

### Day 1: Fix Test Failure
```
You: "The bi-temporal query test is failing. Fix it."

Claude: [investigates, fixes bug in timestamp comparison, re-runs tests, all pass]
```

### Day 1: Review
```
You: "REVIEW REQUIREMENT: REQ-001-01. Do the full god-level review."

Claude: [thorough review, finds 2 MEDIUM issues: missing error handling, hardcoded timeout]
```

### Day 1: Fix Issues
```
You: "FIX REVIEW ISSUES: REQ-001-01

      ISSUE: MEDIUM - Missing error handling for Qdrant connection failure
      ISSUE: MEDIUM - Hardcoded 30s timeout"

Claude: [adds try/except with retry, moves timeout to config, verifies tests pass]
```

### Day 1: Final Review
```
You: "REVIEW REQUIREMENT: REQ-001-01 again after fixes."

Claude: [reviews again, no issues found, verdict: READY]
```

### Day 1: Update Docs
```
You: "UPDATE DOCS FOR: REQ-001-01"

Claude: [updates checklist, marks criteria complete, adds changelog entry]
```

### Done!
Move to REQ-001-02.

---

## Tips

1. **One requirement at a time** - Don't try to implement multiple at once
2. **Small iterations** - Better to test frequently than do big bang
3. **Trust the process** - The review prompt catches things
4. **Document deviations** - If you do something different, explain why
5. **Keep the checklist updated** - Future you will thank you

---

## The Nuclear Option: Full Phase Review

When you complete an entire phase (all requirements in REQ-00X), use this:

```
FULL PHASE REVIEW: Phase [N] - [Name]

All requirements in REQ-00X are marked complete. Do a final phase review.

1. Read the entire requirement document
2. Verify EVERY acceptance criterion is actually met
3. Run the FULL test suite for this component
4. Check for any orphaned code or incomplete features
5. Verify documentation is complete
6. Run performance benchmarks if applicable

Use the God-Level Review Prompt from the requirement doc.

Report:
- Overall phase status: COMPLETE / INCOMPLETE
- Any gaps found
- Any technical debt incurred
- Recommendations for next phase
```

