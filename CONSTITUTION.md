# draagon-ai Constitution

**Core Development Principles - Inviolable Rules**

This document defines the fundamental principles that govern all development on draagon-ai.
These principles are **ABSOLUTE** and take precedence over convenience, deadlines, or scope.

---

## 1. Testing Integrity

### 1.1 NEVER Weaken Tests to Pass (ABSOLUTE RULE)

**Tests exist to validate the system. The system must rise to meet the tests, not the other way around.**

When a test fails, the ONLY acceptable responses are:

1. **Fix the system** - Address the root cause of the failure
2. **Document the gap** - If unfixable now, document as a known limitation with a plan
3. **Investigate deeper** - The failure may reveal a larger issue

**FORBIDDEN actions:**
- Lowering thresholds (e.g., 80% → 60%)
- Removing failing test cases
- Adding pytest.skip without root cause analysis
- Increasing tolerance on assertions
- Changing expected values to match wrong output
- Adding partial credit to mask failures
- Commenting out assertions

**Example:**
```python
# ❌ FORBIDDEN: Weakening to pass
def test_accuracy():
    assert accuracy >= 0.60  # Changed from 0.80 because test was failing

# ✅ REQUIRED: Fix the system
def test_accuracy():
    assert accuracy >= 0.80  # System was fixed to achieve this
```

### 1.2 Always Fix Root Issues When Found

**If you discover a bug while working on something else, FIX IT IMMEDIATELY.**

This applies even if:
- The bug is "out of scope" for your current task
- The fix is "small" and seems trivial
- You're "almost done" with something else
- It might require more investigation

The codebase should always improve, never regress. Every interaction is an opportunity to make things better.

**Examples of root issues to fix immediately:**
- Wrong constant value
- Missing edge case
- Logical error
- Incorrect assumption
- Typo in critical string
- Wrong data type
- Missing null check
- Off-by-one error

### 1.3 Tests Must Be Designed to Fail Initially

**Good tests challenge the system. If all tests pass on first run, the tests are too easy.**

Test suites should be tiered:

| Tier | Purpose | Expected Pass Rate |
|------|---------|-------------------|
| **Tier 1: Industry Standard** | Production readiness baseline | Must pass 100% |
| **Tier 2: Advanced** | Push system limits | May fail initially (target: improve) |
| **Tier 3: Frontier** | Unsolved problems | Expected to fail (research targets) |

When implementing new functionality:
1. Write Tier 2/3 tests that expose limitations
2. Run tests and see them fail
3. Improve the system
4. Repeat until Tier 2 passes

### 1.4 Prevent Test Overfitting

**A system that only passes its own test suite is fragile.**

Anti-overfitting measures:
- Include novel test cases not used during development
- Use randomized/parameterized testing with diverse inputs
- Test on out-of-domain examples
- Include adversarial inputs designed to break the system
- Regularly add new test cases from real-world failures
- Test with inputs the developer hasn't seen
- Use holdout test sets for final validation

Signs of overfitting:
- High accuracy on test suite, low accuracy in production
- Tests are suspiciously specific to implementation details
- Edge cases are absent
- Tests only cover "happy path"

### 1.5 Multi-Step Integration Tests

**Complex systems need multi-step tests that verify knowledge builds correctly.**

Multi-step tests should:
- Chain operations that depend on previous results
- Verify context carries forward correctly
- Test that dynamic insertion actually works
- Push limits with long sequences (10+, 20+, 50+ steps)
- Include failure recovery scenarios
- Test rollback and partial success cases

### 1.6 Benchmark Against Industry Standards

**Self-assessment is meaningless. Compare to published benchmarks.**

For every NLP/AI component:
1. Identify relevant published benchmarks (BioScope, CoNLL, TempEval, ATOMIC, etc.)
2. Calibrate thresholds to published results
3. Track performance over time
4. Document gaps from state-of-the-art
5. Never claim "state of the art" without evidence

---

## 2. LLM-First Architecture

### 2.1 Never Use Regex for Semantic Understanding

**The LLM handles ALL semantic analysis. Regex is for syntax, not meaning.**

| Task | ❌ WRONG | ✅ RIGHT |
|------|----------|----------|
| Detect user corrections | Regex: `r"actually\|no,\|wrong"` | LLM analyzes intent |
| Classify intents | Keyword matching | LLM decision prompt |
| Detect learning opportunities | Pattern: `r"remember\|my .* is"` | LLM semantic detection |
| Parse dates/times | Regex patterns | LLM extracts structured data |

**Exceptions (non-semantic tasks):**
- Security blocklist patterns
- TTS text transformations
- URL/email validation
- Entity ID resolution (exact matching)
- Parsing structured LLM output (XML extraction)

### 2.2 XML Output Format for LLM Prompts

**ALWAYS use XML format for LLM output, NOT JSON.**

XML advantages:
- Fewer escaping issues
- Better streaming (incremental parsing)
- More robust parsing (easier recovery)
- Clearer nesting (self-documenting)

---

## 3. Honest Benchmarking

### 3.1 Never Inflate Performance Claims

**Report actual performance, not best-case scenarios.**

When reporting benchmarks:
- Use holdout test sets
- Report confidence intervals
- Include failure cases
- Document methodology
- Compare to published baselines

### 3.2 Acknowledge Limitations

**Every system has limitations. Document them honestly.**

Required documentation:
- Known failure modes
- Edge cases not handled
- Domains not covered
- Performance degradation conditions

---

## 4. Code Quality

### 4.1 No Dead Code

Remove unused code. Don't comment it out "just in case."

### 4.2 No Magic Numbers

Constants should be named and documented.

### 4.3 Clear Error Messages

Errors should explain what went wrong and suggest fixes.

### 4.4 Type Hints

All public interfaces should have type hints.

---

## 5. Research-Driven Development

### 5.1 Cite Sources

When implementing algorithms or approaches:
- Reference the paper/source
- Note any modifications made
- Link to relevant documentation

### 5.2 ATOMIC 2020 Relation Types for Commonsense

When implementing commonsense inference, use the 23 ATOMIC 2020 relations:

**Social-Interaction (9 types):**
- xIntent, xNeed, xAttr, xEffect, xWant, xReact
- oEffect, oWant, oReact

**Physical-Entity (6 types):**
- ObjectUse, AtLocation, MadeOf, HasProperty, CapableOf, DesiredBy

**Event-Centered (8 types):**
- isAfter, isBefore, HasSubEvent, HinderedBy, Causes
- xReason, isFilledBy, HasFirstSubEvent

### 5.3 Epistemic vs Deontic Disambiguation

When implementing modality detection:

**Key disambiguation criteria:**
1. **Scope** - What does the modal operate on?
2. **Source** - Who/what imposes the modality?
3. **Potential barrier** - What would prevent realization?

**Context clues for disambiguation:**
- `must + have + past participle` → Usually EPISTEMIC
- `must + action verb + now/by/deadline` → Usually DEONTIC
- `can + infinitive` → Ambiguous (ability vs permission vs possibility)

---

## 6. Documentation

### 6.1 Code Should Be Self-Documenting

But complex logic needs explanation.

### 6.2 Keep Documentation Updated

Stale documentation is worse than no documentation.

### 6.3 Document Decisions

When making architectural decisions, document:
- What alternatives were considered
- Why this choice was made
- What trade-offs were accepted

---

## Enforcement

These principles are enforced through:
1. Code review
2. Automated tests
3. CI/CD checks
4. Regular audits

Violations should be caught and corrected immediately.

---

**Last Updated:** 2025-12-31
**Version:** 1.0.0
