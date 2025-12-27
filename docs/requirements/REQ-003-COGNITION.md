# REQ-003: Cognitive Services Consolidation

**Priority:** Medium
**Estimated Effort:** Medium (1-2 weeks)
**Dependencies:** REQ-001 (Memory System)
**Blocks:** REQ-006 (Roxy Refactor)

---

## 1. Overview

### 1.1 Current State
Both Roxy and draagon-ai have cognitive services, with significant overlap:

| Service | Roxy Version | draagon-ai Version |
|---------|--------------|-------------------|
| Belief Reconciliation | `belief_reconciliation.py` | `cognition/beliefs.py` |
| Curiosity Engine | `curiosity_engine.py` | `cognition/curiosity.py` |
| Opinion Formation | `opinion_formation.py` | `cognition/opinions.py` |
| Learning Service | `learning.py` | `cognition/learning.py` |
| Identity Manager | `roxy_self.py` | `cognition/identity.py` |
| Proactive Questions | `proactive_questions.py` | `cognition/proactive_questions.py` |

Most Roxy services are thin wrappers or duplicates with slight differences.

### 1.2 Target State
- All cognitive logic lives in draagon-ai core
- Roxy uses thin adapters for Roxy-specific needs
- No duplicate implementations
- Personality/identity data is Roxy-specific content, not code

### 1.3 Success Metrics
- Single source of truth for cognitive logic
- Roxy cognitive services become pure adapters
- Existing cognitive behavior unchanged
- Test coverage maintained

---

## 2. Detailed Requirements

### 2.1 Belief Reconciliation via Core Service

**ID:** REQ-003-01
**Priority:** High

#### Description
Replace Roxy's belief reconciliation with draagon-ai's `BeliefReconciliationService`.

#### Acceptance Criteria
- [ ] Roxy uses `BeliefReconciliationService` from draagon-ai
- [ ] Adapter provides Roxy's LLM and Memory as protocols
- [ ] Conflict detection works across users
- [ ] Belief confidence is adjusted by source credibility
- [ ] All existing belief tests pass

#### Adapter Pattern
```python
# Roxy adapter
class RoxyBeliefAdapter:
    def __init__(self, llm: LLMService, memory: MemoryService):
        self.service = BeliefReconciliationService(
            llm=RoxyLLMAdapter(llm),
            memory=RoxyMemoryAdapter(memory),
        )

    async def reconcile(self, observation: str, user_id: str) -> ReconciliationResult:
        return await self.service.reconcile(
            observation=observation,
            source_user_id=user_id,
            credibility_provider=RoxyCredibilityProvider(),
        )
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | New observation | Belief created | Integration |
| T02 | Conflicting observation | Reconciliation triggered | Integration |
| T03 | Same user correction | Old belief updated | Unit |
| T04 | Low credibility source | Lower belief confidence | Unit |

---

### 2.2 Curiosity Engine via Core Service

**ID:** REQ-003-02
**Priority:** Medium

#### Description
Replace Roxy's curiosity engine with draagon-ai's `CuriosityEngine`.

#### Acceptance Criteria
- [ ] Roxy uses `CuriosityEngine` from draagon-ai
- [ ] Knowledge gaps are detected in conversations
- [ ] Questions are prioritized by importance and user tolerance
- [ ] Questions are queued for appropriate timing
- [ ] Rate limiting preserved (3/day, 30 min gap)

#### Adapter Pattern
```python
class RoxyCuriosityAdapter:
    def __init__(self, llm: LLMService, memory: MemoryService):
        self.engine = CuriosityEngine(
            llm=RoxyLLMAdapter(llm),
            memory=RoxyMemoryAdapter(memory),
        )

    async def analyze_for_gaps(self, conversation: str, user_id: str) -> list[CuriousQuestion]:
        return await self.engine.detect_gaps(
            conversation=conversation,
            trait_provider=RoxyTraitProvider(),
        )
```

---

### 2.3 Opinion Formation via Core Service

**ID:** REQ-003-03
**Priority:** Medium

#### Description
Replace Roxy's opinion formation with draagon-ai's `OpinionFormationService`.

#### Acceptance Criteria
- [ ] Opinions form consistently with personality
- [ ] Opinions can change with good arguments
- [ ] Opinion history is tracked
- [ ] Graceful fallback on LLM failure

---

### 2.4 Learning Service via Core Service

**ID:** REQ-003-04
**Priority:** High

#### Description
Replace Roxy's learning service with draagon-ai's `LearningService`.

#### Acceptance Criteria
- [ ] Skills, facts, insights extracted from interactions
- [ ] Corrections detected and processed
- [ ] Verification of claims works
- [ ] Memory actions (create/update/delete) work
- [ ] Failure-triggered relearning works

#### Note
The learning service is complex and Roxy-specific. May need extension points in draagon-ai rather than full replacement.

---

### 2.5 Identity Manager Integration

**ID:** REQ-003-05
**Priority:** High

#### Description
Use draagon-ai's `IdentityManager` for Roxy's identity (RoxySelf), with Roxy-specific content.

#### Current RoxySelf
```python
# Roxy-specific content (stays in Roxy)
values = {
    "truth_seeking": 0.95,
    "epistemic_humility": 0.90,
    "helpfulness": 0.95,
}

traits = {
    "verification_threshold": 0.7,
    "curiosity_intensity": 0.6,
    "debate_persistence": 0.4,
}
```

#### Target Structure
- **draagon-ai:** `IdentityManager` protocol + default implementation
- **Roxy:** `roxy_persona.yaml` with personality content
- **Adapter:** Loads YAML, provides to IdentityManager

#### Acceptance Criteria
- [ ] Personality loaded from YAML config
- [ ] Traits are mutable via IdentityManager
- [ ] Preferences form through interaction
- [ ] Opinions tracked with history
- [ ] Serialization to/from Qdrant works

---

### 2.6 Remove Duplicate Roxy Cognitive Services

**ID:** REQ-003-06
**Priority:** Medium

#### Description
After adapters are working, archive or remove duplicate cognitive code from Roxy.

#### Files to Review
```
src/roxy/services/
  belief_reconciliation.py   # → Use draagon-ai + adapter
  curiosity_engine.py        # → Use draagon-ai + adapter
  opinion_formation.py       # → Use draagon-ai + adapter
  learning.py                # → May need extension points
  roxy_self.py               # → Use draagon-ai + config file
  proactive_questions.py     # → Use draagon-ai + adapter
```

#### Acceptance Criteria
- [ ] Each file reviewed for unique logic
- [ ] Unique logic moved to draagon-ai or kept as extension
- [ ] Duplicate logic removed
- [ ] Imports updated throughout codebase
- [ ] No orphaned code

---

### 2.7 Adapter Layer for Roxy-Specific Needs

**ID:** REQ-003-07
**Priority:** Medium

#### Description
Create adapter layer in Roxy for any cognitive needs that are truly Roxy-specific.

#### Roxy-Specific Items
- Voice-optimized responses (TTS)
- Home Assistant context
- Multi-user household model
- Custom personality content

#### Adapter Structure
```
src/roxy/adapters/
  cognitive/
    __init__.py
    beliefs.py         # RoxyBeliefAdapter
    curiosity.py       # RoxyCuriosityAdapter
    opinions.py        # RoxyOpinionAdapter
    learning.py        # RoxyLearningAdapter (if needed)
    identity.py        # RoxyIdentityAdapter
```

---

### 2.8 Unit Tests

**ID:** REQ-003-08
**Priority:** High

#### Coverage Requirements
- Minimum 90% coverage for adapters
- All cognitive paths tested
- Verify adapters are thin (no business logic)

---

### 2.9 Integration Tests

**ID:** REQ-003-09
**Priority:** High

#### Test Scenarios
1. Belief formation from observation
2. Belief conflict resolution
3. Curiosity gap detection
4. Opinion formation and update
5. Learning extraction from conversation

---

## 3. Implementation Plan

### 3.1 Sequence
1. Audit: Compare Roxy vs draagon-ai cognitive services
2. Identify: Unique logic in each implementation
3. Merge: Move unique logic to draagon-ai where appropriate
4. Create adapters for each service (REQ-003-01 through 05)
5. Archive duplicate code (REQ-003-06)
6. Finalize adapter layer (REQ-003-07)
7. Unit tests (REQ-003-08)
8. Integration tests (REQ-003-09)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| Subtle behavioral differences | Extensive A/B testing |
| Breaking personality traits | Snapshot current behavior |
| Missing edge cases | Preserve existing tests |

---

## 4. Review Checklist

### Functional Completeness
- [ ] All cognitive services work through adapters
- [ ] Personality traits preserved
- [ ] No regression in behavior
- [ ] Roxy-specific needs addressed

### Code Quality
- [ ] Adapters are thin (no logic)
- [ ] No duplicate implementations
- [ ] Clean protocol boundaries
- [ ] Proper dependency injection

### Test Coverage
- [ ] Unit tests ≥ 90%
- [ ] All cognitive paths tested
- [ ] Regression tests pass

---

## 5. God-Level Review Prompt

```
COGNITIVE SERVICES REVIEW - REQ-003

Context: Consolidating cognitive services from Roxy into draagon-ai core,
leaving only thin adapters in Roxy.

Review the implementation against these specific criteria:

1. ADAPTER THINNESS
   - Are adapters truly thin (just protocol adaptation)?
   - Is there any business logic in adapters that should be in core?
   - Are adapters using dependency injection correctly?

2. SERVICE PARITY
   - Does belief reconciliation work identically?
   - Does curiosity engine detect the same gaps?
   - Do opinions form consistently?
   - Does learning extract the same information?

3. PERSONALITY PRESERVATION
   - Are RoxySelf traits preserved?
   - Can personality be configured via YAML?
   - Do traits affect behavior correctly?

4. CODE CLEANUP
   - Is duplicate code removed?
   - Are imports updated everywhere?
   - Is there orphaned code?

5. UNIQUE LOGIC
   - Was any unique Roxy logic preserved?
   - Was it moved to draagon-ai or kept as extension?
   - Is the decision documented?

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

