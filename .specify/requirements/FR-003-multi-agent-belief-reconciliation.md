# FR-003: Multi-Agent Belief Reconciliation

**Feature:** Reconcile conflicting beliefs from multiple agents
**Status:** Planned
**Priority:** P0 (Foundation)
**Complexity:** High
**Phase:** Cognitive Swarm Phase 3

---

## Overview

Implement multi-agent belief reconciliation that detects conflicts between agents' observations, weighs them by agent credibility, and synthesizes consolidated beliefs or flags unresolvable conflicts for human clarification. This extends single-agent belief reconciliation to handle inter-agent disagreement.

### Research Foundation

- **MAST Framework**: Inter-agent misalignment causes 34.7% of multi-agent failures
- **MemoryAgentBench**: Current systems fail at conflict resolution (45% success rate)
- **Epistemic Logic**: BDI (Beliefs, Desires, Intentions) agent architectures
- **MultiAgentBench Werewolf**: Trust/disclosure failures due to lack of belief tracking

---

## Functional Requirements

### FR-003.1: Conflict Detection from Shared Observations

**Requirement:** Detect when agents have contradictory observations

**Acceptance Criteria:**
- Input: list of `SharedObservation` with `conflicts_with` markers
- Group observations by topic (semantic clustering)
- Identify topics with 2+ observations from different agents
- Flag as potential conflict if semantic similarity > threshold but content differs

**Test Approach:**
- Agent A observes: "The meeting is at 3pm in Room A"
- Agent B observes: "The meeting is at 4pm in Room B"
- Both marked as FACT belief candidates
- `reconcile_multi_agent()` detects conflict
- Assert grouped under "meeting time" topic

**Constitution Check:** ✅ LLM-First (semantic grouping via embeddings, NOT regex)

---

### FR-003.2: Agent Credibility Weighting

**Requirement:** Weight observations by source agent's track record

**Acceptance Criteria:**
- Input: `agent_credibilities: dict[str, float]` (0-1 scale)
- Higher credibility agents' observations weighted more heavily
- Credibility derived from:
  - Skill success rate (from transactive memory)
  - Historical accuracy (from metacognitive reflection)
- Default credibility: 0.5 (neutral prior)

**Test Approach:**
- Agent A (credibility 0.9) says "Meeting at 3pm"
- Agent B (credibility 0.3) says "Meeting at 4pm"
- Reconciliation favors Agent A's observation
- Consolidated belief: "Meeting at 3pm" with higher confidence

**Constitution Check:** ✅ Research-grounded (credibility weighting from BDI architectures)

---

### FR-003.3: LLM-Based Conflict Analysis

**Requirement:** Use LLM to analyze conflicts and synthesize beliefs

**Acceptance Criteria:**
- Prompt includes:
  - All conflicting observations
  - Agent credibilities
  - Topic context
- LLM determines:
  - Do observations truly conflict?
  - Which is most likely correct?
  - Can they be synthesized?
  - If unresolvable, what clarifying question?
- Response in XML format (not JSON)

**Test Approach:**
- Temporal conflict:
  - Agent A (Monday): "Project on track"
  - Agent B (Wednesday): "Project delayed"
- LLM recognizes both are correct for their time context
- Consolidated belief: "Project was on track Monday but delayed by Wednesday"
- Assert `conflicts=False` (resolved)

**Constitution Check:** ✅ XML Output Format (MANDATORY)

---

### FR-003.4: Reconciliation Result Types

**Requirement:** Return structured reconciliation result

**Acceptance Criteria:**
- `ReconciliationResult` includes:
  - `resolved: bool` - was conflict resolved?
  - `consolidated_belief: AgentBelief | None`
  - `confidence: float` (0-1)
  - `needs_human_clarification: bool`
  - `clarification_question: str | None`
  - `observations_considered: list[str]` (audit trail)
  - `reasoning: str` (why this conclusion)
- If resolved: consolidated_belief populated
- If unresolved: clarification_question populated

**Test Approach:**
- Simple contradiction (resolvable):
  - Agent A (high credibility): "6 cats"
  - Agent B (low credibility): "5 cats"
  - Assert `resolved=True`, `consolidated_belief="6 cats"`, `confidence>0.7`
- Unresolvable:
  - Agent A (equal credibility): "Meeting at 3pm"
  - Agent B (equal credibility): "Meeting at 4pm"
  - Assert `resolved=False`, `needs_human_clarification=True`
  - Assert clarification_question like "What time is the meeting scheduled?"

**Constitution Check:** ✅ Cognitive authenticity (genuine belief synthesis)

---

### FR-003.5: Partial Overlap Handling

**Requirement:** Recognize when observations partially overlap vs contradict

**Acceptance Criteria:**
- Observations that are compatible but differ in specificity should merge, not conflict
- Example:
  - Agent A: "John has 3 cats and 2 dogs"
  - Agent B: "John has several pets including cats"
  - Result: Merge to "John has 3 cats and 2 dogs" (A is more specific)
- LLM determines compatibility

**Test Approach:**
- Agent A: "User works from home Mon-Wed"
- Agent B: "User sometimes works from home"
- LLM recognizes B is subset of A
- Consolidated: "User works from home Mon-Wed"
- Assert `conflicts=False` in LLM response

**Constitution Check:** ✅ LLM-First (semantic compatibility, not pattern matching)

---

### FR-003.6: Source Credibility Conflict

**Requirement:** Weigh observations by source authority, not just agent credibility

**Acceptance Criteria:**
- Observation metadata includes `source_authority` (e.g., "CEO earnings call" vs "Reddit rumor")
- Source authority overrides agent credibility
- Prompt includes both agent credibility and source authority

**Test Approach:**
- Agent A (credibility 0.5): "Revenue up 10% (from CEO earnings call)"
- Agent B (credibility 0.8): "Revenue might be up 15% (from Reddit rumor)"
- LLM weighs CEO source higher
- Consolidated belief: "Revenue up 10%"
- Assert confidence reflects source authority

**Constitution Check:** ✅ Cognitive authenticity (real-world source weighting)

---

### FR-003.7: Audit Trail

**Requirement:** Maintain full audit trail of reconciliation decisions

**Acceptance Criteria:**
- `ReconciliationResult` includes:
  - All observation IDs considered
  - Reasoning from LLM
  - Credibility scores used
- Stored for metacognitive reflection
- Can reconstruct why a belief was formed

**Test Approach:**
- Reconcile 3 conflicting observations
- Assert `observations_considered` contains all 3 IDs
- Assert `reasoning` is non-empty
- Query: "Why do we believe X?"
- System can explain: "Agent A (credibility 0.9) said X, Agent B (0.4) said Y, chose X due to credibility"

---

## Data Structures

```python
@dataclass
class ReconciliationResult:
    resolved: bool
    consolidated_belief: AgentBelief | None
    confidence: float
    needs_human_clarification: bool = False
    clarification_question: str | None = None
    observations_considered: list[str] = field(default_factory=list)
    reasoning: str = ""

class MultiAgentBeliefReconciliation:
    def __init__(
        self,
        base_belief_service: BeliefReconciliationService,
        llm: LLMProvider,
    )

    async def reconcile_multi_agent(
        self,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult

    async def _reconcile_topic(
        self,
        topic: str,
        observations: list[SharedObservation],
        agent_credibilities: dict[str, float],
    ) -> ReconciliationResult

    async def _single_observation_to_belief(
        self,
        observation: SharedObservation,
    ) -> ReconciliationResult

    def _group_by_topic(
        self,
        observations: list[SharedObservation],
    ) -> dict[str, list[SharedObservation]]

    def _parse_reconciliation_response(
        self,
        response: str,
        observations: list[SharedObservation],
    ) -> ReconciliationResult

    def _combine_results(
        self,
        results: list[ReconciliationResult],
    ) -> ReconciliationResult
```

---

## Integration Points

### Upstream Dependencies
- `SharedObservation` (from FR-001)
- `BeliefReconciliationService` (existing single-agent belief system)
- `LLMProvider` protocol
- `AgentBelief`, `BeliefType` (existing belief structures)

### Downstream Consumers
- `ParallelCognitiveOrchestrator` (calls after detecting conflicts)
- `MetacognitiveReflectionService` (uses audit trail)
- Agent belief caches (updated with consolidated beliefs)

---

## Success Criteria

### Cognitive Metrics
- **Conflict Resolution Rate**: >90% of conflicts resolved or flagged for human
- **Resolution Accuracy**: >85% of resolutions align with ground truth (benchmark)
- **Audit Completeness**: 100% of decisions have reasoning trail

### Performance Metrics
- **Latency**: <2s to reconcile 5 conflicting observations
- **LLM Calls**: 1 LLM call per topic (not per observation pair)

### Benchmark Targets
- **MemoryAgentBench Conflict Resolution**: Target 75% (vs 50% SOTA)
- **MultiAgentBench Werewolf**: Target 55% task score (vs 36.33% SOTA)

---

## Testing Strategy

### Unit Tests
- `test_simple_contradiction()`: High credibility agent wins
- `test_temporal_conflict()`: Time-aware reconciliation
- `test_partial_overlap()`: Compatible observations merge
- `test_source_authority()`: CEO beats Reddit
- `test_unresolvable()`: Equal credibility triggers clarification

### Integration Tests
- `test_multi_topic_reconciliation()`: Multiple topics reconciled independently
- `test_audit_trail()`: Full reasoning reconstructable
- `test_belief_cache_update()`: Reconciled beliefs propagate to agents

### Cognitive Tests
- `test_MemoryAgentBench_conflict_resolution()`: Run official benchmark
- `test_MultiAgentBench_werewolf()`: Trust-based belief sharing

---

## Implementation Tasks

1. **Core Reconciliation Logic** (3 days)
   - Implement `MultiAgentBeliefReconciliation` class
   - `reconcile_multi_agent()` main flow
   - Topic grouping logic

2. **LLM Integration** (2 days)
   - Design reconciliation prompt
   - XML response parsing
   - Handle malformed responses

3. **Credibility Weighting** (2 days)
   - Agent credibility integration
   - Source authority metadata
   - Credibility score calculation

4. **Result Structures** (1 day)
   - `ReconciliationResult` dataclass
   - Audit trail population
   - `consolidated_belief` creation

5. **Edge Cases** (2 days)
   - Partial overlap detection
   - Temporal conflicts
   - Multi-topic conflicts
   - Unresolvable conflicts

6. **Testing** (3 days)
   - Unit tests for all conflict types
   - Integration with orchestrator
   - Benchmark runs

**Total Estimate:** 13 days

---

## XML Prompt Template

```xml
<prompt>
Analyze these observations from different agents about "{topic}":

<observations>
  <observation agent_id="agent_a" credibility="0.9">
    Meeting is at 3pm in Room A
  </observation>
  <observation agent_id="agent_b" credibility="0.4">
    Meeting is at 4pm in Room B
  </observation>
</observations>

Determine:
1. Do these observations conflict? (yes/no)
2. If yes, which is most likely correct based on credibility?
3. Can you synthesize a consolidated understanding?
4. If unresolvable, what clarifying question would help?

Respond in XML:
<reconciliation>
    <conflicts>yes/no</conflicts>
    <consolidated_belief>The reconciled understanding, or null if unresolvable</consolidated_belief>
    <confidence>0.0-1.0</confidence>
    <needs_clarification>true/false</needs_clarification>
    <clarification_question>Question to ask if needed</clarification_question>
    <reasoning>Why you reached this conclusion</reasoning>
</reconciliation>
</prompt>
```

---

## Constitution Compliance

- ✅ **LLM-First**: Semantic conflict detection, compatibility analysis (NO regex)
- ✅ **XML Output Format**: All LLM prompts return XML
- ✅ **Research-Grounded**: MAST framework, BDI architectures, MemoryAgentBench
- ✅ **Cognitive Authenticity**: Genuine belief synthesis, credibility weighting
- ✅ **Test Outcomes**: Tests verify correct belief formation, not specific LLM tokens

---

**Document Status:** Draft
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
