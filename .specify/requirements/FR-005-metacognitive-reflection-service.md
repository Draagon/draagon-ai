# FR-005: Metacognitive Reflection Service

**Feature:** Post-task reflection for self-improvement
**Status:** Planned
**Priority:** P1 (Enhancement)
**Complexity:** Medium
**Phase:** Cognitive Swarm Phase 5

---

## Overview

Implement a metacognitive reflection service that analyzes completed multi-agent tasks to identify what worked, what didn't, and how to improve. Unlike passive logging, this actively learns from experience and updates the system's understanding of its own capabilities.

### Research Foundation

- **ICML 2025 Position Paper**: "Truly self-improving agents require intrinsic metacognitive learning"
- **Metacognitive Components**: Knowledge (self-assessment), Planning (learning strategy), Evaluation (reflection)
- **MAST Framework**: Task verification issues (13.48% of failures) addressed by reflection
- **MemoryAgentBench**: Memory agents fail to learn from past errors—reflection addresses this

---

## Functional Requirements

### FR-005.1: Post-Task Reflection Analysis

**Requirement:** Analyze completed multi-agent task execution

**Acceptance Criteria:**
- Input: `TaskContext`, `OrchestratorResult`
- LLM analyzes:
  - Which agents contributed effectively? Why?
  - Which agents struggled? What were the issues?
  - Were there coordination failures?
  - What patterns led to success/failure?
  - How should we adjust for similar future tasks?
- Returns structured `ReflectionResult`

**Test Approach:**
- Execute task with 3 agents: researcher (succeeds), critic (timeout), executor (exception)
- `reflect_on_task()` called
- Assert `effective_agents` includes researcher with reasoning
- Assert `ineffective_agents` includes critic ("timeout") and executor ("exception")
- Assert suggestions include "increase timeout for critic role"

**Constitution Check:** ✅ LLM-First (semantic analysis, NOT hardcoded rules)

---

### FR-005.2: Identify Successful Patterns

**Requirement:** Extract patterns that led to task success

**Acceptance Criteria:**
- `ReflectionResult.successful_patterns: list[str]`
- Examples:
  - "Parallel execution of researcher + critic worked well"
  - "Expertise routing to weather agent was correct"
  - "Belief reconciliation prevented conflict escalation"
- Patterns stored for future reference

**Test Approach:**
- Task succeeds with 3 agents in parallel, belief reconciliation used
- Reflection identifies: "Parallel execution completed 2x faster than sequential would have"
- Assert pattern stored
- Similar task later: pattern influences orchestration decision

**Constitution Check:** ✅ Cognitive authenticity (genuine pattern learning)

---

### FR-005.3: Identify Failure Patterns

**Requirement:** Extract patterns that led to task failure or inefficiency

**Acceptance Criteria:**
- `ReflectionResult.failure_patterns: list[str]`
- Examples:
  - "Agent A exceeded timeout on complex queries"
  - "Lack of coordination led to duplicate work"
  - "Belief conflict unresolved due to equal credibility"
- Patterns flagged for improvement

**Test Approach:**
- Task fails with 2 agents producing same output (duplicate work)
- Reflection identifies: "Agents didn't check shared memory before starting research"
- Assert failure pattern stored
- Next task: orchestrator reminds agents to check shared context first

**Constitution Check:** ✅ Test outcomes (validates learning from failure, not specific error codes)

---

### FR-005.4: Agent Effectiveness Scoring

**Requirement:** Score individual agent contributions

**Acceptance Criteria:**
- `ReflectionResult.effective_agents: list[tuple[str, str]]` - (agent_id, why_effective)
- `ReflectionResult.ineffective_agents: list[tuple[str, str]]` - (agent_id, why_ineffective)
- Based on:
  - Success/failure of agent
  - Contribution to final result
  - Coordination behavior
  - Time efficiency

**Test Approach:**
- 3 agents: A (fast, correct), B (slow, correct), C (fast, incorrect)
- Reflection ranks:
  - Effective: A ("Fast and accurate"), B ("Accurate but slow")
  - Ineffective: C ("Fast but incorrect")
- Assert reasoning captured for each

---

### FR-005.5: Coordination Issue Detection

**Requirement:** Identify coordination failures between agents

**Acceptance Criteria:**
- `ReflectionResult.coordination_issues: list[str]`
- Examples:
  - "Agents A and B duplicated research on same topic"
  - "Belief conflict between A and B unresolved"
  - "Agent C didn't access shared working memory"
- LLM analyzes interaction patterns

**Test Approach:**
- 2 agents both research "weather forecasting"
- Neither checks shared memory to see other's work
- Reflection identifies: "Duplicate work on weather forecasting"
- Assert coordination_issue captured

**Constitution Check:** ✅ LLM-First (semantic coordination analysis)

---

### FR-005.6: Expertise Updates from Reflection

**Requirement:** Update transactive memory based on reflection insights

**Acceptance Criteria:**
- `ReflectionResult.expertise_updates: dict[str, dict[str, float]]` - agent → topic → confidence_delta
- Examples:
  - agent_weather performed well on "forecasting" → +0.1 confidence
  - agent_calendar failed on "weather" → -0.15 confidence
- Applied to `TransactiveMemory` after reflection

**Test Approach:**
- Agent A succeeds on "coding" task
- Reflection suggests: agent_a, topic="coding", confidence_delta=+0.1
- Assert `transactive_memory.update_expertise()` called with delta
- Agent A's expertise in coding increases

**Constitution Check:** ✅ Cognitive authenticity (genuine learning from outcomes)

---

### FR-005.7: Store Insights as Learnings

**Requirement:** Convert reflection insights to permanent learnings

**Acceptance Criteria:**
- `ReflectionResult.new_insights: list[str]`
- Examples:
  - "Calendar queries often need location context"
  - "Weather forecasts >5 days out are unreliable"
  - "Parallel critic + researcher pattern reduces errors"
- Stored via `LearningService` as INSIGHT type
- Available for future decision-making

**Test Approach:**
- Reflection produces insight: "Parallel execution works well for research tasks"
- Assert `learning_service.store_learning(content="...", learning_type=INSIGHT)` called
- Query later: "How should I orchestrate a research task?"
- Insight retrieved and influences decision

**Constitution Check:** ✅ Memory architecture (insights → semantic memory layer)

---

### FR-005.8: Suggested Changes for Improvement

**Requirement:** Generate actionable improvement suggestions

**Acceptance Criteria:**
- `ReflectionResult.suggested_changes: list[str]`
- Examples:
  - "Increase timeout for critic agents from 60s to 120s"
  - "Add weather expertise check before routing weather queries"
  - "Use barrier sync instead of fork-join for conflict-prone tasks"
- Suggestions inform future configuration

**Test Approach:**
- Task times out frequently for critic agents
- Reflection suggests: "Increase timeout for critic role"
- Assert suggestion captured
- Next orchestration: timeout adjusted based on suggestion

---

## Data Structures

```python
@dataclass
class ReflectionResult:
    task_id: str

    # What happened
    overall_success: bool
    task_duration_ms: float
    agents_involved: list[str]

    # What worked
    successful_patterns: list[str]
    effective_agents: list[tuple[str, str]]  # (agent_id, why_effective)

    # What didn't work
    failure_patterns: list[str]
    ineffective_agents: list[tuple[str, str]]  # (agent_id, why_ineffective)
    coordination_issues: list[str]

    # Improvements
    suggested_changes: list[str]
    expertise_updates: dict[str, dict[str, float]]  # agent → topic → confidence_delta

    # Learnings to store
    new_insights: list[str]

class MetacognitiveReflectionService:
    def __init__(
        self,
        llm: LLMProvider,
        transactive_memory: TransactiveMemory,
        learning_service: LearningService,
    )

    async def reflect_on_task(
        self,
        task_context: TaskContext,
        orchestration_result: OrchestratorResult,
    ) -> ReflectionResult

    async def _apply_reflection(self, result: ReflectionResult) -> None

    def _build_reflection_context(
        self,
        task_context: TaskContext,
        result: OrchestratorResult,
    ) -> dict[str, str]

    def _parse_reflection_response(
        self,
        response: str,
        task_context: TaskContext,
        result: OrchestratorResult,
    ) -> ReflectionResult
```

---

## Integration Points

### Upstream Dependencies
- `TaskContext`, `OrchestratorResult` (orchestration)
- `LLMProvider` protocol
- `TransactiveMemory` (FR-004)
- `LearningService` (existing cognition)
- `SharedWorkingMemory` (FR-001) for coordination analysis

### Downstream Consumers
- Orchestrator (calls after task completion)
- Configuration service (applies suggested changes)
- Analytics/monitoring (tracks patterns over time)

---

## Success Criteria

### Cognitive Metrics
- **Pattern Identification Rate**: 80%+ of tasks produce at least 1 actionable insight
- **Improvement Accuracy**: 70%+ of suggested changes improve future task performance
- **Expertise Update Accuracy**: Confidence deltas align with ground truth within 0.1

### Performance Metrics
- **Reflection Latency**: <5s for typical task (3 agents, 30s duration)
- **Storage Efficiency**: Insights stored without duplication
- **Async Processing**: Reflection runs in background, doesn't block response

---

## Testing Strategy

### Unit Tests
- `test_successful_pattern_extraction()`: Patterns from success
- `test_failure_pattern_extraction()`: Patterns from failures
- `test_agent_effectiveness_scoring()`: Agent ranking
- `test_coordination_issue_detection()`: Duplicate work detected
- `test_expertise_updates()`: Confidence deltas correct
- `test_insight_storage()`: Learnings stored

### Integration Tests
- `test_reflection_after_orchestration()`: End-to-end flow
- `test_apply_reflection()`: Expertise and insights updated
- `test_suggested_changes_applied()`: Configuration adjusted

### Cognitive Tests
- `test_improvement_over_time()`: 10 similar tasks, performance improves
- `test_pattern_reuse()`: Successful patterns influence future tasks

---

## Implementation Tasks

1. **Core Reflection Logic** (2 days)
   - `MetacognitiveReflectionService` class
   - `reflect_on_task()` main flow
   - Context building from TaskContext + OrchestratorResult

2. **LLM Integration** (2 days)
   - Design reflection prompt (XML output)
   - Parse XML response into `ReflectionResult`
   - Handle malformed responses

3. **Pattern Extraction** (2 days)
   - Successful pattern identification
   - Failure pattern identification
   - Coordination issue detection

4. **Application of Learnings** (2 days)
   - `_apply_reflection()` implementation
   - Update transactive memory
   - Store insights via learning service

5. **Suggested Changes** (1 day)
   - Extract actionable suggestions
   - Format for configuration updates

6. **Testing** (2 days)
   - Unit tests for all components
   - Integration tests with orchestrator
   - Longitudinal improvement tests

**Total Estimate:** 11 days

---

## XML Prompt Template

```xml
<prompt>
Reflect on this multi-agent task execution:

<task>
  <query>{task_context.query}</query>
  <duration_ms>{orchestration_result.duration_ms}</duration_ms>
  <overall_success>{orchestration_result.success}</overall_success>
</task>

<agent_results>
  <agent id="agent_a" success="true" duration_ms="15000">
    Completed research successfully
  </agent>
  <agent id="agent_b" success="false" duration_ms="60000" error="timeout">
    Exceeded timeout
  </agent>
</agent_results>

<coordination_events>
  - Agent A wrote 5 observations to shared memory
  - Agent B read 3 observations from shared memory
  - Belief conflict detected and reconciled
</coordination_events>

<conflicts_encountered>
  - Agent A vs Agent B: meeting time (resolved)
</conflicts_encountered>

Analyze:
1. Which agents contributed most effectively? Why?
2. Which agents struggled? What were the issues?
3. Were there coordination failures? What caused them?
4. What patterns led to success or failure?
5. How should we adjust for similar future tasks?
6. What did we learn that should be stored?

Respond in XML:
<reflection>
    <successful_patterns>
        <pattern>Description of what worked</pattern>
    </successful_patterns>
    <failure_patterns>
        <pattern>Description of what didn't work</pattern>
    </failure_patterns>
    <effective_agents>
        <agent id="agent_id">Why they were effective</agent>
    </effective_agents>
    <ineffective_agents>
        <agent id="agent_id">Why they struggled</agent>
    </ineffective_agents>
    <coordination_issues>
        <issue>Description of coordination problem</issue>
    </coordination_issues>
    <suggested_changes>
        <change>What to do differently next time</change>
    </suggested_changes>
    <expertise_updates>
        <update agent_id="agent_id" topic="topic" confidence_delta="+0.1"/>
    </expertise_updates>
    <new_insights>
        <insight>Something we learned that should be remembered</insight>
    </new_insights>
</reflection>
</prompt>
```

---

## Constitution Compliance

- ✅ **LLM-First**: Semantic analysis of patterns, coordination, effectiveness (NO hardcoded rules)
- ✅ **XML Output Format**: All LLM prompts return XML
- ✅ **Research-Grounded**: ICML 2025 metacognitive learning, MAST framework
- ✅ **Cognitive Authenticity**: Genuine self-improvement, not superficial logging
- ✅ **Async-First**: Reflection runs after response sent, non-blocking
- ✅ **Test Outcomes**: Tests verify improvement over time, not specific LLM tokens

---

**Document Status:** Draft
**Last Updated:** 2025-12-30
**Author:** draagon-ai team
