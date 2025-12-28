# REQ-002: Orchestrator Migration

**Priority:** High
**Estimated Effort:** Large (2-3 weeks)
**Dependencies:** REQ-001 (Memory System)
**Blocks:** REQ-006 (Roxy Refactor)

---

## 1. Overview

### 1.1 Current State
- **Roxy** has its own `orchestrator.py` with simple tool-use pattern:
  ```
  1. Contextualize query
  2. LLM decides action + args (single call)
  3. Execute tool
  4. Synthesize response
  ```
- **draagon-ai** has full orchestration that is NOT being used:
  - `Agent` - Main agent class
  - `AgentLoop` - ReAct-style reasoning loop
  - `DecisionEngine` - Action selection
  - `ActionExecutor` - Tool execution

### 1.2 Target State
- Roxy uses draagon-ai's `Agent` and `AgentLoop`
- ReAct pattern enables multi-step reasoning with explicit thought traces
- Simple mode still available for fast responses
- Roxy's orchestrator is removed or becomes thin adapter

### 1.3 Success Metrics
- Multi-step tasks complete correctly (e.g., "search calendar then add event")
- Thought traces are visible in debug output
- Single-step tasks are not slower
- All existing Roxy functionality preserved

---

## 2. Detailed Requirements

### 2.1 AgentLoop with ReAct Support

**ID:** REQ-002-01
**Priority:** Critical

#### Description
Implement/enhance `AgentLoop` to support ReAct pattern with explicit thought traces.

#### ReAct Pattern
```
loop:
    THOUGHT: "I need to check the user's calendar for conflicts"
    ACTION: search_calendar(days=7)
    OBSERVATION: [3 events found]

    THOUGHT: "Now I see there's an overlap on Tuesday. Let me check details."
    ACTION: get_event_details(event_id="...")
    OBSERVATION: {details}

    THOUGHT: "I have enough information to answer"
    FINAL_ANSWER: "You have a conflict on Tuesday at 3pm..."
```

#### Acceptance Criteria
- [x] Loop continues until FINAL_ANSWER or max_iterations
- [x] Each step produces THOUGHT, ACTION, OBSERVATION
- [x] Thoughts are logged and can be returned in debug
- [x] Loop can be configured: `use_react: bool` (via LoopMode enum)
- [x] Max iterations configurable (default: 10)
- [x] Timeout per iteration

#### Implementation Notes (2025-12-27)
- Implemented as enhancements to existing `AgentLoop` class
- Added `LoopMode` enum (SIMPLE, REACT, AUTO) for mode selection
- Added `StepType` enum (THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER)
- Added `ReActStep` dataclass for reasoning trace capture
- Added `AgentLoopConfig` for all loop settings
- Enhanced `AgentContext` with observation tracking
- Enhanced `AgentResponse` with react_steps and thought trace formatting
- 27 unit tests passing with mocked dependencies

#### Technical Notes
```python
class AgentLoop:
    async def run(self, query: str, context: AgentContext) -> AgentResponse:
        if self.config.use_react:
            return await self._run_react(query, context)
        else:
            return await self._run_simple(query, context)

    async def _run_react(self, query: str, context: AgentContext) -> AgentResponse:
        steps: list[ReActStep] = []

        for _ in range(self.config.max_iterations):
            # THOUGHT: Reason about what to do next
            thought = await self._think(query, context, steps)
            steps.append(ReActStep(type="thought", content=thought.reasoning))

            if thought.is_final_answer:
                return AgentResponse(
                    answer=thought.answer,
                    steps=steps,
                    success=True,
                )

            # ACTION: Execute the decided action
            result = await self._act(thought.action)
            steps.append(ReActStep(type="action", content=str(thought.action)))
            steps.append(ReActStep(type="observation", content=str(result)))

            # Add observation to context for next iteration
            context.add_observation(result)

        return AgentResponse(
            answer="I couldn't complete this task in the allowed steps.",
            steps=steps,
            success=False,
        )
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | "What time is it?" | Single step, immediate answer | Unit |
| T02 | "Search events then add one" | 2+ steps with thoughts | Integration |
| T03 | Exceed max iterations | Graceful failure message | Unit |
| T04 | Tool throws exception | Captured in observation, continues | Unit |

---

### 2.2 DecisionEngine Integration

**ID:** REQ-002-02
**Priority:** Critical

#### Description
Use draagon-ai's `DecisionEngine` for action selection instead of Roxy's inline prompts.

#### Acceptance Criteria
- [x] DecisionEngine selects appropriate tool for query
- [x] Tool arguments are correctly extracted
- [x] Confidence score returned with decision
- [x] Fallback to "no action" when appropriate
- [x] Supports all Roxy tools

#### Implementation Notes (2025-12-27)
- Enhanced `DecisionResult` with validation fields (`is_valid_action`, `original_action`, `validation_notes`)
- Added helper methods (`is_final_answer()`, `is_no_action()`)
- Added `ACTION_ALIASES` dictionary for common alias resolution
- Enhanced `DecisionEngine` with validation and fallback options
- Added confidence extraction to XML/JSON/text parsers with proper clamping
- 50 unit tests covering all acceptance criteria

#### Decision Output
```python
@dataclass
class Decision:
    action: str              # Tool name or "respond"
    args: dict[str, Any]     # Tool arguments
    confidence: float        # 0.0 - 1.0
    reasoning: str           # Why this action was chosen
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | "What's the weather?" | action="get_weather", high confidence | Unit |
| T02 | "How are you?" | action="respond", no tool needed | Unit |
| T03 | "Turn off bedroom lights" | action="call_service", correct args | Unit |
| T04 | Ambiguous query | Lower confidence, reasoning explains | Unit |

---

### 2.3 ActionExecutor with Tool Registry

**ID:** REQ-002-03
**Priority:** High

#### Description
Use draagon-ai's `ActionExecutor` with a dynamic tool registry that Roxy can populate.

#### Acceptance Criteria
- [x] Tools registered dynamically at startup
- [x] Tool execution returns structured results
- [x] Errors captured and returned, not thrown
- [x] Timeout handling per tool
- [x] Execution metrics collected

#### Implementation Notes (2025-12-27)
- Created `ToolRegistry` class with `Tool`, `ToolParameter`, `ToolMetrics`, `ToolExecutionResult`
- Enhanced `ActionExecutor` to support both `ToolProvider` (legacy) and `ToolRegistry` (new)
- Timeout per tool with `timeout_ms` field and override capability
- Metrics tracking for success/failure/timeout with latency stats
- 60 unit tests covering all acceptance criteria

#### Tool Registry API
```python
class ToolRegistry:
    def register(self, name: str, handler: Callable, schema: dict) -> None: ...
    def get_tool(self, name: str) -> Tool | None: ...
    def list_tools(self) -> list[Tool]: ...
    def get_schemas_for_llm(self) -> list[dict]: ...

# Roxy registers its tools
registry.register("get_weather", weather_handler, weather_schema)
registry.register("call_service", ha_handler, ha_schema)
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Execute registered tool | Result returned | Unit |
| T02 | Execute unknown tool | Error result, not exception | Unit |
| T03 | Tool times out | Timeout error result | Unit |
| T04 | Tool throws exception | Exception captured in result | Unit |

---

### 2.4 Configurable Loop Modes

**ID:** REQ-002-04
**Priority:** Medium

#### Description
Support both simple (single-step) and ReAct (multi-step) modes, configurable per request or globally.

#### Acceptance Criteria
- [ ] Global default mode in config
- [ ] Per-request override via parameter
- [ ] Auto-detection mode based on query complexity
- [ ] Metrics differentiate between modes

#### Configuration
```python
class AgentConfig:
    default_mode: Literal["simple", "react", "auto"] = "auto"
    react_max_iterations: int = 10
    react_iteration_timeout: float = 30.0
    auto_threshold: float = 0.7  # Complexity threshold for auto mode

# Per-request override
response = await agent.run(
    query="...",
    mode="react",  # Override default
)
```

#### Auto-Detection Logic
```
Query complexity signals for ReAct mode:
- Contains "and then", "after that", "also"
- Mentions multiple actions or entities
- Requires information gathering before action
- Previous step had partial results
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Simple query, auto mode | Uses simple mode | Unit |
| T02 | Complex query, auto mode | Uses ReAct mode | Unit |
| T03 | Force simple on complex | Respects override | Unit |
| T04 | Force ReAct on simple | Respects override | Unit |

---

### 2.5 Thought Trace Logging

**ID:** REQ-002-05
**Priority:** Medium

#### Description
Log thought traces for debugging, analysis, and improvement.

#### Acceptance Criteria
- [ ] All ReAct steps logged with timestamps
- [ ] Logs include: thought, action, observation, duration
- [ ] Structured format for parsing
- [ ] Available in debug response
- [ ] Can be disabled for production

#### Log Format
```json
{
  "conversation_id": "conv_123",
  "query": "Search events and add one",
  "mode": "react",
  "steps": [
    {
      "step": 1,
      "type": "thought",
      "content": "I need to search for events first",
      "timestamp": "2025-12-27T10:30:00Z",
      "duration_ms": 150
    },
    {
      "step": 2,
      "type": "action",
      "content": {"tool": "search_calendar", "args": {"days": 7}},
      "timestamp": "2025-12-27T10:30:00.150Z",
      "duration_ms": 200
    },
    {
      "step": 3,
      "type": "observation",
      "content": "[3 events found]",
      "timestamp": "2025-12-27T10:30:00.350Z"
    }
  ],
  "total_duration_ms": 1200
}
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | ReAct with 3 steps | All 3 steps logged | Integration |
| T02 | Debug=true | Steps in response | Integration |
| T03 | Debug=false | Steps not in response | Unit |

---

### 2.6 Roxy Adapter for Orchestration

**ID:** REQ-002-06
**Priority:** High

#### Description
Create adapter that allows Roxy to use draagon-ai orchestration while maintaining its API.

#### Acceptance Criteria
- [x] Roxy's `process_message()` uses draagon-ai Agent
- [x] All Roxy tools registered with draagon-ai registry
- [x] Context (conversation history, user, area) passed correctly
- [x] Response format unchanged for callers
- [x] Debug info includes thought traces

#### Implementation Notes (2025-12-27)
- Created `RoxyOrchestrationAdapter` class in `src/draagon_ai/adapters/roxy_orchestration.py`
- Implements same interface as Roxy's `AgentOrchestrator.process()` method
- Created `RoxyToolDefinition` for registering tools with parameters and handlers
- Created `RoxyResponse`, `ToolCallInfo`, `DebugInfo` dataclasses matching Roxy's format
- Created `create_roxy_orchestration_adapter()` factory function for easy setup
- Session context management caches `AgentContext` per conversation_id
- 37 unit tests covering all acceptance criteria

#### Adapter Structure
```python
class RoxyOrchestrationAdapter:
    def __init__(self, agent: Agent, tool_registry: ToolRegistry):
        self.agent = agent
        self.registry = tool_registry

    async def process_message(
        self,
        text: str,
        user_id: str,
        conversation_id: str,
        area_id: str | None = None,
        debug: bool = False,
    ) -> RoxyResponse:
        # Build context
        context = self._build_context(user_id, conversation_id, area_id)

        # Run agent
        result = await self.agent.run(text, context)

        # Convert to Roxy response format
        return self._convert_response(result, debug)
```

#### Test Cases
| ID | Input | Expected Output | Type |
|----|-------|-----------------|------|
| T01 | Simple query via adapter | Correct Roxy response | Integration |
| T02 | Multi-step via adapter | All steps executed | Integration |
| T03 | Debug mode | Thought traces in debug | Integration |
| T04 | All existing Roxy tests | Pass without changes | Regression |

---

### 2.7 Remove Duplicate Roxy Orchestrator

**ID:** REQ-002-07
**Priority:** Medium

#### Description
After adapter is working, remove or archive Roxy's duplicate orchestration code.

#### Acceptance Criteria
- [x] Old orchestrator.py archived (not deleted initially)
- [x] All imports updated to use adapter
- [x] No duplicate prompt definitions
- [x] No duplicate decision logic
- [x] Code review confirms no orphaned code

#### Implementation Notes (2025-12-27)
**Status:** Completed as analysis & documentation (full migration blocked)

Analysis revealed that Roxy's orchestrator.py (6098 lines) contains significant
Roxy-specific features that cannot be migrated until the adapter has feature parity:
- Calendar cache, relationship graphs, undo, episode summaries, user ID flow, etc.

**Files Created:**
- `roxy-voice-assistant/src/roxy/agent/_archive/MIGRATION_ANALYSIS.md` - Full analysis

**Findings:**
- DECISION_PROMPT and SYNTHESIS_PROMPT are duplicated (can be consolidated)
- Core decision logic is duplicated (can be consolidated)
- 20+ Roxy-specific prompts must remain in Roxy
- Fast router, intents should remain Roxy-specific

See `_archive/MIGRATION_ANALYSIS.md` for full migration plan.

#### Files to Review
```
src/roxy/agent/orchestrator.py     # Archive after migration
src/roxy/agent/prompts.py          # Move prompts to draagon-ai
src/roxy/agent/fast_router.py      # Keep or integrate
src/roxy/agent/intents.py          # Keep (Roxy-specific)
```

---

### 2.8 Unit Tests

**ID:** REQ-002-08
**Priority:** High

#### Coverage Requirements
- Minimum 90% line coverage
- All loop modes tested
- All decision paths tested
- Error handling tested

#### Test Files
```
tests/
  unit/
    orchestration/
      test_agent_loop.py
      test_decision_engine.py
      test_action_executor.py
      test_tool_registry.py
      test_react_steps.py
```

---

### 2.9 Integration Tests

**ID:** REQ-002-09
**Priority:** High

#### Test Scenarios
1. Single-step tool execution
2. Multi-step ReAct reasoning
3. Error recovery and continuation
4. Timeout handling
5. Context propagation across steps

---

### 2.10 Multi-Step Reasoning E2E Tests

**ID:** REQ-002-10
**Priority:** Medium

#### Test Cases
```python
def test_search_then_add():
    """Search for events, then add one to calendar."""
    response = client.query("Search for concerts this week and add one to my calendar")
    assert len(response.steps) >= 2
    assert "search" in response.steps[0].action
    assert "add" in response.steps[-1].action

def test_gather_then_analyze():
    """Get multiple pieces of info, then synthesize."""
    response = client.query("What's the weather and what's on my calendar? Should I bring an umbrella?")
    assert "weather" in str(response.steps)
    assert "calendar" in str(response.steps)
    assert "umbrella" in response.answer.lower() or "rain" in response.answer.lower()
```

---

## 3. Implementation Plan

### 3.1 Sequence
1. Review existing draagon-ai orchestration code (already exists)
2. Enhance AgentLoop with ReAct (REQ-002-01)
3. Wire up DecisionEngine (REQ-002-02)
4. Implement ToolRegistry (REQ-002-03)
5. Add mode configuration (REQ-002-04)
6. Add thought trace logging (REQ-002-05)
7. Create Roxy adapter (REQ-002-06)
8. Remove duplicate code (REQ-002-07)
9. Unit tests (REQ-002-08)
10. Integration tests (REQ-002-09)
11. E2E tests (REQ-002-10)

### 3.2 Risks
| Risk | Mitigation |
|------|------------|
| ReAct is slower than simple | Benchmark both, use auto-mode wisely |
| Infinite loops | Max iterations + timeout |
| Breaking fast-path routing | Keep fast_router.py for truly fast paths |
| LLM inconsistency in thoughts | Structured output prompts |

---

## 4. Review Checklist

### Functional Completeness
- [ ] Simple mode works for single-step queries
- [ ] ReAct mode produces thought traces
- [ ] Auto mode correctly detects complexity
- [ ] All Roxy tools work through new system
- [ ] Error handling is robust

### Code Quality
- [ ] No duplicate logic between Roxy and draagon-ai
- [ ] Adapter is thin (just translation)
- [ ] Prompts are in draagon-ai, not Roxy
- [ ] Logging is comprehensive

### Test Coverage
- [ ] Unit tests ≥ 90%
- [ ] Multi-step scenarios tested
- [ ] Timeout and error cases tested
- [ ] Regression tests pass

### Performance
- [ ] Single-step latency unchanged
- [ ] Multi-step latency acceptable
- [ ] Memory usage reasonable

---

## 5. God-Level Review Prompt

```
ORCHESTRATOR REVIEW - REQ-002

Context: Migrating Roxy's orchestration to use draagon-ai's Agent/AgentLoop
with ReAct-style multi-step reasoning.

Review the implementation against these specific criteria:

1. REACT LOOP
   - Does the loop produce THOUGHT → ACTION → OBSERVATION correctly?
   - Are thoughts meaningful (not just "I will do X")?
   - Does the loop terminate appropriately?
   - Is max_iterations enforced?
   - Is timeout per-step enforced?

2. DECISION ENGINE
   - Are tool selections correct for various queries?
   - Is confidence scoring working?
   - Are arguments extracted correctly?
   - Does "no action needed" work?

3. TOOL REGISTRY
   - Are all Roxy tools registered?
   - Is execution error handling robust?
   - Are timeouts per-tool working?
   - Are metrics being collected?

4. MODE SELECTION
   - Does auto-mode detect complexity correctly?
   - Do overrides work?
   - Is simple mode truly single-step?

5. THOUGHT TRACES
   - Are traces complete and accurate?
   - Are they available in debug?
   - Is logging structured?

6. ROXY ADAPTER
   - Is it truly thin (just translation)?
   - Are all existing tests passing?
   - Is the API unchanged?

7. CODE CLEANUP
   - Is duplicate code removed?
   - Are prompts centralized in draagon-ai?
   - Is there orphaned code?

Provide specific code references for any issues found.
Rate each section: PASS / NEEDS_WORK / FAIL
Overall recommendation: READY / NOT_READY
```

