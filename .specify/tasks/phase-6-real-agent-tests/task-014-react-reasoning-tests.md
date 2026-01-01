# TASK-014: Implement ReAct Reasoning Tests (FR-010.5)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P2 (Nice-to-have - validates advanced reasoning)
**Effort**: 1.5 days
**Status**: Pending
**Dependencies**: TASK-009, TASK-010

## Description

Implement integration tests for multi-step ReAct reasoning: THOUGHT → ACTION → OBSERVATION loops with tool usage and final answer synthesis.

**Core Principle:** Test that agents can perform complex multi-step reasoning with explicit thought traces.

## Acceptance Criteria

- [ ] Test ReAct mode produces THOUGHT/ACTION/OBSERVATION steps
- [ ] Test tools invoked correctly within reasoning loop
- [ ] Test observations integrated into subsequent reasoning
- [ ] Test final answer synthesizes all steps
- [ ] Test trace correctness (logical flow)
- [ ] Performance: 3-7 steps for complex queries

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_react.py`

**Example Tests:**
```python
from draagon_ai.orchestration import LoopMode, ReActStep

@pytest.mark.react_integration
class TestReActReasoning:
    """Test multi-step ReAct reasoning."""

    @pytest.mark.asyncio
    async def test_react_multi_step_trace(self, agent, evaluator):
        """Agent uses ReAct for complex query."""

        # Configure for ReAct mode
        agent.config.loop_mode = LoopMode.REACT

        response = await agent.process(
            "What's the weather like in Portland today, and should I bring an umbrella?"
        )

        # Check trace contains expected steps
        assert len(response.react_trace) > 0

        # Should have THOUGHT steps
        thoughts = [s for s in response.react_trace if s.step_type == "THOUGHT"]
        assert len(thoughts) > 0

        # Should have ACTION steps (get_weather)
        actions = [s for s in response.react_trace if s.step_type == "ACTION"]
        assert len(actions) > 0

        # Should have OBSERVATION steps
        observations = [s for s in response.react_trace if s.step_type == "OBSERVATION"]
        assert len(observations) > 0

        # Final answer should address both questions
        result = await evaluator.evaluate_correctness(
            query="What's the weather and should I bring umbrella?",
            expected_outcome="Mentions current weather and umbrella recommendation",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_react_tool_usage(self, agent, tool_registry):
        """ReAct mode correctly invokes tools during reasoning."""

        agent.config.loop_mode = LoopMode.REACT

        # Register search tool
        @tool(name="search_web", description="Search the web")
        async def search_web(query: str) -> dict:
            return {"results": ["Transformers are neural networks...", "BERT paper..."]}

        tool_registry.register(search_web)

        response = await agent.process(
            "Search for recent papers on transformer models and summarize findings"
        )

        # Should have used search tool
        actions = [s for s in response.react_trace if s.step_type == "ACTION"]
        assert any("search" in s.content.lower() for s in actions)

    @pytest.mark.asyncio
    async def test_react_observation_integration(self, agent):
        """Observations from actions feed into next thought."""

        agent.config.loop_mode = LoopMode.REACT

        response = await agent.process(
            "What's 5 factorial, and is it greater than 100?"
        )

        # Should have:
        # THOUGHT: Need to calculate 5!
        # ACTION: calculate(5!)
        # OBSERVATION: Result is 120
        # THOUGHT: 120 > 100, so yes
        # ANSWER: 5! = 120, which is greater than 100

        trace = response.react_trace

        # Find observation step
        obs_step = next((s for s in trace if s.step_type == "OBSERVATION"), None)
        assert obs_step is not None

        # Next thought should reference observation
        obs_index = trace.index(obs_step)
        if obs_index + 1 < len(trace):
            next_step = trace[obs_index + 1]
            # Should reference the 120 result in reasoning
            assert "120" in next_step.content or "greater" in next_step.content.lower()

    @pytest.mark.asyncio
    async def test_react_max_steps_limit(self, agent):
        """ReAct reasoning respects max steps limit."""

        agent.config.loop_mode = LoopMode.REACT
        agent.config.max_react_steps = 5

        response = await agent.process(
            "Solve this complex multi-step problem..."
        )

        # Should not exceed max steps
        assert len(response.react_trace) <= 5

    @pytest.mark.asyncio
    async def test_react_final_answer_synthesis(self, agent, evaluator):
        """Final answer synthesizes information from all steps."""

        agent.config.loop_mode = LoopMode.REACT

        response = await agent.process(
            "What's the capital of France, and what's its population?"
        )

        # Should have gathered both pieces of info
        result = await evaluator.evaluate_correctness(
            query="What's the capital and population of France?",
            expected_outcome="Mentions Paris and approximate population",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_react_vs_simple_mode_comparison(self, agent_factory, evaluator):
        """ReAct mode provides better reasoning for complex queries."""

        # Create two agents with different modes
        simple_agent = await agent_factory.create(
            AppProfile(name="simple", loop_mode=LoopMode.SIMPLE)
        )
        react_agent = await agent_factory.create(
            AppProfile(name="react", loop_mode=LoopMode.REACT)
        )

        complex_query = "What's the weather in Portland, and based on that, should I bring an umbrella?"

        simple_response = await simple_agent.process(complex_query)
        react_response = await react_agent.process(complex_query)

        # Both should get correct answer
        simple_eval = await evaluator.evaluate_correctness(
            query=complex_query,
            expected_outcome="Mentions weather and umbrella recommendation",
            actual_response=simple_response.answer,
        )

        react_eval = await evaluator.evaluate_correctness(
            query=complex_query,
            expected_outcome="Mentions weather and umbrella recommendation",
            actual_response=react_response.answer,
        )

        # ReAct should have explicit reasoning trace
        assert len(react_response.react_trace) > 0
        assert len(simple_response.react_trace) == 0

        # Both should be correct (but ReAct is more transparent)
        assert react_eval.correct
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_react_multi_step_trace` - THOUGHT/ACTION/OBSERVATION steps
- [ ] `test_react_tool_usage` - Tool invocation in reasoning
- [ ] `test_react_observation_integration` - Observations feed into thoughts
- [ ] `test_react_max_steps_limit` - Step limit enforcement
- [ ] `test_react_final_answer_synthesis` - Answer synthesis
- [ ] `test_react_vs_simple_mode_comparison` - Mode comparison

**Performance Tests:**
- [ ] ReAct queries: <10s for multi-step
- [ ] Average steps: 3-7 for complex queries
- [ ] Step latency: <2s per step

**Cognitive Tests:**
- [ ] Thought progression is logical
- [ ] Observations correctly integrated
- [ ] Final answer uses all gathered info

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_react.py` - ReAct tests

**Modify:**
- `src/draagon_ai/orchestration/loop.py` - Add `react_trace` to AgentResponse if missing

## Pre-Implementation Work

**Verify ReAct Implementation:**

Check that `AgentLoop` has:
1. **ReActStep dataclass:**
```python
@dataclass
class ReActStep:
    step_type: Literal["THOUGHT", "ACTION", "OBSERVATION"]
    content: str
    timestamp: datetime
```

2. **AgentResponse includes trace:**
```python
@dataclass
class AgentResponse:
    answer: str
    confidence: float
    react_trace: list[ReActStep] = field(default_factory=list)  # NEW
    # ... other fields
```

3. **LoopMode.REACT implemented:**
```python
class LoopMode(Enum):
    SIMPLE = "simple"
    REACT = "react"
```

**Estimated Effort:** 2 hours to verify/add missing fields

## Success Metrics

- ✅ Average steps per complex query: 3-7
- ✅ Tool invocation accuracy: >90%
- ✅ Final answer relevance: >85% (LLM-as-judge)
- ✅ Trace coherence: >80%
- ✅ All tests pass with >90% success rate

## Notes

**ReAct Paper Reference:**
- Yao et al. 2022: "ReAct: Synergizing Reasoning and Acting in Language Models"
- Interleaves reasoning (THOUGHT) with actions (tool calls)
- Observations from actions inform next reasoning step

**Performance vs Accuracy Tradeoff:**
- ReAct mode is slower (more LLM calls) but more transparent
- Simple mode is faster but reasoning is implicit
- Use ReAct for complex queries, Simple for straightforward ones

**Cost Control:**
- ReAct mode: ~3-7 LLM calls per query
- Estimated cost: ~$0.20 for full test suite
