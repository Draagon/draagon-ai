# TASK-010: Implement Core Agent Processing Tests (FR-010.1)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P0 (Critical - validates basic agent functionality)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-009 (Real agent fixtures)

## Description

Implement integration tests for core agent query processing pipeline: query → decision → action → response. Tests the complete `AgentLoop.process()` flow with real LLM and decision engine.

**Core Principle:** Test outcomes, not processes. Validate that agent gets correct results, not specific methods used.

## Acceptance Criteria

- [ ] Test simple query → direct answer (no tools)
- [ ] Test query requiring tool → tool execution → answer
- [ ] Test confidence-based responses (low confidence = hedging)
- [ ] Test error recovery (graceful degradation)
- [ ] Test session persistence across multiple queries
- [ ] Test LLM tier selection (simple → local, complex → complex)
- [ ] All evaluations use LLM-as-judge (NO string matching)
- [ ] Tests complete within performance budget (2s simple, 5s tool)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_core.py`

**Example Test:**
```python
@pytest.mark.agent_integration
class TestAgentCore:
    """Test core agent query processing."""

    @pytest.mark.asyncio
    async def test_simple_query_direct_answer(self, agent, evaluator):
        """Agent answers simple query without tools."""
        response = await agent.process("What is 2+2?")

        result = await evaluator.evaluate_correctness(
            query="What is 2+2?",
            expected_outcome="Agent says 4",
            actual_response=response.answer,
        )

        assert result.correct
        assert response.confidence > 0.9

    @pytest.mark.asyncio
    async def test_query_requires_tool(self, agent, tool_registry, evaluator):
        """Agent uses tool when needed."""

        # Register test tool
        @tool(name="get_weather", description="Get weather for location")
        async def get_weather(location: str) -> dict:
            return {"temp": 72, "condition": "sunny"}

        tool_registry.register(get_weather)

        response = await agent.process("What's the weather in Portland?")

        result = await evaluator.evaluate_correctness(
            query="What's the weather in Portland?",
            expected_outcome="Mentions temperature and sunny condition",
            actual_response=response.answer,
        )

        assert result.correct
        # Don't assert WHICH tool was used - test outcome, not process

    @pytest.mark.asyncio
    async def test_confidence_affects_response(self, agent, evaluator):
        """Low confidence leads to hedging."""
        response = await agent.process("What's the population of Xanadu?")

        # Fictional place should have low confidence
        assert response.confidence < 0.5

        # Should hedge (but don't regex - use evaluator)
        result = await evaluator.evaluate_coherence(
            response=response.answer,
            criteria="Response should hedge or express uncertainty",
        )
        assert result.score > 0.7

    @pytest.mark.asyncio
    async def test_session_persistence(self, agent, evaluator):
        """Agent maintains context across queries in session."""
        # First query establishes context
        await agent.process("My name is Alice", session_id="session_1")

        # Second query uses context
        response = await agent.process(
            "What's my name?",
            session_id="session_1"
        )

        result = await evaluator.evaluate_correctness(
            query="What's my name?",
            expected_outcome="Agent says Alice",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.tier_integration
    @pytest.mark.asyncio
    async def test_simple_query_uses_local_tier(self, agent):
        """Simple queries should use fast local tier."""
        response = await agent.process("What is 2+2?")
        assert response.model_tier == "local"

    @pytest.mark.tier_integration
    @pytest.mark.asyncio
    async def test_complex_query_uses_complex_tier(self, agent):
        """Complex reasoning should use complex tier."""
        response = await agent.process(
            "Analyze the trade-offs between microservices and monoliths"
        )
        assert response.model_tier in ["complex", "deep"]

    @pytest.mark.asyncio
    async def test_error_recovery(self, agent, tool_registry, evaluator):
        """Agent recovers gracefully from tool failures."""

        @tool(name="broken_tool")
        async def broken_tool() -> str:
            raise Exception("Simulated failure")

        tool_registry.register(broken_tool)

        response = await agent.process("Use the broken tool")

        # Should not crash - should provide graceful error message
        assert response is not None
        assert len(response.answer) > 0

        # Should communicate error to user
        result = await evaluator.evaluate_coherence(
            response=response.answer,
            criteria="Response should acknowledge error or inability to complete",
        )
        assert result.score > 0.6
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_simple_query_direct_answer` - Basic agent processing
- [ ] `test_query_requires_tool` - Tool execution integration
- [ ] `test_confidence_affects_response` - Confidence calibration
- [ ] `test_session_persistence` - Context tracking
- [ ] `test_simple_query_uses_local_tier` - Tier selection (simple)
- [ ] `test_complex_query_uses_complex_tier` - Tier selection (complex)
- [ ] `test_error_recovery` - Graceful degradation

**Performance Tests:**
- [ ] Simple queries complete in <2s
- [ ] Tool queries complete in <5s
- [ ] No memory leaks across 100 queries

**Cognitive Tests:**
- [ ] Agent maintains session context in working memory
- [ ] Confidence scores correlate with answer correctness

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_core.py` - Core agent tests

**Modify:**
- `tests/integration/agents/conftest.py` - Add test tool helpers if needed

## Success Metrics

- ✅ All tests pass with >95% success rate
- ✅ Simple queries: <2s latency
- ✅ Tool queries: <5s latency
- ✅ Confidence calibration: correlation >0.7 with correctness
- ✅ LLM tier selection: >90% accuracy
- ✅ Error recovery: No crashes, graceful messages

## Notes

**LLM Tier Validation:**
This requires `AgentResponse` to include `model_tier` field. If missing, add to `src/draagon_ai/orchestration/loop.py`:

```python
@dataclass
class AgentResponse:
    answer: str
    confidence: float
    model_tier: str  # "local" | "complex" | "deep"
    # ... other fields
```

**Cost Control:**
- Each test makes 1-3 LLM calls (~$0.001-0.003)
- Full test suite: ~$0.02-0.05
- Use session-scoped `real_llm` to minimize provider initialization

**Flakiness:**
- LLM responses are non-deterministic
- Use LLM-as-judge for semantic validation (more robust than string matching)
- Retry flaky tests up to 2 times with pytest-rerunfailures
