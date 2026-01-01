# TASK-015: Implement Tool Execution Tests (FR-010.6)

**Phase**: 6 (Real Agent Integration Tests - FR-010)
**Priority**: P1 (Important - validates tool system)
**Effort**: 1.5 days
**Status**: Pending
**Dependencies**: TASK-009, TASK-010

## Description

Implement integration tests for tool discovery, registration, execution, timeout enforcement, error handling, and metrics collection.

**Core Principle:** Test the complete tool lifecycle from discovery to execution with real agent integration.

## Acceptance Criteria

- [ ] Test tool discovery from `@tool` decorator
- [ ] Test parameter validation before execution
- [ ] Test timeout enforcement (terminate long-running tools)
- [ ] Test error handling and graceful degradation
- [ ] Test metrics collection (invocation count, success rate, latency)
- [ ] All tool execution uses real ToolRegistry (not mocks)

## Technical Notes

**Test File:** `tests/integration/agents/test_agent_tools.py`

**Example Tests:**
```python
from draagon_ai.tools import tool, ToolRegistry

@pytest.mark.tool_integration
class TestToolExecution:
    """Test tool discovery and execution."""

    @pytest.mark.asyncio
    async def test_tool_discovery(self, tool_registry):
        """Agent discovers registered tools."""

        @tool(name="test_tool", description="A test tool")
        async def my_tool(arg: str) -> str:
            return f"Result: {arg}"

        tool_registry.register(my_tool)

        # Tool should be in registry
        tools = tool_registry.list_tools()
        assert "test_tool" in tools

    @pytest.mark.asyncio
    async def test_tool_parameter_validation(self, agent, tool_registry):
        """Agent validates tool parameters before execution."""

        @tool(
            name="add",
            description="Add two numbers",
            parameters={
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            }
        )
        async def add_numbers(a: int, b: int) -> int:
            return a + b

        tool_registry.register(add_numbers)

        # Valid parameters
        result = await agent.execute_tool("add", {"a": 2, "b": 3})
        assert result.success
        assert result.data == 5

        # Invalid parameters (missing required)
        result = await agent.execute_tool("add", {"a": 2})
        assert not result.success
        assert "missing" in result.error.lower() or "required" in result.error.lower()

        # Invalid parameters (wrong type)
        result = await agent.execute_tool("add", {"a": "foo", "b": "bar"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_tool_timeout(self, agent, tool_registry):
        """Tools that exceed timeout are terminated."""

        @tool(name="slow_tool", timeout=1.0)
        async def slow_operation() -> str:
            await asyncio.sleep(5)  # Exceeds timeout
            return "done"

        tool_registry.register(slow_operation)

        result = await agent.execute_tool("slow_tool", {})

        assert not result.success
        assert result.timeout
        assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, agent, tool_registry):
        """Tool execution errors are captured gracefully."""

        @tool(name="error_tool")
        async def error_operation() -> str:
            raise ValueError("Simulated error")

        tool_registry.register(error_operation)

        result = await agent.execute_tool("error_tool", {})

        assert not result.success
        assert "error" in result.error.lower() or "ValueError" in result.error

    @pytest.mark.asyncio
    async def test_tool_metrics_collection(self, tool_registry):
        """Tool metrics track invocations and success rate."""

        @tool(name="flaky_tool")
        async def flaky() -> str:
            import random
            if random.random() < 0.5:
                raise Exception("Random failure")
            return "success"

        tool_registry.register(flaky)

        # Execute multiple times
        for _ in range(20):
            try:
                await tool_registry.execute("flaky_tool", {})
            except:
                pass

        # Check metrics
        metrics = tool_registry.get_metrics("flaky_tool")
        assert metrics.invocation_count == 20
        assert 0.3 < metrics.success_rate < 0.7  # ~50% success
        assert metrics.average_latency > 0

    @pytest.mark.asyncio
    async def test_tool_metrics_success_rate(self, tool_registry):
        """Success rate accurately tracks tool reliability."""

        @tool(name="reliable_tool")
        async def reliable() -> str:
            return "success"

        @tool(name="unreliable_tool")
        async def unreliable() -> str:
            raise Exception("Always fails")

        tool_registry.register(reliable)
        tool_registry.register(unreliable)

        # Execute reliable tool
        for _ in range(10):
            await tool_registry.execute("reliable_tool", {})

        # Execute unreliable tool
        for _ in range(10):
            try:
                await tool_registry.execute("unreliable_tool", {})
            except:
                pass

        # Check metrics
        reliable_metrics = tool_registry.get_metrics("reliable_tool")
        unreliable_metrics = tool_registry.get_metrics("unreliable_tool")

        assert reliable_metrics.success_rate == 1.0
        assert unreliable_metrics.success_rate == 0.0

    @pytest.mark.asyncio
    async def test_tool_latency_tracking(self, tool_registry):
        """Tool latency is tracked accurately."""

        @tool(name="fast_tool")
        async def fast() -> str:
            return "done"

        @tool(name="slow_tool")
        async def slow() -> str:
            await asyncio.sleep(0.5)
            return "done"

        tool_registry.register(fast)
        tool_registry.register(slow)

        # Execute each
        await tool_registry.execute("fast_tool", {})
        await tool_registry.execute("slow_tool", {})

        # Check latency
        fast_metrics = tool_registry.get_metrics("fast_tool")
        slow_metrics = tool_registry.get_metrics("slow_tool")

        assert fast_metrics.average_latency < 100  # <100ms
        assert slow_metrics.average_latency >= 500  # >=500ms

    @pytest.mark.asyncio
    async def test_tool_in_agent_query(self, agent, tool_registry, evaluator):
        """Agent integrates tool execution into query response."""

        @tool(
            name="get_time",
            description="Get the current time"
        )
        async def get_time() -> dict:
            return {"time": "3:00 PM", "timezone": "PST"}

        tool_registry.register(get_time)

        response = await agent.process("What time is it?")

        result = await evaluator.evaluate_correctness(
            query="What time is it?",
            expected_outcome="Mentions 3:00 PM or 3 PM",
            actual_response=response.answer,
        )
        assert result.correct

    @pytest.mark.asyncio
    async def test_tool_selection_based_on_description(self, agent, tool_registry):
        """Agent selects correct tool based on description."""

        @tool(name="weather", description="Get weather information")
        async def get_weather(location: str) -> dict:
            return {"temp": 72, "condition": "sunny"}

        @tool(name="time", description="Get current time")
        async def get_time() -> dict:
            return {"time": "3:00 PM"}

        tool_registry.register(get_weather)
        tool_registry.register(get_time)

        # Ask weather question - should use weather tool
        response = await agent.process("What's the weather?")
        # We can't assert WHICH tool was used (test outcomes, not processes)
        # But we can verify the response contains weather info

        # Ask time question - should use time tool
        response = await agent.process("What time is it?")
        # Again, verify outcome (contains time), not process (which tool)
```

## Testing Requirements

**Integration Tests:**
- [ ] `test_tool_discovery` - Registration and listing
- [ ] `test_tool_parameter_validation` - Parameter checking
- [ ] `test_tool_timeout` - Timeout enforcement
- [ ] `test_tool_error_handling` - Error capture
- [ ] `test_tool_metrics_collection` - Metrics tracking
- [ ] `test_tool_metrics_success_rate` - Success rate calculation
- [ ] `test_tool_latency_tracking` - Latency measurement
- [ ] `test_tool_in_agent_query` - Agent integration
- [ ] `test_tool_selection_based_on_description` - Tool selection

**Performance Tests:**
- [ ] Tool discovery: 100% of registered tools
- [ ] Parameter validation: <10ms overhead
- [ ] Timeout enforcement: ±100ms accuracy
- [ ] Metrics collection: 100% accurate
- [ ] Tool execution: <500ms average (excluding tool work)

**Cognitive Tests:** N/A (tool system is infrastructure)

## Files to Create/Modify

**Create:**
- `tests/integration/agents/test_agent_tools.py` - Tool tests

**Modify:**
- None (uses existing ToolRegistry)

## Pre-Implementation Work

**Verify ToolRegistry Metrics:**

From the codebase review, `ToolRegistry` already has `ToolMetrics` with:
- `invocation_count`, `success_count`, `failure_count`, `timeout_count`
- `success_rate`, `avg_latency_ms` properties
- `record_success()`, `record_failure()`, `record_timeout()` methods
- `get_metrics(name)` method

**Verify these methods exist:**
1. `tool_registry.list_tools()` - Returns list of tool names
2. `tool_registry.execute(name, args)` - Executes tool by name
3. `tool_registry.get_metrics(name)` - Returns ToolMetrics

**Estimated Effort:** 1 hour to verify API matches tests

## Success Metrics

- ✅ Tool discovery: 100% of registered tools
- ✅ Parameter validation accuracy: >95%
- ✅ Timeout enforcement: ±100ms
- ✅ Metrics accuracy: 100%
- ✅ Success rate tracking: 100% accurate
- ✅ Latency tracking: ±10ms accuracy
- ✅ All tests pass with >95% success rate

## Notes

**Tool Execution Flow:**
1. Discovery: @tool decorator registers with ToolRegistry
2. Validation: Check parameters match schema
3. Execution: Invoke tool handler with timeout
4. Metrics: Record invocation, success/failure, latency
5. Result: Return ToolResult with data or error

**Timeout Implementation:**
```python
async def execute_with_timeout(tool, args, timeout):
    try:
        result = await asyncio.wait_for(tool.handler(args), timeout=timeout)
        return ToolResult(success=True, data=result)
    except asyncio.TimeoutError:
        return ToolResult(success=False, timeout=True, error="Tool timeout")
    except Exception as e:
        return ToolResult(success=False, error=str(e))
```

**Cost Control:**
- Tool tests don't make LLM calls (except for agent integration tests)
- Estimated cost: <$0.05 for full test suite
