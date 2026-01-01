"""Tool execution integration tests (FR-010.6).

Tests the complete tool lifecycle:
- Tool registration and discovery
- Parameter validation
- Timeout enforcement
- Error handling
- Metrics collection (invocation count, success rate, latency)

These tests validate the ToolRegistry works correctly.
"""

import asyncio
import os
import pytest
import time

from draagon_ai.orchestration.registry import (
    Tool,
    ToolParameter,
    ToolMetrics,
    ToolRegistry,
    ToolExecutionResult,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def fresh_registry():
    """Fresh tool registry for each test."""
    return ToolRegistry()


# =============================================================================
# Tool Discovery Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolDiscovery:
    """Test tool registration and discovery."""

    def test_register_tool(self, fresh_registry):
        """Can register a tool with the registry."""

        async def my_handler(args):
            return {"result": "success"}

        tool = Tool(
            name="test_tool",
            description="A test tool",
            handler=my_handler,
        )

        fresh_registry.register(tool)

        assert fresh_registry.has_tool("test_tool")
        assert "test_tool" in fresh_registry.list_tools()

    def test_register_multiple_tools(self, fresh_registry):
        """Can register multiple tools."""

        async def handler1(args):
            return "1"

        async def handler2(args):
            return "2"

        fresh_registry.register(
            Tool(name="tool1", description="Tool 1", handler=handler1)
        )
        fresh_registry.register(
            Tool(name="tool2", description="Tool 2", handler=handler2)
        )

        assert len(fresh_registry) == 2
        assert "tool1" in fresh_registry
        assert "tool2" in fresh_registry

    def test_list_tools(self, fresh_registry):
        """list_tools() returns all registered tool names."""

        async def handler(args):
            return None

        fresh_registry.register(
            Tool(name="alpha", description="Alpha", handler=handler)
        )
        fresh_registry.register(
            Tool(name="beta", description="Beta", handler=handler)
        )
        fresh_registry.register(
            Tool(name="gamma", description="Gamma", handler=handler)
        )

        tools = fresh_registry.list_tools()
        assert len(tools) == 3
        assert set(tools) == {"alpha", "beta", "gamma"}

    def test_get_tool(self, fresh_registry):
        """Can retrieve a tool by name."""

        async def handler(args):
            return "result"

        fresh_registry.register(
            Tool(name="my_tool", description="My tool", handler=handler)
        )

        tool = fresh_registry.get_tool("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"
        assert tool.description == "My tool"

    def test_get_unknown_tool(self, fresh_registry):
        """get_tool returns None for unknown tools."""
        assert fresh_registry.get_tool("nonexistent") is None

    def test_unregister_tool(self, fresh_registry):
        """Can unregister a tool."""

        async def handler(args):
            return None

        fresh_registry.register(
            Tool(name="removable", description="Removable", handler=handler)
        )
        assert fresh_registry.has_tool("removable")

        result = fresh_registry.unregister("removable")
        assert result is True
        assert not fresh_registry.has_tool("removable")

    def test_unregister_unknown_tool(self, fresh_registry):
        """Unregistering unknown tool returns False."""
        result = fresh_registry.unregister("nonexistent")
        assert result is False


# =============================================================================
# Tool Parameter Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolParameters:
    """Test tool parameter definitions."""

    def test_tool_with_parameters(self, fresh_registry):
        """Tool can define parameters with schemas."""

        async def add(args):
            return args["a"] + args["b"]

        tool = Tool(
            name="add",
            description="Add two numbers",
            handler=add,
            parameters=[
                ToolParameter(
                    name="a",
                    type="integer",
                    description="First number",
                    required=True,
                ),
                ToolParameter(
                    name="b",
                    type="integer",
                    description="Second number",
                    required=True,
                ),
            ],
        )

        fresh_registry.register(tool)

        retrieved = fresh_registry.get_tool("add")
        assert len(retrieved.parameters) == 2
        assert retrieved.parameters[0].name == "a"
        assert retrieved.parameters[1].name == "b"

    def test_optional_parameters(self, fresh_registry):
        """Parameters can be optional with defaults."""

        async def greet(args):
            name = args.get("name", "World")
            return f"Hello, {name}!"

        tool = Tool(
            name="greet",
            description="Greet someone",
            handler=greet,
            parameters=[
                ToolParameter(
                    name="name",
                    type="string",
                    description="Name to greet",
                    required=False,
                    default="World",
                ),
            ],
        )

        fresh_registry.register(tool)

        retrieved = fresh_registry.get_tool("greet")
        assert retrieved.parameters[0].required is False
        assert retrieved.parameters[0].default == "World"

    def test_enum_parameter(self, fresh_registry):
        """Parameters can have enum constraints."""

        async def set_mode(args):
            return f"Mode set to {args['mode']}"

        tool = Tool(
            name="set_mode",
            description="Set operation mode",
            handler=set_mode,
            parameters=[
                ToolParameter(
                    name="mode",
                    type="string",
                    description="Mode to set",
                    required=True,
                    enum=["fast", "normal", "slow"],
                ),
            ],
        )

        fresh_registry.register(tool)

        retrieved = fresh_registry.get_tool("set_mode")
        assert retrieved.parameters[0].enum == ["fast", "normal", "slow"]

    def test_to_prompt_format(self, fresh_registry):
        """Tool generates human-readable prompt format."""

        async def search(args):
            return []

        tool = Tool(
            name="search",
            description="Search the database",
            handler=search,
            parameters=[
                ToolParameter(
                    name="query", type="string", description="Search query", required=True
                ),
                ToolParameter(
                    name="limit", type="integer", description="Max results", required=False
                ),
            ],
        )

        prompt = tool.to_prompt_format()
        assert "search" in prompt
        assert "query" in prompt
        assert "limit" in prompt
        assert "optional" in prompt.lower()


# =============================================================================
# Tool Execution Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolExecution:
    """Test tool execution."""

    @pytest.mark.asyncio
    async def test_execute_simple_tool(self, fresh_registry):
        """Can execute a simple tool."""

        async def echo(args):
            return args.get("message", "no message")

        fresh_registry.register(
            Tool(name="echo", description="Echo back", handler=echo)
        )

        result = await fresh_registry.execute("echo", {"message": "hello"})

        assert result.success
        assert result.result == "hello"
        assert result.tool_name == "echo"

    @pytest.mark.asyncio
    async def test_execute_tool_with_calculation(self, fresh_registry):
        """Tool can perform calculations."""

        async def multiply(args):
            return args["a"] * args["b"]

        fresh_registry.register(
            Tool(
                name="multiply",
                description="Multiply two numbers",
                handler=multiply,
                parameters=[
                    ToolParameter(name="a", type="number", description="First number"),
                    ToolParameter(name="b", type="number", description="Second number"),
                ],
            )
        )

        result = await fresh_registry.execute("multiply", {"a": 6, "b": 7})

        assert result.success
        assert result.result == 42

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, fresh_registry):
        """Executing unknown tool returns error."""
        result = await fresh_registry.execute("nonexistent", {})

        assert not result.success
        assert "unknown" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_context(self, fresh_registry):
        """Tool receives context when provided."""

        async def context_tool(args, context=None):
            user = context.get("user_id", "unknown") if context else "no context"
            return f"User: {user}"

        fresh_registry.register(
            Tool(name="context_tool", description="Uses context", handler=context_tool)
        )

        result = await fresh_registry.execute(
            "context_tool", {}, context={"user_id": "doug"}
        )

        assert result.success
        assert "doug" in result.result


# =============================================================================
# Timeout Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolTimeout:
    """Test tool timeout enforcement."""

    @pytest.mark.asyncio
    async def test_tool_respects_timeout(self, fresh_registry):
        """Slow tool is terminated at timeout."""

        async def slow_tool(args):
            await asyncio.sleep(5)  # Exceeds timeout
            return "done"

        fresh_registry.register(
            Tool(
                name="slow_tool",
                description="A slow tool",
                handler=slow_tool,
                timeout_ms=500,  # 500ms timeout
            )
        )

        start = time.time()
        result = await fresh_registry.execute("slow_tool", {})
        elapsed = time.time() - start

        assert not result.success
        assert result.timed_out
        assert "timed out" in result.error.lower() or "timeout" in result.error.lower()
        # Should complete within 1 second (with some margin)
        assert elapsed < 2.0

    @pytest.mark.asyncio
    async def test_timeout_override(self, fresh_registry):
        """Can override default timeout."""

        async def slow_tool(args):
            await asyncio.sleep(2)
            return "done"

        fresh_registry.register(
            Tool(
                name="slow",
                description="Slow",
                handler=slow_tool,
                timeout_ms=30000,  # Default 30s
            )
        )

        # Override with short timeout
        result = await fresh_registry.execute(
            "slow", {}, timeout_override_ms=500
        )

        assert not result.success
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_fast_tool_completes_before_timeout(self, fresh_registry):
        """Fast tool completes successfully before timeout."""

        async def fast_tool(args):
            return "done quickly"

        fresh_registry.register(
            Tool(
                name="fast",
                description="Fast",
                handler=fast_tool,
                timeout_ms=5000,
            )
        )

        result = await fresh_registry.execute("fast", {})

        assert result.success
        assert not result.timed_out
        assert result.result == "done quickly"


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolErrorHandling:
    """Test tool error handling."""

    @pytest.mark.asyncio
    async def test_tool_exception_captured(self, fresh_registry):
        """Tool exceptions are captured gracefully."""

        async def error_tool(args):
            raise ValueError("Simulated error")

        fresh_registry.register(
            Tool(name="error_tool", description="Errors", handler=error_tool)
        )

        result = await fresh_registry.execute("error_tool", {})

        assert not result.success
        assert "ValueError" in result.error or "Simulated error" in result.error

    @pytest.mark.asyncio
    async def test_tool_key_error_captured(self, fresh_registry):
        """Missing key errors are captured."""

        async def key_error_tool(args):
            return args["missing_key"]  # Will raise KeyError

        fresh_registry.register(
            Tool(name="key_error", description="Key error", handler=key_error_tool)
        )

        result = await fresh_registry.execute("key_error", {})

        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_tool_type_error_captured(self, fresh_registry):
        """Type errors are captured."""

        async def type_error_tool(args):
            return 1 + "string"  # Will raise TypeError

        fresh_registry.register(
            Tool(name="type_error", description="Type error", handler=type_error_tool)
        )

        result = await fresh_registry.execute("type_error", {})

        assert not result.success
        assert result.error is not None


# =============================================================================
# Metrics Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolMetrics:
    """Test tool metrics collection."""

    @pytest.mark.asyncio
    async def test_invocation_count_tracked(self, fresh_registry):
        """Invocation count is tracked."""

        async def counter_tool(args):
            return "counted"

        fresh_registry.register(
            Tool(name="counter", description="Counts", handler=counter_tool)
        )

        # Execute 5 times
        for _ in range(5):
            await fresh_registry.execute("counter", {})

        metrics = fresh_registry.get_metrics("counter")
        assert metrics.invocation_count == 5

    @pytest.mark.asyncio
    async def test_success_count_tracked(self, fresh_registry):
        """Success count is tracked."""

        async def success_tool(args):
            return "success"

        fresh_registry.register(
            Tool(name="success", description="Succeeds", handler=success_tool)
        )

        for _ in range(3):
            await fresh_registry.execute("success", {})

        metrics = fresh_registry.get_metrics("success")
        assert metrics.success_count == 3
        assert metrics.failure_count == 0

    @pytest.mark.asyncio
    async def test_failure_count_tracked(self, fresh_registry):
        """Failure count is tracked."""

        async def fail_tool(args):
            raise Exception("Always fails")

        fresh_registry.register(
            Tool(name="fail", description="Fails", handler=fail_tool)
        )

        for _ in range(3):
            await fresh_registry.execute("fail", {})

        metrics = fresh_registry.get_metrics("fail")
        assert metrics.failure_count == 3
        assert metrics.success_count == 0

    @pytest.mark.asyncio
    async def test_timeout_count_tracked(self, fresh_registry):
        """Timeout count is tracked."""

        async def timeout_tool(args):
            await asyncio.sleep(2)
            return "done"

        fresh_registry.register(
            Tool(
                name="timeout",
                description="Times out",
                handler=timeout_tool,
                timeout_ms=100,
            )
        )

        for _ in range(2):
            await fresh_registry.execute("timeout", {})

        metrics = fresh_registry.get_metrics("timeout")
        assert metrics.timeout_count == 2

    @pytest.mark.asyncio
    async def test_success_rate_calculated(self, fresh_registry):
        """Success rate is calculated correctly."""

        call_count = 0

        async def mixed_tool(args):
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return "success"
            raise Exception("fail")

        fresh_registry.register(
            Tool(name="mixed", description="Mixed results", handler=mixed_tool)
        )

        # 3 successes, 2 failures
        for _ in range(5):
            await fresh_registry.execute("mixed", {})

        metrics = fresh_registry.get_metrics("mixed")
        assert metrics.success_rate == 0.6  # 3/5

    @pytest.mark.asyncio
    async def test_latency_tracked(self, fresh_registry):
        """Latency is tracked accurately."""

        async def timed_tool(args):
            await asyncio.sleep(0.1)  # 100ms
            return "done"

        fresh_registry.register(
            Tool(name="timed", description="Timed", handler=timed_tool)
        )

        await fresh_registry.execute("timed", {})

        metrics = fresh_registry.get_metrics("timed")
        # Should be around 100ms, allow some variance
        assert metrics.avg_latency_ms >= 90
        assert metrics.avg_latency_ms < 300

    @pytest.mark.asyncio
    async def test_average_latency_calculated(self, fresh_registry):
        """Average latency is calculated over multiple invocations."""

        async def variable_tool(args):
            delay = args.get("delay", 0.05)
            await asyncio.sleep(delay)
            return "done"

        fresh_registry.register(
            Tool(name="variable", description="Variable", handler=variable_tool)
        )

        # Two invocations with different delays
        await fresh_registry.execute("variable", {"delay": 0.05})  # 50ms
        await fresh_registry.execute("variable", {"delay": 0.15})  # 150ms

        metrics = fresh_registry.get_metrics("variable")
        # Average should be around 100ms
        assert metrics.invocation_count == 2
        assert 80 < metrics.avg_latency_ms < 200

    @pytest.mark.asyncio
    async def test_last_invoked_tracked(self, fresh_registry):
        """Last invocation timestamp is tracked."""

        async def timestamp_tool(args):
            return "done"

        fresh_registry.register(
            Tool(name="timestamp", description="Timestamp", handler=timestamp_tool)
        )

        before = time.time()
        await fresh_registry.execute("timestamp", {})
        after = time.time()

        metrics = fresh_registry.get_metrics("timestamp")
        assert metrics.last_invoked is not None
        # Timestamp should be between before and after
        invoked_ts = metrics.last_invoked.timestamp()
        assert before <= invoked_ts <= after + 1

    @pytest.mark.asyncio
    async def test_last_error_tracked(self, fresh_registry):
        """Last error message is tracked."""

        async def error_tool(args):
            raise ValueError("Specific error message")

        fresh_registry.register(
            Tool(name="error", description="Errors", handler=error_tool)
        )

        await fresh_registry.execute("error", {})

        metrics = fresh_registry.get_metrics("error")
        assert metrics.last_error is not None
        assert "Specific error message" in metrics.last_error

    def test_reset_metrics(self, fresh_registry):
        """Can reset metrics for a tool."""

        async def counter(args):
            return None

        tool = Tool(name="resettable", description="Resettable", handler=counter)
        tool.metrics.invocation_count = 10
        tool.metrics.success_count = 8

        fresh_registry.register(tool)

        fresh_registry.reset_metrics("resettable")

        metrics = fresh_registry.get_metrics("resettable")
        assert metrics.invocation_count == 0
        assert metrics.success_count == 0

    def test_get_all_metrics(self, fresh_registry):
        """Can get metrics for all tools."""

        async def handler(args):
            return None

        fresh_registry.register(
            Tool(name="tool1", description="Tool 1", handler=handler)
        )
        fresh_registry.register(
            Tool(name="tool2", description="Tool 2", handler=handler)
        )

        all_metrics = fresh_registry.get_all_metrics()

        assert "tool1" in all_metrics
        assert "tool2" in all_metrics
        assert isinstance(all_metrics["tool1"], ToolMetrics)


# =============================================================================
# Schema Generation Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolSchemas:
    """Test tool schema generation."""

    def test_openai_function_format(self, fresh_registry):
        """Tool generates OpenAI function format."""

        async def search(args):
            return []

        tool = Tool(
            name="search",
            description="Search documents",
            handler=search,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    required=True,
                ),
            ],
        )

        schema = tool.to_openai_function()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert "parameters" in schema["function"]
        assert "query" in schema["function"]["parameters"]["properties"]

    def test_get_openai_tools(self, fresh_registry):
        """Registry returns all tools in OpenAI format."""

        async def handler(args):
            return None

        fresh_registry.register(
            Tool(name="tool1", description="Tool 1", handler=handler)
        )
        fresh_registry.register(
            Tool(name="tool2", description="Tool 2", handler=handler)
        )

        tools = fresh_registry.get_openai_tools()

        assert len(tools) == 2
        assert all(t["type"] == "function" for t in tools)

    def test_get_descriptions(self, fresh_registry):
        """Registry returns formatted descriptions."""

        async def handler(args):
            return None

        fresh_registry.register(
            Tool(name="tool_a", description="Does A", handler=handler)
        )
        fresh_registry.register(
            Tool(name="tool_b", description="Does B", handler=handler)
        )

        descriptions = fresh_registry.get_descriptions()

        assert "tool_a" in descriptions
        assert "tool_b" in descriptions
        assert "Does A" in descriptions
        assert "Does B" in descriptions


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.tool_integration
class TestToolPerformance:
    """Test tool execution performance."""

    @pytest.mark.asyncio
    async def test_registration_performance(self, fresh_registry):
        """Tool registration is fast."""
        import time

        async def handler(args):
            return None

        start = time.time()
        for i in range(100):
            fresh_registry.register(
                Tool(name=f"tool_{i}", description=f"Tool {i}", handler=handler)
            )
        elapsed = time.time() - start

        # Registration of 100 tools should be fast (<100ms)
        assert elapsed < 0.1
        assert len(fresh_registry) == 100

    @pytest.mark.asyncio
    async def test_lookup_performance(self, fresh_registry):
        """Tool lookup is fast."""
        import time

        async def handler(args):
            return None

        # Register 100 tools
        for i in range(100):
            fresh_registry.register(
                Tool(name=f"tool_{i}", description=f"Tool {i}", handler=handler)
            )

        start = time.time()
        for i in range(1000):
            fresh_registry.get_tool(f"tool_{i % 100}")
        elapsed = time.time() - start

        # 1000 lookups should be very fast (<50ms)
        assert elapsed < 0.05

    @pytest.mark.asyncio
    async def test_execution_overhead(self, fresh_registry):
        """Tool execution overhead is minimal."""
        import time

        async def fast_handler(args):
            return "done"

        fresh_registry.register(
            Tool(name="fast", description="Fast", handler=fast_handler)
        )

        # Warm up
        await fresh_registry.execute("fast", {})

        start = time.time()
        for _ in range(100):
            await fresh_registry.execute("fast", {})
        elapsed = time.time() - start

        # 100 executions of minimal handler should be fast
        avg_ms = (elapsed * 1000) / 100
        assert avg_ms < 5  # <5ms average per execution


# =============================================================================
# AgentLoop Integration Tests (Require Full Wiring)
# =============================================================================


@pytest.mark.tool_integration
@pytest.mark.skip(reason="Requires full AgentLoop wiring with ActionExecutor - not yet integrated")
class TestToolAgentIntegration:
    """Test tool integration with AgentLoop.

    These tests require the full agent stack with ActionExecutor configured.
    Skipped until agent-tool integration is complete.
    """

    @pytest.mark.asyncio
    async def test_agent_uses_tool_in_response(self, agent, tool_registry, evaluator):
        """Agent uses tool to answer query."""
        pass

    @pytest.mark.asyncio
    async def test_agent_selects_correct_tool(self, agent, tool_registry):
        """Agent selects appropriate tool based on query."""
        pass
