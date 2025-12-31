"""Tests for ToolRegistry and ActionExecutor with registry (REQ-002-03).

Tests:
- Tool registration and lookup
- Tool execution with timeout handling
- Metrics collection
- Schema generation for LLM
- ActionExecutor integration with registry
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.orchestration import (
    ToolRegistry,
    Tool,
    ToolParameter,
    ToolMetrics,
    ToolExecutionResult,
    ActionExecutor,
    ActionResult,
)
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_behavior():
    """Create a simple behavior for testing."""
    return Behavior(
        behavior_id="test",
        name="Test Behavior",
        description="For testing",
        actions=[
            Action(name="answer", description="Provide an answer"),
            Action(name="get_weather", description="Get weather"),
            Action(name="search_web", description="Search the web"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide action for: {question}",
            synthesis_prompt="Synthesize: {tool_results}",
        ),
    )


async def simple_handler(args: dict, context: dict | None = None) -> dict:
    """Simple async handler for testing."""
    return {"result": "success", "args": args}


async def slow_handler(args: dict, context: dict | None = None) -> dict:
    """Handler that takes time (for timeout testing)."""
    await asyncio.sleep(2)  # 2 second delay
    return {"result": "completed"}


async def error_handler(args: dict, context: dict | None = None) -> dict:
    """Handler that raises an error."""
    raise ValueError("Test error")


# =============================================================================
# ToolParameter Tests
# =============================================================================


class TestToolParameter:
    """Tests for ToolParameter dataclass."""

    def test_required_parameter(self):
        """Test required parameter with defaults."""
        param = ToolParameter(
            name="query",
            type="string",
            description="Search query",
        )

        assert param.name == "query"
        assert param.type == "string"
        assert param.description == "Search query"
        assert param.required is True
        assert param.enum is None
        assert param.default is None

    def test_optional_parameter_with_default(self):
        """Test optional parameter with default value."""
        param = ToolParameter(
            name="limit",
            type="integer",
            description="Max results",
            required=False,
            default=10,
        )

        assert param.required is False
        assert param.default == 10

    def test_enum_parameter(self):
        """Test parameter with enum values."""
        param = ToolParameter(
            name="format",
            type="string",
            description="Output format",
            enum=["json", "xml", "csv"],
        )

        assert param.enum == ["json", "xml", "csv"]


# =============================================================================
# ToolMetrics Tests
# =============================================================================


class TestToolMetrics:
    """Tests for ToolMetrics tracking."""

    def test_initial_metrics(self):
        """Test initial metric values."""
        metrics = ToolMetrics()

        assert metrics.invocation_count == 0
        assert metrics.success_count == 0
        assert metrics.failure_count == 0
        assert metrics.timeout_count == 0
        assert metrics.success_rate == 1.0  # Default when no invocations
        assert metrics.avg_latency_ms == 0.0

    def test_record_success(self):
        """Test recording a successful invocation."""
        metrics = ToolMetrics()
        metrics.record_success(100.0)

        assert metrics.invocation_count == 1
        assert metrics.success_count == 1
        assert metrics.success_rate == 1.0
        assert metrics.avg_latency_ms == 100.0
        assert metrics.last_invoked is not None

    def test_record_failure(self):
        """Test recording a failed invocation."""
        metrics = ToolMetrics()
        metrics.record_failure(50.0, "Connection timeout")

        assert metrics.invocation_count == 1
        assert metrics.failure_count == 1
        assert metrics.success_rate == 0.0
        assert metrics.last_error == "Connection timeout"

    def test_record_timeout(self):
        """Test recording a timeout."""
        metrics = ToolMetrics()
        metrics.record_timeout(30000.0)

        assert metrics.invocation_count == 1
        assert metrics.timeout_count == 1
        assert metrics.last_error == "Timeout after 30000.0ms"

    def test_mixed_invocations(self):
        """Test success rate with mixed results."""
        metrics = ToolMetrics()
        metrics.record_success(100.0)
        metrics.record_success(100.0)
        metrics.record_failure(50.0, "error")

        assert metrics.invocation_count == 3
        assert metrics.success_count == 2
        assert metrics.failure_count == 1
        assert abs(metrics.success_rate - 2 / 3) < 0.01


# =============================================================================
# Tool Tests
# =============================================================================


class TestTool:
    """Tests for Tool dataclass."""

    def test_tool_creation(self):
        """Test creating a tool with basic properties."""
        tool = Tool(
            name="get_weather",
            description="Get current weather",
            handler=simple_handler,
        )

        assert tool.name == "get_weather"
        assert tool.description == "Get current weather"
        assert tool.parameters == []
        assert tool.timeout_ms == 30000
        assert tool.requires_confirmation is False

    def test_tool_with_parameters(self):
        """Test tool with parameters."""
        tool = Tool(
            name="search",
            description="Search the web",
            handler=simple_handler,
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
                ToolParameter(
                    name="limit",
                    type="integer",
                    description="Max results",
                    required=False,
                    default=5,
                ),
            ],
        )

        assert len(tool.parameters) == 2
        assert tool.parameters[0].name == "query"
        assert tool.parameters[1].required is False

    def test_to_openai_function(self):
        """Test OpenAI function format conversion."""
        tool = Tool(
            name="search",
            description="Search the web",
            handler=simple_handler,
            parameters=[
                ToolParameter(name="query", type="string", description="Search query"),
            ],
        )

        schema = tool.to_openai_function()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "search"
        assert schema["function"]["description"] == "Search the web"
        assert "parameters" in schema["function"]
        assert schema["function"]["parameters"]["properties"]["query"]["type"] == "string"
        assert "query" in schema["function"]["parameters"]["required"]

    def test_to_openai_function_no_params(self):
        """Test OpenAI function format with no parameters."""
        tool = Tool(
            name="get_time",
            description="Get current time",
            handler=simple_handler,
        )

        schema = tool.to_openai_function()

        assert "parameters" not in schema["function"]

    def test_to_prompt_format(self):
        """Test human-readable prompt format."""
        tool = Tool(
            name="search",
            description="Search the web",
            handler=simple_handler,
            parameters=[
                ToolParameter(name="query", type="string", description="Query"),
                ToolParameter(
                    name="limit", type="integer", description="Max", required=False
                ),
            ],
        )

        prompt = tool.to_prompt_format()

        assert "search" in prompt
        assert "query: string" in prompt
        assert "limit: integer (optional)" in prompt

    def test_to_prompt_format_no_params(self):
        """Test prompt format with no parameters."""
        tool = Tool(
            name="get_time",
            description="Get current time",
            handler=simple_handler,
        )

        prompt = tool.to_prompt_format()

        assert "get_time" in prompt
        assert "(no arguments)" in prompt

    def test_to_schema_dict(self):
        """Test schema dictionary conversion."""
        tool = Tool(
            name="search",
            description="Search the web",
            handler=simple_handler,
            parameters=[
                ToolParameter(
                    name="query",
                    type="string",
                    description="Search query",
                    enum=["web", "images"],
                ),
            ],
            returns="List of search results",
        )

        schema = tool.to_schema_dict()

        assert schema["name"] == "search"
        assert schema["description"] == "Search the web"
        assert schema["returns"] == "List of search results"
        assert "query" in schema["parameters"]
        assert schema["parameters"]["query"]["enum"] == ["web", "images"]


# =============================================================================
# ToolRegistry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_empty_registry(self):
        """Test empty registry."""
        registry = ToolRegistry()

        assert len(registry) == 0
        assert registry.list_tools() == []

    def test_register_tool_object(self):
        """Test registering a Tool object."""
        registry = ToolRegistry()
        tool = Tool(
            name="get_time",
            description="Get current time",
            handler=simple_handler,
        )

        registry.register(tool)

        assert len(registry) == 1
        assert "get_time" in registry
        assert registry.has_tool("get_time")

    def test_register_with_kwargs(self):
        """Test registering with keyword arguments."""
        registry = ToolRegistry()

        registry.register(
            name="get_time",
            handler=simple_handler,
            description="Get current time",
        )

        assert "get_time" in registry
        tool = registry.get_tool("get_time")
        assert tool is not None
        assert tool.description == "Get current time"

    def test_register_with_schema(self):
        """Test registering with schema dict."""
        registry = ToolRegistry()

        registry.register(
            name="search",
            handler=simple_handler,
            schema={
                "description": "Search the web",
                "parameters": {
                    "query": {
                        "type": "string",
                        "description": "Search query",
                        "required": True,
                    },
                },
            },
        )

        tool = registry.get_tool("search")
        assert tool is not None
        assert len(tool.parameters) == 1
        assert tool.parameters[0].name == "query"

    def test_register_requires_tool_or_name_handler(self):
        """Test that registration requires either tool or name/handler."""
        registry = ToolRegistry()

        with pytest.raises(ValueError):
            registry.register()  # No arguments

        with pytest.raises(ValueError):
            registry.register(name="test")  # No handler

    def test_unregister_tool(self):
        """Test unregistering a tool."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))

        assert registry.unregister("test") is True
        assert "test" not in registry

    def test_unregister_nonexistent(self):
        """Test unregistering non-existent tool."""
        registry = ToolRegistry()

        assert registry.unregister("nonexistent") is False

    def test_get_tool(self):
        """Test getting a tool by name."""
        registry = ToolRegistry()
        tool = Tool(name="test", description="Test tool", handler=simple_handler)
        registry.register(tool)

        retrieved = registry.get_tool("test")
        assert retrieved is tool

    def test_get_tool_not_found(self):
        """Test getting non-existent tool."""
        registry = ToolRegistry()

        assert registry.get_tool("nonexistent") is None

    def test_list_tools(self):
        """Test listing all tool names."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="", handler=simple_handler))
        registry.register(Tool(name="tool2", description="", handler=simple_handler))

        tools = registry.list_tools()

        assert "tool1" in tools
        assert "tool2" in tools

    def test_get_all_tools(self):
        """Test getting all Tool objects."""
        registry = ToolRegistry()
        tool1 = Tool(name="tool1", description="", handler=simple_handler)
        tool2 = Tool(name="tool2", description="", handler=simple_handler)
        registry.register(tool1)
        registry.register(tool2)

        tools = registry.get_all_tools()

        assert len(tools) == 2

    def test_get_descriptions(self):
        """Test getting formatted descriptions."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="First", handler=simple_handler))
        registry.register(Tool(name="tool2", description="Second", handler=simple_handler))

        descriptions = registry.get_descriptions()

        assert "tool1" in descriptions
        assert "tool2" in descriptions

    def test_get_openai_tools(self):
        """Test getting OpenAI format schemas."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="search",
                description="Search",
                handler=simple_handler,
                parameters=[
                    ToolParameter(name="q", type="string", description="Query"),
                ],
            )
        )

        schemas = registry.get_openai_tools()

        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "search"

    def test_get_schemas_for_llm(self):
        """Test getting schemas for LLM context."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="Test", handler=simple_handler))

        schemas = registry.get_schemas_for_llm()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful tool execution."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))

        result = await registry.execute("test", {"key": "value"})

        assert result.success is True
        assert result.result == {"result": "success", "args": {"key": "value"}}
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """Test executing unknown tool returns error result."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent", {})

        assert result.success is False
        assert "Unknown tool" in result.error

    @pytest.mark.asyncio
    async def test_execute_with_timeout(self):
        """Test tool execution timeout."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="slow",
                description="Slow tool",
                handler=slow_handler,
                timeout_ms=100,  # 100ms timeout, handler takes 2s
            )
        )

        result = await registry.execute("slow", {})

        assert result.success is False
        assert result.timed_out is True
        assert "timed out" in result.error.lower()

    @pytest.mark.asyncio
    async def test_execute_with_timeout_override(self):
        """Test timeout override."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="slow",
                description="Slow tool",
                handler=slow_handler,
                timeout_ms=30000,  # 30s default
            )
        )

        # Override with 100ms timeout
        result = await registry.execute("slow", {}, timeout_override_ms=100)

        assert result.success is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_with_error(self):
        """Test tool execution error captured."""
        registry = ToolRegistry()
        registry.register(Tool(name="error", description="", handler=error_handler))

        result = await registry.execute("error", {})

        assert result.success is False
        assert "Test error" in result.error
        assert result.timed_out is False

    @pytest.mark.asyncio
    async def test_execute_updates_metrics(self):
        """Test that execution updates metrics."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))

        await registry.execute("test", {})
        await registry.execute("test", {})

        metrics = registry.get_metrics("test")
        assert metrics is not None
        assert metrics.invocation_count == 2
        assert metrics.success_count == 2

    def test_get_all_metrics(self):
        """Test getting all tool metrics."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="", handler=simple_handler))
        registry.register(Tool(name="tool2", description="", handler=simple_handler))

        all_metrics = registry.get_all_metrics()

        assert "tool1" in all_metrics
        assert "tool2" in all_metrics

    def test_reset_metrics(self):
        """Test resetting metrics."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))

        # Record some activity
        metrics = registry.get_metrics("test")
        metrics.record_success(100.0)
        assert metrics.invocation_count == 1

        # Reset
        registry.reset_metrics("test")
        metrics = registry.get_metrics("test")
        assert metrics.invocation_count == 0

    def test_reset_all_metrics(self):
        """Test resetting all metrics."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="", handler=simple_handler))
        registry.register(Tool(name="tool2", description="", handler=simple_handler))

        # Record activity
        registry.get_metrics("tool1").record_success(100.0)
        registry.get_metrics("tool2").record_success(100.0)

        # Reset all
        registry.reset_metrics()

        assert registry.get_metrics("tool1").invocation_count == 0
        assert registry.get_metrics("tool2").invocation_count == 0

    def test_contains(self):
        """Test __contains__ magic method."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))

        assert "test" in registry
        assert "nonexistent" not in registry

    def test_iter(self):
        """Test __iter__ magic method."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="", handler=simple_handler))
        registry.register(Tool(name="tool2", description="", handler=simple_handler))

        names = list(registry)

        assert "tool1" in names
        assert "tool2" in names


# =============================================================================
# ActionExecutor with ToolRegistry Tests
# =============================================================================


class TestActionExecutorWithRegistry:
    """Tests for ActionExecutor using ToolRegistry."""

    def test_init_with_registry(self):
        """Test initialization with ToolRegistry."""
        registry = ToolRegistry()
        executor = ActionExecutor(tool_registry=registry)

        assert executor.tool_registry is registry
        assert executor._use_registry is True

    def test_init_requires_provider_or_registry(self):
        """Test that init requires either provider or registry."""
        with pytest.raises(ValueError):
            ActionExecutor()

    @pytest.mark.asyncio
    async def test_execute_via_registry(self, simple_behavior):
        """Test executing action via registry."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_weather",
                description="Get weather",
                handler=simple_handler,
            )
        )
        executor = ActionExecutor(tool_registry=registry)

        result = await executor.execute(
            "get_weather",
            {"location": "NYC"},
            simple_behavior,
            {},
        )

        assert result.success is True
        assert result.action_name == "get_weather"

    @pytest.mark.asyncio
    async def test_execute_answer_action(self, simple_behavior):
        """Test that answer action doesn't use registry."""
        registry = ToolRegistry()
        executor = ActionExecutor(tool_registry=registry)

        result = await executor.execute(
            "answer",
            {"answer": "Hello!"},
            simple_behavior,
            {},
        )

        assert result.success is True
        assert result.direct_answer == "Hello!"

    @pytest.mark.asyncio
    async def test_execute_with_timeout_tracking(self, simple_behavior):
        """Test timeout tracking in result."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_weather",
                description="Get weather",
                handler=slow_handler,
                timeout_ms=100,
            )
        )
        executor = ActionExecutor(tool_registry=registry)

        result = await executor.execute(
            "get_weather",
            {},
            simple_behavior,
            {},
        )

        assert result.success is False
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_execute_with_timeout_override(self, simple_behavior):
        """Test timeout override in execute."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="get_weather",
                description="Get weather",
                handler=slow_handler,
                timeout_ms=30000,  # Long default
            )
        )
        executor = ActionExecutor(tool_registry=registry)

        result = await executor.execute(
            "get_weather",
            {},
            simple_behavior,
            {},
            timeout_override_ms=100,  # Short override
        )

        assert result.success is False
        assert result.timed_out is True

    def test_list_tools(self):
        """Test listing tools from registry."""
        registry = ToolRegistry()
        registry.register(Tool(name="tool1", description="", handler=simple_handler))
        registry.register(Tool(name="tool2", description="", handler=simple_handler))
        executor = ActionExecutor(tool_registry=registry)

        tools = executor.list_tools()

        assert "tool1" in tools
        assert "tool2" in tools

    def test_get_tool_description(self):
        """Test getting tool description."""
        registry = ToolRegistry()
        registry.register(
            Tool(name="test", description="Test description", handler=simple_handler)
        )
        executor = ActionExecutor(tool_registry=registry)

        desc = executor.get_tool_description("test")

        assert desc == "Test description"

    def test_get_schemas_for_llm(self):
        """Test getting schemas for LLM."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="Test", handler=simple_handler))
        executor = ActionExecutor(tool_registry=registry)

        schemas = executor.get_schemas_for_llm()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "test"

    @pytest.mark.asyncio
    async def test_get_metrics(self, simple_behavior):
        """Test getting execution metrics."""
        registry = ToolRegistry()
        registry.register(
            Tool(name="get_weather", description="", handler=simple_handler)
        )
        executor = ActionExecutor(tool_registry=registry)

        # Execute a few times
        await executor.execute("get_weather", {}, simple_behavior, {})
        await executor.execute("get_weather", {}, simple_behavior, {})

        metrics = executor.get_metrics("get_weather")

        assert metrics["invocation_count"] == 2
        assert metrics["success_count"] == 2

    @pytest.mark.asyncio
    async def test_get_all_metrics(self, simple_behavior):
        """Test getting all metrics."""
        registry = ToolRegistry()
        registry.register(
            Tool(name="get_weather", description="", handler=simple_handler)
        )
        registry.register(
            Tool(name="search_web", description="", handler=simple_handler)
        )
        executor = ActionExecutor(tool_registry=registry)

        await executor.execute("get_weather", {}, simple_behavior, {})
        await executor.execute("search_web", {}, simple_behavior, {})

        metrics = executor.get_metrics()

        assert "get_weather" in metrics
        assert "search_web" in metrics

    def test_has_tool(self):
        """Test checking if tool exists."""
        registry = ToolRegistry()
        registry.register(Tool(name="test", description="", handler=simple_handler))
        executor = ActionExecutor(tool_registry=registry)

        assert executor.has_tool("test") is True
        assert executor.has_tool("nonexistent") is False

    def test_requires_confirmation_from_registry(self, simple_behavior):
        """Test requires_confirmation checks registry tool."""
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="dangerous",
                description="Dangerous action",
                handler=simple_handler,
                requires_confirmation=True,
            )
        )
        executor = ActionExecutor(tool_registry=registry)

        assert executor.requires_confirmation("dangerous", simple_behavior) is True


# =============================================================================
# ActionExecutor with ToolProvider Tests (Legacy)
# =============================================================================


class TestActionExecutorWithProvider:
    """Tests for ActionExecutor using ToolProvider (legacy path)."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock tool provider."""
        from draagon_ai.orchestration.protocols import ToolResult

        provider = MagicMock()
        provider.execute = AsyncMock(
            return_value=ToolResult(
                tool_name="test",
                success=True,
                result={"data": "test"},
            )
        )
        provider.list_tools = MagicMock(return_value=["test", "search"])
        provider.get_tool_description = MagicMock(return_value="Test tool")
        return provider

    def test_init_with_provider(self, mock_provider):
        """Test initialization with ToolProvider."""
        executor = ActionExecutor(tool_provider=mock_provider)

        assert executor.tool_provider is mock_provider
        assert executor._use_registry is False

    @pytest.mark.asyncio
    async def test_execute_via_provider(self, mock_provider, simple_behavior):
        """Test executing action via provider."""
        executor = ActionExecutor(tool_provider=mock_provider)

        result = await executor.execute(
            "get_weather",
            {"location": "NYC"},
            simple_behavior,
            {},
        )

        assert result.success is True
        mock_provider.execute.assert_called_once()

    def test_list_tools_from_provider(self, mock_provider):
        """Test listing tools from provider."""
        executor = ActionExecutor(tool_provider=mock_provider)

        tools = executor.list_tools()

        assert "test" in tools
        assert "search" in tools

    def test_get_tool_description_from_provider(self, mock_provider):
        """Test getting description from provider."""
        executor = ActionExecutor(tool_provider=mock_provider)

        desc = executor.get_tool_description("test")

        assert desc == "Test tool"

    def test_get_schemas_empty_for_provider(self, mock_provider):
        """Test schemas not available for provider mode."""
        executor = ActionExecutor(tool_provider=mock_provider)

        schemas = executor.get_schemas_for_llm()

        assert schemas == []

    def test_get_metrics_empty_for_provider(self, mock_provider):
        """Test metrics not available for provider mode."""
        executor = ActionExecutor(tool_provider=mock_provider)

        metrics = executor.get_metrics()

        assert metrics == {}

    def test_has_tool_from_provider(self, mock_provider):
        """Test has_tool with provider."""
        executor = ActionExecutor(tool_provider=mock_provider)

        assert executor.has_tool("test") is True
        assert executor.has_tool("nonexistent") is False


# =============================================================================
# Additional Tests for Execution Coverage (REQ-002-08)
# =============================================================================


class TestExecutorBuiltInActions:
    """Tests for built-in action handling (more_details, clarify)."""

    @pytest.fixture
    def empty_registry(self):
        """Create a minimal empty registry for testing built-ins."""
        return ToolRegistry()

    @pytest.fixture
    def behavior(self):
        """Create a behavior with built-in actions."""
        return Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[
                Action(name="more_details", description="Get more details"),
                Action(name="clarify", description="Ask for clarification"),
            ],
        )

    @pytest.mark.asyncio
    async def test_more_details_action(self, behavior, empty_registry):
        """Test that more_details returns pending_details from context."""
        executor = ActionExecutor(tool_registry=empty_registry)
        context = {"pending_details": "Here are the full details you requested."}

        result = await executor.execute(
            action_name="more_details",
            args={},
            behavior=behavior,
            context=context,
        )

        assert result.success is True
        assert result.direct_answer == "Here are the full details you requested."

    @pytest.mark.asyncio
    async def test_more_details_empty_context(self, behavior, empty_registry):
        """Test more_details with no pending_details."""
        executor = ActionExecutor(tool_registry=empty_registry)

        result = await executor.execute(
            action_name="more_details",
            args={},
            behavior=behavior,
            context={},
        )

        assert result.success is True
        assert result.direct_answer == ""

    @pytest.mark.asyncio
    async def test_clarify_action(self, behavior, empty_registry):
        """Test that clarify returns question from args."""
        executor = ActionExecutor(tool_registry=empty_registry)

        result = await executor.execute(
            action_name="clarify",
            args={"question": "What do you mean by X?"},
            behavior=behavior,
            context={},
        )

        assert result.success is True
        assert result.direct_answer == "What do you mean by X?"

    @pytest.mark.asyncio
    async def test_clarify_default_question(self, behavior, empty_registry):
        """Test clarify with no question arg uses default."""
        executor = ActionExecutor(tool_registry=empty_registry)

        result = await executor.execute(
            action_name="clarify",
            args={},
            behavior=behavior,
            context={},
        )

        assert result.success is True
        assert result.direct_answer == "Could you clarify?"


class TestExecutorExecuteMultiple:
    """Tests for execute_multiple."""

    @pytest.fixture
    def registry_executor(self):
        """Create an executor with registry and tools."""
        registry = ToolRegistry()
        registry.register(Tool(
            name="get_time",
            description="Get time",
            handler=simple_handler,
        ))
        registry.register(Tool(
            name="get_weather",
            description="Get weather",
            handler=simple_handler,
        ))
        return ActionExecutor(tool_registry=registry)

    @pytest.fixture
    def behavior(self):
        """Create a behavior with multiple actions."""
        return Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[
                Action(name="get_time", description="Get time"),
                Action(name="get_weather", description="Get weather"),
            ],
        )

    @pytest.mark.asyncio
    async def test_execute_multiple_actions(self, registry_executor, behavior):
        """Test executing multiple actions in sequence."""
        actions = [
            ("get_time", {"format": "12h"}),
            ("get_weather", {"location": "NYC"}),
        ]

        results = await registry_executor.execute_multiple(
            actions=actions,
            behavior=behavior,
            context={},
        )

        assert len(results) == 2
        assert all(r.success for r in results)


class TestExecutorValidation:
    """Tests for action validation."""

    @pytest.fixture
    def empty_registry(self):
        """Create a minimal empty registry for validation tests."""
        return ToolRegistry()

    @pytest.fixture
    def behavior(self):
        """Create a behavior with parameterized actions."""
        from draagon_ai.behaviors import ActionParameter
        return Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[
                Action(
                    name="search",
                    description="Search",
                    parameters={
                        "query": ActionParameter(
                            name="query",
                            description="Search query",
                            type="string",
                            required=True,
                        ),
                    },
                ),
                Action(
                    name="blocked_action",
                    description="Blocked",
                ),
            ],
            constraints=MagicMock(
                blocked_actions=["blocked_action"],
                requires_user_confirmation=["confirm_action"],
            ),
        )

    def test_validate_unknown_action(self, behavior, empty_registry):
        """Test validation of unknown action."""
        executor = ActionExecutor(tool_registry=empty_registry)

        is_valid, error = executor.validate_action("unknown", {}, behavior)

        assert is_valid is False
        assert "Unknown action" in error

    def test_validate_missing_required_param(self, behavior, empty_registry):
        """Test validation catches missing required parameter."""
        executor = ActionExecutor(tool_registry=empty_registry)

        is_valid, error = executor.validate_action("search", {}, behavior)

        assert is_valid is False
        assert "Missing required parameter" in error

    def test_validate_blocked_action(self, behavior, empty_registry):
        """Test validation of blocked action."""
        executor = ActionExecutor(tool_registry=empty_registry)

        is_valid, error = executor.validate_action("blocked_action", {}, behavior)

        assert is_valid is False
        assert "blocked" in error.lower()

    def test_validate_valid_action(self, behavior, empty_registry):
        """Test validation of valid action passes."""
        executor = ActionExecutor(tool_registry=empty_registry)

        is_valid, error = executor.validate_action("search", {"query": "test"}, behavior)

        assert is_valid is True
        assert error is None


class TestExecutorConfirmation:
    """Tests for requires_confirmation."""

    @pytest.fixture
    def empty_registry(self):
        """Create a minimal empty registry for confirmation tests."""
        return ToolRegistry()

    @pytest.fixture
    def behavior(self):
        """Create a behavior with confirmation requirements."""
        return Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[
                Action(
                    name="delete",
                    description="Delete",
                    requires_confirmation=True,
                ),
                Action(
                    name="search",
                    description="Search",
                ),
            ],
            constraints=MagicMock(
                blocked_actions=[],
                requires_user_confirmation=["list_action"],
            ),
        )

    def test_requires_confirmation_from_action(self, behavior, empty_registry):
        """Test confirmation required from action definition."""
        executor = ActionExecutor(tool_registry=empty_registry)

        assert executor.requires_confirmation("delete", behavior) is True
        assert executor.requires_confirmation("search", behavior) is False

    def test_requires_confirmation_from_constraints(self, behavior, empty_registry):
        """Test confirmation required from constraints list."""
        executor = ActionExecutor(tool_registry=empty_registry)

        assert executor.requires_confirmation("list_action", behavior) is True

    def test_requires_confirmation_from_registry(self, behavior):
        """Test confirmation required from registry tool."""
        registry = ToolRegistry()
        registry.register(Tool(
            name="registry_confirm",
            description="Needs confirmation",
            handler=simple_handler,
            requires_confirmation=True,
        ))
        executor = ActionExecutor(tool_registry=registry)

        assert executor.requires_confirmation("registry_confirm", behavior) is True


class TestExecutorFormatting:
    """Tests for result formatting methods."""

    @pytest.fixture
    def executor(self):
        """Create an executor with empty registry for formatting tests."""
        return ActionExecutor(tool_registry=ToolRegistry())

    def test_format_result_value_none(self, executor):
        """Test formatting None returns 'Done.'"""
        result = executor._format_result_value(None)

        assert result == "Done."

    def test_format_result_value_string(self, executor):
        """Test formatting string returns as-is."""
        result = executor._format_result_value("Hello world")

        assert result == "Hello world"

    def test_format_result_value_dict_with_time(self, executor):
        """Test formatting dict with time field."""
        result = executor._format_result_value({"time": "3:00 PM"})

        assert "3:00 PM" in result

    def test_format_result_value_dict_with_weather(self, executor):
        """Test formatting dict with weather field."""
        result = executor._format_result_value({"weather": "Sunny, 72°F"})

        assert "Sunny, 72°F" in result

    def test_format_result_value_dict_with_events(self, executor):
        """Test formatting dict with events field."""
        result = executor._format_result_value({"events": [1, 2, 3]})

        assert "3 event(s)" in result

    def test_format_result_value_dict_empty_events(self, executor):
        """Test formatting dict with empty events."""
        result = executor._format_result_value({"events": []})

        assert "No events found" in result

    def test_format_result_value_dict_generic(self, executor):
        """Test formatting generic dict."""
        result = executor._format_result_value({"key": "value"})

        assert "key" in result or "value" in result

    def test_format_result_value_list_empty(self, executor):
        """Test formatting empty list."""
        result = executor._format_result_value([])

        assert "No results found" in result

    def test_format_result_value_list_with_items(self, executor):
        """Test formatting list with items shows actual content."""
        result = executor._format_result_value([1, 2, 3, 4])

        # Format changed to show actual content: "1; 2; 3; 4"
        assert "1" in result
        assert "2" in result
        assert "3" in result
        assert "4" in result

    def test_format_result_value_other_type(self, executor):
        """Test formatting other types returns str()."""
        result = executor._format_result_value(42)

        assert result == "42"


class TestExecutorNoTools:
    """Tests for executor with empty registry (no tools registered)."""

    @pytest.fixture
    def executor(self):
        """Create an executor with an empty registry (no tools)."""
        return ActionExecutor(tool_registry=ToolRegistry())

    def test_has_tool_empty_registry(self, executor):
        """Test has_tool returns False with empty registry."""
        assert executor.has_tool("anything") is False

    def test_list_tools_empty_registry(self, executor):
        """Test list_tools returns empty with empty registry."""
        tools = executor.list_tools()

        assert tools == []

    def test_get_description_empty_registry(self, executor):
        """Test get_tool_description returns None with empty registry."""
        desc = executor.get_tool_description("anything")

        assert desc is None
