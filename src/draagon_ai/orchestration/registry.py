"""Tool registry for dynamic tool registration and management.

This module provides a ToolRegistry that allows applications to dynamically
register tools at startup. The registry manages tool definitions, schemas,
and execution metrics.

REQ-002-03: ActionExecutor with Tool Registry
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Definition Types
# =============================================================================


@dataclass
class ToolParameter:
    """Definition of a tool parameter.

    Attributes:
        name: Parameter name
        type: JSON Schema type (string, number, boolean, object, array, integer)
        description: Human-readable description
        required: Whether the parameter is required
        enum: Optional list of allowed values
        default: Optional default value
    """

    name: str
    type: str  # "string", "number", "boolean", "object", "array", "integer"
    description: str
    required: bool = True
    enum: list[str] | None = None
    default: Any = None


@dataclass
class ToolMetrics:
    """Execution metrics for a tool.

    Tracks success/failure rates and latency statistics.
    """

    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    timeout_count: int = 0
    total_latency_ms: float = 0.0
    last_invoked: datetime | None = None
    last_error: str | None = None

    @property
    def success_rate(self) -> float:
        """Get the success rate (0.0 to 1.0)."""
        if self.invocation_count == 0:
            return 1.0
        return self.success_count / self.invocation_count

    @property
    def avg_latency_ms(self) -> float:
        """Get average latency in milliseconds."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_latency_ms / self.invocation_count

    def record_success(self, latency_ms: float) -> None:
        """Record a successful invocation."""
        self.invocation_count += 1
        self.success_count += 1
        self.total_latency_ms += latency_ms
        self.last_invoked = datetime.now()

    def record_failure(self, latency_ms: float, error: str) -> None:
        """Record a failed invocation."""
        self.invocation_count += 1
        self.failure_count += 1
        self.total_latency_ms += latency_ms
        self.last_invoked = datetime.now()
        self.last_error = error

    def record_timeout(self, timeout_ms: float) -> None:
        """Record a timeout."""
        self.invocation_count += 1
        self.timeout_count += 1
        self.total_latency_ms += timeout_ms
        self.last_invoked = datetime.now()
        self.last_error = f"Timeout after {timeout_ms}ms"


# Type alias for tool handlers
ToolHandler = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class Tool:
    """Definition of an available tool with structured schema.

    A Tool encapsulates everything needed to execute and describe
    a capability:
    - Handler function for execution
    - Parameters with types and descriptions
    - Timeout configuration
    - Execution metrics

    Attributes:
        name: Unique tool name (e.g., "get_weather", "search_web")
        description: Human-readable description for LLM
        handler: Async function that executes the tool
        parameters: List of parameter definitions
        returns: Description of return type
        timeout_ms: Execution timeout in milliseconds (default: 30000)
        requires_confirmation: Whether user confirmation is needed
        metrics: Execution metrics (auto-tracked)
    """

    name: str
    description: str
    handler: ToolHandler
    parameters: list[ToolParameter] = field(default_factory=list)
    returns: str = "object"
    timeout_ms: int = 30000  # Default 30 second timeout
    requires_confirmation: bool = False
    metrics: ToolMetrics = field(default_factory=ToolMetrics)

    def to_openai_function(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format.

        Returns:
            Dictionary in OpenAI function schema format
        """
        properties = {}
        required = []

        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

            if param.required:
                required.append(param.name)

        schema: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
            },
        }

        if properties:
            schema["function"]["parameters"] = {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        return schema

    def to_prompt_format(self) -> str:
        """Convert to human-readable prompt format.

        Returns:
            String suitable for injection into prompts
        """
        if not self.parameters:
            return f"- {self.name}: {self.description} (no arguments)"

        args = []
        for p in self.parameters:
            req = "" if p.required else " (optional)"
            args.append(f"{p.name}: {p.type}{req}")

        return f"- {self.name}({', '.join(args)}): {self.description}"

    def to_schema_dict(self) -> dict[str, Any]:
        """Convert to simple schema dictionary.

        Returns:
            Dictionary with name, description, and parameters
        """
        params = {}
        for p in self.parameters:
            params[p.name] = {
                "type": p.type,
                "description": p.description,
                "required": p.required,
            }
            if p.enum:
                params[p.name]["enum"] = p.enum
            if p.default is not None:
                params[p.name]["default"] = p.default

        return {
            "name": self.name,
            "description": self.description,
            "parameters": params,
            "returns": self.returns,
        }


# =============================================================================
# Tool Execution Result
# =============================================================================


@dataclass
class ToolExecutionResult:
    """Result from executing a tool.

    Contains the execution result along with success status,
    timing, and any error information.
    """

    tool_name: str
    success: bool
    result: Any = None
    error: str | None = None
    latency_ms: float = 0.0
    timed_out: bool = False
    executed_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# Tool Registry
# =============================================================================


class ToolRegistry:
    """Registry of available tools with structured schemas.

    The ToolRegistry manages tool registration, lookup, and execution.
    It provides:
    - Dynamic tool registration at startup
    - Schema generation for LLM prompts
    - Execution with timeout handling
    - Metrics collection

    Example:
        registry = ToolRegistry()

        # Register a tool
        registry.register(
            Tool(
                name="get_weather",
                description="Get current weather",
                handler=weather_handler,
                parameters=[
                    ToolParameter(
                        name="location",
                        type="string",
                        description="Location name",
                        required=False,
                    )
                ],
            )
        )

        # Execute a tool
        result = await registry.execute("get_weather", {"location": "NYC"})

        # Get schemas for LLM
        schemas = registry.get_schemas_for_llm()
    """

    def __init__(self) -> None:
        """Initialize an empty tool registry."""
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        tool: Tool | None = None,
        *,
        name: str | None = None,
        handler: ToolHandler | None = None,
        schema: dict[str, Any] | None = None,
        description: str = "",
        parameters: list[ToolParameter] | None = None,
        timeout_ms: int = 30000,
    ) -> None:
        """Register a tool.

        Can be called with a Tool object or with separate arguments.

        Args:
            tool: Complete Tool object (preferred)
            name: Tool name (if not using Tool object)
            handler: Async handler function (if not using Tool object)
            schema: Optional schema dict with parameters (alternative to parameters list)
            description: Tool description
            parameters: List of ToolParameter objects
            timeout_ms: Execution timeout in milliseconds

        Raises:
            ValueError: If neither tool nor name/handler are provided
        """
        if tool is not None:
            self._tools[tool.name] = tool
            logger.debug(f"Registered tool: {tool.name}")
            return

        if name is None or handler is None:
            raise ValueError("Must provide either a Tool object or name and handler")

        # Build parameters from schema if provided
        param_list: list[ToolParameter] = []
        if schema and "parameters" in schema:
            for pname, pdef in schema["parameters"].items():
                param_list.append(
                    ToolParameter(
                        name=pname,
                        type=pdef.get("type", "string"),
                        description=pdef.get("description", ""),
                        required=pdef.get("required", True),
                        enum=pdef.get("enum"),
                        default=pdef.get("default"),
                    )
                )
        elif parameters:
            param_list = parameters

        # Determine description - use explicit description, then schema description
        tool_description = description
        if not tool_description and schema:
            tool_description = schema.get("description", "")

        tool = Tool(
            name=name,
            description=tool_description,
            handler=handler,
            parameters=param_list,
            timeout_ms=timeout_ms,
        )

        self._tools[name] = tool
        logger.debug(f"Registered tool: {name}")

    def unregister(self, name: str) -> bool:
        """Unregister a tool.

        Args:
            name: Name of tool to remove

        Returns:
            True if tool was removed, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.debug(f"Unregistered tool: {name}")
            return True
        return False

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool object or None if not found
        """
        return self._tools.get(name)

    def has_tool(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        return name in self._tools

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def get_all_tools(self) -> list[Tool]:
        """Get all registered tools.

        Returns:
            List of Tool objects
        """
        return list(self._tools.values())

    def get_tool_description(self, name: str) -> str | None:
        """Get description of a tool.

        Args:
            name: Tool name

        Returns:
            Description or None if not found
        """
        tool = self._tools.get(name)
        return tool.description if tool else None

    def get_descriptions(self) -> str:
        """Get formatted tool descriptions for prompts.

        Returns:
            Multi-line string with all tool descriptions
        """
        return "\n".join(tool.to_prompt_format() for tool in self._tools.values())

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI function calling format.

        Returns:
            List of OpenAI function schemas
        """
        return [tool.to_openai_function() for tool in self._tools.values()]

    def get_schemas_for_llm(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM context.

        Returns:
            List of schema dictionaries
        """
        return [tool.to_schema_dict() for tool in self._tools.values()]

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: dict[str, Any] | None = None,
        timeout_override_ms: int | None = None,
    ) -> ToolExecutionResult:
        """Execute a tool with timeout handling.

        Args:
            name: Tool name
            args: Tool arguments
            context: Optional execution context
            timeout_override_ms: Override tool's default timeout

        Returns:
            ToolExecutionResult with success/error information
        """
        tool = self._tools.get(name)

        if tool is None:
            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )

        timeout_ms = timeout_override_ms or tool.timeout_ms
        start_time = time.time()

        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                tool.handler(args, context=context) if context else tool.handler(args),
                timeout=timeout_ms / 1000.0,
            )

            latency_ms = (time.time() - start_time) * 1000
            tool.metrics.record_success(latency_ms)

            return ToolExecutionResult(
                tool_name=name,
                success=True,
                result=result,
                latency_ms=latency_ms,
            )

        except asyncio.TimeoutError:
            latency_ms = (time.time() - start_time) * 1000
            tool.metrics.record_timeout(latency_ms)

            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=f"Tool execution timed out after {timeout_ms}ms",
                latency_ms=latency_ms,
                timed_out=True,
            )

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)
            tool.metrics.record_failure(latency_ms, error_msg)

            logger.error(f"Tool {name} failed: {error_msg}", exc_info=True)

            return ToolExecutionResult(
                tool_name=name,
                success=False,
                error=error_msg,
                latency_ms=latency_ms,
            )

    def get_metrics(self, name: str) -> ToolMetrics | None:
        """Get metrics for a specific tool.

        Args:
            name: Tool name

        Returns:
            ToolMetrics or None if not found
        """
        tool = self._tools.get(name)
        return tool.metrics if tool else None

    def get_all_metrics(self) -> dict[str, ToolMetrics]:
        """Get metrics for all tools.

        Returns:
            Dictionary mapping tool names to their metrics
        """
        return {name: tool.metrics for name, tool in self._tools.items()}

    def reset_metrics(self, name: str | None = None) -> None:
        """Reset metrics for a tool or all tools.

        Args:
            name: Tool name (None = reset all)
        """
        if name:
            tool = self._tools.get(name)
            if tool:
                tool.metrics = ToolMetrics()
        else:
            for tool in self._tools.values():
                tool.metrics = ToolMetrics()

    def __len__(self) -> int:
        """Get number of registered tools."""
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        """Check if tool is registered."""
        return name in self._tools

    def __iter__(self):
        """Iterate over tool names."""
        return iter(self._tools)
