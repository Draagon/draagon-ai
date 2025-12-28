"""Action execution layer.

The action executor takes decisions and executes them using
the appropriate tools.

REQ-002-03: ActionExecutor with Tool Registry
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..behaviors import Action, Behavior
from .protocols import ToolCall, ToolProvider, ToolResult
from .registry import ToolRegistry, ToolExecutionResult


@dataclass
class ActionResult:
    """Result from executing an action."""

    # What was executed
    action_name: str
    success: bool

    # Results
    result: Any = None
    formatted_result: str = ""

    # For errors
    error: str | None = None

    # Timing
    latency_ms: float = 0.0
    executed_at: datetime = field(default_factory=datetime.now)

    # If the action returns a direct answer
    direct_answer: str | None = None

    # Timeout tracking (REQ-002-03)
    timed_out: bool = False


class ActionExecutor:
    """Executes actions using registered tools.

    The executor maps behavior actions to tool handlers and
    manages execution with proper error handling.

    Supports two modes (REQ-002-03):
    1. ToolProvider protocol (legacy) - for compatibility with existing code
    2. ToolRegistry (new) - for dynamic tool registration with metrics and timeouts

    Example with ToolRegistry:
        registry = ToolRegistry()
        registry.register(Tool(name="get_weather", handler=weather_handler, ...))
        executor = ActionExecutor(tool_registry=registry)

    Example with ToolProvider:
        executor = ActionExecutor(tool_provider=my_provider)
    """

    def __init__(
        self,
        tool_provider: ToolProvider | None = None,
        tool_registry: ToolRegistry | None = None,
    ):
        """Initialize the action executor.

        Args:
            tool_provider: Provider that implements tool execution (legacy)
            tool_registry: Tool registry with dynamic registration (new, preferred)

        Raises:
            ValueError: If neither tool_provider nor tool_registry is provided
        """
        if tool_provider is None and tool_registry is None:
            raise ValueError("Must provide either tool_provider or tool_registry")

        self.tool_provider = tool_provider
        self.tool_registry = tool_registry
        self._use_registry = tool_registry is not None

    async def execute(
        self,
        action_name: str,
        args: dict[str, Any],
        behavior: Behavior,
        context: dict[str, Any],
        timeout_override_ms: int | None = None,
    ) -> ActionResult:
        """Execute an action.

        Args:
            action_name: Name of the action to execute
            args: Arguments for the action
            behavior: Behavior the action belongs to
            context: Execution context
            timeout_override_ms: Override tool's default timeout (REQ-002-03)

        Returns:
            ActionResult with execution details
        """
        start_time = datetime.now()

        # Get the action definition
        action = behavior.get_action(action_name)

        # Handle special cases
        if action_name == "answer":
            # Direct answer doesn't need tool execution
            return ActionResult(
                action_name=action_name,
                success=True,
                result=args.get("answer", ""),
                direct_answer=args.get("answer", ""),
            )

        if action_name == "more_details":
            # Return pending details
            return ActionResult(
                action_name=action_name,
                success=True,
                result=context.get("pending_details", ""),
                direct_answer=context.get("pending_details", ""),
            )

        if action_name == "clarify":
            # Return clarification question
            return ActionResult(
                action_name=action_name,
                success=True,
                result=args.get("question", "Could you clarify?"),
                direct_answer=args.get("question", "Could you clarify?"),
            )

        # Map action to tool
        tool_name = self._get_tool_name(action, action_name)

        # Use ToolRegistry if available, otherwise fallback to ToolProvider
        if self._use_registry:
            return await self._execute_via_registry(
                action_name, tool_name, args, context, timeout_override_ms
            )
        else:
            return await self._execute_via_provider(
                action_name, tool_name, args, context, start_time
            )

    async def _execute_via_registry(
        self,
        action_name: str,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
        timeout_override_ms: int | None,
    ) -> ActionResult:
        """Execute via ToolRegistry with timeout and metrics (REQ-002-03).

        Args:
            action_name: Original action name
            tool_name: Tool to execute
            args: Tool arguments
            context: Execution context
            timeout_override_ms: Optional timeout override

        Returns:
            ActionResult with execution details
        """
        assert self.tool_registry is not None

        result = await self.tool_registry.execute(
            tool_name, args, context, timeout_override_ms
        )

        return ActionResult(
            action_name=action_name,
            success=result.success,
            result=result.result,
            formatted_result=self._format_registry_result(tool_name, result),
            error=result.error,
            latency_ms=result.latency_ms,
            timed_out=result.timed_out,
        )

    async def _execute_via_provider(
        self,
        action_name: str,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
        start_time: datetime,
    ) -> ActionResult:
        """Execute via ToolProvider (legacy path).

        Args:
            action_name: Original action name
            tool_name: Tool to execute
            args: Tool arguments
            context: Execution context
            start_time: When execution started

        Returns:
            ActionResult with execution details
        """
        assert self.tool_provider is not None

        # Build tool call
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=args,
        )

        try:
            # Execute via tool provider
            tool_result = await self.tool_provider.execute(tool_call, context)

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ActionResult(
                action_name=action_name,
                success=tool_result.success,
                result=tool_result.result,
                formatted_result=self._format_result(tool_name, tool_result),
                error=tool_result.error,
                latency_ms=latency,
            )

        except Exception as e:
            latency = (datetime.now() - start_time).total_seconds() * 1000

            return ActionResult(
                action_name=action_name,
                success=False,
                error=str(e),
                latency_ms=latency,
            )

    async def execute_multiple(
        self,
        actions: list[tuple[str, dict[str, Any]]],
        behavior: Behavior,
        context: dict[str, Any],
    ) -> list[ActionResult]:
        """Execute multiple actions.

        Args:
            actions: List of (action_name, args) tuples
            behavior: Behavior the actions belong to
            context: Execution context

        Returns:
            List of ActionResults
        """
        results = []
        for action_name, args in actions:
            result = await self.execute(action_name, args, behavior, context)
            results.append(result)
        return results

    def _get_tool_name(self, action: Action | None, action_name: str) -> str:
        """Get the tool name for an action.

        Args:
            action: Action definition (may be None)
            action_name: Name of the action

        Returns:
            Tool name to use
        """
        if action and action.handler:
            return action.handler

        # Default: use action name as tool name
        return action_name

    def _format_result(self, tool_name: str, result: ToolResult) -> str:
        """Format a tool result for display/synthesis.

        Args:
            tool_name: Name of the tool
            result: Tool execution result

        Returns:
            Formatted string
        """
        if not result.success:
            return f"Error: {result.error}"

        return self._format_result_value(result.result)

    def _format_registry_result(
        self, tool_name: str, result: ToolExecutionResult
    ) -> str:
        """Format a ToolRegistry execution result for display/synthesis.

        Args:
            tool_name: Name of the tool
            result: Tool execution result from registry

        Returns:
            Formatted string
        """
        if not result.success:
            if result.timed_out:
                return f"Timeout: Tool {tool_name} did not respond in time"
            return f"Error: {result.error}"

        return self._format_result_value(result.result)

    def _format_result_value(self, r: Any) -> str:
        """Format a result value for display.

        Args:
            r: Result value from tool execution

        Returns:
            Formatted string
        """
        if r is None:
            return "Done."

        if isinstance(r, str):
            return r

        if isinstance(r, dict):
            # Check for common fields
            if "time" in r:
                return f"The time is {r['time']}"
            if "weather" in r:
                return f"Weather: {r['weather']}"
            if "events" in r:
                events = r["events"]
                if not events:
                    return "No events found."
                return f"Found {len(events)} event(s)."

            # Generic dict formatting
            return str(r)

        if isinstance(r, list):
            if not r:
                return "No results found."
            return f"Found {len(r)} result(s)."

        return str(r)

    def validate_action(
        self,
        action_name: str,
        args: dict[str, Any],
        behavior: Behavior,
    ) -> tuple[bool, str | None]:
        """Validate an action before execution.

        Args:
            action_name: Name of the action
            args: Arguments to validate
            behavior: Behavior containing action definition

        Returns:
            Tuple of (is_valid, error_message)
        """
        action = behavior.get_action(action_name)

        if not action:
            return False, f"Unknown action: {action_name}"

        # Check required parameters
        for param_name, param in action.parameters.items():
            if param.required and param_name not in args:
                if param.default is None:
                    return False, f"Missing required parameter: {param_name}"

        # Check blocked actions
        if action_name in behavior.constraints.blocked_actions:
            return False, f"Action is blocked: {action_name}"

        return True, None

    def requires_confirmation(
        self,
        action_name: str,
        behavior: Behavior,
    ) -> bool:
        """Check if an action requires user confirmation.

        Args:
            action_name: Name of the action
            behavior: Behavior containing action definition

        Returns:
            True if confirmation is required
        """
        action = behavior.get_action(action_name)

        if action and action.requires_confirmation:
            return True

        if action_name in behavior.constraints.requires_user_confirmation:
            return True

        # Check registry tool if using registry mode
        if self._use_registry and self.tool_registry:
            tool = self.tool_registry.get_tool(action_name)
            if tool and tool.requires_confirmation:
                return True

        return False

    # =========================================================================
    # Registry convenience methods (REQ-002-03)
    # =========================================================================

    def list_tools(self) -> list[str]:
        """List all available tool names.

        Returns:
            List of tool names
        """
        if self._use_registry and self.tool_registry:
            return self.tool_registry.list_tools()
        elif self.tool_provider:
            return self.tool_provider.list_tools()
        return []

    def get_tool_description(self, name: str) -> str | None:
        """Get description for a tool.

        Args:
            name: Tool name

        Returns:
            Description or None if not found
        """
        if self._use_registry and self.tool_registry:
            return self.tool_registry.get_tool_description(name)
        elif self.tool_provider:
            return self.tool_provider.get_tool_description(name)
        return None

    def get_schemas_for_llm(self) -> list[dict[str, Any]]:
        """Get tool schemas formatted for LLM context.

        Returns:
            List of schema dictionaries
        """
        if self._use_registry and self.tool_registry:
            return self.tool_registry.get_schemas_for_llm()
        return []

    def get_metrics(self, name: str | None = None) -> dict[str, Any]:
        """Get execution metrics for tool(s).

        Args:
            name: Tool name (None = all tools)

        Returns:
            Metrics dictionary
        """
        if not self._use_registry or not self.tool_registry:
            return {}

        if name:
            metrics = self.tool_registry.get_metrics(name)
            if metrics:
                return {
                    "invocation_count": metrics.invocation_count,
                    "success_count": metrics.success_count,
                    "failure_count": metrics.failure_count,
                    "timeout_count": metrics.timeout_count,
                    "success_rate": metrics.success_rate,
                    "avg_latency_ms": metrics.avg_latency_ms,
                    "last_error": metrics.last_error,
                }
            return {}

        all_metrics = self.tool_registry.get_all_metrics()
        return {
            tool_name: {
                "invocation_count": m.invocation_count,
                "success_count": m.success_count,
                "failure_count": m.failure_count,
                "timeout_count": m.timeout_count,
                "success_rate": m.success_rate,
                "avg_latency_ms": m.avg_latency_ms,
            }
            for tool_name, m in all_metrics.items()
        }

    def has_tool(self, name: str) -> bool:
        """Check if a tool is available.

        Args:
            name: Tool name

        Returns:
            True if tool exists
        """
        if self._use_registry and self.tool_registry:
            return self.tool_registry.has_tool(name)
        elif self.tool_provider:
            return name in self.tool_provider.list_tools()
        return False
