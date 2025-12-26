"""Action execution layer.

The action executor takes decisions and executes them using
the appropriate tools.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..behaviors import Action, Behavior
from .protocols import ToolCall, ToolProvider, ToolResult


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


class ActionExecutor:
    """Executes actions using registered tools.

    The executor maps behavior actions to tool handlers and
    manages execution with proper error handling.
    """

    def __init__(
        self,
        tool_provider: ToolProvider,
    ):
        """Initialize the action executor.

        Args:
            tool_provider: Provider that implements tool execution
        """
        self.tool_provider = tool_provider

    async def execute(
        self,
        action_name: str,
        args: dict[str, Any],
        behavior: Behavior,
        context: dict[str, Any],
    ) -> ActionResult:
        """Execute an action.

        Args:
            action_name: Name of the action to execute
            args: Arguments for the action
            behavior: Behavior the action belongs to
            context: Execution context

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

        # Handle common result types
        r = result.result

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

        return False
