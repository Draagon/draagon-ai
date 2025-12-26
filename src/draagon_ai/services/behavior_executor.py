"""Behavior Executor - Runtime for executing behaviors.

This module provides the execution runtime that allows behaviors to:
1. Process user queries using their decision prompts
2. Execute selected actions
3. Format responses using synthesis prompts
4. Track execution metrics

This is the "real" implementation that makes behaviors work in production.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol

from draagon_ai.behaviors.types import (
    Action,
    Behavior,
    BehaviorStatus,
)
from draagon_ai.llm.base import LLMProvider, ModelTier

logger = logging.getLogger(__name__)


# =============================================================================
# Tool System
# =============================================================================


class ToolHandler(Protocol):
    """Protocol for tool handlers."""

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the tool with given parameters."""
        ...


@dataclass
class ToolDefinition:
    """Definition of an available tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler


@dataclass
class ToolResult:
    """Result from executing a tool."""

    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    execution_time_ms: float = 0.0


class ToolRegistry:
    """Registry of available tools for behavior execution."""

    def __init__(self):
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        handler: ToolHandler,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool handler."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=handler,
        )

    def get(self, name: str) -> ToolDefinition | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    async def execute(self, name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name."""
        tool = self._tools.get(name)
        if not tool:
            return ToolResult(
                tool_name=name,
                success=False,
                result=None,
                error=f"Tool not found: {name}",
            )

        start = time.time()
        try:
            result = await tool.handler(**kwargs)
            return ToolResult(
                tool_name=name,
                success=True,
                result=result,
                execution_time_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Tool {name} execution failed: {e}")
            return ToolResult(
                tool_name=name,
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(time.time() - start) * 1000,
            )


# =============================================================================
# Execution Context and Results
# =============================================================================


@dataclass
class ExecutionContext:
    """Context for behavior execution."""

    user_id: str
    query: str
    conversation_history: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExecutionDecision:
    """Decision made by the behavior's decision prompt."""

    action_name: str
    parameters: dict[str, Any]
    reasoning: str
    confidence: float = 1.0
    raw_response: str = ""


@dataclass
class ExecutionResult:
    """Complete result from executing a behavior."""

    success: bool
    response: str
    action_taken: str | None = None
    action_result: Any = None
    decision: ExecutionDecision | None = None
    tool_result: ToolResult | None = None
    error: str | None = None

    # Timing
    decision_time_ms: float = 0.0
    execution_time_ms: float = 0.0
    synthesis_time_ms: float = 0.0
    total_time_ms: float = 0.0


# =============================================================================
# Behavior Executor
# =============================================================================


class BehaviorExecutor:
    """Executes behaviors against user queries.

    This is the runtime that makes behaviors work:
    1. Takes a user query
    2. Uses the behavior's decision prompt to select an action
    3. Executes the action using registered tools
    4. Formats the response using the synthesis prompt

    Usage:
        executor = BehaviorExecutor(llm=my_llm)
        executor.register_tool("set_timer", set_timer_handler)

        result = await executor.execute(behavior, context)
        print(result.response)
    """

    def __init__(
        self,
        llm: LLMProvider,
        tool_registry: ToolRegistry | None = None,
    ):
        """Initialize the executor.

        Args:
            llm: LLM provider for decision and synthesis
            tool_registry: Optional pre-configured tool registry
        """
        self.llm = llm
        self.tools = tool_registry or ToolRegistry()

        # Execution stats
        self._executions = 0
        self._successes = 0
        self._failures = 0

    def register_tool(
        self,
        name: str,
        handler: ToolHandler,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Register a tool for behavior execution."""
        self.tools.register(name, handler, description, parameters)

    async def execute(
        self,
        behavior: Behavior,
        context: ExecutionContext,
    ) -> ExecutionResult:
        """Execute a behavior for a given context.

        Args:
            behavior: The behavior to execute
            context: Execution context with query and metadata

        Returns:
            ExecutionResult with response and execution details
        """
        start_total = time.time()
        self._executions += 1

        try:
            # Validate behavior is executable
            if not behavior.prompts:
                return ExecutionResult(
                    success=False,
                    response="I'm sorry, I'm not configured properly.",
                    error="Behavior has no prompts",
                    total_time_ms=(time.time() - start_total) * 1000,
                )

            if behavior.status in (BehaviorStatus.RETIRED, BehaviorStatus.DEPRECATED):
                return ExecutionResult(
                    success=False,
                    response="This feature is currently unavailable.",
                    error="Behavior is disabled",
                    total_time_ms=(time.time() - start_total) * 1000,
                )

            # Phase 1: Decision - Choose action
            decision, decision_time = await self._make_decision(behavior, context)

            if not decision:
                return ExecutionResult(
                    success=False,
                    response="I couldn't understand your request. Could you rephrase it?",
                    error="Decision phase failed",
                    decision_time_ms=decision_time,
                    total_time_ms=(time.time() - start_total) * 1000,
                )

            # Phase 2: Execute - Run the action
            tool_result, exec_time = await self._execute_action(
                behavior, decision, context
            )

            # Phase 3: Synthesize - Format response
            response, synth_time = await self._synthesize_response(
                behavior, decision, tool_result, context
            )

            self._successes += 1

            return ExecutionResult(
                success=True,
                response=response,
                action_taken=decision.action_name,
                action_result=tool_result.result if tool_result else None,
                decision=decision,
                tool_result=tool_result,
                decision_time_ms=decision_time,
                execution_time_ms=exec_time,
                synthesis_time_ms=synth_time,
                total_time_ms=(time.time() - start_total) * 1000,
            )

        except Exception as e:
            self._failures += 1
            logger.error(f"Behavior execution failed: {e}", exc_info=True)
            return ExecutionResult(
                success=False,
                response="I encountered an error processing your request.",
                error=str(e),
                total_time_ms=(time.time() - start_total) * 1000,
            )

    async def _make_decision(
        self,
        behavior: Behavior,
        context: ExecutionContext,
    ) -> tuple[ExecutionDecision | None, float]:
        """Use decision prompt to choose an action."""
        start = time.time()

        if not behavior.prompts or not behavior.prompts.decision_prompt:
            return None, (time.time() - start) * 1000

        # Build action descriptions
        action_docs = []
        for action in behavior.actions:
            params = []
            for pname, pinfo in action.parameters.items():
                required = pinfo.required if hasattr(pinfo, "required") else True
                params.append(f"    - {pname}: {pinfo.description} (required: {required})")

            action_docs.append(
                f"ACTION: {action.name}\n"
                f"  Description: {action.description}\n"
                f"  Parameters:\n" + "\n".join(params) if params else ""
            )

        # Format the decision prompt
        prompt = behavior.prompts.decision_prompt
        prompt = prompt.replace("{{query}}", context.query)
        prompt = prompt.replace("{query}", context.query)
        prompt = prompt.replace("{{context}}", str(context.metadata))
        prompt = prompt.replace("{context}", str(context.metadata))
        prompt = prompt.replace("{{actions}}", "\n\n".join(action_docs))
        prompt = prompt.replace("{actions}", "\n\n".join(action_docs))

        # Add available actions if not in prompt
        if "AVAILABLE ACTIONS" not in prompt:
            prompt = f"AVAILABLE ACTIONS:\n{''.join(action_docs)}\n\n{prompt}"

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="You are a decision-making assistant. Analyze the query and choose the most appropriate action.",
                temperature=0.3,
                tier=ModelTier.FAST,
            )

            decision = self._parse_decision(response, behavior)
            return decision, (time.time() - start) * 1000

        except Exception as e:
            logger.error(f"Decision failed: {e}")
            return None, (time.time() - start) * 1000

    def _parse_decision(
        self,
        response: str,
        behavior: Behavior,
    ) -> ExecutionDecision | None:
        """Parse LLM response into a decision."""
        # Try XML format first
        action_match = re.search(
            r'<action\s+name=["\']([^"\']+)["\']',
            response,
            re.IGNORECASE,
        )

        if action_match:
            action_name = action_match.group(1)

            # Extract parameters
            params = {}
            param_matches = re.finditer(
                r'<parameter\s+name=["\']([^"\']+)["\']>([^<]*)</parameter>',
                response,
                re.IGNORECASE,
            )
            for match in param_matches:
                params[match.group(1)] = match.group(2).strip()

            # Extract reasoning
            reasoning_match = re.search(
                r"<reasoning>([^<]*)</reasoning>",
                response,
                re.IGNORECASE,
            )
            reasoning = reasoning_match.group(1).strip() if reasoning_match else ""

            return ExecutionDecision(
                action_name=action_name,
                parameters=params,
                reasoning=reasoning,
                raw_response=response,
            )

        # Fallback: Try to find action name in response
        for action in behavior.actions:
            if action.name.lower() in response.lower():
                return ExecutionDecision(
                    action_name=action.name,
                    parameters={},
                    reasoning="Action matched in response",
                    raw_response=response,
                )

        return None

    async def _execute_action(
        self,
        behavior: Behavior,
        decision: ExecutionDecision,
        context: ExecutionContext,
    ) -> tuple[ToolResult | None, float]:
        """Execute the decided action."""
        start = time.time()

        # Find the action
        action = None
        for a in behavior.actions:
            if a.name == decision.action_name:
                action = a
                break

        if not action:
            return ToolResult(
                tool_name=decision.action_name,
                success=False,
                result=None,
                error=f"Action not found: {decision.action_name}",
            ), (time.time() - start) * 1000

        # Execute via tool registry
        tool_result = await self.tools.execute(
            decision.action_name,
            **decision.parameters,
        )

        # Update action stats
        action.usage_count += 1
        action.last_used = datetime.now()
        if tool_result.success:
            # Update success rate
            action.success_rate = (
                (action.success_rate * (action.usage_count - 1) + 1.0)
                / action.usage_count
            )
        else:
            action.success_rate = (
                (action.success_rate * (action.usage_count - 1))
                / action.usage_count
            )

        return tool_result, (time.time() - start) * 1000

    async def _synthesize_response(
        self,
        behavior: Behavior,
        decision: ExecutionDecision,
        tool_result: ToolResult | None,
        context: ExecutionContext,
    ) -> tuple[str, float]:
        """Synthesize the final response."""
        start = time.time()

        if not behavior.prompts or not behavior.prompts.synthesis_prompt:
            # Default response
            if tool_result and tool_result.success:
                return f"Done. {tool_result.result}", (time.time() - start) * 1000
            return "I've completed the action.", (time.time() - start) * 1000

        # Format the synthesis prompt
        prompt = behavior.prompts.synthesis_prompt

        action_result = ""
        if tool_result:
            if tool_result.success:
                action_result = str(tool_result.result)
            else:
                action_result = f"Error: {tool_result.error}"

        prompt = prompt.replace("{{action_result}}", action_result)
        prompt = prompt.replace("{action_result}", action_result)
        prompt = prompt.replace("{{action}}", decision.action_name)
        prompt = prompt.replace("{action}", decision.action_name)
        prompt = prompt.replace("{{query}}", context.query)
        prompt = prompt.replace("{query}", context.query)
        prompt = prompt.replace("{{style}}", behavior.prompts.style_guidelines or "friendly")
        prompt = prompt.replace("{style}", behavior.prompts.style_guidelines or "friendly")

        try:
            response = await self.llm.generate(
                prompt=prompt,
                system_prompt="Format a natural response for a voice assistant.",
                temperature=0.5,
                tier=ModelTier.FAST,
            )

            # Clean up response
            response = response.strip()
            if response.startswith('"') and response.endswith('"'):
                response = response[1:-1]

            return response, (time.time() - start) * 1000

        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            if tool_result and tool_result.success:
                return "Done.", (time.time() - start) * 1000
            return "I completed the action.", (time.time() - start) * 1000

    @property
    def stats(self) -> dict[str, Any]:
        """Get execution statistics."""
        return {
            "total_executions": self._executions,
            "successes": self._successes,
            "failures": self._failures,
            "success_rate": (
                self._successes / self._executions if self._executions > 0 else 0.0
            ),
        }


# =============================================================================
# Common Tool Implementations (for testing and common use cases)
# =============================================================================


def create_mock_tool_registry() -> ToolRegistry:
    """Create a tool registry with mock tools for testing."""
    registry = ToolRegistry()

    async def mock_set_timer(duration: str = "5 minutes", **kwargs) -> str:
        return f"Timer set for {duration}"

    async def mock_cancel_timer(timer_name: str = "timer", **kwargs) -> str:
        return f"Timer '{timer_name}' cancelled"

    async def mock_list_timers(**kwargs) -> list[str]:
        return ["5 minute timer", "pasta timer"]

    async def mock_control_device(device_action: str = "", **kwargs) -> str:
        return f"Device action completed: {device_action}"

    async def mock_get_device_state(device: str = "", **kwargs) -> str:
        return f"{device} is currently on"

    async def mock_get_events(time_range: str = "today", **kwargs) -> list[str]:
        return ["Meeting at 3pm", "Call with team at 4pm"]

    async def mock_create_event(event_details: str = "", **kwargs) -> str:
        return f"Event created: {event_details}"

    # Register all mock tools
    registry.register("set_timer", mock_set_timer, "Set a timer")
    registry.register("cancel_timer", mock_cancel_timer, "Cancel a timer")
    registry.register("list_timers", mock_list_timers, "List active timers")
    registry.register("control_device", mock_control_device, "Control a device")
    registry.register("get_device_state", mock_get_device_state, "Get device state")
    registry.register("get_events", mock_get_events, "Get calendar events")
    registry.register("create_event", mock_create_event, "Create calendar event")

    return registry
