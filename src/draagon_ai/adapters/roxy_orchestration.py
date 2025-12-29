"""Adapter for integrating Roxy voice assistant with draagon-ai orchestration.

This module provides an adapter that allows Roxy to use draagon-ai's Agent
and AgentLoop for orchestration instead of its own orchestrator. This enables
Roxy to leverage the ReAct reasoning pattern and unified tool registry.

The adapter bridges:
- Roxy's process() interface → draagon-ai's Agent.process()
- Roxy's tools → draagon-ai ToolRegistry
- AgentResponse → ChatResponse

REQ-002-06: Roxy adapter for orchestration

Migration Status (REQ-002-07):
==============================
This adapter provides core orchestration but does NOT yet implement all
Roxy-specific features. See roxy-voice-assistant/src/roxy/agent/_archive/
MIGRATION_ANALYSIS.md for the full migration plan.

Features implemented:
- Tool registration and execution via ToolRegistry
- Query processing via Agent.process()
- Response conversion to RoxyResponse format
- Debug info with thought traces and ReAct steps
- Session context management per conversation

Features NOT yet implemented (remain in Roxy's orchestrator.py):
- Calendar cache management
- Conversation mode detection
- Relationship graph queries
- Undo functionality
- Episode summaries
- User identification flow
- Proactive question timing
- Belief reconciliation integration
- Sentiment analysis integration
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, TYPE_CHECKING

from draagon_ai.behaviors import Behavior, VOICE_ASSISTANT_TEMPLATE
from draagon_ai.orchestration import (
    Agent,
    AgentConfig,
    AgentContext,
    AgentResponse,
    ToolRegistry,
    Tool,
    ToolParameter,
    LoopMode,
    ReActStep,
    StepType,
)
from draagon_ai.orchestration.protocols import LLMProvider, MemoryProvider

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Response Types (compatible with Roxy)
# =============================================================================


@dataclass
class ToolCallInfo:
    """Information about a tool call (mirrors Roxy's ToolCallInfo)."""

    tool: str
    args: Any = None
    result: Any = None
    elapsed_ms: int | None = None
    error: str | None = None


@dataclass
class DebugInfo:
    """Debug information (mirrors Roxy's DebugInfo)."""

    latency_ms: int = 0
    llm_calls: int = 0
    llm_retries: int = 0
    knowledge_found: int = 0
    memories_found: int = 0
    parallel_executions: int = 0
    router_used: bool = False
    router_decision: dict[str, Any] | None = None
    multi_step_reasoning: bool = False
    command_security: dict[str, Any] | None = None
    llm_classification: dict[str, Any] | None = None
    tools_called: list[ToolCallInfo] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    # New fields for enhanced debugging
    confidence: float | None = None
    fast_path: str | None = None
    thoughts: list[str] | None = None
    # CRAG metrics
    crag_enabled: bool = False
    crag_grading: dict[str, int] | None = None
    crag_knowledge_strips: int = 0
    # Provider info
    llm_provider: str | None = None
    model_tier: str | None = None
    # Device/area context
    area_id: str | None = None
    # ReAct loop info
    loop_mode: str | None = None
    iterations_used: int = 0
    react_steps: list[dict[str, Any]] | None = None
    # Timing breakdown (like native orchestrator)
    timings: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoxyResponse:
    """Response format (mirrors Roxy's ChatResponse)."""

    response: str
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    iterations: int = 1
    success: bool = True
    conversation_id: str = "default"
    debug: DebugInfo | None = None


# =============================================================================
# Tool Definitions
# =============================================================================

# Type alias for Roxy tool handlers
RoxyToolHandler = Callable[..., Coroutine[Any, Any, Any]]


@dataclass
class RoxyToolDefinition:
    """Definition of a Roxy tool for registration.

    This captures the essential information needed to register
    a Roxy tool with the draagon-ai ToolRegistry.

    Attributes:
        name: Tool name (e.g., "get_time", "search_web")
        description: Human-readable description
        handler: Async function that executes the tool
        parameters: List of parameter definitions
        timeout_ms: Execution timeout in milliseconds
        requires_confirmation: Whether user confirmation is needed
    """

    name: str
    description: str
    handler: RoxyToolHandler
    parameters: list[ToolParameter] = field(default_factory=list)
    timeout_ms: int = 30000
    requires_confirmation: bool = False


# =============================================================================
# Adapter
# =============================================================================


class RoxyOrchestrationAdapter:
    """Adapter for using draagon-ai orchestration with Roxy.

    This adapter allows Roxy to use draagon-ai's Agent and AgentLoop
    for orchestration while maintaining Roxy's existing API contract.

    Features:
    - Same process() interface as Roxy's AgentOrchestrator
    - All Roxy tools registered with draagon-ai ToolRegistry
    - Context (conversation history, user, area) passed correctly
    - Response format unchanged for callers
    - Debug info includes thought traces from ReAct

    Example:
        from draagon_ai.adapters import RoxyOrchestrationAdapter
        from draagon_ai.orchestration import ToolRegistry

        # Create adapter with LLM provider
        adapter = RoxyOrchestrationAdapter(
            llm=my_llm_provider,
            memory=my_memory_provider,
        )

        # Register Roxy's tools
        adapter.register_tool(RoxyToolDefinition(
            name="get_time",
            description="Get current time",
            handler=get_time_handler,
        ))

        # Process messages with same API as Roxy
        response = await adapter.process(
            query="What time is it?",
            user_id="doug",
            conversation_id="conv_123",
            debug=True,
        )
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        behavior: Behavior | None = None,
        agent_id: str = "roxy",
        agent_name: str = "Roxy",
        personality_intro: str = "",
        default_model_tier: str = "local",
        loop_mode: LoopMode = LoopMode.AUTO,
    ):
        """Initialize the orchestration adapter.

        Args:
            llm: LLM provider for reasoning
            memory: Optional memory provider for context
            behavior: Behavior definition (defaults to VOICE_ASSISTANT_TEMPLATE)
            agent_id: Unique agent identifier
            agent_name: Human-readable agent name
            personality_intro: Personality introduction for prompts
            default_model_tier: Default LLM model tier
            loop_mode: AgentLoop mode (SIMPLE, REACT, AUTO)
        """
        self.llm = llm
        self.memory = memory
        self.behavior = behavior or VOICE_ASSISTANT_TEMPLATE
        self.loop_mode = loop_mode

        # Create tool registry
        self.tool_registry = ToolRegistry()

        # Track registered tools
        self._registered_tools: dict[str, RoxyToolDefinition] = {}

        # Agent will be created when tools are registered
        self._agent: Agent | None = None
        self._agent_config = AgentConfig(
            agent_id=agent_id,
            name=agent_name,
            personality_intro=personality_intro,
            default_model_tier=default_model_tier,
        )

        # Session contexts for conversation continuity
        self._contexts: dict[str, AgentContext] = {}

        # Optional mode detection callback (set by Roxy to enable conversation mode detection)
        # Signature: async def(query: str, conversation_id: str, history: list[dict]) -> dict
        # Returns: {"mode": "task"|"support"|..., "mode_detection_ms": int}
        self._mode_detector: Any = None

        logger.info(
            f"RoxyOrchestrationAdapter initialized: agent={agent_id}, mode={loop_mode.value}"
        )

    def register_tool(self, tool_def: RoxyToolDefinition) -> None:
        """Register a Roxy tool with the adapter.

        Args:
            tool_def: Tool definition with name, handler, and parameters
        """
        # Store definition
        self._registered_tools[tool_def.name] = tool_def

        # Register with ToolRegistry
        tool = Tool(
            name=tool_def.name,
            description=tool_def.description,
            handler=tool_def.handler,
            parameters=tool_def.parameters,
            timeout_ms=tool_def.timeout_ms,
            requires_confirmation=tool_def.requires_confirmation,
        )
        self.tool_registry.register(tool)

        logger.debug(f"Registered tool: {tool_def.name}")

    def register_tools(self, tools: list[RoxyToolDefinition]) -> None:
        """Register multiple Roxy tools.

        Args:
            tools: List of tool definitions
        """
        for tool in tools:
            self.register_tool(tool)

    def set_mode_detector(self, detector: Any) -> None:
        """Set the conversation mode detector callback.

        The callback should be an async function with signature:
            async def(query: str, conversation_id: str, history: list[dict]) -> dict

        It should return a dict with:
            - mode: str (e.g., "task", "support", "casual", "brainstorm", "learning")
            - mode_detection_ms: int (time taken in milliseconds)

        Args:
            detector: Async callback function for mode detection
        """
        self._mode_detector = detector
        logger.debug("Mode detector callback registered")

    def _ensure_agent(self) -> Agent:
        """Ensure the Agent is created with registered tools.

        Returns:
            The Agent instance
        """
        if self._agent is None:
            from draagon_ai.orchestration import ActionExecutor

            # Create action executor with our tool registry
            action_executor = ActionExecutor(tool_registry=self.tool_registry)

            # Create the agent
            self._agent = Agent(
                config=self._agent_config,
                behavior=self.behavior,
                llm=self.llm,
                memory=self.memory,
                tools=None,  # We use ToolRegistry, not ToolProvider
            )

            # Replace the action executor with ours
            self._agent._action_executor = action_executor
            self._agent._loop.action_executor = action_executor

            # Configure loop mode
            self._agent._loop.config.mode = self.loop_mode

            logger.info(
                f"Agent created with {len(self.tool_registry)} tools, mode={self.loop_mode.value}"
            )

        return self._agent

    def _get_or_create_context(
        self,
        conversation_id: str,
        user_id: str,
        area_id: str | None,
        debug: bool,
    ) -> AgentContext:
        """Get or create a session context.

        Args:
            conversation_id: Conversation identifier
            user_id: User identifier
            area_id: Area/room identifier
            debug: Debug mode flag

        Returns:
            AgentContext for this session
        """
        if conversation_id not in self._contexts:
            self._contexts[conversation_id] = AgentContext(
                user_id=user_id,
                session_id=conversation_id,
                area_id=area_id,
                debug=debug,
            )
        else:
            # Update mutable fields
            ctx = self._contexts[conversation_id]
            ctx.area_id = area_id
            ctx.debug = debug

        return self._contexts[conversation_id]

    def _extract_tool_calls(
        self, agent_response: AgentResponse
    ) -> list[ToolCallInfo]:
        """Extract tool call info from AgentResponse.

        Args:
            agent_response: Response from the agent

        Returns:
            List of ToolCallInfo objects
        """
        tool_calls = []

        for step in agent_response.react_steps:
            if step.type == StepType.ACTION:
                # Parse action content to get tool name and args
                tool_name = "unknown"
                tool_args = {}

                # ReActStep has action_name and action_args directly
                tool_name = step.action_name or "unknown"
                tool_args = step.action_args or {}

                # Also check content for action info (legacy format)
                if tool_name == "unknown" and isinstance(step.content, dict):
                    tool_name = step.content.get("action", step.content.get("tool", "unknown"))
                    tool_args = step.content.get("args", step.content.get("arguments", {}))
                elif tool_name == "unknown" and isinstance(step.content, str):
                    tool_name = step.content

                tool_calls.append(ToolCallInfo(
                    tool=tool_name,
                    args=tool_args,
                    elapsed_ms=int(step.duration_ms) if step.duration_ms else None,
                ))

            elif step.type == StepType.OBSERVATION:
                # Match observation to previous action
                if tool_calls and tool_calls[-1].result is None:
                    if isinstance(step.content, dict) and "error" in step.content:
                        tool_calls[-1].error = step.content.get("error")
                    else:
                        tool_calls[-1].result = step.content

        return tool_calls

    def _extract_thoughts(self, agent_response: AgentResponse) -> list[str]:
        """Extract thought strings from ReAct steps.

        Args:
            agent_response: Response from the agent

        Returns:
            List of thought strings
        """
        thoughts = []
        for step in agent_response.react_steps:
            if step.type == StepType.THOUGHT:
                if isinstance(step.content, str):
                    thoughts.append(step.content)
                elif isinstance(step.content, dict):
                    thoughts.append(step.content.get("reasoning", str(step.content)))
        return thoughts

    def _convert_react_steps(
        self, steps: list[ReActStep]
    ) -> list[dict[str, Any]]:
        """Convert ReActStep objects to dictionaries.

        Args:
            steps: List of ReActStep objects

        Returns:
            List of step dictionaries
        """
        return [
            {
                "type": step.type.value,
                "content": step.content,
                "duration_ms": step.duration_ms,
                "timestamp": step.timestamp.isoformat() if step.timestamp else None,
            }
            for step in steps
        ]

    def _convert_response(
        self,
        agent_response: AgentResponse,
        conversation_id: str,
        latency_ms: int,
        debug: bool,
        area_id: str | None,
        extra_timings: dict[str, Any] | None = None,
    ) -> RoxyResponse:
        """Convert AgentResponse to RoxyResponse.

        Args:
            agent_response: Response from draagon-ai Agent
            conversation_id: Conversation identifier
            latency_ms: Total processing time
            debug: Whether to include debug info
            area_id: Area/room ID
            extra_timings: Additional timing info (e.g., mode detection)

        Returns:
            RoxyResponse compatible with Roxy's ChatResponse
        """
        tool_calls = self._extract_tool_calls(agent_response)

        response = RoxyResponse(
            response=agent_response.response,
            tool_calls=tool_calls,
            iterations=agent_response.iterations_used,
            success=agent_response.success,
            conversation_id=conversation_id,
        )

        if debug:
            thoughts = self._extract_thoughts(agent_response)
            react_steps = self._convert_react_steps(agent_response.react_steps)

            # Extract confidence from decision if available
            confidence = None
            if agent_response.decision:
                confidence = agent_response.decision.confidence

            # Extract memory count from debug_info
            memories_found = agent_response.debug_info.get("memories_found", 0)

            # Build timings dict with any extra timing info
            timings: dict[str, Any] = {}
            if extra_timings:
                timings.update(extra_timings)

            response.debug = DebugInfo(
                latency_ms=latency_ms,
                llm_calls=agent_response.iterations_used,
                tools_called=tool_calls,
                multi_step_reasoning=len(agent_response.react_steps) > 1,
                router_used=True,
                confidence=confidence,
                thoughts=thoughts if thoughts else None,
                area_id=area_id,
                loop_mode=agent_response.loop_mode.value if agent_response.loop_mode else None,
                iterations_used=agent_response.iterations_used,
                react_steps=react_steps if react_steps else None,
                memories_found=memories_found,
                timings=timings,
            )

        return response

    async def process(
        self,
        query: str,
        user_id: str = "default",
        conversation_id: str = "default",
        debug: bool = False,
        area_id: str | None = None,
        timezone: str | None = None,
        **kwargs: Any,
    ) -> RoxyResponse:
        """Process a query using draagon-ai orchestration.

        This method has the same interface as Roxy's AgentOrchestrator.process()
        to enable drop-in replacement.

        Args:
            query: The user's query text
            user_id: User identifier
            conversation_id: Conversation identifier for context
            debug: Whether to include debug information
            area_id: Area/room ID for room-aware commands
            timezone: IANA timezone string (for future use)
            **kwargs: Additional context (ignored for compatibility)

        Returns:
            RoxyResponse with the result (compatible with ChatResponse)
        """
        import asyncio

        start_time = time.time()
        extra_timings: dict[str, Any] = {}

        # Ensure agent is created
        agent = self._ensure_agent()

        # Get or create context
        context = self._get_or_create_context(
            conversation_id=conversation_id,
            user_id=user_id,
            area_id=area_id,
            debug=debug,
        )

        try:
            # Run mode detection in parallel with agent processing (if detector is set)
            mode_task = None
            if self._mode_detector is not None:
                # Get conversation history from context
                history = list(context.conversation_history) if hasattr(context, 'conversation_history') else []
                mode_task = asyncio.create_task(
                    self._mode_detector(query, conversation_id, history)
                )

            # Process through draagon-ai Agent
            agent_response = await agent.process(
                query=query,
                user_id=user_id,
                session_id=conversation_id,
                area_id=area_id,
                debug=debug,
            )

            # Await mode detection result if running
            if mode_task is not None:
                try:
                    mode_result = await mode_task
                    if mode_result:
                        # Store mode in timings for test compatibility
                        extra_timings["conversation_mode"] = mode_result.get("mode", "task")
                        extra_timings["mode_detection_ms"] = mode_result.get("mode_detection_ms", 0)
                except Exception as e:
                    logger.warning(f"Mode detection failed: {e}")
                    extra_timings["conversation_mode"] = "task"  # Default to task on error

            latency_ms = int((time.time() - start_time) * 1000)

            # Convert to Roxy format
            return self._convert_response(
                agent_response=agent_response,
                conversation_id=conversation_id,
                latency_ms=latency_ms,
                debug=debug,
                area_id=area_id,
                extra_timings=extra_timings if extra_timings else None,
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            latency_ms = int((time.time() - start_time) * 1000)

            response = RoxyResponse(
                response=f"I encountered an error: {str(e)}",
                success=False,
                conversation_id=conversation_id,
            )

            if debug:
                response.debug = DebugInfo(
                    latency_ms=latency_ms,
                    errors=[{"error": str(e), "type": type(e).__name__}],
                    timings=extra_timings if extra_timings else {},
                )

            return response

    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear a conversation's history.

        Args:
            conversation_id: Conversation to clear

        Returns:
            True if conversation existed and was cleared
        """
        if conversation_id in self._contexts:
            self._contexts[conversation_id].conversation_history.clear()
            self._contexts[conversation_id].pending_details = None
            return True
        return False

    def get_registered_tools(self) -> list[str]:
        """Get list of registered tool names.

        Returns:
            List of tool names
        """
        return list(self._registered_tools.keys())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Get tool schemas for LLM context.

        Returns:
            List of tool schema dictionaries
        """
        return self.tool_registry.get_schemas_for_llm()

    @property
    def agent(self) -> Agent | None:
        """Get the underlying Agent (may be None if not yet initialized)."""
        return self._agent

    @property
    def agent_id(self) -> str:
        """Get the agent's ID."""
        return self._agent_config.agent_id

    @property
    def agent_name(self) -> str:
        """Get the agent's name."""
        return self._agent_config.name


# =============================================================================
# Factory Functions
# =============================================================================


def create_roxy_orchestration_adapter(
    llm: LLMProvider,
    memory: MemoryProvider | None = None,
    tools: list[RoxyToolDefinition] | None = None,
    agent_id: str = "roxy",
    agent_name: str = "Roxy",
    personality_intro: str = "You are Roxy, a helpful voice assistant.",
    loop_mode: LoopMode = LoopMode.AUTO,
) -> RoxyOrchestrationAdapter:
    """Create a configured RoxyOrchestrationAdapter.

    This is the recommended way to create an adapter for Roxy integration.

    Args:
        llm: LLM provider for reasoning
        memory: Optional memory provider for context
        tools: List of tool definitions to register
        agent_id: Unique agent identifier
        agent_name: Human-readable agent name
        personality_intro: Personality introduction for prompts
        loop_mode: AgentLoop mode (SIMPLE, REACT, AUTO)

    Returns:
        Configured RoxyOrchestrationAdapter

    Example:
        adapter = create_roxy_orchestration_adapter(
            llm=my_llm,
            memory=my_memory,
            tools=[
                RoxyToolDefinition(name="get_time", ...),
                RoxyToolDefinition(name="get_weather", ...),
            ],
        )
    """
    adapter = RoxyOrchestrationAdapter(
        llm=llm,
        memory=memory,
        agent_id=agent_id,
        agent_name=agent_name,
        personality_intro=personality_intro,
        loop_mode=loop_mode,
    )

    if tools:
        adapter.register_tools(tools)

    return adapter
