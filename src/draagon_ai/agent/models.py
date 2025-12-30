"""Models for the agent loop.

This module defines request/response models and supporting structures
for the agent orchestration loop.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    """Types of actions the agent can take."""

    ANSWER = "answer"              # Direct response
    SEARCH_WEB = "search_web"      # Web search
    SEARCH_MEMORY = "search_memory"  # Memory/knowledge search
    TOOL_CALL = "tool_call"        # Execute a tool
    CLARIFY = "clarify"            # Ask for clarification
    UNDO = "undo"                  # Undo previous action
    MORE_DETAILS = "more_details"  # Expand on previous response


class ModelTier(str, Enum):
    """Model tiers for routing."""

    FAST = "fast"           # 8B models, greetings/simple
    STANDARD = "standard"   # 20B models, most queries
    COMPLEX = "complex"     # 70B models, reasoning


@dataclass
class AgentRequest:
    """Request to the agent loop.

    Attributes:
        query: User's input text
        user_id: User identifier
        conversation_id: Conversation/session identifier
        area_id: Physical area (for voice assistants)
        device_id: Device identifier
        timezone: User's timezone
        debug: Whether to include debug info
        metadata: Additional request metadata
    """

    query: str
    user_id: str
    conversation_id: str | None = None
    area_id: str | None = None
    device_id: str | None = None
    timezone: str | None = None
    debug: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallInfo:
    """Information about a tool call made during processing.

    Attributes:
        tool: Tool name
        args: Arguments passed to tool
        result: Tool result (may be truncated for display)
        full_result: Complete tool result
        elapsed_ms: Execution time in milliseconds
        success: Whether tool executed successfully
        error: Error message if failed
    """

    tool: str
    args: dict[str, Any]
    result: Any = None
    full_result: Any = None
    elapsed_ms: int = 0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "tool": self.tool,
            "args": self.args,
            "result": self.result,
            "elapsed_ms": self.elapsed_ms,
            "success": self.success,
            "error": self.error,
        }


@dataclass
class DebugInfo:
    """Debug information about request processing.

    Attributes:
        fast_path: Fast-path route used (if any)
        router_used: Whether decision LLM was used
        router_decision: Raw decision output
        action: Action taken
        model_tier: Model tier used
        llm_calls: Number of LLM calls
        llm_provider: LLM provider used
        confidence: Decision confidence
        contextualized: Whether query was contextualized
        original_query: Original query (if contextualized)
        standalone_query: Standalone query (if contextualized)
        context_intent: Detected intent
        memory_found: Number of memories found
        knowledge_found: Number of knowledge items found
        cross_chat_found: Number of cross-chat results
        crag_enabled: Whether CRAG was used
        crag_grading: CRAG grading results
        timings: Operation timings in ms
        area_id: Area context used
    """

    fast_path: str | None = None
    router_used: bool = False
    router_decision: dict[str, Any] | None = None
    action: str | None = None
    model_tier: str | None = None
    llm_calls: int = 0
    llm_provider: str | None = None
    confidence: float = 0.0
    contextualized: bool = False
    original_query: str | None = None
    standalone_query: str | None = None
    context_intent: str | None = None
    memory_found: int = 0
    knowledge_found: int = 0
    cross_chat_found: int = 0
    crag_enabled: bool = False
    crag_grading: dict[str, int] | None = None
    timings: dict[str, int] = field(default_factory=dict)
    area_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "fast_path": self.fast_path,
            "router_used": self.router_used,
            "router_decision": self.router_decision,
            "action": self.action,
            "model_tier": self.model_tier,
            "llm_calls": self.llm_calls,
            "llm_provider": self.llm_provider,
            "confidence": self.confidence,
            "contextualized": self.contextualized,
            "original_query": self.original_query,
            "standalone_query": self.standalone_query,
            "context_intent": self.context_intent,
            "memory_found": self.memory_found,
            "knowledge_found": self.knowledge_found,
            "cross_chat_found": self.cross_chat_found,
            "crag_enabled": self.crag_enabled,
            "crag_grading": self.crag_grading,
            "timings": self.timings,
            "area_id": self.area_id,
        }


@dataclass
class AgentResponse:
    """Response from the agent loop.

    Attributes:
        response: Text response to user
        success: Whether request was successful
        action: Action that was taken
        tool_calls: Tools called during processing
        debug: Debug information (if requested)
        pending_details: Full response for "tell me more"
        metadata: Additional response metadata
    """

    response: str
    success: bool = True
    action: str | None = None
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    debug: DebugInfo | None = None
    pending_details: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "response": self.response,
            "success": self.success,
            "action": self.action,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "metadata": self.metadata,
        }
        if self.debug:
            result["debug"] = self.debug.to_dict()
        if self.pending_details:
            result["pending_details"] = self.pending_details
        return result


@dataclass
class FastRouteResult:
    """Result from fast-path routing.

    Attributes:
        response: Response text
        route_type: Type of fast route used
        tool_calls: Any tools called
        undoable_action: Action that can be undone
        skip_reflection: Whether to skip post-response reflection
    """

    response: str
    route_type: str
    tool_calls: list[ToolCallInfo] = field(default_factory=list)
    undoable_action: Any | None = None
    skip_reflection: bool = True

    def to_agent_response(self, debug: DebugInfo | None = None) -> AgentResponse:
        """Convert to AgentResponse."""
        if debug:
            debug.fast_path = self.route_type
            debug.action = self.route_type
        return AgentResponse(
            response=self.response,
            success=True,
            action=self.route_type,
            tool_calls=self.tool_calls,
            debug=debug,
        )
