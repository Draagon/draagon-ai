"""Agent loop and orchestration for Draagon AI.

This module provides the core agent loop that orchestrates:
- Fast-path routing for common queries
- Context gathering (RAG, cross-chat, episodic)
- LLM decision making with multi-modal context
- Tool execution with undo support
- Async post-response reflection and learning
"""

from .models import (
    AgentRequest,
    AgentResponse,
    ToolCallInfo,
    DebugInfo,
    FastRouteResult,
)
from .fast_router import FastRouter, FastRouteHandler
from .loop import AgentLoop, AgentLoopConfig

__all__ = [
    # Models
    "AgentRequest",
    "AgentResponse",
    "ToolCallInfo",
    "DebugInfo",
    "FastRouteResult",
    # Fast routing
    "FastRouter",
    "FastRouteHandler",
    # Main loop
    "AgentLoop",
    "AgentLoopConfig",
]
