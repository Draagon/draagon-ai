"""Adapters for integrating draagon-ai with Roxy Voice Assistant.

These adapters bridge Roxy's existing services (LLMService, MemoryService,
ToolExecutor) to the draagon-ai protocol interfaces. This allows Roxy to
use draagon-ai's behavior system while keeping its existing infrastructure.

Usage:
    from roxy.services.llm import LLMService
    from roxy.services.memory import MemoryService
    from roxy.tools.executor import ToolExecutor

    from draagon_ai.adapters.roxy import (
        RoxyLLMAdapter,
        RoxyMemoryAdapter,
        RoxyToolAdapter,
    )
    from draagon_ai.orchestration import Agent
    from draagon_ai.behaviors import VOICE_ASSISTANT_TEMPLATE

    # Create adapters wrapping existing services
    llm_provider = RoxyLLMAdapter(LLMService())
    memory_provider = RoxyMemoryAdapter(MemoryService())
    tool_provider = RoxyToolAdapter(ToolExecutor())

    # Create agent using draagon-ai behavior
    agent = Agent(
        behavior=VOICE_ASSISTANT_TEMPLATE,
        llm_provider=llm_provider,
        memory_provider=memory_provider,
        tool_provider=tool_provider,
    )

    # Process user request
    response = await agent.process("What time is it?", user_id="doug")
"""

from __future__ import annotations

import logging
from typing import Any, Literal, TYPE_CHECKING

from draagon_ai.orchestration.protocols import (
    LLMMessage,
    LLMResponse,
    MemorySearchResult,
    ToolCall,
    ToolResult,
)

if TYPE_CHECKING:
    # Import types only for type checking to avoid circular imports
    # These will be the actual Roxy services at runtime
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Adapter
# =============================================================================


class RoxyLLMAdapter:
    """Adapts Roxy's LLMService to the draagon-ai LLMProvider protocol.

    This adapter wraps Roxy's multi-provider LLM service (Groq/Claude/Ollama)
    and exposes it through draagon-ai's simpler LLMProvider interface.

    Features preserved:
    - Multi-tier model selection (fast/complex/deep)
    - Provider failover (Groq -> Claude)
    - JSON mode for structured output
    - Response caching (when enabled)

    Example:
        from roxy.services.llm import LLMService
        from draagon_ai.adapters.roxy import RoxyLLMAdapter

        roxy_llm = LLMService()
        provider = RoxyLLMAdapter(roxy_llm)

        response = await provider.chat([
            LLMMessage(role="user", content="Hello!")
        ])
    """

    def __init__(self, llm_service: Any):
        """Initialize the adapter.

        Args:
            llm_service: Roxy's LLMService instance
        """
        self._llm = llm_service

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a chat completion.

        Args:
            messages: Conversation messages in draagon-ai format
            model: Model hint - "fast", "complex", or "deep" for Roxy's tiers
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional options passed to underlying service
                - json_mode: bool - Request JSON output
                - force_provider: "groq" | "claude" | "ollama"
                - use_fast_model: bool - Use 8B vs 70B on Groq

        Returns:
            LLM response with content and metadata
        """
        # Convert draagon-ai messages to Roxy format
        roxy_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        # Map model hint to Roxy's tier system
        force_provider: Literal["groq", "claude", "ollama"] | None = None
        use_fast_model = True

        if model:
            model_lower = model.lower()
            if model_lower in ("fast", "local", "8b"):
                use_fast_model = True
                force_provider = "groq"
            elif model_lower in ("complex", "70b"):
                use_fast_model = False
                force_provider = "groq"
            elif model_lower in ("deep", "opus", "claude"):
                force_provider = "claude"
            elif model_lower == "ollama":
                force_provider = "ollama"

        # Override with explicit kwargs
        if "force_provider" in kwargs:
            force_provider = kwargs.pop("force_provider")
        if "use_fast_model" in kwargs:
            use_fast_model = kwargs.pop("use_fast_model")

        json_mode = kwargs.pop("json_mode", False)

        # Call Roxy's LLM service
        result = await self._llm.chat(
            messages=roxy_messages,
            max_tokens=max_tokens or 300,
            temperature=temperature,
            json_mode=json_mode,
            force_provider=force_provider,
            use_fast_model=use_fast_model,
        )

        # Convert to draagon-ai response format
        return LLMResponse(
            content=result.get("content", ""),
            model=result.get("model", ""),
            usage=result.get("usage", {}),
            finish_reason="stop" if not result.get("error") else "error",
        )

    async def embed(self, text: str) -> list[float] | None:
        """Generate embeddings for text.

        Exposes Roxy's Ollama-based embedding service.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None on failure
        """
        return await self._llm.embed(text)


# =============================================================================
# Memory Adapter
# =============================================================================


class RoxyMemoryAdapter:
    """Adapts Roxy's MemoryService to the draagon-ai MemoryProvider protocol.

    This adapter wraps Roxy's Qdrant-backed memory service and exposes it
    through draagon-ai's MemoryProvider interface.

    Features preserved:
    - Multi-type memories (fact, skill, preference, episodic, etc.)
    - Self-RAG retrieval assessment
    - CRAG chunk grading
    - Contradiction detection
    - Recency boosting
    - Cross-chat search

    Example:
        from roxy.services.memory import MemoryService
        from draagon_ai.adapters.roxy import RoxyMemoryAdapter

        roxy_memory = MemoryService(settings, llm)
        provider = RoxyMemoryAdapter(roxy_memory)

        results = await provider.search("user preferences", user_id="doug")
    """

    # Map draagon-ai memory types to Roxy's MemoryType enum
    TYPE_MAPPING = {
        "fact": "fact",
        "skill": "skill",
        "preference": "preference",
        "episodic": "episodic",
        "instruction": "instruction",
        "knowledge": "knowledge",
        "insight": "insight",
        "unknown": "fact",  # Default to fact for unknown types
    }

    def __init__(self, memory_service: Any):
        """Initialize the adapter.

        Args:
            memory_service: Roxy's MemoryService instance
        """
        self._memory = memory_service

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        memory_types: list[str] | None = None,
        **kwargs: Any,
    ) -> list[MemorySearchResult]:
        """Search for relevant memories.

        Args:
            query: Search query
            user_id: User to search memories for
            limit: Maximum results to return
            memory_types: Filter by memory types (fact, skill, etc.)
            **kwargs: Additional options
                - use_self_rag: bool - Use Self-RAG assessment (default True)
                - use_crag: bool - Use CRAG grading
                - include_knowledge: bool - Include knowledge base
                - score_threshold: float - Minimum similarity score
                - household_id: str - Include household shared memories

        Returns:
            List of matching memory results
        """
        use_self_rag = kwargs.pop("use_self_rag", True)
        use_crag = kwargs.pop("use_crag", False)
        include_knowledge = kwargs.pop("include_knowledge", True)
        score_threshold = kwargs.pop("score_threshold", 0.4)
        household_id = kwargs.pop("household_id", None)

        # Convert memory types to Roxy's MemoryType enum if provided
        roxy_types = None
        if memory_types:
            roxy_types = [
                self._get_roxy_memory_type(t)
                for t in memory_types
                if t in self.TYPE_MAPPING
            ]

        # Choose search method based on options
        if use_crag:
            result = await self._memory.search_with_crag(
                query=query,
                user_id=user_id,
                limit=limit,
            )
            raw_results = result.get("results", [])
        elif use_self_rag:
            result = await self._memory.search_with_self_rag(
                query=query,
                user_id=user_id,
                limit=limit,
            )
            raw_results = result.get("results", [])
        else:
            raw_results = await self._memory.search(
                query=query,
                user_id=user_id,
                limit=limit,
                score_threshold=score_threshold,
                include_knowledge=include_knowledge,
                memory_types=roxy_types,
                household_id=household_id,
            )

        # Convert to draagon-ai format
        return [
            MemorySearchResult(
                memory_id=r.get("id", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
                memory_type=r.get("type", "unknown"),
                metadata={
                    "scope": r.get("scope"),
                    "source": r.get("source"),
                    "created_at": r.get("created_at"),
                    "importance": r.get("importance"),
                    "knowledge_strip": r.get("knowledge_strip"),  # From CRAG
                },
                entities=r.get("entities", []),
            )
            for r in raw_results
        ]

    async def store(
        self,
        content: str,
        user_id: str,
        memory_type: str,
        entities: list[str] | None = None,
        importance: float = 0.5,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> str:
        """Store a new memory.

        Args:
            content: Memory content
            user_id: User this memory belongs to
            memory_type: Type of memory (fact, skill, preference, etc.)
            entities: Entities mentioned
            importance: Importance score (0-1)
            metadata: Additional metadata
            **kwargs: Additional options
                - conversation_id: str - Conversation context
                - scope: str - "private", "shared", or "system"
                - household_id: str - Household ID for shared memories
                - skip_extraction: bool - Skip LLM entity extraction

        Returns:
            ID of the stored memory
        """
        conversation_id = kwargs.pop("conversation_id", None)
        scope = kwargs.pop("scope", "private")
        household_id = kwargs.pop("household_id", None)
        skip_extraction = kwargs.pop("skip_extraction", False)

        # Get Roxy's memory type enum
        roxy_type = self._get_roxy_memory_type(memory_type)

        result = await self._memory.store(
            content=content,
            user_id=user_id,
            scope=scope,
            memory_type=roxy_type,
            importance=importance,
            entities=entities,
            conversation_id=conversation_id,
            skip_extraction=skip_extraction,
            household_id=household_id,
            metadata=metadata,
        )

        return result.get("memory_id", "")

    def _get_roxy_memory_type(self, type_str: str) -> Any:
        """Convert string type to Roxy's MemoryType enum.

        This is done lazily to avoid import issues.
        """
        # Access Roxy's MemoryType through the service
        if hasattr(self._memory, "MemoryType"):
            MemoryType = self._memory.MemoryType
        else:
            # Fallback: return string (Roxy accepts both)
            return self.TYPE_MAPPING.get(type_str, "fact")

        type_map = {
            "fact": MemoryType.FACT,
            "skill": MemoryType.SKILL,
            "preference": MemoryType.PREFERENCE,
            "episodic": MemoryType.EPISODIC,
            "instruction": MemoryType.INSTRUCTION,
            "knowledge": MemoryType.KNOWLEDGE,
            "insight": MemoryType.INSIGHT,
        }
        return type_map.get(type_str, MemoryType.FACT)


# =============================================================================
# Tool Adapter
# =============================================================================


class RoxyToolAdapter:
    """Adapts Roxy's ToolExecutor to the draagon-ai ToolProvider protocol.

    This adapter wraps Roxy's tool execution system and exposes it through
    draagon-ai's ToolProvider interface. It enables behavior actions to be
    executed by Roxy's existing tool infrastructure.

    Features preserved:
    - All registered tools (time, weather, calendar, HA, commands, etc.)
    - MCP tool support
    - Parallel execution
    - Tool result formatting

    Example:
        from roxy.tools.executor import ToolExecutor
        from draagon_ai.adapters.roxy import RoxyToolAdapter

        roxy_executor = ToolExecutor()
        provider = RoxyToolAdapter(roxy_executor)

        result = await provider.execute(
            ToolCall(tool_name="get_time", arguments={}),
            context={"user_id": "doug"}
        )
    """

    def __init__(self, tool_executor: Any):
        """Initialize the adapter.

        Args:
            tool_executor: Roxy's ToolExecutor instance
        """
        self._executor = tool_executor

    async def execute(
        self,
        tool_call: ToolCall,
        context: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_call: The tool to execute with arguments
            context: Execution context with:
                - user_id: str (required)
                - conversation_id: str (optional)
                - area_id: str (optional, for room-aware commands)
                - timezone: str (optional, IANA timezone)

        Returns:
            Tool execution result
        """
        import time
        start_time = time.time()

        # Extract context values with defaults
        user_id = context.get("user_id", "unknown")
        conversation_id = context.get("conversation_id", "default")
        area_id = context.get("area_id")
        timezone = context.get("timezone")

        try:
            # Execute via Roxy's executor
            result = await self._executor.execute(
                tool_name=tool_call.tool_name,
                args=tool_call.arguments,
                user_id=user_id,
                conversation_id=conversation_id,
                area_id=area_id,
                timezone=timezone,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            # Convert Roxy's ToolCallInfo to draagon-ai ToolResult
            if result.error:
                return ToolResult(
                    tool_name=tool_call.tool_name,
                    success=False,
                    result=None,
                    error=result.error,
                    latency_ms=elapsed_ms,
                )

            return ToolResult(
                tool_name=tool_call.tool_name,
                success=True,
                result=result.result,
                error=None,
                latency_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error(f"Tool execution error: {tool_call.tool_name}: {e}")
            return ToolResult(
                tool_name=tool_call.tool_name,
                success=False,
                result=None,
                error=str(e),
                latency_ms=elapsed_ms,
            )

    def list_tools(self) -> list[str]:
        """List available tool names.

        Returns:
            List of tool names that can be executed
        """
        tools = list(self._executor.registry.list_tools())

        # Include MCP tools if available
        if hasattr(self._executor, "mcp") and self._executor.mcp.is_available:
            mcp_tools = [t.name for t in self._executor.mcp.list_tools()]
            tools.extend(mcp_tools)

        return tools

    def get_tool_description(self, tool_name: str) -> str | None:
        """Get description of a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description or None if not found
        """
        tool = self._executor.registry.get(tool_name)
        if tool:
            return tool.description

        # Check MCP tools
        if hasattr(self._executor, "mcp") and self._executor.mcp.is_available:
            for mcp_tool in self._executor.mcp.list_tools():
                if mcp_tool.name == tool_name:
                    return mcp_tool.description

        return None

    def get_tool_schema(self, tool_name: str) -> dict[str, Any] | None:
        """Get the full schema for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool schema including parameters, or None if not found
        """
        tool = self._executor.registry.get(tool_name)
        if tool:
            return {
                "name": tool.name,
                "description": tool.description,
                "parameters": [
                    {
                        "name": p.name,
                        "type": p.param_type,
                        "description": p.description,
                        "required": p.required,
                        "default": p.default,
                    }
                    for p in tool.parameters
                ],
                "returns": tool.returns,
            }
        return None
