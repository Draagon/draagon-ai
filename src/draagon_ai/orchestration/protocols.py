"""Protocol definitions for orchestration dependencies.

These protocols define the interfaces that applications must implement
to use the agent orchestration system. This allows draagon-ai to work
with any LLM, memory, or tool system.
"""

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# =============================================================================
# LLM Provider Protocol
# =============================================================================


@dataclass
class LLMMessage:
    """A message in a conversation."""

    role: str  # "system", "user", "assistant"
    content: str
    name: str | None = None  # Optional name for multi-agent
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[dict] = field(default_factory=list)
    finish_reason: str = "stop"


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers.

    Applications must implement this protocol to provide LLM capabilities
    to agents. The implementation can use any backend (Groq, OpenAI, Ollama, etc.).

    Example:
        class GroqLLMProvider:
            async def chat(
                self,
                messages: list[LLMMessage],
                model: str | None = None,
                **kwargs,
            ) -> LLMResponse:
                # Call Groq API
                return LLMResponse(content=response)
    """

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
            messages: Conversation messages
            model: Model to use (implementation-specific)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific options

        Returns:
            LLM response with content and metadata
        """
        ...


# =============================================================================
# Memory Provider Protocol
# =============================================================================


@dataclass
class MemorySearchResult:
    """A result from memory search."""

    memory_id: str
    content: str
    score: float
    memory_type: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    entities: list[str] = field(default_factory=list)


@dataclass
class Memory:
    """A memory record."""

    memory_id: str
    content: str
    memory_type: str  # "fact", "skill", "episodic", "knowledge", etc.
    user_id: str
    importance: float = 0.5
    entities: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory/knowledge providers.

    Applications must implement this protocol to provide memory capabilities.
    The implementation can use any backend (Qdrant, Pinecone, in-memory, etc.).

    Example:
        class QdrantMemoryProvider:
            async def search(
                self,
                query: str,
                user_id: str,
                limit: int = 5,
            ) -> list[MemorySearchResult]:
                # Search Qdrant
                return results
    """

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
            memory_types: Filter by memory types
            **kwargs: Additional provider-specific options

        Returns:
            List of matching memory results
        """
        ...

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
            memory_type: Type of memory
            entities: Entities mentioned
            importance: Importance score (0-1)
            metadata: Additional metadata
            **kwargs: Additional provider-specific options

        Returns:
            ID of the stored memory
        """
        ...


# =============================================================================
# Tool Provider Protocol
# =============================================================================


@dataclass
class ToolCall:
    """A request to execute a tool."""

    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str | None = None


@dataclass
class ToolResult:
    """Result from tool execution."""

    tool_name: str
    success: bool
    result: Any
    error: str | None = None
    latency_ms: float = 0.0


@runtime_checkable
class ToolProvider(Protocol):
    """Protocol for tool/action providers.

    Applications must implement this protocol to provide tool execution.
    Tools are the handlers for behavior actions.

    Example:
        class MyToolProvider:
            async def execute(
                self,
                tool_call: ToolCall,
                context: dict,
            ) -> ToolResult:
                if tool_call.tool_name == "get_time":
                    return ToolResult(
                        tool_name="get_time",
                        success=True,
                        result={"time": "3:45 PM"},
                    )
    """

    async def execute(
        self,
        tool_call: ToolCall,
        context: dict[str, Any],
    ) -> ToolResult:
        """Execute a tool call.

        Args:
            tool_call: The tool to execute with arguments
            context: Execution context (user_id, area, etc.)

        Returns:
            Tool execution result
        """
        ...

    def list_tools(self) -> list[str]:
        """List available tool names.

        Returns:
            List of tool names that can be executed
        """
        ...

    def get_tool_description(self, tool_name: str) -> str | None:
        """Get description of a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Tool description or None if not found
        """
        ...


# =============================================================================
# Search Provider Protocol (Optional)
# =============================================================================


@dataclass
class SearchResult:
    """A web search result."""

    title: str
    url: str
    snippet: str
    score: float = 0.0


@runtime_checkable
class SearchProvider(Protocol):
    """Protocol for web search providers.

    Optional protocol for agents that need web search capabilities.

    Example:
        class SearXNGSearchProvider:
            async def search(
                self,
                query: str,
                num_results: int = 5,
            ) -> list[SearchResult]:
                # Search SearXNG
                return results
    """

    async def search(
        self,
        query: str,
        num_results: int = 5,
        **kwargs: Any,
    ) -> list[SearchResult]:
        """Execute a web search.

        Args:
            query: Search query
            num_results: Maximum results to return
            **kwargs: Additional provider-specific options

        Returns:
            List of search results
        """
        ...
