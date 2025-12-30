"""Tools module for Draagon AI.

Provides:
- @tool decorator for declarative tool registration
- MCP client integration for external tool servers
- Tool discovery and registry management

Example:
    from draagon_ai.tools import tool, get_all_tools

    @tool(
        name="get_time",
        description="Get the current time",
        category="utilities",
    )
    async def get_time(args: dict, **context) -> dict:
        return {"time": datetime.now().isoformat()}

    # Get all registered tools
    tools = get_all_tools()

Note:
    MCP client functionality requires the 'mcp' optional dependency.
    Install with: pip install draagon-ai[mcp]

    Check MCP_AVAILABLE before using MCP classes:
        from draagon_ai.tools import MCP_AVAILABLE, MCPClient
        if MCP_AVAILABLE:
            client = MCPClient()
"""

# Decorator-based registration
from draagon_ai.tools.decorator import (
    tool,
    get_global_registry,
    get_all_tools,
    get_tool,
    get_tool_metadata,
    get_tools_by_category,
    get_tools_by_tag,
    list_categories,
    list_tags,
    discover_tools,
    clear_registry,
    set_registry,
    ToolMetadata,
)

# MCP client integration
from draagon_ai.tools.mcp_client import (
    MCP_AVAILABLE,
    MCPClient,
    MCPServerConfig,
    MCPTool,
    create_mcp_client,
)

# Re-export core types from orchestration for convenience
from draagon_ai.orchestration.registry import (
    Tool,
    ToolParameter,
    ToolRegistry,
    ToolMetrics,
    ToolExecutionResult,
)

__all__ = [
    # Decorator
    "tool",
    # Registry access
    "get_global_registry",
    "get_all_tools",
    "get_tool",
    "get_tool_metadata",
    "get_tools_by_category",
    "get_tools_by_tag",
    "list_categories",
    "list_tags",
    # Discovery
    "discover_tools",
    "clear_registry",
    "set_registry",
    # Metadata types
    "ToolMetadata",
    # Core types (re-exported from orchestration)
    "Tool",
    "ToolParameter",
    "ToolRegistry",
    "ToolMetrics",
    "ToolExecutionResult",
    # MCP
    "MCP_AVAILABLE",
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "create_mcp_client",
]
