"""Tools module for Draagon AI.

Provides MCP client integration and tool management.

Note:
    MCP client functionality requires the 'mcp' optional dependency.
    Install with: pip install draagon-ai[mcp]

    Check MCP_AVAILABLE before using MCP classes:
        from draagon_ai.tools import MCP_AVAILABLE, MCPClient
        if MCP_AVAILABLE:
            client = MCPClient()
"""

from draagon_ai.tools.mcp_client import (
    MCP_AVAILABLE,
    MCPClient,
    MCPServerConfig,
    MCPTool,
    create_mcp_client,
)

__all__ = [
    "MCP_AVAILABLE",
    "MCPClient",
    "MCPServerConfig",
    "MCPTool",
    "create_mcp_client",
]
