"""MCP Server for Draagon AI Memory.

This module provides an MCP (Model Context Protocol) server that exposes
draagon-ai memory operations as tools. This enables Claude Code, VS Code,
and other MCP-compatible applications to share a common knowledge base.

Example:
    # Start the MCP server
    python -m draagon_ai.mcp.server

    # Or use programmatically
    from draagon_ai.mcp import MemoryMCPServer, MCPConfig

    config = MCPConfig(
        qdrant_url="http://192.168.168.216:6333",
        ollama_url="http://192.168.168.200:11434",
    )
    server = MemoryMCPServer(config)
    server.run()

Configuration for Claude Code (~/.config/claude-code/claude_code_config.json):
    {
      "mcpServers": {
        "memory": {
          "command": "python",
          "args": ["-m", "draagon_ai.mcp.server"],
          "env": {
            "QDRANT_URL": "http://192.168.168.216:6333",
            "OLLAMA_URL": "http://192.168.168.200:11434"
          }
        }
      }
    }
"""

from draagon_ai.mcp.config import (
    ClientConfig,
    MCPConfig,
    MCPScope,
    SCOPE_HIERARCHY,
    can_read_scope,
    can_write_scope,
    get_readable_scopes,
    get_scope_level,
)
from draagon_ai.mcp.server import MemoryMCPServer, create_memory_mcp_server

__all__ = [
    # Config
    "ClientConfig",
    "MCPConfig",
    "MCPScope",
    # Scope access control
    "SCOPE_HIERARCHY",
    "can_read_scope",
    "can_write_scope",
    "get_readable_scopes",
    "get_scope_level",
    # Server
    "MemoryMCPServer",
    "create_memory_mcp_server",
]
