"""MCP Client for Draagon AI.

Provides integration with Model Context Protocol servers, allowing Draagon AI
agents to use tools exposed by external MCP servers (Home Assistant, Calendar,
SearXNG, etc.)

Example:
    from draagon_ai.tools import MCPClient, MCPServerConfig

    # Configure MCP servers
    servers = [
        MCPServerConfig(
            name="home-assistant",
            command="mcp-server-home-assistant",
            args=["--url", "http://192.168.168.206:8123"],
            env={"HA_TOKEN": "..."},
        ),
        MCPServerConfig(
            name="searxng",
            command="mcp-server-searxng",
            args=["--url", "http://192.168.168.213:8080"],
        ),
    ]

    # Create client and connect
    client = MCPClient()
    await client.connect_all(servers)

    # List available tools
    tools = await client.list_all_tools()

    # Call a tool
    result = await client.call_tool("home-assistant", "get_entity", {
        "entity_id": "light.bedroom"
    })

Note:
    Requires the 'mcp' optional dependency: pip install draagon-ai[mcp]
"""

import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

# Optional MCP dependency - make imports conditional
try:
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.types import CallToolResult, TextContent

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    ClientSession = None  # type: ignore
    StdioServerParameters = None  # type: ignore
    stdio_client = None  # type: ignore
    CallToolResult = None  # type: ignore
    TextContent = None  # type: ignore

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class MCPServerConfig:
    """Configuration for connecting to an MCP server."""

    name: str
    """Unique name for this server connection."""

    command: str
    """Command to start the MCP server (e.g., 'mcp-server-home-assistant')."""

    args: list[str] = field(default_factory=list)
    """Arguments to pass to the server command."""

    env: dict[str, str] = field(default_factory=dict)
    """Environment variables to set for the server process."""

    enabled: bool = True
    """Whether this server is enabled."""


@dataclass
class MCPTool:
    """A tool available from an MCP server."""

    name: str
    """Tool name (scoped with server: 'server_name.tool_name')."""

    server: str
    """Name of the MCP server providing this tool."""

    original_name: str
    """Original tool name from the server."""

    description: str
    """Tool description."""

    input_schema: dict[str, Any]
    """JSON schema for tool input."""


# =============================================================================
# MCP Client
# =============================================================================


def _check_mcp_available() -> None:
    """Check if MCP package is available, raise ImportError if not."""
    if not MCP_AVAILABLE:
        raise ImportError(
            "MCP package not installed. Install with: pip install draagon-ai[mcp]"
        )


class MCPClient:
    """Client for connecting to multiple MCP servers.

    Manages connections to MCP servers and provides a unified interface
    for listing and calling tools across all connected servers.

    Note:
        Requires the 'mcp' optional dependency: pip install draagon-ai[mcp]
    """

    def __init__(self) -> None:
        """Initialize the MCP client.

        Raises:
            ImportError: If MCP package is not installed.
        """
        _check_mcp_available()
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}  # type: ignore[type-arg]
        self._tools: dict[str, MCPTool] = {}
        self._initialized = False

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to a single MCP server.

        Args:
            config: Server configuration.

        Raises:
            ConnectionError: If connection fails.
        """
        if not config.enabled:
            logger.info(f"MCP server '{config.name}' is disabled, skipping")
            return

        try:
            logger.info(f"Connecting to MCP server: {config.name}")

            # Create server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env if config.env else None,
            )

            # Create stdio transport and enter context
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            # Create session and enter context
            session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the session
            await session.initialize()

            # Store session
            self._sessions[config.name] = session

            # Fetch and cache tools
            await self._cache_tools(config.name, session)

            logger.info(
                f"Connected to MCP server '{config.name}' with "
                f"{len([t for t in self._tools.values() if t.server == config.name])} tools"
            )

        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
            raise ConnectionError(f"Failed to connect to {config.name}: {e}") from e

    async def connect_all(
        self,
        configs: list[MCPServerConfig],
        fail_fast: bool = False,
    ) -> dict[str, Exception | None]:
        """Connect to multiple MCP servers.

        Args:
            configs: List of server configurations.
            fail_fast: If True, stop on first failure. If False, continue
                connecting to other servers even if some fail.

        Returns:
            Dict mapping server names to exceptions (None if successful).
        """
        results: dict[str, Exception | None] = {}

        for config in configs:
            try:
                await self.connect(config)
                results[config.name] = None
            except Exception as e:
                results[config.name] = e
                if fail_fast:
                    break

        self._initialized = True
        return results

    async def disconnect(self) -> None:
        """Disconnect from all MCP servers."""
        await self._exit_stack.aclose()
        self._sessions.clear()
        self._tools.clear()
        self._initialized = False
        logger.info("Disconnected from all MCP servers")

    # =========================================================================
    # Tool Discovery
    # =========================================================================

    async def _cache_tools(self, server_name: str, session: ClientSession) -> None:
        """Cache tools from a server."""
        try:
            tools_result = await session.list_tools()

            for tool in tools_result.tools:
                scoped_name = f"{server_name}.{tool.name}"
                self._tools[scoped_name] = MCPTool(
                    name=scoped_name,
                    server=server_name,
                    original_name=tool.name,
                    description=tool.description or "",
                    input_schema=tool.inputSchema or {},
                )
        except Exception as e:
            logger.warning(f"Failed to list tools from '{server_name}': {e}")

    async def refresh_tools(self, server_name: str | None = None) -> None:
        """Refresh the tool cache.

        Args:
            server_name: Specific server to refresh, or None for all.
        """
        if server_name:
            if server_name in self._sessions:
                # Clear existing tools from this server
                self._tools = {
                    name: tool
                    for name, tool in self._tools.items()
                    if tool.server != server_name
                }
                await self._cache_tools(server_name, self._sessions[server_name])
        else:
            # Refresh all
            self._tools.clear()
            for name, session in self._sessions.items():
                await self._cache_tools(name, session)

    def list_tools(self, server_name: str | None = None) -> list[MCPTool]:
        """List available tools.

        Args:
            server_name: Filter to specific server, or None for all.

        Returns:
            List of available tools.
        """
        if server_name:
            return [t for t in self._tools.values() if t.server == server_name]
        return list(self._tools.values())

    def get_tool(self, scoped_name: str) -> MCPTool | None:
        """Get a specific tool by scoped name.

        Args:
            scoped_name: Tool name in 'server.tool_name' format.

        Returns:
            Tool info or None if not found.
        """
        return self._tools.get(scoped_name)

    def get_tools_for_llm(self) -> list[dict[str, Any]]:
        """Get tools formatted for LLM function calling.

        Returns tool definitions in a format suitable for OpenAI/Claude
        tool use APIs.

        Returns:
            List of tool definitions.
        """
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
            for tool in self._tools.values()
        ]

    # =========================================================================
    # Tool Execution
    # =========================================================================

    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool on an MCP server.

        Args:
            server_name: Name of the server.
            tool_name: Name of the tool (without server prefix).
            arguments: Tool arguments.

        Returns:
            Tool result as a dict with 'content' and 'is_error' keys.

        Raises:
            ValueError: If server not connected or tool not found.
        """
        if server_name not in self._sessions:
            raise ValueError(f"Not connected to server: {server_name}")

        scoped_name = f"{server_name}.{tool_name}"
        if scoped_name not in self._tools:
            raise ValueError(f"Tool not found: {scoped_name}")

        session = self._sessions[server_name]

        try:
            result = await session.call_tool(tool_name, arguments or {})
            return self._format_result(result)
        except Exception as e:
            logger.error(f"Error calling tool {scoped_name}: {e}")
            return {"content": str(e), "is_error": True}

    async def call_tool_scoped(
        self,
        scoped_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a tool using its scoped name.

        Args:
            scoped_name: Full tool name in 'server.tool_name' format.
            arguments: Tool arguments.

        Returns:
            Tool result.

        Raises:
            ValueError: If tool not found.
        """
        tool = self._tools.get(scoped_name)
        if not tool:
            raise ValueError(f"Tool not found: {scoped_name}")

        return await self.call_tool(tool.server, tool.original_name, arguments)

    def _format_result(self, result: CallToolResult) -> dict[str, Any]:
        """Format a tool result into a standard dict."""
        content_parts = []

        for item in result.content:
            if isinstance(item, TextContent):
                content_parts.append(item.text)
            else:
                # Handle other content types (images, etc.)
                content_parts.append(str(item))

        return {
            "content": "\n".join(content_parts) if content_parts else "",
            "is_error": result.isError or False,
        }

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "MCPClient":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context, disconnecting from all servers."""
        await self.disconnect()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def connected_servers(self) -> list[str]:
        """List of connected server names."""
        return list(self._sessions.keys())

    @property
    def is_initialized(self) -> bool:
        """Whether the client has been initialized."""
        return self._initialized


# =============================================================================
# Factory Functions
# =============================================================================


async def create_mcp_client(
    configs: list[MCPServerConfig],
    fail_fast: bool = False,
) -> MCPClient:
    """Create and connect an MCP client.

    Convenience function that creates a client and connects to all servers.

    Args:
        configs: List of server configurations.
        fail_fast: Stop on first connection failure.

    Returns:
        Connected MCP client.

    Raises:
        ImportError: If MCP package is not installed.

    Example:
        client = await create_mcp_client([
            MCPServerConfig(name="ha", command="mcp-server-home-assistant"),
        ])

    Note:
        Requires the 'mcp' optional dependency: pip install draagon-ai[mcp]
    """
    _check_mcp_available()
    client = MCPClient()
    await client.connect_all(configs, fail_fast=fail_fast)
    return client
