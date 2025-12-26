"""End-to-end tests for MCP integration with real servers.

These tests require actual MCP servers to be available.
They are marked as slow/integration tests.
"""

import pytest
import asyncio

from draagon_ai.tools import (
    MCP_AVAILABLE,
    MCPClient,
    MCPServerConfig,
)


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
class TestMCPFetchServer:
    """Test MCP integration with mcp-server-fetch."""

    @pytest.fixture
    def fetch_config(self):
        """Config for mcp-server-fetch."""
        import sys
        return MCPServerConfig(
            name="fetch",
            command=sys.executable,  # Use the current Python interpreter
            args=["-m", "mcp_server_fetch"],
        )

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_connect_to_fetch_server(self, fetch_config):
        """Test connecting to the fetch MCP server."""
        async with MCPClient() as client:
            await client.connect(fetch_config)

            # Verify connected
            assert "fetch" in client.connected_servers

            # List tools
            tools = client.list_tools()
            assert len(tools) > 0

            # Should have a fetch tool
            tool_names = [t.original_name for t in tools]
            assert "fetch" in tool_names or any("fetch" in n.lower() for n in tool_names)

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_call_fetch_tool(self, fetch_config):
        """Test calling the fetch tool to get a webpage."""
        async with MCPClient() as client:
            await client.connect(fetch_config)

            # Find the fetch tool
            tools = client.list_tools()
            fetch_tool = next(
                (t for t in tools if "fetch" in t.original_name.lower()),
                None
            )

            if not fetch_tool:
                pytest.skip("No fetch tool found")

            # Call the fetch tool
            result = await client.call_tool(
                "fetch",
                fetch_tool.original_name,
                {"url": "https://example.com"}
            )

            # Should get content back (fetch server returns cleaned markdown)
            assert not result.get("is_error"), f"Error: {result.get('content')}"
            content = result.get("content", "")
            assert "example.com" in content.lower() or "iana" in content.lower(), f"Unexpected content: {content}"


# Note: Tests for Roxy's MCP service wrapper are in the roxy-voice-assistant repo
# in tests/suites/draagon_ai/test_mcp_e2e.py
