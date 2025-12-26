"""Tests for draagon_ai MCP client.

These tests verify the MCP client works correctly with mock servers.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.tools import (
    MCP_AVAILABLE,
    MCPClient,
    MCPServerConfig,
    MCPTool,
    create_mcp_client,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_session():
    """Create a mock MCP session."""
    session = MagicMock()
    session.initialize = AsyncMock()

    # Mock list_tools response
    tool_mock = MagicMock()
    tool_mock.name = "get_entity"
    tool_mock.description = "Get a Home Assistant entity"
    tool_mock.inputSchema = {"type": "object", "properties": {"entity_id": {"type": "string"}}}

    tools_result = MagicMock()
    tools_result.tools = [tool_mock]
    session.list_tools = AsyncMock(return_value=tools_result)

    # Mock call_tool response
    from dataclasses import dataclass

    @dataclass
    class MockTextContent:
        text: str
        type: str = "text"

    @dataclass
    class MockToolResult:
        content: list
        isError: bool = False

    session.call_tool = AsyncMock(return_value=MockToolResult(
        content=[MockTextContent(text='{"state": "on", "entity_id": "light.bedroom"}')],
        isError=False,
    ))

    return session


@pytest.fixture
def server_config():
    """Create a test server config."""
    return MCPServerConfig(
        name="test-server",
        command="test-mcp-server",
        args=["--port", "8080"],
        env={"API_KEY": "test123"},
    )


# =============================================================================
# MCPServerConfig Tests
# =============================================================================


class TestMCPServerConfig:
    """Tests for MCPServerConfig dataclass."""

    def test_config_creation(self):
        """Test creating a server config."""
        config = MCPServerConfig(
            name="home-assistant",
            command="mcp-server-home-assistant",
            args=["--url", "http://localhost:8123"],
            env={"HA_TOKEN": "secret"},
        )

        assert config.name == "home-assistant"
        assert config.command == "mcp-server-home-assistant"
        assert config.args == ["--url", "http://localhost:8123"]
        assert config.env == {"HA_TOKEN": "secret"}
        assert config.enabled is True  # Default

    def test_config_disabled(self):
        """Test disabled server config."""
        config = MCPServerConfig(
            name="disabled-server",
            command="some-command",
            enabled=False,
        )

        assert config.enabled is False

    def test_config_defaults(self):
        """Test default values."""
        config = MCPServerConfig(
            name="minimal",
            command="cmd",
        )

        assert config.args == []
        assert config.env == {}
        assert config.enabled is True


# =============================================================================
# MCPTool Tests
# =============================================================================


class TestMCPTool:
    """Tests for MCPTool dataclass."""

    def test_tool_creation(self):
        """Test creating a tool."""
        tool = MCPTool(
            name="home-assistant.get_entity",
            server="home-assistant",
            original_name="get_entity",
            description="Get entity state",
            input_schema={"type": "object"},
        )

        assert tool.name == "home-assistant.get_entity"
        assert tool.server == "home-assistant"
        assert tool.original_name == "get_entity"
        assert tool.description == "Get entity state"


# =============================================================================
# MCPClient Tests
# =============================================================================


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
class TestMCPClient:
    """Tests for MCPClient."""

    def test_client_initialization(self):
        """Test client can be initialized."""
        client = MCPClient()

        assert client.connected_servers == []
        assert client.is_initialized is False
        assert client.list_tools() == []

    @pytest.mark.asyncio
    async def test_connect_disabled_server(self, server_config):
        """Test connecting to a disabled server skips it."""
        server_config.enabled = False

        client = MCPClient()
        await client.connect(server_config)

        assert server_config.name not in client.connected_servers

    @pytest.mark.asyncio
    async def test_list_tools_empty(self):
        """Test listing tools when none connected."""
        client = MCPClient()

        tools = client.list_tools()

        assert tools == []

    @pytest.mark.asyncio
    async def test_get_tool_not_found(self):
        """Test getting a non-existent tool."""
        client = MCPClient()

        tool = client.get_tool("nonexistent.tool")

        assert tool is None

    @pytest.mark.asyncio
    async def test_call_tool_not_connected(self):
        """Test calling a tool on a non-connected server."""
        client = MCPClient()

        with pytest.raises(ValueError, match="Not connected to server"):
            await client.call_tool("nonexistent", "tool", {})

    @pytest.mark.asyncio
    async def test_get_tools_for_llm(self):
        """Test getting tools formatted for LLM."""
        client = MCPClient()
        # Manually add a tool for testing
        client._tools["test.get_time"] = MCPTool(
            name="test.get_time",
            server="test",
            original_name="get_time",
            description="Get the current time",
            input_schema={"type": "object", "properties": {}},
        )

        llm_tools = client.get_tools_for_llm()

        assert len(llm_tools) == 1
        assert llm_tools[0]["name"] == "test.get_time"
        assert llm_tools[0]["description"] == "Get the current time"
        assert "parameters" in llm_tools[0]

    @pytest.mark.asyncio
    async def test_list_tools_by_server(self):
        """Test listing tools filtered by server."""
        client = MCPClient()

        # Add tools from different servers
        client._tools["server1.tool1"] = MCPTool(
            name="server1.tool1",
            server="server1",
            original_name="tool1",
            description="Tool 1",
            input_schema={},
        )
        client._tools["server2.tool2"] = MCPTool(
            name="server2.tool2",
            server="server2",
            original_name="tool2",
            description="Tool 2",
            input_schema={},
        )

        # List all
        all_tools = client.list_tools()
        assert len(all_tools) == 2

        # Filter by server
        server1_tools = client.list_tools("server1")
        assert len(server1_tools) == 1
        assert server1_tools[0].name == "server1.tool1"

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test client can be used as async context manager."""
        async with MCPClient() as client:
            assert client is not None
            assert client.is_initialized is False


# =============================================================================
# Integration Tests with Mocks
# =============================================================================


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
class TestMCPClientWithMocks:
    """Integration tests using mocked MCP infrastructure."""

    @pytest.mark.asyncio
    async def test_connect_and_list_tools(self, mock_session, server_config):
        """Test connecting to a server and listing tools."""
        with patch("draagon_ai.tools.mcp_client.stdio_client") as mock_stdio:
            with patch("draagon_ai.tools.mcp_client.ClientSession") as mock_session_cls:
                # Setup mocks
                mock_transport = (MagicMock(), MagicMock())  # read, write streams
                mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                async with MCPClient() as client:
                    await client.connect(server_config)

                    # Verify connected
                    assert server_config.name in client.connected_servers

                    # Verify tools are cached
                    tools = client.list_tools()
                    assert len(tools) == 1
                    assert tools[0].name == "test-server.get_entity"
                    assert tools[0].description == "Get a Home Assistant entity"

    @pytest.mark.asyncio
    async def test_call_tool_success(self, mock_session, server_config):
        """Test calling a tool successfully."""
        with patch("draagon_ai.tools.mcp_client.stdio_client") as mock_stdio:
            with patch("draagon_ai.tools.mcp_client.ClientSession") as mock_session_cls:
                # Setup mocks
                mock_transport = (MagicMock(), MagicMock())
                mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                async with MCPClient() as client:
                    await client.connect(server_config)

                    # Call tool
                    result = await client.call_tool(
                        "test-server",
                        "get_entity",
                        {"entity_id": "light.bedroom"},
                    )

                    assert result["is_error"] is False
                    assert "light.bedroom" in result["content"]

    @pytest.mark.asyncio
    async def test_call_tool_scoped(self, mock_session, server_config):
        """Test calling a tool using scoped name."""
        with patch("draagon_ai.tools.mcp_client.stdio_client") as mock_stdio:
            with patch("draagon_ai.tools.mcp_client.ClientSession") as mock_session_cls:
                # Setup mocks
                mock_transport = (MagicMock(), MagicMock())
                mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                async with MCPClient() as client:
                    await client.connect(server_config)

                    # Call using scoped name
                    result = await client.call_tool_scoped(
                        "test-server.get_entity",
                        {"entity_id": "light.bedroom"},
                    )

                    assert result["is_error"] is False

    @pytest.mark.asyncio
    async def test_connect_all_partial_failure(self, mock_session):
        """Test connecting to multiple servers with one failing."""
        config1 = MCPServerConfig(name="server1", command="cmd1")
        config2 = MCPServerConfig(name="server2", command="cmd2")

        call_count = 0

        async def mock_connect_behavior(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call succeeds
                return (MagicMock(), MagicMock())
            else:
                # Second call fails
                raise ConnectionError("Failed to connect")

        with patch("draagon_ai.tools.mcp_client.stdio_client") as mock_stdio:
            with patch("draagon_ai.tools.mcp_client.ClientSession") as mock_session_cls:
                mock_stdio.return_value.__aenter__ = mock_connect_behavior
                mock_stdio.return_value.__aexit__ = AsyncMock()

                mock_session_cls.return_value.__aenter__ = AsyncMock(return_value=mock_session)
                mock_session_cls.return_value.__aexit__ = AsyncMock()

                async with MCPClient() as client:
                    results = await client.connect_all([config1, config2])

                    # First succeeded
                    assert results["server1"] is None
                    # Second failed
                    assert results["server2"] is not None
                    assert isinstance(results["server2"], Exception)

                    # Client should be initialized even with partial failure
                    assert client.is_initialized


# =============================================================================
# Factory Function Tests
# =============================================================================


@pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP package not installed")
class TestCreateMCPClient:
    """Tests for create_mcp_client factory function."""

    @pytest.mark.asyncio
    async def test_create_client_empty_configs(self):
        """Test creating client with no servers."""
        client = await create_mcp_client([])

        assert client.is_initialized
        assert client.connected_servers == []

        await client.disconnect()


# =============================================================================
# Graceful Fallback Tests (when MCP not installed)
# =============================================================================


class TestMCPNotAvailable:
    """Test behavior when MCP package is not installed."""

    def test_mcp_available_constant(self):
        """Test MCP_AVAILABLE constant is defined."""
        # MCP_AVAILABLE should be either True or False
        assert isinstance(MCP_AVAILABLE, bool)

    def test_imports_succeed(self):
        """Test that imports succeed even without MCP."""
        # This should not raise even if MCP is not installed
        from draagon_ai.tools import MCP_AVAILABLE, MCPServerConfig, MCPTool

        assert MCPServerConfig is not None
        assert MCPTool is not None
