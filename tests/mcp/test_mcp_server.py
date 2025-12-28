"""Unit tests for the Memory MCP Server.

Tests cover:
- Configuration (MCPConfig, ClientConfig, MCPScope)
- Scope/type mapping
- Server initialization
- Tool registration
- Tool execution (with mocked memory provider)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass
from typing import Any

from draagon_ai.mcp.config import MCPConfig, ClientConfig, MCPScope
from draagon_ai.mcp.server import (
    MemoryMCPServer,
    create_memory_mcp_server,
    map_scope_to_draagon,
    map_type_to_draagon,
    SCOPE_MAPPING,
    TYPE_MAPPING,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockSearchResult:
    """Mock search result for testing."""

    id: str
    content: str
    memory_type: str
    scope: str
    score: float
    importance: float = 0.5
    entities: list[str] = None
    created_at: str = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


@pytest.fixture
def mock_memory_provider():
    """Create a mock memory provider."""
    provider = AsyncMock()

    # Mock store
    provider.store = AsyncMock(return_value={"memory_id": "test-memory-123"})

    # Mock search
    provider.search = AsyncMock(
        return_value=[
            MockSearchResult(
                id="mem-1",
                content="Test memory content",
                memory_type="FACT",
                scope="USER",
                score=0.95,
                importance=0.7,
                entities=["test", "memory"],
            ),
            MockSearchResult(
                id="mem-2",
                content="Another memory",
                memory_type="SKILL",
                scope="CONTEXT",
                score=0.85,
                importance=0.5,
            ),
        ]
    )

    # Mock get
    provider.get = AsyncMock(
        return_value=MockSearchResult(
            id="mem-1",
            content="Test memory content",
            memory_type="FACT",
            scope="USER",
            score=1.0,
            importance=0.7,
            entities=["test"],
        )
    )

    # Mock delete
    provider.delete = AsyncMock(return_value=True)

    # Mock close
    provider.close = AsyncMock()

    return provider


@pytest.fixture
def mcp_config():
    """Create a test MCP config."""
    return MCPConfig(
        qdrant_url="http://test:6333",
        ollama_url="http://test:11434",
        qdrant_collection="test_memories",
        require_auth=False,
        default_user_id="test-user",
        default_agent_id="test-agent",
        log_level="ERROR",  # Quiet logs in tests
    )


@pytest.fixture
def mcp_server(mcp_config, mock_memory_provider):
    """Create a test MCP server with mocked memory."""
    server = MemoryMCPServer(config=mcp_config, memory_provider=mock_memory_provider)
    return server


# =============================================================================
# Test MCPScope
# =============================================================================


class TestMCPScope:
    """Tests for MCPScope enum."""

    def test_scope_values(self):
        """Test scope enum values."""
        assert MCPScope.PRIVATE.value == "private"
        assert MCPScope.SHARED.value == "shared"
        assert MCPScope.SYSTEM.value == "system"

    def test_scope_is_string(self):
        """Test scope inherits from str."""
        assert isinstance(MCPScope.PRIVATE, str)
        assert MCPScope.PRIVATE == "private"


# =============================================================================
# Test ClientConfig
# =============================================================================


class TestClientConfig:
    """Tests for ClientConfig."""

    def test_default_values(self):
        """Test default client config values."""
        config = ClientConfig(client_id="test", name="Test Client")

        assert config.client_id == "test"
        assert config.name == "Test Client"
        assert config.allowed_scopes == [MCPScope.PRIVATE]
        assert config.default_user_id is None
        assert config.default_agent_id == "claude-code"
        assert config.max_requests_per_minute == 60

    def test_custom_values(self):
        """Test client config with custom values."""
        config = ClientConfig(
            client_id="custom",
            name="Custom Client",
            allowed_scopes=[MCPScope.PRIVATE, MCPScope.SHARED],
            default_user_id="custom-user",
            default_agent_id="custom-agent",
            max_requests_per_minute=120,
        )

        assert config.client_id == "custom"
        assert len(config.allowed_scopes) == 2
        assert config.default_user_id == "custom-user"
        assert config.max_requests_per_minute == 120


# =============================================================================
# Test MCPConfig
# =============================================================================


class TestMCPConfig:
    """Tests for MCPConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = MCPConfig()

        assert "6333" in config.qdrant_url
        assert "11434" in config.ollama_url
        assert config.embedding_model == "nomic-embed-text"
        assert config.embedding_dimension == 768
        assert config.require_auth is False
        assert config.search_limit_default == 5
        assert config.search_limit_max == 50

    def test_custom_values(self):
        """Test config with custom values."""
        config = MCPConfig(
            qdrant_url="http://custom:1234",
            ollama_url="http://custom:5678",
            embedding_model="custom-model",
            embedding_dimension=512,
            require_auth=True,
        )

        assert config.qdrant_url == "http://custom:1234"
        assert config.ollama_url == "http://custom:5678"
        assert config.embedding_model == "custom-model"
        assert config.embedding_dimension == 512
        assert config.require_auth is True

    def test_from_env(self, monkeypatch):
        """Test config from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://env:6333")
        monkeypatch.setenv("OLLAMA_URL", "http://env:11434")
        monkeypatch.setenv("MCP_REQUIRE_AUTH", "true")
        monkeypatch.setenv("MCP_DEFAULT_USER", "env-user")

        config = MCPConfig.from_env()

        assert config.qdrant_url == "http://env:6333"
        assert config.ollama_url == "http://env:11434"
        assert config.require_auth is True
        assert config.default_user_id == "env-user"

    def test_add_client(self):
        """Test adding a client."""
        config = MCPConfig()
        config.add_client(
            api_key="test-key",
            client_id="test-client",
            name="Test Client",
            scopes=[MCPScope.PRIVATE, MCPScope.SHARED],
            user_id="test-user",
        )

        client = config.get_client("test-key")
        assert client is not None
        assert client.client_id == "test-client"
        assert client.name == "Test Client"
        assert len(client.allowed_scopes) == 2
        assert client.default_user_id == "test-user"

    def test_get_client_not_found(self):
        """Test getting non-existent client."""
        config = MCPConfig()
        assert config.get_client("invalid-key") is None
        assert config.get_client(None) is None


# =============================================================================
# Test Scope/Type Mapping
# =============================================================================


class TestScopeMapping:
    """Tests for scope mapping functions."""

    def test_scope_mapping_values(self):
        """Test all scope mappings exist."""
        assert "private" in SCOPE_MAPPING
        assert "shared" in SCOPE_MAPPING
        assert "system" in SCOPE_MAPPING

        assert SCOPE_MAPPING["private"] == "USER"
        assert SCOPE_MAPPING["shared"] == "CONTEXT"
        assert SCOPE_MAPPING["system"] == "WORLD"

    def test_map_scope_to_draagon(self):
        """Test scope mapping function."""
        assert map_scope_to_draagon("private") == "USER"
        assert map_scope_to_draagon("shared") == "CONTEXT"
        assert map_scope_to_draagon("system") == "WORLD"

    def test_map_scope_case_insensitive(self):
        """Test scope mapping is case insensitive."""
        assert map_scope_to_draagon("PRIVATE") == "USER"
        assert map_scope_to_draagon("Private") == "USER"

    def test_map_scope_unknown(self):
        """Test unknown scope defaults to USER."""
        assert map_scope_to_draagon("unknown") == "USER"
        assert map_scope_to_draagon("") == "USER"


class TestTypeMapping:
    """Tests for memory type mapping functions."""

    def test_type_mapping_values(self):
        """Test all type mappings exist."""
        assert "fact" in TYPE_MAPPING
        assert "skill" in TYPE_MAPPING
        assert "insight" in TYPE_MAPPING
        assert "preference" in TYPE_MAPPING
        assert "episodic" in TYPE_MAPPING
        assert "instruction" in TYPE_MAPPING

    def test_map_type_to_draagon(self):
        """Test type mapping function."""
        assert map_type_to_draagon("fact") == "FACT"
        assert map_type_to_draagon("skill") == "SKILL"
        assert map_type_to_draagon("insight") == "INSIGHT"
        assert map_type_to_draagon("preference") == "PREFERENCE"

    def test_map_type_case_insensitive(self):
        """Test type mapping is case insensitive."""
        assert map_type_to_draagon("FACT") == "FACT"
        assert map_type_to_draagon("Skill") == "SKILL"

    def test_map_type_unknown(self):
        """Test unknown type defaults to FACT."""
        assert map_type_to_draagon("unknown") == "FACT"
        assert map_type_to_draagon("") == "FACT"


# =============================================================================
# Test MemoryMCPServer
# =============================================================================


class TestMemoryMCPServerInit:
    """Tests for server initialization."""

    def test_server_init_default_config(self):
        """Test server init with default config."""
        server = MemoryMCPServer()

        assert server.config is not None
        assert server.mcp is not None
        assert server._memory_provider is None
        assert server._memory_initialized is False

    def test_server_init_custom_config(self, mcp_config):
        """Test server init with custom config."""
        server = MemoryMCPServer(config=mcp_config)

        assert server.config.qdrant_url == "http://test:6333"
        assert server.config.default_user_id == "test-user"

    def test_server_init_with_memory(self, mcp_config, mock_memory_provider):
        """Test server init with pre-configured memory."""
        server = MemoryMCPServer(
            config=mcp_config, memory_provider=mock_memory_provider
        )

        assert server._memory_provider is mock_memory_provider

    def test_server_has_mcp_instance(self, mcp_server):
        """Test server has FastMCP instance."""
        assert mcp_server.mcp is not None
        assert mcp_server.mcp.name == mcp_server.config.server_name


def lookup_tool(mcp_server, tool_name: str):
    """Helper to get a tool from the server."""
    return mcp_server.mcp._tool_manager._tools.get(tool_name)


class TestMemoryMCPServerTools:
    """Tests for server tool execution."""

    @pytest.mark.asyncio
    async def test_memory_store(self, mcp_server, mock_memory_provider):
        """Test memory.store tool."""
        store_tool = lookup_tool(mcp_server, "memory_store")

        assert store_tool is not None

        # Call the tool
        result = await store_tool.fn(
            content="Test content",
            memory_type="fact",
            scope="private",
            entities=["test"],
        )

        assert result["success"] is True
        assert result["memory_id"] == "test-memory-123"

        # Verify memory provider was called
        mock_memory_provider.store.assert_called_once()
        call_kwargs = mock_memory_provider.store.call_args.kwargs
        assert call_kwargs["content"] == "Test content"
        assert call_kwargs["memory_type"] == "FACT"
        assert call_kwargs["scope"] == "USER"

    @pytest.mark.asyncio
    async def test_memory_search(self, mcp_server, mock_memory_provider):
        """Test memory.search tool."""
        search_tool = lookup_tool(mcp_server, "memory_search")
        assert search_tool is not None

        result = await search_tool.fn(query="test query", limit=5)

        assert result["success"] is True
        assert result["count"] == 2
        assert len(result["results"]) == 2
        assert result["results"][0]["content"] == "Test memory content"
        assert result["results"][0]["score"] == 0.95

    @pytest.mark.asyncio
    async def test_memory_search_with_filters(self, mcp_server, mock_memory_provider):
        """Test memory.search with type and scope filters."""
        search_tool = lookup_tool(mcp_server, "memory_search")

        result = await search_tool.fn(
            query="test query", limit=10, memory_types=["fact", "skill"], scope="shared"
        )

        assert result["success"] is True

        # Verify filters were passed
        call_kwargs = mock_memory_provider.search.call_args.kwargs
        assert call_kwargs["memory_types"] == ["FACT", "SKILL"]
        assert call_kwargs["scopes"] == ["CONTEXT"]

    @pytest.mark.asyncio
    async def test_memory_list(self, mcp_server, mock_memory_provider):
        """Test memory.list tool."""
        list_tool = lookup_tool(mcp_server, "memory_list")
        assert list_tool is not None

        result = await list_tool.fn(memory_type="fact", limit=10)

        assert result["success"] is True
        assert result["count"] == 2  # From mock

    @pytest.mark.asyncio
    async def test_memory_get(self, mcp_server, mock_memory_provider):
        """Test memory.get tool."""
        get_tool = lookup_tool(mcp_server, "memory_get")
        assert get_tool is not None

        result = await get_tool.fn(memory_id="mem-1")

        assert result["success"] is True
        assert result["memory"]["id"] == "mem-1"
        assert result["memory"]["content"] == "Test memory content"

    @pytest.mark.asyncio
    async def test_memory_get_not_found(self, mcp_server, mock_memory_provider):
        """Test memory.get with non-existent ID."""
        mock_memory_provider.get.return_value = None

        tool = lookup_tool(mcp_server, "memory_get")
        result = await tool.fn(memory_id="non-existent")

        assert result["success"] is False
        assert "not found" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_memory_delete(self, mcp_server, mock_memory_provider):
        """Test memory.delete tool."""
        delete_tool = lookup_tool(mcp_server, "memory_delete")
        assert delete_tool is not None

        result = await delete_tool.fn(memory_id="mem-1")

        assert result["success"] is True
        mock_memory_provider.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_delete_not_found(self, mcp_server, mock_memory_provider):
        """Test memory.delete with non-existent ID."""
        mock_memory_provider.delete.return_value = False

        delete_tool = lookup_tool(mcp_server, "memory_delete")
        result = await delete_tool.fn(memory_id="non-existent")

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_beliefs_reconcile(self, mcp_server, mock_memory_provider):
        """Test beliefs.reconcile tool."""
        reconcile_tool = lookup_tool(mcp_server, "beliefs_reconcile")
        assert reconcile_tool is not None

        result = await reconcile_tool.fn(
            observation="Test observation", source="user", confidence=0.9
        )

        assert result["success"] is True
        assert result["memory_id"] is not None

        # Verify it was stored as a fact with observation metadata
        call_kwargs = mock_memory_provider.store.call_args.kwargs
        assert call_kwargs["content"] == "Test observation"
        assert call_kwargs["metadata"]["is_observation"] is True
        assert call_kwargs["metadata"]["confidence"] == 0.9


class TestMemoryMCPServerUserAgent:
    """Tests for user/agent ID resolution."""

    def test_get_user_id_explicit(self, mcp_server):
        """Test explicit user ID is used."""
        result = mcp_server._get_user_id("explicit-user")
        assert result == "explicit-user"

    def test_get_user_id_default(self, mcp_server):
        """Test default user ID when not specified."""
        result = mcp_server._get_user_id(None)
        assert result == "test-user"  # From mcp_config fixture

    def test_get_user_id_from_client(self, mcp_server):
        """Test user ID from client context."""
        mcp_server._client_context = ClientConfig(
            client_id="test",
            name="Test",
            default_user_id="client-user",
        )

        result = mcp_server._get_user_id(None)
        assert result == "client-user"

    def test_get_agent_id_explicit(self, mcp_server):
        """Test explicit agent ID is used."""
        result = mcp_server._get_agent_id("explicit-agent")
        assert result == "explicit-agent"

    def test_get_agent_id_default(self, mcp_server):
        """Test default agent ID when not specified."""
        result = mcp_server._get_agent_id(None)
        assert result == "test-agent"  # From mcp_config fixture


class TestMemoryMCPServerLimits:
    """Tests for search/list limits."""

    @pytest.mark.asyncio
    async def test_search_limit_clamped(self, mcp_server, mock_memory_provider):
        """Test search limit is clamped to max."""
        search_tool = lookup_tool(mcp_server, "memory_search")

        # Request more than max
        await search_tool.fn(query="test", limit=1000)

        # Verify limit was clamped
        call_kwargs = mock_memory_provider.search.call_args.kwargs
        assert call_kwargs["limit"] <= mcp_server.config.search_limit_max

    @pytest.mark.asyncio
    async def test_list_limit_clamped(self, mcp_server, mock_memory_provider):
        """Test list limit is clamped to max."""
        list_tool = lookup_tool(mcp_server, "memory_list")

        # Request more than max
        await list_tool.fn(limit=1000)

        # Verify limit was clamped
        call_kwargs = mock_memory_provider.search.call_args.kwargs
        assert call_kwargs["limit"] <= mcp_server.config.list_limit_max


# =============================================================================
# Test Factory Function
# =============================================================================


class TestFactoryFunction:
    """Tests for create_memory_mcp_server factory."""

    def test_create_with_defaults(self):
        """Test factory with default config."""
        server = create_memory_mcp_server()

        assert isinstance(server, MemoryMCPServer)
        assert server.config is not None

    def test_create_with_config(self, mcp_config):
        """Test factory with custom config."""
        server = create_memory_mcp_server(config=mcp_config)

        assert server.config.qdrant_url == "http://test:6333"

    def test_create_with_memory(self, mcp_config, mock_memory_provider):
        """Test factory with pre-configured memory."""
        server = create_memory_mcp_server(
            config=mcp_config, memory_provider=mock_memory_provider
        )

        assert server._memory_provider is mock_memory_provider


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in tools."""

    @pytest.mark.asyncio
    async def test_store_error_handling(self, mcp_server, mock_memory_provider):
        """Test error handling in store."""
        mock_memory_provider.store.side_effect = Exception("Store failed")

        store_tool = lookup_tool(mcp_server, "memory_store")
        result = await store_tool.fn(content="test", memory_type="fact")

        assert result["success"] is False
        assert "Store failed" in result["error"]

    @pytest.mark.asyncio
    async def test_search_error_handling(self, mcp_server, mock_memory_provider):
        """Test error handling in search."""
        mock_memory_provider.search.side_effect = Exception("Search failed")

        search_tool = lookup_tool(mcp_server, "memory_search")
        result = await search_tool.fn(query="test")

        assert result["success"] is False
        assert "Search failed" in result["error"]
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_get_error_handling(self, mcp_server, mock_memory_provider):
        """Test error handling in get."""
        mock_memory_provider.get.side_effect = Exception("Get failed")

        tool = lookup_tool(mcp_server, "memory_get")
        result = await tool.fn(memory_id="test")

        assert result["success"] is False
        assert "Get failed" in result["error"]


# =============================================================================
# Test Tool Registration
# =============================================================================


class TestToolRegistration:
    """Tests for tool registration."""

    def test_all_tools_registered(self, mcp_server):
        """Test all expected tools are registered."""
        tools = mcp_server.mcp._tool_manager._tools

        tool_names = [t.name for t in tools.values()]

        assert "memory_store" in tool_names
        assert "memory_search" in tool_names
        assert "memory_list" in tool_names
        assert "memory_get" in tool_names
        assert "memory_delete" in tool_names
        assert "beliefs_reconcile" in tool_names

    def test_tool_count(self, mcp_server):
        """Test correct number of tools registered."""
        tools = mcp_server.mcp._tool_manager._tools

        # Should have 6 tools
        assert len(tools) == 6
