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

from draagon_ai.mcp.config import (
    MCPConfig,
    ClientConfig,
    MCPScope,
    SCOPE_HIERARCHY,
    can_read_scope,
    can_write_scope,
    get_readable_scopes,
    get_scope_level,
)
from draagon_ai.mcp.server import (
    AuthResult,
    AuthAuditEntry,
    MCPAuthenticator,
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
class MockMemoryInResult:
    """Mock Memory object inside a SearchResult."""

    id: str
    content: str
    memory_type: str
    scope: str
    importance: float = 0.5
    entities: list[str] = None
    created_at: str = None

    def __post_init__(self):
        if self.entities is None:
            self.entities = []


@dataclass
class MockSearchResult:
    """Mock search result for testing (mimics SearchResult with .memory attribute)."""

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
        # Create a memory attribute that mirrors the result's properties
        self.memory = MockMemoryInResult(
            id=self.id,
            content=self.content,
            memory_type=self.memory_type,
            scope=self.scope,
            importance=self.importance,
            entities=self.entities,
            created_at=self.created_at,
        )


@dataclass
class MockMemory:
    """Mock Memory object for testing."""
    id: str
    content: str = ""
    memory_type: str = "fact"
    scope: str = "user"


@pytest.fixture
def mock_memory_provider():
    """Create a mock memory provider."""
    provider = AsyncMock()

    # Mock store - returns a Memory-like object with .id attribute
    provider.store = AsyncMock(return_value=MockMemory(id="test-memory-123"))

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

        # Scope values are lowercase to match MemoryScope enum
        assert SCOPE_MAPPING["private"] == "user"
        assert SCOPE_MAPPING["shared"] == "context"
        assert SCOPE_MAPPING["system"] == "world"

    def test_map_scope_to_draagon(self):
        """Test scope mapping function."""
        assert map_scope_to_draagon("private") == "user"
        assert map_scope_to_draagon("shared") == "context"
        assert map_scope_to_draagon("system") == "world"

    def test_map_scope_case_insensitive(self):
        """Test scope mapping is case insensitive."""
        assert map_scope_to_draagon("PRIVATE") == "user"
        assert map_scope_to_draagon("Private") == "user"

    def test_map_scope_unknown(self):
        """Test unknown scope defaults to user."""
        assert map_scope_to_draagon("unknown") == "user"
        assert map_scope_to_draagon("") == "user"


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
        # Type values are lowercase to match MemoryType enum
        assert map_type_to_draagon("fact") == "fact"
        assert map_type_to_draagon("skill") == "skill"
        assert map_type_to_draagon("insight") == "insight"
        assert map_type_to_draagon("preference") == "preference"

    def test_map_type_case_insensitive(self):
        """Test type mapping is case insensitive."""
        assert map_type_to_draagon("FACT") == "fact"
        assert map_type_to_draagon("Skill") == "skill"

    def test_map_type_unknown(self):
        """Test unknown type defaults to fact."""
        assert map_type_to_draagon("unknown") == "fact"
        assert map_type_to_draagon("") == "fact"


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
        # Type and scope values are lowercase to match enum values
        assert call_kwargs["memory_type"] == "fact"
        assert call_kwargs["scope"] == "user"

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

        # Verify filters were passed (lowercase to match enum values)
        call_kwargs = mock_memory_provider.search.call_args.kwargs
        assert call_kwargs["memory_types"] == ["fact", "skill"]
        assert call_kwargs["scopes"] == ["context"]

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


# =============================================================================
# Test Scope Hierarchy Functions
# =============================================================================


class TestScopeHierarchy:
    """Tests for scope hierarchy and access control functions."""

    def test_scope_hierarchy_values(self):
        """Test scope hierarchy levels are correct."""
        assert SCOPE_HIERARCHY[MCPScope.PRIVATE] == 0
        assert SCOPE_HIERARCHY[MCPScope.SHARED] == 1
        assert SCOPE_HIERARCHY[MCPScope.SYSTEM] == 2

    def test_get_scope_level(self):
        """Test get_scope_level function."""
        assert get_scope_level(MCPScope.PRIVATE) == 0
        assert get_scope_level(MCPScope.SHARED) == 1
        assert get_scope_level(MCPScope.SYSTEM) == 2

    def test_get_scope_level_string(self):
        """Test get_scope_level with string input."""
        assert get_scope_level("private") == 0
        assert get_scope_level("shared") == 1
        assert get_scope_level("system") == 2

    def test_can_read_scope_system_always_readable(self):
        """Test SYSTEM scope is always readable."""
        # Even with only PRIVATE access, can read SYSTEM
        assert can_read_scope([MCPScope.PRIVATE], MCPScope.SYSTEM) is True
        assert can_read_scope([MCPScope.SHARED], MCPScope.SYSTEM) is True
        assert can_read_scope([MCPScope.SYSTEM], MCPScope.SYSTEM) is True

    def test_can_read_scope_own_scopes(self):
        """Test reading from own scopes is allowed."""
        assert can_read_scope([MCPScope.PRIVATE], MCPScope.PRIVATE) is True
        assert can_read_scope([MCPScope.SHARED], MCPScope.SHARED) is True
        assert can_read_scope([MCPScope.SYSTEM], MCPScope.SYSTEM) is True

    def test_can_read_scope_denied(self):
        """Test reading from non-allowed scopes is denied."""
        # PRIVATE only client cannot read SHARED
        assert can_read_scope([MCPScope.PRIVATE], MCPScope.SHARED) is False

    def test_can_read_scope_empty_list(self):
        """Test empty scope list denies all reads."""
        assert can_read_scope([], MCPScope.PRIVATE) is False
        assert can_read_scope([], MCPScope.SHARED) is False
        # System is always readable
        assert can_read_scope([], MCPScope.SYSTEM) is True

    def test_can_write_scope_allowed(self):
        """Test writing to allowed scopes works."""
        assert can_write_scope([MCPScope.PRIVATE], MCPScope.PRIVATE) is True
        assert can_write_scope([MCPScope.SHARED], MCPScope.SHARED) is True
        assert can_write_scope([MCPScope.SYSTEM], MCPScope.SYSTEM) is True

    def test_can_write_scope_denied(self):
        """Test writing to non-allowed scopes is denied."""
        assert can_write_scope([MCPScope.PRIVATE], MCPScope.SHARED) is False
        assert can_write_scope([MCPScope.PRIVATE], MCPScope.SYSTEM) is False
        assert can_write_scope([MCPScope.SHARED], MCPScope.SYSTEM) is False

    def test_can_write_scope_multiple_allowed(self):
        """Test multiple allowed scopes for write."""
        scopes = [MCPScope.PRIVATE, MCPScope.SHARED]
        assert can_write_scope(scopes, MCPScope.PRIVATE) is True
        assert can_write_scope(scopes, MCPScope.SHARED) is True
        assert can_write_scope(scopes, MCPScope.SYSTEM) is False

    def test_get_readable_scopes_private_only(self):
        """Test readable scopes for PRIVATE-only client."""
        readable = get_readable_scopes([MCPScope.PRIVATE])
        assert MCPScope.PRIVATE in readable
        assert MCPScope.SYSTEM in readable  # Always readable
        assert MCPScope.SHARED not in readable

    def test_get_readable_scopes_shared(self):
        """Test readable scopes for SHARED client."""
        readable = get_readable_scopes([MCPScope.SHARED])
        assert MCPScope.SHARED in readable
        assert MCPScope.SYSTEM in readable

    def test_get_readable_scopes_all(self):
        """Test readable scopes for full-access client."""
        all_scopes = [MCPScope.PRIVATE, MCPScope.SHARED, MCPScope.SYSTEM]
        readable = get_readable_scopes(all_scopes)
        assert len(readable) == 3
        assert all(s in readable for s in all_scopes)


# =============================================================================
# Test Scope Enforcement in Server
# =============================================================================


class TestScopeEnforcement:
    """Tests for scope enforcement in MemoryMCPServer."""

    @pytest.fixture
    def restricted_client(self):
        """Create a client with only PRIVATE scope."""
        return ClientConfig(
            client_id="restricted",
            name="Restricted Client",
            allowed_scopes=[MCPScope.PRIVATE],
        )

    @pytest.fixture
    def shared_client(self):
        """Create a client with PRIVATE and SHARED scopes."""
        return ClientConfig(
            client_id="shared",
            name="Shared Client",
            allowed_scopes=[MCPScope.PRIVATE, MCPScope.SHARED],
        )

    @pytest.fixture
    def full_access_client(self):
        """Create a client with all scopes."""
        return ClientConfig(
            client_id="full",
            name="Full Access Client",
            allowed_scopes=[MCPScope.PRIVATE, MCPScope.SHARED, MCPScope.SYSTEM],
        )

    def test_get_allowed_scopes_no_context(self, mcp_server):
        """Test allowed scopes without client context."""
        # Uses config's allowed_scopes
        scopes = mcp_server._get_allowed_scopes()
        assert len(scopes) == 3  # Default config has all scopes

    def test_get_allowed_scopes_with_context(self, mcp_server, restricted_client):
        """Test allowed scopes with client context."""
        mcp_server.set_client_context(restricted_client)
        scopes = mcp_server._get_allowed_scopes()
        assert scopes == [MCPScope.PRIVATE]

    def test_check_write_permission_allowed(self, mcp_server, restricted_client):
        """Test write permission check when allowed."""
        mcp_server.set_client_context(restricted_client)
        allowed, error = mcp_server._check_write_permission("private")
        assert allowed is True
        assert error is None

    def test_check_write_permission_denied(self, mcp_server, restricted_client):
        """Test write permission check when denied."""
        mcp_server.set_client_context(restricted_client)
        allowed, error = mcp_server._check_write_permission("shared")
        assert allowed is False
        assert "Permission denied" in error
        assert "shared" in error.lower()

    def test_check_write_permission_invalid_scope(self, mcp_server):
        """Test write permission check with invalid scope."""
        allowed, error = mcp_server._check_write_permission("invalid")
        assert allowed is False
        assert "Invalid scope" in error

    def test_check_read_permission_allowed(self, mcp_server, shared_client):
        """Test read permission check when allowed."""
        mcp_server.set_client_context(shared_client)
        # SHARED client can read SHARED
        allowed, error = mcp_server._check_read_permission("shared")
        assert allowed is True
        assert error is None

    def test_check_read_permission_system_always_allowed(self, mcp_server, restricted_client):
        """Test SYSTEM scope is always readable."""
        mcp_server.set_client_context(restricted_client)
        allowed, error = mcp_server._check_read_permission("system")
        assert allowed is True
        assert error is None

    def test_check_read_permission_denied(self, mcp_server, restricted_client):
        """Test read permission check when denied."""
        mcp_server.set_client_context(restricted_client)
        allowed, error = mcp_server._check_read_permission("shared")
        assert allowed is False
        assert "Permission denied" in error

    def test_get_search_scopes_all_readable(self, mcp_server, shared_client):
        """Test search scopes returns all readable scopes."""
        mcp_server.set_client_context(shared_client)
        scopes = mcp_server._get_search_scopes(None)
        # Should include SHARED, PRIVATE, and SYSTEM (always readable)
        assert len(scopes) == 3

    def test_get_search_scopes_specific_allowed(self, mcp_server, shared_client):
        """Test search scopes with specific allowed scope."""
        mcp_server.set_client_context(shared_client)
        scopes = mcp_server._get_search_scopes("shared")
        assert scopes == ["context"]  # Mapped scope (lowercase)

    def test_get_search_scopes_specific_denied(self, mcp_server, restricted_client):
        """Test search scopes falls back when specific scope denied."""
        mcp_server.set_client_context(restricted_client)
        # Request SHARED but only have PRIVATE access
        scopes = mcp_server._get_search_scopes("shared")
        # Should fall back to all readable scopes (lowercase)
        assert "user" in scopes  # PRIVATE
        assert "world" in scopes  # SYSTEM (always readable)


class TestScopeEnforcementInTools:
    """Tests for scope enforcement in MCP tools."""

    @pytest.fixture
    def restricted_server(self, mcp_config, mock_memory_provider):
        """Create server with restricted client."""
        server = MemoryMCPServer(config=mcp_config, memory_provider=mock_memory_provider)
        server.set_client_context(
            ClientConfig(
                client_id="restricted",
                name="Restricted",
                allowed_scopes=[MCPScope.PRIVATE],
            )
        )
        return server

    @pytest.mark.asyncio
    async def test_store_denied_for_system_scope(self, restricted_server):
        """Test store is denied for unauthorized scope."""
        store_tool = lookup_tool(restricted_server, "memory_store")

        result = await store_tool.fn(
            content="Test content",
            memory_type="fact",
            scope="system",  # Not allowed for restricted client
        )

        assert result["success"] is False
        assert "Permission denied" in result["error"]

    @pytest.mark.asyncio
    async def test_store_allowed_for_private_scope(self, restricted_server, mock_memory_provider):
        """Test store is allowed for authorized scope."""
        # Re-add mock since fixture creates new server
        restricted_server._memory_provider = mock_memory_provider

        store_tool = lookup_tool(restricted_server, "memory_store")

        result = await store_tool.fn(
            content="Test content",
            memory_type="fact",
            scope="private",  # Allowed
        )

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_search_filters_to_readable_scopes(self, restricted_server, mock_memory_provider):
        """Test search only includes readable scopes."""
        restricted_server._memory_provider = mock_memory_provider

        search_tool = lookup_tool(restricted_server, "memory_search")
        await search_tool.fn(query="test")

        # Verify scopes passed to memory provider (lowercase)
        call_kwargs = mock_memory_provider.search.call_args.kwargs
        scopes = call_kwargs.get("scopes", [])
        # Should only include readable scopes (user + world)
        assert "user" in scopes or "world" in scopes

    @pytest.mark.asyncio
    async def test_search_denies_explicit_unauthorized_scope(self, restricted_server):
        """Test search denies explicit unauthorized scope."""
        search_tool = lookup_tool(restricted_server, "memory_search")

        result = await search_tool.fn(
            query="test",
            scope="shared",  # Not allowed for restricted client
        )

        assert result["success"] is False
        assert "Permission denied" in result["error"]


# =============================================================================
# Test Client Context Management
# =============================================================================


class TestClientContextManagement:
    """Tests for client context management."""

    def test_set_client_context(self, mcp_server):
        """Test setting client context."""
        client = ClientConfig(
            client_id="test-client",
            name="Test",
            allowed_scopes=[MCPScope.PRIVATE, MCPScope.SHARED],
            default_user_id="context-user",
        )

        mcp_server.set_client_context(client)

        assert mcp_server._client_context is client
        assert mcp_server._get_user_id(None) == "context-user"
        assert mcp_server._get_allowed_scopes() == [MCPScope.PRIVATE, MCPScope.SHARED]

    def test_clear_client_context(self, mcp_server):
        """Test clearing client context."""
        client = ClientConfig(client_id="test", name="Test")
        mcp_server.set_client_context(client)

        # Clear by setting to None
        mcp_server._client_context = None

        # Should fall back to config defaults
        assert mcp_server._get_user_id(None) == mcp_server.config.default_user_id


# =============================================================================
# Test Authentication
# =============================================================================


class TestMCPAuthenticator:
    """Tests for MCPAuthenticator."""

    @pytest.fixture
    def auth_config(self):
        """Create config with authentication enabled."""
        config = MCPConfig(require_auth=True)
        config.add_client(
            api_key="test-api-key-12345",
            client_id="test-client",
            name="Test Client",
            scopes=[MCPScope.PRIVATE, MCPScope.SHARED],
            user_id="test-user",
        )
        return config

    @pytest.fixture
    def authenticator(self, auth_config):
        """Create authenticator with test config."""
        return MCPAuthenticator(auth_config)

    def test_auth_not_required(self):
        """Test authentication when not required."""
        config = MCPConfig(require_auth=False)
        auth = MCPAuthenticator(config)

        result = auth.authenticate(None)
        assert result.authenticated is True
        assert result.error is None

    def test_auth_required_no_key(self, authenticator):
        """Test auth required but no key provided."""
        result = authenticator.authenticate(None)
        assert result.authenticated is False
        assert "Authentication required" in result.error

    def test_auth_invalid_key(self, authenticator):
        """Test auth with invalid key."""
        result = authenticator.authenticate("invalid-key")
        assert result.authenticated is False
        assert "Invalid API key" in result.error

    def test_auth_valid_key(self, authenticator):
        """Test auth with valid key."""
        result = authenticator.authenticate("test-api-key-12345")
        assert result.authenticated is True
        assert result.client_config is not None
        assert result.client_config.client_id == "test-client"

    def test_auth_client_config(self, authenticator):
        """Test client config is returned on success."""
        result = authenticator.authenticate("test-api-key-12345")
        assert result.client_config.allowed_scopes == [MCPScope.PRIVATE, MCPScope.SHARED]
        assert result.client_config.default_user_id == "test-user"


class TestAuthAuditLog:
    """Tests for authentication audit logging."""

    @pytest.fixture
    def auth_config(self):
        """Create config with authentication enabled."""
        config = MCPConfig(require_auth=True)
        config.add_client(
            api_key="valid-key-12345678",
            client_id="test-client",
            name="Test Client",
        )
        return config

    @pytest.fixture
    def authenticator(self, auth_config):
        """Create authenticator with test config."""
        return MCPAuthenticator(auth_config)

    def test_successful_auth_logged(self, authenticator):
        """Test successful auth is logged."""
        authenticator.authenticate("valid-key-12345678")
        log = authenticator.get_audit_log()
        assert len(log) == 1
        assert log[0]["success"] is True
        assert log[0]["client_id"] == "test-client"

    def test_failed_auth_logged(self, authenticator):
        """Test failed auth is logged."""
        authenticator.authenticate("invalid-key")
        log = authenticator.get_audit_log()
        assert len(log) == 1
        assert log[0]["success"] is False
        assert log[0]["error"] == "Invalid API key"

    def test_api_key_masked(self, authenticator):
        """Test API key is masked in log."""
        authenticator.authenticate("valid-key-12345678")
        log = authenticator.get_audit_log()
        # Key should be masked (first 4 + ... + last 4)
        assert "vali" in log[0]["api_key_prefix"]
        assert "5678" in log[0]["api_key_prefix"]
        assert "..." in log[0]["api_key_prefix"]
        # Full key should NOT appear
        assert "valid-key-12345678" not in log[0]["api_key_prefix"]

    def test_short_key_masked(self, authenticator):
        """Test short API key is masked."""
        authenticator.authenticate("short")
        log = authenticator.get_audit_log()
        # Short keys show first 2 chars + ...
        assert log[0]["api_key_prefix"].startswith("sh")
        assert "..." in log[0]["api_key_prefix"]

    def test_audit_log_limit(self, authenticator):
        """Test audit log limit."""
        for i in range(10):
            authenticator.authenticate(f"key-{i}")

        log = authenticator.get_audit_log(limit=5)
        assert len(log) == 5

    def test_no_key_logged(self, authenticator):
        """Test no key is logged as (none)."""
        authenticator.authenticate(None)
        log = authenticator.get_audit_log()
        assert log[0]["api_key_prefix"] == "(none)"


class TestServerAuthentication:
    """Tests for server-level authentication integration."""

    @pytest.fixture
    def auth_server(self, mock_memory_provider):
        """Create server with auth enabled."""
        config = MCPConfig(
            require_auth=True,
            qdrant_url="http://test:6333",
            log_level="ERROR",
        )
        config.add_client(
            api_key="server-test-key-1234",
            client_id="server-client",
            name="Server Client",
            scopes=[MCPScope.PRIVATE],
            user_id="auth-user",
        )
        return MemoryMCPServer(config=config, memory_provider=mock_memory_provider)

    def test_server_has_authenticator(self, auth_server):
        """Test server has authenticator."""
        assert hasattr(auth_server, "authenticator")
        assert isinstance(auth_server.authenticator, MCPAuthenticator)

    def test_server_authenticate_success(self, auth_server):
        """Test server authentication sets context."""
        result = auth_server.authenticate("server-test-key-1234")

        assert result.authenticated is True
        assert auth_server._client_context is not None
        assert auth_server._client_context.client_id == "server-client"

    def test_server_authenticate_failure(self, auth_server):
        """Test server authentication failure."""
        result = auth_server.authenticate("wrong-key")

        assert result.authenticated is False
        # Context should not be set on failure
        assert auth_server._client_context is None

    def test_server_get_audit_log(self, auth_server):
        """Test server audit log access."""
        auth_server.authenticate("server-test-key-1234")
        auth_server.authenticate("invalid-key")

        log = auth_server.get_auth_audit_log()
        assert len(log) == 2

    def test_authenticated_client_scope_enforcement(self, auth_server):
        """Test authenticated client has scope restrictions."""
        auth_server.authenticate("server-test-key-1234")

        # Client only has PRIVATE scope
        allowed, error = auth_server._check_write_permission("private")
        assert allowed is True

        allowed, error = auth_server._check_write_permission("shared")
        assert allowed is False
        assert "Permission denied" in error
