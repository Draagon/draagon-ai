"""Integration tests for Memory MCP Server.

These tests verify end-to-end functionality with real dependencies.
They require Qdrant and Ollama to be running.

To run these tests:
    pytest tests/mcp/test_mcp_integration.py -v --run-integration

Or run the manual integration test:
    python tests/mcp/test_mcp_integration.py
"""

import asyncio
import os
import pytest
from datetime import datetime
from typing import Any

# Skip all integration tests unless explicitly requested
pytestmark = pytest.mark.integration


# =============================================================================
# Integration Test Configuration
# =============================================================================

QDRANT_URL = os.environ.get("QDRANT_URL", "http://192.168.168.216:6333")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://192.168.168.200:11434")
TEST_COLLECTION = "test_mcp_integration"


# =============================================================================
# Integration Tests (require real services)
# =============================================================================


class TestMCPServerIntegration:
    """Integration tests for MCP server with real services."""

    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server with real Qdrant connection."""
        from draagon_ai.mcp import MCPConfig, MemoryMCPServer, MCPScope

        config = MCPConfig(
            qdrant_url=QDRANT_URL,
            ollama_url=OLLAMA_URL,
            qdrant_collection=TEST_COLLECTION,
            require_auth=False,
            log_level="DEBUG",
        )

        server = MemoryMCPServer(config=config)
        yield server

        # Cleanup
        await server.close()

    @pytest.mark.asyncio
    async def test_store_and_search_memory(self, mcp_server):
        """Test storing and searching a memory end-to-end."""
        # Get the store tool
        store_tool = mcp_server.mcp._tool_manager._tools.get("memory_store")
        search_tool = mcp_server.mcp._tool_manager._tools.get("memory_search")

        # Store a unique memory
        unique_content = f"Integration test memory created at {datetime.utcnow().isoformat()}"
        store_result = await store_tool.fn(
            content=unique_content,
            memory_type="fact",
            scope="private",
            entities=["integration", "test"],
        )

        assert store_result["success"] is True
        assert store_result["memory_id"] is not None

        # Search for it
        search_result = await search_tool.fn(
            query="integration test memory",
            limit=5,
        )

        assert search_result["success"] is True
        assert search_result["count"] > 0

        # Verify our memory is in results
        found = any(
            "integration test memory" in r["content"].lower()
            for r in search_result["results"]
        )
        assert found, "Stored memory not found in search results"

    @pytest.mark.asyncio
    async def test_full_crud_cycle(self, mcp_server):
        """Test create, read, update (via delete+create), delete cycle."""
        store_tool = mcp_server.mcp._tool_manager._tools.get("memory_store")
        get_tool = mcp_server.mcp._tool_manager._tools.get("memory_get")
        delete_tool = mcp_server.mcp._tool_manager._tools.get("memory_delete")

        # Create
        create_result = await store_tool.fn(
            content="CRUD test memory - original content",
            memory_type="fact",
            scope="private",
        )
        assert create_result["success"] is True
        memory_id = create_result["memory_id"]

        # Read
        get_result = await get_tool.fn(memory_id=memory_id)
        assert get_result["success"] is True
        assert "CRUD test memory" in get_result["memory"]["content"]

        # Delete
        delete_result = await delete_tool.fn(memory_id=memory_id)
        assert delete_result["success"] is True

        # Verify deleted
        get_after_delete = await get_tool.fn(memory_id=memory_id)
        assert get_after_delete["success"] is False

    @pytest.mark.asyncio
    async def test_scope_isolation(self, mcp_server):
        """Test that scope filtering works correctly."""
        from draagon_ai.mcp import MCPScope, ClientConfig

        store_tool = mcp_server.mcp._tool_manager._tools.get("memory_store")
        search_tool = mcp_server.mcp._tool_manager._tools.get("memory_search")

        # Store as PRIVATE scope
        await store_tool.fn(
            content="Private scope test memory",
            memory_type="fact",
            scope="private",
        )

        # Set restricted client context (PRIVATE only)
        mcp_server.set_client_context(
            ClientConfig(
                client_id="restricted",
                name="Restricted",
                allowed_scopes=[MCPScope.PRIVATE],
            )
        )

        # Search should only return PRIVATE scope results
        result = await search_tool.fn(
            query="scope test memory",
            limit=10,
        )

        assert result["success"] is True
        # All results should be PRIVATE or WORLD scope
        for r in result["results"]:
            assert r["scope"] in ["USER", "WORLD"], f"Unexpected scope: {r['scope']}"


# =============================================================================
# Manual Integration Test Runner
# =============================================================================


async def run_manual_integration_test():
    """Run manual integration tests for Claude Code integration.

    This function tests the MCP server end-to-end and can be used
    to verify the server works before configuring Claude Code.
    """
    print("\n" + "=" * 60)
    print("Memory MCP Server Integration Test")
    print("=" * 60)

    try:
        from draagon_ai.mcp import MCPConfig, MemoryMCPServer, MCPScope
    except ImportError as e:
        print(f"\nError: Could not import MCP server: {e}")
        print("Make sure draagon-ai is installed with MCP support.")
        return False

    # Create server
    print(f"\n1. Creating MCP server...")
    print(f"   Qdrant URL: {QDRANT_URL}")
    print(f"   Ollama URL: {OLLAMA_URL}")
    print(f"   Collection: {TEST_COLLECTION}")

    config = MCPConfig(
        qdrant_url=QDRANT_URL,
        ollama_url=OLLAMA_URL,
        qdrant_collection=TEST_COLLECTION,
        require_auth=False,
        log_level="INFO",
    )
    server = MemoryMCPServer(config=config)
    print("   ✓ Server created")

    # List tools
    print(f"\n2. Checking registered tools...")
    tools = server.mcp._tool_manager._tools
    print(f"   Found {len(tools)} tools:")
    for name in tools:
        print(f"   - {name}")
    assert len(tools) == 6, f"Expected 6 tools, got {len(tools)}"
    print("   ✓ All tools registered")

    # Test store
    print(f"\n3. Testing memory.store...")
    store_tool = tools.get("memory_store")
    test_content = f"Integration test at {datetime.utcnow().isoformat()}"
    result = await store_tool.fn(
        content=test_content,
        memory_type="fact",
        scope="private",
        entities=["integration", "test"],
    )
    print(f"   Result: {result}")
    assert result["success"] is True, f"Store failed: {result.get('error')}"
    memory_id = result["memory_id"]
    print(f"   ✓ Memory stored: {memory_id}")

    # Test search
    print(f"\n4. Testing memory.search...")
    search_tool = tools.get("memory_search")
    result = await search_tool.fn(query="integration test", limit=5)
    print(f"   Found {result['count']} results")
    assert result["success"] is True, f"Search failed: {result.get('error')}"
    print("   ✓ Search working")

    # Test get
    print(f"\n5. Testing memory.get...")
    get_tool = tools.get("memory_get")
    result = await get_tool.fn(memory_id=memory_id)
    assert result["success"] is True, f"Get failed: {result.get('error')}"
    print(f"   Content: {result['memory']['content'][:50]}...")
    print("   ✓ Get working")

    # Test delete
    print(f"\n6. Testing memory.delete...")
    delete_tool = tools.get("memory_delete")
    result = await delete_tool.fn(memory_id=memory_id)
    assert result["success"] is True, f"Delete failed: {result.get('error')}"
    print("   ✓ Delete working")

    # Cleanup
    await server.close()

    print("\n" + "=" * 60)
    print("✓ All integration tests passed!")
    print("=" * 60)

    # Print Claude Code configuration
    print("\n" + "-" * 60)
    print("Claude Code Configuration")
    print("-" * 60)
    print("""
To use with Claude Code, add to your claude_code_config.json:

{
  "mcpServers": {
    "memory": {
      "command": "python",
      "args": ["-m", "draagon_ai.mcp.server"],
      "env": {
        "QDRANT_URL": "%s",
        "OLLAMA_URL": "%s"
      }
    }
  }
}

Then restart Claude Code and try:
- "Remember that my favorite color is blue"
- "What is my favorite color?"
""" % (QDRANT_URL, OLLAMA_URL))

    return True


if __name__ == "__main__":
    success = asyncio.run(run_manual_integration_test())
    exit(0 if success else 1)
