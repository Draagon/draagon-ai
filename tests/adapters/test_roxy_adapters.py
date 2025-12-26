"""Tests for Roxy adapters.

These tests use mock versions of Roxy's services to verify that the adapters
correctly translate between Roxy's interfaces and draagon-ai's protocols.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.adapters.roxy import (
    RoxyLLMAdapter,
    RoxyMemoryAdapter,
    RoxyToolAdapter,
)
from draagon_ai.orchestration.protocols import (
    LLMMessage,
    LLMResponse,
    MemorySearchResult,
    ToolCall,
    ToolResult,
)


# =============================================================================
# Mock Roxy Services
# =============================================================================


class MockRoxyLLMService:
    """Mock of Roxy's LLMService for testing."""

    def __init__(self):
        self.chat = AsyncMock(return_value={
            "content": "Hello! How can I help you?",
            "model": "llama-3.1-8b-instant",
            "provider": "groq",
            "elapsed_ms": 150,
            "usage": {"prompt_tokens": 10, "completion_tokens": 8},
        })
        self.embed = AsyncMock(return_value=[0.1, 0.2, 0.3] * 256)  # 768 dims


class MockRoxyMemoryService:
    """Mock of Roxy's MemoryService for testing."""

    def __init__(self):
        self.search = AsyncMock(return_value=[
            {
                "id": "mem_123",
                "content": "Doug's birthday is March 15",
                "score": 0.85,
                "type": "fact",
                "scope": "private",
                "entities": ["doug", "birthday"],
                "created_at": "2024-01-15T10:00:00Z",
                "importance": 0.8,
            },
            {
                "id": "mem_456",
                "content": "Doug prefers dark mode",
                "score": 0.72,
                "type": "preference",
                "scope": "private",
                "entities": ["doug", "dark mode"],
                "created_at": "2024-01-10T10:00:00Z",
                "importance": 0.6,
            },
        ])
        self.search_with_self_rag = AsyncMock(return_value={
            "results": [
                {
                    "id": "mem_789",
                    "content": "WiFi password is hunter2",
                    "score": 0.91,
                    "type": "fact",
                    "entities": ["wifi"],
                }
            ],
            "assessment": {"score": 0.85, "relevant": True},
        })
        self.search_with_crag = AsyncMock(return_value={
            "results": [
                {
                    "id": "mem_abc",
                    "content": "How to restart Roxy: systemctl restart roxy",
                    "score": 0.88,
                    "type": "skill",
                    "knowledge_strip": "systemctl restart roxy",
                }
            ],
            "grading": {"relevant": 1, "irrelevant": 0, "ambiguous": 0},
        })
        self.store = AsyncMock(return_value={
            "success": True,
            "memory_id": "mem_new_123",
            "type": "fact",
        })


class MockRoxyToolExecutor:
    """Mock of Roxy's ToolExecutor for testing."""

    def __init__(self):
        # Mock tool registry
        self.registry = MagicMock()
        self.registry.list_tools.return_value = [
            "get_time", "get_weather", "search_knowledge"
        ]
        self.registry.get.return_value = MagicMock(
            name="get_time",
            description="Get the current time and date",
            parameters=[],
            returns="Object with time, date, day",
        )

        # Mock MCP service
        self.mcp = MagicMock()
        self.mcp.is_available = False
        self.mcp.list_tools.return_value = []

        # Mock execute method
        self.execute = AsyncMock()

    def set_execute_result(self, result: dict):
        """Helper to set execute return value."""
        mock_result = MagicMock()
        mock_result.result = result.get("result")
        mock_result.error = result.get("error")
        self.execute.return_value = mock_result


# =============================================================================
# LLM Adapter Tests
# =============================================================================


class TestRoxyLLMAdapter:
    """Tests for RoxyLLMAdapter."""

    @pytest.fixture
    def llm_service(self):
        return MockRoxyLLMService()

    @pytest.fixture
    def adapter(self, llm_service):
        return RoxyLLMAdapter(llm_service)

    @pytest.mark.asyncio
    async def test_chat_basic(self, adapter, llm_service):
        """Test basic chat completion."""
        messages = [
            LLMMessage(role="system", content="You are helpful."),
            LLMMessage(role="user", content="Hello!"),
        ]

        response = await adapter.chat(messages)

        assert isinstance(response, LLMResponse)
        assert response.content == "Hello! How can I help you?"
        assert response.model == "llama-3.1-8b-instant"

        # Verify Roxy service was called correctly
        llm_service.chat.assert_called_once()
        call_args = llm_service.chat.call_args
        assert call_args.kwargs["messages"] == [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

    @pytest.mark.asyncio
    async def test_chat_with_model_hint_fast(self, adapter, llm_service):
        """Test model hint 'fast' uses Groq 8B."""
        messages = [LLMMessage(role="user", content="Quick question")]

        await adapter.chat(messages, model="fast")

        call_args = llm_service.chat.call_args
        assert call_args.kwargs["force_provider"] == "groq"
        assert call_args.kwargs["use_fast_model"] is True

    @pytest.mark.asyncio
    async def test_chat_with_model_hint_complex(self, adapter, llm_service):
        """Test model hint 'complex' uses Groq 70B."""
        messages = [LLMMessage(role="user", content="Complex reasoning")]

        await adapter.chat(messages, model="complex")

        call_args = llm_service.chat.call_args
        assert call_args.kwargs["force_provider"] == "groq"
        assert call_args.kwargs["use_fast_model"] is False

    @pytest.mark.asyncio
    async def test_chat_with_model_hint_deep(self, adapter, llm_service):
        """Test model hint 'deep' uses Claude."""
        messages = [LLMMessage(role="user", content="Nuanced question")]

        await adapter.chat(messages, model="deep")

        call_args = llm_service.chat.call_args
        assert call_args.kwargs["force_provider"] == "claude"

    @pytest.mark.asyncio
    async def test_chat_with_json_mode(self, adapter, llm_service):
        """Test JSON mode is passed through."""
        messages = [LLMMessage(role="user", content="Give me JSON")]

        await adapter.chat(messages, json_mode=True)

        call_args = llm_service.chat.call_args
        assert call_args.kwargs["json_mode"] is True

    @pytest.mark.asyncio
    async def test_chat_with_temperature(self, adapter, llm_service):
        """Test temperature is passed through."""
        messages = [LLMMessage(role="user", content="Be creative")]

        await adapter.chat(messages, temperature=0.9)

        call_args = llm_service.chat.call_args
        assert call_args.kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_embed(self, adapter, llm_service):
        """Test embedding generation."""
        result = await adapter.embed("test text")

        assert result is not None
        assert len(result) == 768
        llm_service.embed.assert_called_once_with("test text")


# =============================================================================
# Memory Adapter Tests
# =============================================================================


class TestRoxyMemoryAdapter:
    """Tests for RoxyMemoryAdapter."""

    @pytest.fixture
    def memory_service(self):
        return MockRoxyMemoryService()

    @pytest.fixture
    def adapter(self, memory_service):
        return RoxyMemoryAdapter(memory_service)

    @pytest.mark.asyncio
    async def test_search_basic(self, adapter, memory_service):
        """Test basic memory search."""
        results = await adapter.search(
            query="Doug's birthday",
            user_id="doug",
            use_self_rag=False,  # Use basic search
        )

        assert len(results) == 2
        assert all(isinstance(r, MemorySearchResult) for r in results)
        assert results[0].memory_id == "mem_123"
        assert results[0].content == "Doug's birthday is March 15"
        assert results[0].score == 0.85
        assert results[0].memory_type == "fact"
        assert "doug" in results[0].entities

    @pytest.mark.asyncio
    async def test_search_with_self_rag(self, adapter, memory_service):
        """Test search with Self-RAG assessment."""
        results = await adapter.search(
            query="WiFi password",
            user_id="doug",
            use_self_rag=True,  # Default
        )

        memory_service.search_with_self_rag.assert_called_once()
        assert len(results) == 1
        assert results[0].content == "WiFi password is hunter2"

    @pytest.mark.asyncio
    async def test_search_with_crag(self, adapter, memory_service):
        """Test search with CRAG grading."""
        results = await adapter.search(
            query="How to restart Roxy",
            user_id="doug",
            use_crag=True,
        )

        memory_service.search_with_crag.assert_called_once()
        assert len(results) == 1
        assert results[0].memory_type == "skill"
        assert "knowledge_strip" in results[0].metadata

    @pytest.mark.asyncio
    async def test_store_basic(self, adapter, memory_service):
        """Test storing a memory."""
        memory_id = await adapter.store(
            content="Lisa's favorite color is blue",
            user_id="doug",
            memory_type="fact",
            entities=["lisa", "color", "blue"],
            importance=0.7,
        )

        assert memory_id == "mem_new_123"
        memory_service.store.assert_called_once()

        call_args = memory_service.store.call_args
        assert call_args.kwargs["content"] == "Lisa's favorite color is blue"
        assert call_args.kwargs["user_id"] == "doug"
        assert call_args.kwargs["importance"] == 0.7

    @pytest.mark.asyncio
    async def test_store_with_options(self, adapter, memory_service):
        """Test storing with additional options."""
        await adapter.store(
            content="Household rule",
            user_id="doug",
            memory_type="instruction",
            scope="shared",
            household_id="mealing_home",
        )

        call_args = memory_service.store.call_args
        assert call_args.kwargs["scope"] == "shared"
        assert call_args.kwargs["household_id"] == "mealing_home"


# =============================================================================
# Tool Adapter Tests
# =============================================================================


class TestRoxyToolAdapter:
    """Tests for RoxyToolAdapter."""

    @pytest.fixture
    def tool_executor(self):
        return MockRoxyToolExecutor()

    @pytest.fixture
    def adapter(self, tool_executor):
        return RoxyToolAdapter(tool_executor)

    @pytest.mark.asyncio
    async def test_execute_success(self, adapter, tool_executor):
        """Test successful tool execution."""
        tool_executor.set_execute_result({
            "result": {"time": "3:45 PM", "date": "December 26, 2024"},
        })

        result = await adapter.execute(
            ToolCall(tool_name="get_time", arguments={}),
            context={"user_id": "doug"},
        )

        assert isinstance(result, ToolResult)
        assert result.success is True
        assert result.tool_name == "get_time"
        assert result.result["time"] == "3:45 PM"
        assert result.error is None
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_execute_failure(self, adapter, tool_executor):
        """Test tool execution with error."""
        tool_executor.set_execute_result({
            "error": "Service unavailable",
        })

        result = await adapter.execute(
            ToolCall(tool_name="get_weather", arguments={}),
            context={"user_id": "doug"},
        )

        assert result.success is False
        assert result.error == "Service unavailable"
        assert result.result is None

    @pytest.mark.asyncio
    async def test_execute_with_context(self, adapter, tool_executor):
        """Test context values are passed to executor."""
        tool_executor.set_execute_result({"result": {}})

        await adapter.execute(
            ToolCall(tool_name="call_service", arguments={"domain": "light"}),
            context={
                "user_id": "doug",
                "conversation_id": "conv_123",
                "area_id": "bedroom",
                "timezone": "America/New_York",
            },
        )

        call_args = tool_executor.execute.call_args
        assert call_args.kwargs["user_id"] == "doug"
        assert call_args.kwargs["conversation_id"] == "conv_123"
        assert call_args.kwargs["area_id"] == "bedroom"
        assert call_args.kwargs["timezone"] == "America/New_York"

    def test_list_tools(self, adapter):
        """Test listing available tools."""
        tools = adapter.list_tools()

        assert "get_time" in tools
        assert "get_weather" in tools
        assert "search_knowledge" in tools

    def test_list_tools_with_mcp(self, adapter, tool_executor):
        """Test listing tools includes MCP tools when available."""
        tool_executor.mcp.is_available = True
        tool_executor.mcp.list_tools.return_value = [
            MagicMock(name="calendar.list-events"),
            MagicMock(name="calendar.create-event"),
        ]

        tools = adapter.list_tools()

        assert len(tools) >= 5  # 3 local + 2 MCP

    def test_get_tool_description(self, adapter):
        """Test getting tool description."""
        desc = adapter.get_tool_description("get_time")

        assert desc == "Get the current time and date"

    def test_get_tool_description_not_found(self, adapter, tool_executor):
        """Test getting description for unknown tool."""
        tool_executor.registry.get.return_value = None

        desc = adapter.get_tool_description("unknown_tool")

        assert desc is None


# =============================================================================
# Integration Tests
# =============================================================================


class TestAdapterIntegration:
    """Integration tests for adapters working together."""

    @pytest.mark.asyncio
    async def test_llm_and_memory_together(self):
        """Test LLM and Memory adapters can work in a pipeline."""
        llm_service = MockRoxyLLMService()
        memory_service = MockRoxyMemoryService()

        llm = RoxyLLMAdapter(llm_service)
        memory = RoxyMemoryAdapter(memory_service)

        # Search for memories
        memories = await memory.search("user info", user_id="doug")
        assert len(memories) > 0

        # Use memory content in LLM prompt
        context = "\n".join(m.content for m in memories)
        messages = [
            LLMMessage(role="system", content=f"Context:\n{context}"),
            LLMMessage(role="user", content="When is Doug's birthday?"),
        ]

        response = await llm.chat(messages)
        assert response.content  # Got a response

    @pytest.mark.asyncio
    async def test_tool_with_llm_decision(self):
        """Test tool execution based on LLM decision."""
        llm_service = MockRoxyLLMService()
        llm_service.chat.return_value = {
            "content": '{"action": "get_time", "reason": "User asked for time"}',
            "model": "llama-3.1-8b-instant",
        }

        tool_executor = MockRoxyToolExecutor()
        tool_executor.set_execute_result({
            "result": {"time": "3:45 PM"},
        })

        llm = RoxyLLMAdapter(llm_service)
        tool = RoxyToolAdapter(tool_executor)

        # Simulate decision flow
        decision = await llm.chat(
            [LLMMessage(role="user", content="What time is it?")],
            json_mode=True,
        )
        assert "get_time" in decision.content

        # Execute the tool
        result = await tool.execute(
            ToolCall(tool_name="get_time", arguments={}),
            context={"user_id": "doug"},
        )
        assert result.success
        assert result.result["time"] == "3:45 PM"
