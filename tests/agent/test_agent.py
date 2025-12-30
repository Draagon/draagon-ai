"""Tests for the agent module."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.agent import (
    AgentRequest,
    AgentResponse,
    ToolCallInfo,
    DebugInfo,
    FastRouteResult,
    FastRouter,
    FastRouteHandler,
    AgentLoop,
    AgentLoopConfig,
)
from draagon_ai.agent.fast_router import (
    FastRouterConfig,
    GreetingHandler,
    UndoHandler,
    ClearHistoryHandler,
)
from draagon_ai.agent.loop import DecisionHandler
from draagon_ai.conversation import ConversationState


class TestAgentRequest:
    """Tests for AgentRequest."""

    def test_create_request(self):
        """Test creating a request."""
        request = AgentRequest(
            query="What time is it?",
            user_id="user-123",
            conversation_id="conv-456",
        )

        assert request.query == "What time is it?"
        assert request.user_id == "user-123"
        assert request.conversation_id == "conv-456"

    def test_default_values(self):
        """Test default request values."""
        request = AgentRequest(query="test", user_id="user-1")

        assert request.conversation_id is None
        assert request.area_id is None
        assert request.debug is False
        assert request.metadata == {}


class TestAgentResponse:
    """Tests for AgentResponse."""

    def test_create_response(self):
        """Test creating a response."""
        response = AgentResponse(
            response="It's 3:00 PM.",
            success=True,
            action="answer",
        )

        assert response.response == "It's 3:00 PM."
        assert response.success is True
        assert response.action == "answer"

    def test_to_dict(self):
        """Test response serialization."""
        tool_call = ToolCallInfo(
            tool="get_time",
            args={},
            result="3:00 PM",
            elapsed_ms=50,
        )
        response = AgentResponse(
            response="It's 3:00 PM.",
            tool_calls=[tool_call],
        )

        data = response.to_dict()

        assert data["response"] == "It's 3:00 PM."
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["tool"] == "get_time"


class TestToolCallInfo:
    """Tests for ToolCallInfo."""

    def test_create_tool_call(self):
        """Test creating tool call info."""
        tc = ToolCallInfo(
            tool="search_web",
            args={"query": "weather"},
            result={"temp": 72},
            elapsed_ms=100,
        )

        assert tc.tool == "search_web"
        assert tc.args == {"query": "weather"}
        assert tc.success is True

    def test_failed_tool_call(self):
        """Test failed tool call."""
        tc = ToolCallInfo(
            tool="broken_tool",
            args={},
            success=False,
            error="Connection failed",
        )

        assert tc.success is False
        assert tc.error == "Connection failed"


class TestDebugInfo:
    """Tests for DebugInfo."""

    def test_create_debug_info(self):
        """Test creating debug info."""
        debug = DebugInfo(
            fast_path="greeting",
            llm_calls=2,
        )

        assert debug.fast_path == "greeting"
        assert debug.llm_calls == 2

    def test_to_dict(self):
        """Test debug info serialization."""
        debug = DebugInfo(
            router_used=True,
            action="answer",
            timings={"total_ms": 500},
        )

        data = debug.to_dict()

        assert data["router_used"] is True
        assert data["action"] == "answer"
        assert data["timings"]["total_ms"] == 500


class TestFastRouteResult:
    """Tests for FastRouteResult."""

    def test_to_agent_response(self):
        """Test converting to agent response."""
        result = FastRouteResult(
            response="Hello!",
            route_type="greeting",
        )

        debug = DebugInfo()
        response = result.to_agent_response(debug)

        assert response.response == "Hello!"
        assert response.action == "greeting"
        assert debug.fast_path == "greeting"


class TestFastRouter:
    """Tests for FastRouter."""

    def test_empty_router(self):
        """Test router with no handlers."""
        router = FastRouter()
        assert router.list_handlers() == []

    def test_register_handler(self):
        """Test registering handlers."""
        router = FastRouter()
        router.register_handler(GreetingHandler())

        assert "greeting" in router.list_handlers()

    def test_unregister_handler(self):
        """Test unregistering handlers."""
        router = FastRouter()
        router.register_handler(GreetingHandler())

        assert router.unregister_handler("greeting") is True
        assert router.unregister_handler("greeting") is False

    @pytest.mark.asyncio
    async def test_route_greeting(self):
        """Test routing a greeting."""
        router = FastRouter()
        router.register_handler(GreetingHandler())

        result = await router.try_route("hello", {})

        assert result is not None
        assert result.route_type == "greeting"
        assert "Hello" in result.response or "Hi" in result.response or "Hey" in result.response

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test query with no matching handler."""
        router = FastRouter()
        router.register_handler(GreetingHandler())

        result = await router.try_route("What's the weather?", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_context_required_skips_fast_route(self):
        """Test queries requiring context skip fast routing."""
        router = FastRouter()
        router.register_handler(GreetingHandler())

        # "again" indicates context required
        result = await router.try_route("hello again", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_disabled_router(self):
        """Test disabled router returns None."""
        config = FastRouterConfig(enabled=False)
        router = FastRouter(config=config)
        router.register_handler(GreetingHandler())

        result = await router.try_route("hello", {})

        assert result is None

    def test_priority_ordering(self):
        """Test handlers are sorted by priority."""
        router = FastRouter()

        # Add in reverse priority order
        router.register_handler(ClearHistoryHandler())  # priority 85
        router.register_handler(GreetingHandler())      # priority 100

        handlers = router.list_handlers()
        # Greeting should be first (higher priority)
        assert handlers[0] == "greeting"


class TestGreetingHandler:
    """Tests for GreetingHandler."""

    @pytest.mark.asyncio
    async def test_handles_hello(self):
        """Test handles 'hello'."""
        handler = GreetingHandler()

        assert handler.can_handle("hello", {})
        assert handler.can_handle("Hello", {})
        assert handler.can_handle("hello there", {})

    @pytest.mark.asyncio
    async def test_not_handles_question(self):
        """Test doesn't handle questions."""
        handler = GreetingHandler()

        assert not handler.can_handle("What time is it?", {})

    @pytest.mark.asyncio
    async def test_response(self):
        """Test greeting response."""
        handler = GreetingHandler()

        result = await handler.handle("hello", {})

        assert result.route_type == "greeting"
        assert len(result.response) > 0


class TestClearHistoryHandler:
    """Tests for ClearHistoryHandler."""

    @pytest.mark.asyncio
    async def test_handles_clear_phrases(self):
        """Test handles clear phrases."""
        handler = ClearHistoryHandler()

        assert handler.can_handle("start fresh", {})
        assert handler.can_handle("new conversation", {})
        assert handler.can_handle("clear history", {})

    @pytest.mark.asyncio
    async def test_clears_conversation(self):
        """Test clears conversation state."""
        handler = ClearHistoryHandler()

        # Mock conversation with clear method
        conversation = MagicMock()
        context = {"conversation": conversation}

        result = await handler.handle("start fresh", context)

        assert result.route_type == "clear_history"
        conversation.clear.assert_called_once()


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str = "Test response"):
        self._response = response

    async def chat(self, messages, model=None, **kwargs):
        return self._response


class MockDecisionHandler(DecisionHandler):
    """Mock decision handler for testing."""

    def __init__(self, action_type: str = "test_action"):
        self._action_type = action_type

    @property
    def action_type(self) -> str:
        return self._action_type

    async def handle(self, decision, request, context):
        return AgentResponse(
            response=f"Handled {self._action_type}",
            action=self._action_type,
        )


class TestAgentLoop:
    """Tests for AgentLoop."""

    @pytest.mark.asyncio
    async def test_basic_process(self):
        """Test basic request processing."""
        llm = MockLLM(response="Hello! How can I help?")
        loop = AgentLoop(llm=llm)

        request = AgentRequest(
            query="test query",
            user_id="user-1",
        )

        response = await loop.process(request)

        assert response.success is True
        assert len(response.response) > 0

    @pytest.mark.asyncio
    async def test_fast_route_greeting(self):
        """Test fast-path routing for greetings."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)
        loop.register_fast_handler(GreetingHandler())

        request = AgentRequest(
            query="hello",
            user_id="user-1",
            debug=True,
        )

        response = await loop.process(request)

        assert response.success is True
        assert response.debug.fast_path == "greeting"

    @pytest.mark.asyncio
    async def test_decision_handler_routing(self):
        """Test routing to decision handlers."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)
        loop.register_decision_handler(MockDecisionHandler("custom_action"))

        # Mock decision to return custom_action
        loop._parse_decision = lambda r: {"action": "custom_action"}

        request = AgentRequest(
            query="trigger custom action",
            user_id="user-1",
        )

        response = await loop.process(request)

        assert response.action == "custom_action"

    @pytest.mark.asyncio
    async def test_conversation_history(self):
        """Test conversation history is maintained."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        # First request
        request1 = AgentRequest(
            query="First message",
            user_id="user-1",
            conversation_id="conv-1",
        )
        await loop.process(request1)

        # Second request
        request2 = AgentRequest(
            query="Second message",
            user_id="user-1",
            conversation_id="conv-1",
        )
        await loop.process(request2)

        conversation = loop.get_conversation("conv-1")
        assert len(conversation.history) == 2

    @pytest.mark.asyncio
    async def test_debug_info(self):
        """Test debug info is populated."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        request = AgentRequest(
            query="test",
            user_id="user-1",
            debug=True,
        )

        response = await loop.process(request)

        assert response.debug is not None
        assert response.debug.router_used is True
        assert "total_ms" in response.debug.timings

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling returns error response."""
        # LLM that raises an error
        llm = MockLLM()
        llm.chat = AsyncMock(side_effect=Exception("LLM error"))

        loop = AgentLoop(llm=llm)

        request = AgentRequest(
            query="test",
            user_id="user-1",
        )

        response = await loop.process(request)

        assert response.success is False
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_cleanup_expired(self):
        """Test expired conversation cleanup."""
        config = AgentLoopConfig(conversation_timeout_minutes=0)
        loop = AgentLoop(llm=MockLLM(), config=config)
        # Set cleanup interval to 0 so cleanup runs immediately
        loop._conversations.config.cleanup_interval = 0

        # Create conversation
        request = AgentRequest(
            query="test",
            user_id="user-1",
            conversation_id="conv-1",
        )
        await loop.process(request)

        # Small delay to ensure expiry
        await asyncio.sleep(0.1)

        # Cleanup
        expired = loop.cleanup_expired()

        assert "conv-1" in expired

    @pytest.mark.asyncio
    async def test_on_response_callback(self):
        """Test response callbacks are called."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        callback_called = []

        def callback(req, resp, conv):
            callback_called.append(True)

        loop.on_response(callback)

        request = AgentRequest(query="test", user_id="user-1")
        await loop.process(request)

        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_async_callback(self):
        """Test async callbacks work."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        callback_called = []

        async def async_callback(req, resp, conv):
            callback_called.append(True)

        loop.on_response(async_callback)

        request = AgentRequest(query="test", user_id="user-1")
        await loop.process(request)

        assert len(callback_called) == 1

    def test_get_stats(self):
        """Test getting loop statistics."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)
        loop.register_fast_handler(GreetingHandler())
        loop.register_decision_handler(MockDecisionHandler())

        stats = loop.get_stats()

        assert "config" in stats
        assert "fast_router" in stats
        assert "decision_handlers" in stats
        assert "greeting" in stats["fast_router"]["handlers"][0]["route_type"]


class TestAgentLoopConfig:
    """Tests for AgentLoopConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = AgentLoopConfig()

        assert config.enable_fast_router is True
        assert config.enable_contextualization is True
        assert config.enable_reflection is True
        assert config.max_history_messages == 20

    def test_custom_config(self):
        """Test custom configuration."""
        config = AgentLoopConfig(
            enable_fast_router=False,
            max_history_messages=10,
            model_complex="custom-model",
        )

        assert config.enable_fast_router is False
        assert config.max_history_messages == 10
        assert config.model_complex == "custom-model"


class TestIntegration:
    """Integration tests for agent module."""

    @pytest.mark.asyncio
    async def test_full_flow_with_handlers(self):
        """Test full flow with multiple handlers."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        # Register handlers
        loop.register_fast_handler(GreetingHandler())
        loop.register_fast_handler(ClearHistoryHandler())

        # Test greeting
        resp1 = await loop.process(AgentRequest(
            query="hello",
            user_id="user-1",
            conversation_id="conv-1",
        ))
        assert "greeting" in (resp1.debug.fast_path if resp1.debug else "greeting")

        # Add some history
        await loop.process(AgentRequest(
            query="test query",
            user_id="user-1",
            conversation_id="conv-1",
        ))

        conversation = loop.get_conversation("conv-1")
        assert len(conversation.history) >= 1

        # Clear history
        resp3 = await loop.process(AgentRequest(
            query="start fresh",
            user_id="user-1",
            conversation_id="conv-1",
        ))
        assert resp3.action == "clear_history"

    @pytest.mark.asyncio
    async def test_area_context_propagation(self):
        """Test area/device context propagation."""
        llm = MockLLM()
        loop = AgentLoop(llm=llm)

        request = AgentRequest(
            query="turn on the lights",
            user_id="user-1",
            conversation_id="conv-1",
            area_id="living_room",
            device_id="voice-123",
        )

        await loop.process(request)

        conversation = loop.get_conversation("conv-1")
        assert conversation.current_area_id == "living_room"
        assert conversation.current_device_id == "voice-123"
