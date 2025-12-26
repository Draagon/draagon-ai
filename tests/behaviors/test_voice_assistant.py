"""Tests for the voice assistant behavior template."""

import pytest
from dataclasses import dataclass
from typing import Any

from draagon_ai.behaviors import (
    VOICE_ASSISTANT_TEMPLATE,
    BehaviorRegistry,
    create_voice_assistant_behavior,
    BehaviorStatus,
    BehaviorTier,
)
from draagon_ai.orchestration import (
    Agent,
    AgentConfig,
    AgentResponse,
)
from draagon_ai.orchestration.protocols import (
    LLMMessage,
    LLMResponse,
    LLMProvider,
    MemoryProvider,
    MemorySearchResult,
    ToolCall,
    ToolProvider,
    ToolResult,
)
from draagon_ai.orchestration.decision import DecisionEngine, DecisionContext


# =============================================================================
# Mock Providers
# =============================================================================


class MockLLMProvider:
    """Mock LLM that returns predictable responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[list[LLMMessage]] = []

    async def chat(
        self,
        messages: list[LLMMessage],
        model: str | None = None,
        **kwargs,
    ) -> LLMResponse:
        self.calls.append(messages)

        # Get the user message
        user_msg = next((m for m in messages if m.role == "user"), None)
        query = user_msg.content if user_msg else ""

        # Check for matching response
        for pattern, response in self.responses.items():
            if pattern.lower() in query.lower():
                return LLMResponse(content=response)

        # Default response
        return LLMResponse(
            content="""<response>
<action>answer</action>
<reasoning>Default response</reasoning>
<answer>I understand your question.</answer>
<model_tier>local</model_tier>
</response>"""
        )


class MockMemoryProvider:
    """Mock memory provider."""

    def __init__(self, memories: list[MemorySearchResult] | None = None):
        self.memories = memories or []
        self.stored: list[dict] = []

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        **kwargs,
    ) -> list[MemorySearchResult]:
        return self.memories[:limit]

    async def store(
        self,
        content: str,
        user_id: str,
        memory_type: str,
        **kwargs,
    ) -> str:
        self.stored.append({
            "content": content,
            "user_id": user_id,
            "memory_type": memory_type,
        })
        return f"mem_{len(self.stored)}"


class MockToolProvider:
    """Mock tool provider."""

    def __init__(self, results: dict[str, Any] | None = None):
        self.results = results or {}
        self.calls: list[ToolCall] = []

    async def execute(
        self,
        tool_call: ToolCall,
        context: dict,
    ) -> ToolResult:
        self.calls.append(tool_call)

        result = self.results.get(tool_call.tool_name, {"success": True})

        return ToolResult(
            tool_name=tool_call.tool_name,
            success=result.get("success", True),
            result=result.get("result", {}),
            error=result.get("error"),
        )

    def list_tools(self) -> list[str]:
        return list(self.results.keys())

    def get_tool_description(self, tool_name: str) -> str | None:
        return f"Mock tool: {tool_name}"


# =============================================================================
# Template Tests
# =============================================================================


class TestVoiceAssistantTemplate:
    """Tests for the voice assistant template."""

    def test_template_exists(self):
        """Template should be defined."""
        assert VOICE_ASSISTANT_TEMPLATE is not None
        assert VOICE_ASSISTANT_TEMPLATE.behavior_id == "voice_assistant"

    def test_template_has_actions(self):
        """Template should have actions defined."""
        assert len(VOICE_ASSISTANT_TEMPLATE.actions) > 0

        action_names = [a.name for a in VOICE_ASSISTANT_TEMPLATE.actions]
        assert "answer" in action_names
        assert "get_time" in action_names
        assert "get_weather" in action_names
        assert "home_assistant" in action_names
        assert "search_web" in action_names

    def test_template_has_triggers(self):
        """Template should have triggers defined."""
        assert len(VOICE_ASSISTANT_TEMPLATE.triggers) > 0

        trigger_names = [t.name for t in VOICE_ASSISTANT_TEMPLATE.triggers]
        assert "greeting" in trigger_names
        assert "time_query" in trigger_names

    def test_template_has_prompts(self):
        """Template should have prompts defined."""
        assert VOICE_ASSISTANT_TEMPLATE.prompts is not None
        assert VOICE_ASSISTANT_TEMPLATE.prompts.decision_prompt
        assert VOICE_ASSISTANT_TEMPLATE.prompts.synthesis_prompt

    def test_template_has_test_cases(self):
        """Template should have test cases defined."""
        assert len(VOICE_ASSISTANT_TEMPLATE.test_cases) > 0

    def test_template_is_active(self):
        """Template should be in active status."""
        assert VOICE_ASSISTANT_TEMPLATE.status == BehaviorStatus.ACTIVE
        assert VOICE_ASSISTANT_TEMPLATE.tier == BehaviorTier.CORE


class TestCreateVoiceAssistant:
    """Tests for the create_voice_assistant_behavior factory."""

    def test_create_default(self):
        """Create with default settings."""
        behavior = create_voice_assistant_behavior()
        assert behavior.behavior_id == "voice_assistant"
        assert len(behavior.actions) == len(VOICE_ASSISTANT_TEMPLATE.actions)

    def test_create_custom_id(self):
        """Create with custom ID."""
        behavior = create_voice_assistant_behavior(
            behavior_id="my_assistant",
            name="My Assistant",
        )
        assert behavior.behavior_id == "my_assistant"
        assert behavior.name == "My Assistant"

    def test_exclude_actions(self):
        """Create excluding specific actions."""
        behavior = create_voice_assistant_behavior(
            exclude_actions=["execute_command"],
        )

        action_names = [a.name for a in behavior.actions]
        assert "execute_command" not in action_names
        assert "answer" in action_names  # Other actions still present


class TestBehaviorRegistry:
    """Tests for behavior registry with voice assistant."""

    def test_register_template(self):
        """Should register the template."""
        registry = BehaviorRegistry()
        registry.register(VOICE_ASSISTANT_TEMPLATE)

        assert registry.get("voice_assistant") == VOICE_ASSISTANT_TEMPLATE

    def test_find_by_trigger(self):
        """Should find behavior by trigger."""
        registry = BehaviorRegistry()
        registry.register(VOICE_ASSISTANT_TEMPLATE)

        matches = registry.find_by_trigger("hello", {})
        assert len(matches) > 0
        assert matches[0].behavior_id == "voice_assistant"


# =============================================================================
# Decision Engine Tests
# =============================================================================


class TestDecisionEngine:
    """Tests for the decision engine with voice assistant behavior."""

    @pytest.fixture
    def mock_llm(self):
        """Create mock LLM with predefined responses."""
        return MockLLMProvider({
            "what time": """<response>
<action>get_time</action>
<reasoning>User wants current time</reasoning>
<model_tier>local</model_tier>
</response>""",
            "turn on": """<response>
<action>home_assistant</action>
<reasoning>Smart home control</reasoning>
<ha_domain>light</ha_domain>
<ha_service>turn_on</ha_service>
<ha_entity>bedroom</ha_entity>
<model_tier>local</model_tier>
</response>""",
            "hello": """<response>
<action>answer</action>
<reasoning>Greeting</reasoning>
<answer>Hello! How can I help you?</answer>
<model_tier>local</model_tier>
</response>""",
        })

    @pytest.fixture
    def decision_engine(self, mock_llm):
        """Create decision engine with mock LLM."""
        return DecisionEngine(llm=mock_llm)

    @pytest.mark.asyncio
    async def test_decide_time_query(self, decision_engine, mock_llm):
        """Should decide to use get_time for time queries."""
        context = DecisionContext(
            user_id="test_user",
            assistant_intro="You are a helpful assistant.",
        )

        result = await decision_engine.decide(
            behavior=VOICE_ASSISTANT_TEMPLATE,
            query="What time is it?",
            context=context,
        )

        assert result.action == "get_time"

    @pytest.mark.asyncio
    async def test_decide_greeting(self, decision_engine, mock_llm):
        """Should answer greetings directly."""
        context = DecisionContext(
            user_id="test_user",
            assistant_intro="You are a helpful assistant.",
        )

        result = await decision_engine.decide(
            behavior=VOICE_ASSISTANT_TEMPLATE,
            query="Hello!",
            context=context,
        )

        assert result.action == "answer"
        assert result.answer is not None

    @pytest.mark.asyncio
    async def test_decide_smart_home(self, decision_engine, mock_llm):
        """Should decide to use home_assistant for device control."""
        context = DecisionContext(
            user_id="test_user",
            assistant_intro="You are a helpful assistant.",
        )

        result = await decision_engine.decide(
            behavior=VOICE_ASSISTANT_TEMPLATE,
            query="Turn on the bedroom lights",
            context=context,
        )

        assert result.action == "home_assistant"
        assert result.args.get("domain") == "light"
        assert result.args.get("service") == "turn_on"


# =============================================================================
# Agent Integration Tests
# =============================================================================


class TestAgent:
    """Integration tests for the Agent with voice assistant behavior."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock providers."""
        llm = MockLLMProvider({
            "hello": """<response>
<action>answer</action>
<reasoning>Greeting</reasoning>
<answer>Hello! How can I help you today?</answer>
<model_tier>local</model_tier>
</response>""",
        })

        memory = MockMemoryProvider()

        tools = MockToolProvider({
            "get_time": {"success": True, "result": {"time": "3:45 PM"}},
            "get_weather": {"success": True, "result": {"weather": "72°F, sunny"}},
        })

        return llm, memory, tools

    @pytest.fixture
    def agent(self, mock_providers):
        """Create agent with mock providers."""
        llm, memory, tools = mock_providers

        return Agent(
            config=AgentConfig(
                agent_id="test_agent",
                name="Test Agent",
                personality_intro="You are a helpful test assistant.",
            ),
            behavior=VOICE_ASSISTANT_TEMPLATE,
            llm=llm,
            memory=memory,
            tools=tools,
        )

    @pytest.mark.asyncio
    async def test_process_greeting(self, agent):
        """Should process a greeting."""
        response = await agent.process(
            query="Hello!",
            user_id="test_user",
        )

        assert response.success
        assert "hello" in response.response.lower() or "help" in response.response.lower()

    @pytest.mark.asyncio
    async def test_process_with_debug(self, agent):
        """Should include debug info when requested."""
        response = await agent.process(
            query="Hello!",
            user_id="test_user",
            debug=True,
        )

        assert response.debug_info is not None
        assert "decision" in response.debug_info

    @pytest.mark.asyncio
    async def test_session_management(self, agent):
        """Should maintain session history."""
        # First message
        await agent.process(
            query="Hello!",
            user_id="test_user",
            session_id="session_1",
        )

        # Check session was created
        session = agent.get_session("session_1")
        assert session is not None
        assert len(session.conversation_history) == 1

        # Second message in same session
        await agent.process(
            query="Thanks!",
            user_id="test_user",
            session_id="session_1",
        )

        assert len(session.conversation_history) == 2

    @pytest.mark.asyncio
    async def test_clear_session(self, agent):
        """Should clear session history."""
        await agent.process(
            query="Hello!",
            user_id="test_user",
            session_id="session_1",
        )

        assert agent.clear_session("session_1") is True

        session = agent.get_session("session_1")
        assert session is not None
        assert len(session.conversation_history) == 0
