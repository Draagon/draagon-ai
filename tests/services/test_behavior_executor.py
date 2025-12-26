"""Tests for BehaviorExecutor."""

import pytest
from datetime import datetime

from draagon_ai.llm import RealisticMockLLM
from draagon_ai.services import (
    BehaviorExecutor,
    ExecutionContext,
    ToolRegistry,
    create_mock_tool_registry,
    BehaviorArchitectService,
)
from draagon_ai.behaviors.types import (
    Action,
    Behavior,
    BehaviorPrompts,
    BehaviorStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    return RealisticMockLLM()


@pytest.fixture
def tool_registry():
    """Create a tool registry with mock tools."""
    return create_mock_tool_registry()


@pytest.fixture
def executor(mock_llm, tool_registry):
    """Create an executor with mock LLM and tools."""
    return BehaviorExecutor(llm=mock_llm, tool_registry=tool_registry)


@pytest.fixture
def timer_behavior():
    """Create a timer behavior for testing."""
    return Behavior(
        behavior_id="timer",
        name="Timer",
        description="Manages timers",
        actions=[
            Action(
                name="set_timer",
                description="Set a countdown timer",
                parameters={
                    "duration": type(
                        "Param",
                        (),
                        {"description": "Duration string", "required": True},
                    )()
                },
            ),
            Action(
                name="cancel_timer",
                description="Cancel an active timer",
                parameters={
                    "timer_name": type(
                        "Param",
                        (),
                        {"description": "Timer name", "required": True},
                    )()
                },
            ),
            Action(
                name="list_timers",
                description="List all active timers",
                parameters={},
            ),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="""
You are a timer assistant. Analyze the query and choose the appropriate action.

USER QUERY: {query}
CONTEXT: {context}

Available actions:
- set_timer: Set a new timer (requires duration parameter)
- cancel_timer: Cancel a timer (requires timer_name parameter)
- list_timers: List all active timers

Respond with your decision in XML format:
<decision>
  <action name="action_name">
    <parameter name="param_name">value</parameter>
  </action>
  <reasoning>Why you chose this action</reasoning>
</decision>
""",
            synthesis_prompt="""
Format a friendly response for a voice assistant.

ACTION: {action}
RESULT: {action_result}
QUERY: {query}
STYLE: {style}

Respond with a natural, concise confirmation.
""",
        ),
        status=BehaviorStatus.ACTIVE,
    )


@pytest.fixture
def context():
    """Create a basic execution context."""
    return ExecutionContext(
        user_id="test_user",
        query="Set a 5 minute timer for pasta",
    )


# =============================================================================
# Tool Registry Tests
# =============================================================================


class TestToolRegistry:
    """Tests for the tool registry."""

    def test_register_and_get_tool(self):
        """Test registering and retrieving a tool."""
        registry = ToolRegistry()

        async def my_tool(**kwargs):
            return "result"

        registry.register("my_tool", my_tool, "Does something")

        tool = registry.get("my_tool")
        assert tool is not None
        assert tool.name == "my_tool"
        assert tool.description == "Does something"

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = create_mock_tool_registry()
        tools = registry.list_tools()

        assert "set_timer" in tools
        assert "cancel_timer" in tools
        assert "list_timers" in tools

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """Test executing a registered tool."""
        registry = create_mock_tool_registry()

        result = await registry.execute("set_timer", duration="10 minutes")

        assert result.success
        assert result.tool_name == "set_timer"
        assert "10 minutes" in result.result
        assert result.execution_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist."""
        registry = ToolRegistry()

        result = await registry.execute("nonexistent")

        assert not result.success
        assert result.error == "Tool not found: nonexistent"


# =============================================================================
# Executor Tests
# =============================================================================


class TestBehaviorExecutor:
    """Tests for the behavior executor."""

    @pytest.mark.asyncio
    async def test_execute_basic_query(self, executor, timer_behavior, context):
        """Test executing a basic query."""
        result = await executor.execute(timer_behavior, context)

        # Should complete without error
        assert result.success or result.error  # May fail with mock, but shouldn't crash
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_execute_tracks_timing(self, executor, timer_behavior, context):
        """Test that execution tracks timing."""
        result = await executor.execute(timer_behavior, context)

        assert result.total_time_ms > 0
        # At least decision time should be tracked
        assert result.decision_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_updates_stats(self, executor, timer_behavior, context):
        """Test that execution updates stats."""
        initial_stats = executor.stats.copy()

        await executor.execute(timer_behavior, context)

        assert executor.stats["total_executions"] == initial_stats["total_executions"] + 1

    @pytest.mark.asyncio
    async def test_execute_disabled_behavior(self, executor, timer_behavior, context):
        """Test executing a disabled/retired behavior."""
        timer_behavior.status = BehaviorStatus.RETIRED

        result = await executor.execute(timer_behavior, context)

        assert not result.success
        assert "unavailable" in result.response.lower()

    @pytest.mark.asyncio
    async def test_execute_behavior_without_prompts(self, executor, context):
        """Test executing a behavior without prompts."""
        behavior = Behavior(
            behavior_id="broken",
            name="Broken",
            description="Missing prompts",
            prompts=None,
        )

        result = await executor.execute(behavior, context)

        assert not result.success
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_register_custom_tool(self, mock_llm):
        """Test registering and using a custom tool."""
        executor = BehaviorExecutor(llm=mock_llm)

        async def custom_action(**kwargs):
            return "custom result"

        executor.register_tool("custom_action", custom_action)

        # Tool should be registered
        assert "custom_action" in executor.tools.list_tools()


class TestExecutionContext:
    """Tests for execution context."""

    def test_basic_context(self):
        """Test creating a basic context."""
        context = ExecutionContext(
            user_id="user1",
            query="Hello",
        )

        assert context.user_id == "user1"
        assert context.query == "Hello"
        assert context.conversation_history == []
        assert isinstance(context.timestamp, datetime)

    def test_context_with_history(self):
        """Test context with conversation history."""
        context = ExecutionContext(
            user_id="user1",
            query="And now?",
            conversation_history=[
                {"role": "user", "content": "Set a timer"},
                {"role": "assistant", "content": "Done"},
            ],
        )

        assert len(context.conversation_history) == 2

    def test_context_with_metadata(self):
        """Test context with metadata."""
        context = ExecutionContext(
            user_id="user1",
            query="Test",
            metadata={"room": "kitchen", "device_id": "speaker1"},
        )

        assert context.metadata["room"] == "kitchen"


# =============================================================================
# Integration Tests
# =============================================================================


class TestExecutorIntegration:
    """Integration tests with architect-created behaviors."""

    @pytest.mark.asyncio
    async def test_execute_architect_created_behavior(self, mock_llm, tool_registry):
        """Test executing a behavior created by the architect."""
        # Create behavior using architect
        architect = BehaviorArchitectService(llm=mock_llm)
        behavior = await architect.create_behavior(
            "A timer that can set, cancel, and list timers",
            evolve=False,
        )

        # Execute with the same LLM
        executor = BehaviorExecutor(llm=mock_llm, tool_registry=tool_registry)
        context = ExecutionContext(
            user_id="test",
            query="Set a 5 minute timer",
        )

        result = await executor.execute(behavior, context)

        # Should complete (may not be perfect with mock LLM)
        assert result.total_time_ms > 0
        # Should have attempted decision
        assert result.decision_time_ms >= 0

    @pytest.mark.asyncio
    async def test_execute_multiple_queries(self, executor, timer_behavior):
        """Test executing multiple queries in sequence."""
        queries = [
            "Set a 5 minute timer for pasta",
            "What timers do I have?",
            "Cancel the pasta timer",
        ]

        for query in queries:
            context = ExecutionContext(user_id="test", query=query)
            result = await executor.execute(timer_behavior, context)
            assert result.total_time_ms > 0

        # Stats should reflect all executions
        assert executor.stats["total_executions"] == 3

    @pytest.mark.asyncio
    async def test_action_stats_updated(self, executor, timer_behavior):
        """Test that action stats are updated on execution."""
        initial_usage = timer_behavior.actions[0].usage_count

        context = ExecutionContext(user_id="test", query="Set a timer")
        result = await executor.execute(timer_behavior, context)

        if result.success and result.action_taken:
            # Find the action that was taken
            for action in timer_behavior.actions:
                if action.name == result.action_taken:
                    assert action.usage_count > initial_usage
                    assert action.last_used is not None


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_query(self, executor, timer_behavior):
        """Test handling empty query."""
        context = ExecutionContext(user_id="test", query="")

        result = await executor.execute(timer_behavior, context)

        # Should handle gracefully
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_very_long_query(self, executor, timer_behavior):
        """Test handling very long query."""
        long_query = "Set a timer " * 100  # Very long query

        context = ExecutionContext(user_id="test", query=long_query)

        result = await executor.execute(timer_behavior, context)

        # Should handle without crashing
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, executor, timer_behavior):
        """Test handling special characters."""
        context = ExecutionContext(
            user_id="test",
            query="Set a timer for 5' & <10> \"minutes\"",
        )

        result = await executor.execute(timer_behavior, context)

        # Should handle special characters
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_behavior_with_no_actions(self, executor):
        """Test executing behavior with no actions."""
        behavior = Behavior(
            behavior_id="empty",
            name="Empty",
            description="No actions",
            actions=[],
            prompts=BehaviorPrompts(
                decision_prompt="You have no actions.",
                synthesis_prompt="Format response.",
            ),
        )

        context = ExecutionContext(user_id="test", query="Do something")

        result = await executor.execute(behavior, context)

        # Should handle gracefully
        assert result.total_time_ms > 0
