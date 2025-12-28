"""End-to-end tests for multi-step reasoning (REQ-002-10).

These tests verify complex multi-step reasoning workflows:
1. Search-then-add: Search for information, then take action based on results
2. Gather-then-analyze: Collect multiple pieces of info, then synthesize
3. Error recovery: Handle failures gracefully in multi-step chains
4. Thought trace quality: Ensure reasoning is meaningful and traceable

Unlike integration tests (REQ-002-09), these tests focus on realistic
multi-step scenarios with properly sequenced mock responses.
"""

import pytest
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock

from draagon_ai.orchestration import (
    Agent,
    AgentConfig,
    AgentLoop,
    AgentLoopConfig,
    AgentContext,
    AgentResponse,
    DecisionEngine,
    ActionExecutor,
    ToolRegistry,
    Tool,
    ToolParameter,
    LoopMode,
    StepType,
)
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts


pytestmark = [
    pytest.mark.e2e,
]


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def tool_registry():
    """Create a tool registry with realistic tools for E2E testing."""
    registry = ToolRegistry()

    # Calendar search tool
    async def search_calendar_handler(args: dict, context: dict | None = None) -> list[dict]:
        days = args.get("days", 7)
        # Return realistic calendar data
        return [
            {
                "id": "evt_001",
                "title": "Taylor Swift Concert",
                "date": "2025-12-28",
                "time": "7:00 PM",
                "location": "Wells Fargo Center",
            },
            {
                "id": "evt_002",
                "title": "Team Meeting",
                "date": "2025-12-29",
                "time": "10:00 AM",
                "location": "Office",
            },
            {
                "id": "evt_003",
                "title": "Dentist Appointment",
                "date": "2025-12-30",
                "time": "2:00 PM",
                "location": "Dr. Smith's Office",
            },
        ][:days]

    registry.register(Tool(
        name="search_calendar",
        description="Search calendar for events",
        handler=search_calendar_handler,
        parameters=[
            ToolParameter(
                name="days",
                type="integer",
                description="Number of days to search",
                required=False,
                default=7,
            ),
        ],
    ))

    # Add calendar event tool
    async def add_calendar_event_handler(args: dict, context: dict | None = None) -> dict:
        return {
            "success": True,
            "event_id": "evt_new_001",
            "title": args.get("title", "New Event"),
            "date": args.get("date", "2025-12-28"),
            "time": args.get("time"),
            "location": args.get("location"),
        }

    registry.register(Tool(
        name="add_calendar_event",
        description="Add a new event to the calendar",
        handler=add_calendar_event_handler,
        parameters=[
            ToolParameter(name="title", type="string", description="Event title", required=True),
            ToolParameter(name="date", type="string", description="Event date", required=True),
            ToolParameter(name="time", type="string", description="Event time", required=False),
            ToolParameter(name="location", type="string", description="Event location", required=False),
        ],
    ))

    # Weather tool
    async def get_weather_handler(args: dict, context: dict | None = None) -> dict:
        return {
            "condition": "Rainy",
            "temperature": "45°F",
            "precipitation": "80%",
            "forecast": "Heavy rain expected in the afternoon",
        }

    registry.register(Tool(
        name="get_weather",
        description="Get current weather",
        handler=get_weather_handler,
    ))

    # Web search tool
    async def web_search_handler(args: dict, context: dict | None = None) -> list[dict]:
        query = args.get("query", "")
        return [
            {
                "title": f"Result 1 for {query}",
                "url": "https://example.com/1",
                "snippet": f"Information about {query}...",
            },
            {
                "title": f"Result 2 for {query}",
                "url": "https://example.com/2",
                "snippet": f"More details on {query}...",
            },
        ]

    registry.register(Tool(
        name="web_search",
        description="Search the web for information",
        handler=web_search_handler,
        parameters=[
            ToolParameter(name="query", type="string", description="Search query", required=True),
        ],
    ))

    # Memory search tool
    async def search_memory_handler(args: dict, context: dict | None = None) -> list[dict]:
        return [
            {"content": "User prefers morning meetings", "type": "preference"},
            {"content": "User works from home on Fridays", "type": "fact"},
        ]

    registry.register(Tool(
        name="search_memory",
        description="Search user memories",
        handler=search_memory_handler,
        parameters=[
            ToolParameter(name="query", type="string", description="Memory query", required=True),
        ],
    ))

    # Error-prone tool for testing recovery
    call_counter = {"count": 0}

    async def flaky_service_handler(args: dict, context: dict | None = None) -> dict:
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            raise ConnectionError("Service temporarily unavailable")
        return {"status": "success", "data": "Retrieved on retry"}

    registry.register(Tool(
        name="flaky_service",
        description="A service that may fail temporarily",
        handler=flaky_service_handler,
    ))

    return registry


@pytest.fixture
def behavior():
    """Create a behavior for E2E testing."""
    return Behavior(
        behavior_id="e2e_test",
        name="E2E Test Behavior",
        description="Behavior for end-to-end multi-step testing",
        actions=[
            Action(name="answer", description="Provide a direct answer"),
            Action(name="search_calendar", description="Search calendar events"),
            Action(name="add_calendar_event", description="Add a calendar event"),
            Action(name="get_weather", description="Get weather information"),
            Action(name="web_search", description="Search the web"),
            Action(name="search_memory", description="Search user memories"),
            Action(name="flaky_service", description="Access flaky service"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide what action to take for: {question}",
            synthesis_prompt="Synthesize response from: {tool_results}",
        ),
    )


@pytest.fixture
def agent_context():
    """Create an agent context for E2E testing."""
    return AgentContext(
        user_id="e2e_test_user",
        session_id="e2e_test_session",
        conversation_id="e2e_test_conv",
        debug=True,
    )


# =============================================================================
# Search-Then-Add Tests
# =============================================================================


class TestSearchThenAdd:
    """Tests for search-then-add multi-step patterns (REQ-002-10 Scenario 1)."""

    @pytest.mark.anyio
    async def test_search_calendar_then_add_event(self, tool_registry, behavior, agent_context):
        """Test searching calendar and then adding an event based on results."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Step 1: Search calendar first
                response.content = '{"action": "search_calendar", "args": {"days": 7}, "reasoning": "Need to check what events already exist this week before adding a new one"}'
            elif call_count[0] == 2:
                # Step 2: Add event based on search results
                response.content = '{"action": "add_calendar_event", "args": {"title": "Follow-up Meeting", "date": "2025-12-31", "time": "3:00 PM"}, "reasoning": "Found existing meetings, scheduling follow-up for a day with no conflicts"}'
            else:
                # Step 3: Final answer
                response.content = """<response>
                <action>answer</action>
                <answer>I've added a Follow-up Meeting on December 31st at 3:00 PM, avoiding conflicts with your existing events.</answer>
                <reasoning>Successfully searched calendar and added new event</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Check my calendar and add a follow-up meeting when I'm free",
            behavior=behavior,
            context=agent_context,
        )

        # Verify multi-step execution
        assert response.success is True
        assert len(response.tool_results) == 2

        # First tool was search
        assert response.tool_results[0].action_name == "search_calendar"
        assert isinstance(response.tool_results[0].result, list)

        # Second tool was add
        assert response.tool_results[1].action_name == "add_calendar_event"
        assert response.tool_results[1].result["success"] is True

        # Verify ReAct steps were recorded
        assert response.loop_mode == LoopMode.REACT
        assert response.iterations_used >= 3

    @pytest.mark.anyio
    async def test_web_search_then_add_to_calendar(self, tool_registry, behavior, agent_context):
        """Test searching web for event info, then adding to calendar."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Step 1: Search web for concert info
                response.content = '{"action": "web_search", "args": {"query": "Coldplay concert Philadelphia 2025"}, "reasoning": "Need to find concert details before adding to calendar"}'
            elif call_count[0] == 2:
                # Step 2: Add event with found info
                response.content = '{"action": "add_calendar_event", "args": {"title": "Coldplay Concert", "date": "2025-07-15", "location": "Citizens Bank Park"}, "reasoning": "Found concert details, adding to calendar"}'
            else:
                # Step 3: Final answer
                response.content = """<response>
                <action>answer</action>
                <answer>I found the Coldplay concert on July 15th at Citizens Bank Park and added it to your calendar!</answer>
                <reasoning>Successfully found and added event</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Find the Coldplay concert in Philadelphia and add it to my calendar",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2
        assert response.tool_results[0].action_name == "web_search"
        assert response.tool_results[1].action_name == "add_calendar_event"


# =============================================================================
# Gather-Then-Analyze Tests
# =============================================================================


class TestGatherThenAnalyze:
    """Tests for gather-then-analyze multi-step patterns (REQ-002-10 Scenario 2)."""

    @pytest.mark.anyio
    async def test_weather_and_calendar_synthesis(self, tool_registry, behavior, agent_context):
        """Test gathering weather and calendar, then synthesizing advice."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Step 1: Get weather
                response.content = '{"action": "get_weather", "args": {}, "reasoning": "Need to check weather conditions first"}'
            elif call_count[0] == 2:
                # Step 2: Get calendar
                response.content = '{"action": "search_calendar", "args": {"days": 1}, "reasoning": "Need to check today\'s schedule"}'
            else:
                # Step 3: Synthesize and answer
                response.content = """<response>
                <action>answer</action>
                <answer>It's going to rain today (80% chance). You have a dentist appointment at 2 PM - definitely bring an umbrella!</answer>
                <reasoning>Combined weather and calendar info to give practical advice</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="What's the weather and what's on my calendar? Should I bring an umbrella?",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2

        # Both tools were called
        tool_names = [r.action_name for r in response.tool_results]
        assert "get_weather" in tool_names
        assert "search_calendar" in tool_names

        # Final answer synthesizes both
        assert response.response is not None
        assert "umbrella" in response.response.lower() or "rain" in response.response.lower()

    @pytest.mark.anyio
    async def test_memory_and_preferences_synthesis(self, tool_registry, behavior, agent_context):
        """Test gathering memories and preferences to personalize response."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Step 1: Search memories about preferences
                response.content = '{"action": "search_memory", "args": {"query": "meeting preferences"}, "reasoning": "Check user preferences before scheduling"}'
            elif call_count[0] == 2:
                # Step 2: Check calendar
                response.content = '{"action": "search_calendar", "args": {"days": 5}, "reasoning": "Find available slots matching preferences"}'
            else:
                # Step 3: Personalized recommendation
                response.content = """<response>
                <action>answer</action>
                <answer>Based on your preference for morning meetings, I found Tuesday at 10 AM is available. Since you work from home on Fridays, I avoided that day.</answer>
                <reasoning>Used memories to personalize scheduling recommendation</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="When should I schedule a meeting this week?",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2

        # Memories and calendar both checked
        tool_names = [r.action_name for r in response.tool_results]
        assert "search_memory" in tool_names
        assert "search_calendar" in tool_names


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Tests for error recovery in multi-step reasoning (REQ-002-10 Scenario 3)."""

    @pytest.mark.anyio
    async def test_retry_after_service_failure(self, tool_registry, behavior, agent_context):
        """Test that the agent retries after a service failure."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # First attempt - will fail
                response.content = '{"action": "flaky_service", "args": {}, "reasoning": "Accessing the service"}'
            elif call_count[0] == 2:
                # Retry after seeing error
                response.content = '{"action": "flaky_service", "args": {}, "reasoning": "Service failed, retrying..."}'
            else:
                # Success after retry
                response.content = """<response>
                <action>answer</action>
                <answer>Successfully retrieved data after retry!</answer>
                <reasoning>Service recovered on second attempt</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Get data from the flaky service",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2

        # First call failed
        assert response.tool_results[0].success is False
        assert "unavailable" in str(response.tool_results[0].error).lower()

        # Second call succeeded
        assert response.tool_results[1].success is True

    @pytest.mark.anyio
    async def test_fallback_to_alternative(self, tool_registry, behavior, agent_context):
        """Test falling back to alternative action after failure."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Try flaky service
                response.content = '{"action": "flaky_service", "args": {}, "reasoning": "Trying primary service"}'
            elif call_count[0] == 2:
                # Fall back to web search after error
                response.content = '{"action": "web_search", "args": {"query": "alternative data source"}, "reasoning": "Primary service failed, using web search as fallback"}'
            else:
                # Answer using fallback data
                response.content = """<response>
                <action>answer</action>
                <answer>The primary service was unavailable, but I found the information via web search.</answer>
                <reasoning>Successfully recovered using alternative</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Get data, use web search if needed",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True

        # First call failed, second succeeded with different tool
        assert response.tool_results[0].action_name == "flaky_service"
        assert response.tool_results[0].success is False
        assert response.tool_results[1].action_name == "web_search"
        assert response.tool_results[1].success is True


# =============================================================================
# Thought Trace Quality Tests
# =============================================================================


class TestThoughtTraceQuality:
    """Tests for thought trace quality and completeness (REQ-002-10 Scenario 4)."""

    @pytest.mark.anyio
    async def test_thought_traces_captured(self, tool_registry, behavior, agent_context):
        """Test that thought traces are properly captured in ReAct mode."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = '{"action": "get_weather", "args": {}, "reasoning": "First, I need to check the current weather conditions to provide accurate advice"}'
            else:
                response.content = """<response>
                <action>answer</action>
                <answer>It's rainy with 80% precipitation. Bring an umbrella!</answer>
                <reasoning>Weather data shows rain is likely, so umbrella recommendation is appropriate</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="Should I bring an umbrella?",
            behavior=behavior,
            context=agent_context,
        )

        # Verify ReAct steps are present
        assert len(response.react_steps) >= 2

        # Should have ACTION and OBSERVATION steps
        step_types = [step.type for step in response.react_steps]
        assert StepType.ACTION in step_types
        assert StepType.OBSERVATION in step_types

    @pytest.mark.anyio
    async def test_debug_info_contains_thoughts(self, tool_registry, behavior, agent_context):
        """Test that debug info includes thought traces."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = '{"action": "search_calendar", "args": {"days": 3}, "reasoning": "Checking upcoming calendar events"}'
            else:
                response.content = """<response>
                <action>answer</action>
                <answer>You have 3 events in the next 3 days.</answer>
                <reasoning>Calendar search complete</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="What's on my calendar?",
            behavior=behavior,
            context=agent_context,
        )

        # Debug info should be populated with thought trace in ReAct mode
        assert "thought_trace" in response.debug_info
        # Thought trace should contain steps with actions
        thought_trace = response.debug_info["thought_trace"]
        assert isinstance(thought_trace, list)
        assert len(thought_trace) > 0
        # Should have a step with the search_calendar action
        action_names = [step.get("action_name") for step in thought_trace if step.get("action_name")]
        assert "search_calendar" in action_names

    @pytest.mark.anyio
    async def test_thought_trace_formatting(self, tool_registry, behavior, agent_context):
        """Test that thought traces can be formatted for display."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = '{"action": "get_weather", "args": {}, "reasoning": "Checking weather first"}'
            elif call_count[0] == 2:
                response.content = '{"action": "search_calendar", "args": {"days": 1}, "reasoning": "Now checking today\'s events"}'
            else:
                response.content = """<response>
                <action>answer</action>
                <answer>Rainy weather, you have a meeting at 2 PM.</answer>
                <reasoning>Combined info to answer</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="What's the weather and what do I have today?",
            behavior=behavior,
            context=agent_context,
        )

        # Get formatted thought trace
        thought_trace = response.get_thought_trace()

        # Should be a list of step dictionaries
        assert isinstance(thought_trace, list)
        assert len(thought_trace) > 0

        # Each step should have required fields
        for step in thought_trace:
            assert "step" in step
            assert "type" in step
            assert "content" in step
            assert "timestamp" in step

        # Should have both weather and calendar actions
        action_names = [step.get("action_name") for step in thought_trace if step.get("action_name")]
        assert "get_weather" in action_names
        assert "search_calendar" in action_names


# =============================================================================
# Complex Multi-Step Scenarios
# =============================================================================


class TestComplexScenarios:
    """Tests for complex real-world multi-step scenarios."""

    @pytest.mark.anyio
    async def test_four_step_planning_workflow(self, tool_registry, behavior, agent_context):
        """Test a 4-step planning workflow: search web, check calendar, check weather, add event."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                response.content = '{"action": "web_search", "args": {"query": "outdoor events this weekend"}, "reasoning": "Finding outdoor activities"}'
            elif call_count[0] == 2:
                response.content = '{"action": "get_weather", "args": {}, "reasoning": "Check if weather is suitable for outdoor events"}'
            elif call_count[0] == 3:
                response.content = '{"action": "search_calendar", "args": {"days": 2}, "reasoning": "Check for scheduling conflicts"}'
            elif call_count[0] == 4:
                response.content = '{"action": "add_calendar_event", "args": {"title": "Outdoor Festival", "date": "2025-12-28", "time": "2:00 PM", "location": "City Park"}, "reasoning": "Weather looks good and no conflicts, adding event"}'
            else:
                response.content = """<response>
                <action>answer</action>
                <answer>I found an Outdoor Festival, checked that the weather is okay despite some rain chance, confirmed you're free, and added it to your calendar for December 28th at 2 PM!</answer>
                <reasoning>Completed full planning workflow</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=10),
        )

        response = await loop.process(
            query="Find an outdoor event this weekend, check if weather and my schedule allow, and add it to my calendar",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 4
        assert response.iterations_used == 5

        # Verify all 4 tools were called in sequence
        expected_tools = ["web_search", "get_weather", "search_calendar", "add_calendar_event"]
        actual_tools = [r.action_name for r in response.tool_results]
        assert actual_tools == expected_tools

    @pytest.mark.anyio
    async def test_conditional_branching(self, tool_registry, behavior, agent_context):
        """Test that agent can make conditional decisions based on intermediate results."""
        mock_llm = MagicMock()
        call_count = [0]

        async def mock_chat(*args, **kwargs):
            call_count[0] += 1
            response = MagicMock()
            if call_count[0] == 1:
                # Check weather first
                response.content = '{"action": "get_weather", "args": {}, "reasoning": "Need to check weather conditions"}'
            elif call_count[0] == 2:
                # Based on rainy weather, recommend indoor activity
                response.content = '{"action": "web_search", "args": {"query": "indoor activities near me"}, "reasoning": "Weather is rainy (80% precipitation), searching for indoor alternatives"}'
            else:
                response.content = """<response>
                <action>answer</action>
                <answer>It's going to rain, so I found some indoor activities for you instead of outdoor ones.</answer>
                <reasoning>Adapted recommendation based on weather</reasoning>
                </response>"""
            return response

        mock_llm.chat = mock_chat

        decision_engine = DecisionEngine(llm=mock_llm)
        action_executor = ActionExecutor(tool_registry=tool_registry)

        loop = AgentLoop(
            llm=mock_llm,
            decision_engine=decision_engine,
            action_executor=action_executor,
            config=AgentLoopConfig(mode=LoopMode.REACT, max_iterations=5),
        )

        response = await loop.process(
            query="What activities should I do today?",
            behavior=behavior,
            context=agent_context,
        )

        assert response.success is True
        assert len(response.tool_results) == 2

        # First checked weather, then searched for indoor based on rain
        assert response.tool_results[0].action_name == "get_weather"
        assert response.tool_results[1].action_name == "web_search"
        # The web_search result contains "indoor" in the title/snippet (from our mock)
        # The title is "Result 1 for indoor activities near me"
        assert "indoor" in response.tool_results[1].result[0]["title"].lower()
