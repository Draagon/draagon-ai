"""Tests for the PromptBuilder."""

import pytest
from draagon_ai.prompts.builder import (
    PromptBuilder,
    ActionDef,
    CapabilityDef,
    create_default_builder,
    CORE_ACTIONS,
    CALENDAR_CAPABILITY,
    HOME_ASSISTANT_CAPABILITY,
)
from draagon_ai.prompts.domains.templates import (
    DECISION_TEMPLATE,
    FAST_ROUTE_TEMPLATE,
)


class TestActionDef:
    """Tests for ActionDef dataclass."""

    def test_create_basic(self):
        """Test creating a basic action definition."""
        action = ActionDef("test_action", "A test action")
        assert action.name == "test_action"
        assert action.description == "A test action"
        assert action.source == "core"

    def test_create_with_source(self):
        """Test creating action with custom source."""
        action = ActionDef("test", "desc", source="extension:foo")
        assert action.source == "extension:foo"


class TestCapabilityDef:
    """Tests for CapabilityDef dataclass."""

    def test_create_basic(self):
        """Test creating a basic capability."""
        cap = CapabilityDef(name="test_cap")
        assert cap.name == "test_cap"
        assert cap.actions == []
        assert cap.fast_route_actions == []

    def test_create_with_actions(self):
        """Test creating capability with actions."""
        actions = [ActionDef("action1", "desc1"), ActionDef("action2", "desc2")]
        cap = CapabilityDef(name="test_cap", actions=actions)
        assert len(cap.actions) == 2
        assert cap.actions[0].name == "action1"


class TestPromptBuilder:
    """Tests for PromptBuilder class."""

    def test_add_core_action(self):
        """Test adding a single core action."""
        builder = PromptBuilder()
        builder.add_core_action("test", "A test action")

        actions = builder.get_all_actions()
        assert len(actions) == 1
        assert actions[0].name == "test"
        assert actions[0].source == "core"

    def test_add_core_actions(self):
        """Test adding multiple core actions."""
        builder = PromptBuilder()
        actions = [
            ActionDef("action1", "desc1"),
            ActionDef("action2", "desc2"),
        ]
        builder.add_core_actions(actions)

        all_actions = builder.get_all_actions()
        assert len(all_actions) == 2

    def test_add_capability(self):
        """Test adding a capability."""
        builder = PromptBuilder()
        cap = CapabilityDef(
            name="test_cap",
            actions=[ActionDef("cap_action", "A capability action")]
        )
        builder.add_capability("test_cap", cap)

        actions = builder.get_all_actions()
        assert len(actions) == 1
        assert actions[0].name == "cap_action"

    def test_add_capability_actions(self):
        """Test adding capability actions via tuple format."""
        builder = PromptBuilder()
        builder.add_capability_actions("calendar", [
            ("list_events", "List calendar events"),
            ("create_event", "Create an event"),
        ])

        actions = builder.get_all_actions()
        assert len(actions) == 2
        assert actions[0].source == "capability:calendar"

    def test_add_extension_actions(self):
        """Test adding extension actions."""
        builder = PromptBuilder()
        builder.add_extension_actions("storytelling", [
            ("tell_story", "Tell a story"),
            ("add_character", "Add a character"),
        ])

        actions = builder.get_all_actions()
        assert len(actions) == 2
        assert actions[0].source == "extension:storytelling"

    def test_add_mcp_actions(self):
        """Test adding MCP server actions."""
        builder = PromptBuilder()
        builder.add_mcp_actions("fetch", [
            ("fetch_url", "Fetch content from URL"),
        ])

        actions = builder.get_all_actions()
        assert len(actions) == 1
        assert actions[0].source == "mcp:fetch"

    def test_format_actions_section(self):
        """Test formatting actions as prompt section."""
        builder = PromptBuilder()
        builder.add_core_action("answer", "Respond directly")
        builder.add_capability_actions("calendar", [
            ("list_events", "List events"),
        ])

        section = builder.format_actions_section()
        assert "AVAILABLE ACTIONS:" in section
        assert "answer: Respond directly" in section
        assert "CALENDAR ACTIONS:" in section
        assert "list_events: List events" in section

    def test_build_template(self):
        """Test building a prompt from template."""
        builder = PromptBuilder()
        builder.add_core_action("answer", "Respond directly")

        template = "Test prompt with {available_actions} placeholder"
        result = builder.build(template)

        assert "AVAILABLE ACTIONS:" in result
        assert "answer: Respond directly" in result
        assert "{available_actions}" not in result

    def test_build_with_kwargs(self):
        """Test building with additional kwargs."""
        builder = PromptBuilder()

        template = "Hello {name}, your actions: {available_actions}"
        result = builder.build(template, name="TestBot")

        assert "Hello TestBot" in result
        assert "AVAILABLE ACTIONS:" in result

    def test_chaining(self):
        """Test that builder methods return self for chaining."""
        builder = (
            PromptBuilder()
            .add_core_action("action1", "desc1")
            .add_capability_actions("cal", [("action2", "desc2")])
            .add_custom_section("custom", "Custom content")
        )

        assert len(builder.get_all_actions()) == 2


class TestCreateDefaultBuilder:
    """Tests for create_default_builder function."""

    def test_default_all_capabilities(self):
        """Test default builder has all capabilities."""
        builder = create_default_builder()
        actions = builder.get_all_actions()

        # Should have core actions plus all capability actions
        action_names = [a.name for a in actions]
        assert "answer" in action_names
        assert "get_time" in action_names
        assert "get_calendar_events" in action_names
        assert "home_assistant" in action_names

    def test_specific_capabilities(self):
        """Test builder with specific capabilities."""
        builder = create_default_builder(capabilities=["calendar"])
        actions = builder.get_all_actions()

        action_names = [a.name for a in actions]
        assert "get_calendar_events" in action_names
        assert "home_assistant" not in action_names

    def test_empty_capabilities(self):
        """Test builder with no capabilities."""
        builder = create_default_builder(capabilities=[])
        actions = builder.get_all_actions()

        # Should still have core actions
        action_names = [a.name for a in actions]
        assert "answer" in action_names
        assert "get_time" in action_names
        # But no capability actions
        assert "get_calendar_events" not in action_names


class TestPrebuiltCapabilities:
    """Tests for pre-built capability definitions."""

    def test_core_actions_exist(self):
        """Test CORE_ACTIONS has expected actions."""
        action_names = [a.name for a in CORE_ACTIONS]
        assert "answer" in action_names
        assert "clarify" in action_names
        assert "get_time" in action_names
        assert "get_weather" in action_names

    def test_calendar_capability(self):
        """Test CALENDAR_CAPABILITY is properly defined."""
        assert CALENDAR_CAPABILITY.name == "calendar"
        action_names = [a.name for a in CALENDAR_CAPABILITY.actions]
        assert "get_calendar_events" in action_names
        assert "create_calendar_event" in action_names

    def test_home_assistant_capability(self):
        """Test HOME_ASSISTANT_CAPABILITY is properly defined."""
        assert HOME_ASSISTANT_CAPABILITY.name == "home_assistant"
        action_names = [a.name for a in HOME_ASSISTANT_CAPABILITY.actions]
        assert "home_assistant" in action_names
        # Should have fast route action
        assert len(HOME_ASSISTANT_CAPABILITY.fast_route_actions) > 0


class TestBuildDecisionPrompt:
    """Tests for building the decision prompt from template."""

    def test_build_decision_prompt(self):
        """Test building a full decision prompt."""
        builder = create_default_builder(capabilities=["calendar", "home_assistant"])

        prompt = builder.build(
            DECISION_TEMPLATE,
            assistant_name="TestBot",
            assistant_intro="A test assistant",
            examples="Example 1\nExample 2",
        )

        # Should have substituted assistant name
        assert "TestBot" in prompt
        assert "A test assistant" in prompt

        # Should have available actions
        assert "AVAILABLE ACTIONS:" in prompt
        assert "get_calendar_events" in prompt
        assert "home_assistant" in prompt

        # Should have examples
        assert "Example 1" in prompt

    def test_build_minimal_prompt(self):
        """Test building with minimal capabilities."""
        builder = create_default_builder(capabilities=[])

        prompt = builder.build(
            DECISION_TEMPLATE,
            assistant_name="MinimalBot",
            assistant_intro="Minimal",
            examples="",
        )

        assert "MinimalBot" in prompt
        assert "answer" in prompt  # Core actions still present
        assert "get_calendar_events" not in prompt
