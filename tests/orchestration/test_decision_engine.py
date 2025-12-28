"""Tests for DecisionEngine integration (REQ-002-02).

Tests the enhanced DecisionEngine:
- Action validation against behavior actions
- Action alias resolution
- Confidence extraction from responses
- Fallback to "answer" for unknown actions
- Helper methods (is_final_answer, is_no_action)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.orchestration import (
    DecisionEngine,
    DecisionResult,
    DecisionContext,
    ACTION_ALIASES,
)
from draagon_ai.orchestration.protocols import LLMResponse
from draagon_ai.behaviors import Behavior, Action, BehaviorPrompts


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.chat = AsyncMock()
    return llm


@pytest.fixture
def simple_behavior():
    """Create a behavior with common actions for testing."""
    return Behavior(
        behavior_id="test",
        name="Test Behavior",
        description="For testing",
        actions=[
            Action(name="answer", description="Provide a direct answer"),
            Action(name="search_web", description="Search the web"),
            Action(name="get_time", description="Get current time"),
            Action(name="call_service", description="Call a home service"),
        ],
        prompts=BehaviorPrompts(
            decision_prompt="Decide action for: {question}",
            synthesis_prompt="Synthesize: {tool_results}",
        ),
    )


@pytest.fixture
def decision_context():
    """Create a basic decision context."""
    return DecisionContext(
        user_id="test_user",
        assistant_intro="I am a helpful assistant.",
        conversation_history="",
        gathered_context="",
    )


# =============================================================================
# DecisionResult Tests
# =============================================================================


class TestDecisionResult:
    """Tests for DecisionResult dataclass and helper methods."""

    def test_default_values(self):
        """Test DecisionResult has correct default values."""
        result = DecisionResult(action="answer")

        assert result.action == "answer"
        assert result.reasoning == ""
        assert result.args == {}
        assert result.answer is None
        assert result.model_tier == "local"
        assert result.confidence == 1.0
        assert result.additional_actions == []
        assert result.memory_update is None
        assert result.is_valid_action is True
        assert result.original_action is None
        assert result.validation_notes == ""
        assert result.raw_response == ""

    def test_is_final_answer_true(self):
        """Test is_final_answer returns True for answer action with answer."""
        result = DecisionResult(
            action="answer",
            answer="The answer is 42.",
        )

        assert result.is_final_answer() is True

    def test_is_final_answer_false_no_answer(self):
        """Test is_final_answer returns False when answer is None."""
        result = DecisionResult(action="answer", answer=None)

        assert result.is_final_answer() is False

    def test_is_final_answer_false_different_action(self):
        """Test is_final_answer returns False for non-answer action."""
        result = DecisionResult(
            action="search_web",
            answer="Some text",  # Has answer but wrong action
        )

        assert result.is_final_answer() is False

    def test_is_no_action_true_for_no_action(self):
        """Test is_no_action returns True for 'no_action'."""
        result = DecisionResult(action="no_action")
        assert result.is_no_action() is True

    def test_is_no_action_true_for_none(self):
        """Test is_no_action returns True for 'none'."""
        result = DecisionResult(action="none")
        assert result.is_no_action() is True

    def test_is_no_action_true_for_empty(self):
        """Test is_no_action returns True for empty string."""
        result = DecisionResult(action="")
        assert result.is_no_action() is True

    def test_is_no_action_false_for_valid_action(self):
        """Test is_no_action returns False for valid actions."""
        result = DecisionResult(action="search_web")
        assert result.is_no_action() is False

    def test_full_result_with_all_fields(self):
        """Test DecisionResult with all fields populated."""
        result = DecisionResult(
            action="search_web",
            reasoning="User wants to search",
            args={"query": "weather"},
            answer=None,
            model_tier="complex",
            confidence=0.85,
            additional_actions=["get_time"],
            memory_update={"action": "store", "content": "test"},
            is_valid_action=True,
            original_action="search",
            validation_notes="Resolved alias",
            raw_response="<response>...</response>",
        )

        assert result.action == "search_web"
        assert result.confidence == 0.85
        assert result.args["query"] == "weather"
        assert result.original_action == "search"
        assert result.is_final_answer() is False
        assert result.is_no_action() is False


# =============================================================================
# ACTION_ALIASES Tests
# =============================================================================


class TestActionAliases:
    """Tests for the ACTION_ALIASES dictionary."""

    def test_respond_maps_to_answer(self):
        """Test 'respond' alias maps to 'answer'."""
        assert ACTION_ALIASES["respond"] == "answer"

    def test_reply_maps_to_answer(self):
        """Test 'reply' alias maps to 'answer'."""
        assert ACTION_ALIASES["reply"] == "answer"

    def test_say_maps_to_answer(self):
        """Test 'say' alias maps to 'answer'."""
        assert ACTION_ALIASES["say"] == "answer"

    def test_search_maps_to_search_web(self):
        """Test 'search' alias maps to 'search_web'."""
        assert ACTION_ALIASES["search"] == "search_web"

    def test_web_search_maps_to_search_web(self):
        """Test 'web_search' alias maps to 'search_web'."""
        assert ACTION_ALIASES["web_search"] == "search_web"

    def test_lookup_maps_to_search_web(self):
        """Test 'lookup' alias maps to 'search_web'."""
        assert ACTION_ALIASES["lookup"] == "search_web"

    def test_no_action_maps_to_answer(self):
        """Test 'no_action' alias maps to 'answer'."""
        assert ACTION_ALIASES["no_action"] == "answer"

    def test_none_maps_to_answer(self):
        """Test 'none' alias maps to 'answer'."""
        assert ACTION_ALIASES["none"] == "answer"


# =============================================================================
# DecisionEngine Initialization Tests
# =============================================================================


class TestDecisionEngineInit:
    """Tests for DecisionEngine initialization."""

    def test_default_initialization(self, mock_llm):
        """Test default DecisionEngine initialization."""
        engine = DecisionEngine(llm=mock_llm)

        assert engine.llm is mock_llm
        assert engine.default_model_tier == "local"
        assert engine.validate_actions is True
        assert engine.fallback_to_answer is True
        assert engine.action_aliases == ACTION_ALIASES

    def test_custom_model_tier(self, mock_llm):
        """Test initialization with custom model tier."""
        engine = DecisionEngine(llm=mock_llm, default_model_tier="complex")

        assert engine.default_model_tier == "complex"

    def test_disable_validation(self, mock_llm):
        """Test initialization with validation disabled."""
        engine = DecisionEngine(llm=mock_llm, validate_actions=False)

        assert engine.validate_actions is False

    def test_disable_fallback(self, mock_llm):
        """Test initialization with fallback disabled."""
        engine = DecisionEngine(llm=mock_llm, fallback_to_answer=False)

        assert engine.fallback_to_answer is False

    def test_custom_aliases(self, mock_llm):
        """Test initialization with custom aliases."""
        custom_aliases = {"custom": "answer"}
        engine = DecisionEngine(llm=mock_llm, action_aliases=custom_aliases)

        # Should merge with defaults
        assert engine.action_aliases["custom"] == "answer"
        assert engine.action_aliases["respond"] == "answer"  # Default still there


# =============================================================================
# Action Validation Tests
# =============================================================================


class TestActionValidation:
    """Tests for action validation logic."""

    def test_valid_action_passes(self, mock_llm, simple_behavior):
        """Test that valid action passes validation."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(action="search_web")

        validated = engine._validate_action(result, simple_behavior)

        assert validated.is_valid_action is True
        assert validated.action == "search_web"
        assert validated.original_action is None

    def test_alias_resolved_to_canonical(self, mock_llm, simple_behavior):
        """Test that alias is resolved to canonical action name."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(action="respond")

        validated = engine._validate_action(result, simple_behavior)

        assert validated.action == "answer"
        assert validated.original_action == "respond"
        assert "Alias" in validated.validation_notes

    def test_search_alias_resolved(self, mock_llm, simple_behavior):
        """Test 'search' alias resolves to 'search_web'."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(action="search")

        validated = engine._validate_action(result, simple_behavior)

        assert validated.action == "search_web"
        assert validated.original_action == "search"

    def test_case_insensitive_validation(self, mock_llm, simple_behavior):
        """Test action validation is case insensitive."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(action="SEARCH_WEB")

        validated = engine._validate_action(result, simple_behavior)

        assert validated.is_valid_action is True

    def test_unknown_action_falls_back_to_answer(self, mock_llm, simple_behavior):
        """Test unknown action falls back to 'answer'."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(
            action="unknown_action",
            reasoning="Some reasoning here",
        )

        validated = engine._validate_action(result, simple_behavior)

        assert validated.action == "answer"
        assert validated.is_valid_action is False
        assert validated.original_action == "unknown_action"
        assert "fallback" in validated.validation_notes.lower()
        # Reasoning should become answer if answer was None
        assert validated.answer == "Some reasoning here"

    def test_unknown_action_no_fallback(self, mock_llm, simple_behavior):
        """Test unknown action without fallback enabled."""
        engine = DecisionEngine(llm=mock_llm, fallback_to_answer=False)
        result = DecisionResult(action="unknown_action")

        validated = engine._validate_action(result, simple_behavior)

        assert validated.action == "unknown_action"
        assert validated.is_valid_action is False
        assert validated.original_action == "unknown_action"

    def test_unknown_action_preserves_existing_answer(self, mock_llm, simple_behavior):
        """Test fallback doesn't overwrite existing answer."""
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(
            action="unknown_action",
            reasoning="Some reasoning",
            answer="Existing answer",
        )

        validated = engine._validate_action(result, simple_behavior)

        assert validated.action == "answer"
        assert validated.answer == "Existing answer"  # Not overwritten

    def test_no_answer_in_behavior_no_fallback(self, mock_llm):
        """Test fallback doesn't work if behavior has no 'answer' action."""
        behavior = Behavior(
            behavior_id="limited",
            name="Limited Behavior",
            description="No answer action",
            actions=[
                Action(name="search_web", description="Search"),
            ],
            prompts=BehaviorPrompts(
                decision_prompt="Decide",
                synthesis_prompt="Synthesize",
            ),
        )
        engine = DecisionEngine(llm=mock_llm)
        result = DecisionResult(action="unknown_action")

        validated = engine._validate_action(result, behavior)

        assert validated.is_valid_action is False
        assert "no 'answer' fallback" in validated.validation_notes.lower()


# =============================================================================
# get_valid_actions Tests
# =============================================================================


class TestGetValidActions:
    """Tests for get_valid_actions method."""

    def test_returns_all_action_names(self, mock_llm, simple_behavior):
        """Test returns all action names from behavior."""
        engine = DecisionEngine(llm=mock_llm)

        actions = engine.get_valid_actions(simple_behavior)

        assert "answer" in actions
        assert "search_web" in actions
        assert "get_time" in actions
        assert "call_service" in actions
        assert len(actions) == 4

    def test_empty_behavior_returns_empty_list(self, mock_llm):
        """Test returns empty list for behavior with no actions."""
        behavior = Behavior(
            behavior_id="empty",
            name="Empty",
            description="No actions",
            prompts=BehaviorPrompts(
                decision_prompt="Decide",
                synthesis_prompt="Synthesize",
            ),
        )
        engine = DecisionEngine(llm=mock_llm)

        actions = engine.get_valid_actions(behavior)

        assert actions == []


# =============================================================================
# XML Parsing Tests
# =============================================================================


class TestXMLParsing:
    """Tests for XML response parsing."""

    def test_parse_xml_with_confidence(self, mock_llm, simple_behavior):
        """Test parsing XML response with confidence element."""
        engine = DecisionEngine(llm=mock_llm)
        xml_response = """
        <response>
            <action>search_web</action>
            <reasoning>User wants to find information</reasoning>
            <confidence>0.85</confidence>
            <query>weather today</query>
        </response>
        """

        mock_response = LLMResponse(content=xml_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.action == "search_web"
        assert result.confidence == 0.85
        assert result.args.get("query") == "weather today"

    def test_parse_xml_confidence_clamped_to_1(self, mock_llm, simple_behavior):
        """Test confidence > 1.0 is clamped to 1.0."""
        engine = DecisionEngine(llm=mock_llm)
        xml_response = """
        <response>
            <action>answer</action>
            <confidence>1.5</confidence>
            <answer>Hello</answer>
        </response>
        """

        mock_response = LLMResponse(content=xml_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 1.0

    def test_parse_xml_confidence_clamped_to_0(self, mock_llm, simple_behavior):
        """Test confidence < 0.0 is clamped to 0.0."""
        engine = DecisionEngine(llm=mock_llm)
        xml_response = """
        <response>
            <action>answer</action>
            <confidence>-0.5</confidence>
            <answer>Hello</answer>
        </response>
        """

        mock_response = LLMResponse(content=xml_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 0.0

    def test_parse_xml_invalid_confidence_uses_default(self, mock_llm, simple_behavior):
        """Test invalid confidence uses default 1.0."""
        engine = DecisionEngine(llm=mock_llm)
        xml_response = """
        <response>
            <action>answer</action>
            <confidence>not_a_number</confidence>
            <answer>Hello</answer>
        </response>
        """

        mock_response = LLMResponse(content=xml_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 1.0

    def test_parse_xml_no_confidence_uses_default(self, mock_llm, simple_behavior):
        """Test missing confidence element uses default 1.0."""
        engine = DecisionEngine(llm=mock_llm)
        xml_response = """
        <response>
            <action>answer</action>
            <answer>Hello</answer>
        </response>
        """

        mock_response = LLMResponse(content=xml_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 1.0


# =============================================================================
# JSON Parsing Tests
# =============================================================================


class TestJSONParsing:
    """Tests for JSON response parsing."""

    def test_parse_json_with_confidence(self, mock_llm, simple_behavior):
        """Test parsing JSON response with confidence field."""
        engine = DecisionEngine(llm=mock_llm)
        json_response = '{"action": "search_web", "confidence": 0.75, "args": {"query": "news"}}'

        mock_response = LLMResponse(content=json_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.action == "search_web"
        assert result.confidence == 0.75

    def test_parse_json_confidence_clamped(self, mock_llm, simple_behavior):
        """Test JSON confidence is clamped to valid range."""
        engine = DecisionEngine(llm=mock_llm)
        json_response = '{"action": "answer", "confidence": 2.5}'

        mock_response = LLMResponse(content=json_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 1.0

    def test_parse_json_invalid_confidence(self, mock_llm, simple_behavior):
        """Test invalid JSON confidence uses default."""
        engine = DecisionEngine(llm=mock_llm)
        json_response = '{"action": "answer", "confidence": "high"}'

        mock_response = LLMResponse(content=json_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.confidence == 1.0

    def test_parse_json_with_additional_actions(self, mock_llm, simple_behavior):
        """Test parsing JSON with additional_actions field."""
        engine = DecisionEngine(llm=mock_llm)
        json_response = '{"action": "search_web", "additional_actions": ["get_time", "answer"]}'

        mock_response = LLMResponse(content=json_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.additional_actions == ["get_time", "answer"]

    def test_parse_json_with_memory_update(self, mock_llm, simple_behavior):
        """Test parsing JSON with memory_update field."""
        engine = DecisionEngine(llm=mock_llm)
        json_response = '{"action": "answer", "memory_update": {"action": "store", "content": "test"}}'

        mock_response = LLMResponse(content=json_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.memory_update == {"action": "store", "content": "test"}


# =============================================================================
# Text Parsing Tests
# =============================================================================


class TestTextParsing:
    """Tests for plain text response parsing (fallback)."""

    def test_text_fallback_lower_confidence(self, mock_llm, simple_behavior):
        """Test text fallback parser has lower confidence."""
        engine = DecisionEngine(llm=mock_llm)
        text_response = "I'll help you with the time."

        mock_response = LLMResponse(content=text_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        # Should detect "time" and match get_time action
        assert result.confidence < 1.0

    def test_text_fallback_detects_action_keyword(self, mock_llm, simple_behavior):
        """Test text fallback detects action keywords."""
        engine = DecisionEngine(llm=mock_llm)
        text_response = "Let me search_web for that information."

        mock_response = LLMResponse(content=text_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.action == "search_web"
        assert result.confidence == 0.5  # Matched action = 0.5

    def test_text_fallback_no_match_lowest_confidence(self, mock_llm, simple_behavior):
        """Test text fallback with no action match has lowest confidence."""
        engine = DecisionEngine(llm=mock_llm)
        text_response = "Hello there, how can I help you?"

        mock_response = LLMResponse(content=text_response)
        result = engine._parse_decision(mock_response, simple_behavior)

        assert result.action == "answer"  # Default
        assert result.confidence == 0.3  # No match = 0.3


# =============================================================================
# Full Decide Flow Tests
# =============================================================================


class TestDecideFlow:
    """Tests for the full decide() method."""

    @pytest.mark.asyncio
    async def test_decide_with_validation(self, mock_llm, simple_behavior, decision_context):
        """Test full decide flow with action validation."""
        engine = DecisionEngine(llm=mock_llm)

        # Mock LLM to return XML with alias
        mock_llm.chat.return_value = LLMResponse(
            content="""
            <response>
                <action>respond</action>
                <answer>Hello there!</answer>
            </response>
            """
        )

        result = await engine.decide(simple_behavior, "Hello", decision_context)

        # Should validate and resolve alias
        assert result.action == "answer"  # Resolved from "respond"
        assert result.original_action == "respond"
        assert result.answer == "Hello there!"

    @pytest.mark.asyncio
    async def test_decide_stores_raw_response(self, mock_llm, simple_behavior, decision_context):
        """Test decide stores raw response in result."""
        engine = DecisionEngine(llm=mock_llm)

        raw_response = """
        <response>
            <action>answer</action>
            <answer>Test</answer>
        </response>
        """
        mock_llm.chat.return_value = LLMResponse(content=raw_response)

        result = await engine.decide(simple_behavior, "Hello", decision_context)

        # Raw response is stored as-is from LLM
        assert "<response>" in result.raw_response
        assert "<action>answer</action>" in result.raw_response

    @pytest.mark.asyncio
    async def test_decide_without_validation(self, mock_llm, simple_behavior, decision_context):
        """Test decide without validation passes through unknown actions."""
        engine = DecisionEngine(llm=mock_llm, validate_actions=False)

        mock_llm.chat.return_value = LLMResponse(
            content='{"action": "unknown_custom_action"}'
        )

        result = await engine.decide(simple_behavior, "Hello", decision_context)

        # Without validation, unknown action passes through
        assert result.action == "unknown_custom_action"
        assert result.is_valid_action is True  # Not validated


# =============================================================================
# Integration Tests
# =============================================================================


class TestDecisionEngineIntegration:
    """Integration tests for DecisionEngine with behavior."""

    @pytest.mark.asyncio
    async def test_all_aliases_resolve_correctly(self, mock_llm, simple_behavior):
        """Test all common aliases resolve to valid actions."""
        engine = DecisionEngine(llm=mock_llm)

        test_cases = [
            ("respond", "answer"),
            ("reply", "answer"),
            ("say", "answer"),
            ("search", "search_web"),
            ("web_search", "search_web"),
            ("lookup", "search_web"),
            ("find", "search_web"),
            ("no_action", "answer"),
            ("none", "answer"),
        ]

        for alias, expected in test_cases:
            result = DecisionResult(action=alias)
            validated = engine._validate_action(result, simple_behavior)
            assert validated.action == expected, f"Alias '{alias}' should resolve to '{expected}'"

    def test_behavior_with_custom_actions(self, mock_llm):
        """Test validation works with custom behavior actions."""
        custom_behavior = Behavior(
            behavior_id="custom",
            name="Custom",
            description="Custom actions",
            actions=[
                Action(name="cast_spell", description="Cast a magic spell"),
                Action(name="roll_dice", description="Roll dice"),
                Action(name="answer", description="Respond"),
            ],
            prompts=BehaviorPrompts(
                decision_prompt="Decide",
                synthesis_prompt="Synthesize",
            ),
        )
        engine = DecisionEngine(llm=mock_llm)

        # Valid custom action
        result = engine._validate_action(
            DecisionResult(action="cast_spell"),
            custom_behavior,
        )
        assert result.is_valid_action is True

        # Unknown action falls back
        result = engine._validate_action(
            DecisionResult(action="search_web"),
            custom_behavior,
        )
        assert result.action == "answer"  # Fallback
        assert result.is_valid_action is False
