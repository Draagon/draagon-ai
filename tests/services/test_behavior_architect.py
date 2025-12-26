"""Tests for BehaviorArchitectService.

Tests the meta-service that creates, tests, and evolves behaviors.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import asdict

from draagon_ai.services.behavior_architect import (
    BehaviorArchitectService,
    BehaviorDesign,
    MutationPrompt,
    RESEARCH_PROMPT,
    DESIGN_PROMPT,
    INITIAL_MUTATION_PROMPTS,
)
from draagon_ai.behaviors.types import (
    Action,
    ActionParameter,
    Behavior,
    BehaviorPrompts,
    BehaviorStatus,
    BehaviorTestCase,
    BehaviorTier,
    DomainResearchResult,
    EvolutionConfig,
    Trigger,
)
from draagon_ai.behaviors.registry import BehaviorRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="Mock response")
    return llm


@pytest.fixture
def mock_web_search():
    """Create a mock web search provider."""
    search = AsyncMock()
    search.search = AsyncMock(return_value=[
        {"title": "Best Practices", "snippet": "Some tips", "url": "http://example.com"},
    ])
    return search


@pytest.fixture
def mock_registry():
    """Create a mock behavior registry."""
    registry = MagicMock(spec=BehaviorRegistry)
    registry.get_all.return_value = []
    return registry


@pytest.fixture
def architect(mock_llm, mock_web_search, mock_registry):
    """Create a BehaviorArchitectService with mocks."""
    return BehaviorArchitectService(
        llm=mock_llm,
        web_search=mock_web_search,
        registry=mock_registry,
    )


# =============================================================================
# Research Phase Tests
# =============================================================================


class TestResearchPhase:
    """Tests for the research phase."""

    @pytest.mark.asyncio
    async def test_research_domain_calls_llm(self, architect, mock_llm):
        """Should call LLM with research prompt."""
        mock_llm.generate.return_value = """
        <research>
          <core_tasks>
            <task>Set timers</task>
            <task>Track multiple timers</task>
          </core_tasks>
          <suggested_actions>
            <action name="set_timer" description="Set a new timer">
              <parameter name="duration" type="int" required="true">Duration in seconds</parameter>
            </action>
          </suggested_actions>
          <triggers>
            <trigger>set a timer for</trigger>
          </triggers>
          <constraints>
            <constraint>Don't delete timers without confirmation</constraint>
          </constraints>
          <domain_knowledge>Timers are countdown mechanisms.</domain_knowledge>
          <sources>
            <source>http://example.com</source>
          </sources>
        </research>
        """

        result = await architect.research_domain(
            "A behavior for managing kitchen timers"
        )

        assert mock_llm.generate.called
        assert isinstance(result, DomainResearchResult)
        assert "kitchen timers" in result.domain.lower() or "timer" in result.domain.lower()
        assert len(result.core_tasks) >= 1

    @pytest.mark.asyncio
    async def test_research_uses_web_search(self, architect, mock_web_search):
        """Should search web when enabled."""
        await architect.research_domain(
            "A behavior for aquarium management",
            search_web=True,
        )

        assert mock_web_search.search.called

    @pytest.mark.asyncio
    async def test_research_skips_web_when_disabled(self, architect, mock_web_search):
        """Should not search web when disabled."""
        await architect.research_domain(
            "A behavior for todo lists",
            search_web=False,
        )

        assert not mock_web_search.search.called

    @pytest.mark.asyncio
    async def test_research_fallback_on_bad_xml(self, architect, mock_llm):
        """Should handle malformed XML gracefully."""
        mock_llm.generate.return_value = """
        Here are the core tasks:
        - Task 1
        - Task 2
        """

        result = await architect.research_domain("Some behavior")

        # Should still return a result with fallback parsing
        assert isinstance(result, DomainResearchResult)
        assert result.domain == "Some behavior"


# =============================================================================
# Design Phase Tests
# =============================================================================


class TestDesignPhase:
    """Tests for the design phase."""

    @pytest.mark.asyncio
    async def test_design_behavior_from_research(self, architect, mock_llm):
        """Should create design from research results."""
        mock_llm.generate.return_value = """
        <design>
          <behavior_id>kitchen_timer</behavior_id>
          <name>Kitchen Timer</name>
          <description>Manages cooking timers</description>
          <actions>
            <action name="set_timer" requires_confirmation="false">
              <description>Set a new timer</description>
              <parameter name="minutes" type="int" required="true">Duration</parameter>
              <trigger>set a timer</trigger>
            </action>
          </actions>
          <triggers>
            <trigger name="main" priority="70">
              <semantic>timer related requests</semantic>
              <keyword>timer</keyword>
            </trigger>
          </triggers>
          <constraints>
            <style>Be concise</style>
          </constraints>
          <domain_context>Kitchen timing</domain_context>
        </design>
        """

        research = DomainResearchResult(
            domain="kitchen timers",
            core_tasks=["Set timers", "Track timers"],
            suggested_actions=[{"name": "set_timer", "description": "Set a timer"}],
        )

        design = await architect.design_behavior(research)

        assert isinstance(design, BehaviorDesign)
        assert design.behavior_id == "kitchen_timer"
        assert len(design.actions) >= 1
        assert design.actions[0].name == "set_timer"

    @pytest.mark.asyncio
    async def test_design_creates_actions(self, architect, mock_llm):
        """Should create action objects with parameters."""
        mock_llm.generate.return_value = """
        <design>
          <behavior_id>test_behavior</behavior_id>
          <name>Test</name>
          <description>Test behavior</description>
          <actions>
            <action name="action_one">
              <description>First action</description>
              <parameter name="param1" type="string" required="true">A parameter</parameter>
              <parameter name="param2" type="int" required="false">Optional param</parameter>
            </action>
          </actions>
          <triggers/>
          <constraints/>
        </design>
        """

        research = DomainResearchResult(domain="test")
        design = await architect.design_behavior(research)

        action = design.actions[0]
        assert action.name == "action_one"
        assert "param1" in action.parameters
        assert action.parameters["param1"].required is True
        assert action.parameters["param2"].required is False


# =============================================================================
# Build Phase Tests
# =============================================================================


class TestBuildPhase:
    """Tests for the build phase."""

    @pytest.mark.asyncio
    async def test_build_creates_behavior(self, architect, mock_llm):
        """Should create complete Behavior from design."""
        # First call for prompts
        mock_llm.generate.side_effect = [
            """
            <prompts>
              <decision_prompt>You are a timer assistant. Choose the right action.</decision_prompt>
              <synthesis_prompt>Format the response clearly.</synthesis_prompt>
            </prompts>
            """,
            """
            <test_cases>
              <test id="test_1" name="Basic Timer" priority="high">
                <description>Test basic timer setting</description>
                <user_query>Set a 5 minute timer</user_query>
                <expected_actions>
                  <action>set_timer</action>
                </expected_actions>
              </test>
            </test_cases>
            """,
        ]

        design = BehaviorDesign(
            behavior_id="timer",
            name="Timer",
            description="Timer behavior",
            actions=[
                Action(name="set_timer", description="Set a timer"),
            ],
        )

        behavior, test_cases = await architect.build_behavior(design)

        assert isinstance(behavior, Behavior)
        assert behavior.behavior_id == "timer"
        assert behavior.prompts is not None
        assert "timer" in behavior.prompts.decision_prompt.lower() or "action" in behavior.prompts.decision_prompt.lower()
        assert len(test_cases) >= 1

    @pytest.mark.asyncio
    async def test_build_sets_correct_tier(self, architect, mock_llm):
        """Generated behaviors should be GENERATED tier."""
        mock_llm.generate.side_effect = [
            "<prompts><decision_prompt>Test</decision_prompt><synthesis_prompt>Test</synthesis_prompt></prompts>",
            "<test_cases></test_cases>",
        ]

        design = BehaviorDesign(
            behavior_id="test",
            name="Test",
            description="Test",
        )

        behavior, _ = await architect.build_behavior(design)

        assert behavior.tier == BehaviorTier.GENERATED
        assert behavior.status == BehaviorStatus.TESTING


# =============================================================================
# Test Phase Tests
# =============================================================================


class TestTestPhase:
    """Tests for the test phase."""

    @pytest.mark.asyncio
    async def test_run_tests_returns_results(self, architect, mock_llm):
        """Should run tests and return results."""
        mock_llm.generate.return_value = """
        <decision>
          <action name="set_timer">
            <parameter name="minutes">5</parameter>
          </action>
        </decision>
        """

        behavior = Behavior(
            behavior_id="timer",
            name="Timer",
            description="Test",
            actions=[Action(name="set_timer", description="Set timer")],
            prompts=BehaviorPrompts(
                decision_prompt="Choose action: {{actions}}",
                synthesis_prompt="Format response",
            ),
        )

        test_cases = [
            BehaviorTestCase(
                test_id="test_1",
                name="Basic Test",
                user_query="Set a 5 minute timer",
                expected_actions=["set_timer"],
            ),
        ]

        results = await architect._run_tests(behavior, test_cases)

        assert results.total_tests == 1
        assert results.passed >= 0  # May or may not pass based on LLM response

    @pytest.mark.asyncio
    async def test_extract_action_from_xml(self, architect):
        """Should extract action name from XML response."""
        response = '<action name="do_something"><param>value</param></action>'
        action = architect._extract_action_from_response(response)
        assert action == "do_something"

    @pytest.mark.asyncio
    async def test_extract_action_from_text(self, architect):
        """Should extract action name from text response."""
        response = "I will use set_timer action with parameter 5 minutes"
        action = architect._extract_action_from_response(response)
        assert action == "set_timer"


# =============================================================================
# Iteration Phase Tests
# =============================================================================


class TestIterationPhase:
    """Tests for the iteration phase."""

    @pytest.mark.asyncio
    async def test_analyze_failures_identifies_patterns(self, architect, mock_llm):
        """Should identify failure patterns."""
        mock_llm.generate.return_value = """
        <analysis>
          <patterns>
            <pattern type="wrong_action" count="2">Selected wrong action</pattern>
          </patterns>
          <root_causes>
            <cause id="cause_1">Decision prompt unclear</cause>
          </root_causes>
          <fixes>
            <fix cause_id="cause_1" target="decision_prompt">Add more specific guidance</fix>
          </fixes>
        </analysis>
        """

        from draagon_ai.behaviors.types import TestResults, TestOutcome

        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            test_cases=[
                BehaviorTestCase(test_id="t1", name="Test 1", expected_actions=["action_a"]),
            ],
        )

        results = TestResults(
            total_tests=2,
            passed=0,
            failed=2,
            test_outcomes={
                "t1": TestOutcome(test_id="t1", passed=False, failure_reason="Wrong action"),
                "t2": TestOutcome(test_id="t2", passed=False, failure_reason="Wrong action"),
            },
        )

        analysis = await architect._analyze_failures(behavior, results)

        assert len(analysis.patterns) >= 1
        assert len(analysis.root_causes) >= 1
        assert len(analysis.suggested_fixes) >= 1


# =============================================================================
# Evolution Phase Tests
# =============================================================================


class TestEvolutionPhase:
    """Tests for the evolution phase."""

    def test_initial_mutation_prompts(self):
        """Should have initial mutation prompts."""
        assert len(INITIAL_MUTATION_PROMPTS) >= 3

    def test_mutation_prompt_structure(self):
        """Mutation prompts should have required fields."""
        for prompt in INITIAL_MUTATION_PROMPTS:
            assert prompt.prompt_id
            assert prompt.content
            assert isinstance(prompt.fitness, float)

    @pytest.mark.asyncio
    async def test_mutate_behavior_changes_prompt(self, architect, mock_llm):
        """Should mutate behavior prompts."""
        mock_llm.generate.return_value = "This is the mutated prompt with more detail."

        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            prompts=BehaviorPrompts(
                decision_prompt="Original prompt",
                synthesis_prompt="Synthesis",
            ),
        )

        mutation = MutationPrompt(
            prompt_id="test",
            content="Add more detail",
        )

        mutated = await architect._mutate_behavior(behavior, mutation)

        assert mutated.prompts.decision_prompt != behavior.prompts.decision_prompt
        assert mutated.parent_behavior_id == behavior.behavior_id

    def test_compute_prompt_similarity(self, architect):
        """Should compute similarity between prompts."""
        prompt1 = "You are a helpful assistant that helps with tasks"
        prompt2 = "You are a helpful assistant that manages timers"
        prompt3 = "Completely different words here about something else"

        sim1_2 = architect._compute_prompt_similarity(prompt1, prompt2)
        sim1_3 = architect._compute_prompt_similarity(prompt1, prompt3)

        assert sim1_2 > sim1_3  # More similar prompts should have higher score
        assert 0 <= sim1_2 <= 1
        assert 0 <= sim1_3 <= 1

    def test_tournament_selection(self, architect):
        """Should select higher fitness behaviors."""
        from draagon_ai.behaviors.types import BehaviorMetrics

        behaviors = []
        for i in range(10):
            b = Behavior(
                behavior_id=f"b_{i}",
                name=f"Behavior {i}",
                description="Test",
                metrics=BehaviorMetrics(fitness_score=i / 10),
            )
            behaviors.append(b)

        selected = architect._tournament_select(behaviors, tournament_size=3, count=3)

        assert len(selected) == 3
        # Selected should generally have higher fitness
        avg_selected = sum(b.metrics.fitness_score for b in selected) / len(selected)
        avg_all = sum(b.metrics.fitness_score for b in behaviors) / len(behaviors)
        # Not guaranteed, but likely
        assert avg_selected >= avg_all * 0.5


# =============================================================================
# Registration Phase Tests
# =============================================================================


class TestRegistrationPhase:
    """Tests for the registration phase."""

    @pytest.mark.asyncio
    async def test_register_behavior(self, architect, mock_registry):
        """Should register behavior with correct settings."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
        )

        result = await architect.register_behavior(behavior)

        assert result == "test"
        mock_registry.register.assert_called_once()
        mock_registry.save_behavior.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_sets_staging_status(self, architect, mock_registry):
        """Registered behaviors should be in STAGING by default."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
        )

        await architect.register_behavior(behavior)

        # Check the behavior passed to register
        call_args = mock_registry.register.call_args[0][0]
        assert call_args.status == BehaviorStatus.STAGING
        assert call_args.tier == BehaviorTier.GENERATED


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for behavior validation."""

    def test_validate_empty_behavior(self, architect):
        """Should catch missing required fields."""
        behavior = Behavior(
            behavior_id="",
            name="",
            description="",
        )

        issues = architect.validate_behavior(behavior)

        error_messages = [i.message for i in issues if i.severity == "error"]
        assert any("ID" in m for m in error_messages)
        assert any("name" in m for m in error_messages)

    def test_validate_no_prompts(self, architect):
        """Should warn about missing prompts."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            prompts=None,
        )

        issues = architect.validate_behavior(behavior)

        assert any(i.severity == "error" and "prompts" in i.message.lower() for i in issues)

    def test_validate_no_actions(self, architect):
        """Should warn about missing actions."""
        behavior = Behavior(
            behavior_id="test",
            name="Test",
            description="Test",
            actions=[],
        )

        issues = architect.validate_behavior(behavior)

        assert any(i.severity == "warning" and "actions" in i.message.lower() for i in issues)

    def test_validate_complete_behavior(self, architect):
        """Complete behavior should have minimal issues."""
        behavior = Behavior(
            behavior_id="complete_behavior",
            name="Complete Behavior",
            description="A fully specified behavior",
            actions=[
                Action(name="action_one", description="Does something"),
            ],
            prompts=BehaviorPrompts(
                decision_prompt="Decide what to do",
                synthesis_prompt="Format the response",
            ),
            test_cases=[
                BehaviorTestCase(test_id="t1", name="Test 1"),
                BehaviorTestCase(test_id="t2", name="Test 2"),
                BehaviorTestCase(test_id="t3", name="Test 3"),
                BehaviorTestCase(test_id="t4", name="Test 4"),
                BehaviorTestCase(test_id="t5", name="Test 5"),
            ],
        )

        issues = architect.validate_behavior(behavior)

        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestFullWorkflow:
    """Integration tests for the full workflow."""

    @pytest.mark.asyncio
    async def test_create_simple_behavior(self, architect, mock_llm, mock_web_search):
        """Should create a behavior from description."""
        # Set up mock responses for each phase
        mock_llm.generate.side_effect = [
            # Research
            """
            <research>
              <core_tasks><task>Count items</task></core_tasks>
              <suggested_actions>
                <action name="count" description="Count something"/>
              </suggested_actions>
              <triggers><trigger>count</trigger></triggers>
              <constraints/>
              <domain_knowledge>Counting basics</domain_knowledge>
              <sources/>
            </research>
            """,
            # Design
            """
            <design>
              <behavior_id>counter</behavior_id>
              <name>Counter</name>
              <description>Counts things</description>
              <actions>
                <action name="count"><description>Count items</description></action>
              </actions>
              <triggers><trigger name="main" priority="50"><semantic>count</semantic></trigger></triggers>
              <constraints/>
            </design>
            """,
            # Prompts
            """
            <prompts>
              <decision_prompt>Count things when asked</decision_prompt>
              <synthesis_prompt>Report the count</synthesis_prompt>
            </prompts>
            """,
            # Test cases
            """
            <test_cases>
              <test id="t1" name="Count Test">
                <user_query>Count to 5</user_query>
                <expected_actions><action>count</action></expected_actions>
              </test>
            </test_cases>
            """,
            # Test run response
            '<decision><action name="count"/></decision>',
        ]

        behavior = await architect.create_behavior(
            "A simple counter behavior",
            evolve=False,  # Skip evolution for this test
        )

        assert behavior is not None
        assert behavior.behavior_id == "counter"
        assert len(behavior.actions) >= 1


# =============================================================================
# Template Tests
# =============================================================================


class TestBehaviorArchitectTemplate:
    """Tests for the Behavior Architect template."""

    def test_template_exists(self):
        """Template should be importable."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
        assert BEHAVIOR_ARCHITECT_TEMPLATE is not None

    def test_template_is_core_tier(self):
        """Architect should be CORE tier."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
        assert BEHAVIOR_ARCHITECT_TEMPLATE.tier == BehaviorTier.CORE

    def test_template_is_active(self):
        """Architect should be ACTIVE status."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
        assert BEHAVIOR_ARCHITECT_TEMPLATE.status == BehaviorStatus.ACTIVE

    def test_template_has_actions(self):
        """Architect should have all phase actions."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE

        action_names = [a.name for a in BEHAVIOR_ARCHITECT_TEMPLATE.actions]

        # Check key actions exist
        assert "research_domain" in action_names
        assert "draft_behavior_structure" in action_names
        assert "add_action" in action_names
        assert "draft_decision_prompt" in action_names
        assert "generate_test_cases" in action_names
        assert "run_tests" in action_names
        assert "register_behavior" in action_names
        assert "evolve_behavior" in action_names

    def test_template_has_prompts(self):
        """Architect should have decision and synthesis prompts."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE

        assert BEHAVIOR_ARCHITECT_TEMPLATE.prompts is not None
        assert BEHAVIOR_ARCHITECT_TEMPLATE.prompts.decision_prompt
        assert BEHAVIOR_ARCHITECT_TEMPLATE.prompts.synthesis_prompt

    def test_template_has_test_cases(self):
        """Architect should have test cases."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE

        assert len(BEHAVIOR_ARCHITECT_TEMPLATE.test_cases) >= 5

    def test_template_not_evolvable(self):
        """Architect itself should not be auto-evolved."""
        from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE

        assert BEHAVIOR_ARCHITECT_TEMPLATE.is_evolvable is False

    def test_create_behavior_architect(self):
        """Should create fresh architect instance."""
        from draagon_ai.behaviors.templates import create_behavior_architect

        architect = create_behavior_architect()

        assert architect.behavior_id == "behavior_architect"
        assert architect.tier == BehaviorTier.CORE
