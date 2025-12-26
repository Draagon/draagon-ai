"""Behavior Architect Template.

The meta-behavior that creates, tests, and evolves other behaviors.
This is the "god-level agentic AI architect" - a CORE tier behavior
that requires maximum trust as it can create new behaviors.

Usage:
    from draagon_ai.behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
    from draagon_ai.services import BehaviorArchitectService

    # Create the service
    architect_service = BehaviorArchitectService(
        llm=my_llm,
        web_search=my_search,
        registry=my_registry,
    )

    # Create a new behavior
    behavior = await architect_service.create_behavior(
        "A behavior for managing kitchen timers"
    )
"""

from dataclasses import field
from datetime import datetime

from ..types import (
    Action,
    ActionExample,
    ActionParameter,
    Behavior,
    BehaviorConstraints,
    BehaviorPrompts,
    BehaviorStatus,
    BehaviorTestCase,
    BehaviorTier,
    Trigger,
)


# =============================================================================
# Prompts
# =============================================================================

BEHAVIOR_ARCHITECT_DECISION_PROMPT = '''You are the Behavior Architect, a meta-agent that creates AI behaviors.

## Your Role
You design, build, test, and evolve AI behaviors from natural language descriptions.
You have the highest trust level (CORE tier) and can create new behaviors.

## Available Actions
{actions}

## Current State
Phase: {phase}
Iteration: {iteration}
Design: {design_summary}

## Context
{context}

## User Request
{query}

## Decision Guidelines

### Research Phase
- Use `research_domain` to understand what the behavior should do
- Search web for best practices if the domain is unfamiliar
- Look at existing behaviors for patterns to follow

### Design Phase
- Use `draft_behavior_structure` to create the basic structure
- Use `add_action` for each capability the behavior needs
- Use `add_trigger` to define when the behavior activates
- Actions should be intuitive and well-documented

### Build Phase
- Use `draft_decision_prompt` to create the brain of the behavior
- Use `draft_synthesis_prompt` to shape the voice
- The decision prompt is critical - be thorough

### Test Phase
- Use `generate_test_cases` to create comprehensive tests
- Use `run_tests` to validate the behavior
- Aim for 80%+ pass rate

### Iterate Phase
- Use `analyze_failures` to understand what went wrong
- Use `apply_fix` to improve prompts and actions
- Iterate until tests pass or max iterations reached

### Register Phase
- Use `register_behavior` to add to the registry
- Generated behaviors start in STAGING status
- They need monitoring before going ACTIVE

## Response Format
Respond with your decision in XML:

<decision>
  <action name="action_name">
    <parameter name="param1">value1</parameter>
    <parameter name="param2">value2</parameter>
  </action>
  <reasoning>Why this action is appropriate</reasoning>
</decision>

Choose the most appropriate action for the current phase and state.
'''

BEHAVIOR_ARCHITECT_SYNTHESIS_PROMPT = '''You are formatting the Behavior Architect's response.

## Action Result
{action_result}

## Current State
Phase: {phase}
Progress: {progress}

## Style Guidelines
- Be clear and technical but accessible
- Explain what was done and what comes next
- If there are issues, explain them clearly
- Celebrate successes (behaviors created, tests passing)

## Response Guidelines

### For Research Results
Summarize the key findings:
- Core tasks identified
- Suggested actions
- Domain knowledge gathered

### For Design Progress
Report on structure:
- Actions defined
- Triggers configured
- Constraints set

### For Build Results
Describe what was created:
- Prompts generated
- Test cases created

### For Test Results
Report on quality:
- Pass rate
- Key failures (if any)
- Next steps

### For Registration
Confirm success:
- Behavior ID
- Initial status
- What to monitor

Format your response naturally, as if explaining to a developer.
'''


# =============================================================================
# Actions
# =============================================================================

ARCHITECT_ACTIONS = [
    # Research Phase
    Action(
        name="research_domain",
        description="""Research a domain to understand what a behavior should do.

        Uses web search and existing behavior patterns to build comprehensive
        domain knowledge. This is typically the first action when creating
        a new behavior.""",
        parameters={
            "description": ActionParameter(
                name="description",
                description="User's description of desired behavior",
                type="string",
                required=True,
            ),
            "search_web": ActionParameter(
                name="search_web",
                description="Whether to search web for best practices",
                type="bool",
                required=False,
                default=True,
            ),
            "search_existing": ActionParameter(
                name="search_existing",
                description="Whether to search existing behaviors for patterns",
                type="bool",
                required=False,
                default=True,
            ),
        },
        triggers=["need to understand", "research", "learn about domain"],
        examples=[
            ActionExample(
                user_query="Create a behavior for managing kitchen timers",
                action_call={
                    "name": "research_domain",
                    "args": {
                        "description": "managing kitchen timers",
                        "search_web": True,
                    },
                },
                expected_outcome="Research results with timer management best practices",
            ),
        ],
        handler="architect_service.research_domain",
    ),

    # Design Phase
    Action(
        name="draft_behavior_structure",
        description="""Create the initial structure for a new behavior.

        Defines the behavior's ID, name, description, and sets up the
        foundation for actions and triggers.""",
        parameters={
            "behavior_id": ActionParameter(
                name="behavior_id",
                description="Snake_case identifier for the behavior",
                type="string",
                required=True,
                validation_pattern=r"^[a-z][a-z0-9_]*$",
            ),
            "name": ActionParameter(
                name="name",
                description="Human-readable name",
                type="string",
                required=True,
            ),
            "description": ActionParameter(
                name="description",
                description="What this behavior enables",
                type="string",
                required=True,
            ),
            "domain_context": ActionParameter(
                name="domain_context",
                description="Background knowledge for this domain",
                type="string",
                required=False,
            ),
        },
        triggers=["create structure", "start design", "define behavior"],
        handler="architect_service.draft_behavior_structure",
    ),
    Action(
        name="add_action",
        description="""Add an action to the behavior being designed.

        Each action represents something the behavior can do. Actions
        should be intuitive and well-documented.""",
        parameters={
            "name": ActionParameter(
                name="name",
                description="Snake_case action name",
                type="string",
                required=True,
            ),
            "description": ActionParameter(
                name="description",
                description="What this action does",
                type="string",
                required=True,
            ),
            "parameters": ActionParameter(
                name="parameters",
                description="Action parameters as dict",
                type="dict",
                required=False,
            ),
            "triggers": ActionParameter(
                name="triggers",
                description="Phrases that suggest this action",
                type="list",
                required=False,
            ),
            "requires_confirmation": ActionParameter(
                name="requires_confirmation",
                description="Whether user confirmation is needed",
                type="bool",
                required=False,
                default=False,
            ),
        },
        triggers=["add action", "define capability", "add function"],
        handler="architect_service.add_action",
    ),
    Action(
        name="add_trigger",
        description="""Add a trigger pattern for behavior activation.

        Triggers determine when a behavior should activate. They can be
        semantic (LLM evaluates) or keyword-based (regex).""",
        parameters={
            "name": ActionParameter(
                name="name",
                description="Trigger identifier",
                type="string",
                required=True,
            ),
            "semantic_patterns": ActionParameter(
                name="semantic_patterns",
                description="Natural language patterns",
                type="list",
                required=False,
            ),
            "keyword_patterns": ActionParameter(
                name="keyword_patterns",
                description="Regex patterns for quick matching",
                type="list",
                required=False,
            ),
            "priority": ActionParameter(
                name="priority",
                description="Priority for conflict resolution (0-100)",
                type="int",
                required=False,
                default=50,
            ),
        },
        triggers=["add trigger", "define activation", "set when to activate"],
        handler="architect_service.add_trigger",
    ),

    # Build Phase
    Action(
        name="draft_decision_prompt",
        description="""Draft the decision prompt that determines which action to take.

        This is the "brain" of the behavior. It should include:
        - Role definition
        - Available actions with descriptions
        - Decision criteria
        - Response format
        - Domain-specific guidance""",
        parameters={
            "role_description": ActionParameter(
                name="role_description",
                description="How to describe the agent's role",
                type="string",
                required=True,
            ),
            "decision_criteria": ActionParameter(
                name="decision_criteria",
                description="Rules for choosing actions",
                type="list",
                required=False,
            ),
            "domain_knowledge": ActionParameter(
                name="domain_knowledge",
                description="Domain-specific knowledge to embed",
                type="string",
                required=False,
            ),
        },
        triggers=["create decision prompt", "draft brain", "write decision logic"],
        handler="architect_service.draft_decision_prompt",
    ),
    Action(
        name="draft_synthesis_prompt",
        description="""Draft the synthesis prompt that formats responses.

        This shapes the "voice" of the behavior. It defines:
        - Output style and tone
        - Formatting requirements
        - Error handling""",
        parameters={
            "style_guidelines": ActionParameter(
                name="style_guidelines",
                description="How responses should be styled",
                type="list",
                required=False,
            ),
            "tone": ActionParameter(
                name="tone",
                description="Overall tone (friendly, professional, etc.)",
                type="string",
                required=False,
                default="helpful",
            ),
        },
        triggers=["create synthesis prompt", "format voice", "define style"],
        handler="architect_service.draft_synthesis_prompt",
    ),

    # Test Phase
    Action(
        name="generate_test_cases",
        description="""Generate test cases for the behavior.

        Creates positive tests (correct usage), negative tests (edge cases),
        and comprehensive coverage of all actions.""",
        parameters={
            "test_type": ActionParameter(
                name="test_type",
                description="Type: 'positive', 'negative', 'edge_case', 'all'",
                type="string",
                required=False,
                default="all",
                enum_values=["positive", "negative", "edge_case", "all"],
            ),
            "count": ActionParameter(
                name="count",
                description="Number of tests to generate per category",
                type="int",
                required=False,
                default=5,
            ),
        },
        triggers=["generate tests", "create test cases", "write tests"],
        handler="architect_service.generate_test_cases",
    ),
    Action(
        name="run_tests",
        description="""Run test cases against the behavior.

        Executes tests and reports pass/fail rates.""",
        parameters={
            "test_ids": ActionParameter(
                name="test_ids",
                description="Specific tests to run, or 'all'",
                type="list",
                required=False,
            ),
        },
        triggers=["run tests", "validate behavior", "check tests"],
        handler="architect_service.run_tests",
    ),

    # Iterate Phase
    Action(
        name="analyze_failures",
        description="""Analyze test failures to identify patterns and root causes.

        Reviews failed tests and suggests fixes.""",
        parameters={},
        triggers=["analyze failures", "why failing", "find problems"],
        handler="architect_service.analyze_failures",
    ),
    Action(
        name="apply_fix",
        description="""Apply a suggested fix to the behavior.

        Modifies prompts, actions, or triggers based on failure analysis.""",
        parameters={
            "fix_id": ActionParameter(
                name="fix_id",
                description="ID of the fix to apply",
                type="string",
                required=True,
            ),
            "target": ActionParameter(
                name="target",
                description="What to fix: 'decision_prompt', 'synthesis_prompt', 'action', 'trigger'",
                type="string",
                required=True,
                enum_values=["decision_prompt", "synthesis_prompt", "action", "trigger"],
            ),
        },
        triggers=["apply fix", "fix issue", "resolve problem"],
        handler="architect_service.apply_fix",
    ),

    # Evolution Phase
    Action(
        name="evolve_behavior",
        description="""Evolve the behavior using genetic algorithms.

        Creates a population of variants, evaluates fitness, and
        evolves towards better performance. Includes self-referential
        mutation where the mutation prompts themselves evolve.""",
        parameters={
            "generations": ActionParameter(
                name="generations",
                description="Number of evolution generations",
                type="int",
                required=False,
                default=5,
            ),
            "population_size": ActionParameter(
                name="population_size",
                description="Size of evolution population",
                type="int",
                required=False,
                default=6,
            ),
        },
        triggers=["evolve behavior", "optimize", "improve via evolution"],
        requires_confirmation=True,  # Evolution can take time
        handler="architect_service.evolve_behavior",
    ),

    # Register Phase
    Action(
        name="register_behavior",
        description="""Register the behavior in the registry.

        Adds the behavior to the registry with appropriate tier and status.
        Generated behaviors start in STAGING for monitoring.""",
        parameters={
            "tier": ActionParameter(
                name="tier",
                description="Trust tier: 'generated' or 'experimental'",
                type="string",
                required=False,
                default="generated",
                enum_values=["generated", "experimental"],
            ),
            "status": ActionParameter(
                name="status",
                description="Initial status: 'staging', 'testing'",
                type="string",
                required=False,
                default="staging",
                enum_values=["staging", "testing"],
            ),
        },
        triggers=["register behavior", "add to registry", "save behavior"],
        handler="architect_service.register_behavior",
    ),
    Action(
        name="promote_behavior",
        description="""Promote a behavior from STAGING to ACTIVE.

        Should only be done after monitoring shows good performance.""",
        parameters={
            "behavior_id": ActionParameter(
                name="behavior_id",
                description="ID of behavior to promote",
                type="string",
                required=True,
            ),
            "reason": ActionParameter(
                name="reason",
                description="Reason for promotion",
                type="string",
                required=True,
            ),
        },
        triggers=["promote behavior", "activate", "go live"],
        requires_confirmation=True,
        handler="architect_service.promote_behavior",
    ),
]


# =============================================================================
# Triggers
# =============================================================================

ARCHITECT_TRIGGERS = [
    Trigger(
        name="create_behavior",
        description="User wants to create a new behavior",
        semantic_patterns=[
            "create a behavior for",
            "build a new behavior",
            "I need an agent that",
            "make a behavior that can",
            "design a behavior to",
        ],
        keyword_patterns=[
            r"create\s+(a\s+)?behavior",
            r"build\s+(a\s+)?behavior",
            r"new\s+behavior",
        ],
        priority=90,
    ),
    Trigger(
        name="improve_behavior",
        description="User wants to improve an existing behavior",
        semantic_patterns=[
            "improve this behavior",
            "make it better",
            "evolve the behavior",
            "optimize the prompts",
        ],
        keyword_patterns=[
            r"improve\s+(the\s+)?behavior",
            r"evolve\s+(the\s+)?behavior",
            r"optimize",
        ],
        priority=80,
    ),
    Trigger(
        name="test_behavior",
        description="User wants to test a behavior",
        semantic_patterns=[
            "test the behavior",
            "run the tests",
            "validate this",
        ],
        keyword_patterns=[
            r"test\s+(the\s+)?behavior",
            r"run\s+tests",
        ],
        priority=70,
    ),
]


# =============================================================================
# Constraints
# =============================================================================

ARCHITECT_CONSTRAINTS = BehaviorConstraints(
    # Evolution and registration need confirmation
    requires_user_confirmation=["evolve_behavior", "promote_behavior"],

    # Rate limit expensive operations
    rate_limits={
        "research_domain": 10,  # per minute
        "evolve_behavior": 1,
        "run_tests": 5,
    },

    # Style
    style_guidelines=[
        "Be technical but accessible",
        "Explain decisions clearly",
        "Report progress at each phase",
        "Warn about potential issues",
    ],
)


# =============================================================================
# Test Cases
# =============================================================================

ARCHITECT_TEST_CASES = [
    BehaviorTestCase(
        test_id="test_research_kitchen_timer",
        name="Research Kitchen Timer Behavior",
        description="Should research domain when asked to create timer behavior",
        user_query="Create a behavior for managing kitchen timers",
        expected_actions=["research_domain"],
        priority="high",
        tags=["research", "creation"],
    ),
    BehaviorTestCase(
        test_id="test_design_after_research",
        name="Design After Research",
        description="Should proceed to design after research is complete",
        user_query="Great, now design the behavior structure",
        context={"phase": "design", "research_complete": True},
        expected_actions=["draft_behavior_structure"],
        priority="high",
        tags=["design", "workflow"],
    ),
    BehaviorTestCase(
        test_id="test_add_action",
        name="Add Action to Design",
        description="Should add actions when requested",
        user_query="Add an action for setting a timer",
        context={"phase": "design", "has_structure": True},
        expected_actions=["add_action"],
        priority="high",
        tags=["design", "actions"],
    ),
    BehaviorTestCase(
        test_id="test_generate_tests",
        name="Generate Test Cases",
        description="Should generate tests when behavior is built",
        user_query="Generate tests for this behavior",
        context={"phase": "build", "prompts_created": True},
        expected_actions=["generate_test_cases"],
        priority="high",
        tags=["testing"],
    ),
    BehaviorTestCase(
        test_id="test_run_tests",
        name="Run Tests",
        description="Should run tests when requested",
        user_query="Run the tests",
        context={"phase": "test", "has_test_cases": True},
        expected_actions=["run_tests"],
        priority="high",
        tags=["testing"],
    ),
    BehaviorTestCase(
        test_id="test_analyze_failures",
        name="Analyze Test Failures",
        description="Should analyze failures when tests fail",
        user_query="Some tests failed, what went wrong?",
        context={"phase": "iterate", "has_failures": True},
        expected_actions=["analyze_failures"],
        priority="high",
        tags=["iteration", "debugging"],
    ),
    BehaviorTestCase(
        test_id="test_register_behavior",
        name="Register Completed Behavior",
        description="Should register behavior when complete",
        user_query="Register this behavior",
        context={"phase": "register", "tests_passing": True},
        expected_actions=["register_behavior"],
        priority="high",
        tags=["registration"],
    ),
    # Negative tests
    BehaviorTestCase(
        test_id="test_no_evolution_without_tests",
        name="No Evolution Without Tests",
        description="Should not evolve without test cases",
        user_query="Evolve this behavior",
        context={"phase": "build", "has_test_cases": False},
        forbidden_actions=["evolve_behavior"],
        expected_actions=["generate_test_cases"],
        priority="high",
        tags=["negative", "evolution"],
    ),
    BehaviorTestCase(
        test_id="test_no_register_without_testing",
        name="No Registration Without Testing",
        description="Should not register without running tests",
        user_query="Register this now",
        context={"phase": "build", "tests_run": False},
        forbidden_actions=["register_behavior"],
        expected_actions=["run_tests"],
        priority="high",
        tags=["negative", "registration"],
    ),
]


# =============================================================================
# Template
# =============================================================================

BEHAVIOR_ARCHITECT_TEMPLATE = Behavior(
    behavior_id="behavior_architect",
    name="Behavior Architect",
    description="""Meta-behavior that creates, tests, and evolves other behaviors.

    The Behavior Architect is a god-level agentic AI that can design new behaviors
    from natural language descriptions. It follows a rigorous process:

    1. Research: Understand the domain via web search and existing patterns
    2. Design: Create behavior structure (actions, triggers, constraints)
    3. Build: Generate prompts and test cases
    4. Test: Run tests and analyze results
    5. Iterate: Fix failures and improve
    6. Evolve: Optional genetic algorithm optimization
    7. Register: Add to registry with appropriate trust level

    This is a CORE tier behavior with maximum trust, as it can create new behaviors
    that will be executed by agents.""",
    version="1.0.0",

    tier=BehaviorTier.CORE,
    status=BehaviorStatus.ACTIVE,

    actions=ARCHITECT_ACTIONS,
    triggers=ARCHITECT_TRIGGERS,
    prompts=BehaviorPrompts(
        decision_prompt=BEHAVIOR_ARCHITECT_DECISION_PROMPT,
        synthesis_prompt=BEHAVIOR_ARCHITECT_SYNTHESIS_PROMPT,
    ),
    constraints=ARCHITECT_CONSTRAINTS,

    domain_context="""The Behavior Architect creates AI behaviors - pluggable modules that define
    what an agent CAN DO. Behaviors are data structures (not code) that include:
    - Actions: Specific capabilities with parameters and triggers
    - Triggers: Patterns that activate the behavior
    - Prompts: LLM prompts for decision and synthesis
    - Constraints: Safety rules and style guidelines
    - Tests: Validation cases

    Behaviors follow a trust hierarchy:
    - CORE: Built-in, maximum trust (like this architect)
    - ADDON: Official packages, high trust
    - APPLICATION: Host app provided, medium trust
    - GENERATED: Auto-created by architect, low trust (needs monitoring)
    - EXPERIMENTAL: Under testing, minimal trust

    Generated behaviors start in STAGING status and need monitoring before
    being promoted to ACTIVE.""",

    author="draagon_ai",
    is_evolvable=False,  # The architect itself should not be auto-evolved
    test_cases=ARCHITECT_TEST_CASES,
)


def create_behavior_architect() -> Behavior:
    """Create a fresh instance of the Behavior Architect.

    Returns:
        Behavior instance configured as the Behavior Architect
    """
    from copy import deepcopy
    architect = deepcopy(BEHAVIOR_ARCHITECT_TEMPLATE)
    architect.created_at = datetime.now()
    architect.updated_at = datetime.now()
    return architect
