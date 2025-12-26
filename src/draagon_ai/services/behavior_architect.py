"""Behavior Architect Service.

The meta-service that creates, tests, and evolves behaviors.
This is the "god-level agentic AI architect" that can design new behaviors
from natural language descriptions.

Architecture:
- Research Phase: Understand the domain via web search and existing behaviors
- Design Phase: Create behavior structure (actions, triggers, constraints)
- Build Phase: Generate prompts and test cases
- Test Phase: Run tests and analyze failures
- Iterate Phase: Fix failures and improve
- Evolve Phase: Genetic algorithm optimization
- Register Phase: Add to registry with appropriate trust level
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Protocol
import re
import xml.etree.ElementTree as ET

from ..behaviors.types import (
    Action,
    ActionExample,
    ActionParameter,
    Behavior,
    BehaviorConstraints,
    BehaviorMetrics,
    BehaviorPrompts,
    BehaviorStatus,
    BehaviorTestCase,
    BehaviorTier,
    DomainResearchResult,
    EvolutionConfig,
    FailureAnalysis,
    TestOutcome,
    TestResults,
    Trigger,
    ValidationIssue,
)
from ..behaviors.registry import BehaviorRegistry


# =============================================================================
# Protocols for Dependencies
# =============================================================================


class LLMProvider(Protocol):
    """Protocol for LLM inference."""

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.7,
    ) -> str:
        """Generate text from prompt."""
        ...


class WebSearchProvider(Protocol):
    """Protocol for web search."""

    async def search(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web and return results."""
        ...


# =============================================================================
# Design Types
# =============================================================================


@dataclass
class BehaviorDesign:
    """Intermediate design state for a behavior being created."""

    behavior_id: str
    name: str
    description: str

    # Core structure
    actions: list[Action] = field(default_factory=list)
    triggers: list[Trigger] = field(default_factory=list)
    constraints: BehaviorConstraints = field(default_factory=BehaviorConstraints)

    # Context
    domain_context: str = ""
    research: DomainResearchResult | None = None

    # State for agentic loop
    current_phase: str = "research"  # research, design, build, test, iterate, evolve, register
    iteration_count: int = 0
    max_iterations: int = 5

    # Validation
    validation_issues: list[ValidationIssue] = field(default_factory=list)


@dataclass
class MutationPrompt:
    """A prompt that describes how to mutate other prompts.

    Used in self-referential mutation where the mutation prompts
    themselves evolve over generations.
    """

    prompt_id: str
    content: str
    fitness: float = 0.0
    usage_count: int = 0
    success_count: int = 0


# =============================================================================
# Prompts
# =============================================================================

RESEARCH_PROMPT = '''You are researching a domain to create an AI behavior.

USER REQUEST:
{description}

EXISTING SIMILAR BEHAVIORS:
{existing_behaviors}

WEB SEARCH RESULTS:
{search_results}

Analyze this information and produce a structured research result.

Think through:
1. What are the CORE TASKS this behavior should do?
2. What specific ACTIONS should be available?
3. What user inputs (TRIGGERS) should activate this behavior?
4. What CONSTRAINTS should limit this behavior?
5. What domain KNOWLEDGE should be embedded in the prompts?

Be thorough but focused. This research will guide behavior creation.

Respond in XML format:
<research>
  <core_tasks>
    <task>Description of a core task</task>
    <task>Another core task</task>
  </core_tasks>
  <suggested_actions>
    <action name="action_name" description="What it does">
      <parameter name="param_name" type="string" required="true">Description</parameter>
    </action>
  </suggested_actions>
  <triggers>
    <trigger>Pattern or phrase that should activate this behavior</trigger>
  </triggers>
  <constraints>
    <constraint>Something this behavior should NOT do</constraint>
  </constraints>
  <domain_knowledge>
    Relevant background knowledge for this domain that should be included
    in the behavior's prompts to help the LLM make good decisions.
  </domain_knowledge>
  <sources>
    <source>URL or reference</source>
  </sources>
</research>
'''

DESIGN_PROMPT = '''You are designing an AI behavior based on research.

RESEARCH RESULTS:
{research}

USER CONSTRAINTS:
{user_constraints}

Design a complete behavior structure with:

1. ACTIONS - For each core task, define an action with:
   - name: snake_case identifier
   - description: Clear explanation of what it does
   - parameters: Required and optional inputs
   - triggers: Phrases that suggest this action

2. TRIGGERS - Patterns that activate this behavior:
   - semantic_patterns: Natural language patterns (LLM evaluates)
   - keyword_patterns: Regex patterns for quick matching
   - priority: 0-100, higher wins conflicts

3. CONSTRAINTS - Safety and style rules:
   - Actions requiring confirmation
   - Rate limits
   - Style guidelines

Focus on usability and robustness. Actions should be intuitive.

Respond in XML format:
<design>
  <behavior_id>snake_case_id</behavior_id>
  <name>Human Readable Name</name>
  <description>What this behavior enables</description>

  <actions>
    <action name="action_name" requires_confirmation="false">
      <description>What this action does</description>
      <parameter name="param" type="string" required="true">Parameter description</parameter>
      <trigger>Phrase that suggests this action</trigger>
      <example query="User says this" outcome="Expected result"/>
    </action>
  </actions>

  <triggers>
    <trigger name="main_trigger" priority="70">
      <semantic>Natural language pattern</semantic>
      <keyword>regex_pattern</keyword>
    </trigger>
  </triggers>

  <constraints>
    <require_confirmation>action_name</require_confirmation>
    <rate_limit action="action_name" per_minute="10"/>
    <style>Be concise and helpful</style>
  </constraints>

  <domain_context>
    Background knowledge to embed in prompts
  </domain_context>
</design>
'''

PROMPT_GENERATION_PROMPT = '''You are writing prompts for an AI behavior.

BEHAVIOR DESIGN:
{design}

DOMAIN KNOWLEDGE:
{domain_knowledge}

Write two prompts:

1. DECISION PROMPT - Determines which action to take
   Requirements:
   - Define the role clearly
   - List all available actions with descriptions
   - Explain when to use each action
   - Include decision criteria
   - Specify response format (XML)
   - Include domain-specific guidance
   - Be thorough - this is the "brain" of the behavior

2. SYNTHESIS PROMPT - Formats the response to the user
   Requirements:
   - Define output style
   - Include domain-appropriate tone
   - Specify formatting requirements
   - Handle error cases gracefully

The decision prompt should help the LLM pick the RIGHT action.
The synthesis prompt should help format a GOOD response.

Respond with:
<prompts>
  <decision_prompt>
Your decision prompt here. Include placeholders like:
- {{actions}} - Will be replaced with action list
- {{context}} - Will be replaced with context
- {{query}} - Will be replaced with user query
  </decision_prompt>
  <synthesis_prompt>
Your synthesis prompt here. Include placeholders like:
- {{action_result}} - Will be replaced with action output
- {{style}} - Will be replaced with style guidelines
  </synthesis_prompt>
</prompts>
'''

TEST_GENERATION_PROMPT = '''You are generating test cases for an AI behavior.

BEHAVIOR:
ID: {behavior_id}
Name: {name}
Description: {description}

ACTIONS:
{actions}

TRIGGERS:
{triggers}

Generate comprehensive test cases:

1. POSITIVE TESTS - One for each action, showing correct usage
2. NEGATIVE TESTS - Common mistakes, edge cases, invalid inputs
3. EDGE CASES - Ambiguous inputs, boundary conditions

Each test should validate that the behavior:
- Picks the RIGHT action for the input
- Includes expected elements in the response
- Avoids forbidden actions

Generate at least:
- 1 positive test per action
- 3 negative tests (wrong action risks)
- 3 edge case tests (ambiguous inputs)

Respond with:
<test_cases>
  <test id="test_001" name="Test Name" priority="high">
    <description>What this test validates</description>
    <user_query>What the user says</user_query>
    <context key="value">Optional context</context>
    <expected_actions>
      <action>expected_action_name</action>
    </expected_actions>
    <expected_response_contains>
      <phrase>Something the response should include</phrase>
    </expected_response_contains>
    <forbidden_actions>
      <action>action_that_should_not_be_used</action>
    </forbidden_actions>
  </test>
</test_cases>
'''

FAILURE_ANALYSIS_PROMPT = '''You are analyzing test failures for an AI behavior.

BEHAVIOR: {behavior_id}

TEST RESULTS:
{test_results}

FAILED TESTS:
{failed_tests}

Analyze the failures and identify:

1. PATTERNS - What types of failures are occurring?
   - Wrong action selected
   - Missing required content
   - Forbidden action used
   - Other issues

2. ROOT CAUSES - Why are these failures happening?
   - Prompt unclear about when to use action
   - Action description misleading
   - Trigger too broad or narrow
   - Missing edge case handling

3. SUGGESTED FIXES - How to fix each root cause?
   - Prompt modifications
   - Action description changes
   - Trigger adjustments
   - New test cases needed

Respond with:
<analysis>
  <patterns>
    <pattern type="wrong_action" count="3">Description of pattern</pattern>
  </patterns>
  <root_causes>
    <cause id="cause_1">Specific root cause</cause>
  </root_causes>
  <fixes>
    <fix cause_id="cause_1" target="decision_prompt">
      Specific fix to apply
    </fix>
  </fixes>
</analysis>
'''

# Initial mutation prompts for evolution
INITIAL_MUTATION_PROMPTS = [
    MutationPrompt(
        prompt_id="expand_detail",
        content="Make this prompt more detailed and specific. Add examples where helpful.",
    ),
    MutationPrompt(
        prompt_id="simplify",
        content="Simplify this prompt. Remove redundancy. Be more direct and clear.",
    ),
    MutationPrompt(
        prompt_id="add_constraints",
        content="Add constraints and edge case handling. Be more explicit about what NOT to do.",
    ),
    MutationPrompt(
        prompt_id="improve_structure",
        content="Improve the structure and organization. Use clearer headings and sections.",
    ),
    MutationPrompt(
        prompt_id="domain_focus",
        content="Make this prompt more domain-specific. Add expert-level knowledge and terminology.",
    ),
]


# =============================================================================
# Service
# =============================================================================


class BehaviorArchitectService:
    """Service for creating, testing, and evolving behaviors.

    This is the meta-agent that can create other behaviors from
    natural language descriptions. It follows an agentic loop:

    1. Research: Understand the domain
    2. Design: Create behavior structure
    3. Build: Generate prompts and tests
    4. Test: Run tests
    5. Iterate: Fix failures
    6. Evolve: Optimize via genetic algorithms (optional)
    7. Register: Add to registry

    Usage:
        architect = BehaviorArchitectService(
            llm=my_llm_provider,
            web_search=my_search_provider,
            registry=my_registry,
        )

        behavior = await architect.create_behavior(
            "A behavior for managing kitchen timers"
        )
    """

    def __init__(
        self,
        llm: LLMProvider,
        web_search: WebSearchProvider | None = None,
        registry: BehaviorRegistry | None = None,
        tool_executor: Callable[[str, dict], Any] | None = None,
    ):
        """Initialize the architect service.

        Args:
            llm: LLM provider for generation
            web_search: Optional web search for research
            registry: Optional registry for existing behaviors
            tool_executor: Optional tool executor for testing
        """
        self._llm = llm
        self._web_search = web_search
        self._registry = registry
        self._tool_executor = tool_executor
        self._mutation_prompts = list(INITIAL_MUTATION_PROMPTS)

    # =========================================================================
    # Main Entry Point
    # =========================================================================

    async def create_behavior(
        self,
        description: str,
        user_constraints: dict[str, Any] | None = None,
        evolve: bool = False,
        evolution_config: EvolutionConfig | None = None,
    ) -> Behavior:
        """Create a complete behavior from a description.

        This is the main entry point that runs the full agentic loop.

        Args:
            description: Natural language description of desired behavior
            user_constraints: Optional constraints from user
            evolve: Whether to run evolution after initial creation
            evolution_config: Optional evolution configuration

        Returns:
            Complete, tested Behavior ready for registration
        """
        # Phase 1: Research
        research = await self.research_domain(description)

        # Phase 2: Design
        design = await self.design_behavior(research, user_constraints)

        # Phase 3: Build
        behavior, test_cases = await self.build_behavior(design)

        # Phase 4-5: Test & Iterate
        behavior = await self.test_and_iterate(behavior, test_cases)

        # Phase 6: Evolve (optional)
        if evolve:
            config = evolution_config or EvolutionConfig()
            result = await self.evolve_behavior(behavior, test_cases, config)
            if result.approved:
                behavior = result.evolved_behavior

        # Phase 7: Register
        if self._registry:
            await self.register_behavior(behavior)

        return behavior

    # =========================================================================
    # Phase 1: Research
    # =========================================================================

    async def research_domain(
        self,
        description: str,
        search_web: bool = True,
        search_existing: bool = True,
    ) -> DomainResearchResult:
        """Research a domain to understand what a behavior should do.

        Args:
            description: User's description of desired behavior
            search_web: Whether to search web for best practices
            search_existing: Whether to search existing behaviors for patterns

        Returns:
            Structured research result
        """
        # Gather existing behaviors for context
        existing_behaviors = ""
        if search_existing and self._registry:
            similar = self._find_similar_behaviors(description)
            if similar:
                existing_behaviors = self._format_behaviors_for_context(similar)

        # Search web for domain knowledge
        search_results = ""
        if search_web and self._web_search:
            # Extract key terms for search
            search_query = f"best practices for {description}"
            results = await self._web_search.search(search_query, max_results=5)
            search_results = self._format_search_results(results)

        # Generate research using LLM
        prompt = RESEARCH_PROMPT.format(
            description=description,
            existing_behaviors=existing_behaviors or "None found",
            search_results=search_results or "No web search performed",
        )

        response = await self._llm.generate(
            prompt,
            system_prompt="You are a domain research expert helping design AI behaviors.",
            temperature=0.7,
        )

        # Parse XML response
        return self._parse_research_response(response, description)

    def _find_similar_behaviors(self, description: str) -> list[Behavior]:
        """Find behaviors similar to the description."""
        if not self._registry:
            return []

        # Simple keyword matching for now
        keywords = set(description.lower().split())
        matches = []

        for behavior in self._registry.get_all():
            behavior_text = f"{behavior.name} {behavior.description}".lower()
            overlap = len(keywords & set(behavior_text.split()))
            if overlap >= 2:  # At least 2 keyword matches
                matches.append(behavior)

        return matches[:3]  # Top 3

    def _format_behaviors_for_context(self, behaviors: list[Behavior]) -> str:
        """Format behaviors for LLM context."""
        lines = []
        for b in behaviors:
            lines.append(f"- {b.name}: {b.description}")
            for action in b.actions[:3]:  # Top 3 actions
                lines.append(f"  - Action: {action.name} - {action.description}")
        return "\n".join(lines)

    def _format_search_results(self, results: list[dict]) -> str:
        """Format search results for LLM context."""
        lines = []
        for r in results:
            title = r.get("title", "Untitled")
            snippet = r.get("snippet", "")[:200]
            url = r.get("url", "")
            lines.append(f"- {title}\n  {snippet}\n  Source: {url}")
        return "\n\n".join(lines)

    def _parse_research_response(
        self, response: str, domain: str
    ) -> DomainResearchResult:
        """Parse XML research response into structured result."""
        result = DomainResearchResult(domain=domain)

        try:
            # Extract XML content
            xml_match = re.search(r"<research>(.*?)</research>", response, re.DOTALL)
            if not xml_match:
                # Fallback: try to extract key sections
                return self._parse_research_fallback(response, domain)

            xml_content = f"<research>{xml_match.group(1)}</research>"
            root = ET.fromstring(xml_content)

            # Parse core tasks
            for task in root.findall(".//core_tasks/task"):
                if task.text:
                    result.core_tasks.append(task.text.strip())

            # Parse suggested actions
            for action in root.findall(".//suggested_actions/action"):
                action_dict = {
                    "name": action.get("name", "unnamed"),
                    "description": action.get("description", ""),
                    "parameters": [],
                }
                for param in action.findall("parameter"):
                    action_dict["parameters"].append({
                        "name": param.get("name", "param"),
                        "type": param.get("type", "string"),
                        "required": param.get("required", "true") == "true",
                        "description": param.text.strip() if param.text else "",
                    })
                result.suggested_actions.append(action_dict)

            # Parse triggers
            for trigger in root.findall(".//triggers/trigger"):
                if trigger.text:
                    result.suggested_triggers.append(trigger.text.strip())

            # Parse constraints
            for constraint in root.findall(".//constraints/constraint"):
                if constraint.text:
                    result.constraints.append(constraint.text.strip())

            # Parse domain knowledge
            knowledge = root.find(".//domain_knowledge")
            if knowledge is not None and knowledge.text:
                result.domain_knowledge = knowledge.text.strip()

            # Parse sources
            for source in root.findall(".//sources/source"):
                if source.text:
                    result.sources.append(source.text.strip())

        except ET.ParseError:
            return self._parse_research_fallback(response, domain)

        return result

    def _parse_research_fallback(self, response: str, domain: str) -> DomainResearchResult:
        """Fallback parsing when XML fails."""
        result = DomainResearchResult(domain=domain)

        # Try to extract sections by header patterns
        lines = response.split("\n")
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            lower = line.lower()
            if "core task" in lower or "main task" in lower:
                current_section = "tasks"
            elif "action" in lower and "suggest" in lower:
                current_section = "actions"
            elif "trigger" in lower:
                current_section = "triggers"
            elif "constraint" in lower:
                current_section = "constraints"
            elif line.startswith("-") or line.startswith("*"):
                content = line.lstrip("-* ").strip()
                if current_section == "tasks":
                    result.core_tasks.append(content)
                elif current_section == "triggers":
                    result.suggested_triggers.append(content)
                elif current_section == "constraints":
                    result.constraints.append(content)

        # If we got nothing, add generic defaults
        if not result.core_tasks:
            result.core_tasks = ["Handle user requests", "Provide helpful responses"]

        return result

    # =========================================================================
    # Phase 2: Design
    # =========================================================================

    async def design_behavior(
        self,
        research: DomainResearchResult,
        user_constraints: dict[str, Any] | None = None,
    ) -> BehaviorDesign:
        """Design a behavior structure from research.

        Args:
            research: Research results
            user_constraints: Optional user-provided constraints

        Returns:
            BehaviorDesign with actions, triggers, constraints
        """
        prompt = DESIGN_PROMPT.format(
            research=research.to_summary(),
            user_constraints=str(user_constraints or "None specified"),
        )

        response = await self._llm.generate(
            prompt,
            system_prompt="You are an AI behavior architect designing robust agent behaviors.",
            temperature=0.7,
        )

        return self._parse_design_response(response, research)

    def _parse_design_response(
        self, response: str, research: DomainResearchResult
    ) -> BehaviorDesign:
        """Parse XML design response into BehaviorDesign."""
        design = BehaviorDesign(
            behavior_id="new_behavior",
            name="New Behavior",
            description="Auto-generated behavior",
            research=research,
            domain_context=research.domain_knowledge,
        )

        try:
            # Extract XML
            xml_match = re.search(r"<design>(.*?)</design>", response, re.DOTALL)
            if not xml_match:
                return self._design_from_research(research)

            xml_content = f"<design>{xml_match.group(1)}</design>"
            root = ET.fromstring(xml_content)

            # Parse identity
            bid = root.find("behavior_id")
            if bid is not None and bid.text:
                design.behavior_id = bid.text.strip()

            name = root.find("name")
            if name is not None and name.text:
                design.name = name.text.strip()

            desc = root.find("description")
            if desc is not None and desc.text:
                design.description = desc.text.strip()

            # Parse actions
            for action_elem in root.findall(".//actions/action"):
                action = self._parse_action_element(action_elem)
                design.actions.append(action)

            # Parse triggers
            for trigger_elem in root.findall(".//triggers/trigger"):
                trigger = self._parse_trigger_element(trigger_elem)
                design.triggers.append(trigger)

            # Parse constraints
            constraints_elem = root.find("constraints")
            if constraints_elem is not None:
                design.constraints = self._parse_constraints_element(constraints_elem)

            # Parse domain context
            context = root.find("domain_context")
            if context is not None and context.text:
                design.domain_context = context.text.strip()

        except ET.ParseError:
            return self._design_from_research(research)

        return design

    def _parse_action_element(self, elem: ET.Element) -> Action:
        """Parse an action XML element."""
        action = Action(
            name=elem.get("name", "unnamed_action"),
            description="",
            requires_confirmation=elem.get("requires_confirmation", "false") == "true",
        )

        desc = elem.find("description")
        if desc is not None and desc.text:
            action.description = desc.text.strip()

        # Parse parameters
        for param_elem in elem.findall("parameter"):
            param = ActionParameter(
                name=param_elem.get("name", "param"),
                description=param_elem.text.strip() if param_elem.text else "",
                type=param_elem.get("type", "string"),
                required=param_elem.get("required", "true") == "true",
            )
            action.parameters[param.name] = param

        # Parse triggers
        for trigger in elem.findall("trigger"):
            if trigger.text:
                action.triggers.append(trigger.text.strip())

        # Parse examples
        for example in elem.findall("example"):
            action.examples.append(ActionExample(
                user_query=example.get("query", ""),
                action_call={"name": action.name},
                expected_outcome=example.get("outcome", ""),
            ))

        return action

    def _parse_trigger_element(self, elem: ET.Element) -> Trigger:
        """Parse a trigger XML element."""
        trigger = Trigger(
            name=elem.get("name", "main"),
            description="",
            priority=int(elem.get("priority", "50")),
        )

        for semantic in elem.findall("semantic"):
            if semantic.text:
                trigger.semantic_patterns.append(semantic.text.strip())

        for keyword in elem.findall("keyword"):
            if keyword.text:
                trigger.keyword_patterns.append(keyword.text.strip())

        return trigger

    def _parse_constraints_element(self, elem: ET.Element) -> BehaviorConstraints:
        """Parse constraints XML element."""
        constraints = BehaviorConstraints()

        for confirm in elem.findall("require_confirmation"):
            if confirm.text:
                constraints.requires_user_confirmation.append(confirm.text.strip())

        for limit in elem.findall("rate_limit"):
            action = limit.get("action", "")
            per_minute = int(limit.get("per_minute", "60"))
            if action:
                constraints.rate_limits[action] = per_minute

        for style in elem.findall("style"):
            if style.text:
                constraints.style_guidelines.append(style.text.strip())

        return constraints

    def _design_from_research(self, research: DomainResearchResult) -> BehaviorDesign:
        """Create a basic design directly from research (fallback)."""
        # Generate behavior_id from domain
        behavior_id = re.sub(r"[^a-z0-9]+", "_", research.domain.lower()).strip("_")

        design = BehaviorDesign(
            behavior_id=behavior_id,
            name=research.domain.title(),
            description=f"Behavior for {research.domain}",
            research=research,
            domain_context=research.domain_knowledge,
        )

        # Convert suggested actions
        for suggested in research.suggested_actions:
            action = Action(
                name=suggested.get("name", "action"),
                description=suggested.get("description", ""),
            )
            for param in suggested.get("parameters", []):
                action.parameters[param["name"]] = ActionParameter(
                    name=param["name"],
                    description=param.get("description", ""),
                    type=param.get("type", "string"),
                    required=param.get("required", True),
                )
            design.actions.append(action)

        # Create default trigger
        design.triggers.append(Trigger(
            name="main",
            description=f"Activate for {research.domain}",
            semantic_patterns=research.suggested_triggers[:3],
            priority=50,
        ))

        return design

    # =========================================================================
    # Phase 3: Build
    # =========================================================================

    async def build_behavior(
        self, design: BehaviorDesign
    ) -> tuple[Behavior, list[BehaviorTestCase]]:
        """Build a complete Behavior from a design.

        Args:
            design: The behavior design

        Returns:
            Tuple of (Behavior, list of test cases)
        """
        # Generate prompts
        prompts = await self._generate_prompts(design)

        # Generate test cases
        test_cases = await self._generate_test_cases(design)

        # Assemble behavior
        behavior = Behavior(
            behavior_id=design.behavior_id,
            name=design.name,
            description=design.description,
            tier=BehaviorTier.GENERATED,
            status=BehaviorStatus.TESTING,
            actions=design.actions,
            triggers=design.triggers,
            prompts=prompts,
            constraints=design.constraints,
            domain_context=design.domain_context,
            test_cases=test_cases,
            author="behavior_architect",
            is_evolvable=True,
        )

        return behavior, test_cases

    async def _generate_prompts(self, design: BehaviorDesign) -> BehaviorPrompts:
        """Generate decision and synthesis prompts."""
        # Format actions for context
        actions_text = []
        for action in design.actions:
            params = ", ".join(
                f"{p.name}: {p.type}" for p in action.parameters.values()
            )
            actions_text.append(
                f"- {action.name}({params}): {action.description}"
            )

        prompt = PROMPT_GENERATION_PROMPT.format(
            design=f"ID: {design.behavior_id}\nName: {design.name}\nDescription: {design.description}\n\nActions:\n" + "\n".join(actions_text),
            domain_knowledge=design.domain_context,
        )

        response = await self._llm.generate(
            prompt,
            system_prompt="You are an expert prompt engineer creating prompts for AI behaviors.",
            temperature=0.7,
        )

        return self._parse_prompts_response(response)

    def _parse_prompts_response(self, response: str) -> BehaviorPrompts:
        """Parse prompts XML response."""
        decision_prompt = ""
        synthesis_prompt = ""

        try:
            # Extract XML
            xml_match = re.search(r"<prompts>(.*?)</prompts>", response, re.DOTALL)
            if xml_match:
                xml_content = f"<prompts>{xml_match.group(1)}</prompts>"
                root = ET.fromstring(xml_content)

                decision = root.find("decision_prompt")
                if decision is not None and decision.text:
                    decision_prompt = decision.text.strip()

                synthesis = root.find("synthesis_prompt")
                if synthesis is not None and synthesis.text:
                    synthesis_prompt = synthesis.text.strip()
        except ET.ParseError:
            # Fallback: try to find prompts by markers
            if "decision_prompt" in response.lower():
                parts = response.split("synthesis_prompt", 1)
                if len(parts) == 2:
                    decision_prompt = parts[0].replace("decision_prompt", "").strip()
                    synthesis_prompt = parts[1].strip()

        # Ensure we have something
        if not decision_prompt:
            decision_prompt = "Analyze the user query and select the best action."
        if not synthesis_prompt:
            synthesis_prompt = "Format the response clearly and helpfully."

        return BehaviorPrompts(
            decision_prompt=decision_prompt,
            synthesis_prompt=synthesis_prompt,
        )

    async def _generate_test_cases(
        self, design: BehaviorDesign
    ) -> list[BehaviorTestCase]:
        """Generate test cases for the behavior."""
        # Format for prompt
        actions_text = "\n".join(
            f"- {a.name}: {a.description}" for a in design.actions
        )
        triggers_text = "\n".join(
            f"- {t.name}: {', '.join(t.semantic_patterns[:2])}" for t in design.triggers
        )

        prompt = TEST_GENERATION_PROMPT.format(
            behavior_id=design.behavior_id,
            name=design.name,
            description=design.description,
            actions=actions_text,
            triggers=triggers_text,
        )

        response = await self._llm.generate(
            prompt,
            system_prompt="You are a QA engineer generating comprehensive test cases.",
            temperature=0.8,  # Slightly higher for variety
        )

        return self._parse_test_cases_response(response)

    def _parse_test_cases_response(
        self, response: str
    ) -> list[BehaviorTestCase]:
        """Parse test cases XML response."""
        test_cases = []

        try:
            xml_match = re.search(r"<test_cases>(.*?)</test_cases>", response, re.DOTALL)
            if not xml_match:
                return self._generate_default_test_cases()

            xml_content = f"<test_cases>{xml_match.group(1)}</test_cases>"
            root = ET.fromstring(xml_content)

            for test_elem in root.findall("test"):
                test = BehaviorTestCase(
                    test_id=test_elem.get("id", f"test_{len(test_cases)+1}"),
                    name=test_elem.get("name", "Unnamed Test"),
                    priority=test_elem.get("priority", "medium"),
                )

                desc = test_elem.find("description")
                if desc is not None and desc.text:
                    test.description = desc.text.strip()

                query = test_elem.find("user_query")
                if query is not None and query.text:
                    test.user_query = query.text.strip()

                # Parse context
                for ctx in test_elem.findall("context"):
                    key = ctx.get("key")
                    if key and ctx.text:
                        test.context[key] = ctx.text.strip()

                # Parse expected actions
                for action in test_elem.findall(".//expected_actions/action"):
                    if action.text:
                        test.expected_actions.append(action.text.strip())

                # Parse expected response contains
                for phrase in test_elem.findall(".//expected_response_contains/phrase"):
                    if phrase.text:
                        test.expected_response_contains.append(phrase.text.strip())

                # Parse forbidden actions
                for action in test_elem.findall(".//forbidden_actions/action"):
                    if action.text:
                        test.forbidden_actions.append(action.text.strip())

                test_cases.append(test)

        except ET.ParseError:
            return self._generate_default_test_cases()

        return test_cases if test_cases else self._generate_default_test_cases()

    def _generate_default_test_cases(self) -> list[BehaviorTestCase]:
        """Generate minimal default test cases."""
        return [
            BehaviorTestCase(
                test_id="test_basic_greeting",
                name="Basic Greeting",
                description="Test that the behavior responds to greetings",
                user_query="Hello",
                expected_response_contains=["hello", "hi", "greet"],
            ),
            BehaviorTestCase(
                test_id="test_basic_help",
                name="Help Request",
                description="Test that the behavior can explain itself",
                user_query="What can you do?",
            ),
        ]

    # =========================================================================
    # Phase 4-5: Test & Iterate
    # =========================================================================

    async def test_and_iterate(
        self,
        behavior: Behavior,
        test_cases: list[BehaviorTestCase],
        max_iterations: int = 5,
    ) -> Behavior:
        """Test the behavior and iterate on failures.

        Args:
            behavior: The behavior to test
            test_cases: Test cases to run
            max_iterations: Maximum iteration attempts

        Returns:
            Improved behavior
        """
        for iteration in range(max_iterations):
            # Run tests
            results = await self._run_tests(behavior, test_cases)

            # Check if passing
            if results.pass_rate >= 0.8:  # 80% pass rate threshold
                behavior.test_results = results
                behavior.status = BehaviorStatus.STAGING
                return behavior

            # Analyze failures
            analysis = await self._analyze_failures(behavior, results)

            if analysis.no_failures:
                break

            # Apply fixes
            behavior = await self._apply_fixes(behavior, analysis)

        # Update status based on final results
        behavior.test_results = results
        if results.pass_rate >= 0.8:
            behavior.status = BehaviorStatus.STAGING
        else:
            behavior.status = BehaviorStatus.TESTING

        return behavior

    async def _run_tests(
        self,
        behavior: Behavior,
        test_cases: list[BehaviorTestCase],
    ) -> TestResults:
        """Run test cases against a behavior."""
        results = TestResults(
            total_tests=len(test_cases),
            run_at=datetime.now(),
        )

        start_time = datetime.now()

        for test in test_cases:
            outcome = await self._run_single_test(behavior, test)
            results.test_outcomes[test.test_id] = outcome

            if outcome.passed:
                results.passed += 1
            else:
                results.failed += 1

        results.duration_seconds = (datetime.now() - start_time).total_seconds()
        results.pass_rate = results.passed / results.total_tests if results.total_tests > 0 else 0.0

        return results

    async def _run_single_test(
        self,
        behavior: Behavior,
        test: BehaviorTestCase,
    ) -> TestOutcome:
        """Run a single test case."""
        start_time = datetime.now()

        try:
            # Simulate running the behavior's decision prompt
            # In a real implementation, this would use the orchestrator
            if not behavior.prompts:
                return TestOutcome(
                    test_id=test.test_id,
                    passed=False,
                    failure_reason="Behavior has no prompts",
                )

            # Build prompt with test context
            context = {
                "query": test.user_query,
                "context": test.context,
            }

            # Format actions for the decision prompt
            actions_text = "\n".join(
                f"- {a.name}: {a.description}" for a in behavior.actions
            )

            decision_prompt = behavior.prompts.decision_prompt.replace(
                "{{actions}}", actions_text
            ).replace(
                "{{query}}", test.user_query
            ).replace(
                "{{context}}", str(test.context)
            )

            # Get LLM decision
            response = await self._llm.generate(
                decision_prompt,
                temperature=0.3,  # Lower temp for consistency
            )

            # Extract action from response
            actual_action = self._extract_action_from_response(response)

            # Check test expectations
            passed = True
            failure_reason = None

            # Check expected actions
            if test.expected_actions:
                if actual_action not in test.expected_actions:
                    passed = False
                    failure_reason = f"Expected one of {test.expected_actions}, got {actual_action}"

            # Check forbidden actions
            if actual_action in test.forbidden_actions:
                passed = False
                failure_reason = f"Used forbidden action: {actual_action}"

            # Check response contains
            if test.expected_response_contains and passed:
                response_lower = response.lower()
                for phrase in test.expected_response_contains:
                    if phrase.lower() not in response_lower:
                        passed = False
                        failure_reason = f"Response missing expected phrase: {phrase}"
                        break

            latency = (datetime.now() - start_time).total_seconds() * 1000

            return TestOutcome(
                test_id=test.test_id,
                passed=passed,
                actual_action=actual_action,
                actual_response=response[:500],  # Truncate
                failure_reason=failure_reason,
                latency_ms=latency,
            )

        except Exception as e:
            return TestOutcome(
                test_id=test.test_id,
                passed=False,
                failure_reason=f"Test execution error: {str(e)}",
            )

    def _extract_action_from_response(self, response: str) -> str | None:
        """Extract action name from LLM response."""
        # Try XML format first
        action_match = re.search(r"<action[^>]*name=[\"']([^\"']+)[\"']", response)
        if action_match:
            return action_match.group(1)

        # Try simple action: format
        action_match = re.search(r"action:\s*(\w+)", response, re.IGNORECASE)
        if action_match:
            return action_match.group(1)

        # Try to find action name pattern
        action_match = re.search(r"(?:use|call|execute|invoke)\s+(\w+)", response, re.IGNORECASE)
        if action_match:
            return action_match.group(1)

        return None

    async def _analyze_failures(
        self,
        behavior: Behavior,
        results: TestResults,
    ) -> FailureAnalysis:
        """Analyze test failures to identify patterns and fixes."""
        failed_tests = [
            (tid, outcome)
            for tid, outcome in results.test_outcomes.items()
            if not outcome.passed
        ]

        if not failed_tests:
            return FailureAnalysis(no_failures=True)

        # Format for analysis prompt
        test_results_text = f"Pass rate: {results.pass_rate:.1%}\nPassed: {results.passed}\nFailed: {results.failed}"

        failed_tests_text = "\n\n".join([
            f"Test: {tid}\nExpected: {self._get_test_expectations(behavior, tid)}\nGot: {outcome.actual_action}\nReason: {outcome.failure_reason}"
            for tid, outcome in failed_tests
        ])

        prompt = FAILURE_ANALYSIS_PROMPT.format(
            behavior_id=behavior.behavior_id,
            test_results=test_results_text,
            failed_tests=failed_tests_text,
        )

        response = await self._llm.generate(
            prompt,
            system_prompt="You are a QA analyst identifying root causes of test failures.",
            temperature=0.5,
        )

        return self._parse_failure_analysis(response)

    def _get_test_expectations(self, behavior: Behavior, test_id: str) -> str:
        """Get expected outcomes for a test."""
        for test in behavior.test_cases:
            if test.test_id == test_id:
                return f"actions={test.expected_actions}, forbidden={test.forbidden_actions}"
        return "unknown"

    def _parse_failure_analysis(self, response: str) -> FailureAnalysis:
        """Parse failure analysis response."""
        analysis = FailureAnalysis()

        try:
            xml_match = re.search(r"<analysis>(.*?)</analysis>", response, re.DOTALL)
            if xml_match:
                xml_content = f"<analysis>{xml_match.group(1)}</analysis>"
                root = ET.fromstring(xml_content)

                # Parse patterns
                for pattern in root.findall(".//patterns/pattern"):
                    if pattern.text:
                        analysis.patterns.append(pattern.text.strip())

                # Parse root causes
                for cause in root.findall(".//root_causes/cause"):
                    if cause.text:
                        analysis.root_causes.append(cause.text.strip())

                # Parse fixes
                for fix in root.findall(".//fixes/fix"):
                    fix_dict = {
                        "cause_id": fix.get("cause_id", ""),
                        "target": fix.get("target", "decision_prompt"),
                        "description": fix.text.strip() if fix.text else "",
                    }
                    analysis.suggested_fixes.append(fix_dict)

        except ET.ParseError:
            # Basic fallback
            if "prompt" in response.lower():
                analysis.root_causes.append("Prompt may need clarification")
                analysis.suggested_fixes.append({
                    "cause_id": "prompt",
                    "target": "decision_prompt",
                    "description": "Add more specific guidance for action selection",
                })

        return analysis

    async def _apply_fixes(
        self,
        behavior: Behavior,
        analysis: FailureAnalysis,
    ) -> Behavior:
        """Apply suggested fixes to the behavior."""
        if not analysis.suggested_fixes or not behavior.prompts:
            return behavior

        for fix in analysis.suggested_fixes:
            target = fix.get("target", "")
            description = fix.get("description", "")

            if target == "decision_prompt" and description:
                # Apply fix to decision prompt
                behavior.prompts.decision_prompt = await self._apply_prompt_fix(
                    behavior.prompts.decision_prompt,
                    description,
                )
            elif target == "synthesis_prompt" and description:
                behavior.prompts.synthesis_prompt = await self._apply_prompt_fix(
                    behavior.prompts.synthesis_prompt,
                    description,
                )

        return behavior

    async def _apply_prompt_fix(self, prompt: str, fix_description: str) -> str:
        """Apply a fix to a prompt."""
        apply_prompt = f"""Improve this prompt based on the suggested fix:

CURRENT PROMPT:
{prompt}

SUGGESTED FIX:
{fix_description}

Return the improved prompt. Keep the same overall structure but apply the fix.
Only return the improved prompt text, no explanation."""

        improved = await self._llm.generate(
            apply_prompt,
            temperature=0.5,
        )

        return improved.strip()

    # =========================================================================
    # Phase 6: Evolution
    # =========================================================================

    async def evolve_behavior(
        self,
        behavior: Behavior,
        test_cases: list[BehaviorTestCase],
        config: EvolutionConfig,
    ) -> "BehaviorEvolutionResult":
        """Evolve a behavior using genetic algorithms.

        This implements the enhanced evolution from the plan,
        including self-referential mutation prompts.
        """
        from ..behaviors.types import BehaviorEvolutionResult

        # Split test cases
        split_idx = int(len(test_cases) * config.train_test_split)
        train_cases = test_cases[:split_idx]
        holdout_cases = test_cases[split_idx:]

        if len(train_cases) < 3 or len(holdout_cases) < 1:
            # Not enough test cases for evolution
            return BehaviorEvolutionResult(
                original_behavior=behavior,
                evolved_behavior=behavior,
                original_fitness=0.0,
                evolved_fitness=0.0,
                overfitting_gap=0.0,
                generations_run=0,
                approved=False,
            )

        # Initialize population
        population = [behavior]
        for _ in range(config.population_size - 1):
            mutant = await self._mutate_behavior(
                behavior,
                self._mutation_prompts[_ % len(self._mutation_prompts)],
            )
            population.append(mutant)

        # Track original fitness
        original_results = await self._run_tests(behavior, train_cases)
        original_fitness = original_results.pass_rate

        best_behavior = behavior
        best_fitness = original_fitness

        # Evolution loop
        generations_without_improvement = 0
        for generation in range(config.generations):
            # Evaluate fitness on train set
            fitness_scores = []
            for variant in population:
                results = await self._run_tests(variant, train_cases)
                variant.metrics.fitness_score = results.pass_rate
                fitness_scores.append(results.pass_rate)

            # Apply fitness sharing for diversity
            self._apply_fitness_sharing(population)

            # Tournament selection
            parents = self._tournament_select(
                population,
                config.tournament_size,
                config.population_size // 2,
            )

            # Create offspring
            offspring = []
            for i in range(0, len(parents) - 1, 2):
                # Crossover
                if len(parents) > i + 1:
                    child = await self._crossover(parents[i], parents[i + 1])
                    offspring.append(child)

                # Mutation
                mutation = self._weighted_random_choice(self._mutation_prompts)
                mutant = await self._mutate_behavior(parents[i], mutation)
                offspring.append(mutant)

            # Elitism + offspring
            population = sorted(
                population,
                key=lambda b: b.metrics.fitness_score,
                reverse=True,
            )[:config.elitism_count] + offspring

            # Trim to population size
            population = population[:config.population_size]

            # Track best
            current_best = max(population, key=lambda b: b.metrics.fitness_score)
            if current_best.metrics.fitness_score > best_fitness:
                best_fitness = current_best.metrics.fitness_score
                best_behavior = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1

            # Early stopping
            if generations_without_improvement >= config.max_generations_without_improvement:
                break

            # Evolve mutation prompts every 2 generations
            if generation > 0 and generation % 2 == 0:
                await self._evolve_mutation_prompts()

        # Validate on holdout
        holdout_results = await self._run_tests(best_behavior, holdout_cases)
        holdout_fitness = holdout_results.pass_rate

        # Check for overfitting
        overfitting_gap = best_fitness - holdout_fitness
        approved = overfitting_gap < config.overfitting_threshold

        return BehaviorEvolutionResult(
            original_behavior=behavior,
            evolved_behavior=best_behavior,
            original_fitness=original_fitness,
            evolved_fitness=holdout_fitness,
            overfitting_gap=overfitting_gap,
            generations_run=generation + 1,
            approved=approved,
        )

    async def _mutate_behavior(
        self,
        behavior: Behavior,
        mutation: MutationPrompt,
    ) -> Behavior:
        """Apply a mutation to a behavior."""
        if not behavior.prompts:
            return behavior

        # Mutate decision prompt
        mutate_prompt = f"""Apply this mutation to the prompt:

MUTATION INSTRUCTION:
{mutation.content}

CURRENT PROMPT:
{behavior.prompts.decision_prompt}

Return the mutated prompt. Apply the mutation instruction but keep the core functionality."""

        mutated_decision = await self._llm.generate(
            mutate_prompt,
            temperature=0.8,  # Higher temp for variety
        )

        # Create new behavior with mutated prompt
        from copy import deepcopy
        mutated = deepcopy(behavior)
        mutated.prompts = BehaviorPrompts(
            decision_prompt=mutated_decision.strip(),
            synthesis_prompt=behavior.prompts.synthesis_prompt,
        )
        mutated.parent_behavior_id = behavior.behavior_id

        return mutated

    async def _crossover(
        self,
        parent1: Behavior,
        parent2: Behavior,
    ) -> Behavior:
        """Create offspring by crossing two behaviors."""
        if not parent1.prompts or not parent2.prompts:
            return parent1

        # Simple crossover: take parts from each parent
        crossover_prompt = f"""Create a combined prompt from these two prompts:

PROMPT 1:
{parent1.prompts.decision_prompt[:500]}

PROMPT 2:
{parent2.prompts.decision_prompt[:500]}

Combine the best elements from both. Keep the core functionality."""

        combined = await self._llm.generate(
            crossover_prompt,
            temperature=0.6,
        )

        from copy import deepcopy
        child = deepcopy(parent1)
        child.prompts = BehaviorPrompts(
            decision_prompt=combined.strip(),
            synthesis_prompt=parent1.prompts.synthesis_prompt,
        )

        return child

    def _apply_fitness_sharing(self, population: list[Behavior]) -> None:
        """Apply fitness sharing to maintain diversity."""
        sigma = 0.3  # Similarity threshold

        for i, b1 in enumerate(population):
            niche_count = 0
            for j, b2 in enumerate(population):
                if i != j and b1.prompts and b2.prompts:
                    similarity = self._compute_prompt_similarity(
                        b1.prompts.decision_prompt,
                        b2.prompts.decision_prompt,
                    )
                    if similarity > sigma:
                        niche_count += 1 - (similarity - sigma) / (1 - sigma)

            # Reduce fitness by niche count
            if niche_count > 0:
                b1.metrics.fitness_score /= (1 + niche_count)

    def _compute_prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Compute similarity between two prompts (0-1)."""
        # Simple word overlap
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _tournament_select(
        self,
        population: list[Behavior],
        tournament_size: int,
        count: int,
    ) -> list[Behavior]:
        """Select parents via tournament selection."""
        import random
        selected = []

        for _ in range(count):
            tournament = random.sample(
                population,
                min(tournament_size, len(population)),
            )
            winner = max(tournament, key=lambda b: b.metrics.fitness_score)
            selected.append(winner)

        return selected

    def _weighted_random_choice(
        self,
        prompts: list[MutationPrompt],
    ) -> MutationPrompt:
        """Choose a mutation prompt weighted by fitness."""
        import random

        # Add small base fitness to avoid zero weights
        weights = [max(0.1, p.fitness) for p in prompts]
        total = sum(weights)
        r = random.uniform(0, total)

        cumulative = 0
        for prompt, weight in zip(prompts, weights):
            cumulative += weight
            if r <= cumulative:
                return prompt

        return prompts[-1]

    async def _evolve_mutation_prompts(self) -> None:
        """Evolve the mutation prompts themselves (self-referential)."""
        # Score mutation prompts by their success rate
        for prompt in self._mutation_prompts:
            if prompt.usage_count > 0:
                prompt.fitness = prompt.success_count / prompt.usage_count

        # Sort by fitness
        self._mutation_prompts.sort(key=lambda p: p.fitness, reverse=True)

        # Keep top half, regenerate bottom half
        half = len(self._mutation_prompts) // 2
        if half > 0:
            top_performers = self._mutation_prompts[:half]

            # Generate new mutations inspired by top performers
            for i in range(half, len(self._mutation_prompts)):
                source = top_performers[i % len(top_performers)]
                new_content = await self._generate_new_mutation(source)
                self._mutation_prompts[i] = MutationPrompt(
                    prompt_id=f"evolved_{i}",
                    content=new_content,
                )

    async def _generate_new_mutation(self, source: MutationPrompt) -> str:
        """Generate a new mutation prompt inspired by a successful one."""
        prompt = f"""Create a new mutation instruction inspired by this successful one:

SUCCESSFUL MUTATION:
{source.content}

Create a DIFFERENT but related mutation instruction. Keep the same spirit but change the approach."""

        response = await self._llm.generate(prompt, temperature=0.9)
        return response.strip()

    # =========================================================================
    # Phase 7: Register
    # =========================================================================

    async def register_behavior(
        self,
        behavior: Behavior,
        tier: BehaviorTier = BehaviorTier.GENERATED,
        status: BehaviorStatus = BehaviorStatus.STAGING,
    ) -> str:
        """Register a behavior in the registry.

        Args:
            behavior: The behavior to register
            tier: The trust tier (default: GENERATED)
            status: The lifecycle status (default: STAGING)

        Returns:
            The registered behavior ID
        """
        behavior.tier = tier
        behavior.status = status
        behavior.is_evolvable = True
        behavior.updated_at = datetime.now()

        if self._registry:
            self._registry.register(behavior)
            self._registry.save_behavior(behavior)

        return behavior.behavior_id

    # =========================================================================
    # Validation
    # =========================================================================

    def validate_behavior(self, behavior: Behavior) -> list[ValidationIssue]:
        """Validate a behavior for completeness and correctness."""
        issues = []

        # Check identity
        if not behavior.behavior_id:
            issues.append(ValidationIssue(
                severity="error",
                message="Behavior ID is required",
                field="behavior_id",
            ))

        if not behavior.name:
            issues.append(ValidationIssue(
                severity="error",
                message="Behavior name is required",
                field="name",
            ))

        # Check actions
        if not behavior.actions:
            issues.append(ValidationIssue(
                severity="warning",
                message="Behavior has no actions",
                field="actions",
            ))

        for action in behavior.actions:
            if not action.description:
                issues.append(ValidationIssue(
                    severity="warning",
                    message=f"Action '{action.name}' has no description",
                    field=f"actions.{action.name}",
                ))

        # Check prompts
        if not behavior.prompts:
            issues.append(ValidationIssue(
                severity="error",
                message="Behavior has no prompts",
                field="prompts",
            ))
        else:
            if not behavior.prompts.decision_prompt:
                issues.append(ValidationIssue(
                    severity="error",
                    message="Decision prompt is required",
                    field="prompts.decision_prompt",
                ))
            if not behavior.prompts.synthesis_prompt:
                issues.append(ValidationIssue(
                    severity="warning",
                    message="Synthesis prompt is empty",
                    field="prompts.synthesis_prompt",
                ))

        # Check test cases
        if not behavior.test_cases:
            issues.append(ValidationIssue(
                severity="warning",
                message="Behavior has no test cases",
                field="test_cases",
            ))
        elif len(behavior.test_cases) < 5:
            issues.append(ValidationIssue(
                severity="info",
                message="Consider adding more test cases (currently {})".format(
                    len(behavior.test_cases)
                ),
                field="test_cases",
            ))

        return issues
