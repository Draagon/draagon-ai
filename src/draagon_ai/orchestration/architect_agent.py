"""Behavior Architect Agent integration.

Provides a specialized agent that uses the Behavior Architect to create
new behaviors through natural conversation.

Usage:
    from draagon_ai.orchestration import create_architect_agent

    architect = await create_architect_agent(
        llm=my_llm,
        web_search=my_search,
        registry=my_registry,
    )

    # Create a behavior through conversation
    response = await architect.create_behavior_interactive(
        "Create a behavior for managing a home aquarium"
    )
"""

from dataclasses import dataclass
from typing import Any

from ..behaviors import Behavior, BehaviorRegistry
from ..behaviors.templates import BEHAVIOR_ARCHITECT_TEMPLATE
from ..services import BehaviorArchitectService
from .agent import Agent, AgentConfig
from .protocols import LLMProvider, MemoryProvider, ToolProvider


@dataclass
class ArchitectResult:
    """Result from behavior creation."""

    behavior: Behavior | None
    success: bool
    message: str
    phases_completed: list[str]
    test_pass_rate: float = 0.0
    evolution_applied: bool = False


class ArchitectAgent(Agent):
    """Specialized agent for creating behaviors.

    Extends the base Agent with behavior creation capabilities.
    The agent uses the Behavior Architect behavior to guide
    conversation about what behavior to create.
    """

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        tools: ToolProvider | None = None,
        behavior_registry: BehaviorRegistry | None = None,
        web_search: Any = None,
    ):
        """Initialize the architect agent.

        Args:
            llm: LLM provider for reasoning
            memory: Optional memory provider
            tools: Optional tool provider
            behavior_registry: Registry for storing created behaviors
            web_search: Optional web search for research phase
        """
        # Initialize with Behavior Architect behavior
        super().__init__(
            config=AgentConfig(
                agent_id="behavior_architect",
                name="Behavior Architect",
                personality_intro="""You are the Behavior Architect, a meta-agent that creates AI behaviors.
You can design, build, test, and evolve behaviors from natural language descriptions.
Guide the user through the behavior creation process, asking clarifying questions
when needed and explaining what you're doing at each phase.""",
                enable_learning=False,  # Architect doesn't need learning
            ),
            behavior=BEHAVIOR_ARCHITECT_TEMPLATE,
            llm=llm,
            memory=memory,
            tools=tools,
            behavior_registry=behavior_registry,
        )

        # Create the service that does the actual work
        self._architect_service = BehaviorArchitectService(
            llm=llm,
            web_search=web_search,
            registry=behavior_registry,
        )

        # Track creation state
        self._current_design: Any = None
        self._current_behavior: Behavior | None = None
        self._phases_completed: list[str] = []

    @property
    def architect_service(self) -> BehaviorArchitectService:
        """Get the underlying architect service."""
        return self._architect_service

    async def create_behavior(
        self,
        description: str,
        user_constraints: dict[str, Any] | None = None,
        evolve: bool = False,
    ) -> ArchitectResult:
        """Create a behavior from a description (non-interactive).

        This is a direct creation path that doesn't require conversation.
        Use this when you have a clear description and constraints.

        Args:
            description: Natural language description of the behavior
            user_constraints: Optional constraints
            evolve: Whether to evolve the behavior after creation

        Returns:
            ArchitectResult with the created behavior
        """
        try:
            behavior = await self._architect_service.create_behavior(
                description=description,
                user_constraints=user_constraints,
                evolve=evolve,
            )

            test_pass_rate = 0.0
            if behavior.test_results:
                test_pass_rate = behavior.test_results.pass_rate

            return ArchitectResult(
                behavior=behavior,
                success=True,
                message=f"Created behavior '{behavior.name}' with {len(behavior.actions)} actions",
                phases_completed=["research", "design", "build", "test", "register"],
                test_pass_rate=test_pass_rate,
                evolution_applied=evolve,
            )

        except Exception as e:
            return ArchitectResult(
                behavior=None,
                success=False,
                message=f"Failed to create behavior: {str(e)}",
                phases_completed=self._phases_completed,
            )

    async def research(self, description: str) -> dict[str, Any]:
        """Research a domain for behavior creation.

        Args:
            description: What kind of behavior to research

        Returns:
            Research results dict
        """
        research = await self._architect_service.research_domain(description)
        self._phases_completed.append("research")

        return {
            "domain": research.domain,
            "core_tasks": research.core_tasks,
            "suggested_actions": research.suggested_actions,
            "triggers": research.suggested_triggers,
            "constraints": research.constraints,
            "knowledge": research.domain_knowledge,
            "sources": research.sources,
        }

    async def design(
        self,
        research_results: dict[str, Any] | None = None,
        description: str = "",
        user_constraints: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Design a behavior structure.

        Args:
            research_results: Results from research phase
            description: Behavior description (if no research)
            user_constraints: Optional user constraints

        Returns:
            Design dict
        """
        from ..behaviors.types import DomainResearchResult

        if research_results:
            research = DomainResearchResult(
                domain=research_results.get("domain", ""),
                core_tasks=research_results.get("core_tasks", []),
                suggested_actions=research_results.get("suggested_actions", []),
                suggested_triggers=research_results.get("triggers", []),
                constraints=research_results.get("constraints", []),
                domain_knowledge=research_results.get("knowledge", ""),
                sources=research_results.get("sources", []),
            )
        else:
            research = await self._architect_service.research_domain(description)

        design = await self._architect_service.design_behavior(
            research, user_constraints
        )
        self._current_design = design
        self._phases_completed.append("design")

        return {
            "behavior_id": design.behavior_id,
            "name": design.name,
            "description": design.description,
            "actions": [
                {"name": a.name, "description": a.description}
                for a in design.actions
            ],
            "triggers": [t.name for t in design.triggers],
        }

    async def build(self) -> dict[str, Any]:
        """Build the behavior from the current design.

        Returns:
            Build results dict
        """
        if not self._current_design:
            raise ValueError("No design available. Run design() first.")

        behavior, test_cases = await self._architect_service.build_behavior(
            self._current_design
        )
        self._current_behavior = behavior
        self._phases_completed.append("build")

        return {
            "behavior_id": behavior.behavior_id,
            "name": behavior.name,
            "actions": len(behavior.actions),
            "test_cases": len(test_cases),
            "has_prompts": behavior.prompts is not None,
        }

    async def test(self) -> dict[str, Any]:
        """Test the current behavior.

        Returns:
            Test results dict
        """
        if not self._current_behavior:
            raise ValueError("No behavior available. Run build() first.")

        behavior = await self._architect_service.test_and_iterate(
            self._current_behavior,
            self._current_behavior.test_cases,
        )
        self._current_behavior = behavior
        self._phases_completed.append("test")

        results = behavior.test_results
        return {
            "pass_rate": results.pass_rate if results else 0.0,
            "passed": results.passed if results else 0,
            "failed": results.failed if results else 0,
            "total": results.total_tests if results else 0,
            "status": behavior.status.value,
        }

    async def register(self) -> dict[str, Any]:
        """Register the current behavior.

        Returns:
            Registration results dict
        """
        if not self._current_behavior:
            raise ValueError("No behavior available. Run build() first.")

        behavior_id = await self._architect_service.register_behavior(
            self._current_behavior
        )
        self._phases_completed.append("register")

        return {
            "behavior_id": behavior_id,
            "tier": self._current_behavior.tier.value,
            "status": self._current_behavior.status.value,
        }

    def get_current_behavior(self) -> Behavior | None:
        """Get the behavior currently being created."""
        return self._current_behavior

    def reset(self) -> None:
        """Reset the creation state."""
        self._current_design = None
        self._current_behavior = None
        self._phases_completed = []


async def create_architect_agent(
    llm: LLMProvider,
    web_search: Any = None,
    registry: BehaviorRegistry | None = None,
    memory: MemoryProvider | None = None,
    tools: ToolProvider | None = None,
) -> ArchitectAgent:
    """Create a Behavior Architect agent.

    Factory function for creating an ArchitectAgent with all dependencies.

    Args:
        llm: LLM provider for reasoning
        web_search: Optional web search provider for research
        registry: Optional behavior registry
        memory: Optional memory provider
        tools: Optional tool provider

    Returns:
        Configured ArchitectAgent

    Example:
        architect = await create_architect_agent(
            llm=GroqLLM(),
            web_search=SearXNG(),
            registry=BehaviorRegistry(),
        )

        result = await architect.create_behavior(
            "A behavior for managing kitchen timers"
        )

        if result.success:
            print(f"Created: {result.behavior.name}")
    """
    return ArchitectAgent(
        llm=llm,
        web_search=web_search,
        behavior_registry=registry,
        memory=memory,
        tools=tools,
    )
