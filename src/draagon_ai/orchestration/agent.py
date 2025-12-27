"""Agent class - the main entry point for using draagon-ai.

The Agent class combines:
- Behavior: What the agent can do
- Persona: Who the agent is
- Tools: How the agent interacts with the world
- Cognition: How the agent learns and remembers
"""

from dataclasses import dataclass, field
from typing import Any

from ..behaviors import Behavior, BehaviorRegistry
from .protocols import LLMProvider, MemoryProvider, ToolProvider
from .loop import AgentContext, AgentLoop, AgentResponse
from .decision import DecisionEngine
from .execution import ActionExecutor


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    # Identity
    agent_id: str
    name: str = ""

    # Personality introduction (injected into prompts)
    personality_intro: str = ""

    # Default model tier
    default_model_tier: str = "local"

    # Feature flags
    enable_learning: bool = True
    enable_proactive: bool = False

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class Agent:
    """An AI agent with behavior, personality, and tools.

    The Agent class is the main entry point for using draagon-ai.
    It combines all the components needed to create a functional agent.

    Example:
        from draagon_ai.orchestration import Agent, AgentConfig
        from draagon_ai.behaviors import VOICE_ASSISTANT_TEMPLATE

        agent = Agent(
            config=AgentConfig(
                agent_id="assistant",
                name="Assistant",
                personality_intro="You are a helpful voice assistant.",
            ),
            behavior=VOICE_ASSISTANT_TEMPLATE,
            llm=my_llm_provider,
            memory=my_memory_provider,
            tools=my_tool_provider,
        )

        response = await agent.process(
            query="What time is it?",
            user_id="doug",
        )
    """

    def __init__(
        self,
        config: AgentConfig,
        behavior: Behavior,
        llm: LLMProvider,
        memory: MemoryProvider | None = None,
        tools: ToolProvider | None = None,
        behavior_registry: BehaviorRegistry | None = None,
    ):
        """Initialize an agent.

        Args:
            config: Agent configuration
            behavior: Primary behavior for this agent
            llm: LLM provider for reasoning
            memory: Optional memory provider for context
            tools: Optional tool provider for actions
            behavior_registry: Optional registry for multi-behavior support
        """
        self.config = config
        self.behavior = behavior
        self.llm = llm
        self.memory = memory
        self.tools = tools
        self.registry = behavior_registry

        # Create internal components
        self._decision_engine = DecisionEngine(
            llm=llm,
            default_model_tier=config.default_model_tier,
        )

        self._action_executor = ActionExecutor(tools) if tools else None

        self._loop = AgentLoop(
            llm=llm,
            memory=memory,
            decision_engine=self._decision_engine,
            action_executor=self._action_executor,
        )

        # State
        self._sessions: dict[str, AgentContext] = {}

    @property
    def agent_id(self) -> str:
        """Get the agent's ID."""
        return self.config.agent_id

    @property
    def name(self) -> str:
        """Get the agent's name."""
        return self.config.name or self.config.agent_id

    async def process(
        self,
        query: str,
        user_id: str,
        session_id: str | None = None,
        area_id: str | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> AgentResponse:
        """Process a user query.

        This is the main entry point for agent interaction.

        Args:
            query: User's query text
            user_id: User identifier
            session_id: Optional session ID for conversation continuity
            area_id: Optional area/room ID for context
            debug: Include debug info in response
            **kwargs: Additional context

        Returns:
            AgentResponse with the result
        """
        # Get or create session context
        session_key = session_id or f"{user_id}_default"
        context = self._get_or_create_context(
            session_key=session_key,
            user_id=user_id,
            session_id=session_id,
            area_id=area_id,
            debug=debug,
        )

        # Process through the loop
        response = await self._loop.process(
            query=query,
            behavior=self.behavior,
            context=context,
            assistant_intro=self.config.personality_intro,
        )

        # Update conversation history
        self._update_history(context, query, response.response)

        # Store pending details for "tell me more"
        if response.full_response:
            context.pending_details = response.full_response

        return response

    async def set_behavior(self, behavior: Behavior) -> None:
        """Change the agent's behavior.

        Args:
            behavior: New behavior to use
        """
        self.behavior = behavior

    async def get_behavior(self) -> Behavior:
        """Get the current behavior.

        Returns:
            Current behavior
        """
        return self.behavior

    def get_session(self, session_id: str) -> AgentContext | None:
        """Get an existing session context.

        Args:
            session_id: Session identifier

        Returns:
            Session context or None
        """
        return self._sessions.get(session_id)

    def clear_session(self, session_id: str) -> bool:
        """Clear a session's conversation history.

        Args:
            session_id: Session to clear

        Returns:
            True if session existed and was cleared
        """
        if session_id in self._sessions:
            self._sessions[session_id].conversation_history.clear()
            self._sessions[session_id].pending_details = None
            return True
        return False

    def _get_or_create_context(
        self,
        session_key: str,
        user_id: str,
        session_id: str | None,
        area_id: str | None,
        debug: bool,
    ) -> AgentContext:
        """Get or create a session context.

        Args:
            session_key: Key for session lookup
            user_id: User identifier
            session_id: Session identifier
            area_id: Area/room identifier
            debug: Debug mode flag

        Returns:
            AgentContext for this session
        """
        if session_key not in self._sessions:
            self._sessions[session_key] = AgentContext(
                user_id=user_id,
                session_id=session_id or session_key,
                area_id=area_id,
                debug=debug,
            )
        else:
            # Update mutable fields
            ctx = self._sessions[session_key]
            ctx.area_id = area_id
            ctx.debug = debug

        return self._sessions[session_key]

    def _update_history(
        self,
        context: AgentContext,
        query: str,
        response: str,
    ) -> None:
        """Update conversation history with a new turn.

        Args:
            context: Session context
            query: User's query
            response: Agent's response
        """
        context.conversation_history.append({
            "user": query,
            "assistant": response,
        })

        # Limit history size
        max_history = 10
        if len(context.conversation_history) > max_history:
            context.conversation_history = context.conversation_history[-max_history:]


class MultiAgent:
    """Manager for multiple agents.

    Use this when you need multiple agents with different behaviors
    that can collaborate or be selected based on context.

    Example:
        multi = MultiAgent(
            llm=my_llm,
            registry=my_registry,
        )

        multi.add_agent("assistant", VOICE_ASSISTANT_TEMPLATE)
        multi.add_agent("dm", DUNGEON_MASTER_TEMPLATE)

        # Select agent based on query
        response = await multi.process(
            query="Roll initiative!",
            user_id="player1",
        )
    """

    def __init__(
        self,
        llm: LLMProvider,
        registry: BehaviorRegistry,
        memory: MemoryProvider | None = None,
        tools: ToolProvider | None = None,
    ):
        """Initialize multi-agent manager.

        Args:
            llm: LLM provider
            registry: Behavior registry
            memory: Optional memory provider
            tools: Optional tool provider
        """
        self.llm = llm
        self.registry = registry
        self.memory = memory
        self.tools = tools
        self._agents: dict[str, Agent] = {}
        self._default_agent: str | None = None

    def add_agent(
        self,
        agent_id: str,
        behavior: Behavior,
        personality_intro: str = "",
        set_default: bool = False,
    ) -> Agent:
        """Add an agent with a specific behavior.

        Args:
            agent_id: Unique agent identifier
            behavior: Behavior for this agent
            personality_intro: Personality introduction
            set_default: Set as default agent

        Returns:
            Created Agent instance
        """
        agent = Agent(
            config=AgentConfig(
                agent_id=agent_id,
                name=behavior.name,
                personality_intro=personality_intro,
            ),
            behavior=behavior,
            llm=self.llm,
            memory=self.memory,
            tools=self.tools,
            behavior_registry=self.registry,
        )

        self._agents[agent_id] = agent

        if set_default or self._default_agent is None:
            self._default_agent = agent_id

        return agent

    def get_agent(self, agent_id: str) -> Agent | None:
        """Get an agent by ID.

        Args:
            agent_id: Agent identifier

        Returns:
            Agent or None
        """
        return self._agents.get(agent_id)

    async def process(
        self,
        query: str,
        user_id: str,
        agent_id: str | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """Process a query with an agent.

        If agent_id is not specified, uses the default agent
        or selects based on behavior triggers.

        Args:
            query: User's query
            user_id: User identifier
            agent_id: Optional specific agent to use
            **kwargs: Additional context

        Returns:
            AgentResponse from the selected agent
        """
        # Select agent
        if agent_id:
            agent = self._agents.get(agent_id)
            if not agent:
                raise ValueError(f"Unknown agent: {agent_id}")
        elif self._default_agent:
            agent = self._agents[self._default_agent]
        else:
            raise ValueError("No agents registered")

        return await agent.process(query=query, user_id=user_id, **kwargs)

    async def select_agent(self, query: str, context: dict) -> Agent | None:
        """Select the best agent for a query based on triggers.

        Args:
            query: User's query
            context: Additional context

        Returns:
            Best matching agent or None
        """
        # Find behaviors matching the query
        matching = self.registry.find_by_trigger(query, context)

        if not matching:
            return self._agents.get(self._default_agent) if self._default_agent else None

        # Find agent with matching behavior
        for behavior in matching:
            for agent_id, agent in self._agents.items():
                if agent.behavior.behavior_id == behavior.behavior_id:
                    return agent

        return self._agents.get(self._default_agent) if self._default_agent else None
