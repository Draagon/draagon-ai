"""Agent context for Draagon AI cognitive engine.

The AgentContext is the central abstraction that replaces global singletons.
It contains everything needed to operate as a specific agent.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

from draagon_ai.core.types import MemoryScope

if TYPE_CHECKING:
    from draagon_ai.core.identity import AgentIdentity


@dataclass
class MemoryScopeConfig:
    """Configuration for what memory scopes an agent can access."""

    # Primary scopes this agent can read/write
    agent_scope: str  # e.g., "agent:roxy"
    user_scope_prefix: str  # e.g., "user:" - combined with user_id

    # Shared scopes (read access, maybe write)
    context_scope: str | None = None  # e.g., "context:mealing_home"
    world_scope: str = "world"  # Usually shared by all

    # What this agent can write to
    can_write_context: bool = True
    can_write_world: bool = False  # Usually only admin/system

    def get_read_scopes(self, user_id: str | None = None) -> list[str]:
        """Get all scopes this agent can read from."""
        scopes = [self.agent_scope, self.world_scope]

        if self.context_scope:
            scopes.append(self.context_scope)

        if user_id:
            scopes.append(f"{self.user_scope_prefix}{user_id}")

        return scopes

    def get_write_scope(self, scope_type: MemoryScope, user_id: str | None = None) -> str | None:
        """Get the scope to write to for a given type."""
        if scope_type == MemoryScope.AGENT:
            return self.agent_scope
        elif scope_type == MemoryScope.USER and user_id:
            return f"{self.user_scope_prefix}{user_id}"
        elif scope_type == MemoryScope.CONTEXT and self.can_write_context:
            return self.context_scope
        elif scope_type == MemoryScope.WORLD and self.can_write_world:
            return self.world_scope
        return None


@dataclass
class AgentConfig:
    """Configuration for agent behavior."""

    # LLM settings
    llm_provider: str = "groq"  # "groq", "openai", "anthropic", "ollama"
    llm_model: str = "llama-3.3-70b-versatile"
    llm_fast_model: str = "llama-3.1-8b-instant"

    # Embedding settings
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"

    # Behavior settings
    max_response_length: int = 100  # Default word limit for responses
    enable_curiosity: bool = True
    enable_learning: bool = True
    enable_opinions: bool = True

    # Rate limiting
    max_questions_per_day: int = 3
    min_question_gap_minutes: int = 30

    # Custom prompt overrides (if any)
    prompt_overrides: dict[str, str] = field(default_factory=dict)


@dataclass
class SessionContext:
    """Context for a single conversation session."""

    session_id: str
    user_id: str
    started_at: datetime = field(default_factory=datetime.now)

    # Conversation state
    conversation_history: list[dict[str, Any]] = field(default_factory=list)
    pending_details: str | None = None  # For "tell me more" feature

    # Session-specific state
    session_memories: list[str] = field(default_factory=list)  # Memory IDs accessed
    tools_used: list[str] = field(default_factory=list)

    # Area/device context (for voice assistants)
    area_id: str | None = None
    device_id: str | None = None


@dataclass
class AgentContext:
    """Everything needed to operate as a specific agent.

    This is the central abstraction that gets passed through the cognitive
    pipeline instead of relying on global singletons.
    """

    # Agent identification
    agent_id: str  # "roxy", "max", "dungeon_master"
    agent_name: str  # Display name

    # Agent identity (personality, beliefs, values)
    identity: "AgentIdentity"

    # Memory access configuration
    memory_scope: MemoryScopeConfig

    # Agent configuration
    config: AgentConfig = field(default_factory=AgentConfig)

    # Runtime state (not persisted)
    session: SessionContext | None = None

    @classmethod
    def create(
        cls,
        agent_id: str,
        identity: "AgentIdentity",
        context_id: str | None = None,
        config: AgentConfig | None = None,
    ) -> "AgentContext":
        """Create an agent context with proper scoping.

        Args:
            agent_id: Unique identifier for this agent
            identity: The agent's identity (personality, beliefs, etc.)
            context_id: Optional shared context (e.g., "mealing_home", "dragon_quest")
            config: Optional agent configuration
        """
        memory_scope = MemoryScopeConfig(
            agent_scope=f"agent:{agent_id}",
            user_scope_prefix="user:",
            context_scope=f"context:{context_id}" if context_id else None,
        )

        return cls(
            agent_id=agent_id,
            agent_name=identity.name,
            identity=identity,
            memory_scope=memory_scope,
            config=config or AgentConfig(),
        )

    def with_session(
        self,
        session_id: str,
        user_id: str,
        area_id: str | None = None,
        device_id: str | None = None,
    ) -> "AgentContext":
        """Create a new context with session information.

        Returns a new AgentContext with the session attached.
        Does not modify the original.
        """
        session = SessionContext(
            session_id=session_id,
            user_id=user_id,
            area_id=area_id,
            device_id=device_id,
        )

        return AgentContext(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            identity=self.identity,
            memory_scope=self.memory_scope,
            config=self.config,
            session=session,
        )

    @property
    def user_id(self) -> str | None:
        """Get current user ID from session."""
        return self.session.user_id if self.session else None

    @property
    def conversation_id(self) -> str | None:
        """Get current session/conversation ID."""
        return self.session.session_id if self.session else None
