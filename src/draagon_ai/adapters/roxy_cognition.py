"""Roxy Cognition Adapters.

Adapters that allow Roxy to use draagon-ai's cognitive services while providing
Roxy-specific LLM, Memory, Credibility, and Trait implementations.

REQ-003-01: Belief reconciliation using core service.
REQ-003-02: Curiosity engine using core service.
REQ-003-03: Opinion formation using core service.
REQ-003-04: Learning service using core service.
"""

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from draagon_ai.cognition.beliefs import (
    BeliefReconciliationService,
    CredibilityProvider,
    ReconciliationResult,
)
from draagon_ai.cognition.curiosity import (
    CuriosityEngine,
    CuriousQuestion,
    KnowledgeGap,
    QuestionPriority,
    QuestionPurpose,
    QuestionType,
    TraitProvider,
)
from draagon_ai.cognition.learning import (
    CredibilityProvider as LearningCredibilityProvider,
    FailureType,
    LearningCandidate,
    LearningResult,
    LearningService,
    LearningType,
    MemoryAction,
    SearchProvider,
    SkillConfidence,
    UserProvider,
    VerificationResult,
)
from draagon_ai.cognition.opinions import (
    FormedOpinion,
    IdentityManager,
    OpinionBasis,
    OpinionFormationService,
    OpinionRequest,
    OpinionStrength,
)
from draagon_ai.core import (
    AgentIdentity,
    CoreValue,
    GuidingPrinciple,
    Opinion,
    PersonalityTrait,
    Preference,
    WorldviewBelief,
)
from draagon_ai.core.types import (
    AgentBelief,
    BeliefType,
    ObservationScope,
    UserObservation,
)
from draagon_ai.llm import LLMProvider, ChatResponse, ModelTier
from draagon_ai.memory import MemoryProvider, MemoryType, MemoryScope, Memory, SearchResult

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols for Roxy Services
# =============================================================================


@runtime_checkable
class RoxyLLMService(Protocol):
    """Protocol for Roxy's LLM service.

    Roxy's LLMService has a different interface than draagon-ai's LLMProvider.
    This adapter bridges the gap.
    """

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.1,
    ) -> dict[str, Any] | None:
        """Execute a chat completion expecting JSON output.

        Returns:
            Dict with 'content' and 'parsed' keys, or None on failure.
        """
        ...

    async def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str | None:
        """Execute a chat completion.

        Returns:
            The response content string, or None on failure.
        """
        ...


@runtime_checkable
class RoxyMemoryService(Protocol):
    """Protocol for Roxy's Memory service.

    Roxy's MemoryService uses a different interface with user_id, scope strings,
    and different search/store methods.
    """

    async def store(
        self,
        content: str,
        user_id: str,
        scope: str = "system",
        memory_type: Any = None,  # Roxy's MemoryType enum
        importance: float = 0.5,
        entities: list[str] | None = None,
        conversation_id: str | None = None,
        source_user_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Store a memory."""
        ...

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
        include_knowledge: bool = False,
    ) -> list[dict[str, Any]]:
        """Search memories."""
        ...


@runtime_checkable
class RoxyUserService(Protocol):
    """Protocol for Roxy's User service (credibility provider)."""

    def get_user_credibility(self, user_id: str) -> Any | None:
        """Get user credibility object.

        Returns object with 'credibility' float attribute, or None.
        """
        ...


@runtime_checkable
class RoxySelfManager(Protocol):
    """Protocol for Roxy's RoxySelfManager (trait provider).

    Roxy's RoxySelfManager provides trait values and worldview information.
    """

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a trait value with default fallback.

        Args:
            trait_name: Name of the trait (e.g., 'curiosity_intensity')
            default: Default value if trait not found

        Returns:
            Trait value 0.0-1.0
        """
        ...

    async def get_worldview_string(self) -> str:
        """Get a string representation of the agent's worldview.

        Returns:
            Human-readable worldview string
        """
        ...


@runtime_checkable
class RoxySearchService(Protocol):
    """Protocol for Roxy's web search service.

    Roxy's SearchService uses SearXNG for web search.
    """

    async def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search the web using SearXNG.

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of search results with title, snippet, url
        """
        ...


@runtime_checkable
class RoxyFullUserService(Protocol):
    """Protocol for Roxy's full UserService (for learning adapter).

    This extends RoxyUserService with additional methods needed for LearningService.
    """

    def get_user_credibility(self, user_id: str) -> Any | None:
        """Get user credibility object."""
        ...

    def should_verify_correction(
        self,
        user_id: str,
        domain: str | None = None,
    ) -> tuple[bool, float]:
        """Determine if a correction should be verified."""
        ...

    def record_correction_result(
        self,
        user_id: str,
        result: str,
        domain: str | None = None,
        user_was_confident: bool = False,
    ) -> dict[str, Any]:
        """Record the result of a correction verification."""
        ...

    async def get_user(self, user_id: str) -> Any | None:
        """Get user by ID."""
        ...

    def get_user_sync(self, user_id: str) -> Any | None:
        """Get user by ID (synchronous version)."""
        ...


# =============================================================================
# LLM Adapter
# =============================================================================


class RoxyLLMAdapter(LLMProvider):
    """Adapts Roxy's LLMService to draagon-ai's LLMProvider protocol.

    Roxy's LLM service uses different method signatures, so this adapter
    translates between them.
    """

    def __init__(self, roxy_llm: RoxyLLMService):
        """Initialize the adapter.

        Args:
            roxy_llm: Roxy's LLMService instance
        """
        self._llm = roxy_llm

    async def chat(
        self,
        messages: list[dict[str, Any]] | list[Any],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tools: list[Any] | None = None,
        tier: ModelTier = ModelTier.LOCAL,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResponse:
        """Execute a chat completion using Roxy's LLM service.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tools: Tool definitions (not used for belief reconciliation)
            tier: Model tier (ignored, Roxy handles routing)
            response_format: Response format (ignored)

        Returns:
            ChatResponse with content
        """
        # Convert messages to simple dicts if needed
        msg_list = []
        for m in messages:
            if isinstance(m, dict):
                msg_list.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                # Assume it has role and content attributes
                msg_list.append({"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")})

        # Prepend system prompt if provided
        if system_prompt:
            msg_list.insert(0, {"role": "system", "content": system_prompt})

        # Call Roxy's LLM
        result = await self._llm.chat(
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return ChatResponse(
            content=result or "",
            tool_calls=[],
            usage=None,
            model=None,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]] | list[Any],
        *,
        system_prompt: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        tier: ModelTier = ModelTier.LOCAL,
    ) -> AsyncIterator[str]:
        """Stream a chat completion (falls back to non-streaming for Roxy).

        Roxy's LLM service doesn't always support streaming, so we call
        the regular chat method and yield the full response.

        Args:
            messages: Conversation messages
            system_prompt: Optional system prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            tier: Model tier (ignored)

        Yields:
            The full response as a single chunk
        """
        response = await self.chat(
            messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            tier=tier,
        )
        yield response.content


# =============================================================================
# Memory Adapter
# =============================================================================


class RoxyMemoryAdapter(MemoryProvider):
    """Adapts Roxy's MemoryService to draagon-ai's MemoryProvider protocol.

    Roxy's memory service uses user_id and scope strings, while draagon-ai
    uses agent_id, scopes, and enum types.
    """

    # Map draagon-ai MemoryScope to Roxy scope strings
    SCOPE_MAPPING = {
        MemoryScope.AGENT: "system",
        MemoryScope.USER: "user",
        MemoryScope.CONTEXT: "household",
        MemoryScope.WORLD: "system",
        MemoryScope.SESSION: "user",
    }

    # Map draagon-ai MemoryType to Roxy memory type names
    TYPE_MAPPING = {
        MemoryType.FACT: "fact",
        MemoryType.SKILL: "skill",
        MemoryType.PREFERENCE: "preference",
        MemoryType.EPISODIC: "episodic",
        MemoryType.INSIGHT: "insight",
        MemoryType.INSTRUCTION: "instruction",
        MemoryType.OBSERVATION: "fact",  # Roxy stores observations as facts
        MemoryType.BELIEF: "fact",  # Roxy stores beliefs as facts
    }

    def __init__(self, roxy_memory: RoxyMemoryService, agent_id: str = "roxy"):
        """Initialize the adapter.

        Args:
            roxy_memory: Roxy's MemoryService instance
            agent_id: Agent ID (used as user_id prefix in Roxy)
        """
        self._memory = roxy_memory
        self._agent_id = agent_id

    async def initialize(self) -> None:
        """Initialize the memory provider (no-op for Roxy)."""
        pass

    async def store(
        self,
        content: str,
        memory_type: MemoryType,
        scope: MemoryScope,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        importance: float = 0.5,
        confidence: float = 1.0,
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory:
        """Store a memory using Roxy's MemoryService.

        Args:
            content: Memory content
            memory_type: Type of memory
            scope: Visibility scope
            agent_id: Agent ID (maps to user_id in Roxy)
            user_id: User ID
            context_id: Context/conversation ID
            importance: Importance score
            confidence: Confidence score
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            Memory object (simplified)
        """
        # Determine Roxy user_id: use agent namespace for agent-scoped memories
        roxy_user_id = f"{self._agent_id}_system" if scope == MemoryScope.AGENT else (user_id or self._agent_id)

        # Map scope and type
        roxy_scope = self.SCOPE_MAPPING.get(scope, "system")
        roxy_type = self.TYPE_MAPPING.get(memory_type, "fact")

        # Get Roxy's MemoryType enum
        try:
            from roxy.services.memory import MemoryType as RoxyMemoryType
            type_enum = getattr(RoxyMemoryType, roxy_type.upper(), RoxyMemoryType.FACT)
        except ImportError:
            type_enum = None

        # Call Roxy's store
        await self._memory.store(
            content=content,
            user_id=roxy_user_id,
            scope=roxy_scope,
            memory_type=type_enum,
            importance=importance,
            entities=entities or [],
            conversation_id=context_id,
            metadata=metadata or {},
        )

        # Return a simplified Memory object
        return Memory(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=memory_type,
            scope=scope,
            agent_id=agent_id or self._agent_id,
            user_id=user_id,
            context_id=context_id,
            importance=importance,
            confidence=confidence,
            entities=entities or [],
        )

    async def search(
        self,
        query: str,
        *,
        agent_id: str | None = None,
        user_id: str | None = None,
        context_id: str | None = None,
        memory_types: list[MemoryType] | None = None,
        scopes: list[MemoryScope] | None = None,
        limit: int = 5,
        min_score: float | None = None,
    ) -> list[SearchResult]:
        """Search memories using Roxy's MemoryService.

        Args:
            query: Search query
            agent_id: Agent ID
            user_id: User ID
            context_id: Context ID
            memory_types: Memory types to filter by
            scopes: Scopes to search
            limit: Maximum results
            min_score: Minimum similarity score

        Returns:
            List of SearchResult objects
        """
        # Determine Roxy user_id
        roxy_user_id = user_id or f"{self._agent_id}_system"

        # Call Roxy's search
        results = await self._memory.search(
            query=query,
            user_id=roxy_user_id,
            limit=limit,
            include_knowledge=False,
        )

        # Convert to SearchResult objects
        search_results = []
        for r in results:
            metadata = r.get("metadata", {})

            # Create Memory object from result
            memory = Memory(
                id=metadata.get("memory_id", str(uuid.uuid4())),
                content=r.get("content", ""),
                memory_type=MemoryType.FACT,  # Default type
                scope=MemoryScope.AGENT,
                agent_id=agent_id or self._agent_id,
                user_id=r.get("source_user_id") or metadata.get("source_user_id"),
                importance=r.get("importance", 0.5),
                confidence=metadata.get("confidence", 0.8),
                entities=r.get("entities", []),
            )

            search_results.append(SearchResult(
                memory=memory,
                score=r.get("score", 0.8),
            ))

        return search_results

    async def get(self, memory_id: str) -> Memory | None:
        """Get a specific memory by ID (not implemented for Roxy)."""
        logger.warning("RoxyMemoryAdapter.get() not implemented")
        return None

    async def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        importance: float | None = None,
        confidence: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Memory | None:
        """Update a memory (not implemented for Roxy)."""
        logger.warning("RoxyMemoryAdapter.update() not implemented")
        return None

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory (not implemented for Roxy)."""
        logger.warning("RoxyMemoryAdapter.delete() not implemented")
        return False


# =============================================================================
# Credibility Adapter
# =============================================================================


class RoxyCredibilityAdapter(CredibilityProvider):
    """Adapts Roxy's UserService to draagon-ai's CredibilityProvider protocol.

    Roxy tracks multi-dimensional credibility per user. This adapter extracts
    the composite credibility score.
    """

    def __init__(self, roxy_user_service: RoxyUserService):
        """Initialize the adapter.

        Args:
            roxy_user_service: Roxy's UserService instance
        """
        self._user_service = roxy_user_service

    def get_user_credibility(self, user_id: str) -> float | None:
        """Get user credibility score.

        Args:
            user_id: User ID to look up

        Returns:
            Credibility score 0.0-1.0, or None if unknown
        """
        if user_id in ("unknown", "system", "roxy_system"):
            return None

        try:
            credibility = self._user_service.get_user_credibility(user_id)
            if credibility:
                # Roxy's UserCredibility has a 'credibility' attribute
                return getattr(credibility, "credibility", 0.7)
        except Exception as e:
            logger.warning(f"Failed to get credibility for {user_id}: {e}")

        return None


# =============================================================================
# Main Adapter
# =============================================================================


@dataclass
class RoxyBeliefAdapter:
    """Adapter that allows Roxy to use draagon-ai's BeliefReconciliationService.

    This is the main entry point for REQ-003-01. It wraps draagon-ai's belief
    reconciliation service and provides Roxy-compatible methods.

    Example:
        from roxy.services.llm import LLMService
        from roxy.services.memory import MemoryService
        from roxy.services.users import get_user_service

        adapter = RoxyBeliefAdapter(
            llm=LLMService(),
            memory=MemoryService(),
            user_service=get_user_service(),
        )

        result = await adapter.reconcile(
            observation="We have 6 cats",
            user_id="doug",
        )
    """

    llm: RoxyLLMService
    memory: RoxyMemoryService
    user_service: RoxyUserService | None = None
    agent_name: str = "Roxy"
    agent_id: str = "roxy"

    _service: BeliefReconciliationService | None = None

    def _get_service(self) -> BeliefReconciliationService:
        """Get or create the underlying service."""
        if self._service is None:
            # Create adapters
            llm_adapter = RoxyLLMAdapter(self.llm)
            memory_adapter = RoxyMemoryAdapter(self.memory, agent_id=self.agent_id)

            # Create credibility provider if user service provided
            credibility = None
            if self.user_service:
                credibility = RoxyCredibilityAdapter(self.user_service)

            # Create the service
            self._service = BeliefReconciliationService(
                llm=llm_adapter,
                memory=memory_adapter,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                credibility_provider=credibility,
            )

        return self._service

    # =========================================================================
    # Roxy-Compatible Methods
    # =========================================================================

    async def create_observation(
        self,
        statement: str,
        user_id: str,
        context: str | None = None,
        conversation_id: str | None = None,
    ) -> UserObservation:
        """Create an observation from a user statement.

        Args:
            statement: What the user said
            user_id: Who said it
            context: Conversation context
            conversation_id: Conversation ID

        Returns:
            The created observation
        """
        return await self._get_service().create_observation(
            statement=statement,
            user_id=user_id,
            context=context,
            conversation_id=conversation_id,
        )

    async def reconcile(
        self,
        observation: str,
        user_id: str,
        context: str | None = None,
    ) -> ReconciliationResult | None:
        """Reconcile an observation into a belief.

        This is the main entry point for belief reconciliation.
        It creates an observation and triggers reconciliation.

        Args:
            observation: The user's statement
            user_id: Who said it
            context: Conversation context

        Returns:
            ReconciliationResult with the formed belief
        """
        # Create observation (this also triggers reconciliation)
        await self.create_observation(
            statement=observation,
            user_id=user_id,
            context=context,
        )

        # Return the latest reconciliation for this topic
        return await self._get_service().reconcile_topic(
            topic=observation,
            user_id=user_id,
        )

    async def get_belief(self, topic: str) -> AgentBelief | None:
        """Get the agent's belief about a topic.

        Args:
            topic: What to look up

        Returns:
            AgentBelief if found, None otherwise
        """
        return await self._get_service().get_belief(topic)

    async def get_beliefs_needing_clarification(
        self,
        limit: int = 10,
    ) -> list[AgentBelief]:
        """Get beliefs that need clarification.

        Args:
            limit: Maximum number of beliefs

        Returns:
            List of beliefs needing clarification
        """
        return await self._get_service().get_beliefs_needing_clarification(limit)

    async def get_unverified_beliefs(self, limit: int = 10) -> list[AgentBelief]:
        """Get beliefs that haven't been verified.

        Args:
            limit: Maximum number of beliefs

        Returns:
            List of unverified beliefs
        """
        return await self._get_service().get_unverified_beliefs(limit)

    async def mark_verified(
        self,
        belief_id: str,
        verification_source: str,
        new_confidence: float | None = None,
    ) -> bool:
        """Mark a belief as verified.

        Args:
            belief_id: The belief to update
            verification_source: How it was verified
            new_confidence: Optional new confidence level

        Returns:
            True if updated successfully
        """
        return await self._get_service().mark_verified(
            belief_id=belief_id,
            verification_source=verification_source,
            new_confidence=new_confidence,
        )

    async def update_belief_confidence(
        self,
        belief_id: str,
        new_confidence: float,
        reason: str,
    ) -> bool:
        """Update a belief's confidence level.

        Args:
            belief_id: The belief to update
            new_confidence: New confidence (0.0-1.0)
            reason: Why the confidence changed

        Returns:
            True if updated successfully
        """
        return await self._get_service().update_belief_confidence(
            belief_id=belief_id,
            new_confidence=new_confidence,
            reason=reason,
        )

    async def get_beliefs_for_context(
        self,
        query: str,
        user_id: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get relevant beliefs for decision context.

        Args:
            query: The query to find relevant beliefs for
            user_id: User ID for scoping
            min_confidence: Minimum confidence threshold
            limit: Maximum number of beliefs

        Returns:
            List of belief dicts with content and metadata
        """
        return await self._get_service().get_beliefs_for_context(
            query=query,
            user_id=user_id,
            min_confidence=min_confidence,
            limit=limit,
        )

    async def resolve_conflict(
        self,
        topic: str,
        observations: list[dict[str, Any]],
        current_belief: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a conflict between observations.

        Args:
            topic: The topic with conflicting info
            observations: The conflicting observations
            current_belief: Current belief on the topic if any

        Returns:
            Resolution result with strategy and suggested actions
        """
        return await self._get_service().resolve_conflict(
            topic=topic,
            observations=observations,
            current_belief=current_belief,
        )


# =============================================================================
# Trait Adapter (REQ-003-02)
# =============================================================================


class RoxyTraitAdapter(TraitProvider):
    """Adapts Roxy's RoxySelfManager to draagon-ai's TraitProvider protocol.

    Roxy's RoxySelfManager provides trait values for the agent's personality.
    This adapter bridges it to the TraitProvider protocol used by CuriosityEngine.
    """

    def __init__(self, roxy_self_manager: RoxySelfManager):
        """Initialize the adapter.

        Args:
            roxy_self_manager: Roxy's RoxySelfManager instance
        """
        self._roxy_self_manager = roxy_self_manager

    def get_trait_value(self, trait_name: str, default: float = 0.5) -> float:
        """Get a trait value from RoxySelfManager.

        Args:
            trait_name: Name of the trait (e.g., 'curiosity_intensity')
            default: Default value if trait not found

        Returns:
            Trait value 0.0-1.0
        """
        return self._roxy_self_manager.get_trait_value(trait_name, default)


# =============================================================================
# Curiosity Adapter (REQ-003-02)
# =============================================================================


@dataclass
class RoxyCuriosityAdapter:
    """Adapter that allows Roxy to use draagon-ai's CuriosityEngine.

    This is the main entry point for REQ-003-02. It wraps draagon-ai's curiosity
    engine and provides Roxy-compatible methods.

    Example:
        from roxy.services.llm import LLMService
        from roxy.services.memory import MemoryService
        from roxy.services.roxy_self import RoxySelfManager

        adapter = RoxyCuriosityAdapter(
            llm=LLMService(),
            memory=MemoryService(),
            roxy_self_manager=RoxySelfManager(),
        )

        questions = await adapter.analyze_for_curiosity(
            conversation="User: I'm planning a trip to Japan...",
            user_id="doug",
        )
    """

    llm: RoxyLLMService
    memory: RoxyMemoryService
    roxy_self_manager: RoxySelfManager
    agent_name: str = "Roxy"
    agent_id: str = "roxy"
    ask_cooldown_hours: int = 24

    _engine: CuriosityEngine | None = None

    def _get_engine(self) -> CuriosityEngine:
        """Get or create the underlying engine."""
        if self._engine is None:
            # Create adapters
            llm_adapter = RoxyLLMAdapter(self.llm)
            memory_adapter = RoxyMemoryAdapter(self.memory, agent_id=self.agent_id)
            trait_adapter = RoxyTraitAdapter(self.roxy_self_manager)

            # Create the engine
            self._engine = CuriosityEngine(
                llm=llm_adapter,
                memory=memory_adapter,
                trait_provider=trait_adapter,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
                ask_cooldown_hours=self.ask_cooldown_hours,
            )

        return self._engine

    # =========================================================================
    # Roxy-Compatible Methods
    # =========================================================================

    async def analyze_for_curiosity(
        self,
        conversation: str,
        user_id: str,
        topic_hint: str | None = None,
    ) -> list[CuriousQuestion]:
        """Analyze a conversation for things to be curious about.

        Called after interactions to identify questions.

        Args:
            conversation: The conversation text
            user_id: The user involved
            topic_hint: Optional topic hint

        Returns:
            List of CuriousQuestion objects generated
        """
        # Get worldview from RoxySelfManager if available
        worldview_str = None
        try:
            worldview_str = await self.roxy_self_manager.get_worldview_string()
        except Exception as e:
            logger.debug(f"Could not get worldview: {e}")

        return await self._get_engine().analyze_for_curiosity(
            conversation=conversation,
            user_id=user_id,
            topic_hint=topic_hint,
            worldview_str=worldview_str,
        )

    async def get_question_for_moment(
        self,
        user_id: str,
        conversation_context: str,
    ) -> CuriousQuestion | None:
        """Get a question appropriate for the current moment.

        Called when there's a natural pause or conversation end.

        Args:
            user_id: Current user
            conversation_context: Recent context

        Returns:
            A question to ask, or None if not appropriate
        """
        return await self._get_engine().get_question_for_moment(
            user_id=user_id,
            conversation_context=conversation_context,
        )

    async def mark_question_asked(self, question_id: str) -> None:
        """Mark a question as asked.

        Args:
            question_id: The question that was asked
        """
        await self._get_engine().mark_question_asked(question_id)

    async def process_answer(
        self,
        question_id: str,
        response: str,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """Process user's answer to a question.

        Args:
            question_id: The question that was answered
            response: User's response
            user_id: User ID for storage

        Returns:
            Extracted information including what agent should do next
        """
        return await self._get_engine().process_answer(
            question_id=question_id,
            response=response,
            user_id=user_id,
        )

    async def load_questions_from_storage(self) -> None:
        """Load queued questions from storage."""
        await self._get_engine().load_questions_from_storage()

    def get_pending_questions(self) -> list[CuriousQuestion]:
        """Get all pending questions (for dashboard).

        Returns:
            List of unanswered, unexpired questions
        """
        return self._get_engine().get_pending_questions()

    def get_knowledge_gaps_count(self) -> int:
        """Get count of known knowledge gaps.

        Returns:
            Number of tracked knowledge gaps
        """
        return self._get_engine().get_knowledge_gaps_count()

    def get_curiosity_level(self) -> float:
        """Get the current curiosity level from traits.

        Returns:
            Curiosity intensity 0.0-1.0
        """
        return self.roxy_self_manager.get_trait_value("curiosity_intensity", default=0.7)


# =============================================================================
# Identity Adapter (REQ-003-03)
# =============================================================================


class RoxyIdentityAdapter(IdentityManager):
    """Adapts Roxy's RoxySelfManager to draagon-ai's IdentityManager protocol.

    The IdentityManager protocol is used by OpinionFormationService to load
    and save agent identity. This adapter bridges RoxySelfManager to that protocol.

    Key mappings:
    - RoxySelf.values -> AgentIdentity.values
    - RoxySelf.worldview -> AgentIdentity.worldview
    - RoxySelf.principles -> AgentIdentity.principles
    - RoxySelf.traits -> AgentIdentity.traits
    - RoxySelf.preferences -> AgentIdentity.preferences
    - RoxySelf.opinions -> AgentIdentity.opinions
    """

    def __init__(
        self,
        roxy_self_manager: RoxySelfManager,
        agent_name: str = "Roxy",
        agent_id: str = "roxy",
    ):
        """Initialize the adapter.

        Args:
            roxy_self_manager: Roxy's RoxySelfManager instance
            agent_name: Name of the agent
            agent_id: Unique agent ID
        """
        self._roxy_self_manager = roxy_self_manager
        self._agent_name = agent_name
        self._agent_id = agent_id
        self._dirty = False
        self._cached_identity: AgentIdentity | None = None

    async def load(self) -> AgentIdentity:
        """Load the agent's identity from RoxySelfManager.

        Converts RoxySelf to AgentIdentity format.
        """
        # Get RoxySelf through the manager
        # Note: We need to call the manager's load method
        roxy_self = await self._load_roxy_self()

        # Convert to AgentIdentity
        identity = AgentIdentity(
            agent_id=self._agent_id,
            name=self._agent_name,
        )

        # Map values
        for name, value in roxy_self.get("values", {}).items():
            identity.values[name] = CoreValue(
                strength=value.get("strength", 0.9),
                description=value.get("description", ""),
                formed_through=value.get("formed_through", ""),
            )

        # Map worldview beliefs
        for name, wb in roxy_self.get("worldview", {}).items():
            identity.worldview[name] = WorldviewBelief(
                name=name,
                description=wb.get("description", ""),
                conviction=wb.get("conviction", 0.7),
                influences=wb.get("influences", []),
                open_to_revision=wb.get("open_to_revision", True),
                caveats=wb.get("caveats", []),
            )

        # Map principles
        for name, p in roxy_self.get("principles", {}).items():
            identity.principles[name] = GuidingPrinciple(
                name=name,
                description=p.get("description", ""),
                application=p.get("application", ""),
                source=p.get("source", ""),
                strength=p.get("strength", 0.9),
            )

        # Map traits
        for name, trait in roxy_self.get("traits", {}).items():
            identity.traits[name] = PersonalityTrait(
                value=trait.get("value", 0.5),
                description=trait.get("description", ""),
            )

        # Map preferences
        for name, pref in roxy_self.get("preferences", {}).items():
            identity.preferences[name] = Preference(
                name=name,
                value=pref.get("value", ""),
                reason=pref.get("reason", ""),
                confidence=pref.get("confidence", 0.7),
                formed_at=datetime.fromisoformat(pref["formed_at"]) if pref.get("formed_at") else datetime.now(),
            )

        # Map opinions
        for topic, op in roxy_self.get("opinions", {}).items():
            open_to_rev = op.get("open_to_revision", op.get("open_to_change", True))
            identity.opinions[topic] = Opinion(
                topic=topic,
                stance=op.get("stance", ""),
                basis=op.get("basis", ""),
                confidence=op.get("confidence", 0.5),
                open_to_change=open_to_rev,
                open_to_revision=open_to_rev,
                reasoning=op.get("reasoning", ""),
                caveats=op.get("caveats", []),
            )

        self._cached_identity = identity
        return identity

    async def _load_roxy_self(self) -> dict[str, Any]:
        """Load RoxySelf data from the manager.

        Returns a dict representation for easier mapping.
        """
        # The RoxySelfManager returns a RoxySelf object
        # We need to convert it to a dict for mapping
        roxy_self = await self._roxy_self_manager.load()

        # If it's already a dict, use it directly
        if isinstance(roxy_self, dict):
            return roxy_self

        # Otherwise, extract attributes
        # This handles both actual RoxySelf objects and Mock objects
        result: dict[str, Any] = {}

        # Extract each attribute if available
        for attr in ["values", "worldview", "principles", "traits", "preferences", "opinions"]:
            try:
                value = getattr(roxy_self, attr, {})
                if hasattr(value, "items"):
                    # Convert to dict of dicts
                    result[attr] = {
                        k: self._to_dict(v) for k, v in value.items()
                    }
                else:
                    result[attr] = {}
            except Exception:
                result[attr] = {}

        return result

    def _to_dict(self, obj: Any) -> dict[str, Any]:
        """Convert an object to a dict."""
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        if hasattr(obj, "_asdict"):
            return obj._asdict()
        return {"value": str(obj)}

    def mark_dirty(self) -> None:
        """Mark identity as needing save."""
        self._dirty = True
        # Also mark Roxy's manager as dirty
        try:
            self._roxy_self_manager.mark_dirty()
        except Exception:
            pass

    async def save_if_dirty(self) -> bool:
        """Save identity if modified.

        Delegates to RoxySelfManager's save method.
        """
        if not self._dirty:
            return False

        try:
            # Delegate to Roxy's manager
            result = await self._roxy_self_manager.save_if_dirty()
            if result:
                self._dirty = False
            return result
        except Exception as e:
            logger.warning(f"Failed to save identity: {e}")
            return False


# =============================================================================
# Opinion Adapter (REQ-003-03)
# =============================================================================


@dataclass
class RoxyOpinionAdapter:
    """Adapter that allows Roxy to use draagon-ai's OpinionFormationService.

    This is the main entry point for REQ-003-03. It wraps draagon-ai's opinion
    formation service and provides Roxy-compatible methods.

    Example:
        from roxy.services.llm import LLMService
        from roxy.services.memory import MemoryService
        from roxy.services.roxy_self import RoxySelfManager

        adapter = RoxyOpinionAdapter(
            llm=LLMService(),
            memory=MemoryService(),
            roxy_self_manager=RoxySelfManager(),
        )

        opinion = await adapter.form_opinion(
            topic="pineapple on pizza",
            context="User asked about food preferences",
            user_id="doug",
        )
    """

    llm: RoxyLLMService
    memory: RoxyMemoryService
    roxy_self_manager: RoxySelfManager
    agent_name: str = "Roxy"
    agent_id: str = "roxy"

    _service: OpinionFormationService | None = None

    def _get_service(self) -> OpinionFormationService:
        """Get or create the underlying service."""
        if self._service is None:
            # Create adapters
            llm_adapter = RoxyLLMAdapter(self.llm)
            memory_adapter = RoxyMemoryAdapter(self.memory, agent_id=self.agent_id)
            identity_adapter = RoxyIdentityAdapter(
                self.roxy_self_manager,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
            )

            # Create the service
            self._service = OpinionFormationService(
                llm=llm_adapter,
                memory=memory_adapter,
                identity_manager=identity_adapter,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
            )

        return self._service

    # =========================================================================
    # Roxy-Compatible Methods
    # =========================================================================

    async def form_opinion(
        self,
        topic: str,
        context: str,
        user_id: str,
    ) -> FormedOpinion | None:
        """Form an opinion on a topic.

        Args:
            topic: The topic to form an opinion on
            context: Conversational context
            user_id: User asking for the opinion

        Returns:
            FormedOpinion or None if unable to form one
        """
        request = OpinionRequest(
            topic=topic,
            user_id=user_id,
            context=context,
        )
        return await self._get_service().form_opinion(request)

    async def form_preference(
        self,
        topic: str,
        context: str,
        user_id: str,
        options: list[str] | None = None,
    ) -> Preference | None:
        """Form a preference on a topic.

        Args:
            topic: The preference topic (e.g., "favorite color")
            context: Conversational context
            user_id: User asking
            options: Optional list of choices

        Returns:
            Preference or None
        """
        request = OpinionRequest(
            topic=topic,
            user_id=user_id,
            context=context,
            is_preference_request=True,
            options=options,
        )
        return await self._get_service().form_preference(request)

    async def get_opinion(self, topic: str) -> Opinion | None:
        """Get existing opinion on a topic.

        Args:
            topic: The topic to look up

        Returns:
            Opinion or None
        """
        return await self._get_service().get_opinion(topic)

    async def get_preference(self, topic: str) -> Preference | None:
        """Get existing preference on a topic.

        Args:
            topic: The topic to look up

        Returns:
            Preference or None
        """
        return await self._get_service().get_preference(topic)

    async def get_or_form_opinion(
        self,
        topic: str,
        context: str,
        user_id: str,
    ) -> FormedOpinion | None:
        """Get existing opinion or form a new one.

        Args:
            topic: The topic
            context: Context for formation if needed
            user_id: User asking

        Returns:
            FormedOpinion (from existing or newly formed)
        """
        return await self._get_service().get_or_form_opinion(
            topic=topic,
            context=context,
            user_id=user_id,
        )

    async def consider_updating_opinion(
        self,
        topic: str,
        new_info: str,
    ) -> bool:
        """Consider whether to update an opinion based on new info.

        Args:
            topic: The topic with the opinion
            new_info: New information to consider

        Returns:
            True if opinion was updated
        """
        return await self._get_service().consider_updating_opinion(
            topic=topic,
            new_info=new_info,
        )


# =============================================================================
# Search Adapter (REQ-003-04)
# =============================================================================


class RoxySearchAdapter(SearchProvider):
    """Adapts Roxy's SearchService to draagon-ai's SearchProvider protocol.

    Roxy's SearchService uses SearXNG for web search. This adapter bridges
    it to the SearchProvider protocol used by LearningService.
    """

    def __init__(self, roxy_search: RoxySearchService):
        """Initialize the adapter.

        Args:
            roxy_search: Roxy's SearchService instance
        """
        self._search = roxy_search

    async def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search the web using Roxy's SearchService.

        Args:
            query: Search query
            limit: Max results to return

        Returns:
            List of search results with title, snippet/content, url
        """
        results = await self._search.search(query, limit)

        # Normalize result format (Roxy uses 'snippet', draagon-ai expects 'content' too)
        normalized = []
        for r in results:
            result = {
                "title": r.get("title", ""),
                "url": r.get("url", ""),
            }
            # Include both snippet and content for compatibility
            snippet = r.get("snippet", r.get("content", ""))
            result["snippet"] = snippet
            result["content"] = snippet
            normalized.append(result)

        return normalized


# =============================================================================
# Learning Credibility Adapter (REQ-003-04)
# =============================================================================


class RoxyLearningCredibilityAdapter(LearningCredibilityProvider):
    """Adapts Roxy's UserService to draagon-ai's CredibilityProvider protocol.

    This is an extended version of RoxyCredibilityAdapter that implements
    the full CredibilityProvider protocol required by LearningService.
    """

    def __init__(self, roxy_user_service: RoxyFullUserService):
        """Initialize the adapter.

        Args:
            roxy_user_service: Roxy's UserService instance
        """
        self._user_service = roxy_user_service

    def should_verify_correction(
        self,
        user_id: str,
        domain: str | None = None,
    ) -> tuple[bool, float]:
        """Determine if a correction should be verified.

        Args:
            user_id: User making the correction
            domain: Domain of the correction (optional)

        Returns:
            Tuple of (should_verify, threshold)
        """
        # Handle system/unknown users
        if user_id in ("unknown", "system", "roxy_system"):
            return True, 0.7  # Always verify with default threshold

        try:
            return self._user_service.should_verify_correction(user_id, domain)
        except Exception as e:
            logger.warning(f"Failed to check verification for {user_id}: {e}")
            return True, 0.7

    def record_correction_result(
        self,
        user_id: str,
        result: str,
        domain: str | None = None,
        user_was_confident: bool = False,
    ) -> dict[str, Any]:
        """Record the result of a correction verification.

        Args:
            user_id: User who made the correction
            result: Verification result string
            domain: Domain of the correction
            user_was_confident: Whether user seemed confident

        Returns:
            Updated credibility info
        """
        if user_id in ("unknown", "system", "roxy_system"):
            return {"user_id": user_id, "credibility": 0.7}

        try:
            return self._user_service.record_correction_result(
                user_id=user_id,
                result=result,
                domain=domain,
                user_was_confident=user_was_confident,
            )
        except Exception as e:
            logger.warning(f"Failed to record correction for {user_id}: {e}")
            return {"user_id": user_id, "error": str(e)}

    def get_user_credibility(self, user_id: str) -> Any | None:
        """Get credibility info for a user.

        Args:
            user_id: User ID to look up

        Returns:
            Credibility object or None if unknown
        """
        if user_id in ("unknown", "system", "roxy_system"):
            return None

        try:
            return self._user_service.get_user_credibility(user_id)
        except Exception as e:
            logger.warning(f"Failed to get credibility for {user_id}: {e}")
            return None


# =============================================================================
# User Provider Adapter (REQ-003-04)
# =============================================================================


class RoxyUserProviderAdapter(UserProvider):
    """Adapts Roxy's UserService to draagon-ai's UserProvider protocol.

    Provides user information for the LearningService.
    """

    def __init__(self, roxy_user_service: RoxyFullUserService):
        """Initialize the adapter.

        Args:
            roxy_user_service: Roxy's UserService instance
        """
        self._user_service = roxy_user_service

    async def get_user(self, user_id: str) -> Any | None:
        """Get user by ID.

        Args:
            user_id: User ID to look up

        Returns:
            User object or None if not found
        """
        try:
            return await self._user_service.get_user(user_id)
        except Exception as e:
            logger.warning(f"Failed to get user {user_id}: {e}")
            return None

    async def get_display_name(self, user_id: str) -> str:
        """Get user's display name.

        Args:
            user_id: User ID to look up

        Returns:
            Display name or user_id if not found
        """
        try:
            # Try async first
            user = await self._user_service.get_user(user_id)
            if user:
                # Roxy's User has display_name attribute
                return getattr(user, "display_name", user_id)
        except Exception:
            pass

        # Fall back to sync version
        try:
            user = self._user_service.get_user_sync(user_id)
            if user:
                return getattr(user, "display_name", user_id)
        except Exception:
            pass

        return user_id


# =============================================================================
# Learning Adapter (REQ-003-04)
# =============================================================================


@dataclass
class RoxyLearningAdapter:
    """Adapter that allows Roxy to use draagon-ai's LearningService.

    This is the main entry point for REQ-003-04. It wraps draagon-ai's learning
    service and provides Roxy-compatible methods.

    The LearningService handles:
    - Detecting learnable content from interactions (skills, facts, insights)
    - Extracting structured learnings with entities and scope
    - Verifying user corrections against web sources
    - Relearning skills when they fail
    - Tracking skill confidence with decay

    Example:
        from roxy.services.llm import LLMService
        from roxy.services.memory import MemoryService
        from roxy.services.search import SearchService
        from roxy.services.users import get_user_service

        adapter = RoxyLearningAdapter(
            llm=LLMService(),
            memory=MemoryService(),
            search=SearchService(),
            user_service=get_user_service(),
        )

        result = await adapter.process_interaction(
            user_query="The WiFi password is hunter2",
            response="Got it, I'll remember that.",
            tool_calls=[],
            user_id="doug",
        )
    """

    llm: RoxyLLMService
    memory: RoxyMemoryService
    search: RoxySearchService
    user_service: RoxyFullUserService
    agent_name: str = "Roxy"
    agent_id: str = "roxy"

    _service: LearningService | None = None

    def _get_service(self) -> LearningService:
        """Get or create the underlying service."""
        if self._service is None:
            # Create adapters
            llm_adapter = RoxyLLMAdapter(self.llm)
            memory_adapter = RoxyMemoryAdapter(self.memory, agent_id=self.agent_id)
            search_adapter = RoxySearchAdapter(self.search)
            credibility_adapter = RoxyLearningCredibilityAdapter(self.user_service)
            user_adapter = RoxyUserProviderAdapter(self.user_service)

            # Create the service
            self._service = LearningService(
                llm=llm_adapter,
                memory=memory_adapter,
                search_provider=search_adapter,
                credibility_provider=credibility_adapter,
                user_provider=user_adapter,
                agent_name=self.agent_name,
                agent_id=self.agent_id,
            )

        return self._service

    # =========================================================================
    # Roxy-Compatible Methods
    # =========================================================================

    async def process_interaction(
        self,
        user_query: str,
        response: str,
        tool_calls: list[dict[str, Any]],
        user_id: str,
        conversation_id: str | None = None,
        conversation_mode: str = "voice",
        previous_response: str | None = None,
    ) -> LearningResult:
        """Process an interaction to extract learnings.

        Main entry point for learning from user interactions.
        Detects skills, facts, insights, corrections, and more.

        Args:
            user_query: What the user said
            response: Roxy's response
            tool_calls: Tool calls made during the interaction
            user_id: User ID
            conversation_id: Conversation ID for context
            conversation_mode: "voice" or "text"
            previous_response: Previous response for correction detection

        Returns:
            LearningResult with detected learnings
        """
        return await self._get_service().process_interaction(
            user_query=user_query,
            response=response,
            tool_calls=tool_calls,
            user_id=user_id,
            conversation_id=conversation_id or "",
            conversation_mode=conversation_mode,
            previous_response=previous_response,
        )

    async def process_tool_failure(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        tool_result: str,
        skill_used: dict[str, Any] | None = None,
        user_id: str = "system",
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Process a tool failure for potential relearning.

        When a tool fails, this method:
        1. Detects the failure type
        2. Searches for correct information
        3. Extracts a corrected skill
        4. Updates or replaces the stored skill

        Args:
            tool_name: Name of the failed tool
            tool_args: Arguments passed to the tool
            tool_result: Error message or result
            skill_used: Memory dict of skill that was used (if any)
            user_id: User ID for context
            conversation_id: Conversation ID

        Returns:
            Dict with relearning info
        """
        return await self._get_service().process_tool_failure(
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result,
            skill_used=skill_used,
            user_id=user_id,
            conversation_id=conversation_id or "",
        )

    async def record_skill_success(
        self,
        skill_id: str,
        skill_content: str = "",
    ) -> None:
        """Record a successful skill execution.

        Boosts the skill's confidence score.

        Args:
            skill_id: ID of the skill that succeeded
            skill_content: Content of the skill
        """
        await self._get_service().record_skill_success(skill_id, skill_content)

    def get_skill_confidence(self, skill_id: str) -> float | None:
        """Get the confidence score for a skill.

        Args:
            skill_id: ID of the skill

        Returns:
            Confidence score 0.0-1.0, or None if not tracked
        """
        return self._get_service().get_skill_confidence(skill_id)

    def get_degraded_skills(self, threshold: float = 0.3) -> list[SkillConfidence]:
        """Get skills that have degraded below a threshold.

        Args:
            threshold: Minimum confidence threshold

        Returns:
            List of SkillConfidence objects below threshold
        """
        all_skills = self._get_service().get_degraded_skills()
        return [s for s in all_skills if s.confidence < threshold]

    async def detect_household_conflicts(
        self,
        content: str,
        user_id: str,
        entities: list[str],
    ) -> list[dict[str, Any]]:
        """Detect conflicts with other household members' knowledge.

        For multi-user households, check if this learning conflicts
        with what other users have stated.

        Note: This is a stub implementation. Full household conflict detection
        requires a LearningExtension to be provided to the LearningService.
        Without an extension, this returns an empty list (no conflicts).

        Args:
            content: The content being stored
            user_id: User making the statement
            entities: Extracted entities

        Returns:
            List of conflict dicts (empty if no conflicts or no extension)
        """
        # LearningService.detect_household_conflicts is on the LearningExtension
        # protocol, not the LearningService itself. Since Roxy doesn't currently
        # provide a LearningExtension, we return an empty list.
        service = self._get_service()
        if service.extension is not None:
            return await service.extension.detect_household_conflicts(
                content=content,
                user_id=user_id,
                entities=entities,
            )
        # No extension configured - return empty (no conflicts detected)
        return []

    def get_skill_stats(self) -> dict[str, Any]:
        """Get skill tracking statistics.

        Returns:
            Dict with skill counts and degradation info
        """
        service = self._get_service()
        skills = service.get_degraded_skills()
        return {
            "total_tracked": len(service._skill_confidence),
            "degraded_count": len([s for s in skills if s.needs_relearning()]),
            "average_confidence": sum(s.confidence for s in skills) / len(skills) if skills else 1.0,
        }
