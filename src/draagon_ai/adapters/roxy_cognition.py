"""Roxy Cognition Adapters.

Adapters that allow Roxy to use draagon-ai's cognitive services while providing
Roxy-specific LLM, Memory, and Credibility implementations.

REQ-003-01: Belief reconciliation using core service.
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
