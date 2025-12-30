"""Belief Reconciliation for Draagon AI.

Forms agent beliefs from multiple user observations, handling:
- Multiple users reporting the same/different information
- Conflicting claims requiring reconciliation
- Verification status tracking
- Confidence calibration based on source credibility

This is the bridge between Layer 1 (raw observations) and Layer 2 (agent beliefs)
in the cognitive architecture.
"""

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from draagon_ai.core import AgentContext
from draagon_ai.core.types import (
    AgentBelief,
    BeliefType,
    ObservationScope,
    UserObservation,
)
from draagon_ai.llm import LLMProvider, ModelTier
from draagon_ai.memory import MemoryProvider, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Prompts
# =============================================================================

BELIEF_FORMATION_PROMPT = """You are {agent_name}'s belief formation system. Your job is to analyze
observations from users and form {agent_name}'s own beliefs about the world.

OBSERVATIONS TO RECONCILE:
{observations}

EXISTING BELIEF (if any):
{existing_belief}

Analyze these observations and determine:

1. **Consistency Check**: Do all observations agree, or are there conflicts?
2. **Source Credibility**: Consider who said what. Multiple sources agreeing = stronger belief.
3. **Recency**: More recent observations may supersede older ones.
4. **Specificity**: More specific information often supersedes general.

Form a belief that:
- Represents {agent_name}'s best understanding
- Notes any conflicts or uncertainties
- Has appropriate confidence based on evidence

Output XML:
<belief>
    <content>The belief statement (as {agent_name} would hold it)</content>
    <belief_type>household_fact | verified_fact | unverified_claim | inferred | user_preference | agent_preference</belief_type>
    <confidence>0.0-1.0</confidence>
    <has_conflict>true or false</has_conflict>
    <conflict_description>describe any conflicts, or empty</conflict_description>
    <needs_clarification>true or false</needs_clarification>
    <clarification_priority>0.0-1.0</clarification_priority>
    <reasoning>Brief explanation of how you formed this belief</reasoning>
</belief>

Examples:
- 3 users say "We have 6 cats" → confidence 0.95, household_fact
- 1 user says "5 cats", another says "6 cats" → confidence 0.6, needs_clarification
- 1 user mentions casually "the WiFi password is X" → confidence 0.85, household_fact
"""

CONFLICT_RESOLUTION_PROMPT = """You are {agent_name}'s conflict resolution system. Users have provided
conflicting information about the same topic.

CONFLICTING OBSERVATIONS:
{observations}

CURRENT BELIEF:
{current_belief}

Your task:
1. Analyze the conflict
2. Suggest how {agent_name} should handle it
3. Formulate a question {agent_name} could ask to resolve the conflict

Output XML:
<conflict_resolution>
    <conflict_type>factual_disagreement | outdated_info | perspective_difference | unclear</conflict_type>
    <resolution_strategy>ask_user | prefer_recent | prefer_majority | keep_uncertain | verify_externally</resolution_strategy>
    <suggested_question>A natural question {agent_name} could ask to resolve this</suggested_question>
    <temporary_belief>What {agent_name} should believe until resolved</temporary_belief>
    <temporary_confidence>0.0-1.0</temporary_confidence>
</conflict_resolution>
"""

OBSERVATION_EXTRACTION_PROMPT = """Extract a formal observation from this user statement.

USER STATEMENT: {statement}
USER ID: {user_id}
CONTEXT: {context}

Determine:
1. What claim is being made?
2. What scope should this observation have?
   - "private": Secrets (passwords, personal info only this user should access)
   - "personal": User's own fact (their preferences, their birthday)
   - "household": Shared family info (number of pets, family members' birthdays)
3. How confident does the user seem?

Output XML:
<observation>
    <content>The observation/claim in clear terms</content>
    <scope>private | personal | household</scope>
    <confidence_expressed>0.0-1.0</confidence_expressed>
    <entities>
        <entity>key</entity>
        <entity>entities</entity>
    </entities>
    <is_correction>true or false</is_correction>
    <corrects_topic>topic being corrected, or empty</corrects_topic>
</observation>
"""


# =============================================================================
# Protocols for host-provided services
# =============================================================================

@runtime_checkable
class CredibilityProvider(Protocol):
    """Protocol for user credibility lookup.

    Host applications implement this to provide user credibility scores.
    """

    def get_user_credibility(self, user_id: str) -> float | None:
        """Get credibility score for a user.

        Args:
            user_id: User ID to look up

        Returns:
            Credibility score 0.0-1.0, or None if unknown
        """
        ...


# =============================================================================
# Result Types
# =============================================================================

@dataclass
class ReconciliationResult:
    """Result of reconciling observations into a belief."""
    belief: AgentBelief
    action: str  # "created", "updated", "conflict_detected", "unchanged"
    observations_used: list[str]  # Observation IDs
    conflict_info: dict[str, Any] | None = None


# =============================================================================
# Service
# =============================================================================

class BeliefReconciliationService:
    """Forms and maintains agent beliefs from user observations.

    Key responsibilities:
    - Convert raw user statements into formal observations
    - Aggregate multiple observations about the same topic
    - Detect and handle conflicts
    - Form beliefs with appropriate confidence
    - Track verification status

    This service is backend-agnostic. It uses LLMProvider and MemoryProvider
    interfaces, allowing host applications to plug in their specific backends.
    """

    # Record types for storage
    OBSERVATION_RECORD_TYPE = "user_observation"
    BELIEF_RECORD_TYPE = "agent_belief"

    def __init__(
        self,
        llm: LLMProvider,
        memory: MemoryProvider,
        agent_name: str = "the agent",
        agent_id: str = "agent",
        credibility_provider: CredibilityProvider | None = None,
    ):
        """Initialize the belief reconciliation service.

        Args:
            llm: LLM provider for reasoning
            memory: Memory provider for storage
            agent_name: Name of the agent (for prompts)
            agent_id: Agent ID for namespacing storage
            credibility_provider: Optional provider for user credibility scores
        """
        self.llm = llm
        self.memory = memory
        self.agent_name = agent_name
        self.agent_id = agent_id
        self.credibility_provider = credibility_provider

    # =========================================================================
    # Observation Creation
    # =========================================================================

    async def create_observation(
        self,
        statement: str,
        user_id: str,
        context: str | None = None,
        conversation_id: str | None = None,
    ) -> UserObservation:
        """Create an observation from a user statement.

        Observations are immutable records of what users said.
        They are the raw input that beliefs are formed from.

        Args:
            statement: What the user said
            user_id: Who said it
            context: Conversation context
            conversation_id: Conversation ID

        Returns:
            The created observation
        """
        # Use LLM to extract formal observation
        prompt = OBSERVATION_EXTRACTION_PROMPT.format(
            statement=statement,
            user_id=user_id,
            context=context or "General conversation",
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract the observation."},
            ],
            max_tokens=300,
            tier=ModelTier.LOCAL,
        )

        # Parse JSON from response
        parsed = self._parse_json_response(response.content)

        if not parsed:
            # Fallback to simple observation
            logger.warning("LLM observation extraction failed, using fallback")
            return UserObservation(
                observation_id=str(uuid.uuid4()),
                content=statement,
                source_user_id=user_id,
                scope=ObservationScope.HOUSEHOLD,
                timestamp=datetime.now(),
                conversation_id=conversation_id,
                confidence_expressed=0.8,
                context=context,
            )

        # Map scope string to enum
        scope_map = {
            "private": ObservationScope.PRIVATE,
            "personal": ObservationScope.PERSONAL,
            "household": ObservationScope.HOUSEHOLD,
        }
        scope = scope_map.get(parsed.get("scope", "household"), ObservationScope.HOUSEHOLD)

        observation = UserObservation(
            observation_id=str(uuid.uuid4()),
            content=parsed.get("content", statement),
            source_user_id=user_id,
            scope=scope,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            confidence_expressed=parsed.get("confidence_expressed", 0.8),
            context=context,
        )

        # Store the observation
        await self._store_observation(observation, parsed.get("entities", []))

        logger.info(
            f"Created observation from {user_id}: '{observation.content[:50]}...' "
            f"(scope={scope.value}, confidence={observation.confidence_expressed})"
        )

        # Check if this observation should trigger belief formation
        if parsed.get("is_correction"):
            # This is a correction - needs immediate reconciliation
            await self.reconcile_topic(
                topic=parsed.get("corrects_topic", observation.content),
                user_id=user_id,
                new_observation=observation,
            )
        else:
            # Regular observation - queue for belief formation
            await self._queue_for_reconciliation(observation, parsed.get("entities", []))

        return observation

    async def _store_observation(
        self,
        observation: UserObservation,
        entities: list[str],
    ) -> None:
        """Store an observation in memory."""
        await self.memory.store(
            content=observation.content,
            memory_type=MemoryType.OBSERVATION,
            scope=self._map_observation_scope(observation.scope),
            agent_id=self.agent_id,
            user_id=observation.source_user_id,
            importance=observation.confidence_expressed,
            entities=entities,
            metadata={
                "record_type": self.OBSERVATION_RECORD_TYPE,
                "observation_id": observation.observation_id,
                "source_user_id": observation.source_user_id,
                "observation_scope": observation.scope.value,
                "confidence_expressed": observation.confidence_expressed,
                "context": observation.context,
                "conversation_id": observation.conversation_id,
                "timestamp": observation.timestamp.isoformat(),
            },
        )

    async def _queue_for_reconciliation(
        self,
        observation: UserObservation,
        entities: list[str],
    ) -> None:
        """Queue an observation for belief reconciliation."""
        # Find related observations
        if entities:
            query = " ".join(entities)
        else:
            query = observation.content[:100]

        from draagon_ai.memory import MemoryScope
        related = await self.memory.search(
            query=query,
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=5,
        )

        # Filter to just observation records
        related_observations = [
            self._search_result_to_dict(r) for r in related
            if r.memory.embedding is None or True  # Include all results
        ]
        related_observations = [
            r for r in related_observations
            if r.get("metadata", {}).get("record_type") == self.OBSERVATION_RECORD_TYPE
        ]

        # If we have multiple observations on the same topic, reconcile
        if len(related_observations) >= 1:
            await self.reconcile_observations(
                observations=[observation],
                existing_observations=related_observations,
            )

    # =========================================================================
    # Belief Formation
    # =========================================================================

    async def reconcile_observations(
        self,
        observations: list[UserObservation],
        existing_observations: list[dict[str, Any]] | None = None,
    ) -> ReconciliationResult | None:
        """Reconcile observations into a belief.

        Args:
            observations: New observations to process
            existing_observations: Previously stored observations (from search)

        Returns:
            ReconciliationResult with the formed belief
        """
        if not observations and not existing_observations:
            return None

        # Combine all observations
        all_obs_text = []
        obs_ids = []

        for obs in observations:
            all_obs_text.append(
                f"[{obs.source_user_id}] ({obs.timestamp.strftime('%Y-%m-%d %H:%M')}): "
                f"{obs.content} (confidence: {obs.confidence_expressed})"
            )
            obs_ids.append(obs.observation_id)

        if existing_observations:
            for obs_dict in existing_observations:
                metadata = obs_dict.get("metadata", {})
                all_obs_text.append(
                    f"[{obs_dict.get('source_user_id', 'unknown')}] "
                    f"({metadata.get('timestamp', 'unknown')}): "
                    f"{obs_dict.get('content', '')} "
                    f"(confidence: {metadata.get('confidence_expressed', 0.8)})"
                )
                if metadata.get("observation_id"):
                    obs_ids.append(metadata["observation_id"])

        # Check for existing belief on this topic
        existing_belief = await self._find_existing_belief(
            observations[0].content if observations else existing_observations[0].get("content", "")
        )

        # Use LLM to form belief
        prompt = BELIEF_FORMATION_PROMPT.format(
            agent_name=self.agent_name,
            observations="\n".join(all_obs_text),
            existing_belief=existing_belief or "(None)",
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Form a belief from these observations."},
            ],
            max_tokens=400,
            tier=ModelTier.LOCAL,
        )

        parsed = self._parse_json_response(response.content)

        if not parsed:
            logger.warning("Belief formation failed")
            return None

        # Map belief type
        type_map = {
            "household_fact": BeliefType.HOUSEHOLD_FACT,
            "verified_fact": BeliefType.VERIFIED_FACT,
            "unverified_claim": BeliefType.UNVERIFIED_CLAIM,
            "inferred": BeliefType.INFERRED,
            "user_preference": BeliefType.USER_PREFERENCE,
            "agent_preference": BeliefType.AGENT_PREFERENCE,
        }
        belief_type = type_map.get(
            parsed.get("belief_type", "unverified_claim"),
            BeliefType.UNVERIFIED_CLAIM
        )

        # Calculate base confidence from LLM
        base_confidence = parsed.get("confidence", 0.7)

        # Adjust confidence based on source user credibility
        source_users = set()
        for obs in observations:
            source_users.add(obs.source_user_id)
        if existing_observations:
            for obs_dict in existing_observations:
                source_users.add(obs_dict.get("source_user_id", "unknown"))

        adjusted_confidence = self._adjust_confidence_for_credibility(
            base_confidence, list(source_users)
        )

        # Create belief
        belief = AgentBelief(
            belief_id=str(uuid.uuid4()),
            content=parsed.get("content", observations[0].content if observations else ""),
            belief_type=belief_type,
            confidence=adjusted_confidence,
            supporting_observations=obs_ids,
            conflicting_observations=[],
            verified=belief_type == BeliefType.VERIFIED_FACT,
            needs_clarification=parsed.get("needs_clarification", False),
            clarification_priority=parsed.get("clarification_priority", 0.0),
        )

        # Handle conflicts
        action = "created"
        conflict_info = None

        if parsed.get("has_conflict"):
            action = "conflict_detected"
            conflict_info = {
                "description": parsed.get("conflict_description"),
                "resolution_needed": True,
            }
            await self._store_pending_belief_conflict(belief, conflict_info)

        # Store the belief
        await self._store_belief(belief)

        logger.info(
            f"Formed belief: '{belief.content[:50]}...' "
            f"(type={belief.belief_type.value}, confidence={belief.confidence}, "
            f"action={action})"
        )

        return ReconciliationResult(
            belief=belief,
            action=action,
            observations_used=obs_ids,
            conflict_info=conflict_info,
        )

    async def reconcile_topic(
        self,
        topic: str,
        user_id: str,
        new_observation: UserObservation | None = None,
    ) -> ReconciliationResult | None:
        """Reconcile all observations about a topic.

        Args:
            topic: The topic to reconcile (search query)
            user_id: User requesting reconciliation
            new_observation: Optional new observation to include

        Returns:
            ReconciliationResult with the formed/updated belief
        """
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query=topic,
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=10,
        )

        existing_obs = [
            self._search_result_to_dict(r) for r in results
        ]
        existing_obs = [
            r for r in existing_obs
            if r.get("metadata", {}).get("record_type") == self.OBSERVATION_RECORD_TYPE
        ]

        observations = [new_observation] if new_observation else []

        return await self.reconcile_observations(
            observations=observations,
            existing_observations=existing_obs,
        )

    async def _find_existing_belief(self, topic: str) -> str | None:
        """Find existing belief about a topic."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query=topic,
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=3,
        )

        for r in results:
            result_dict = self._search_result_to_dict(r)
            if result_dict.get("metadata", {}).get("record_type") == self.BELIEF_RECORD_TYPE:
                return result_dict.get("content")

        return None

    def _adjust_confidence_for_credibility(
        self,
        base_confidence: float,
        source_users: list[str],
    ) -> float:
        """Adjust belief confidence based on source user credibility."""
        if not source_users or not self.credibility_provider:
            return base_confidence

        try:
            credibility_scores = []

            for user_id in source_users:
                if user_id in ("unknown", "system", f"{self.agent_id}_system"):
                    continue

                credibility = self.credibility_provider.get_user_credibility(user_id)
                if credibility is not None:
                    credibility_scores.append(credibility)

            if not credibility_scores:
                return base_confidence

            avg_credibility = sum(credibility_scores) / len(credibility_scores)

            # Adjust confidence
            if avg_credibility > 0.8:
                adjustment = 0.15 * ((avg_credibility - 0.8) / 0.2)
            elif avg_credibility < 0.5:
                adjustment = -0.30 * ((0.5 - avg_credibility) / 0.5)
            else:
                adjustment = 0.0

            # Multiple agreeing sources boost confidence
            if len(credibility_scores) > 1:
                adjustment += 0.05 * min(len(credibility_scores) - 1, 3)

            adjusted = base_confidence + adjustment
            adjusted = max(0.1, min(1.0, adjusted))

            if adjustment != 0:
                logger.debug(
                    f"Adjusted belief confidence: {base_confidence:.2f} -> {adjusted:.2f} "
                    f"(avg credibility: {avg_credibility:.2f}, sources: {len(credibility_scores)})"
                )

            return adjusted

        except Exception as e:
            logger.warning(f"Credibility adjustment failed (using base): {e}")
            return base_confidence

    async def _store_belief(self, belief: AgentBelief) -> None:
        """Store a belief in memory."""
        from draagon_ai.memory import MemoryScope

        await self.memory.store(
            content=belief.content,
            memory_type=MemoryType.BELIEF,
            scope=MemoryScope.AGENT,
            agent_id=self.agent_id,
            importance=belief.confidence,
            entities=["agent_belief"],
            metadata={
                "record_type": self.BELIEF_RECORD_TYPE,
                "belief_id": belief.belief_id,
                "belief_type": belief.belief_type.value,
                "confidence": belief.confidence,
                "supporting_observations": belief.supporting_observations,
                "conflicting_observations": belief.conflicting_observations,
                "verified": belief.verified,
                "verification_source": belief.verification_source,
                "needs_clarification": belief.needs_clarification,
                "clarification_priority": belief.clarification_priority,
                "created_at": belief.created_at.isoformat(),
                "updated_at": belief.updated_at.isoformat(),
            },
        )

    async def _store_pending_belief_conflict(
        self,
        belief: AgentBelief,
        conflict_info: dict[str, Any],
    ) -> None:
        """Store a conflict for later resolution."""
        from draagon_ai.memory import MemoryScope

        await self.memory.store(
            content=(
                f"BELIEF CONFLICT: {conflict_info.get('description', 'Unknown conflict')}\n"
                f"Belief: {belief.content}"
            ),
            memory_type=MemoryType.INSIGHT,
            scope=MemoryScope.AGENT,
            agent_id=self.agent_id,
            importance=0.9,
            entities=["belief_conflict", "needs_resolution"],
            metadata={
                "record_type": "pending_belief_conflict",
                "belief_id": belief.belief_id,
                "conflict_info": conflict_info,
                "needs_resolution": True,
            },
        )

    # =========================================================================
    # Belief Retrieval
    # =========================================================================

    async def get_belief(self, topic: str) -> AgentBelief | None:
        """Get the agent's belief about a topic."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query=topic,
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=3,
        )

        for r in results:
            result_dict = self._search_result_to_dict(r)
            metadata = result_dict.get("metadata", {})
            if metadata.get("record_type") == self.BELIEF_RECORD_TYPE:
                return AgentBelief(
                    belief_id=metadata.get("belief_id", str(uuid.uuid4())),
                    content=result_dict.get("content", ""),
                    belief_type=BeliefType(metadata.get("belief_type", "unverified_claim")),
                    confidence=metadata.get("confidence", 0.5),
                    supporting_observations=metadata.get("supporting_observations", []),
                    conflicting_observations=metadata.get("conflicting_observations", []),
                    verified=metadata.get("verified", False),
                    verification_source=metadata.get("verification_source"),
                    needs_clarification=metadata.get("needs_clarification", False),
                    clarification_priority=metadata.get("clarification_priority", 0.0),
                )

        return None

    async def get_beliefs_needing_clarification(
        self,
        limit: int = 10,
    ) -> list[AgentBelief]:
        """Get beliefs that need clarification."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query="needs clarification conflict uncertain",
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=limit,
        )

        beliefs = []
        for r in results:
            result_dict = self._search_result_to_dict(r)
            metadata = result_dict.get("metadata", {})
            if (metadata.get("record_type") == self.BELIEF_RECORD_TYPE and
                metadata.get("needs_clarification")):
                beliefs.append(AgentBelief(
                    belief_id=metadata.get("belief_id", str(uuid.uuid4())),
                    content=result_dict.get("content", ""),
                    belief_type=BeliefType(metadata.get("belief_type", "unverified_claim")),
                    confidence=metadata.get("confidence", 0.5),
                    supporting_observations=metadata.get("supporting_observations", []),
                    conflicting_observations=metadata.get("conflicting_observations", []),
                    verified=metadata.get("verified", False),
                    needs_clarification=True,
                    clarification_priority=metadata.get("clarification_priority", 0.0),
                ))

        beliefs.sort(key=lambda b: b.clarification_priority, reverse=True)
        return beliefs

    async def get_unverified_beliefs(self, limit: int = 10) -> list[AgentBelief]:
        """Get beliefs that haven't been verified."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query="unverified claim fact",
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=limit * 2,
        )

        beliefs = []
        for r in results:
            result_dict = self._search_result_to_dict(r)
            metadata = result_dict.get("metadata", {})
            if (metadata.get("record_type") == self.BELIEF_RECORD_TYPE and
                not metadata.get("verified", False)):
                beliefs.append(AgentBelief(
                    belief_id=metadata.get("belief_id", str(uuid.uuid4())),
                    content=result_dict.get("content", ""),
                    belief_type=BeliefType(metadata.get("belief_type", "unverified_claim")),
                    confidence=metadata.get("confidence", 0.5),
                    supporting_observations=metadata.get("supporting_observations", []),
                    conflicting_observations=metadata.get("conflicting_observations", []),
                    verified=False,
                    needs_clarification=metadata.get("needs_clarification", False),
                    clarification_priority=metadata.get("clarification_priority", 0.0),
                ))

        return beliefs[:limit]

    # =========================================================================
    # Belief Updates
    # =========================================================================

    async def mark_verified(
        self,
        belief_id: str,
        verification_source: str,
        new_confidence: float | None = None,
    ) -> bool:
        """Mark a belief as verified."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query=f"belief_id:{belief_id}",
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=5,
        )

        for r in results:
            result_dict = self._search_result_to_dict(r)
            metadata = result_dict.get("metadata", {})
            if metadata.get("belief_id") == belief_id:
                metadata["verified"] = True
                metadata["verification_source"] = verification_source
                metadata["last_verified"] = datetime.now().isoformat()
                if new_confidence:
                    metadata["confidence"] = new_confidence
                metadata["belief_type"] = BeliefType.VERIFIED_FACT.value

                await self.memory.store(
                    content=result_dict.get("content", ""),
                    memory_type=MemoryType.BELIEF,
                    scope=MemoryScope.AGENT,
                    agent_id=self.agent_id,
                    importance=metadata.get("confidence", 0.9),
                    entities=["agent_belief", "verified"],
                    metadata=metadata,
                )

                logger.info(f"Marked belief {belief_id} as verified via {verification_source}")
                return True

        return False

    async def update_belief_confidence(
        self,
        belief_id: str,
        new_confidence: float,
        reason: str,
    ) -> bool:
        """Update a belief's confidence level."""
        from draagon_ai.memory import MemoryScope

        results = await self.memory.search(
            query=f"belief_id:{belief_id}",
            agent_id=self.agent_id,
            scopes=[MemoryScope.AGENT],
            limit=5,
        )

        for r in results:
            result_dict = self._search_result_to_dict(r)
            metadata = result_dict.get("metadata", {})
            if metadata.get("belief_id") == belief_id:
                old_confidence = metadata.get("confidence", 0.5)
                metadata["confidence"] = new_confidence
                metadata["updated_at"] = datetime.now().isoformat()

                await self.memory.store(
                    content=result_dict.get("content", ""),
                    memory_type=MemoryType.BELIEF,
                    scope=MemoryScope.AGENT,
                    agent_id=self.agent_id,
                    importance=new_confidence,
                    entities=["agent_belief"],
                    metadata=metadata,
                )

                logger.info(
                    f"Updated belief {belief_id} confidence: {old_confidence} -> {new_confidence} "
                    f"(reason: {reason})"
                )
                return True

        return False

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    async def get_beliefs_for_context(
        self,
        query: str,
        user_id: str | None = None,
        min_confidence: float = 0.0,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Get relevant memories/beliefs for decision context."""
        from draagon_ai.memory import MemoryScope

        scopes = [MemoryScope.AGENT]
        if user_id:
            scopes.append(MemoryScope.USER)

        results = await self.memory.search(
            query=query,
            agent_id=self.agent_id,
            user_id=user_id,
            scopes=scopes,
            limit=limit,
        )

        result_dicts = [self._search_result_to_dict(r) for r in results]

        if min_confidence > 0:
            filtered = []
            for r in result_dicts:
                metadata = r.get("metadata", {})
                confidence = (
                    metadata.get("confidence") or
                    metadata.get("confidence_expressed") or
                    r.get("importance", 0.5)
                )
                if confidence >= min_confidence:
                    filtered.append(r)
            return filtered[:limit]

        return result_dicts

    async def resolve_conflict(
        self,
        topic: str,
        observations: list[dict[str, Any]],
        current_belief: str | None = None,
    ) -> dict[str, Any] | None:
        """Resolve a conflict between observations."""
        obs_text = "\n".join([
            f"[{o.get('source', 'unknown')}] ({o.get('timestamp', 'unknown')}): {o.get('content', '')}"
            for o in observations
        ])

        prompt = CONFLICT_RESOLUTION_PROMPT.format(
            agent_name=self.agent_name,
            observations=obs_text,
            current_belief=current_belief or "(No current belief)",
        )

        response = await self.llm.chat(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "How should I resolve this conflict?"},
            ],
            max_tokens=300,
            tier=ModelTier.LOCAL,
        )

        return self._parse_json_response(response.content)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response content."""
        import json
        import re

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Try direct JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON object in content
            json_match = re.search(r'\{[^{}]*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            return None

    def _search_result_to_dict(self, result) -> dict[str, Any]:
        """Convert a SearchResult to a dict format."""
        from draagon_ai.memory import SearchResult

        if isinstance(result, SearchResult):
            # Get metadata dict, handling both direct attribute and potential method
            metadata = {}
            if hasattr(result.memory, '__dict__'):
                # Build metadata from Memory fields
                memory = result.memory
                metadata = {
                    "record_type": getattr(memory, 'record_type', None),
                    "observation_id": getattr(memory, 'observation_id', None),
                    "belief_id": getattr(memory, 'belief_id', None),
                    "source_user_id": memory.user_id,
                    "confidence": memory.confidence,
                    "confidence_expressed": memory.confidence,
                    "timestamp": memory.created_at.isoformat() if memory.created_at else None,
                    "needs_clarification": getattr(memory, 'needs_clarification', None),
                    "clarification_priority": getattr(memory, 'clarification_priority', None),
                    "verified": getattr(memory, 'verified', None),
                    "verification_source": getattr(memory, 'verification_source', None),
                    "belief_type": getattr(memory, 'belief_type', None),
                    "supporting_observations": getattr(memory, 'supporting_observations', None),
                    "conflicting_observations": getattr(memory, 'conflicting_observations', None),
                }
            return {
                "content": result.memory.content,
                "score": result.score,
                "importance": result.memory.importance,
                "source_user_id": result.memory.user_id,
                "metadata": metadata,
            }

        # Already a dict
        return result

    def _map_observation_scope(self, scope: ObservationScope):
        """Map ObservationScope to MemoryScope."""
        from draagon_ai.memory import MemoryScope

        mapping = {
            ObservationScope.PRIVATE: MemoryScope.USER,
            ObservationScope.PERSONAL: MemoryScope.USER,
            ObservationScope.HOUSEHOLD: MemoryScope.CONTEXT,
        }
        return mapping.get(scope, MemoryScope.AGENT)
