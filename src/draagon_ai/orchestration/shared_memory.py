"""Shared cognitive working memory for multi-agent coordination.

This module implements task-scoped working memory that enables multiple agents
to coordinate through attention-weighted observations. Based on cognitive
psychology research (Miller's Law, Baddeley's Working Memory Model).

Key Features:
- Observation storage with full source attribution
- Miller's Law capacity management (7±2 items per agent)
- Semantic conflict detection (Phase 1: heuristic, Phase 2: embeddings)
- Attention weighting and periodic decay
- Role-based context filtering
- Concurrent access safety

Research Foundation:
- Miller (1956): "The Magical Number Seven, Plus or Minus Two"
- Baddeley & Hitch (1974): Working Memory Model
- MultiAgentBench (ACL 2025): Shared context prevents coordination failures
- Intrinsic Memory Agents: Heterogeneous views improve performance by 38.6%

Example:
    ```python
    from draagon_ai.orchestration.shared_memory import SharedWorkingMemory
    from draagon_ai.orchestration.multi_agent_orchestrator import AgentRole

    # Create shared memory for task
    memory = SharedWorkingMemory(task_id="task_123")

    # Agent A adds observation
    obs = await memory.add_observation(
        content="Meeting is at 3pm",
        source_agent_id="agent_a",
        attention_weight=0.8,
        is_belief_candidate=True,
        belief_type="FACT",
    )

    # Agent B retrieves context (filtered by role)
    context = await memory.get_context_for_agent(
        agent_id="agent_b",
        role=AgentRole.CRITIC,
        max_items=7,  # Miller's Law
    )

    # Apply periodic attention decay
    await memory.apply_attention_decay()

    # Get conflicts for reconciliation
    conflicts = await memory.get_conflicts()
    ```
"""

from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Protocol, runtime_checkable
import asyncio
import logging
import math
import re
import uuid
from collections import Counter

from .multi_agent_orchestrator import AgentRole

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass(frozen=True)
class SharedObservation:
    """An observation in shared working memory.

    Observations are immutable once created. They represent what an agent
    observed or concluded during task execution.

    Note on Immutability:
        This dataclass is frozen (immutable), but `accessed_by` and
        `access_count` need updates for access tracking. We use
        `dataclasses.replace()` in SharedWorkingMemory methods:
        ```python
        new_obs = replace(old_obs, accessed_by=new_set, access_count=new_count)
        self._observations[obs_id] = new_obs
        ```
        This preserves immutability semantics while allowing access tracking.

    Attributes:
        observation_id: Unique identifier (UUID)
        content: The observation content
        source_agent_id: Which agent made this observation
        timestamp: When observation was created
        attention_weight: Current attention (0-1), decays over time
        confidence: Agent's confidence in this observation (0-1)
        is_belief_candidate: Should this become a belief?
        belief_type: Type if belief candidate (FACT, SKILL, PREFERENCE, etc.)
        conflicts_with: List of observation IDs that conflict
        accessed_by: Set of agent IDs that have read this
        access_count: Number of times accessed
    """

    observation_id: str
    content: str
    source_agent_id: str
    timestamp: datetime

    # Cognitive properties
    attention_weight: float = 0.5  # 0-1
    confidence: float = 1.0  # 0-1

    # Belief tracking
    is_belief_candidate: bool = False
    belief_type: str | None = None  # "FACT", "SKILL", "PREFERENCE", etc.

    # Conflict tracking
    conflicts_with: list[str] = field(default_factory=list)

    # Access tracking (updated via replacement pattern due to frozen=True)
    accessed_by: set[str] = field(default_factory=set)
    access_count: int = 0

    def __post_init__(self):
        """Validate observation after creation.

        Raises:
            ValueError: If attention_weight or confidence not in [0, 1]
        """
        if not 0 <= self.attention_weight <= 1:
            raise ValueError(
                f"attention_weight must be 0-1, got {self.attention_weight}"
            )
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be 0-1, got {self.confidence}")


@dataclass
class SharedWorkingMemoryConfig:
    """Configuration for shared working memory.

    Defaults based on cognitive psychology research:
    - max_items_per_agent: 7 (Miller's Law: 7±2)
    - max_total_items: 50 (room for ~7 agents × 7 items each)
    - attention_decay_factor: 0.9 (10% decay per sync)
    - conflict_threshold: 0.7 (semantic similarity for conflicts)
    - sync_interval_iterations: 3 (barrier sync frequency)

    Attributes:
        max_items_per_agent: Maximum observations per agent (Miller's Law)
        max_total_items: Global capacity limit
        attention_decay_factor: Multiply attention by this on decay (0-1)
        conflict_threshold: Semantic similarity threshold for conflicts
        sync_interval_iterations: How often to sync in barrier mode
    """

    # Capacity constraints
    max_items_per_agent: int = 7  # Miller's Law: 7±2
    max_total_items: int = 50  # Global capacity

    # Attention management
    attention_decay_factor: float = 0.9  # Multiply by this on decay

    # Conflict detection
    conflict_threshold: float = 0.7  # Semantic similarity threshold

    # Synchronization
    sync_interval_iterations: int = 3  # Periodic sync frequency


# =============================================================================
# Protocols
# =============================================================================


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers (optional).

    Used for semantic similarity in conflict detection.
    If not provided, SharedWorkingMemory uses simple heuristic
    (same belief_type = potential conflict).

    Phase 1: Not required (heuristic conflict detection)
    Phase 2: Implement for semantic similarity

    Example:
        ```python
        from sentence_transformers import SentenceTransformer
        import numpy as np

        class SentenceTransformerEmbedding:
            def __init__(self):
                self.model = SentenceTransformer("all-MiniLM-L6-v2")

            async def embed(self, text: str) -> list[float]:
                return self.model.encode(text).tolist()

            async def similarity(self, text_a: str, text_b: str) -> float:
                emb_a = await self.embed(text_a)
                emb_b = await self.embed(text_b)
                return np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
        ```
    """

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (typically 384-1536 dimensions)
        """
        ...

    async def similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity between two texts.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        ...


# =============================================================================
# Main Class
# =============================================================================


class SharedWorkingMemory:
    """Task-scoped working memory for multi-agent coordination.

    Provides:
    - Observation storage with source attribution
    - Miller's Law capacity management (7±2 per agent)
    - Semantic conflict detection
    - Attention weighting and decay
    - Role-based context filtering
    - Concurrent access safety

    This implements the working memory layer of the 4-layer cognitive
    architecture, scoped to a single task. It enables multiple agents
    to share observations while maintaining capacity constraints and
    detecting conflicts.

    Example:
        ```python
        memory = SharedWorkingMemory("task_123")

        # Agent A observes
        await memory.add_observation(
            content="User prefers dark mode",
            source_agent_id="agent_a",
            is_belief_candidate=True,
            belief_type="PREFERENCE",
        )

        # Agent B gets context (sees what A observed)
        context = await memory.get_context_for_agent(
            agent_id="agent_b",
            role=AgentRole.RESEARCHER,
        )

        # Periodic attention decay
        await memory.apply_attention_decay()
        ```

    Attributes:
        task_id: Unique identifier for this task
        config: Configuration settings
        embedding_provider: Optional provider for semantic similarity
    """

    def __init__(
        self,
        task_id: str,
        config: SharedWorkingMemoryConfig | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        """Initialize shared working memory.

        Args:
            task_id: Unique identifier for this task
            config: Configuration settings (uses defaults if None)
            embedding_provider: Optional provider for semantic similarity
        """
        self.task_id = task_id
        self.config = config or SharedWorkingMemoryConfig()
        self.embedding_provider = embedding_provider

        # Storage
        self._observations: dict[str, SharedObservation] = {}
        self._conflicts: list[tuple[str, str, str]] = []  # (obs_a, obs_b, reason)
        self._agent_views: dict[str, list[str]] = {}  # agent_id -> [obs_ids]

        # Concurrency control
        self._global_lock = asyncio.Lock()

        logger.debug(
            f"Initialized SharedWorkingMemory for task {task_id} "
            f"(max_per_agent={self.config.max_items_per_agent}, "
            f"max_total={self.config.max_total_items})"
        )

    # =========================================================================
    # Semantic Similarity (Built-in, no external dependencies)
    # =========================================================================

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text into words for similarity comparison.

        Normalizes to lowercase, removes punctuation, filters stopwords.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens
        """
        # Normalize and split
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        words = text.split()

        # Filter stopwords and short words
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "must", "shall", "can", "need", "dare",
            "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after", "above",
            "below", "between", "under", "again", "further", "then", "once", "here",
            "there", "when", "where", "why", "how", "all", "each", "few", "more",
            "most", "other", "some", "such", "no", "nor", "not", "only", "own",
            "same", "so", "than", "too", "very", "just", "and", "but", "if", "or",
            "because", "until", "while", "this", "that", "these", "those", "it",
        }

        return [w for w in words if w not in stopwords and len(w) > 2]

    def _compute_similarity(self, text_a: str, text_b: str) -> float:
        """Compute semantic similarity using TF-IDF weighted cosine similarity.

        This is a lightweight alternative to embeddings that works well for
        detecting conflicting observations about the same topic.

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            Similarity score (0-1, higher = more similar)
        """
        tokens_a = self._tokenize(text_a)
        tokens_b = self._tokenize(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        # Create term frequency vectors
        counter_a = Counter(tokens_a)
        counter_b = Counter(tokens_b)

        # Get all unique terms
        all_terms = set(counter_a.keys()) | set(counter_b.keys())

        if not all_terms:
            return 0.0

        # Compute dot product and magnitudes
        dot_product = 0.0
        magnitude_a = 0.0
        magnitude_b = 0.0

        for term in all_terms:
            freq_a = counter_a.get(term, 0)
            freq_b = counter_b.get(term, 0)

            dot_product += freq_a * freq_b
            magnitude_a += freq_a * freq_a
            magnitude_b += freq_b * freq_b

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        # Cosine similarity
        similarity = dot_product / (math.sqrt(magnitude_a) * math.sqrt(magnitude_b))

        return similarity

    def _content_differs(self, text_a: str, text_b: str) -> bool:
        """Check if two texts have meaningfully different content.

        Two observations conflict if they're about the same topic but
        make different claims. This checks for:
        - Different numbers (e.g., "3pm" vs "4pm")
        - Negation differences (e.g., "is ready" vs "is not ready")
        - Different entities (e.g., "Room A" vs "Room B")

        Args:
            text_a: First text
            text_b: Second text

        Returns:
            True if content meaningfully differs
        """
        text_a = text_a.lower()
        text_b = text_b.lower()

        # Check for different numbers
        numbers_a = set(re.findall(r"\d+(?:\.\d+)?", text_a))
        numbers_b = set(re.findall(r"\d+(?:\.\d+)?", text_b))

        if numbers_a and numbers_b and numbers_a != numbers_b:
            # Both have numbers but they differ
            return True

        # Check for negation differences
        negation_words = {"not", "no", "never", "none", "neither", "nobody", "nothing"}

        has_negation_a = any(word in text_a.split() for word in negation_words)
        has_negation_b = any(word in text_b.split() for word in negation_words)

        if has_negation_a != has_negation_b:
            # One is negated, other is not
            return True

        # Check for different named entities (capitalized words in original)
        # This catches "Room A" vs "Room B", "John" vs "Jane", etc.
        caps_a = set(re.findall(r"\b[A-Z][a-zA-Z]*\b", text_a))
        caps_b = set(re.findall(r"\b[A-Z][a-zA-Z]*\b", text_b))

        # Filter out common words that happen to be capitalized
        common_caps = {"The", "A", "An", "Is", "Are", "It", "This", "That"}
        caps_a = caps_a - common_caps
        caps_b = caps_b - common_caps

        if caps_a and caps_b and len(caps_a & caps_b) == 0:
            # Both have named entities but no overlap
            return True

        # If texts are very similar (>90% overlap), they probably don't conflict
        similarity = self._compute_similarity(text_a, text_b)
        if similarity > 0.9:
            return False

        # Default: if we can't determine, assume different (safer for conflict detection)
        return True

    # =========================================================================
    # Observation Management
    # =========================================================================

    async def add_observation(
        self,
        content: str,
        source_agent_id: str,
        *,
        attention_weight: float = 0.5,
        confidence: float = 1.0,
        is_belief_candidate: bool = False,
        belief_type: str | None = None,
    ) -> SharedObservation:
        """Add observation with automatic conflict detection.

        Steps:
        1. Create observation with UUID
        2. Acquire global lock
        3. Check for conflicts with existing observations
        4. Ensure capacity (evict if needed)
        5. Store observation
        6. Update agent view
        7. Release lock

        Args:
            content: The observation content
            source_agent_id: Which agent made this observation
            attention_weight: Initial attention (0-1, default 0.5)
            confidence: Agent's confidence (0-1, default 1.0)
            is_belief_candidate: Should this become a belief?
            belief_type: Type if belief candidate (FACT, SKILL, etc.)

        Returns:
            The created observation

        Raises:
            ValueError: If attention_weight or confidence not in [0, 1]
        """
        async with self._global_lock:
            observation = SharedObservation(
                observation_id=str(uuid.uuid4()),
                content=content,
                source_agent_id=source_agent_id,
                timestamp=datetime.now(),
                attention_weight=attention_weight,
                confidence=confidence,
                is_belief_candidate=is_belief_candidate,
                belief_type=belief_type,
            )

            # Detect conflicts
            if is_belief_candidate and belief_type:
                conflicts = await self._detect_conflicts(observation)
                if conflicts:
                    # Create new observation with conflicts using replace()
                    observation = replace(observation, conflicts_with=conflicts)
                    for conflict_id in conflicts:
                        self._conflicts.append(
                            (observation.observation_id, conflict_id, "semantic_conflict")
                        )
                    logger.debug(
                        f"Detected {len(conflicts)} conflicts for observation {observation.observation_id}"
                    )

            # Ensure capacity
            await self._ensure_capacity(source_agent_id)

            # Store
            self._observations[observation.observation_id] = observation

            # Update agent view
            if source_agent_id not in self._agent_views:
                self._agent_views[source_agent_id] = []
            self._agent_views[source_agent_id].append(observation.observation_id)

            logger.debug(
                f"Added observation {observation.observation_id} from {source_agent_id} "
                f"(total: {len(self._observations)})"
            )

            return observation

    async def _ensure_capacity(self, source_agent_id: str) -> None:
        """Evict lowest-attention items if over capacity.

        Two-level capacity enforcement:
        1. Per-agent: max items per agent (Miller's Law)
        2. Global: max total items

        Eviction strategy:
        - Lowest attention weight evicted first
        - Preserves per-agent fairness

        Args:
            source_agent_id: Agent about to add observation
        """
        # Per-agent capacity
        agent_obs = self._agent_views.get(source_agent_id, [])
        while len(agent_obs) >= self.config.max_items_per_agent:
            # Find lowest attention item from this agent
            lowest_id = min(
                agent_obs,
                key=lambda oid: self._observations[oid].attention_weight
                if oid in self._observations
                else 0,
            )
            agent_obs.remove(lowest_id)
            if lowest_id in self._observations:
                logger.debug(
                    f"Evicted observation {lowest_id} from {source_agent_id} "
                    f"(per-agent capacity: {self.config.max_items_per_agent})"
                )
                del self._observations[lowest_id]

        # Global capacity
        while len(self._observations) >= self.config.max_total_items:
            # Find lowest attention item overall
            lowest = min(self._observations.values(), key=lambda o: o.attention_weight)
            logger.debug(
                f"Evicted observation {lowest.observation_id} "
                f"(global capacity: {self.config.max_total_items})"
            )
            del self._observations[lowest.observation_id]

            # Remove from agent views
            for agent_id, obs_ids in self._agent_views.items():
                if lowest.observation_id in obs_ids:
                    obs_ids.remove(lowest.observation_id)
                    break

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    async def _detect_conflicts(self, new_observation: SharedObservation) -> list[str]:
        """Detect semantic conflicts with existing observations.

        Conflict Detection Strategy:
        1. Only check belief candidates OF THE SAME TYPE (FACT vs FACT, not FACT vs INSIGHT)
        2. Compute semantic similarity (topic overlap)
        3. Check if content actually differs (conflicting claims)
        4. A conflict is: same type + same topic + different claims

        Different belief types don't conflict - a FACT and an INSIGHT can coexist
        even if they're about the same topic.

        Uses built-in cosine similarity when no embedding provider is available.
        When embedding provider is set, uses it for higher-quality similarity.

        Args:
            new_observation: Observation to check for conflicts

        Returns:
            List of observation IDs that conflict
        """
        conflicts = []

        if not (new_observation.is_belief_candidate and new_observation.belief_type):
            return conflicts

        for obs_id, obs in self._observations.items():
            # Skip same source (agent can't conflict with itself)
            if obs.source_agent_id == new_observation.source_agent_id:
                continue

            # Only check observations that are:
            # 1. Belief candidates
            # 2. Same belief type (FACT conflicts with FACT, not with INSIGHT)
            if not obs.is_belief_candidate:
                continue
            if obs.belief_type != new_observation.belief_type:
                continue

            # Compute semantic similarity
            if self.embedding_provider is not None:
                # Use embedding provider for higher-quality similarity
                similarity = await self.embedding_provider.similarity(
                    new_observation.content,
                    obs.content,
                )
            else:
                # Use built-in cosine similarity (word overlap)
                similarity = self._compute_similarity(
                    new_observation.content,
                    obs.content,
                )

            # Check if content actually differs (conflicting claims)
            content_differs = self._content_differs(new_observation.content, obs.content)

            if not content_differs:
                # Same content = agreement, not conflict
                continue

            # High similarity + different content = semantic conflict
            # (Same topic, different claims)
            if similarity > self.config.conflict_threshold:
                conflicts.append(obs_id)
                logger.debug(
                    f"Semantic conflict detected: "
                    f"'{new_observation.content[:50]}...' vs '{obs.content[:50]}...' "
                    f"(similarity: {similarity:.2f}, type: {new_observation.belief_type})"
                )
            else:
                # Low text similarity but same belief type - check for shared key terms
                # This catches cases like "The price is $100" vs "The price is $200"
                # where the number difference drops similarity but it's clearly same topic
                tokens_new = set(self._tokenize(new_observation.content))
                tokens_old = set(self._tokenize(obs.content))
                shared_tokens = tokens_new & tokens_old

                # Need at least 1 shared meaningful term to consider it same topic
                # (content_differs already verified they make different claims)
                if len(shared_tokens) >= 1:
                    conflicts.append(obs_id)
                    logger.debug(
                        f"Term-based conflict detected: "
                        f"'{new_observation.content[:50]}...' vs '{obs.content[:50]}...' "
                        f"(shared terms: {shared_tokens}, type: {new_observation.belief_type})"
                    )

        return conflicts

    async def flag_conflict(
        self,
        observation_a_id: str,
        observation_b_id: str,
        conflict_reason: str,
    ) -> None:
        """Explicitly flag a conflict for reconciliation.

        Called by orchestrator when it detects a conflict that
        automatic detection missed.

        Args:
            observation_a_id: First observation ID
            observation_b_id: Second observation ID
            conflict_reason: Why they conflict
        """
        self._conflicts.append((observation_a_id, observation_b_id, conflict_reason))

        # Update conflicts_with for both observations using replace()
        if observation_a_id in self._observations:
            obs_a = self._observations[observation_a_id]
            if observation_b_id not in obs_a.conflicts_with:
                conflicts_a = list(obs_a.conflicts_with) + [observation_b_id]
                self._observations[observation_a_id] = replace(
                    obs_a, conflicts_with=conflicts_a
                )

        if observation_b_id in self._observations:
            obs_b = self._observations[observation_b_id]
            if observation_a_id not in obs_b.conflicts_with:
                conflicts_b = list(obs_b.conflicts_with) + [observation_a_id]
                self._observations[observation_b_id] = replace(
                    obs_b, conflicts_with=conflicts_b
                )

        logger.info(
            f"Flagged conflict: {observation_a_id} vs {observation_b_id} ({conflict_reason})"
        )

    async def get_conflicts(
        self,
    ) -> list[tuple[SharedObservation, SharedObservation, str]]:
        """Get all unresolved conflicts for reconciliation.

        Returns:
            List of (observation_a, observation_b, reason) tuples
        """
        result = []
        for obs_a_id, obs_b_id, reason in self._conflicts:
            obs_a = self._observations.get(obs_a_id)
            obs_b = self._observations.get(obs_b_id)
            if obs_a and obs_b:
                result.append((obs_a, obs_b, reason))
        return result

    # =========================================================================
    # Attention Management
    # =========================================================================

    async def apply_attention_decay(self) -> None:
        """Decay attention weights (called periodically).

        Multiplies all attention weights by decay factor (default 0.9).
        This simulates cognitive attention decay over time.
        """
        for obs_id, obs in self._observations.items():
            new_weight = obs.attention_weight * self.config.attention_decay_factor
            self._observations[obs_id] = replace(obs, attention_weight=new_weight)
        logger.debug(
            f"Applied attention decay (factor={self.config.attention_decay_factor})"
        )

    async def boost_attention(self, observation_id: str, boost: float = 0.2) -> None:
        """Boost attention for a specific observation.

        Called when an observation becomes relevant again.

        Args:
            observation_id: Observation to boost
            boost: Amount to increase attention (default 0.2)
        """
        if observation_id in self._observations:
            obs = self._observations[observation_id]
            new_weight = min(1.0, obs.attention_weight + boost)
            self._observations[observation_id] = replace(obs, attention_weight=new_weight)
            logger.debug(
                f"Boosted attention for {observation_id}: "
                f"{obs.attention_weight:.2f} -> {new_weight:.2f}"
            )

    # =========================================================================
    # Context Retrieval
    # =========================================================================

    async def get_context_for_agent(
        self,
        agent_id: str,
        role: AgentRole,
        max_items: int | None = None,
    ) -> list[SharedObservation]:
        """Get relevant context filtered by agent role.

        Filtering by role:
        - CRITIC: Only belief candidates
        - RESEARCHER: All observations
        - EXECUTOR: Only SKILL and FACT types
        - Other roles: All observations

        Sorting:
        - Primary: attention_weight (descending)
        - Secondary: timestamp (descending - recent first)

        Access tracking:
        - Updates accessed_by set
        - Increments access_count

        Args:
            agent_id: Agent requesting context
            role: Agent's role (determines filtering)
            max_items: Max items to return (defaults to config.max_items_per_agent)

        Returns:
            List of observations sorted by relevance
        """
        max_items = max_items or self.config.max_items_per_agent

        # Get all observations
        all_obs = list(self._observations.values())

        # Filter by role
        relevant = self._filter_by_role(all_obs, role)

        # Sort by attention weight + recency
        sorted_obs = sorted(
            relevant,
            key=lambda o: (o.attention_weight, o.timestamp.timestamp()),
            reverse=True,
        )

        # Take top N
        result = sorted_obs[:max_items]

        # Track access using replace()
        for obs in result:
            accessed_by = set(obs.accessed_by)
            accessed_by.add(agent_id)
            self._observations[obs.observation_id] = replace(
                obs,
                accessed_by=accessed_by,
                access_count=obs.access_count + 1,
            )

        logger.debug(
            f"Retrieved {len(result)} observations for {agent_id} (role={role.value})"
        )
        return result

    def _filter_by_role(
        self,
        observations: list[SharedObservation],
        role: AgentRole,
    ) -> list[SharedObservation]:
        """Filter observations by role relevance.

        Args:
            observations: All observations
            role: Agent role

        Returns:
            Filtered observations relevant to role
        """
        if role == AgentRole.CRITIC:
            # Critics see claims and assertions
            return [o for o in observations if o.is_belief_candidate]
        elif role == AgentRole.RESEARCHER:
            # Researchers see everything
            return observations
        elif role == AgentRole.EXECUTOR:
            # Executors see action-related observations
            return [o for o in observations if o.belief_type in ("SKILL", "FACT", None)]
        else:
            # Default: see everything
            return observations

    # =========================================================================
    # Belief Candidates
    # =========================================================================

    async def get_belief_candidates(self) -> list[SharedObservation]:
        """Get observations that should become beliefs.

        Returns only non-conflicting candidates.
        Candidates with conflicts excluded until reconciled.

        Returns:
            List of belief candidate observations
        """
        return [
            obs
            for obs in self._observations.values()
            if obs.is_belief_candidate and not obs.conflicts_with
        ]
