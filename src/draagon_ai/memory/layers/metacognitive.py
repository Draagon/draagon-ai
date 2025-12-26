"""Metacognitive Memory Layer - Skills, strategies, and self-knowledge.

Metacognitive memory stores:
- Skills (procedural knowledge)
- Strategies (problem-solving approaches)
- Insights (meta-learning patterns)
- Behaviors (agent behaviors as graph citizens)

Features:
- Effectiveness tracking (success/failure counts)
- Version history (evolution of skills)
- Self-improvement capability
- Cross-agent sharing potential

Based on research from:
- ACE Framework: Self-referential evolution
- Promptbreeder: Prompt mutation patterns
- Cognitive science: Metacognition and procedural memory
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import logging

from ..temporal_nodes import TemporalNode, NodeType, EdgeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph, GraphSearchResult
from .base import MemoryLayerBase, LayerConfig

logger = logging.getLogger(__name__)


# No TTL for metacognitive by default (permanent)
DEFAULT_TTL = None


@dataclass
class Skill(TemporalNode):
    """A skill (procedural knowledge).

    Skills represent how to do something:
    - Shell commands
    - API usage patterns
    - Problem-solving procedures
    """

    skill_name: str = ""
    skill_type: str = ""  # command, api, procedure, template
    procedure: str = ""

    # Effectiveness tracking
    success_count: int = 0
    failure_count: int = 0
    last_used: datetime | None = None

    # Version history
    version: int = 1
    parent_skill_id: str | None = None

    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness as success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5  # Unknown, assume neutral
        return self.success_count / total

    @property
    def needs_improvement(self) -> bool:
        """Check if skill needs improvement based on effectiveness."""
        total = self.success_count + self.failure_count
        if total < 3:
            return False  # Not enough data
        return self.effectiveness_score < 0.6


@dataclass
class Strategy(TemporalNode):
    """A problem-solving strategy.

    Strategies are higher-level than skills:
    - When to use which approach
    - Multi-step problem solving
    - Context-dependent decisions
    """

    strategy_name: str = ""
    strategy_type: str = ""  # reasoning, decision, planning
    description: str = ""
    applicable_contexts: list[str] = field(default_factory=list)

    # Effectiveness
    success_count: int = 0
    failure_count: int = 0

    # Components
    skill_ids: list[str] = field(default_factory=list)  # Skills this uses

    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness as success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class Insight(TemporalNode):
    """A meta-learning insight.

    Insights are observations about patterns:
    - What works in which situations
    - Failure patterns to avoid
    - Optimization opportunities
    """

    insight_type: str = ""  # pattern, anti_pattern, optimization
    context: str = ""
    recommendation: str = ""
    evidence_count: int = 1


@dataclass
class BehaviorNode(TemporalNode):
    """A behavior as a graph citizen.

    Behaviors are first-class nodes that:
    - Track their own effectiveness
    - Can evolve over time
    - Share knowledge with other behaviors
    """

    behavior_id: str = ""
    behavior_name: str = ""
    description: str = ""

    # Graph relationships
    depends_on: list[str] = field(default_factory=list)  # Other behaviors
    can_delegate_to: list[str] = field(default_factory=list)

    # Knowledge scopes
    learns_from_scope: str = ""
    contributes_to_scope: str = ""

    # Effectiveness
    execution_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # Evolution
    evolution_fitness: float = 0.5
    evolution_generation: int = 0
    prompt_version: int = 1

    # Self-improvement
    can_self_improve: bool = True
    last_improvement: datetime | None = None
    improvement_cooldown: timedelta = field(default_factory=lambda: timedelta(hours=24))

    @property
    def effectiveness_score(self) -> float:
        """Calculate effectiveness as success rate."""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total

    @property
    def ready_for_improvement(self) -> bool:
        """Check if behavior is ready for self-improvement."""
        if not self.can_self_improve:
            return False
        if self.last_improvement is None:
            return True
        return datetime.now() - self.last_improvement >= self.improvement_cooldown


class MetacognitiveMemory(MemoryLayerBase[Skill]):
    """Metacognitive Memory Layer - Procedural and strategic knowledge.

    Key features:
    - Skill CRUD with effectiveness tracking
    - Strategy management
    - Insight collection
    - Behavior integration
    - Version history for evolution

    Example:
        meta = MetacognitiveMemory(graph)

        # Add a skill
        skill = await meta.add_skill(
            name="restart_plex",
            skill_type="command",
            procedure="docker restart plex",
        )

        # Record success/failure
        await meta.record_skill_result(skill.node_id, success=True)

        # Get effective skills
        skills = await meta.get_effective_skills(min_effectiveness=0.8)

        # Add strategy
        strategy = await meta.add_strategy(
            name="media_troubleshooting",
            description="Steps to diagnose media server issues",
            skill_ids=[skill.node_id],
        )
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        ttl: timedelta | None = DEFAULT_TTL,
    ):
        """Initialize metacognitive memory.

        Args:
            graph: The underlying temporal cognitive graph
            ttl: Time-to-live (None = permanent)
        """
        config = LayerConfig(
            max_items=None,
            default_ttl=ttl,
            decay_factor=0.99,  # Very slow decay for metacognitive
            decay_interval=timedelta(days=7),
            importance_threshold=0.9,  # Very high bar (nothing above meta)
            access_threshold=20,
            auto_promote=False,  # No layer above metacognitive
            node_types=[NodeType.SKILL, NodeType.STRATEGY, NodeType.INSIGHT, NodeType.BEHAVIOR],
        )
        super().__init__(graph, config, MemoryLayer.METACOGNITIVE)

        # Skill name index
        self._skill_index: dict[str, str] = {}  # lowercase name -> skill_id

    async def add(
        self,
        content: str,
        *,
        node_type: NodeType = NodeType.SKILL,
        scope_id: str = "agent:default",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TemporalNode:
        """Add a metacognitive item.

        For specific types, prefer:
        - add_skill() for skills
        - add_strategy() for strategies
        - add_insight() for insights

        Args:
            content: The content
            node_type: Type of node
            scope_id: Hierarchical scope
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The created node
        """
        node = await self._graph.add_node(
            content=content,
            node_type=node_type,
            scope_id=scope_id,
            entities=entities,
            importance=0.85,  # Metacognitive is very important
            metadata=metadata or {},
        )
        return node

    async def get(self, node_id: str) -> TemporalNode | None:
        """Get a metacognitive item by ID.

        Args:
            node_id: The node ID

        Returns:
            The node or None
        """
        node = await self._graph.get_node(node_id)
        if not node or node.layer != MemoryLayer.METACOGNITIVE:
            return None
        return node

    # =========================================================================
    # Skill Operations
    # =========================================================================

    async def add_skill(
        self,
        name: str,
        skill_type: str,
        procedure: str,
        *,
        scope_id: str = "agent:default",
        entities: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Skill:
        """Add a new skill.

        Args:
            name: Skill name
            skill_type: Type of skill
            procedure: How to do it
            scope_id: Hierarchical scope
            entities: Extracted entities
            metadata: Additional metadata

        Returns:
            The created Skill
        """
        # Check for existing skill
        existing = await self.get_skill_by_name(name)
        if existing:
            logger.warning(f"Skill '{name}' already exists, returning existing")
            return existing

        node = await self._graph.add_node(
            content=f"Skill: {name} - {procedure}",
            node_type=NodeType.SKILL,
            scope_id=scope_id,
            entities=entities,
            importance=0.85,
            metadata={
                **(metadata or {}),
                "skill_name": name,
                "skill_type": skill_type,
                "procedure": procedure,
                "success_count": 0,
                "failure_count": 0,
                "version": 1,
            },
        )

        # Index
        self._skill_index[name.lower()] = node.node_id

        skill = Skill(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            skill_name=name,
            skill_type=skill_type,
            procedure=procedure,
            version=1,
        )

        logger.debug(f"Added skill: {name}")
        return skill

    async def get_skill(self, skill_id: str) -> Skill | None:
        """Get a skill by ID.

        Args:
            skill_id: The skill node ID

        Returns:
            The Skill or None
        """
        node = await self._graph.get_node(skill_id)
        if not node or node.node_type != NodeType.SKILL:
            return None

        return Skill(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            skill_name=node.metadata.get("skill_name", ""),
            skill_type=node.metadata.get("skill_type", ""),
            procedure=node.metadata.get("procedure", ""),
            success_count=node.metadata.get("success_count", 0),
            failure_count=node.metadata.get("failure_count", 0),
            last_used=datetime.fromisoformat(node.metadata["last_used"]) if node.metadata.get("last_used") else None,
            version=node.metadata.get("version", 1),
            parent_skill_id=node.metadata.get("parent_skill_id"),
        )

    async def get_skill_by_name(self, name: str) -> Skill | None:
        """Get a skill by name.

        Args:
            name: Skill name

        Returns:
            The Skill or None
        """
        skill_id = self._skill_index.get(name.lower())
        if skill_id:
            return await self.get_skill(skill_id)
        return None

    async def record_skill_result(
        self,
        skill_id: str,
        success: bool,
        context: str | None = None,
    ) -> bool:
        """Record a skill execution result.

        Args:
            skill_id: The skill
            success: Whether it succeeded
            context: Optional context

        Returns:
            True if recorded
        """
        node = await self._graph.get_node(skill_id)
        if not node or node.node_type != NodeType.SKILL:
            return False

        if success:
            node.metadata["success_count"] = node.metadata.get("success_count", 0) + 1
        else:
            node.metadata["failure_count"] = node.metadata.get("failure_count", 0) + 1

        node.metadata["last_used"] = datetime.now().isoformat()
        node.reinforce(0.05 if success else -0.05)

        logger.debug(
            f"Recorded skill result: {node.metadata.get('skill_name')} "
            f"{'success' if success else 'failure'}"
        )
        return True

    async def improve_skill(
        self,
        skill_id: str,
        new_procedure: str,
        reason: str = "",
    ) -> Skill | None:
        """Create an improved version of a skill.

        Args:
            skill_id: Original skill
            new_procedure: Improved procedure
            reason: Why it was improved

        Returns:
            The new Skill or None
        """
        old_skill = await self.get_skill(skill_id)
        if not old_skill:
            return None

        # Create new version
        new_node = await self._graph.supersede_node(
            old_node_id=skill_id,
            new_content=f"Skill: {old_skill.skill_name} - {new_procedure}",
        )

        if not new_node:
            return None

        # Update metadata
        new_node.metadata["skill_name"] = old_skill.skill_name
        new_node.metadata["skill_type"] = old_skill.skill_type
        new_node.metadata["procedure"] = new_procedure
        new_node.metadata["version"] = old_skill.version + 1
        new_node.metadata["parent_skill_id"] = skill_id
        new_node.metadata["improvement_reason"] = reason
        new_node.metadata["success_count"] = 0
        new_node.metadata["failure_count"] = 0

        # Update index
        self._skill_index[old_skill.skill_name.lower()] = new_node.node_id

        logger.info(f"Improved skill: {old_skill.skill_name} v{old_skill.version} -> v{old_skill.version + 1}")
        return await self.get_skill(new_node.node_id)

    async def get_effective_skills(
        self,
        min_effectiveness: float = 0.6,
        min_uses: int = 3,
        scope_id: str | None = None,
    ) -> list[Skill]:
        """Get skills above effectiveness threshold.

        Args:
            min_effectiveness: Minimum success rate
            min_uses: Minimum total uses
            scope_id: Optional scope filter

        Returns:
            List of effective skills
        """
        skills = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.SKILL:
                continue
            if node.layer != MemoryLayer.METACOGNITIVE:
                continue
            if scope_id and node.scope_id != scope_id:
                continue

            success = node.metadata.get("success_count", 0)
            failure = node.metadata.get("failure_count", 0)
            total = success + failure

            if total < min_uses:
                continue

            effectiveness = success / total if total > 0 else 0
            if effectiveness >= min_effectiveness:
                skill = await self.get_skill(node_id)
                if skill:
                    skills.append(skill)

        # Sort by effectiveness
        skills.sort(key=lambda s: s.effectiveness_score, reverse=True)
        return skills

    async def get_skills_needing_improvement(self) -> list[Skill]:
        """Get skills that need improvement.

        Returns:
            List of underperforming skills
        """
        skills = []

        for node_id, node in self._graph._nodes.items():
            if node.node_type != NodeType.SKILL:
                continue
            if node.layer != MemoryLayer.METACOGNITIVE:
                continue

            skill = await self.get_skill(node_id)
            if skill and skill.needs_improvement:
                skills.append(skill)

        return skills

    # =========================================================================
    # Strategy Operations
    # =========================================================================

    async def add_strategy(
        self,
        name: str,
        strategy_type: str,
        description: str,
        *,
        applicable_contexts: list[str] | None = None,
        skill_ids: list[str] | None = None,
        scope_id: str = "agent:default",
        metadata: dict[str, Any] | None = None,
    ) -> Strategy:
        """Add a new strategy.

        Args:
            name: Strategy name
            strategy_type: Type of strategy
            description: What it does
            applicable_contexts: When to use it
            skill_ids: Skills it uses
            scope_id: Hierarchical scope
            metadata: Additional metadata

        Returns:
            The created Strategy
        """
        node = await self._graph.add_node(
            content=f"Strategy: {name} - {description}",
            node_type=NodeType.STRATEGY,
            scope_id=scope_id,
            entities=[name],
            importance=0.9,  # Strategies are very important
            metadata={
                **(metadata or {}),
                "strategy_name": name,
                "strategy_type": strategy_type,
                "description": description,
                "applicable_contexts": applicable_contexts or [],
                "skill_ids": skill_ids or [],
                "success_count": 0,
                "failure_count": 0,
            },
        )

        # Link to skills
        for skill_id in (skill_ids or []):
            await self._graph.add_edge(
                source_id=node.node_id,
                target_id=skill_id,
                edge_type=EdgeType.ENABLES,
                label="uses_skill",
            )

        strategy = Strategy(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            strategy_name=name,
            strategy_type=strategy_type,
            description=description,
            applicable_contexts=applicable_contexts or [],
            skill_ids=skill_ids or [],
        )

        logger.debug(f"Added strategy: {name}")
        return strategy

    async def get_strategy(self, strategy_id: str) -> Strategy | None:
        """Get a strategy by ID.

        Args:
            strategy_id: The strategy node ID

        Returns:
            The Strategy or None
        """
        node = await self._graph.get_node(strategy_id)
        if not node or node.node_type != NodeType.STRATEGY:
            return None

        return Strategy(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            strategy_name=node.metadata.get("strategy_name", ""),
            strategy_type=node.metadata.get("strategy_type", ""),
            description=node.metadata.get("description", ""),
            applicable_contexts=node.metadata.get("applicable_contexts", []),
            skill_ids=node.metadata.get("skill_ids", []),
            success_count=node.metadata.get("success_count", 0),
            failure_count=node.metadata.get("failure_count", 0),
        )

    async def find_strategy_for_context(
        self,
        context: str,
        limit: int = 5,
    ) -> list[Strategy]:
        """Find strategies applicable to a context.

        Args:
            context: The context to match
            limit: Maximum strategies

        Returns:
            List of matching strategies
        """
        # Semantic search
        results = await self._graph.search(
            query=context,
            node_types=[NodeType.STRATEGY],
            layers=[MemoryLayer.METACOGNITIVE],
            limit=limit,
        )

        strategies = []
        for result in results:
            strategy = await self.get_strategy(result.node.node_id)
            if strategy:
                strategies.append(strategy)

        return strategies

    # =========================================================================
    # Insight Operations
    # =========================================================================

    async def add_insight(
        self,
        content: str,
        insight_type: str,
        *,
        context: str = "",
        recommendation: str = "",
        scope_id: str = "agent:default",
        metadata: dict[str, Any] | None = None,
    ) -> Insight:
        """Add a meta-learning insight.

        Args:
            content: The insight
            insight_type: Type of insight
            context: When this applies
            recommendation: What to do about it
            scope_id: Hierarchical scope
            metadata: Additional metadata

        Returns:
            The created Insight
        """
        # Check for similar insights
        existing = await self._graph.search(
            query=content,
            node_types=[NodeType.INSIGHT],
            layers=[MemoryLayer.METACOGNITIVE],
            min_score=0.9,
            limit=1,
        )

        if existing:
            # Reinforce existing insight
            existing_node = existing[0].node
            existing_node.metadata["evidence_count"] = existing_node.metadata.get("evidence_count", 1) + 1
            existing_node.restate()
            logger.debug(f"Reinforced existing insight: {existing_node.node_id[:8]}...")

            return Insight(
                node_id=existing_node.node_id,
                content=existing_node.content,
                node_type=existing_node.node_type,
                scope_id=existing_node.scope_id,
                embedding=existing_node.embedding,
                event_time=existing_node.event_time,
                ingestion_time=existing_node.ingestion_time,
                valid_from=existing_node.valid_from,
                valid_until=existing_node.valid_until,
                confidence=existing_node.confidence,
                importance=existing_node.importance,
                stated_count=existing_node.stated_count,
                access_count=existing_node.access_count,
                entities=existing_node.entities,
                metadata=existing_node.metadata,
                created_at=existing_node.created_at,
                updated_at=existing_node.updated_at,
                insight_type=existing_node.metadata.get("insight_type", ""),
                context=existing_node.metadata.get("context", ""),
                recommendation=existing_node.metadata.get("recommendation", ""),
                evidence_count=existing_node.metadata.get("evidence_count", 1),
            )

        node = await self._graph.add_node(
            content=content,
            node_type=NodeType.INSIGHT,
            scope_id=scope_id,
            importance=0.8,
            metadata={
                **(metadata or {}),
                "insight_type": insight_type,
                "context": context,
                "recommendation": recommendation,
                "evidence_count": 1,
            },
        )

        insight = Insight(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            insight_type=insight_type,
            context=context,
            recommendation=recommendation,
            evidence_count=1,
        )

        logger.debug(f"Added insight: {content[:50]}...")
        return insight

    # =========================================================================
    # Behavior Operations
    # =========================================================================

    async def register_behavior(
        self,
        behavior_id: str,
        behavior_name: str,
        description: str,
        *,
        scope_id: str = "agent:default",
        depends_on: list[str] | None = None,
        can_delegate_to: list[str] | None = None,
        learns_from_scope: str = "",
        contributes_to_scope: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> BehaviorNode:
        """Register a behavior as a graph citizen.

        Args:
            behavior_id: Unique behavior ID
            behavior_name: Human-readable name
            description: What the behavior does
            scope_id: Hierarchical scope
            depends_on: Behaviors this requires
            can_delegate_to: Behaviors it can hand off to
            learns_from_scope: Where it reads knowledge
            contributes_to_scope: Where it writes knowledge
            metadata: Additional metadata

        Returns:
            The created BehaviorNode
        """
        node = await self._graph.add_node(
            content=f"Behavior: {behavior_name} - {description}",
            node_type=NodeType.BEHAVIOR,
            scope_id=scope_id,
            entities=[behavior_name],
            importance=0.9,
            metadata={
                **(metadata or {}),
                "behavior_id": behavior_id,
                "behavior_name": behavior_name,
                "description": description,
                "depends_on": depends_on or [],
                "can_delegate_to": can_delegate_to or [],
                "learns_from_scope": learns_from_scope,
                "contributes_to_scope": contributes_to_scope,
                "execution_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "evolution_fitness": 0.5,
                "evolution_generation": 0,
                "prompt_version": 1,
                "can_self_improve": True,
            },
        )

        # Create dependency edges
        for dep_id in (depends_on or []):
            await self._graph.add_edge(
                source_id=node.node_id,
                target_id=dep_id,
                edge_type=EdgeType.ENABLES,
                label="depends_on",
            )

        # Create delegation edges
        for del_id in (can_delegate_to or []):
            await self._graph.add_edge(
                source_id=node.node_id,
                target_id=del_id,
                edge_type=EdgeType.DELEGATES_TO,
            )

        behavior = BehaviorNode(
            node_id=node.node_id,
            content=node.content,
            node_type=node.node_type,
            scope_id=node.scope_id,
            embedding=node.embedding,
            event_time=node.event_time,
            ingestion_time=node.ingestion_time,
            valid_from=node.valid_from,
            valid_until=node.valid_until,
            confidence=node.confidence,
            importance=node.importance,
            stated_count=node.stated_count,
            access_count=node.access_count,
            entities=node.entities,
            metadata=node.metadata,
            created_at=node.created_at,
            updated_at=node.updated_at,
            behavior_id=behavior_id,
            behavior_name=behavior_name,
            description=description,
            depends_on=depends_on or [],
            can_delegate_to=can_delegate_to or [],
            learns_from_scope=learns_from_scope,
            contributes_to_scope=contributes_to_scope,
        )

        logger.debug(f"Registered behavior: {behavior_name}")
        return behavior

    async def record_behavior_execution(
        self,
        behavior_node_id: str,
        success: bool,
        fitness_delta: float = 0.0,
    ) -> bool:
        """Record a behavior execution result.

        Args:
            behavior_node_id: The behavior node
            success: Whether it succeeded
            fitness_delta: Optional fitness adjustment

        Returns:
            True if recorded
        """
        node = await self._graph.get_node(behavior_node_id)
        if not node or node.node_type != NodeType.BEHAVIOR:
            return False

        node.metadata["execution_count"] = node.metadata.get("execution_count", 0) + 1

        if success:
            node.metadata["success_count"] = node.metadata.get("success_count", 0) + 1
        else:
            node.metadata["failure_count"] = node.metadata.get("failure_count", 0) + 1

        # Update fitness
        current_fitness = node.metadata.get("evolution_fitness", 0.5)
        new_fitness = max(0.0, min(1.0, current_fitness + fitness_delta))
        node.metadata["evolution_fitness"] = new_fitness

        node.updated_at = datetime.now()

        return True

    def stats(self) -> dict[str, Any]:
        """Get metacognitive memory statistics.

        Returns:
            Dictionary with stats
        """
        skill_count = 0
        strategy_count = 0
        insight_count = 0
        behavior_count = 0
        total_skill_uses = 0
        total_skill_successes = 0

        for node in self._graph._nodes.values():
            if node.layer != MemoryLayer.METACOGNITIVE:
                continue

            if node.node_type == NodeType.SKILL:
                skill_count += 1
                total_skill_uses += node.metadata.get("success_count", 0) + node.metadata.get("failure_count", 0)
                total_skill_successes += node.metadata.get("success_count", 0)
            elif node.node_type == NodeType.STRATEGY:
                strategy_count += 1
            elif node.node_type == NodeType.INSIGHT:
                insight_count += 1
            elif node.node_type == NodeType.BEHAVIOR:
                behavior_count += 1

        return {
            "skill_count": skill_count,
            "strategy_count": strategy_count,
            "insight_count": insight_count,
            "behavior_count": behavior_count,
            "total_skill_uses": total_skill_uses,
            "avg_skill_effectiveness": total_skill_successes / total_skill_uses if total_skill_uses else 0,
            "indexed_skills": len(self._skill_index),
        }
