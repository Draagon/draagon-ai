"""Memory Promotion Service - Automatic promotion between memory layers.

Handles the flow of memories from transient to permanent storage:
- Working → Episodic: Important items become conversation episodes
- Episodic → Semantic: Entities and facts extracted to semantic layer
- Semantic → Metacognitive: Patterns and skills elevated

Promotion criteria:
- Importance threshold: High importance items are promoted
- Access threshold: Frequently accessed items are promoted
- Time threshold: Items surviving long enough are promoted

Based on research from:
- Multi-store memory model (Atkinson-Shiffrin)
- Zep/Graphiti: Temporal knowledge promotion
- Cognitive science: Memory consolidation
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
import logging

from ..temporal_nodes import TemporalNode, NodeType, EdgeType, MemoryLayer
from ..temporal_graph import TemporalCognitiveGraph
from .base import PromotionResult
from .working import WorkingMemory, WorkingMemoryItem
from .episodic import EpisodicMemory, Episode, Event
from .semantic import SemanticMemory, Entity, Fact
from .metacognitive import MetacognitiveMemory, Skill, Insight

logger = logging.getLogger(__name__)


@dataclass
class PromotionConfig:
    """Configuration for memory promotion."""

    # Thresholds for Working → Episodic
    working_importance_threshold: float = 0.7
    working_access_threshold: int = 3
    working_min_age: timedelta = timedelta(minutes=5)

    # Thresholds for Episodic → Semantic
    episodic_importance_threshold: float = 0.75
    episodic_access_threshold: int = 5
    episodic_min_age: timedelta = timedelta(hours=1)

    # Thresholds for Semantic → Metacognitive
    semantic_importance_threshold: float = 0.85
    semantic_access_threshold: int = 10
    semantic_min_age: timedelta = timedelta(days=7)

    # Processing limits
    batch_size: int = 50
    max_promotions_per_cycle: int = 100


@dataclass
class PromotionStats:
    """Statistics from a promotion cycle."""

    working_to_episodic: int = 0
    episodic_to_semantic: int = 0
    semantic_to_metacognitive: int = 0
    total_promoted: int = 0
    total_candidates: int = 0
    duration_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Promoted {self.total_promoted}/{self.total_candidates} "
            f"(W→E: {self.working_to_episodic}, E→S: {self.episodic_to_semantic}, "
            f"S→M: {self.semantic_to_metacognitive}) in {self.duration_ms:.1f}ms"
        )


class MemoryPromotion:
    """Service for automatic memory promotion between layers.

    The promotion service manages the flow of memories from transient
    to permanent storage. Memories that meet importance and access
    thresholds are elevated to higher-level representations.

    Example:
        promotion = MemoryPromotion(
            graph=graph,
            working=working_memory,
            episodic=episodic_memory,
            semantic=semantic_memory,
            metacognitive=metacognitive_memory,
        )

        # Run a promotion cycle
        stats = await promotion.promote_all()
        print(f"Promoted {stats.total_promoted} items")

        # Promote specific layer
        result = await promotion.promote_working_to_episodic()
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        metacognitive: MetacognitiveMemory,
        config: PromotionConfig | None = None,
    ):
        """Initialize the promotion service.

        Args:
            graph: The underlying temporal cognitive graph
            working: Working memory layer
            episodic: Episodic memory layer
            semantic: Semantic memory layer
            metacognitive: Metacognitive memory layer
            config: Promotion configuration
        """
        self._graph = graph
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._metacognitive = metacognitive
        self._config = config or PromotionConfig()

        self._last_promotion = datetime.now()
        self._promotion_count = 0

    @property
    def config(self) -> PromotionConfig:
        """Get the promotion configuration."""
        return self._config

    async def promote_all(self) -> PromotionStats:
        """Run a full promotion cycle across all layers.

        Returns:
            Statistics from the promotion cycle
        """
        start = datetime.now()
        stats = PromotionStats()

        try:
            # Working → Episodic
            result = await self.promote_working_to_episodic()
            stats.working_to_episodic = result.promoted_count
            stats.total_candidates += len(result.promoted_ids) + len(result.failed_ids)

            # Episodic → Semantic
            result = await self.promote_episodic_to_semantic()
            stats.episodic_to_semantic = result.promoted_count
            stats.total_candidates += len(result.promoted_ids) + len(result.failed_ids)

            # Semantic → Metacognitive
            result = await self.promote_semantic_to_metacognitive()
            stats.semantic_to_metacognitive = result.promoted_count
            stats.total_candidates += len(result.promoted_ids) + len(result.failed_ids)

            stats.total_promoted = (
                stats.working_to_episodic +
                stats.episodic_to_semantic +
                stats.semantic_to_metacognitive
            )

        except Exception as e:
            stats.errors.append(str(e))
            logger.exception("Error during promotion cycle")

        stats.duration_ms = (datetime.now() - start).total_seconds() * 1000
        self._last_promotion = datetime.now()
        self._promotion_count += stats.total_promoted

        logger.info(str(stats))
        return stats

    async def promote_working_to_episodic(self) -> PromotionResult:
        """Promote important working memory items to episodic memory.

        Working memory items that meet importance/access thresholds
        are converted to events within an episode.

        Returns:
            Result of the promotion
        """
        result = PromotionResult(
            source_layer=MemoryLayer.WORKING,
            target_layer=MemoryLayer.EPISODIC,
            promoted_count=0,
        )

        now = datetime.now()
        candidates = await self._working.get_promotion_candidates()

        # Filter by minimum age
        eligible = [
            c for c in candidates
            if (now - c.ingestion_time) >= self._config.working_min_age
        ]

        if not eligible:
            return result

        # Create or get current episode
        current_episode = self._episodic.get_current_episode()
        if not current_episode:
            # Start a new episode for promoted items
            current_episode = await self._episodic.start_episode(
                episode_type="promoted_working",
            )

        for item in eligible[:self._config.batch_size]:
            try:
                # Create event from working memory item
                event = await self._episodic.add_event(
                    episode_id=current_episode.node_id,
                    content=item.content,
                    event_type="promoted_from_working",
                    metadata={
                        "original_id": item.node_id,
                        "attention_weight": item.attention_weight,
                        "activation_level": item.activation_level,
                        "source": item.source,
                    },
                )

                if event:
                    # Delete from working memory
                    await self._working.delete(item.node_id)

                    result.promoted_ids.append(item.node_id)
                    result.promoted_count += 1

                    logger.debug(f"Promoted working item {item.node_id[:8]}... to episodic")

            except Exception as e:
                result.failed_ids.append(item.node_id)
                logger.warning(f"Failed to promote working item {item.node_id}: {e}")

        return result

    async def promote_episodic_to_semantic(self) -> PromotionResult:
        """Promote episodic memories to semantic knowledge.

        Closed episodes that meet thresholds have their entities and
        facts extracted to the semantic layer.

        Returns:
            Result of the promotion
        """
        result = PromotionResult(
            source_layer=MemoryLayer.EPISODIC,
            target_layer=MemoryLayer.SEMANTIC,
            promoted_count=0,
        )

        now = datetime.now()
        candidates = await self._episodic.get_promotion_candidates()

        # Filter: only closed episodes that are old enough
        eligible = [
            c for c in candidates
            if (
                isinstance(c, Episode) and
                c.closed and
                (now - c.ingestion_time) >= self._config.episodic_min_age
            )
        ]

        if not eligible:
            return result

        for episode in eligible[:self._config.batch_size]:
            try:
                # Extract entities from episode
                entities_created = await self._extract_entities_from_episode(episode)

                # Extract facts from episode
                facts_created = await self._extract_facts_from_episode(episode)

                if entities_created or facts_created:
                    # Mark episode as promoted (don't delete - keep for context)
                    episode.metadata["promoted_to_semantic"] = True
                    episode.metadata["promoted_at"] = now.isoformat()

                    result.promoted_ids.append(episode.node_id)
                    result.promoted_count += 1

                    logger.debug(
                        f"Promoted episode {episode.node_id[:8]}... to semantic "
                        f"({entities_created} entities, {facts_created} facts)"
                    )

            except Exception as e:
                result.failed_ids.append(episode.node_id)
                logger.warning(f"Failed to promote episode {episode.node_id}: {e}")

        return result

    async def _extract_entities_from_episode(self, episode: Episode) -> int:
        """Extract entities from an episode.

        Args:
            episode: The episode to process

        Returns:
            Number of entities created
        """
        count = 0

        # Get all events in episode
        events = await self._episodic.get_events(episode.node_id)

        for event in events:
            for entity_name in event.entities:
                # Try to resolve existing entity
                matches = await self._semantic.resolve_entity(entity_name, min_score=0.85)

                if matches:
                    # Link episode to existing entity
                    existing = matches[0]
                    if episode.node_id not in existing.entity.source_episode_ids:
                        existing.entity.source_episode_ids.append(episode.node_id)
                else:
                    # Create new entity
                    await self._semantic.create_entity(
                        name=entity_name,
                        entity_type="extracted",
                        source_episode_ids=[episode.node_id],
                        metadata={"extracted_from": episode.node_id},
                    )
                    count += 1

        return count

    async def _extract_facts_from_episode(self, episode: Episode) -> int:
        """Extract facts from an episode.

        Args:
            episode: The episode to process

        Returns:
            Number of facts created
        """
        count = 0

        # Extract factual content from episode summary
        if episode.summary:
            # Create fact from summary
            await self._semantic.add_fact(
                content=episode.summary,
                predicate="learned_from_episode",
                source_episode_ids=[episode.node_id],
                confidence=0.8,
                metadata={
                    "episode_type": episode.episode_type,
                    "participants": episode.participants,
                },
            )
            count += 1

        return count

    async def promote_semantic_to_metacognitive(self) -> PromotionResult:
        """Promote semantic knowledge to metacognitive layer.

        High-importance, frequently accessed semantic knowledge
        (especially skills and patterns) is elevated to metacognitive.

        Returns:
            Result of the promotion
        """
        result = PromotionResult(
            source_layer=MemoryLayer.SEMANTIC,
            target_layer=MemoryLayer.METACOGNITIVE,
            promoted_count=0,
        )

        now = datetime.now()
        candidates = await self._semantic.get_promotion_candidates()

        # Filter by age and type
        eligible = [
            c for c in candidates
            if (now - c.ingestion_time) >= self._config.semantic_min_age
        ]

        if not eligible:
            return result

        for node in eligible[:self._config.batch_size]:
            try:
                promoted = False

                # Check if this is skill-like content
                if self._is_skill_content(node):
                    # Promote to skill
                    skill = await self._metacognitive.add_skill(
                        name=self._extract_skill_name(node),
                        skill_type="extracted",
                        procedure=node.content,
                        entities=node.entities,
                        metadata={
                            "promoted_from": node.node_id,
                            "promoted_at": now.isoformat(),
                        },
                    )
                    promoted = True

                elif self._is_pattern_content(node):
                    # Promote to insight
                    insight = await self._metacognitive.add_insight(
                        content=node.content,
                        insight_type="extracted_pattern",
                        metadata={
                            "promoted_from": node.node_id,
                            "promoted_at": now.isoformat(),
                        },
                    )
                    promoted = True

                if promoted:
                    # Mark as promoted
                    node.metadata["promoted_to_metacognitive"] = True
                    node.metadata["promoted_at"] = now.isoformat()

                    result.promoted_ids.append(node.node_id)
                    result.promoted_count += 1

                    logger.debug(f"Promoted semantic {node.node_id[:8]}... to metacognitive")

            except Exception as e:
                result.failed_ids.append(node.node_id)
                logger.warning(f"Failed to promote semantic {node.node_id}: {e}")

        return result

    def _is_skill_content(self, node: TemporalNode) -> bool:
        """Check if node content represents a skill.

        Args:
            node: The node to check

        Returns:
            True if skill-like content
        """
        content_lower = node.content.lower()

        # Check for procedural indicators
        skill_indicators = [
            "how to",
            "steps to",
            "procedure",
            "command",
            "to do this",
            "run ",
            "execute",
            "docker",
            "systemctl",
            "api call",
        ]

        return any(indicator in content_lower for indicator in skill_indicators)

    def _is_pattern_content(self, node: TemporalNode) -> bool:
        """Check if node content represents a pattern/insight.

        Args:
            node: The node to check

        Returns:
            True if pattern-like content
        """
        content_lower = node.content.lower()

        # Check for pattern indicators
        pattern_indicators = [
            "pattern",
            "when",
            "if ",
            "usually",
            "tends to",
            "correlat",
            "relationship",
            "works best",
            "optimization",
        ]

        return any(indicator in content_lower for indicator in pattern_indicators)

    def _extract_skill_name(self, node: TemporalNode) -> str:
        """Extract a skill name from node content.

        Args:
            node: The node to extract from

        Returns:
            Extracted skill name
        """
        # Use first entity if available
        if node.entities:
            return node.entities[0]

        # Extract from content
        content = node.content
        if len(content) > 50:
            content = content[:50] + "..."

        # Clean up for skill name
        name = content.replace("How to ", "").replace("Steps to ", "")
        name = name.split(":")[0] if ":" in name else name.split(".")[0]

        return name.strip()

    async def get_promotion_candidates_by_layer(
        self,
        layer: MemoryLayer,
    ) -> list[TemporalNode]:
        """Get promotion candidates for a specific layer.

        Args:
            layer: The source layer

        Returns:
            List of candidates ready for promotion
        """
        if layer == MemoryLayer.WORKING:
            return await self._working.get_promotion_candidates()
        elif layer == MemoryLayer.EPISODIC:
            return await self._episodic.get_promotion_candidates()
        elif layer == MemoryLayer.SEMANTIC:
            return await self._semantic.get_promotion_candidates()
        else:
            return []

    def stats(self) -> dict[str, Any]:
        """Get promotion service statistics.

        Returns:
            Dictionary with stats
        """
        return {
            "last_promotion": self._last_promotion.isoformat(),
            "total_promoted": self._promotion_count,
            "config": {
                "working_importance_threshold": self._config.working_importance_threshold,
                "episodic_importance_threshold": self._config.episodic_importance_threshold,
                "semantic_importance_threshold": self._config.semantic_importance_threshold,
                "batch_size": self._config.batch_size,
            },
        }


class MemoryConsolidator:
    """High-level memory consolidation service.

    Combines promotion with decay and cleanup for memory health.

    Example:
        consolidator = MemoryConsolidator(
            graph=graph,
            working=working,
            episodic=episodic,
            semantic=semantic,
            metacognitive=metacognitive,
        )

        # Run full consolidation
        await consolidator.consolidate()
    """

    def __init__(
        self,
        graph: TemporalCognitiveGraph,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        metacognitive: MetacognitiveMemory,
        promotion_config: PromotionConfig | None = None,
    ):
        """Initialize the consolidator.

        Args:
            graph: The underlying temporal cognitive graph
            working: Working memory layer
            episodic: Episodic memory layer
            semantic: Semantic memory layer
            metacognitive: Metacognitive memory layer
            promotion_config: Promotion configuration
        """
        self._graph = graph
        self._working = working
        self._episodic = episodic
        self._semantic = semantic
        self._metacognitive = metacognitive

        self._promotion = MemoryPromotion(
            graph=graph,
            working=working,
            episodic=episodic,
            semantic=semantic,
            metacognitive=metacognitive,
            config=promotion_config,
        )

    async def consolidate(self) -> dict[str, Any]:
        """Run a full consolidation cycle.

        Steps:
        1. Apply decay to all layers
        2. Cleanup expired items
        3. Run promotions
        4. Return statistics

        Returns:
            Consolidation statistics
        """
        start = datetime.now()
        stats: dict[str, Any] = {
            "decay": {},
            "cleanup": {},
            "promotion": {},
        }

        # 1. Apply decay
        stats["decay"]["working"] = await self._working.apply_decay()
        stats["decay"]["episodic"] = await self._episodic.apply_decay()
        stats["decay"]["semantic"] = await self._semantic.apply_decay()
        stats["decay"]["metacognitive"] = await self._metacognitive.apply_decay()

        # 2. Cleanup expired
        stats["cleanup"]["working"] = await self._working.cleanup_expired()
        stats["cleanup"]["episodic"] = await self._episodic.cleanup_expired()
        stats["cleanup"]["semantic"] = await self._semantic.cleanup_expired()
        stats["cleanup"]["metacognitive"] = await self._metacognitive.cleanup_expired()

        # 3. Run promotions
        promotion_stats = await self._promotion.promote_all()
        stats["promotion"] = {
            "working_to_episodic": promotion_stats.working_to_episodic,
            "episodic_to_semantic": promotion_stats.episodic_to_semantic,
            "semantic_to_metacognitive": promotion_stats.semantic_to_metacognitive,
            "total": promotion_stats.total_promoted,
        }

        stats["duration_ms"] = (datetime.now() - start).total_seconds() * 1000

        logger.info(
            f"Consolidation complete: "
            f"decayed {sum(stats['decay'].values())}, "
            f"cleaned {sum(stats['cleanup'].values())}, "
            f"promoted {stats['promotion']['total']} "
            f"in {stats['duration_ms']:.1f}ms"
        )

        return stats
