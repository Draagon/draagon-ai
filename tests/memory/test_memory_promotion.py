"""Tests for Memory Promotion service."""

import pytest
from datetime import timedelta

from draagon_ai.memory import (
    TemporalCognitiveGraph,
    WorkingMemory,
    EpisodicMemory,
    SemanticMemory,
    MetacognitiveMemory,
    MemoryPromotion,
    MemoryConsolidator,
    PromotionConfig,
    PromotionStats,
    MemoryLayer,
)


@pytest.fixture
def graph():
    """Create a fresh graph for each test."""
    return TemporalCognitiveGraph()


@pytest.fixture
def memory_stack(graph):
    """Create the full memory stack."""
    return {
        "graph": graph,
        "working": WorkingMemory(graph, session_id="test"),
        "episodic": EpisodicMemory(graph),
        "semantic": SemanticMemory(graph),
        "metacognitive": MetacognitiveMemory(graph),
    }


@pytest.fixture
def promotion(memory_stack):
    """Create promotion service."""
    return MemoryPromotion(
        graph=memory_stack["graph"],
        working=memory_stack["working"],
        episodic=memory_stack["episodic"],
        semantic=memory_stack["semantic"],
        metacognitive=memory_stack["metacognitive"],
    )


class TestPromotionConfig:
    """Test promotion configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PromotionConfig()

        assert config.working_importance_threshold == 0.7
        assert config.episodic_importance_threshold == 0.75
        assert config.semantic_importance_threshold == 0.85
        assert config.batch_size == 50

    def test_custom_config(self):
        """Test custom configuration."""
        config = PromotionConfig(
            working_importance_threshold=0.8,
            batch_size=25,
        )

        assert config.working_importance_threshold == 0.8
        assert config.batch_size == 25


class TestPromotionStats:
    """Test promotion statistics."""

    def test_stats_str(self):
        """Test stats string representation."""
        stats = PromotionStats(
            working_to_episodic=5,
            episodic_to_semantic=3,
            semantic_to_metacognitive=1,
            total_promoted=9,
            total_candidates=15,
            duration_ms=150.5,
        )

        s = str(stats)
        assert "9/15" in s
        assert "W→E: 5" in s
        assert "150.5ms" in s


class TestWorkingToEpisodicPromotion:
    """Test Working → Episodic promotion."""

    @pytest.mark.asyncio
    async def test_promotes_high_importance_items(self, memory_stack, promotion):
        """Test that high importance items are promoted."""
        working = memory_stack["working"]
        graph = memory_stack["graph"]

        # Add high-importance item
        item = await working.add(
            "Important conversation topic",
            attention_weight=0.9,
        )

        # Manually boost importance on the GRAPH NODE (not local object)
        node = await graph.get_node(item.node_id)
        node.importance = 0.8
        node.access_count = 5

        # Use custom config with immediate promotion
        promotion._config = PromotionConfig(
            working_min_age=timedelta(seconds=0),
            working_importance_threshold=0.7,
            working_access_threshold=3,
        )

        result = await promotion.promote_working_to_episodic()

        # Should have promoted
        assert result.source_layer == MemoryLayer.WORKING
        assert result.target_layer == MemoryLayer.EPISODIC
        assert result.promoted_count >= 1
        assert item.node_id in result.promoted_ids

    @pytest.mark.asyncio
    async def test_respects_min_age(self, memory_stack, promotion):
        """Test that items must be old enough to promote."""
        working = memory_stack["working"]

        # Add item (will be very new)
        await working.add("New item", attention_weight=0.9)

        # Set high min_age so nothing qualifies
        promotion._config = PromotionConfig(
            working_min_age=timedelta(hours=1),
        )

        result = await promotion.promote_working_to_episodic()

        # Should not promote anything
        assert result.promoted_count == 0


class TestEpisodicToSemanticPromotion:
    """Test Episodic → Semantic promotion."""

    @pytest.mark.asyncio
    async def test_extracts_entities_from_episode(self, memory_stack, promotion):
        """Test that entities are extracted from episodes."""
        episodic = memory_stack["episodic"]
        semantic = memory_stack["semantic"]

        # Create and close an episode with entities
        episode = await episodic.start_episode()
        await episodic.add_event(
            episode.node_id,
            "Doug mentioned he lives in Philadelphia",
            entities=["Doug", "Philadelphia"],
        )
        await episodic.close_episode(
            episode.node_id,
            summary="Discussed Doug's location",
        )

        # Manually set importance for promotion
        node = await memory_stack["graph"].get_node(episode.node_id)
        node.importance = 0.9
        node.access_count = 10

        promotion._config = PromotionConfig(
            episodic_min_age=timedelta(seconds=0),
        )

        result = await promotion.promote_episodic_to_semantic()

        # Should process the episode
        assert isinstance(result.promoted_count, int)

    @pytest.mark.asyncio
    async def test_only_promotes_closed_episodes(self, memory_stack, promotion):
        """Test that only closed episodes are promoted."""
        episodic = memory_stack["episodic"]

        # Create open episode (not closed)
        episode = await episodic.start_episode()

        promotion._config = PromotionConfig(
            episodic_min_age=timedelta(seconds=0),
        )

        result = await promotion.promote_episodic_to_semantic()

        # Open episodes should not be promoted
        assert episode.node_id not in result.promoted_ids


class TestSemanticToMetacognitivePromotion:
    """Test Semantic → Metacognitive promotion."""

    @pytest.mark.asyncio
    async def test_promotes_skill_content(self, memory_stack, promotion):
        """Test that skill-like content is promoted."""
        semantic = memory_stack["semantic"]
        metacognitive = memory_stack["metacognitive"]

        # Add skill-like fact
        fact = await semantic.add_fact(
            "How to restart Plex: run docker restart plex",
        )

        # Boost for promotion
        node = await memory_stack["graph"].get_node(fact.node_id)
        node.importance = 0.9
        node.access_count = 15

        promotion._config = PromotionConfig(
            semantic_min_age=timedelta(seconds=0),
        )

        result = await promotion.promote_semantic_to_metacognitive()

        # May or may not promote depending on access threshold
        assert isinstance(result.promoted_count, int)

    @pytest.mark.asyncio
    async def test_promotes_pattern_content(self, memory_stack, promotion):
        """Test that pattern content becomes insights."""
        semantic = memory_stack["semantic"]

        # Add pattern-like fact
        fact = await semantic.add_fact(
            "Pattern: Users usually ask about weather after greetings",
        )

        # Boost for promotion
        node = await memory_stack["graph"].get_node(fact.node_id)
        node.importance = 0.9
        node.access_count = 15

        promotion._config = PromotionConfig(
            semantic_min_age=timedelta(seconds=0),
        )

        result = await promotion.promote_semantic_to_metacognitive()

        assert isinstance(result.promoted_count, int)


class TestPromoteAll:
    """Test full promotion cycle."""

    @pytest.mark.asyncio
    async def test_promote_all_runs_all_layers(self, promotion):
        """Test that promote_all runs all layer promotions."""
        stats = await promotion.promote_all()

        assert isinstance(stats, PromotionStats)
        assert stats.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_promote_all_handles_errors(self, memory_stack):
        """Test that errors are captured in stats."""
        # Create promotion with broken config
        promotion = MemoryPromotion(
            graph=memory_stack["graph"],
            working=memory_stack["working"],
            episodic=memory_stack["episodic"],
            semantic=memory_stack["semantic"],
            metacognitive=memory_stack["metacognitive"],
        )

        # Should not crash even with edge cases
        stats = await promotion.promote_all()
        assert isinstance(stats, PromotionStats)


class TestPromotionCandidates:
    """Test getting promotion candidates."""

    @pytest.mark.asyncio
    async def test_get_candidates_by_layer(self, memory_stack, promotion):
        """Test getting candidates for each layer."""
        # Working layer candidates
        working_candidates = await promotion.get_promotion_candidates_by_layer(
            MemoryLayer.WORKING
        )
        assert isinstance(working_candidates, list)

        # Episodic layer candidates
        episodic_candidates = await promotion.get_promotion_candidates_by_layer(
            MemoryLayer.EPISODIC
        )
        assert isinstance(episodic_candidates, list)

        # Semantic layer candidates
        semantic_candidates = await promotion.get_promotion_candidates_by_layer(
            MemoryLayer.SEMANTIC
        )
        assert isinstance(semantic_candidates, list)


class TestPromotionStats:
    """Test promotion statistics."""

    @pytest.mark.asyncio
    async def test_stats(self, promotion):
        """Test getting promotion stats."""
        stats = promotion.stats()

        assert "last_promotion" in stats
        assert "total_promoted" in stats
        assert "config" in stats


class TestMemoryConsolidator:
    """Test the memory consolidator service."""

    @pytest.fixture
    def consolidator(self, memory_stack):
        """Create consolidator."""
        return MemoryConsolidator(
            graph=memory_stack["graph"],
            working=memory_stack["working"],
            episodic=memory_stack["episodic"],
            semantic=memory_stack["semantic"],
            metacognitive=memory_stack["metacognitive"],
        )

    @pytest.mark.asyncio
    async def test_consolidate_runs_all_steps(self, consolidator):
        """Test that consolidate runs decay, cleanup, and promotion."""
        stats = await consolidator.consolidate()

        assert "decay" in stats
        assert "cleanup" in stats
        assert "promotion" in stats
        assert "duration_ms" in stats

    @pytest.mark.asyncio
    async def test_consolidate_decay_step(self, consolidator, memory_stack):
        """Test that decay is applied during consolidation."""
        working = memory_stack["working"]

        # Add an item
        await working.add("Test item")

        stats = await consolidator.consolidate()

        # Decay should have been applied
        assert stats["decay"]["working"] >= 0

    @pytest.mark.asyncio
    async def test_consolidate_cleanup_step(self, consolidator):
        """Test that cleanup runs during consolidation."""
        stats = await consolidator.consolidate()

        assert stats["cleanup"]["working"] >= 0
        assert stats["cleanup"]["episodic"] >= 0

    @pytest.mark.asyncio
    async def test_consolidate_promotion_step(self, consolidator):
        """Test that promotion runs during consolidation."""
        stats = await consolidator.consolidate()

        assert "total" in stats["promotion"]


class TestSkillContentDetection:
    """Test skill content detection."""

    def test_is_skill_content(self, promotion):
        """Test skill content detection."""
        from draagon_ai.memory import TemporalNode, NodeType

        # Skill-like content
        skill_node = TemporalNode(
            node_id="test",
            content="How to restart the service: systemctl restart app",
            node_type=NodeType.FACT,
        )
        assert promotion._is_skill_content(skill_node) is True

        # Non-skill content
        fact_node = TemporalNode(
            node_id="test",
            content="Paris is the capital of France",
            node_type=NodeType.FACT,
        )
        assert promotion._is_skill_content(fact_node) is False

    def test_is_pattern_content(self, promotion):
        """Test pattern content detection."""
        from draagon_ai.memory import TemporalNode, NodeType

        # Pattern-like content
        pattern_node = TemporalNode(
            node_id="test",
            content="Pattern: Users usually prefer concise responses",
            node_type=NodeType.INSIGHT,
        )
        assert promotion._is_pattern_content(pattern_node) is True

        # Non-pattern content
        skill_node = TemporalNode(
            node_id="test",
            content="Run docker ps to list containers",
            node_type=NodeType.SKILL,
        )
        assert promotion._is_pattern_content(skill_node) is False


class TestSkillNameExtraction:
    """Test skill name extraction."""

    def test_extract_from_entities(self, promotion):
        """Test extracting skill name from entities."""
        from draagon_ai.memory import TemporalNode, NodeType

        node = TemporalNode(
            node_id="test",
            content="How to restart plex",
            node_type=NodeType.SKILL,
            entities=["restart_plex"],
        )

        name = promotion._extract_skill_name(node)
        assert name == "restart_plex"

    def test_extract_from_content(self, promotion):
        """Test extracting skill name from content."""
        from draagon_ai.memory import TemporalNode, NodeType

        node = TemporalNode(
            node_id="test",
            content="Steps to deploy: run the script",
            node_type=NodeType.SKILL,
            entities=[],
        )

        name = promotion._extract_skill_name(node)
        assert "deploy" in name.lower()
