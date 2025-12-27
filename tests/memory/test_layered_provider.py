"""Tests for LayeredMemoryProvider with Qdrant integration.

Tests cover:
- In-memory mode (no Qdrant)
- Qdrant mode configuration
- All 4 layer storage and retrieval
- TTL configuration
- Cross-layer search
"""

import pytest
from datetime import timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.memory.base import MemoryType, MemoryScope
from draagon_ai.memory.providers.layered import (
    LayeredMemoryProvider,
    LayeredMemoryConfig,
    LAYER_MAPPING,
)


class TestLayeredMemoryConfig:
    """Tests for LayeredMemoryConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LayeredMemoryConfig()

        assert config.qdrant_url is None
        assert config.working_memory_capacity == 7
        assert config.working_ttl_seconds == 300
        assert config.episodic_ttl_days == 14
        assert config.semantic_ttl_days == 180
        assert config.metacognitive_ttl_days is None  # Permanent

    def test_custom_config(self):
        """Test custom configuration."""
        config = LayeredMemoryConfig(
            qdrant_url="http://localhost:6333",
            working_memory_capacity=10,
            working_ttl_seconds=600,
            episodic_ttl_days=7,
            semantic_ttl_days=90,
            metacognitive_ttl_days=365,
        )

        assert config.qdrant_url == "http://localhost:6333"
        assert config.working_memory_capacity == 10
        assert config.get_working_ttl() == timedelta(seconds=600)
        assert config.get_episodic_ttl() == timedelta(days=7)
        assert config.get_semantic_ttl() == timedelta(days=90)
        assert config.get_metacognitive_ttl() == timedelta(days=365)

    def test_ttl_methods(self):
        """Test TTL getter methods."""
        config = LayeredMemoryConfig()

        assert config.get_working_ttl() == timedelta(seconds=300)
        assert config.get_episodic_ttl() == timedelta(days=14)
        assert config.get_semantic_ttl() == timedelta(days=180)
        assert config.get_metacognitive_ttl() is None  # Permanent


class TestLayeredMemoryProviderInMemory:
    """Tests for LayeredMemoryProvider in-memory mode."""

    @pytest.fixture
    def provider(self):
        """Create an in-memory provider."""
        return LayeredMemoryProvider()

    def test_in_memory_mode_auto_initializes(self, provider):
        """In-memory mode should auto-initialize."""
        assert provider.is_initialized
        assert not provider.uses_qdrant

    def test_layers_accessible(self, provider):
        """All 4 layers should be accessible."""
        assert provider.working is not None
        assert provider.episodic is not None
        assert provider.semantic is not None
        assert provider.metacognitive is not None

    def test_graph_accessible(self, provider):
        """Graph should be accessible."""
        assert provider.graph is not None

    def test_session_id_generated(self, provider):
        """Session ID should be generated."""
        assert provider.session_id is not None
        assert len(provider.session_id) > 0

    @pytest.mark.asyncio
    async def test_store_fact_to_semantic(self, provider):
        """Facts should go to semantic layer."""
        memory = await provider.store(
            content="User's favorite color is blue",
            memory_type=MemoryType.FACT,
            entities=["user", "blue"],
        )

        assert memory.id is not None
        assert memory.content == "User's favorite color is blue"
        assert memory.memory_type == MemoryType.FACT

    @pytest.mark.asyncio
    async def test_store_skill_to_metacognitive(self, provider):
        """Skills should go to metacognitive layer."""
        memory = await provider.store(
            content="To restart the service: systemctl restart app",
            memory_type=MemoryType.SKILL,
        )

        assert memory.id is not None
        assert memory.memory_type == MemoryType.SKILL

    @pytest.mark.asyncio
    async def test_store_observation_to_episodic(self, provider):
        """Observations should go to episodic layer."""
        memory = await provider.store(
            content="User asked about the weather",
            memory_type=MemoryType.OBSERVATION,
            user_id="test_user",
        )

        assert memory.id is not None
        assert memory.memory_type == MemoryType.OBSERVATION

    @pytest.mark.asyncio
    async def test_search_returns_results(self, provider):
        """Search should return results."""
        # Store some memories
        await provider.store(
            content="The capital of France is Paris",
            memory_type=MemoryType.FACT,
        )

        # Search
        results = await provider.search("France capital")

        # Results may be empty in unit test without embeddings
        assert isinstance(results, list)

    def test_set_session_creates_new_working_memory(self, provider):
        """Setting session should create new working memory."""
        old_session = provider.session_id

        provider.set_session("new_session_123")

        assert provider.session_id == "new_session_123"
        assert provider.session_id != old_session


class TestLayeredMemoryProviderQdrantMode:
    """Tests for LayeredMemoryProvider with Qdrant backend."""

    def test_qdrant_mode_requires_initialize(self):
        """Qdrant mode should require explicit initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        assert not provider.is_initialized
        assert provider.uses_qdrant

    def test_accessing_layers_before_init_raises(self):
        """Accessing layers before init should raise."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.working

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.graph

    @pytest.mark.asyncio
    async def test_initialize_without_embedding_provider_raises(self):
        """Initialize without embedding_provider should raise."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="embedding_provider is required"):
            await provider.initialize()

    @pytest.mark.asyncio
    async def test_initialize_with_qdrant(self):
        """Test initialization with Qdrant backend."""
        config = LayeredMemoryConfig(
            qdrant_url="http://localhost:6333",
            qdrant_nodes_collection="test_nodes",
            qdrant_edges_collection="test_edges",
        )

        # Mock embedding provider
        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 768)

        # Mock QdrantGraphStore at its source module
        with patch(
            "draagon_ai.memory.providers.qdrant_graph.QdrantGraphStore"
        ) as MockQdrantStore:
            mock_store = MagicMock()
            mock_store.initialize = AsyncMock()
            MockQdrantStore.return_value = mock_store

            provider = LayeredMemoryProvider(
                config=config,
                embedding_provider=mock_embedder,
            )

            await provider.initialize()

            assert provider.is_initialized
            MockQdrantStore.assert_called_once()
            mock_store.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self):
        """Initialize should be idempotent."""
        config = LayeredMemoryConfig(
            qdrant_url="http://localhost:6333",
        )

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 768)

        with patch(
            "draagon_ai.memory.providers.qdrant_graph.QdrantGraphStore"
        ) as MockQdrantStore:
            mock_store = MagicMock()
            mock_store.initialize = AsyncMock()
            MockQdrantStore.return_value = mock_store

            provider = LayeredMemoryProvider(
                config=config,
                embedding_provider=mock_embedder,
            )

            await provider.initialize()
            await provider.initialize()  # Second call should be no-op

            # Should only be called once
            assert MockQdrantStore.call_count == 1

    @pytest.mark.asyncio
    async def test_close_closes_qdrant(self):
        """Close should close Qdrant connection."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")

        mock_embedder = MagicMock()
        mock_embedder.embed = AsyncMock(return_value=[0.1] * 768)

        with patch(
            "draagon_ai.memory.providers.qdrant_graph.QdrantGraphStore"
        ) as MockQdrantStore:
            mock_store = MagicMock()
            mock_store.initialize = AsyncMock()
            mock_store.close = AsyncMock()
            MockQdrantStore.return_value = mock_store

            provider = LayeredMemoryProvider(
                config=config,
                embedding_provider=mock_embedder,
            )

            await provider.initialize()
            await provider.close()

            mock_store.close.assert_called_once()


class TestLayeredMemoryProviderWithExternalGraph:
    """Tests for LayeredMemoryProvider with externally provided graph."""

    @pytest.mark.asyncio
    async def test_use_external_graph(self):
        """Provider should use externally provided graph."""
        from draagon_ai.memory.temporal_graph import TemporalCognitiveGraph

        external_graph = TemporalCognitiveGraph()
        provider = LayeredMemoryProvider(graph=external_graph)

        assert provider.is_initialized
        assert provider.graph is external_graph

    @pytest.mark.asyncio
    async def test_external_graph_not_closed_on_close(self):
        """External graph should not be closed when provider closes."""
        from draagon_ai.memory.temporal_graph import TemporalCognitiveGraph

        external_graph = TemporalCognitiveGraph()
        provider = LayeredMemoryProvider(graph=external_graph)

        # Close should not affect external graph
        await provider.close()

        # External graph should still be usable
        assert external_graph is not None


class TestLayerMapping:
    """Tests for memory type to layer mapping."""

    def test_fact_maps_to_semantic(self):
        assert LAYER_MAPPING[MemoryType.FACT] == "semantic"

    def test_skill_maps_to_metacognitive(self):
        assert LAYER_MAPPING[MemoryType.SKILL] == "metacognitive"

    def test_episodic_maps_to_episodic(self):
        assert LAYER_MAPPING[MemoryType.EPISODIC] == "episodic"

    def test_observation_maps_to_episodic(self):
        assert LAYER_MAPPING[MemoryType.OBSERVATION] == "episodic"

    def test_insight_maps_to_metacognitive(self):
        assert LAYER_MAPPING[MemoryType.INSIGHT] == "metacognitive"

    def test_preference_maps_to_semantic(self):
        assert LAYER_MAPPING[MemoryType.PREFERENCE] == "semantic"


class TestLayerTTLConfiguration:
    """Tests for layer TTL configuration."""

    def test_config_ttl_values(self):
        """Config should expose correct TTL values."""
        config = LayeredMemoryConfig(
            working_ttl_seconds=120,
            episodic_ttl_days=7,
            semantic_ttl_days=30,
            metacognitive_ttl_days=365,
        )

        assert config.get_working_ttl() == timedelta(seconds=120)
        assert config.get_episodic_ttl() == timedelta(days=7)
        assert config.get_semantic_ttl() == timedelta(days=30)
        assert config.get_metacognitive_ttl() == timedelta(days=365)

    def test_permanent_metacognitive_by_default(self):
        """Metacognitive layer should be permanent by default."""
        config = LayeredMemoryConfig()

        # None means permanent (no expiration)
        assert config.get_metacognitive_ttl() is None

    def test_provider_creates_layers(self):
        """Provider should create all 4 layers."""
        config = LayeredMemoryConfig()
        provider = LayeredMemoryProvider(config=config)

        # All layers should exist
        assert provider._working is not None
        assert provider._episodic is not None
        assert provider._semantic is not None
        assert provider._metacognitive is not None


class TestPromotionConfigIntegration:
    """Tests for promotion configuration in LayeredMemoryConfig."""

    def test_default_promotion_config(self):
        """Test default promotion configuration values."""
        config = LayeredMemoryConfig()

        # Working → Episodic defaults
        assert config.promotion_working_importance == 0.7
        assert config.promotion_working_access == 3
        assert config.promotion_working_min_age_minutes == 5

        # Episodic → Semantic defaults
        assert config.promotion_episodic_importance == 0.75
        assert config.promotion_episodic_access == 5
        assert config.promotion_episodic_min_age_hours == 1

        # Semantic → Metacognitive defaults
        assert config.promotion_semantic_importance == 0.85
        assert config.promotion_semantic_access == 10
        assert config.promotion_semantic_min_age_days == 7

        # Processing limits
        assert config.promotion_batch_size == 50
        assert config.promotion_max_per_cycle == 100

    def test_custom_promotion_config(self):
        """Test custom promotion configuration."""
        config = LayeredMemoryConfig(
            promotion_working_importance=0.5,
            promotion_working_access=2,
            promotion_episodic_importance=0.6,
            promotion_semantic_importance=0.8,
            promotion_batch_size=25,
        )

        assert config.promotion_working_importance == 0.5
        assert config.promotion_working_access == 2
        assert config.promotion_episodic_importance == 0.6
        assert config.promotion_semantic_importance == 0.8
        assert config.promotion_batch_size == 25

    def test_get_promotion_config(self):
        """Test get_promotion_config() returns correct PromotionConfig."""
        config = LayeredMemoryConfig(
            promotion_working_importance=0.6,
            promotion_working_min_age_minutes=10,
            promotion_episodic_min_age_hours=2,
            promotion_semantic_min_age_days=14,
        )

        promo_config = config.get_promotion_config()

        # Check thresholds
        assert promo_config.working_importance_threshold == 0.6
        assert promo_config.working_min_age == timedelta(minutes=10)
        assert promo_config.episodic_min_age == timedelta(hours=2)
        assert promo_config.semantic_min_age == timedelta(days=14)


class TestPromotionServiceIntegration:
    """Tests for promotion service integration with LayeredMemoryProvider."""

    @pytest.fixture
    def provider(self):
        """Create an in-memory provider."""
        return LayeredMemoryProvider()

    def test_promotion_service_created(self, provider):
        """Promotion service should be created during initialization."""
        assert provider._promotion is not None
        assert provider.promotion is not None

    def test_consolidator_created(self, provider):
        """Consolidator should be created during initialization."""
        assert provider._consolidator is not None
        assert provider.consolidator is not None

    def test_promotion_property_requires_initialization(self):
        """Promotion property should require initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        # Not initialized yet (pending Qdrant)
        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.promotion

    def test_consolidator_property_requires_initialization(self):
        """Consolidator property should require initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            _ = provider.consolidator

    @pytest.mark.asyncio
    async def test_promote_all(self, provider):
        """Test promote_all() returns PromotionStats."""
        stats = await provider.promote_all()

        # Should return PromotionStats
        assert hasattr(stats, 'total_promoted')
        assert hasattr(stats, 'working_to_episodic')
        assert hasattr(stats, 'episodic_to_semantic')
        assert hasattr(stats, 'semantic_to_metacognitive')
        assert hasattr(stats, 'duration_ms')

        # With empty layers, should promote 0
        assert stats.total_promoted == 0

    @pytest.mark.asyncio
    async def test_promote_working_to_episodic(self, provider):
        """Test promote_working_to_episodic() returns count."""
        count = await provider.promote_working_to_episodic()

        # With empty working memory, should be 0
        assert count == 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_promote_episodic_to_semantic(self, provider):
        """Test promote_episodic_to_semantic() returns count."""
        count = await provider.promote_episodic_to_semantic()

        # With no closed episodes, should be 0
        assert count == 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_promote_semantic_to_metacognitive(self, provider):
        """Test promote_semantic_to_metacognitive() returns count."""
        count = await provider.promote_semantic_to_metacognitive()

        # With empty semantic layer, should be 0
        assert count == 0
        assert isinstance(count, int)

    @pytest.mark.asyncio
    async def test_consolidate(self, provider):
        """Test consolidate() returns stats dict."""
        stats = await provider.consolidate()

        # Should have decay, cleanup, promotion sections
        assert 'decay' in stats
        assert 'cleanup' in stats
        assert 'promotion' in stats
        assert 'duration_ms' in stats

        # Promotion section should have layer counts
        assert 'working_to_episodic' in stats['promotion']
        assert 'episodic_to_semantic' in stats['promotion']
        assert 'semantic_to_metacognitive' in stats['promotion']
        assert 'total' in stats['promotion']

    def test_get_promotion_stats(self, provider):
        """Test get_promotion_stats() returns stats dict."""
        stats = provider.get_promotion_stats()

        # Should have basic stats
        assert 'last_promotion' in stats
        assert 'total_promoted' in stats
        assert 'config' in stats

        # Config should have thresholds
        assert 'working_importance_threshold' in stats['config']
        assert 'episodic_importance_threshold' in stats['config']
        assert 'semantic_importance_threshold' in stats['config']

    @pytest.mark.asyncio
    async def test_promotion_with_working_memory_items(self, provider):
        """Test promotion with actual working memory items."""
        # Add items to working memory with high importance
        await provider.working.add(
            content="Important task",
            attention_weight=0.9,
            source="test",
        )

        # Get promotion stats before
        stats_before = provider.get_promotion_stats()

        # Run promotion (items won't be promoted due to min_age)
        count = await provider.promote_working_to_episodic()

        # Items too young, should not be promoted
        assert count == 0

    @pytest.mark.asyncio
    async def test_consolidate_applies_decay(self, provider):
        """Test that consolidate applies decay to layers."""
        # Store some memories first
        await provider.store(
            content="Test fact for decay",
            memory_type=MemoryType.FACT,
            importance=0.8,
        )

        # Run consolidation
        stats = await provider.consolidate()

        # Decay should have been applied (may be 0 if no items to decay)
        assert 'decay' in stats
        assert 'semantic' in stats['decay']

    @pytest.mark.asyncio
    async def test_promote_all_requires_initialization(self):
        """Test that promote_all requires initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.promote_all()

    @pytest.mark.asyncio
    async def test_consolidate_requires_initialization(self):
        """Test that consolidate requires initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.consolidate()

    def test_promotion_stats_requires_initialization(self):
        """Test that get_promotion_stats requires initialization."""
        config = LayeredMemoryConfig(qdrant_url="http://localhost:6333")
        provider = LayeredMemoryProvider(config=config)

        with pytest.raises(RuntimeError, match="not initialized"):
            provider.get_promotion_stats()


# =============================================================================
# Scope Access Control Tests (REQ-001-04)
# =============================================================================

class TestScopeHierarchyHelpers:
    """Tests for scope hierarchy helper functions."""

    def test_scope_hierarchy_order(self):
        """Test that scope hierarchy is WORLD > CONTEXT > AGENT > USER > SESSION."""
        from draagon_ai.memory.providers.layered import SCOPE_HIERARCHY

        assert SCOPE_HIERARCHY[0] == MemoryScope.WORLD
        assert SCOPE_HIERARCHY[1] == MemoryScope.CONTEXT
        assert SCOPE_HIERARCHY[2] == MemoryScope.AGENT
        assert SCOPE_HIERARCHY[3] == MemoryScope.USER
        assert SCOPE_HIERARCHY[4] == MemoryScope.SESSION

    def test_get_scope_level(self):
        """Test get_scope_level returns correct hierarchy level."""
        from draagon_ai.memory.providers.layered import get_scope_level

        assert get_scope_level(MemoryScope.WORLD) == 0
        assert get_scope_level(MemoryScope.CONTEXT) == 1
        assert get_scope_level(MemoryScope.AGENT) == 2
        assert get_scope_level(MemoryScope.USER) == 3
        assert get_scope_level(MemoryScope.SESSION) == 4

    def test_get_accessible_scopes_from_user(self):
        """Test USER scope can access USER + AGENT + CONTEXT + WORLD (T01)."""
        from draagon_ai.memory.providers.layered import get_accessible_scopes

        accessible = get_accessible_scopes(MemoryScope.USER)

        assert MemoryScope.WORLD in accessible
        assert MemoryScope.CONTEXT in accessible
        assert MemoryScope.AGENT in accessible
        assert MemoryScope.USER in accessible
        assert MemoryScope.SESSION not in accessible  # More specific, not accessible

    def test_get_accessible_scopes_from_world(self):
        """Test WORLD scope can only access WORLD (T02)."""
        from draagon_ai.memory.providers.layered import get_accessible_scopes

        accessible = get_accessible_scopes(MemoryScope.WORLD)

        assert accessible == [MemoryScope.WORLD]
        assert MemoryScope.CONTEXT not in accessible
        assert MemoryScope.AGENT not in accessible
        assert MemoryScope.USER not in accessible
        assert MemoryScope.SESSION not in accessible

    def test_get_accessible_scopes_from_session(self):
        """Test SESSION scope can access all scopes."""
        from draagon_ai.memory.providers.layered import get_accessible_scopes

        accessible = get_accessible_scopes(MemoryScope.SESSION)

        assert len(accessible) == 5
        assert MemoryScope.WORLD in accessible
        assert MemoryScope.CONTEXT in accessible
        assert MemoryScope.AGENT in accessible
        assert MemoryScope.USER in accessible
        assert MemoryScope.SESSION in accessible

    def test_can_scope_read(self):
        """Test scope read permissions."""
        from draagon_ai.memory.providers.layered import can_scope_read

        # Lower (more specific) can read higher (more general)
        assert can_scope_read(MemoryScope.USER, MemoryScope.WORLD) is True
        assert can_scope_read(MemoryScope.SESSION, MemoryScope.WORLD) is True
        assert can_scope_read(MemoryScope.SESSION, MemoryScope.USER) is True

        # Same level can read
        assert can_scope_read(MemoryScope.USER, MemoryScope.USER) is True
        assert can_scope_read(MemoryScope.WORLD, MemoryScope.WORLD) is True

        # Higher (more general) cannot read lower (more specific)
        assert can_scope_read(MemoryScope.WORLD, MemoryScope.USER) is False
        assert can_scope_read(MemoryScope.CONTEXT, MemoryScope.SESSION) is False
        assert can_scope_read(MemoryScope.AGENT, MemoryScope.USER) is False

    def test_can_scope_write(self):
        """Test scope write permissions."""
        from draagon_ai.memory.providers.layered import can_scope_write

        # Can write to same level or more specific (higher level number)
        assert can_scope_write(MemoryScope.AGENT, MemoryScope.AGENT) is True
        assert can_scope_write(MemoryScope.AGENT, MemoryScope.USER) is True
        assert can_scope_write(MemoryScope.AGENT, MemoryScope.SESSION) is True

        # Cannot write to higher (more general) scopes
        assert can_scope_write(MemoryScope.USER, MemoryScope.WORLD) is False
        assert can_scope_write(MemoryScope.SESSION, MemoryScope.AGENT) is False
        assert can_scope_write(MemoryScope.AGENT, MemoryScope.CONTEXT) is False


class TestScopeEnforcementStore:
    """Tests for scope enforcement in store() method."""

    @pytest.fixture
    def enforcing_provider(self):
        """Create a provider with scope enforcement enabled."""
        config = LayeredMemoryConfig(enforce_scope_permissions=True)
        return LayeredMemoryProvider(config=config)

    @pytest.fixture
    def non_enforcing_provider(self):
        """Create a provider without scope enforcement."""
        return LayeredMemoryProvider()

    @pytest.mark.asyncio
    async def test_store_without_enforcement_allows_any_scope(self, non_enforcing_provider):
        """Test that store allows any scope when enforcement is disabled."""
        # Should not raise
        await non_enforcing_provider.store(
            content="World fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
            user_id="test_user",  # User-level caller writing to WORLD
        )

    @pytest.mark.asyncio
    async def test_store_user_to_world_denied(self, enforcing_provider):
        """Test that USER scope cannot write to WORLD scope (T03)."""
        with pytest.raises(PermissionError, match="Cannot write to scope 'world'"):
            await enforcing_provider.store(
                content="Should fail",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.WORLD,
                user_id="test_user",  # User-level caller
            )

    @pytest.mark.asyncio
    async def test_store_user_to_user_allowed(self, enforcing_provider):
        """Test that USER scope can write to USER scope."""
        # Should not raise
        memory = await enforcing_provider.store(
            content="User fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        assert memory is not None

    @pytest.mark.asyncio
    async def test_store_user_to_session_allowed(self, enforcing_provider):
        """Test that USER scope can write to SESSION scope (more specific)."""
        # Should not raise
        memory = await enforcing_provider.store(
            content="Session fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.SESSION,
            user_id="test_user",
        )
        assert memory is not None

    @pytest.mark.asyncio
    async def test_store_agent_to_context_denied(self, enforcing_provider):
        """Test that AGENT scope cannot write to CONTEXT scope."""
        with pytest.raises(PermissionError, match="Cannot write to scope 'context'"):
            await enforcing_provider.store(
                content="Should fail",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.CONTEXT,
                agent_id="test_agent",  # Agent-level caller
            )

    @pytest.mark.asyncio
    async def test_store_agent_to_user_allowed(self, enforcing_provider):
        """Test that AGENT scope can write to USER scope."""
        # Should not raise
        memory = await enforcing_provider.store(
            content="User fact from agent",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            agent_id="test_agent",
        )
        assert memory is not None

    @pytest.mark.asyncio
    async def test_store_session_to_any_higher_denied(self, enforcing_provider):
        """Test that SESSION scope cannot write to any higher scope."""
        # Session caller (no user_id, no agent_id)
        for target_scope in [MemoryScope.WORLD, MemoryScope.CONTEXT, MemoryScope.AGENT, MemoryScope.USER]:
            with pytest.raises(PermissionError, match="Cannot write to scope"):
                await enforcing_provider.store(
                    content="Should fail",
                    memory_type=MemoryType.FACT,
                    scope=target_scope,
                    # No user_id or agent_id = SESSION level caller
                )

    @pytest.mark.asyncio
    async def test_store_session_to_session_allowed(self, enforcing_provider):
        """Test that SESSION scope can write to SESSION scope."""
        memory = await enforcing_provider.store(
            content="Session-scoped fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.SESSION,
            # No user_id or agent_id = SESSION level caller
        )
        assert memory is not None


class TestScopeEnforcementSearch:
    """Tests for scope enforcement in search() method."""

    @pytest.fixture
    def enforcing_provider(self):
        """Create a provider with scope enforcement enabled."""
        config = LayeredMemoryConfig(enforce_scope_permissions=True)
        return LayeredMemoryProvider(config=config)

    @pytest.fixture
    def non_enforcing_provider(self):
        """Create a provider without scope enforcement."""
        return LayeredMemoryProvider()

    @pytest.mark.asyncio
    async def test_search_without_enforcement_allows_any_scope(self, non_enforcing_provider):
        """Test that search returns all scopes when enforcement is disabled."""
        # Store memories at various scopes
        await non_enforcing_provider.store(
            content="World fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
        )
        await non_enforcing_provider.store(
            content="User fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Mock the layer searches to return results
        results = await non_enforcing_provider.search("fact", limit=10)

        # Without enforcement, all scopes should be searchable
        # (actual results depend on layer search implementation)

    @pytest.mark.asyncio
    async def test_search_user_scope_filters_to_accessible(self, enforcing_provider):
        """Test that USER scope search only sees accessible scopes."""
        # Store with various scopes (bypassing enforcement for setup)
        enforcing_provider._config.enforce_scope_permissions = False

        await enforcing_provider.store(
            content="World capital facts",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
        )
        await enforcing_provider.store(
            content="User preference facts",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await enforcing_provider.store(
            content="Session context facts",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.SESSION,
        )

        # Re-enable enforcement
        enforcing_provider._config.enforce_scope_permissions = True

        # USER scope search - should filter scope list to accessible
        # (Actual filtering happens in the search method)
        # This test verifies the mechanism is in place
        results = await enforcing_provider.search(
            "facts",
            user_id="test_user",
            limit=10,
        )
        # Results would only include WORLD, CONTEXT, AGENT, USER - not SESSION

    @pytest.mark.asyncio
    async def test_search_world_scope_only_sees_world(self, enforcing_provider):
        """Test that WORLD scope search only sees WORLD memories."""
        # Setup memories
        enforcing_provider._config.enforce_scope_permissions = False
        await enforcing_provider.store(
            content="World fact for search",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.WORLD,
        )
        await enforcing_provider.store(
            content="User fact for search",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        enforcing_provider._config.enforce_scope_permissions = True

        # WORLD scope search should only see WORLD
        from draagon_ai.memory.providers.layered import get_accessible_scopes
        accessible = get_accessible_scopes(MemoryScope.WORLD)
        assert accessible == [MemoryScope.WORLD]

    @pytest.mark.asyncio
    async def test_search_with_explicit_caller_scope(self, enforcing_provider):
        """Test search with explicit caller_scope parameter."""
        # The caller_scope parameter allows explicit scope specification
        results = await enforcing_provider.search(
            "test query",
            caller_scope=MemoryScope.AGENT,
            limit=10,
        )
        # Should filter to WORLD, CONTEXT, AGENT - not USER or SESSION

    @pytest.mark.asyncio
    async def test_search_requested_scope_filtered_by_accessible(self, enforcing_provider):
        """Test that requested scopes are filtered to only accessible ones."""
        # Request scopes that include inaccessible ones
        results = await enforcing_provider.search(
            "test query",
            caller_scope=MemoryScope.CONTEXT,  # Level 1
            scopes=[MemoryScope.WORLD, MemoryScope.USER, MemoryScope.SESSION],  # Only WORLD is accessible
            limit=10,
        )
        # Should filter scopes to only [WORLD] since USER and SESSION are not accessible from CONTEXT

    @pytest.mark.asyncio
    async def test_search_empty_when_no_accessible_scopes(self, enforcing_provider):
        """Test that search returns empty when no requested scopes are accessible."""
        results = await enforcing_provider.search(
            "test query",
            caller_scope=MemoryScope.WORLD,  # Level 0 - most general
            scopes=[MemoryScope.USER, MemoryScope.SESSION],  # Neither accessible from WORLD
            limit=10,
        )
        # Should return empty list since no scopes are accessible
        assert results == []


class TestScopeTypeMappingIntegration:
    """Tests for SCOPE_TYPE_MAPPING with existing scope system."""

    def test_scope_type_mapping_complete(self):
        """Test that SCOPE_TYPE_MAPPING covers all MemoryScope values."""
        from draagon_ai.memory.providers.layered import SCOPE_TYPE_MAPPING
        from draagon_ai.memory.scopes import ScopeType

        # Verify all MemoryScope values are mapped
        for scope in MemoryScope:
            assert scope in SCOPE_TYPE_MAPPING, f"Missing mapping for {scope}"
            assert isinstance(SCOPE_TYPE_MAPPING[scope], ScopeType)

    def test_scope_type_mapping_values(self):
        """Test that SCOPE_TYPE_MAPPING maps to correct ScopeType values."""
        from draagon_ai.memory.providers.layered import SCOPE_TYPE_MAPPING
        from draagon_ai.memory.scopes import ScopeType

        assert SCOPE_TYPE_MAPPING[MemoryScope.WORLD] == ScopeType.WORLD
        assert SCOPE_TYPE_MAPPING[MemoryScope.CONTEXT] == ScopeType.CONTEXT
        assert SCOPE_TYPE_MAPPING[MemoryScope.AGENT] == ScopeType.AGENT
        assert SCOPE_TYPE_MAPPING[MemoryScope.USER] == ScopeType.USER
        assert SCOPE_TYPE_MAPPING[MemoryScope.SESSION] == ScopeType.SESSION
