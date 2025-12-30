"""Tests for RoxyLayeredAdapter.

Tests cover:
- Store operations with type/scope mapping
- Search operations with type filtering
- Get, update, delete operations
- Promotion and consolidation exposure
- Error handling
- Type mapping correctness
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from draagon_ai.adapters.roxy import (
    RoxyLayeredAdapter,
    RoxyMemoryType,
    RoxyMemoryScope,
    ROXY_TYPE_MAPPING,
    ROXY_SCOPE_MAPPING,
    DRAAGON_TYPE_MAPPING,
    DRAAGON_SCOPE_MAPPING,
)
from draagon_ai.memory.base import Memory, MemoryType, MemoryScope, SearchResult
from draagon_ai.memory.providers.layered import LayeredMemoryProvider


class TestRoxyTypeMappings:
    """Tests for type and scope mappings."""

    def test_roxy_type_mapping_complete(self):
        """Test all RoxyMemoryType values are mapped."""
        for roxy_type in RoxyMemoryType:
            assert roxy_type in ROXY_TYPE_MAPPING
            assert isinstance(ROXY_TYPE_MAPPING[roxy_type], MemoryType)

    def test_roxy_type_mapping_strings(self):
        """Test string values are also mapped."""
        for roxy_type in RoxyMemoryType:
            assert roxy_type.value in ROXY_TYPE_MAPPING
            assert ROXY_TYPE_MAPPING[roxy_type.value] == ROXY_TYPE_MAPPING[roxy_type]

    def test_roxy_scope_mapping_complete(self):
        """Test all RoxyMemoryScope values are mapped."""
        for roxy_scope in RoxyMemoryScope:
            assert roxy_scope in ROXY_SCOPE_MAPPING
            assert isinstance(ROXY_SCOPE_MAPPING[roxy_scope], MemoryScope)

    def test_roxy_scope_mapping_strings(self):
        """Test string values are also mapped."""
        for roxy_scope in RoxyMemoryScope:
            assert roxy_scope.value in ROXY_SCOPE_MAPPING
            assert ROXY_SCOPE_MAPPING[roxy_scope.value] == ROXY_SCOPE_MAPPING[roxy_scope]

    def test_draagon_type_mapping_complete(self):
        """Test all MemoryType values are mapped back to Roxy types."""
        for mem_type in MemoryType:
            assert mem_type in DRAAGON_TYPE_MAPPING
            assert isinstance(DRAAGON_TYPE_MAPPING[mem_type], RoxyMemoryType)

    def test_draagon_scope_mapping_complete(self):
        """Test all MemoryScope values are mapped back to Roxy scopes."""
        for scope in MemoryScope:
            assert scope in DRAAGON_SCOPE_MAPPING
            assert isinstance(DRAAGON_SCOPE_MAPPING[scope], RoxyMemoryScope)


class TestRoxyLayeredAdapterStore:
    """Tests for store operations."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.store = AsyncMock()
        provider.search = AsyncMock(return_value=[])
        provider.get = AsyncMock(return_value=None)
        provider.update = AsyncMock(return_value=None)
        provider.delete = AsyncMock(return_value=True)
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_store_fact_private(self, adapter, mock_provider):
        """Test storing a private fact."""
        mock_provider.store.return_value = Memory(
            id="mem_123",
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
        )

        result = await adapter.store(
            content="Doug's birthday is March 15",
            user_id="doug",
            scope="private",
            memory_type="fact",
        )

        assert result["success"] is True
        assert result["memory_id"] == "mem_123"

        # Verify correct mapping to draagon-ai types
        mock_provider.store.assert_called_once()
        call_kwargs = mock_provider.store.call_args.kwargs
        assert call_kwargs["memory_type"] == MemoryType.FACT
        assert call_kwargs["scope"] == MemoryScope.USER
        assert call_kwargs["user_id"] == "doug"

    @pytest.mark.asyncio
    async def test_store_skill_shared(self, adapter, mock_provider):
        """Test storing a shared skill."""
        mock_provider.store.return_value = Memory(
            id="mem_456",
            content="To restart Roxy: systemctl restart roxy",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.CONTEXT,
            user_id="doug",
            context_id="mealing_home",
        )

        result = await adapter.store(
            content="To restart Roxy: systemctl restart roxy",
            user_id="doug",
            scope="shared",
            memory_type="skill",
            household_id="mealing_home",
        )

        assert result["success"] is True
        call_kwargs = mock_provider.store.call_args.kwargs
        assert call_kwargs["memory_type"] == MemoryType.SKILL
        assert call_kwargs["scope"] == MemoryScope.CONTEXT
        assert call_kwargs["context_id"] == "mealing_home"

    @pytest.mark.asyncio
    async def test_store_with_entities(self, adapter, mock_provider):
        """Test storing with entities."""
        mock_provider.store.return_value = Memory(
            id="mem_789",
            content="Doug's favorite color is blue",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            entities=["doug", "blue"],
        )

        result = await adapter.store(
            content="Doug's favorite color is blue",
            user_id="doug",
            memory_type="preference",
            entities=["doug", "blue"],
        )

        assert result["success"] is True
        call_kwargs = mock_provider.store.call_args.kwargs
        assert call_kwargs["entities"] == ["doug", "blue"]

    @pytest.mark.asyncio
    async def test_store_system_scope(self, adapter, mock_provider):
        """Test storing with system scope maps to WORLD."""
        mock_provider.store.return_value = Memory(
            id="mem_sys",
            content="Python was created in 1991",
            memory_type=MemoryType.KNOWLEDGE,
            scope=MemoryScope.WORLD,
        )

        await adapter.store(
            content="Python was created in 1991",
            user_id="system",
            scope="system",
            memory_type="knowledge",
        )

        call_kwargs = mock_provider.store.call_args.kwargs
        assert call_kwargs["scope"] == MemoryScope.WORLD

    @pytest.mark.asyncio
    async def test_store_permission_error(self, adapter, mock_provider):
        """Test handling permission errors."""
        mock_provider.store.side_effect = PermissionError("Cannot write to WORLD scope")

        result = await adapter.store(
            content="Test",
            user_id="doug",
            scope="system",
            memory_type="fact",
        )

        assert result["success"] is False
        assert "Cannot write" in result["error"]

    @pytest.mark.asyncio
    async def test_store_generic_error(self, adapter, mock_provider):
        """Test handling generic errors."""
        mock_provider.store.side_effect = Exception("Database error")

        result = await adapter.store(
            content="Test",
            user_id="doug",
        )

        assert result["success"] is False
        assert "Database error" in result["error"]


class TestRoxyLayeredAdapterSearch:
    """Tests for search operations."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.search = AsyncMock(return_value=[])
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_search_basic(self, adapter, mock_provider):
        """Test basic search."""
        mock_provider.search.return_value = [
            SearchResult(
                memory=Memory(
                    id="mem_1",
                    content="Doug's birthday is March 15",
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                    user_id="doug",
                    importance=0.8,
                ),
                score=0.95,
            ),
        ]

        results = await adapter.search(
            query="birthday",
            user_id="doug",
            limit=5,
        )

        assert len(results) == 1
        assert results[0]["id"] == "mem_1"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"]["content"] == "Doug's birthday is March 15"
        assert results[0]["payload"]["memory_type"] == "fact"
        assert results[0]["payload"]["scope"] == "private"

    @pytest.mark.asyncio
    async def test_search_with_type_filter(self, adapter, mock_provider):
        """Test search with memory type filter."""
        mock_provider.search.return_value = []

        await adapter.search(
            query="test",
            user_id="doug",
            memory_types=["fact", "preference"],
        )

        call_kwargs = mock_provider.search.call_args.kwargs
        assert MemoryType.FACT in call_kwargs["memory_types"]
        assert MemoryType.PREFERENCE in call_kwargs["memory_types"]

    @pytest.mark.asyncio
    async def test_search_with_min_score(self, adapter, mock_provider):
        """Test search with minimum score filter."""
        await adapter.search(
            query="test",
            user_id="doug",
            min_score=0.7,
        )

        call_kwargs = mock_provider.search.call_args.kwargs
        assert call_kwargs["min_score"] == 0.7

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self, adapter, mock_provider):
        """Test search error returns empty list."""
        mock_provider.search.side_effect = Exception("Search failed")

        results = await adapter.search(
            query="test",
            user_id="doug",
        )

        assert results == []


class TestRoxyLayeredAdapterCRUD:
    """Tests for get, update, delete operations."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.get = AsyncMock(return_value=None)
        provider.update = AsyncMock(return_value=None)
        provider.delete = AsyncMock(return_value=True)
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_get_by_id_found(self, adapter, mock_provider):
        """Test getting a memory by ID when found."""
        mock_provider.get.return_value = Memory(
            id="mem_123",
            content="Test content",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            importance=0.8,
            created_at=datetime(2025, 1, 1, 12, 0, 0),
        )

        result = await adapter.get_by_id("mem_123")

        assert result is not None
        assert result["id"] == "mem_123"
        assert result["payload"]["content"] == "Test content"
        assert result["payload"]["memory_type"] == "fact"

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self, adapter, mock_provider):
        """Test getting a memory by ID when not found."""
        mock_provider.get.return_value = None

        result = await adapter.get_by_id("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_update_memory_success(self, adapter, mock_provider):
        """Test updating a memory successfully."""
        mock_provider.update.return_value = Memory(
            id="mem_123",
            content="Updated content",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
        )

        result = await adapter.update_memory(
            "mem_123",
            {"content": "Updated content", "importance": 0.9},
        )

        assert result["success"] is True
        mock_provider.update.assert_called_once_with(
            "mem_123",
            content="Updated content",
            importance=0.9,
            confidence=None,
            metadata={"content": "Updated content", "importance": 0.9},
        )

    @pytest.mark.asyncio
    async def test_update_memory_not_found(self, adapter, mock_provider):
        """Test updating a memory that doesn't exist."""
        mock_provider.update.return_value = None

        result = await adapter.update_memory("nonexistent", {"content": "new"})

        assert result["success"] is False
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_delete_success(self, adapter, mock_provider):
        """Test deleting a memory successfully."""
        mock_provider.delete.return_value = True

        result = await adapter.delete("mem_123")

        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self, adapter, mock_provider):
        """Test deleting a memory that doesn't exist."""
        mock_provider.delete.return_value = False

        result = await adapter.delete("nonexistent")

        assert result["success"] is False


class TestRoxyLayeredAdapterPromotion:
    """Tests for promotion and consolidation methods."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.promote_all = AsyncMock()
        provider.consolidate = AsyncMock()
        provider.get_promotion_stats = MagicMock()
        provider.get = AsyncMock()
        provider.update = AsyncMock()
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_promote_all(self, adapter, mock_provider):
        """Test promote_all returns stats."""
        mock_stats = MagicMock()
        mock_stats.working_to_episodic = 5
        mock_stats.episodic_to_semantic = 3
        mock_stats.semantic_to_metacognitive = 1
        mock_stats.total_promoted = 9
        mock_stats.duration_ms = 150
        mock_provider.promote_all.return_value = mock_stats

        result = await adapter.promote_all()

        assert result["working_to_episodic"] == 5
        assert result["episodic_to_semantic"] == 3
        assert result["semantic_to_metacognitive"] == 1
        assert result["total_promoted"] == 9
        assert result["duration_ms"] == 150

    @pytest.mark.asyncio
    async def test_consolidate(self, adapter, mock_provider):
        """Test consolidate returns stats."""
        mock_provider.consolidate.return_value = {
            "decay": {"semantic": 10, "episodic": 5},
            "cleanup": {"semantic": 2},
            "promotion": {"total": 3},
        }

        result = await adapter.consolidate()

        assert "decay" in result
        assert "cleanup" in result
        assert "promotion" in result

    def test_get_promotion_stats(self, adapter, mock_provider):
        """Test getting promotion stats."""
        mock_provider.get_promotion_stats.return_value = {
            "last_promotion": "2025-01-01T00:00:00",
            "total_promoted": 100,
        }

        result = adapter.get_promotion_stats()

        assert result["total_promoted"] == 100


class TestRoxyLayeredAdapterReinforce:
    """Tests for reinforce_memory method."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.get = AsyncMock()
        provider.update = AsyncMock()
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_reinforce_memory(self, adapter, mock_provider):
        """Test reinforcing a memory boosts importance."""
        mock_provider.get.return_value = Memory(
            id="mem_123",
            content="Test",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            importance=0.5,
        )

        result = await adapter.reinforce_memory("mem_123", amount=0.1)

        assert result["success"] is True
        assert result["new_importance"] == 0.6
        mock_provider.update.assert_called_once_with("mem_123", importance=0.6)

    @pytest.mark.asyncio
    async def test_reinforce_memory_caps_at_one(self, adapter, mock_provider):
        """Test reinforcing doesn't exceed 1.0."""
        mock_provider.get.return_value = Memory(
            id="mem_123",
            content="Test",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            importance=0.95,
        )

        result = await adapter.reinforce_memory("mem_123", amount=0.2)

        assert result["success"] is True
        assert result["new_importance"] == 1.0

    @pytest.mark.asyncio
    async def test_reinforce_memory_not_found(self, adapter, mock_provider):
        """Test reinforcing a nonexistent memory."""
        mock_provider.get.return_value = None

        result = await adapter.reinforce_memory("nonexistent")

        assert result["success"] is False
        assert "not found" in result["error"]


class TestRoxyLayeredAdapterSearchWithSelfRag:
    """Tests for search_with_self_rag method."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock LayeredMemoryProvider."""
        provider = MagicMock(spec=LayeredMemoryProvider)
        provider.search = AsyncMock(return_value=[])
        return provider

    @pytest.fixture
    def adapter(self, mock_provider):
        """Create adapter with mock provider."""
        return RoxyLayeredAdapter(mock_provider)

    @pytest.mark.asyncio
    async def test_search_with_self_rag(self, adapter, mock_provider):
        """Test search_with_self_rag returns expected structure."""
        mock_provider.search.return_value = [
            SearchResult(
                memory=Memory(
                    id="mem_1",
                    content="Test",
                    memory_type=MemoryType.FACT,
                    scope=MemoryScope.USER,
                ),
                score=0.9,
            ),
        ]

        result = await adapter.search_with_self_rag(
            query="test",
            user_id="doug",
            limit=5,
        )

        assert "results" in result
        assert len(result["results"]) == 1
        assert "contradictions" in result
        assert result["has_contradictions"] is False
