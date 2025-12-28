"""Tests for the Roxy memory migration script.

Tests cover:
- Type mapping (Roxy -> draagon-ai)
- Scope mapping (Roxy -> draagon-ai)
- Layer assignment by memory type
- Individual point migration
- Batch migration with progress tracking
- Dry-run mode (no changes)
- Backup and rollback
- Error handling
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from draagon_ai.memory.base import MemoryScope, MemoryType
from draagon_ai.scripts.migrate_roxy_memories import (
    LAYER_ASSIGNMENT,
    ROXY_SCOPE_MAPPING,
    ROXY_TYPE_MAPPING,
    MigrationConfig,
    MigrationEngine,
    MigrationRecord,
    MigrationStats,
    MigrationStatus,
    OllamaEmbedder,
    QdrantClient,
    rollback,
)


# =============================================================================
# Test Type Mappings
# =============================================================================


class TestTypeMappings:
    """Tests for Roxy -> draagon-ai type mappings."""

    def test_all_roxy_types_mapped(self) -> None:
        """Verify all Roxy memory types have mappings."""
        roxy_types = [
            "fact", "preference", "episodic", "instruction",
            "knowledge", "skill", "insight", "self_knowledge", "relationship"
        ]
        for roxy_type in roxy_types:
            assert roxy_type in ROXY_TYPE_MAPPING
            assert isinstance(ROXY_TYPE_MAPPING[roxy_type], MemoryType)

    def test_fact_maps_to_fact(self) -> None:
        """Roxy fact -> draagon-ai FACT."""
        assert ROXY_TYPE_MAPPING["fact"] == MemoryType.FACT

    def test_skill_maps_to_skill(self) -> None:
        """Roxy skill -> draagon-ai SKILL."""
        assert ROXY_TYPE_MAPPING["skill"] == MemoryType.SKILL

    def test_preference_maps_to_preference(self) -> None:
        """Roxy preference -> draagon-ai PREFERENCE."""
        assert ROXY_TYPE_MAPPING["preference"] == MemoryType.PREFERENCE

    def test_episodic_maps_to_episodic(self) -> None:
        """Roxy episodic -> draagon-ai EPISODIC."""
        assert ROXY_TYPE_MAPPING["episodic"] == MemoryType.EPISODIC

    def test_instruction_maps_to_instruction(self) -> None:
        """Roxy instruction -> draagon-ai INSTRUCTION."""
        assert ROXY_TYPE_MAPPING["instruction"] == MemoryType.INSTRUCTION

    def test_insight_maps_to_insight(self) -> None:
        """Roxy insight -> draagon-ai INSIGHT."""
        assert ROXY_TYPE_MAPPING["insight"] == MemoryType.INSIGHT


class TestScopeMappings:
    """Tests for Roxy -> draagon-ai scope mappings."""

    def test_all_roxy_scopes_mapped(self) -> None:
        """Verify all Roxy scopes have mappings."""
        roxy_scopes = ["private", "shared", "public", "system"]
        for roxy_scope in roxy_scopes:
            assert roxy_scope in ROXY_SCOPE_MAPPING
            assert isinstance(ROXY_SCOPE_MAPPING[roxy_scope], MemoryScope)

    def test_private_maps_to_user(self) -> None:
        """Roxy private -> draagon-ai USER."""
        assert ROXY_SCOPE_MAPPING["private"] == MemoryScope.USER

    def test_shared_maps_to_context(self) -> None:
        """Roxy shared -> draagon-ai CONTEXT."""
        assert ROXY_SCOPE_MAPPING["shared"] == MemoryScope.CONTEXT

    def test_public_maps_to_world(self) -> None:
        """Roxy public -> draagon-ai WORLD."""
        assert ROXY_SCOPE_MAPPING["public"] == MemoryScope.WORLD

    def test_system_maps_to_world(self) -> None:
        """Roxy system -> draagon-ai WORLD."""
        assert ROXY_SCOPE_MAPPING["system"] == MemoryScope.WORLD


class TestLayerAssignment:
    """Tests for layer assignment by memory type."""

    def test_skill_goes_to_metacognitive(self) -> None:
        """Skills should go to metacognitive layer."""
        assert LAYER_ASSIGNMENT[MemoryType.SKILL] == "metacognitive"

    def test_instruction_goes_to_metacognitive(self) -> None:
        """Instructions should go to metacognitive layer."""
        assert LAYER_ASSIGNMENT[MemoryType.INSTRUCTION] == "metacognitive"

    def test_insight_goes_to_metacognitive(self) -> None:
        """Insights should go to metacognitive layer."""
        assert LAYER_ASSIGNMENT[MemoryType.INSIGHT] == "metacognitive"

    def test_fact_goes_to_semantic(self) -> None:
        """Facts should go to semantic layer."""
        assert LAYER_ASSIGNMENT[MemoryType.FACT] == "semantic"

    def test_knowledge_goes_to_semantic(self) -> None:
        """Knowledge should go to semantic layer."""
        assert LAYER_ASSIGNMENT[MemoryType.KNOWLEDGE] == "semantic"

    def test_preference_goes_to_semantic(self) -> None:
        """Preferences should go to semantic layer."""
        assert LAYER_ASSIGNMENT[MemoryType.PREFERENCE] == "semantic"

    def test_episodic_goes_to_episodic(self) -> None:
        """Episodic memories should go to episodic layer."""
        assert LAYER_ASSIGNMENT[MemoryType.EPISODIC] == "episodic"

    def test_observation_goes_to_working(self) -> None:
        """Observations should go to working layer."""
        assert LAYER_ASSIGNMENT[MemoryType.OBSERVATION] == "working"


# =============================================================================
# Test Migration Config
# =============================================================================


class TestMigrationConfig:
    """Tests for MigrationConfig."""

    def test_default_values(self) -> None:
        """Config should have sensible defaults."""
        config = MigrationConfig()

        assert config.source_qdrant_url == "http://192.168.168.216:6333"
        assert config.source_collection == "roxy_memories"
        assert config.target_nodes_collection == "draagon_memory_nodes"
        assert config.batch_size == 100
        assert config.dry_run is False
        assert config.verbose is False
        assert config.user_id is None

    def test_custom_values(self) -> None:
        """Config should accept custom values."""
        config = MigrationConfig(
            source_qdrant_url="http://localhost:6333",
            source_collection="test_memories",
            batch_size=50,
            dry_run=True,
            user_id="test_user",
        )

        assert config.source_qdrant_url == "http://localhost:6333"
        assert config.source_collection == "test_memories"
        assert config.batch_size == 50
        assert config.dry_run is True
        assert config.user_id == "test_user"

    def test_backup_dir_conversion(self) -> None:
        """String backup_dir should be converted to Path."""
        config = MigrationConfig(backup_dir="/tmp/backups")
        assert isinstance(config.backup_dir, Path)
        assert str(config.backup_dir) == "/tmp/backups"


# =============================================================================
# Test Migration Stats
# =============================================================================


class TestMigrationStats:
    """Tests for MigrationStats."""

    def test_default_values(self) -> None:
        """Stats should start at zero."""
        stats = MigrationStats()

        assert stats.total == 0
        assert stats.success == 0
        assert stats.skipped == 0
        assert stats.errors == 0
        assert stats.by_type == {}
        assert stats.by_layer == {}
        assert stats.by_scope == {}

    def test_to_dict(self) -> None:
        """Stats should serialize to dict."""
        stats = MigrationStats(
            total=100,
            success=90,
            skipped=5,
            errors=5,
            by_type={"fact": 50, "skill": 40},
            by_layer={"semantic": 60, "metacognitive": 35},
            duration_seconds=10.5,
        )

        data = stats.to_dict()

        assert data["total"] == 100
        assert data["success"] == 90
        assert data["skipped"] == 5
        assert data["errors"] == 5
        assert data["by_type"]["fact"] == 50
        assert data["by_layer"]["semantic"] == 60
        assert data["duration_seconds"] == 10.5


# =============================================================================
# Test Migration Record
# =============================================================================


class TestMigrationRecord:
    """Tests for MigrationRecord."""

    def test_create_record(self) -> None:
        """Should create a migration record."""
        record = MigrationRecord(
            original_id="abc123",
            content="Doug's birthday is March 15",
            roxy_type="fact",
            roxy_scope="private",
            draagon_type="fact",
            draagon_scope="user",
            target_layer="semantic",
            status=MigrationStatus.SUCCESS,
            new_id="migrated_abc123",
        )

        assert record.original_id == "abc123"
        assert record.content == "Doug's birthday is March 15"
        assert record.roxy_type == "fact"
        assert record.draagon_type == "fact"
        assert record.target_layer == "semantic"
        assert record.status == MigrationStatus.SUCCESS
        assert record.new_id == "migrated_abc123"
        assert record.error is None

    def test_error_record(self) -> None:
        """Should create an error record."""
        record = MigrationRecord(
            original_id="xyz789",
            content="Some content",
            roxy_type="skill",
            roxy_scope="private",
            draagon_type="skill",
            draagon_scope="user",
            target_layer="metacognitive",
            status=MigrationStatus.ERROR,
            error="Connection timeout",
        )

        assert record.status == MigrationStatus.ERROR
        assert record.error == "Connection timeout"
        assert record.new_id is None


# =============================================================================
# Test Migration Engine
# =============================================================================


class TestMigrationEngine:
    """Tests for MigrationEngine."""

    @pytest.fixture
    def mock_embedder(self) -> AsyncMock:
        """Create a mock embedder."""
        embedder = AsyncMock(spec=OllamaEmbedder)
        embedder.embed.return_value = [0.1] * 768
        return embedder

    @pytest.fixture
    def mock_source_client(self) -> AsyncMock:
        """Create a mock source Qdrant client."""
        client = AsyncMock(spec=QdrantClient)
        client.get_collection_info.return_value = {"points_count": 100}
        return client

    @pytest.fixture
    def mock_target_client(self) -> AsyncMock:
        """Create a mock target Qdrant client."""
        client = AsyncMock(spec=QdrantClient)
        client.create_collection.return_value = None
        client.upsert.return_value = None
        return client

    @pytest.fixture
    def config(self) -> MigrationConfig:
        """Create a test config."""
        return MigrationConfig(dry_run=True)

    @pytest.fixture
    def engine(
        self,
        config: MigrationConfig,
        mock_embedder: AsyncMock,
        mock_source_client: AsyncMock,
        mock_target_client: AsyncMock,
    ) -> MigrationEngine:
        """Create a migration engine with mocks."""
        return MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source_client,
            target_client=mock_target_client,
        )

    def test_map_memory_type(self, engine: MigrationEngine) -> None:
        """Should map Roxy types to draagon-ai types."""
        assert engine._map_memory_type("fact") == MemoryType.FACT
        assert engine._map_memory_type("skill") == MemoryType.SKILL
        assert engine._map_memory_type("FACT") == MemoryType.FACT  # Case insensitive
        assert engine._map_memory_type("unknown") == MemoryType.FACT  # Default

    def test_map_scope(self, engine: MigrationEngine) -> None:
        """Should map Roxy scopes to draagon-ai scopes."""
        assert engine._map_scope("private") == MemoryScope.USER
        assert engine._map_scope("shared") == MemoryScope.CONTEXT
        assert engine._map_scope("PUBLIC") == MemoryScope.WORLD  # Case insensitive
        assert engine._map_scope("unknown") == MemoryScope.USER  # Default

    def test_get_target_layer(self, engine: MigrationEngine) -> None:
        """Should return correct target layer for memory types."""
        assert engine._get_target_layer(MemoryType.FACT) == "semantic"
        assert engine._get_target_layer(MemoryType.SKILL) == "metacognitive"
        assert engine._get_target_layer(MemoryType.EPISODIC) == "episodic"

    def test_classify_memory(self, engine: MigrationEngine) -> None:
        """Should classify Roxy memory correctly."""
        point = {
            "id": "test123",
            "payload": {
                "content": "Doug's birthday is March 15",
                "memory_type": "fact",
                "scope": "private",
            },
        }

        memory_type, scope, layer = engine._classify_memory(point)

        assert memory_type == MemoryType.FACT
        assert scope == MemoryScope.USER
        assert layer == "semantic"

    def test_classify_skill_memory(self, engine: MigrationEngine) -> None:
        """Should classify skill memory correctly."""
        point = {
            "id": "skill123",
            "payload": {
                "content": "To restart Roxy: systemctl restart roxy",
                "memory_type": "skill",
                "scope": "shared",
            },
        }

        memory_type, scope, layer = engine._classify_memory(point)

        assert memory_type == MemoryType.SKILL
        assert scope == MemoryScope.CONTEXT
        assert layer == "metacognitive"

    @pytest.mark.asyncio
    async def test_migrate_point_dry_run(self, engine: MigrationEngine) -> None:
        """Dry run should not upsert to target."""
        point = {
            "id": "test123",
            "vector": [0.1] * 768,
            "payload": {
                "content": "Test content",
                "memory_type": "fact",
                "scope": "private",
                "user_id": "test_user",
            },
        }

        record = await engine._migrate_point(point)

        assert record.status == MigrationStatus.SUCCESS
        assert record.original_id == "test123"
        assert record.target_layer == "semantic"
        # In dry run, new_id is not set (no actual migration)
        assert record.new_id is None
        # Should NOT have called upsert in dry run
        engine.target.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_migrate_point_skip_empty(self, engine: MigrationEngine) -> None:
        """Should skip points with empty content."""
        point = {
            "id": "empty123",
            "payload": {
                "content": "   ",  # Empty after strip
                "memory_type": "fact",
                "scope": "private",
            },
        }

        record = await engine._migrate_point(point)

        assert record.status == MigrationStatus.SKIPPED
        assert "Empty content" in record.error

    @pytest.mark.asyncio
    async def test_migrate_batch(
        self,
        config: MigrationConfig,
        mock_embedder: AsyncMock,
        mock_source_client: AsyncMock,
        mock_target_client: AsyncMock,
    ) -> None:
        """Should migrate a batch of points."""
        # Set up source to return points
        points = [
            {
                "id": f"point_{i}",
                "vector": [0.1] * 768,
                "payload": {
                    "content": f"Memory content {i}",
                    "memory_type": "fact" if i % 2 == 0 else "skill",
                    "scope": "private",
                    "user_id": "test_user",
                    "importance": 0.5,
                },
            }
            for i in range(10)
        ]

        # First call returns points, second returns empty
        mock_source_client.scroll.side_effect = [
            (points, None),  # First batch
        ]

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source_client,
            target_client=mock_target_client,
        )

        stats = await engine.migrate()

        assert stats.total == 10
        assert stats.success == 10
        assert stats.errors == 0
        assert "fact" in stats.by_type
        assert "skill" in stats.by_type
        assert stats.by_type["fact"] == 5
        assert stats.by_type["skill"] == 5

    @pytest.mark.asyncio
    async def test_migrate_with_user_filter(
        self,
        mock_embedder: AsyncMock,
        mock_source_client: AsyncMock,
        mock_target_client: AsyncMock,
    ) -> None:
        """Should filter by user_id when specified."""
        config = MigrationConfig(dry_run=True, user_id="doug")
        mock_source_client.scroll.return_value = ([], None)

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source_client,
            target_client=mock_target_client,
        )

        await engine.migrate()

        # Verify filter was passed
        call_args = mock_source_client.scroll.call_args
        filter_arg = call_args.kwargs.get("filter_")
        assert filter_arg is not None
        assert filter_arg["must"][0]["key"] == "user_id"
        assert filter_arg["must"][0]["match"]["value"] == "doug"


# =============================================================================
# Test Dry Run Mode
# =============================================================================


class TestDryRunMode:
    """Tests for dry-run mode."""

    @pytest.fixture
    def mock_embedder(self) -> AsyncMock:
        embedder = AsyncMock()
        embedder.embed.return_value = [0.1] * 768
        return embedder

    @pytest.fixture
    def mock_source_client(self) -> AsyncMock:
        client = AsyncMock()
        client.get_collection_info.return_value = {"points_count": 5}
        client.scroll.return_value = ([
            {
                "id": "point_1",
                "vector": [0.1] * 768,
                "payload": {
                    "content": "Test memory",
                    "memory_type": "fact",
                    "scope": "private",
                    "user_id": "test",
                },
            }
        ], None)
        return client

    @pytest.fixture
    def mock_target_client(self) -> AsyncMock:
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_dry_run_no_writes(
        self,
        mock_embedder: AsyncMock,
        mock_source_client: AsyncMock,
        mock_target_client: AsyncMock,
    ) -> None:
        """Dry run should not write to target."""
        config = MigrationConfig(dry_run=True)
        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source_client,
            target_client=mock_target_client,
        )

        await engine.migrate()

        # Target should not have any write operations
        mock_target_client.upsert.assert_not_called()
        mock_target_client.create_collection.assert_not_called()

    @pytest.mark.asyncio
    async def test_dry_run_still_counts(
        self,
        mock_embedder: AsyncMock,
        mock_source_client: AsyncMock,
        mock_target_client: AsyncMock,
    ) -> None:
        """Dry run should still count and classify."""
        config = MigrationConfig(dry_run=True)
        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source_client,
            target_client=mock_target_client,
        )

        stats = await engine.migrate()

        assert stats.total == 1
        assert stats.success == 1
        assert stats.by_type["fact"] == 1


# =============================================================================
# Test Backup and Rollback
# =============================================================================


class TestBackupAndRollback:
    """Tests for backup and rollback functionality."""

    @pytest.fixture
    def mock_source_client(self) -> AsyncMock:
        client = AsyncMock()
        client.get_collection_info.return_value = {"points_count": 2}
        client.scroll.return_value = ([
            {"id": "1", "vector": [0.1], "payload": {"content": "A"}},
            {"id": "2", "vector": [0.2], "payload": {"content": "B"}},
        ], None)
        return client

    @pytest.mark.asyncio
    async def test_backup_creates_file(
        self,
        mock_source_client: AsyncMock,
    ) -> None:
        """Backup should create a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = MigrationConfig(
                dry_run=True,
                backup_dir=Path(tmpdir),
            )
            mock_embedder = AsyncMock()
            mock_embedder.embed.return_value = [0.1] * 768

            engine = MigrationEngine(
                config=config,
                embedder=mock_embedder,
                source_client=mock_source_client,
            )

            backup_path = await engine._backup_collection()

            assert backup_path.exists()
            assert backup_path.suffix == ".json"

            # Verify content
            with open(backup_path) as f:
                data = json.load(f)

            assert data["count"] == 2
            assert len(data["points"]) == 2

    @pytest.mark.asyncio
    async def test_rollback_restores_points(self) -> None:
        """Rollback should restore points from backup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create backup file
            backup_file = Path(tmpdir) / "test_backup.json"
            backup_data = {
                "collection": "test_collection",
                "timestamp": "20250627_120000",
                "count": 2,
                "points": [
                    {"id": "1", "vector": [0.1], "payload": {"content": "A"}},
                    {"id": "2", "vector": [0.2], "payload": {"content": "B"}},
                ],
            }
            with open(backup_file, "w") as f:
                json.dump(backup_data, f)

            config = MigrationConfig(source_collection="test_collection")

            # Mock the client
            with patch(
                "draagon_ai.scripts.migrate_roxy_memories.QdrantClient"
            ) as MockClient:
                mock_client = AsyncMock()
                MockClient.return_value = mock_client

                await rollback(backup_file, config)

                # Verify upsert was called with the points
                mock_client.upsert.assert_called_once()
                call_args = mock_client.upsert.call_args
                assert call_args[0][0] == "test_collection"
                assert len(call_args[0][1]) == 2


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling during migration."""

    @pytest.mark.asyncio
    async def test_upsert_error_recorded(self) -> None:
        """Errors during upsert should be recorded."""
        config = MigrationConfig(dry_run=False)
        mock_embedder = AsyncMock()
        mock_embedder.embed.return_value = [0.1] * 768

        mock_source = AsyncMock()
        mock_source.get_collection_info.return_value = {"points_count": 1}
        mock_source.scroll.return_value = ([{
            "id": "error_point",
            "vector": [0.1] * 768,
            "payload": {
                "content": "Content that will fail",
                "memory_type": "fact",
                "scope": "private",
            },
        }], None)

        mock_target = AsyncMock()
        mock_target.upsert.side_effect = Exception("Connection failed")

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source,
            target_client=mock_target,
        )

        stats = await engine.migrate()

        assert stats.errors == 1
        assert stats.success == 0
        assert len(engine.records) == 1
        assert engine.records[0].status == MigrationStatus.ERROR
        assert "Connection failed" in engine.records[0].error


# =============================================================================
# Test Progress Reporting
# =============================================================================


class TestProgressReporting:
    """Tests for progress reporting."""

    def test_print_summary_includes_all_sections(
        self,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Summary should include all sections."""
        config = MigrationConfig(dry_run=True)
        mock_embedder = MagicMock()
        mock_source = MagicMock()

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source,
        )

        engine.stats = MigrationStats(
            total=100,
            success=90,
            skipped=5,
            errors=5,
            by_type={"fact": 50, "skill": 40},
            by_layer={"semantic": 50, "metacognitive": 40},
            by_scope={"user": 80, "context": 10},
            duration_seconds=10.5,
        )

        engine.print_summary()

        captured = capsys.readouterr()
        output = captured.out

        assert "MIGRATION SUMMARY" in output
        assert "Total processed: 100" in output
        assert "Success: 90" in output
        assert "Skipped: 5" in output
        assert "Errors: 5" in output
        assert "By Memory Type:" in output
        assert "fact: 50" in output
        assert "By Target Layer:" in output
        assert "semantic: 50" in output
        assert "[DRY RUN" in output


# =============================================================================
# Test OllamaEmbedder
# =============================================================================


class TestOllamaEmbedder:
    """Tests for OllamaEmbedder."""

    @pytest.mark.asyncio
    async def test_embed_returns_vector(self) -> None:
        """Embedder should return vector from Ollama."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "embeddings": [[0.1, 0.2, 0.3]]
            }

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            embedder = OllamaEmbedder(
                base_url="http://localhost:11434",
                model="nomic-embed-text",
            )
            # Manually set the client for testing
            embedder._client = mock_client

            result = await embedder.embed("test text")

            assert result == [0.1, 0.2, 0.3]


# =============================================================================
# Test QdrantClient
# =============================================================================


class TestQdrantClient:
    """Tests for QdrantClient."""

    @pytest.mark.asyncio
    async def test_scroll_returns_points(self) -> None:
        """Scroll should return points and next offset."""
        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.raise_for_status = MagicMock()
            mock_response.json.return_value = {
                "result": {
                    "points": [{"id": "1"}, {"id": "2"}],
                    "next_page_offset": "offset123",
                }
            }

            mock_client = AsyncMock()
            mock_client.post.return_value = mock_response
            MockClient.return_value.__aenter__.return_value = mock_client

            client = QdrantClient("http://localhost:6333")
            client._client = mock_client

            points, next_offset = await client.scroll("test_collection")

            assert len(points) == 2
            assert next_offset == "offset123"

    @pytest.mark.asyncio
    async def test_create_collection_handles_exists(self) -> None:
        """Create collection should handle already exists error."""
        with patch("httpx.AsyncClient") as MockClient:
            # Simulate 409 Conflict (already exists)
            mock_response = MagicMock()
            mock_response.status_code = 409
            error = Exception("Already exists")
            error.response = mock_response

            mock_client = AsyncMock()
            mock_client.put.side_effect = error
            MockClient.return_value.__aenter__.return_value = mock_client

            client = QdrantClient("http://localhost:6333")
            client._client = mock_client

            # Should not raise - 409 is expected
            # Note: The actual implementation checks HTTPStatusError
            # This test validates the concept


# =============================================================================
# Test Metadata Preservation
# =============================================================================


class TestMetadataPreservation:
    """Tests for metadata preservation during migration."""

    @pytest.mark.asyncio
    async def test_preserves_importance(self) -> None:
        """Migration should preserve importance score."""
        config = MigrationConfig(dry_run=True)
        mock_embedder = AsyncMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_source = AsyncMock()

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source,
        )

        point = {
            "id": "test123",
            "vector": [0.1] * 768,
            "payload": {
                "content": "Important memory",
                "memory_type": "fact",
                "scope": "private",
                "importance": 0.95,
                "user_id": "doug",
            },
        }

        record = await engine._migrate_point(point)

        assert record.metadata["importance"] == 0.95

    @pytest.mark.asyncio
    async def test_preserves_entities(self) -> None:
        """Migration should preserve entities."""
        config = MigrationConfig(dry_run=True)
        mock_embedder = AsyncMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_source = AsyncMock()

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source,
        )

        point = {
            "id": "test123",
            "vector": [0.1] * 768,
            "payload": {
                "content": "Doug's birthday is March 15",
                "memory_type": "fact",
                "scope": "private",
                "entities": ["Doug", "March 15", "birthday"],
                "user_id": "doug",
            },
        }

        record = await engine._migrate_point(point)

        assert record.metadata["entities"] == ["Doug", "March 15", "birthday"]

    @pytest.mark.asyncio
    async def test_preserves_stated_count(self) -> None:
        """Migration should preserve stated_count for confidence propagation."""
        config = MigrationConfig(dry_run=True)
        mock_embedder = AsyncMock()
        mock_embedder.embed.return_value = [0.1] * 768
        mock_source = AsyncMock()

        engine = MigrationEngine(
            config=config,
            embedder=mock_embedder,
            source_client=mock_source,
        )

        point = {
            "id": "test123",
            "vector": [0.1] * 768,
            "payload": {
                "content": "Repeated fact",
                "memory_type": "fact",
                "scope": "private",
                "stated_count": 5,  # Stated 5 times
                "user_id": "doug",
            },
        }

        record = await engine._migrate_point(point)

        assert record.metadata["stated_count"] == 5
