#!/usr/bin/env python3
"""Migration script for migrating Roxy memories to draagon-ai layered structure.

This script reads all memories from Roxy's current flat Qdrant collection
and migrates them to the new layered structure (working, episodic, semantic,
metacognitive) using the LayeredMemoryProvider.

Usage:
    # Dry run (no changes)
    python -m draagon_ai.scripts.migrate_roxy_memories --dry-run

    # Migrate all memories
    python -m draagon_ai.scripts.migrate_roxy_memories

    # Migrate specific user
    python -m draagon_ai.scripts.migrate_roxy_memories --user-id doug

    # With verbose output
    python -m draagon_ai.scripts.migrate_roxy_memories --verbose

    # Create backup before migration
    python -m draagon_ai.scripts.migrate_roxy_memories --backup

    # Rollback to backup
    python -m draagon_ai.scripts.migrate_roxy_memories --rollback backup_20250627_120000.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import httpx

from draagon_ai.memory.base import MemoryScope, MemoryType

logger = logging.getLogger(__name__)


# =============================================================================
# Type Mappings (Roxy -> draagon-ai)
# =============================================================================

# Roxy memory types to draagon-ai memory types
ROXY_TYPE_MAPPING = {
    "fact": MemoryType.FACT,
    "preference": MemoryType.PREFERENCE,
    "episodic": MemoryType.EPISODIC,
    "instruction": MemoryType.INSTRUCTION,
    "knowledge": MemoryType.KNOWLEDGE,
    "skill": MemoryType.SKILL,
    "insight": MemoryType.INSIGHT,
    "self_knowledge": MemoryType.SELF_KNOWLEDGE,
    "relationship": MemoryType.RELATIONSHIP,
}

# Roxy scopes to draagon-ai scopes
ROXY_SCOPE_MAPPING = {
    "private": MemoryScope.USER,
    "shared": MemoryScope.CONTEXT,
    "public": MemoryScope.WORLD,
    "system": MemoryScope.WORLD,
}

# Layer assignment by memory type
LAYER_ASSIGNMENT = {
    MemoryType.OBSERVATION: "working",
    MemoryType.EPISODIC: "episodic",
    MemoryType.FACT: "semantic",
    MemoryType.KNOWLEDGE: "semantic",
    MemoryType.PREFERENCE: "semantic",
    MemoryType.BELIEF: "semantic",
    MemoryType.INSTRUCTION: "metacognitive",
    MemoryType.SKILL: "metacognitive",
    MemoryType.INSIGHT: "metacognitive",
    MemoryType.SELF_KNOWLEDGE: "metacognitive",
    MemoryType.RELATIONSHIP: "semantic",
}


# =============================================================================
# Data Classes
# =============================================================================


class MigrationStatus(str, Enum):
    """Status of a memory migration."""

    PENDING = "pending"
    SUCCESS = "success"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class MigrationRecord:
    """Record of a single memory migration."""

    original_id: str
    content: str
    roxy_type: str
    roxy_scope: str
    draagon_type: str
    draagon_scope: str
    target_layer: str
    status: MigrationStatus
    new_id: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MigrationStats:
    """Statistics for the migration run."""

    total: int = 0
    success: int = 0
    skipped: int = 0
    errors: int = 0
    by_type: dict[str, int] = field(default_factory=dict)
    by_layer: dict[str, int] = field(default_factory=dict)
    by_scope: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total": self.total,
            "success": self.success,
            "skipped": self.skipped,
            "errors": self.errors,
            "by_type": self.by_type,
            "by_layer": self.by_layer,
            "by_scope": self.by_scope,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class MigrationConfig:
    """Configuration for the migration."""

    # Qdrant settings
    source_qdrant_url: str = "http://192.168.168.216:6333"
    source_collection: str = "roxy_memories"
    target_nodes_collection: str = "draagon_memory_nodes"
    target_edges_collection: str = "draagon_memory_edges"

    # Embedding settings
    embedding_url: str = "http://192.168.168.200:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dimension: int = 768

    # Migration options
    batch_size: int = 100
    dry_run: bool = False
    verbose: bool = False
    user_id: str | None = None
    backup_dir: Path = field(default_factory=lambda: Path("./backups"))

    def __post_init__(self) -> None:
        if isinstance(self.backup_dir, str):
            self.backup_dir = Path(self.backup_dir)


# =============================================================================
# Embedding Provider
# =============================================================================


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        ...


class OllamaEmbedder:
    """Embedding provider using Ollama."""

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text using Ollama."""
        client = await self._get_client()
        response = await client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": text},
        )
        response.raise_for_status()
        data = response.json()
        # Ollama returns embeddings in data.embeddings[0]
        return data.get("embeddings", [[]])[0]

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None


# =============================================================================
# Qdrant Client
# =============================================================================


class QdrantClient:
    """Simple async Qdrant client for migration."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def scroll(
        self,
        collection: str,
        limit: int = 100,
        offset: str | None = None,
        filter_: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], str | None]:
        """Scroll through collection points.

        Returns:
            Tuple of (points, next_offset)
        """
        client = await self._get_client()
        payload = {
            "limit": limit,
            "with_payload": True,
            "with_vector": True,
        }
        if offset:
            payload["offset"] = offset
        if filter_:
            payload["filter"] = filter_

        response = await client.post(
            f"{self.base_url}/collections/{collection}/points/scroll",
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        result = data.get("result", {})
        points = result.get("points", [])
        next_offset = result.get("next_page_offset")
        return points, next_offset

    async def get_collection_info(self, collection: str) -> dict[str, Any]:
        """Get collection info."""
        client = await self._get_client()
        response = await client.get(
            f"{self.base_url}/collections/{collection}",
        )
        response.raise_for_status()
        return response.json().get("result", {})

    async def upsert(
        self,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        """Upsert points to collection."""
        client = await self._get_client()
        response = await client.put(
            f"{self.base_url}/collections/{collection}/points",
            json={"points": points},
        )
        response.raise_for_status()

    async def create_collection(
        self,
        collection: str,
        vector_size: int,
        distance: str = "Cosine",
    ) -> None:
        """Create a collection if it doesn't exist."""
        client = await self._get_client()
        try:
            response = await client.put(
                f"{self.base_url}/collections/{collection}",
                json={
                    "vectors": {
                        "size": vector_size,
                        "distance": distance,
                    }
                },
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 409:  # Already exists
                raise


# =============================================================================
# Migration Engine
# =============================================================================


class MigrationEngine:
    """Engine for migrating Roxy memories to layered structure."""

    def __init__(
        self,
        config: MigrationConfig,
        embedder: EmbeddingProvider,
        source_client: QdrantClient,
        target_client: QdrantClient | None = None,
    ):
        self.config = config
        self.embedder = embedder
        self.source = source_client
        # Target can be same as source or separate Qdrant instance
        self.target = target_client or source_client
        self.stats = MigrationStats()
        self.records: list[MigrationRecord] = []

    def _map_memory_type(self, roxy_type: str) -> MemoryType:
        """Map Roxy memory type to draagon-ai type."""
        return ROXY_TYPE_MAPPING.get(roxy_type.lower(), MemoryType.FACT)

    def _map_scope(self, roxy_scope: str) -> MemoryScope:
        """Map Roxy scope to draagon-ai scope."""
        return ROXY_SCOPE_MAPPING.get(roxy_scope.lower(), MemoryScope.USER)

    def _get_target_layer(self, memory_type: MemoryType) -> str:
        """Determine target layer for a memory type."""
        return LAYER_ASSIGNMENT.get(memory_type, "semantic")

    def _classify_memory(
        self,
        point: dict[str, Any],
    ) -> tuple[MemoryType, MemoryScope, str]:
        """Classify a Roxy memory into draagon-ai types.

        Returns:
            Tuple of (memory_type, scope, target_layer)
        """
        payload = point.get("payload", {})

        # Get Roxy type and scope
        roxy_type = payload.get("memory_type", "fact")
        roxy_scope = payload.get("scope", "private")

        # Map to draagon-ai types
        memory_type = self._map_memory_type(roxy_type)
        scope = self._map_scope(roxy_scope)
        layer = self._get_target_layer(memory_type)

        return memory_type, scope, layer

    async def _migrate_point(
        self,
        point: dict[str, Any],
    ) -> MigrationRecord:
        """Migrate a single memory point.

        Args:
            point: Qdrant point with payload and vector

        Returns:
            MigrationRecord with status
        """
        payload = point.get("payload", {})
        original_id = str(point.get("id", ""))
        content = payload.get("content", "")

        # Skip empty content
        if not content.strip():
            return MigrationRecord(
                original_id=original_id,
                content=content[:100],
                roxy_type=payload.get("memory_type", "unknown"),
                roxy_scope=payload.get("scope", "unknown"),
                draagon_type="",
                draagon_scope="",
                target_layer="",
                status=MigrationStatus.SKIPPED,
                error="Empty content",
            )

        # Classify memory
        memory_type, scope, layer = self._classify_memory(point)

        record = MigrationRecord(
            original_id=original_id,
            content=content[:100] + ("..." if len(content) > 100 else ""),
            roxy_type=payload.get("memory_type", "fact"),
            roxy_scope=payload.get("scope", "private"),
            draagon_type=memory_type.value,
            draagon_scope=scope.value,
            target_layer=layer,
            status=MigrationStatus.PENDING,
            metadata={
                "user_id": payload.get("user_id"),
                "importance": payload.get("importance", 0.5),
                "entities": payload.get("entities", []),
                "created_at": payload.get("created_at"),
                "source": payload.get("source"),
                "stated_count": payload.get("stated_count", 1),
            },
        )

        if self.config.dry_run:
            record.status = MigrationStatus.SUCCESS
            return record

        try:
            # Get vector from source or regenerate
            vector = point.get("vector")
            if not vector:
                vector = await self.embedder.embed(content)

            # Create new point for target collection
            # Use a new UUID for the migrated point (Qdrant requires UUID or int)
            new_id = str(uuid.uuid4())
            new_point = {
                "id": new_id,
                "vector": vector,
                "payload": {
                    # Core fields
                    "content": content,
                    "memory_type": memory_type.value,
                    "scope": scope.value,
                    "layer": layer,
                    # Preserved metadata
                    "user_id": payload.get("user_id"),
                    "context_id": payload.get("household_id"),
                    "importance": payload.get("importance", 0.5),
                    "confidence": 1.0,
                    "entities": payload.get("entities", []),
                    "source": payload.get("source", "migration"),
                    "created_at": payload.get("created_at", datetime.now().isoformat()),
                    # Migration metadata
                    "migrated_from": original_id,
                    "migration_date": datetime.now().isoformat(),
                    "original_roxy_type": payload.get("memory_type"),
                    "original_roxy_scope": payload.get("scope"),
                    "stated_count": payload.get("stated_count", 1),
                },
            }

            # Upsert to target collection
            await self.target.upsert(
                self.config.target_nodes_collection,
                [new_point],
            )

            record.status = MigrationStatus.SUCCESS
            record.new_id = new_id

        except Exception as e:
            record.status = MigrationStatus.ERROR
            record.error = str(e)
            logger.error(f"Error migrating {original_id}: {e}")

        return record

    async def _backup_collection(self) -> Path:
        """Create a backup of the source collection.

        Returns:
            Path to backup file
        """
        self.config.backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = self.config.backup_dir / f"backup_{timestamp}.json"

        logger.info(f"Creating backup to {backup_file}...")

        all_points = []
        offset = None

        while True:
            points, next_offset = await self.source.scroll(
                self.config.source_collection,
                limit=self.config.batch_size,
                offset=offset,
            )

            all_points.extend(points)

            if not next_offset:
                break
            offset = next_offset

        with open(backup_file, "w") as f:
            json.dump(
                {
                    "collection": self.config.source_collection,
                    "timestamp": timestamp,
                    "count": len(all_points),
                    "points": all_points,
                },
                f,
                indent=2,
            )

        logger.info(f"Backup complete: {len(all_points)} points saved")
        return backup_file

    async def migrate(self, create_backup: bool = False) -> MigrationStats:
        """Run the migration.

        Args:
            create_backup: Create backup before migration

        Returns:
            MigrationStats with results
        """
        start_time = datetime.now()

        # Create backup if requested
        if create_backup and not self.config.dry_run:
            await self._backup_collection()

        # Ensure target collection exists
        if not self.config.dry_run:
            await self.target.create_collection(
                self.config.target_nodes_collection,
                self.config.embedding_dimension,
            )

        # Build filter for user_id if specified
        filter_ = None
        if self.config.user_id:
            filter_ = {
                "must": [
                    {"key": "user_id", "match": {"value": self.config.user_id}}
                ]
            }

        # Get collection info
        try:
            info = await self.source.get_collection_info(self.config.source_collection)
            total_points = info.get("points_count", 0)
            logger.info(f"Source collection has {total_points} total points")
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")

        # Scroll through and migrate
        offset = None
        batch_num = 0

        while True:
            batch_num += 1
            logger.info(f"Processing batch {batch_num}...")

            points, next_offset = await self.source.scroll(
                self.config.source_collection,
                limit=self.config.batch_size,
                offset=offset,
                filter_=filter_,
            )

            if not points:
                break

            # Process batch
            for point in points:
                record = await self._migrate_point(point)
                self.records.append(record)
                self.stats.total += 1

                # Update stats
                if record.status == MigrationStatus.SUCCESS:
                    self.stats.success += 1
                    # Track by type
                    type_key = record.draagon_type
                    self.stats.by_type[type_key] = self.stats.by_type.get(type_key, 0) + 1
                    # Track by layer
                    layer_key = record.target_layer
                    self.stats.by_layer[layer_key] = self.stats.by_layer.get(layer_key, 0) + 1
                    # Track by scope
                    scope_key = record.draagon_scope
                    self.stats.by_scope[scope_key] = self.stats.by_scope.get(scope_key, 0) + 1
                elif record.status == MigrationStatus.SKIPPED:
                    self.stats.skipped += 1
                else:
                    self.stats.errors += 1

                if self.config.verbose:
                    status_icon = {
                        MigrationStatus.SUCCESS: "✓",
                        MigrationStatus.SKIPPED: "→",
                        MigrationStatus.ERROR: "✗",
                    }.get(record.status, "?")
                    logger.info(
                        f"  {status_icon} {record.roxy_type} -> {record.target_layer}/{record.draagon_type}"
                    )

            if not next_offset:
                break
            offset = next_offset

        end_time = datetime.now()
        self.stats.duration_seconds = (end_time - start_time).total_seconds()

        return self.stats

    def print_summary(self) -> None:
        """Print migration summary."""
        print("\n" + "=" * 60)
        print("MIGRATION SUMMARY")
        print("=" * 60)
        print(f"Total processed: {self.stats.total}")
        print(f"  Success: {self.stats.success}")
        print(f"  Skipped: {self.stats.skipped}")
        print(f"  Errors: {self.stats.errors}")
        print(f"Duration: {self.stats.duration_seconds:.2f} seconds")

        if self.stats.by_type:
            print("\nBy Memory Type:")
            for type_name, count in sorted(self.stats.by_type.items()):
                print(f"  {type_name}: {count}")

        if self.stats.by_layer:
            print("\nBy Target Layer:")
            for layer_name, count in sorted(self.stats.by_layer.items()):
                print(f"  {layer_name}: {count}")

        if self.stats.by_scope:
            print("\nBy Scope:")
            for scope_name, count in sorted(self.stats.by_scope.items()):
                print(f"  {scope_name}: {count}")

        if self.config.dry_run:
            print("\n[DRY RUN - No changes made]")

        print("=" * 60)


async def rollback(backup_file: Path, config: MigrationConfig) -> None:
    """Rollback migration from backup file.

    Args:
        backup_file: Path to backup JSON file
        config: Migration config
    """
    logger.info(f"Rolling back from {backup_file}...")

    with open(backup_file, "r") as f:
        backup_data = json.load(f)

    points = backup_data.get("points", [])
    collection = backup_data.get("collection", config.source_collection)

    if not points:
        logger.warning("No points in backup file")
        return

    client = QdrantClient(config.source_qdrant_url)

    try:
        # Upsert in batches
        for i in range(0, len(points), config.batch_size):
            batch = points[i : i + config.batch_size]
            await client.upsert(collection, batch)
            logger.info(f"Restored {min(i + config.batch_size, len(points))}/{len(points)} points")

        logger.info(f"Rollback complete: {len(points)} points restored")
    finally:
        await client.close()


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Roxy memories to draagon-ai layered structure"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview migration without making changes",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--user-id",
        help="Migrate only memories for specific user",
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Create backup before migration",
    )
    parser.add_argument(
        "--rollback",
        metavar="BACKUP_FILE",
        help="Rollback from backup file",
    )
    parser.add_argument(
        "--source-url",
        default="http://192.168.168.216:6333",
        help="Source Qdrant URL",
    )
    parser.add_argument(
        "--source-collection",
        default="roxy_memories",
        help="Source collection name",
    )
    parser.add_argument(
        "--target-collection",
        default="draagon_memory_nodes",
        help="Target collection name",
    )
    parser.add_argument(
        "--embedding-url",
        default="http://192.168.168.200:11434",
        help="Ollama URL for embeddings",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--backup-dir",
        default="./backups",
        help="Directory for backup files",
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    config = MigrationConfig(
        source_qdrant_url=args.source_url,
        source_collection=args.source_collection,
        target_nodes_collection=args.target_collection,
        embedding_url=args.embedding_url,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        verbose=args.verbose,
        user_id=args.user_id,
        backup_dir=Path(args.backup_dir),
    )

    # Handle rollback
    if args.rollback:
        await rollback(Path(args.rollback), config)
        return 0

    # Create clients
    embedder = OllamaEmbedder(config.embedding_url, config.embedding_model)
    source_client = QdrantClient(config.source_qdrant_url)

    try:
        engine = MigrationEngine(
            config=config,
            embedder=embedder,
            source_client=source_client,
        )

        stats = await engine.migrate(create_backup=args.backup)
        engine.print_summary()

        # Return error code if there were errors
        return 1 if stats.errors > 0 else 0

    finally:
        await embedder.close()
        await source_client.close()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
