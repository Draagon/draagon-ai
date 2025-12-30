"""Integration tests for ExtensionConfigService with real Qdrant.

These tests verify the complete configuration flow:
- YAML loading and parsing
- Memory storage and retrieval
- Priority resolution (memory > YAML > default)
- Network context persistence

Requirements:
- Running Qdrant instance (default: http://192.168.168.216:6333)
- qdrant-client package installed
"""

import asyncio
import os
import pytest
from datetime import datetime
from uuid import uuid4

try:
    from qdrant_client import AsyncQdrantClient
    from qdrant_client.models import VectorParams, Distance
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from draagon_ai.extensions.config_service import (
    ExtensionConfigService,
    NetworkService,
)


QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.168.216:6333")
TEST_COLLECTION = f"test_ext_config_{uuid4().hex[:8]}"


async def check_qdrant_connection() -> bool:
    """Check if Qdrant is available."""
    if not QDRANT_CLIENT_AVAILABLE:
        return False
    try:
        client = AsyncQdrantClient(url=QDRANT_URL)
        await client.get_collections()
        return True
    except Exception:
        return False


class SimpleMemoryService:
    """Simple memory service backed by Qdrant for integration tests.

    This is a minimal implementation that stores config as memories.
    """

    def __init__(self, collection: str = TEST_COLLECTION):
        self._client = AsyncQdrantClient(url=QDRANT_URL)
        self._collection = collection
        self._initialized = False

    async def initialize(self):
        """Create collection if needed."""
        if self._initialized:
            return

        collections = await self._client.get_collections()
        if self._collection not in [c.name for c in collections.collections]:
            await self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
        self._initialized = True

    async def store(
        self,
        content: str,
        user_id: str = "system",
        memory_type: str = "fact",
        entities: list[str] | None = None,
        metadata: dict | None = None,
        **kwargs,
    ) -> str:
        """Store a memory."""
        await self.initialize()

        # Qdrant requires UUID or int for point ID
        memory_id = str(uuid4())

        # Simple deterministic embedding based on content hash
        import hashlib
        h = hashlib.sha256(content.encode()).digest()
        embedding = [float(b) / 255.0 for b in h[:32]] * 24  # 768 dims

        payload = {
            "content": content,
            "user_id": user_id,
            "memory_type": memory_type,
            "entities": entities or [],
            **(metadata or {}),
        }

        from qdrant_client.models import PointStruct
        await self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=memory_id, vector=embedding, payload=payload)],
        )

        return memory_id

    async def search(
        self,
        query: str,
        user_id: str = "system",
        limit: int = 5,
        memory_type: str | None = None,
        **kwargs,
    ) -> list:
        """Search memories."""
        await self.initialize()

        # Simple deterministic embedding based on query hash
        import hashlib
        h = hashlib.sha256(query.encode()).digest()
        embedding = [float(b) / 255.0 for b in h[:32]] * 24  # 768 dims

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
        if memory_type:
            conditions.append(
                FieldCondition(key="memory_type", match=MatchValue(value=memory_type))
            )

        results = await self._client.query_points(
            collection_name=self._collection,
            query=embedding,
            query_filter=Filter(must=conditions),
            limit=limit,
        )

        # Convert to expected format
        class MemoryResult:
            def __init__(self, content: str, metadata: dict):
                self.content = content
                self.metadata = metadata

        return [
            MemoryResult(r.payload.get("content", ""), r.payload)
            for r in results.points
        ]

    async def cleanup(self):
        """Delete test collection."""
        try:
            await self._client.delete_collection(self._collection)
        except Exception:
            pass


@pytest.fixture
async def memory_service():
    """Create a real memory service for tests."""
    service = SimpleMemoryService()
    await service.initialize()
    yield service
    await service.cleanup()


@pytest.fixture
def yaml_config():
    """Sample YAML configuration."""
    return {
        "extensions": {
            "security-monitor": {
                "config": {
                    "check_interval_seconds": 300,
                    "notifications": {
                        "voice": {
                            "min_severity": "high",
                        },
                    },
                    "network": {
                        "home_net": "192.168.168.0/24",
                        "known_services": [
                            {"name": "Plex", "ip": "192.168.168.204"},
                        ],
                    },
                },
            },
        },
    }


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
@pytest.mark.skipif(
    not asyncio.get_event_loop().run_until_complete(check_qdrant_connection()),
    reason="Qdrant not available"
)
async def test_config_roundtrip_with_real_qdrant(memory_service, yaml_config):
    """Test storing and retrieving config with real Qdrant."""
    config_service = ExtensionConfigService(
        memory_service=memory_service,
        yaml_config=yaml_config,
    )

    # Set a preference via voice
    await config_service.set(
        extension="security-monitor",
        key="notifications.voice.min_severity",
        value="critical",
        source="voice",
        user_id="doug",
    )

    # Clear cache to force memory lookup
    config_service.clear_cache()

    # Retrieve - should get memory value, not YAML
    value = await config_service.get(
        "security-monitor",
        "notifications.voice.min_severity",
    )

    assert value == "critical"  # From memory, not "high" from YAML


@pytest.mark.asyncio
@pytest.mark.skipif(
    not asyncio.get_event_loop().run_until_complete(check_qdrant_connection()),
    reason="Qdrant not available"
)
async def test_network_service_persistence(memory_service, yaml_config):
    """Test that learned network services persist in Qdrant."""
    config_service = ExtensionConfigService(
        memory_service=memory_service,
        yaml_config=yaml_config,
    )

    # Learn a new network service via voice
    await config_service.add_network_service(
        name="NAS",
        ip="192.168.168.202",
        ports=[445, 139],
        protocols=["smb"],
        notes="Synology storage",
        user_id="doug",
    )

    # Clear cache
    config_service.clear_cache()

    # Get network context - should include both YAML and learned
    context = await config_service.get_network_context("security-monitor")

    names = [s.name for s in context.known_services]
    assert "Plex" in names  # From YAML
    assert "NAS" in names   # From memory

    # Verify learned flag
    nas = next(s for s in context.known_services if s.name == "NAS")
    assert nas.learned is True
    assert nas.ports == [445, 139]


@pytest.mark.asyncio
@pytest.mark.skipif(
    not asyncio.get_event_loop().run_until_complete(check_qdrant_connection()),
    reason="Qdrant not available"
)
async def test_memory_priority_over_yaml(memory_service, yaml_config):
    """Test that memory values take priority over YAML."""
    config_service = ExtensionConfigService(
        memory_service=memory_service,
        yaml_config=yaml_config,
    )

    # YAML says check_interval_seconds = 300
    yaml_value = await config_service.get(
        "security-monitor",
        "check_interval_seconds",
    )
    assert yaml_value == 300

    # User sets via voice to 60
    await config_service.set(
        extension="security-monitor",
        key="check_interval_seconds",
        value=60,
        source="voice",
        user_id="doug",
    )

    # Clear cache
    config_service.clear_cache()

    # Should now return 60 (memory) not 300 (YAML)
    memory_value = await config_service.get(
        "security-monitor",
        "check_interval_seconds",
    )
    assert memory_value == 60


@pytest.mark.asyncio
@pytest.mark.skipif(
    not asyncio.get_event_loop().run_until_complete(check_qdrant_connection()),
    reason="Qdrant not available"
)
async def test_multiple_extensions_isolated(memory_service):
    """Test that different extensions have isolated config."""
    yaml_config = {
        "extensions": {
            "security-monitor": {"config": {"interval": 60}},
            "calendar-sync": {"config": {"interval": 300}},
        },
    }

    config_service = ExtensionConfigService(
        memory_service=memory_service,
        yaml_config=yaml_config,
    )

    # Set different values for each extension
    await config_service.set("security-monitor", "custom_key", "security_value")
    await config_service.set("calendar-sync", "custom_key", "calendar_value")

    config_service.clear_cache()

    # Verify isolation
    sec_value = await config_service.get("security-monitor", "custom_key")
    cal_value = await config_service.get("calendar-sync", "custom_key")

    assert sec_value == "security_value"
    assert cal_value == "calendar_value"
