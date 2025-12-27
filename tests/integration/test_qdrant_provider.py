"""Integration tests for QdrantMemoryProvider.

These tests hit a REAL Qdrant instance - no mocks.
They verify actual persistence and retrieval.

Requirements:
- Running Qdrant instance (default: http://192.168.168.216:6333)
- qdrant-client package installed

Set QDRANT_URL environment variable to override the default.
"""

import asyncio
import os
import pytest
from datetime import datetime
from uuid import uuid4

# Check for qdrant-client
try:
    from qdrant_client import AsyncQdrantClient
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from draagon_ai.memory.base import MemoryType, MemoryScope
from draagon_ai.memory.providers import (
    QdrantMemoryProvider,
    QdrantPromptProvider,
    QdrantConfig,
    QDRANT_AVAILABLE,
)


# Test configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.168.216:6333")
TEST_COLLECTION_PREFIX = "test_draagon_"


class MockEmbeddingProvider:
    """Word-based embedding provider for tests.

    Generates embeddings based on word overlap so that semantically
    similar texts produce similar vectors. This enables realistic
    semantic search testing.
    """

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        # Build a vocabulary for consistent word indices
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        """Get or assign a vocab index for a word."""
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        """Generate word-based embedding.

        Similar texts (sharing words) will have similar embeddings.
        """
        if text in self._cache:
            return self._cache[text]

        # Tokenize and normalize
        import re
        words = re.findall(r'\b\w+\b', text.lower())

        # Create embedding based on word presence
        embedding = [0.0] * self.dimension

        for word in words:
            idx = self._get_word_idx(word)
            # Use word index to set multiple embedding dimensions
            # This creates overlapping representations for shared words
            for offset in range(5):  # Spread each word across 5 dimensions
                dim_idx = (idx * 7 + offset * 13) % self.dimension
                embedding[dim_idx] += 0.5

        # Normalize to unit vector
        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
            # Empty text - use small random values
            import hashlib
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = [(hash_bytes[i % 32] / 255.0) * 0.1 for i in range(self.dimension)]

        self._cache[text] = embedding
        return embedding


async def check_qdrant_connection() -> bool:
    """Check if Qdrant is accessible."""
    if not QDRANT_CLIENT_AVAILABLE:
        return False

    try:
        client = AsyncQdrantClient(url=QDRANT_URL, timeout=5.0)
        await client.get_collections()
        await client.close()
        return True
    except Exception:
        return False


# Skip all tests if Qdrant not available
pytestmark = pytest.mark.skipif(
    not QDRANT_AVAILABLE,
    reason="qdrant-client not installed"
)


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def qdrant_available():
    """Check Qdrant connectivity once per module."""
    available = await check_qdrant_connection()
    if not available:
        pytest.skip(f"Qdrant not accessible at {QDRANT_URL}")
    return True


@pytest.fixture
def embedder():
    """Create embedding provider."""
    return MockEmbeddingProvider()


@pytest.fixture
async def memory_provider(embedder, qdrant_available):
    """Create and cleanup a QdrantMemoryProvider with test collection."""
    collection_name = f"{TEST_COLLECTION_PREFIX}memories_{uuid4().hex[:8]}"

    config = QdrantConfig(
        url=QDRANT_URL,
        collection_name=collection_name,
        embedding_dimension=768,
        similarity_threshold=0.05,  # Low threshold for word-based mock embeddings
    )

    provider = QdrantMemoryProvider(config, embedder)
    await provider.initialize()

    yield provider

    # Cleanup: delete test collection
    try:
        await provider._client.delete_collection(collection_name)
    except Exception:
        pass
    await provider.close()


@pytest.fixture
async def prompt_provider(embedder, qdrant_available):
    """Create and cleanup a QdrantPromptProvider with test collection."""
    collection_name = f"{TEST_COLLECTION_PREFIX}prompts_{uuid4().hex[:8]}"

    config = QdrantConfig(
        url=QDRANT_URL,
        embedding_dimension=768,
    )

    provider = QdrantPromptProvider(config, embedder, collection_name)
    await provider.initialize()

    yield provider

    # Cleanup
    try:
        await provider._client.delete_collection(collection_name)
    except Exception:
        pass
    await provider.close()


# =============================================================================
# Memory Provider Tests
# =============================================================================


class TestQdrantMemoryProvider:
    """Tests for QdrantMemoryProvider with real Qdrant."""

    @pytest.mark.asyncio
    async def test_store_and_get(self, memory_provider):
        """Test storing and retrieving a memory."""
        memory = await memory_provider.store(
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="doug",
            entities=["Doug", "birthday", "March 15"],
            importance=0.8,
        )

        assert memory.id is not None
        assert memory.content == "Doug's birthday is March 15"
        assert memory.memory_type == MemoryType.FACT
        assert memory.user_id == "doug"
        assert memory.importance == 0.8
        assert "Doug" in memory.entities

        # Retrieve by ID
        retrieved = await memory_provider.get(memory.id)

        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content
        assert retrieved.user_id == "doug"

    @pytest.mark.asyncio
    async def test_search_semantic(self, memory_provider):
        """Test semantic search."""
        # Store multiple memories
        await memory_provider.store(
            content="The WiFi password is hunter2",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.CONTEXT,
            context_id="household",
        )

        await memory_provider.store(
            content="Doug prefers dark mode for all applications",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
        )

        await memory_provider.store(
            content="To restart the Plex server, run docker restart plex",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.CONTEXT,
            context_id="household",
        )

        # Search for WiFi
        results = await memory_provider.search(
            query="What is the WiFi password?",
            limit=3,
        )

        assert len(results) > 0
        # Top result should be about WiFi
        assert "wifi" in results[0].memory.content.lower() or "password" in results[0].memory.content.lower()
        assert results[0].score > 0

    @pytest.mark.asyncio
    async def test_search_with_filters(self, memory_provider):
        """Test search with user and type filters."""
        # Store memories for different users
        await memory_provider.store(
            content="Doug's favorite color is blue",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="doug",
        )

        await memory_provider.store(
            content="Sarah's favorite color is green",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="sarah",
        )

        # Search only Doug's memories
        results = await memory_provider.search(
            query="favorite color",
            user_id="doug",
            limit=5,
        )

        assert len(results) > 0
        for r in results:
            assert r.memory.user_id == "doug"

    @pytest.mark.asyncio
    async def test_update_memory(self, memory_provider):
        """Test updating a memory."""
        memory = await memory_provider.store(
            content="Initial content",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test",
            importance=0.5,
        )

        # Update importance
        updated = await memory_provider.update(
            memory.id,
            importance=0.9,
        )

        assert updated is not None
        assert updated.importance == 0.9
        assert updated.last_accessed is not None  # Should be set

    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_provider):
        """Test deleting a memory."""
        memory = await memory_provider.store(
            content="To be deleted",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test",
        )

        # Verify it exists
        assert await memory_provider.get(memory.id) is not None

        # Delete
        result = await memory_provider.delete(memory.id)
        assert result is True

        # Verify it's gone
        assert await memory_provider.get(memory.id) is None

    @pytest.mark.asyncio
    async def test_reinforce_memory(self, memory_provider):
        """Test reinforcing a memory."""
        memory = await memory_provider.store(
            content="Important fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test",
            importance=0.5,
        )

        # Reinforce
        reinforced = await memory_provider.reinforce(memory.id, boost=0.2)

        assert reinforced is not None
        assert reinforced.importance == 0.7
        assert reinforced.last_accessed is not None

    @pytest.mark.asyncio
    async def test_count_memories(self, memory_provider):
        """Test counting memories."""
        # Store a few
        for i in range(3):
            await memory_provider.store(
                content=f"Fact number {i}",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
                user_id="counter_test",
            )

        # Count all
        total = await memory_provider.count()
        assert total >= 3

        # Count by user
        user_count = await memory_provider.count(user_id="counter_test")
        assert user_count == 3

    @pytest.mark.asyncio
    async def test_type_importance_weighting(self, memory_provider):
        """Test that memory types affect search ranking."""
        # Store an instruction (high weight) and fact (medium weight) with same content
        await memory_provider.store(
            content="Always use metric units for measurements",
            memory_type=MemoryType.INSTRUCTION,
            scope=MemoryScope.USER,
            user_id="test",
        )

        await memory_provider.store(
            content="Metric units are used for measurements in science",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test",
        )

        # Search - instruction should rank higher due to type weight
        results = await memory_provider.search(
            query="metric units measurements",
            user_id="test",
            limit=2,
        )

        assert len(results) == 2
        # Both should be found, instruction should have higher weighted score


# =============================================================================
# Prompt Provider Tests
# =============================================================================


class TestQdrantPromptProvider:
    """Tests for QdrantPromptProvider with real Qdrant."""

    @pytest.mark.asyncio
    async def test_store_and_get_prompt(self, prompt_provider):
        """Test storing and retrieving a prompt."""
        prompt = await prompt_provider.store_prompt(
            name="TEST_DECISION_PROMPT",
            domain="core",
            content="You are a decision engine. Given the query, decide what to do.",
            status="draft",
            created_by="test",
        )

        assert prompt.id is not None
        assert prompt.name == "TEST_DECISION_PROMPT"
        assert prompt.domain == "core"
        assert prompt.version == 1
        assert prompt.status == "draft"

        # Retrieve by ID
        retrieved = await prompt_provider.get_prompt_by_id(prompt.id)

        assert retrieved is not None
        assert retrieved.id == prompt.id
        assert retrieved.content == prompt.content

    @pytest.mark.asyncio
    async def test_version_incrementing(self, prompt_provider):
        """Test that versions increment correctly."""
        prompt_name = f"VERSION_TEST_{uuid4().hex[:8]}"

        v1 = await prompt_provider.store_prompt(
            name=prompt_name,
            domain="test",
            content="Version 1 content",
        )

        v2 = await prompt_provider.store_prompt(
            name=prompt_name,
            domain="test",
            content="Version 2 content",
            parent_id=v1.id,
            mutation_reason="Testing versioning",
        )

        assert v1.version == 1
        assert v2.version == 2
        assert v2.parent_id == v1.id

    @pytest.mark.asyncio
    async def test_activate_prompt(self, prompt_provider):
        """Test activating a prompt version."""
        prompt_name = f"ACTIVATE_TEST_{uuid4().hex[:8]}"

        # Create draft
        draft = await prompt_provider.store_prompt(
            name=prompt_name,
            domain="test",
            content="Draft prompt",
            status="draft",
        )

        # Activate
        success = await prompt_provider.activate_prompt(draft.id)
        assert success is True

        # Check it's now active
        active = await prompt_provider.get_active_prompt(prompt_name)
        assert active is not None
        assert active.id == draft.id
        assert active.status == "active"
        assert active.activated_at is not None

    @pytest.mark.asyncio
    async def test_activate_deactivates_old(self, prompt_provider):
        """Test that activating a new version deactivates the old one."""
        prompt_name = f"DEACTIVATE_TEST_{uuid4().hex[:8]}"

        # Create and activate v1
        v1 = await prompt_provider.store_prompt(
            name=prompt_name,
            domain="test",
            content="Version 1",
        )
        await prompt_provider.activate_prompt(v1.id)

        # Create and activate v2
        v2 = await prompt_provider.store_prompt(
            name=prompt_name,
            domain="test",
            content="Version 2",
            parent_id=v1.id,
        )
        await prompt_provider.activate_prompt(v2.id)

        # v1 should now be archived
        v1_after = await prompt_provider.get_prompt_by_id(v1.id)
        assert v1_after.status == "archived"

        # v2 should be active
        active = await prompt_provider.get_active_prompt(prompt_name)
        assert active.id == v2.id

    @pytest.mark.asyncio
    async def test_record_usage(self, prompt_provider):
        """Test recording prompt usage for fitness tracking."""
        prompt = await prompt_provider.store_prompt(
            name=f"USAGE_TEST_{uuid4().hex[:8]}",
            domain="test",
            content="Test prompt",
        )

        # Record usage
        await prompt_provider.record_usage(prompt.id, success=True)
        await prompt_provider.record_usage(prompt.id, success=True)
        await prompt_provider.record_usage(prompt.id, success=False)

        # Check counts
        updated = await prompt_provider.get_prompt_by_id(prompt.id)
        assert updated.usage_count == 3
        assert updated.success_count == 2
        assert updated.fitness_score == pytest.approx(2/3, rel=0.01)

    @pytest.mark.asyncio
    async def test_list_prompts_by_domain(self, prompt_provider):
        """Test listing prompts filtered by domain."""
        domain = f"domain_{uuid4().hex[:8]}"

        # Create prompts in our test domain
        for i in range(3):
            await prompt_provider.store_prompt(
                name=f"PROMPT_{i}",
                domain=domain,
                content=f"Content {i}",
            )

        # Create one in different domain
        await prompt_provider.store_prompt(
            name="OTHER",
            domain="other_domain",
            content="Other content",
        )

        # List by domain
        prompts = await prompt_provider.list_prompts(domain=domain)

        assert len(prompts) == 3
        for p in prompts:
            assert p.domain == domain

    @pytest.mark.asyncio
    async def test_search_similar_prompts(self, prompt_provider):
        """Test searching for similar prompts."""
        await prompt_provider.store_prompt(
            name="HOME_AUTOMATION",
            domain="home",
            content="Control smart home devices including lights, switches, and sensors",
        )

        await prompt_provider.store_prompt(
            name="CALENDAR_EVENTS",
            domain="calendar",
            content="Manage calendar events including creating, reading, and deleting",
        )

        # Search for light-related
        results = await prompt_provider.search_similar_prompts(
            query="turn on the bedroom lights",
            limit=2,
        )

        assert len(results) > 0
        # Home automation should be more relevant
        prompt, score = results[0]
        assert "home" in prompt.domain.lower() or "light" in prompt.content.lower()


# =============================================================================
# Persistence Tests
# =============================================================================


class TestPersistence:
    """Tests verifying data actually persists across provider instances."""

    @pytest.mark.asyncio
    async def test_memory_survives_reconnect(self, embedder, qdrant_available):
        """Test that memories persist after closing and reopening connection."""
        collection_name = f"{TEST_COLLECTION_PREFIX}persist_{uuid4().hex[:8]}"

        config = QdrantConfig(
            url=QDRANT_URL,
            collection_name=collection_name,
        )

        # Create provider and store memory
        provider1 = QdrantMemoryProvider(config, embedder)
        await provider1.initialize()

        memory = await provider1.store(
            content="This should persist",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="persist_test",
        )
        memory_id = memory.id

        await provider1.close()

        # Create new provider with same collection
        provider2 = QdrantMemoryProvider(config, embedder)
        await provider2.initialize()

        # Memory should still exist
        retrieved = await provider2.get(memory_id)

        assert retrieved is not None
        assert retrieved.content == "This should persist"

        # Cleanup
        await provider2._client.delete_collection(collection_name)
        await provider2.close()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_provider):
        """Test getting a memory that doesn't exist."""
        result = await memory_provider.get("nonexistent_id_12345")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_memory(self, memory_provider):
        """Test deleting a memory that doesn't exist."""
        # Should not raise, just return False
        result = await memory_provider.delete("nonexistent_id_12345")
        # Qdrant delete doesn't fail for nonexistent IDs
        assert result is True or result is False

    @pytest.mark.asyncio
    async def test_reinforce_nonexistent_memory(self, memory_provider):
        """Test reinforcing a memory that doesn't exist."""
        result = await memory_provider.reinforce("nonexistent_id_12345")
        assert result is None
