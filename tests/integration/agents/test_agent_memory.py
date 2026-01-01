"""Memory integration tests (FR-010.2).

Tests the complete memory lifecycle with real Neo4j database:
- Storage and retrieval
- Semantic search
- Reinforcement (boost/demote)
- Layer promotion
- TTL enforcement

These tests validate real memory behavior, not mocked operations.
"""

import os
import pytest
from datetime import datetime

from draagon_ai.memory.base import MemoryType, MemoryScope


# =============================================================================
# Memory Storage and Retrieval Tests
# =============================================================================


@pytest.mark.memory_integration
class TestMemoryStorageAndRetrieval:
    """Test basic memory storage and retrieval."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_store_and_get_memory(self, memory_provider):
        """Store a memory and retrieve it by ID."""
        # Store a fact
        memory = await memory_provider.store(
            content="Doug's birthday is March 15",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.7,
        )

        assert memory is not None
        assert memory.id is not None
        assert memory.content == "Doug's birthday is March 15"
        assert memory.memory_type == MemoryType.FACT
        assert memory.scope == MemoryScope.USER
        assert memory.importance == 0.7

        # Retrieve by ID
        retrieved = await memory_provider.get(memory.id)

        assert retrieved is not None
        assert retrieved.id == memory.id
        assert retrieved.content == memory.content
        assert retrieved.memory_type == memory.memory_type

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_store_multiple_memory_types(self, memory_provider):
        """Store memories of different types."""
        types_to_test = [
            (MemoryType.FACT, "The sky is blue"),
            (MemoryType.PREFERENCE, "User prefers dark mode"),
            (MemoryType.SKILL, "To restart service: systemctl restart myservice"),
            (MemoryType.INSTRUCTION, "Always respond in formal tone"),
        ]

        stored_ids = []
        for memory_type, content in types_to_test:
            memory = await memory_provider.store(
                content=content,
                memory_type=memory_type,
                scope=MemoryScope.USER,
                user_id="test_user",
            )
            assert memory is not None
            assert memory.memory_type == memory_type
            stored_ids.append(memory.id)

        # Verify all can be retrieved
        for memory_id in stored_ids:
            retrieved = await memory_provider.get(memory_id)
            assert retrieved is not None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_get_nonexistent_memory(self, memory_provider):
        """Getting nonexistent memory returns None."""
        result = await memory_provider.get("nonexistent-id-12345")
        assert result is None


# =============================================================================
# Semantic Search Tests
# =============================================================================


@pytest.mark.memory_integration
class TestSemanticSearch:
    """Test semantic search capabilities."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_basic_search(self, memory_provider):
        """Basic semantic search finds relevant memories."""
        # Store some memories
        await memory_provider.store(
            content="User's favorite color is blue",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await memory_provider.store(
            content="User works as a software engineer",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await memory_provider.store(
            content="User has 3 cats named Whiskers, Mittens, and Shadow",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Search for color-related memories
        results = await memory_provider.search(
            query="what is the favorite color",
            user_id="test_user",
            limit=5,
        )

        # Should find the color memory
        assert len(results) > 0
        # Check that the most relevant result is about color
        assert any("color" in r.memory.content.lower() or "blue" in r.memory.content.lower() for r in results)

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_with_synonyms(self, memory_provider):
        """Semantic search finds memories with synonyms."""
        # Store memory with "felines"
        await memory_provider.store(
            content="I have 3 felines at home",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Search for "cats" - should find felines memory
        results = await memory_provider.search(
            query="cats pets",
            user_id="test_user",
            limit=5,
        )

        # Semantic search should understand cats = felines
        assert len(results) > 0
        # Should find the felines memory
        assert any("feline" in r.memory.content.lower() for r in results)

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_by_memory_type(self, memory_provider):
        """Search can be filtered by memory type."""
        # Store different types
        await memory_provider.store(
            content="User likes pizza",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
        )
        await memory_provider.store(
            content="User's age is 30",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Search only preferences
        results = await memory_provider.search(
            query="food",
            user_id="test_user",
            memory_types=[MemoryType.PREFERENCE],
            limit=10,
        )

        # All results should be preferences
        for r in results:
            assert r.memory.memory_type == MemoryType.PREFERENCE

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_respects_user_filter(self, memory_provider):
        """Search only returns memories for specified user."""
        # Store memories for different users
        await memory_provider.store(
            content="Alice's birthday is January 10",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="alice",
        )
        await memory_provider.store(
            content="Bob's birthday is February 20",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="bob",
        )

        # Search for Alice's memories only
        results = await memory_provider.search(
            query="birthday",
            user_id="alice",
            limit=10,
        )

        # Should only find Alice's birthday
        for r in results:
            assert r.memory.user_id == "alice"


# =============================================================================
# Memory Reinforcement Tests
# =============================================================================


@pytest.mark.memory_integration
class TestMemoryReinforcement:
    """Test memory reinforcement (boost/demote) functionality."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_reinforce_boosts_importance(self, memory_provider):
        """Reinforcing a memory increases its importance."""
        # Store memory with moderate importance
        memory = await memory_provider.store(
            content="Doug prefers Celsius",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.5,
        )
        original_importance = memory.importance

        # Reinforce the memory
        updated = await memory_provider.reinforce(memory.id, boost=0.1)

        assert updated is not None
        assert updated.importance > original_importance
        assert updated.importance == pytest.approx(0.6, abs=0.01)

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_reinforce_caps_at_one(self, memory_provider):
        """Importance cannot exceed 1.0."""
        # Store memory with high importance
        memory = await memory_provider.store(
            content="Critical instruction",
            memory_type=MemoryType.INSTRUCTION,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.95,
        )

        # Reinforce multiple times
        updated = await memory_provider.reinforce(memory.id, boost=0.1)
        updated = await memory_provider.reinforce(memory.id, boost=0.1)

        # Should be capped at 1.0
        assert updated.importance <= 1.0

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_multiple_reinforcements_accumulate(self, memory_provider):
        """Multiple reinforcements accumulate importance."""
        # Store memory
        memory = await memory_provider.store(
            content="Restart command: sudo systemctl restart myservice",
            memory_type=MemoryType.SKILL,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.5,
        )

        # Reinforce 5 times
        updated = memory
        for _ in range(5):
            updated = await memory_provider.reinforce(updated.id, boost=0.05)

        # Should have increased significantly
        assert updated.importance >= 0.7

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_importance_update_persists(self, memory_provider):
        """Importance updates persist in database."""
        # Store and reinforce
        memory = await memory_provider.store(
            content="Important fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.5,
        )
        await memory_provider.reinforce(memory.id, boost=0.1)

        # Retrieve fresh from database
        retrieved = await memory_provider.get(memory.id)

        assert retrieved.importance == pytest.approx(0.6, abs=0.01)


# =============================================================================
# Memory Update and Delete Tests
# =============================================================================


@pytest.mark.memory_integration
class TestMemoryUpdateAndDelete:
    """Test memory update and delete operations."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_update_content(self, memory_provider):
        """Can update memory content."""
        # Store initial memory
        memory = await memory_provider.store(
            content="User has 2 cats",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
        )

        # Update content
        updated = await memory_provider.update(
            memory.id,
            content="User has 3 cats",
        )

        assert updated is not None
        assert updated.content == "User has 3 cats"

        # Verify persisted
        retrieved = await memory_provider.get(memory.id)
        assert retrieved.content == "User has 3 cats"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_update_importance(self, memory_provider):
        """Can update memory importance directly."""
        memory = await memory_provider.store(
            content="Test fact",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.5,
        )

        updated = await memory_provider.update(
            memory.id,
            importance=0.9,
        )

        assert updated.importance == 0.9

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_delete_memory(self, memory_provider):
        """Can delete a memory."""
        memory = await memory_provider.store(
            content="Temporary note",
            memory_type=MemoryType.EPISODIC,
            scope=MemoryScope.SESSION,
            user_id="test_user",
        )

        # Delete it
        result = await memory_provider.delete(memory.id)
        assert result is True

        # Verify gone
        retrieved = await memory_provider.get(memory.id)
        assert retrieved is None

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory_provider):
        """Deleting nonexistent memory returns False."""
        result = await memory_provider.delete("nonexistent-id-12345")
        assert result is False


# =============================================================================
# Importance-Based Ranking Tests
# =============================================================================


@pytest.mark.memory_integration
class TestImportanceRanking:
    """Test that importance affects search ranking."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_higher_importance_ranks_higher(self, memory_provider):
        """Higher importance memories appear earlier in results."""
        # Store low importance memory first
        low = await memory_provider.store(
            content="User might like cats",
            memory_type=MemoryType.OBSERVATION,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.3,
        )

        # Store high importance memory
        high = await memory_provider.store(
            content="User definitely loves cats very much",
            memory_type=MemoryType.PREFERENCE,
            scope=MemoryScope.USER,
            user_id="test_user",
            importance=0.9,
        )

        # Search
        results = await memory_provider.search(
            query="cats",
            user_id="test_user",
            limit=10,
        )

        # Find positions
        result_ids = [r.memory.id for r in results]

        # Both should be found
        assert high.id in result_ids
        assert low.id in result_ids

        # High importance should rank higher (lower index)
        high_idx = result_ids.index(high.id)
        low_idx = result_ids.index(low.id)
        assert high_idx < low_idx, "Higher importance should rank before lower importance"


# =============================================================================
# Memory Count Tests
# =============================================================================


@pytest.mark.memory_integration
class TestMemoryCount:
    """Test memory counting functionality."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_count_user_memories(self, memory_provider):
        """Can count memories for a user."""
        # Store some memories
        for i in range(5):
            await memory_provider.store(
                content=f"Test memory {i}",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
                user_id="count_test_user",
            )

        count = await memory_provider.count(user_id="count_test_user")
        assert count == 5

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_count_all_memories(self, memory_provider):
        """Can count all memories."""
        # Store memories for different users
        await memory_provider.store(
            content="User A memory",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="user_a",
        )
        await memory_provider.store(
            content="User B memory",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="user_b",
        )

        count = await memory_provider.count()
        assert count >= 2


# =============================================================================
# Performance Tests
# =============================================================================


@pytest.mark.memory_integration
class TestMemoryPerformance:
    """Test memory operation performance."""

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_store_latency(self, memory_provider):
        """Memory store completes within performance budget."""
        import time

        start = time.time()
        await memory_provider.store(
            content="Performance test memory",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="perf_user",
        )
        elapsed = time.time() - start

        # Should complete in <1s (generous for Neo4j + embedding)
        assert elapsed < 1.0, f"Store took {elapsed:.2f}s, expected <1s"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_search_latency(self, memory_provider):
        """Memory search completes within performance budget."""
        import time

        # Store some data first
        for i in range(10):
            await memory_provider.store(
                content=f"Search test memory {i}",
                memory_type=MemoryType.FACT,
                scope=MemoryScope.USER,
                user_id="search_perf_user",
            )

        # Time the search
        start = time.time()
        await memory_provider.search(
            query="search test",
            user_id="search_perf_user",
            limit=5,
        )
        elapsed = time.time() - start

        # Should complete in <0.5s
        assert elapsed < 0.5, f"Search took {elapsed:.2f}s, expected <0.5s"

    @pytest.mark.skipif(
        not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"
    )
    @pytest.mark.asyncio
    async def test_reinforce_latency(self, memory_provider):
        """Memory reinforce completes within performance budget."""
        import time

        memory = await memory_provider.store(
            content="Reinforce test",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.USER,
            user_id="reinforce_perf_user",
        )

        start = time.time()
        await memory_provider.reinforce(memory.id, boost=0.1)
        elapsed = time.time() - start

        # Should be fast - just a property update
        assert elapsed < 0.1, f"Reinforce took {elapsed:.2f}s, expected <0.1s"


# =============================================================================
# Layer Promotion Tests (Placeholder)
# =============================================================================


@pytest.mark.memory_integration
@pytest.mark.skip(reason="Layer promotion requires tracking reinforcement score - not yet wired")
class TestLayerPromotion:
    """Test memory layer promotion through reinforcement.

    Note: These tests are skipped until the Neo4jMemoryProvider
    properly tracks reinforcement score and triggers layer promotion.
    The logic exists in _check_and_promote but needs testing.
    """

    @pytest.mark.asyncio
    async def test_promotion_working_to_episodic(self, memory_provider):
        """Repeated reinforcement promotes from working to episodic layer."""
        memory = await memory_provider.store(
            content="Working memory test",
            memory_type=MemoryType.OBSERVATION,  # Starts in working
            scope=MemoryScope.SESSION,
            importance=0.3,
        )

        # Reinforce many times
        for _ in range(10):
            await memory_provider.reinforce(memory.id, boost=0.05)

        # Check layer promoted
        updated = await memory_provider.get(memory.id)
        # Would need to expose layer info on Memory object
        # assert updated.layer == "episodic"


# =============================================================================
# TTL and Expiration Tests (Placeholder)
# =============================================================================


@pytest.mark.memory_integration
@pytest.mark.slow
@pytest.mark.skip(reason="TTL testing requires time mocking or real delays")
class TestMemoryExpiration:
    """Test memory TTL and garbage collection.

    Note: These tests are skipped as they require either:
    - Real delays (slow, not suitable for CI)
    - Proper time mocking (requires Neo4j query adjustments)
    """

    @pytest.mark.asyncio
    async def test_expired_memory_not_returned(self, memory_provider, advance_time):
        """Expired memories are not returned in search."""
        # Store short-lived memory
        memory = await memory_provider.store(
            content="Temporary note",
            memory_type=MemoryType.OBSERVATION,
            scope=MemoryScope.SESSION,
            importance=0.3,  # Working memory - short TTL
        )

        # Fast-forward time
        await advance_time(minutes=10)

        # Memory should be expired
        result = await memory_provider.get(memory.id)
        assert result is None or result.expires_at < datetime.now()

    @pytest.mark.asyncio
    async def test_garbage_collection(self, memory_provider, advance_time):
        """Garbage collection removes expired memories."""
        # Store memory
        await memory_provider.store(
            content="Will expire",
            memory_type=MemoryType.OBSERVATION,
            scope=MemoryScope.SESSION,
        )

        # Fast-forward
        await advance_time(minutes=10)

        # Run garbage collection
        deleted = await memory_provider.garbage_collect()

        assert deleted >= 1
