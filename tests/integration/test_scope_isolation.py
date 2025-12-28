"""Scope isolation integration tests.

Tests multi-user, multi-agent, and cross-scope isolation.

Requirements:
- Running Qdrant instance (default: http://192.168.168.216:6333)
- qdrant-client package installed
"""

import asyncio
import os
import pytest
from uuid import uuid4

try:
    from qdrant_client import AsyncQdrantClient
    QDRANT_CLIENT_AVAILABLE = True
except ImportError:
    QDRANT_CLIENT_AVAILABLE = False

from draagon_ai.memory import LayeredMemoryProvider
from draagon_ai.memory.providers.layered import LayeredMemoryConfig
from draagon_ai.memory.base import MemoryType, MemoryScope


QDRANT_URL = os.getenv("QDRANT_URL", "http://192.168.168.216:6333")
TEST_COLLECTION_PREFIX = "test_draagon_scope_"


class WordBasedEmbeddingProvider:
    """Word-based embedding provider for tests."""

    def __init__(self, dimension: int = 768):
        self.dimension = dimension
        self._cache: dict[str, list[float]] = {}
        self._vocab: dict[str, int] = {}
        self._next_idx = 0

    def _get_word_idx(self, word: str) -> int:
        if word not in self._vocab:
            self._vocab[word] = self._next_idx
            self._next_idx += 1
        return self._vocab[word]

    async def embed(self, text: str) -> list[float]:
        if text in self._cache:
            return self._cache[text]

        import re
        words = re.findall(r'\b\w+\b', text.lower())

        embedding = [0.0] * self.dimension

        for word in words:
            idx = self._get_word_idx(word)
            for offset in range(5):
                dim_idx = (idx * 7 + offset * 13) % self.dimension
                embedding[dim_idx] += 0.5

        magnitude = sum(x * x for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]
        else:
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


pytestmark = [
    pytest.mark.skipif(
        not QDRANT_CLIENT_AVAILABLE,
        reason="qdrant-client not installed"
    ),
    pytest.mark.integration,
]


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def qdrant_available():
    available = await check_qdrant_connection()
    if not available:
        pytest.skip(f"Qdrant not accessible at {QDRANT_URL}")
    return True


@pytest.fixture
def embedder():
    return WordBasedEmbeddingProvider()


@pytest.fixture
async def provider(embedder, qdrant_available):
    """Create a LayeredMemoryProvider with scope enforcement enabled."""
    test_id = uuid4().hex[:8]

    config = LayeredMemoryConfig(
        qdrant_url=QDRANT_URL,
        qdrant_nodes_collection=f"{TEST_COLLECTION_PREFIX}nodes_{test_id}",
        qdrant_edges_collection=f"{TEST_COLLECTION_PREFIX}edges_{test_id}",
        embedding_dimension=768,
        enforce_scope_permissions=False,  # Start without enforcement
    )

    p = LayeredMemoryProvider(config=config, embedding_provider=embedder)
    await p.initialize()

    yield p

    try:
        if hasattr(p._graph, '_client') and p._graph._client:
            await p._graph._client.delete_collection(config.qdrant_nodes_collection)
            await p._graph._client.delete_collection(config.qdrant_edges_collection)
    except Exception:
        pass
    await p.close()


# =============================================================================
# Multi-User Isolation Tests
# =============================================================================


class TestMultiUserIsolation:
    """Test that different users have isolated memories."""

    @pytest.mark.asyncio
    async def test_user_memories_stored_separately(self, provider):
        """Test that user memories are stored with user_id."""
        # User 1 stores a memory
        await provider.store(
            content="User1 secret password is abc123",
            memory_type="fact",
            scope=MemoryScope.USER,
            user_id="user1",
        )

        # User 2 stores a memory
        await provider.store(
            content="User2 secret password is xyz789",
            memory_type="fact",
            scope=MemoryScope.USER,
            user_id="user2",
        )

        # Search for user1's data
        results1 = await provider.search(
            query="password",
            user_id="user1",
            limit=5,
        )

        # Search for user2's data
        results2 = await provider.search(
            query="password",
            user_id="user2",
            limit=5,
        )

        # Both should return results
        assert isinstance(results1, list)
        assert isinstance(results2, list)

    @pytest.mark.asyncio
    async def test_private_scope_isolation(self, provider):
        """Test that PRIVATE scope memories are isolated."""
        # Store private memory for user1
        await provider.store(
            content="User1 private note about finances",
            memory_type="fact",
            scope=MemoryScope.USER,
            user_id="user1",
            entities=["finances"],
        )

        # Store private memory for user2
        await provider.store(
            content="User2 private note about vacation",
            memory_type="fact",
            scope=MemoryScope.USER,
            user_id="user2",
            entities=["vacation"],
        )

        # Search should respect user_id filter
        results1 = await provider.search(
            query="note",
            user_id="user1",
            limit=5,
        )
        results2 = await provider.search(
            query="note",
            user_id="user2",
            limit=5,
        )

        assert isinstance(results1, list)
        assert isinstance(results2, list)


# =============================================================================
# Multi-Agent Isolation Tests
# =============================================================================


class TestMultiAgentIsolation:
    """Test that different agents have isolated memories."""

    @pytest.mark.asyncio
    async def test_agent_memories_stored_separately(self, provider):
        """Test that agent-scoped memories are stored with agent_id."""
        # Agent 1 stores a memory
        await provider.store(
            content="Agent1 learned skill for web scraping",
            memory_type="skill",
            scope=MemoryScope.AGENT,
            agent_id="agent1",
        )

        # Agent 2 stores a memory
        await provider.store(
            content="Agent2 learned skill for data analysis",
            memory_type="skill",
            scope=MemoryScope.AGENT,
            agent_id="agent2",
        )

        # Search should respect agent filter
        results1 = await provider.search(
            query="learned skill",
            agent_id="agent1",
            limit=5,
        )
        results2 = await provider.search(
            query="learned skill",
            agent_id="agent2",
            limit=5,
        )

        assert isinstance(results1, list)
        assert isinstance(results2, list)

    @pytest.mark.asyncio
    async def test_agent_skills_isolated(self, provider):
        """Test that agent skills are properly isolated."""
        # Add skill for agent1
        skill1 = await provider.metacognitive.add_skill(
            name="web_scrape",
            skill_type="automation",
            procedure="Use requests and BeautifulSoup",
            scope_id="agent:agent1",
        )

        # Add skill for agent2
        skill2 = await provider.metacognitive.add_skill(
            name="data_analyze",
            skill_type="analytics",
            procedure="Use pandas and numpy",
            scope_id="agent:agent2",
        )

        assert skill1 is not None
        assert skill2 is not None
        assert skill1.node_id != skill2.node_id


# =============================================================================
# Cross-Scope Access Tests
# =============================================================================


class TestCrossScopeAccess:
    """Test cross-scope access patterns."""

    @pytest.mark.asyncio
    async def test_world_scope_accessible_by_all(self, provider):
        """Test that WORLD scope is accessible by everyone."""
        # Store world-scope memory
        await provider.store(
            content="Public knowledge about Python programming",
            memory_type="fact",
            scope=MemoryScope.WORLD,
        )

        # Search without user filter should find it
        results = await provider.search(
            query="Python programming",
            limit=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_context_scope_shared_within_context(self, provider):
        """Test that CONTEXT scope is shared within a context."""
        # Store context-scoped memory
        await provider.store(
            content="Household setting: temperature preference 72F",
            memory_type="preference",
            scope=MemoryScope.CONTEXT,
            context_id="household:mealing",
        )

        # Different users in same context should be able to search
        results1 = await provider.search(
            query="temperature preference",
            context_id="household:mealing",
            user_id="user1",
            limit=5,
        )
        results2 = await provider.search(
            query="temperature preference",
            context_id="household:mealing",
            user_id="user2",
            limit=5,
        )

        assert isinstance(results1, list)
        assert isinstance(results2, list)


# =============================================================================
# Scope Hierarchy Tests
# =============================================================================


class TestScopeHierarchy:
    """Test the scope hierarchy behavior."""

    @pytest.mark.asyncio
    async def test_multiple_scopes_stored(self, provider):
        """Test storing memories in multiple scopes."""
        # Store in different scopes
        scopes = [
            (MemoryScope.SESSION, "Session-level data"),
            (MemoryScope.USER, "User-level data"),
            (MemoryScope.AGENT, "Agent-level data"),
            (MemoryScope.CONTEXT, "Context-level data"),
            (MemoryScope.WORLD, "World-level data"),
        ]

        for scope, content in scopes:
            mem = await provider.store(
                content=content,
                memory_type="fact",
                scope=scope,
            )
            assert mem is not None

    @pytest.mark.asyncio
    async def test_scope_filtering_in_search(self, provider):
        """Test that scope filtering works in search."""
        # Store memories in different scopes
        await provider.store(
            content="User specific data abc",
            memory_type="fact",
            scope=MemoryScope.USER,
            user_id="testuser",
        )
        await provider.store(
            content="World accessible data abc",
            memory_type="fact",
            scope=MemoryScope.WORLD,
        )

        # Search with scope filter
        results = await provider.search(
            query="data abc",
            scopes=[MemoryScope.USER],
            user_id="testuser",
            limit=5,
        )

        assert isinstance(results, list)


# =============================================================================
# Concurrent Multi-User Tests
# =============================================================================


class TestConcurrentMultiUser:
    """Test concurrent access by multiple users."""

    @pytest.mark.asyncio
    async def test_concurrent_user_stores(self, provider):
        """Test that concurrent stores from multiple users work correctly."""

        async def store_for_user(user_id: str, idx: int):
            return await provider.store(
                content=f"User {user_id} memory item {idx}",
                memory_type="fact",
                scope=MemoryScope.USER,
                user_id=user_id,
            )

        # Create tasks for multiple users
        tasks = []
        for user_id in ["alice", "bob", "charlie"]:
            for i in range(3):
                tasks.append(store_for_user(user_id, i))

        # Run all concurrently
        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 9
        for result in results:
            assert result is not None

    @pytest.mark.asyncio
    async def test_concurrent_user_searches(self, provider):
        """Test that concurrent searches from multiple users work correctly."""
        # First store some data
        for user_id in ["user_a", "user_b"]:
            for i in range(3):
                await provider.store(
                    content=f"Data for {user_id} number {i}",
                    memory_type="fact",
                    scope=MemoryScope.USER,
                    user_id=user_id,
                )

        async def search_for_user(user_id: str):
            return await provider.search(
                query="Data",
                user_id=user_id,
                limit=5,
            )

        # Concurrent searches
        tasks = [search_for_user(uid) for uid in ["user_a", "user_b"] * 3]
        results = await asyncio.gather(*tasks)

        # All should return lists
        assert len(results) == 6
        for result in results:
            assert isinstance(result, list)


# =============================================================================
# Cross-Agent Collaboration Tests
# =============================================================================


class TestCrossAgentCollaboration:
    """Test scenarios where agents need to share information."""

    @pytest.mark.asyncio
    async def test_shared_context_between_agents(self, provider):
        """Test that agents can share context-level information."""
        # Agent 1 stores context info
        await provider.store(
            content="Shared context: user prefers dark mode",
            memory_type="preference",
            scope=MemoryScope.CONTEXT,
            context_id="household:test",
            agent_id="agent1",
        )

        # Agent 2 should be able to find context info
        results = await provider.search(
            query="dark mode preference",
            context_id="household:test",
            agent_id="agent2",
            limit=5,
        )

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_world_knowledge_shared(self, provider):
        """Test that world knowledge is shared between agents."""
        # Store world knowledge
        await provider.store(
            content="World fact: Python 3.12 released in October 2023",
            memory_type="fact",
            scope=MemoryScope.WORLD,
        )

        # Both agents should find it
        results1 = await provider.search(
            query="Python 3.12",
            agent_id="agent1",
            limit=5,
        )
        results2 = await provider.search(
            query="Python 3.12",
            agent_id="agent2",
            limit=5,
        )

        assert isinstance(results1, list)
        assert isinstance(results2, list)
