"""Tests for HybridRetriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from draagon_ai.retrieval import (
    HybridRetriever,
    RetrievalResult,
    RetrievalConfig,
    CRAGGrade,
)


class MockMemoryProvider:
    """Mock memory provider for testing."""

    def __init__(self, results: list[dict] | None = None):
        self.results = results or []
        self.search_calls = []

    async def search(
        self,
        query: str,
        user_id: str,
        limit: int = 10,
        min_score: float = 0.0,
        memory_types: list | None = None,
        context_id: str | None = None,
        **kwargs,
    ) -> list[dict]:
        self.search_calls.append({
            "query": query,
            "user_id": user_id,
            "limit": limit,
            "min_score": min_score,
        })
        return self.results[:limit]


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "expanded query"):
        self.response = response
        self.chat_calls = []

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int = 500,
        temperature: float = 0.0,
        **kwargs,
    ) -> dict:
        self.chat_calls.append({"messages": messages})
        return {"content": self.response}


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "content": "The WiFi password is hunter2",
            "score": 0.8,
            "memory_type": "fact",
            "importance": 0.9,
            "scope": "private",
        },
        {
            "id": "doc2",
            "content": "Doug's birthday is March 15",
            "score": 0.7,
            "memory_type": "fact",
            "importance": 0.8,
            "scope": "private",
        },
        {
            "id": "doc3",
            "content": "How to restart the server: systemctl restart app",
            "score": 0.6,
            "memory_type": "skill",
            "importance": 0.7,
            "scope": "shared",
        },
        {
            "id": "doc4",
            "content": "Python programming documentation",
            "score": 0.4,
            "memory_type": "knowledge",
            "importance": 0.5,
            "scope": "system",
        },
        {
            "id": "doc5",
            "content": "Random unrelated content",
            "score": 0.2,
            "memory_type": "episodic",
            "importance": 0.3,
            "scope": "private",
        },
    ]


class TestHybridRetriever:
    """Tests for HybridRetriever class."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_result(self, sample_documents):
        """Test basic retrieval returns RetrievalResult."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.retrieve(
            query="WiFi password",
            user_id="test_user",
            k=3,
        )

        assert isinstance(result, RetrievalResult)
        assert len(result.documents) <= 3
        assert result.candidates_count == len(sample_documents)

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self):
        """Test retrieval with no results."""
        memory = MockMemoryProvider([])
        retriever = HybridRetriever(memory)

        result = await retriever.retrieve(
            query="nonexistent",
            user_id="test_user",
        )

        assert result.documents == []
        assert result.relevance_score == 0.0
        assert not result.sufficient
        assert result.strategy_used == "none"

    @pytest.mark.asyncio
    async def test_reranking_boosts_relevant(self, sample_documents):
        """Test that re-ranking boosts relevant documents."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.retrieve(
            query="WiFi password",
            user_id="test_user",
            k=5,
        )

        # Document with "WiFi password" should be boosted to top
        assert result.documents[0]["content"] == "The WiFi password is hunter2"

    @pytest.mark.asyncio
    async def test_crag_grading(self, sample_documents):
        """Test CRAG-style grading."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.retrieve(
            query="test query",
            user_id="test_user",
        )

        assert "relevant" in result.grading
        assert "ambiguous" in result.grading
        assert "irrelevant" in result.grading

        total = sum(result.grading.values())
        assert total == len(sample_documents)

    @pytest.mark.asyncio
    async def test_query_expansion_on_low_quality(self, sample_documents):
        """Test query expansion when quality is low."""
        # Documents with low scores
        low_score_docs = [
            {"id": "1", "content": "text", "score": 0.3, "memory_type": "fact"},
        ]
        memory = MockMemoryProvider(low_score_docs)
        llm = MockLLMProvider("expanded query about WiFi")
        config = RetrievalConfig(enable_query_expansion=True)
        retriever = HybridRetriever(memory, llm, config)

        result = await retriever.retrieve(
            query="wifi",
            user_id="test_user",
            min_relevance=0.9,  # High threshold to trigger expansion
        )

        # Should have tried expansion
        assert len(llm.chat_calls) == 1
        assert result.strategy_used in ("standard", "expanded")

    @pytest.mark.asyncio
    async def test_memory_type_boosts(self, sample_documents):
        """Test that memory type boosts are applied."""
        memory = MockMemoryProvider(sample_documents)
        config = RetrievalConfig(
            memory_type_boosts={"knowledge": 2.0}  # Strong boost for knowledge
        )
        retriever = HybridRetriever(memory, config=config)

        result = await retriever.retrieve(
            query="programming",
            user_id="test_user",
        )

        # Knowledge document should get boosted
        knowledge_doc = next(
            (d for d in result.documents if d["memory_type"] == "knowledge"),
            None,
        )
        if knowledge_doc:
            assert knowledge_doc.get("boosted_score", 0) > knowledge_doc.get("score", 0)

    @pytest.mark.asyncio
    async def test_keyword_matching_boost(self, sample_documents):
        """Test keyword matching boost."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.retrieve(
            query="restart server",
            user_id="test_user",
        )

        # Document with "restart" and "server" should get keyword boost
        skill_doc = next(
            (d for d in result.documents if "restart" in d["content"].lower()),
            None,
        )
        assert skill_doc is not None
        assert skill_doc.get("boosted_score", 0) > skill_doc.get("score", 0)

    @pytest.mark.asyncio
    async def test_search_with_self_rag(self, sample_documents):
        """Test self-RAG search filters by relevance."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.search_with_self_rag(
            query="WiFi password",
            user_id="test_user",
            relevance_threshold=0.5,
        )

        # Should only include high-relevance documents
        for doc in result.documents:
            score = doc.get("boosted_score", doc.get("score", 0))
            assert score >= 0.5

    @pytest.mark.asyncio
    async def test_search_cross_chat(self, sample_documents):
        """Test cross-chat search."""
        memory = MockMemoryProvider(sample_documents)
        retriever = HybridRetriever(memory)

        result = await retriever.search_cross_chat(
            query="test query",
            user_id="test_user",
            limit=3,
        )

        assert "results" in result
        assert "count" in result
        assert "contexts" in result
        assert isinstance(result["contexts"], list)


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_has_relevant(self):
        """Test has_relevant property."""
        result = RetrievalResult(
            documents=[],
            relevance_score=0.5,
            sufficient=True,
            strategy_used="standard",
            candidates_count=10,
            grading={"relevant": 3, "ambiguous": 2, "irrelevant": 5},
        )
        assert result.has_relevant is True

        result.grading = {"relevant": 0, "ambiguous": 2, "irrelevant": 5}
        assert result.has_relevant is False

    def test_mostly_irrelevant(self):
        """Test mostly_irrelevant property."""
        result = RetrievalResult(
            documents=[],
            relevance_score=0.3,
            sufficient=False,
            strategy_used="standard",
            candidates_count=10,
            grading={"relevant": 1, "ambiguous": 1, "irrelevant": 8},
        )
        assert result.mostly_irrelevant is True

        result.grading = {"relevant": 5, "ambiguous": 3, "irrelevant": 2}
        assert result.mostly_irrelevant is False


class TestRetrievalConfig:
    """Tests for RetrievalConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.recall_multiplier == 10
        assert config.max_recall == 100
        assert config.min_relevance == 0.5
        assert config.relevant_threshold == 0.55
        assert config.ambiguous_threshold == 0.4
        assert config.enable_query_expansion is True
        assert "knowledge" in config.memory_type_boosts
        assert "skill" in config.memory_type_boosts

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetrievalConfig(
            recall_multiplier=5,
            min_relevance=0.7,
            enable_query_expansion=False,
        )

        assert config.recall_multiplier == 5
        assert config.min_relevance == 0.7
        assert config.enable_query_expansion is False
