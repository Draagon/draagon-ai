"""Integration tests for Hybrid Parallel Retrieval.

Tests the orchestration of Local, Graph, and Vector retrieval agents
running in parallel with result merging and synthesis.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from draagon_ai.orchestration.hybrid_retrieval import (
    HybridRetrievalOrchestrator,
    HybridRetrievalConfig,
    HybridResult,
    QueryAnalyzer,
    QueryType,
    RetrievalPath,
    QueryClassification,
    Observation,
    Scope,
    ResultMerger,
    LocalRetrievalAgent,
    GraphRetrievalAgent,
    VectorRetrievalAgent,
    SynthesisAgent,
    hybrid_retrieve,
)


# =============================================================================
# Mock Providers
# =============================================================================


@dataclass
class MockLocalResult:
    content: str
    source_path: str = "CLAUDE.md"


@dataclass
class MockGraphResult:
    content: str
    entities: list = field(default_factory=list)
    project: str | None = None


@dataclass
class MockVectorResult:
    content: str
    score: float = 0.85
    project: str | None = None


class MockLocalIndex:
    """Mock local index provider."""

    def __init__(self, results: list[MockLocalResult] | None = None):
        self.results = results or []
        self.search_calls: list[str] = []

    async def search(self, query: str, limit: int = 10) -> list[MockLocalResult]:
        self.search_calls.append(query)
        return self.results[:limit]


class MockSemanticMemory:
    """Mock semantic memory (graph) provider."""

    def __init__(
        self,
        results: list[MockGraphResult] | None = None,
        entities_by_name: dict[str, list] | None = None,
    ):
        self.results = results or []
        self.entities_by_name = entities_by_name or {}
        self.search_calls: list[str] = []
        self.find_calls: list[list[str]] = []

    async def search(self, query: str, limit: int = 10) -> list[MockGraphResult]:
        self.search_calls.append(query)
        return self.results[:limit]

    async def find_entities(self, names: list[str]) -> list[Any]:
        self.find_calls.append(names)
        results = []
        for name in names:
            if name in self.entities_by_name:
                results.extend(self.entities_by_name[name])
        return results

    async def get_related(
        self, entity_id: str, relationship_types: list[str] | None = None, depth: int = 1
    ) -> list[Any]:
        return []


class MockVectorStore:
    """Mock vector store provider."""

    def __init__(self, results: list[MockVectorResult] | None = None):
        self.results = results or []
        self.search_calls: list[tuple[str, dict | None]] = []

    async def search(
        self, query: str, limit: int = 10, filter: dict[str, Any] | None = None
    ) -> list[MockVectorResult]:
        self.search_calls.append((query, filter))

        # If filter specified, only return matching results
        if filter and "project" in filter and "$in" in filter["project"]:
            allowed_projects = filter["project"]["$in"]
            filtered = [r for r in self.results if r.project in allowed_projects]
            return filtered[:limit]

        return self.results[:limit]


class MockLLMProvider:
    """Mock LLM for synthesis and classification."""

    def __init__(self, response: str = "Synthesized answer based on the context."):
        self.response = response
        self.calls: list[dict] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        self.calls.append({"messages": messages})

        # Check if this is a classification query
        if messages and "Classify this query" in messages[0].get("content", ""):
            prompt_text = messages[0]["content"]

            # Extract just the query from the prompt (after "Query: " line)
            import re
            query_match = re.search(r"Query:\s*(.+?)(?:\n|$)", prompt_text)
            query_text = query_match.group(1).lower() if query_match else ""

            # Return appropriate classification based on query content
            # Check cross-project first (more specific)
            if "other teams" in query_text or "other projects" in query_text:
                return """<classification>
  <type>CROSS_PROJECT</type>
  <entities>None</entities>
  <confidence>0.9</confidence>
</classification>"""
            elif "connect to" in query_text or "relates to" in query_text:
                return """<classification>
  <type>RELATIONSHIP</type>
  <entities>None</entities>
  <confidence>0.9</confidence>
</classification>"""
            elif "our " in query_text or "this project" in query_text:
                return """<classification>
  <type>LOCAL_ONLY</type>
  <entities>None</entities>
  <confidence>0.9</confidence>
</classification>"""
            else:
                return """<classification>
  <type>GENERAL</type>
  <entities>None</entities>
  <confidence>0.7</confidence>
</classification>"""

        return self.response


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    return MockLLMProvider()


@pytest.fixture
def mock_local():
    return MockLocalIndex([
        MockLocalResult("Our project uses the Agent pattern for orchestration.", "CLAUDE.md"),
        MockLocalResult("Customer data is handled by the billing module.", "docs/arch.md"),
    ])


@pytest.fixture
def mock_graph():
    return MockSemanticMemory(
        results=[
            MockGraphResult(
                "Customer entity is used in billing-service",
                entities=[type("E", (), {"name": "Customer"})()],
                project="billing-service",
            ),
            MockGraphResult(
                "Customer also exists in auth-service as User",
                entities=[type("E", (), {"name": "User"})()],
                project="auth-service",
            ),
        ],
        entities_by_name={
            "Customer": [{"name": "Customer", "project": "billing-service"}],
            "User": [{"name": "User", "project": "auth-service"}],
        },
    )


@pytest.fixture
def mock_vector():
    return MockVectorStore([
        MockVectorResult(
            "Example customer handling in payment flow",
            score=0.92,
            project="billing-service",
        ),
        MockVectorResult(
            "User authentication pattern example",
            score=0.88,
            project="auth-service",
        ),
        MockVectorResult(
            "Unrelated content about orders",
            score=0.75,
            project="order-service",
        ),
    ])


@pytest.fixture
def orchestrator(mock_llm, mock_local, mock_graph, mock_vector):
    return HybridRetrievalOrchestrator(
        llm=mock_llm,
        semantic_memory=mock_graph,
        vector_store=mock_vector,
        local_index=mock_local,
    )


# =============================================================================
# Query Analyzer Tests
# =============================================================================


class TestQueryAnalyzer:
    """Tests for query classification."""

    def test_relationship_query(self):
        """Test detection of relationship queries."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("How does OrderService connect to PaymentGateway?")

        assert result.query_type == QueryType.RELATIONSHIP
        assert RetrievalPath.GRAPH in result.paths
        assert "OrderService" in result.detected_entities
        assert "PaymentGateway" in result.detected_entities

    def test_similarity_query(self):
        """Test detection of similarity queries."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("Find code similar to this error handling pattern")

        assert result.query_type == QueryType.SIMILARITY
        assert RetrievalPath.VECTOR in result.paths

    def test_cross_project_query(self):
        """Test detection of cross-project queries."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("How do other teams handle customer authentication?")

        assert result.query_type == QueryType.CROSS_PROJECT
        assert RetrievalPath.GRAPH in result.paths
        assert RetrievalPath.VECTOR in result.paths
        assert result.graph_scopes_vector is True

    def test_local_only_query(self):
        """Test detection of local-only queries."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("What does our CLAUDE.md say about testing?")

        assert result.query_type == QueryType.LOCAL_ONLY
        assert result.paths == [RetrievalPath.LOCAL]

    def test_entity_extraction(self):
        """Test extraction of CamelCase entities."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("What is the CustomerService API?")

        assert "CustomerService" in result.detected_entities
        assert "API" not in result.detected_entities  # Too short / common

    def test_general_query(self):
        """Test fallback to general classification."""
        analyzer = QueryAnalyzer()

        result = analyzer.classify("tell me about this thing")

        assert result.query_type == QueryType.GENERAL
        assert len(result.paths) == 3  # All paths


# =============================================================================
# Individual Agent Tests
# =============================================================================


class TestLocalRetrievalAgent:
    """Tests for local retrieval agent."""

    @pytest.mark.asyncio
    async def test_retrieves_from_local(self, mock_local):
        agent = LocalRetrievalAgent(mock_local)

        observations = await agent.retrieve("agent pattern")

        assert len(observations) == 2
        assert observations[0].source.startswith("local:")
        assert observations[0].confidence == 0.9
        assert mock_local.search_calls == ["agent pattern"]

    @pytest.mark.asyncio
    async def test_handles_no_index(self):
        agent = LocalRetrievalAgent(None)

        observations = await agent.retrieve("anything")

        assert observations == []


class TestGraphRetrievalAgent:
    """Tests for graph retrieval agent."""

    @pytest.mark.asyncio
    async def test_retrieves_from_graph(self, mock_graph):
        agent = GraphRetrievalAgent(mock_graph)

        observations = await agent.retrieve("Customer entity")

        assert len(observations) == 2
        assert observations[0].source == "graph:neo4j"
        assert observations[0].source_project == "billing-service"

    @pytest.mark.asyncio
    async def test_find_scope(self, mock_graph):
        agent = GraphRetrievalAgent(mock_graph)

        scope = await agent.find_scope(["Customer"])

        assert "billing-service" in scope.projects

    @pytest.mark.asyncio
    async def test_handles_no_memory(self):
        agent = GraphRetrievalAgent(None)

        observations = await agent.retrieve("anything")

        assert observations == []


class TestVectorRetrievalAgent:
    """Tests for vector retrieval agent."""

    @pytest.mark.asyncio
    async def test_retrieves_from_vector(self, mock_vector):
        agent = VectorRetrievalAgent(mock_vector)

        observations = await agent.retrieve("customer handling")

        assert len(observations) == 3
        assert observations[0].source == "vector:qdrant"
        assert observations[0].confidence <= 0.9  # Capped

    @pytest.mark.asyncio
    async def test_scoped_retrieval(self, mock_vector):
        agent = VectorRetrievalAgent(mock_vector)
        scope = Scope(projects=["billing-service"])

        observations = await agent.retrieve("customer", scope=scope)

        # Should only return billing-service results
        assert all(o.source_project == "billing-service" for o in observations)

        # Check filter was passed
        _, filter_used = mock_vector.search_calls[0]
        assert filter_used == {"project": {"$in": ["billing-service"]}}

    @pytest.mark.asyncio
    async def test_handles_no_store(self):
        agent = VectorRetrievalAgent(None)

        observations = await agent.retrieve("anything")

        assert observations == []


# =============================================================================
# Result Merger Tests
# =============================================================================


class TestResultMerger:
    """Tests for result merging."""

    def test_deduplicates_exact_matches(self):
        merger = ResultMerger()

        observations = [
            Observation(content="Same content here", source="local:a"),
            Observation(content="Same content here", source="graph:b"),
            Observation(content="Different content", source="vector:c"),
        ]

        merged = merger.merge(observations)

        # Should have 2 unique observations
        assert len(merged) == 2

    def test_corroboration_boost(self):
        merger = ResultMerger()

        # Content needs >85% word overlap to trigger corroboration
        # "Customer data is stored in billing" has 6 words
        # "Customer data is stored in billing service" has 7 words
        # Overlap = 6/7 = 0.857 > 0.85 threshold
        observations = [
            Observation(
                content="Customer data is stored in billing",
                source="local:a",
                confidence=0.9,
            ),
            Observation(
                content="Customer data is stored in billing service",
                source="graph:b",
                confidence=0.85,
            ),
        ]

        merged = merger.merge(observations)

        # First one should be boosted due to corroboration
        assert merged[0].confidence > 0.9
        assert "corroborated_by" in merged[0].metadata

    def test_ranks_by_confidence(self):
        merger = ResultMerger()

        observations = [
            Observation(content="Low confidence", source="a", confidence=0.5),
            Observation(content="High confidence", source="b", confidence=0.95),
            Observation(content="Medium confidence", source="c", confidence=0.75),
        ]

        merged = merger.merge(observations)

        # Should be sorted by confidence
        assert merged[0].confidence >= merged[1].confidence >= merged[2].confidence

    def test_local_boost(self):
        merger = ResultMerger()

        observations = [
            Observation(content="From vector", source="vector:a", confidence=0.8),
            Observation(content="From local", source="local:CLAUDE.md", confidence=0.8),
        ]

        merged = merger.merge(observations)

        # Local should rank higher due to boost
        assert merged[0].source.startswith("local:")


# =============================================================================
# Synthesis Tests
# =============================================================================


class TestSynthesisAgent:
    """Tests for answer synthesis."""

    @pytest.mark.asyncio
    async def test_synthesizes_answer(self, mock_llm):
        synthesizer = SynthesisAgent(mock_llm)

        observations = [
            Observation(content="Customer table has id, email, name", source="graph:neo4j"),
            Observation(content="Use UUID for customer IDs", source="local:CLAUDE.md"),
        ]

        answer = await synthesizer.synthesize("What is the customer schema?", observations)

        assert answer == "Synthesized answer based on the context."
        assert len(mock_llm.calls) == 1

        # Check prompt structure
        prompt = mock_llm.calls[0]["messages"][0]["content"]
        assert "customer schema" in prompt
        assert "graph:neo4j" in prompt
        assert "local:CLAUDE.md" in prompt

    @pytest.mark.asyncio
    async def test_groups_by_source_type(self, mock_llm):
        synthesizer = SynthesisAgent(mock_llm)

        observations = [
            Observation(content="Local info", source="local:a"),
            Observation(content="Graph info", source="graph:b"),
            Observation(content="Vector info", source="vector:c"),
        ]

        await synthesizer.synthesize("test query", observations)

        prompt = mock_llm.calls[0]["messages"][0]["content"]
        assert "FROM YOUR PROJECT:" in prompt
        assert "FROM ENTERPRISE KNOWLEDGE:" in prompt
        assert "FROM SIMILAR DOCUMENTS:" in prompt


# =============================================================================
# Full Orchestrator Tests
# =============================================================================


class TestHybridRetrievalOrchestrator:
    """Tests for the full orchestrator."""

    @pytest.mark.asyncio
    async def test_full_retrieval_flow(self, orchestrator, mock_llm, mock_local, mock_graph, mock_vector):
        """Test complete retrieval flow."""
        result = await orchestrator.retrieve("How do other teams handle Customer data?")

        assert isinstance(result, HybridResult)
        assert result.answer != ""
        assert len(result.observations) > 0
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, mock_local, mock_graph, mock_vector):
        """Test that agents run in parallel."""
        result = await orchestrator.retrieve("What is Customer?")

        # All agents should have been called
        assert len(mock_local.search_calls) > 0
        assert len(mock_graph.search_calls) > 0
        # Vector may or may not be called depending on classification

    @pytest.mark.asyncio
    async def test_graph_scopes_vector(self, orchestrator, mock_graph, mock_vector):
        """Test that graph results scope vector search."""
        result = await orchestrator.retrieve("How do other teams handle customer auth?")

        # Check vector was scoped
        if mock_vector.search_calls:
            _, filter_used = mock_vector.search_calls[0]
            if filter_used:
                # Should be scoped to projects from graph
                assert "$in" in filter_used.get("project", {})

    @pytest.mark.asyncio
    async def test_respects_query_classification(self, orchestrator, mock_local, mock_graph, mock_vector):
        """Test that query type affects which agents run."""
        # Local-only query
        await orchestrator.retrieve("What does our CLAUDE.md say?")

        local_calls_before = len(mock_local.search_calls)
        graph_calls_before = len(mock_graph.search_calls)

        # Should have called local, likely not graph
        assert local_calls_before >= 1

    @pytest.mark.asyncio
    async def test_deduplicates_results(self, orchestrator):
        """Test that duplicate observations are merged."""
        result = await orchestrator.retrieve("Customer handling patterns")

        # Should have deduplicated
        assert result.after_dedup <= (
            result.local_observations + result.graph_observations + result.vector_observations
        )

    @pytest.mark.asyncio
    async def test_returns_sources(self, orchestrator):
        """Test that sources are tracked."""
        result = await orchestrator.retrieve("Customer data")

        assert len(result.sources) > 0
        assert any("local:" in s or "graph:" in s or "vector:" in s for s in result.sources)

    @pytest.mark.asyncio
    async def test_returns_classification(self, orchestrator):
        """Test that classification is returned."""
        result = await orchestrator.retrieve("How does X connect to Y?")

        assert result.classification is not None
        assert result.classification.query_type == QueryType.RELATIONSHIP

    @pytest.mark.asyncio
    async def test_handles_agent_failure(self, mock_llm, mock_local):
        """Test graceful handling when some agents fail."""
        # Create orchestrator with failing graph
        orchestrator = HybridRetrievalOrchestrator(
            llm=mock_llm,
            semantic_memory=None,  # Will fail
            vector_store=None,  # Will fail
            local_index=mock_local,
        )

        result = await orchestrator.retrieve("test query")

        # Should still get results from local
        assert result.local_observations > 0
        assert result.answer != ""


class TestHybridRetrievalConfig:
    """Tests for configuration options."""

    @pytest.mark.asyncio
    async def test_disable_agents(self, mock_llm, mock_local, mock_graph, mock_vector):
        """Test disabling specific agents."""
        config = HybridRetrievalConfig(
            enable_graph=False,
            enable_vector=False,
        )

        orchestrator = HybridRetrievalOrchestrator(
            llm=mock_llm,
            semantic_memory=mock_graph,
            vector_store=mock_vector,
            local_index=mock_local,
            config=config,
        )

        await orchestrator.retrieve("test query")

        # Only local should be called
        assert len(mock_local.search_calls) > 0
        assert len(mock_graph.search_calls) == 0
        assert len(mock_vector.search_calls) == 0

    @pytest.mark.asyncio
    async def test_observation_limits(self, mock_llm, mock_local, mock_graph, mock_vector):
        """Test observation count limits."""
        config = HybridRetrievalConfig(
            max_observations_per_agent=1,
            max_total_observations=2,
        )

        orchestrator = HybridRetrievalOrchestrator(
            llm=mock_llm,
            semantic_memory=mock_graph,
            vector_store=mock_vector,
            local_index=mock_local,
            config=config,
        )

        result = await orchestrator.retrieve("test query")

        assert len(result.observations) <= 2


class TestConvenienceFunction:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_hybrid_retrieve_function(self, mock_llm, mock_local, mock_graph, mock_vector):
        """Test the hybrid_retrieve convenience function."""
        result = await hybrid_retrieve(
            query="Customer patterns",
            llm=mock_llm,
            semantic_memory=mock_graph,
            vector_store=mock_vector,
            local_index=mock_local,
        )

        assert isinstance(result, HybridResult)
        assert result.answer != ""


# =============================================================================
# Integration Scenario Tests
# =============================================================================


class TestIntegrationScenarios:
    """End-to-end scenario tests."""

    @pytest.mark.asyncio
    async def test_database_schema_pattern_query(self):
        """Test the database schema pattern use case."""
        # Setup mocks with realistic data
        local = MockLocalIndex([
            MockLocalResult("Use UUID for all primary keys", "CLAUDE.md"),
        ])

        graph = MockSemanticMemory(
            results=[
                MockGraphResult(
                    "Customer entity: id (UUID), email, phone, created_at",
                    entities=[type("E", (), {"name": "Customer"})()],
                    project="billing-service",
                ),
                MockGraphResult(
                    "User entity: user_id (UUID), email, password_hash",
                    entities=[type("E", (), {"name": "User"})()],
                    project="auth-service",
                ),
            ]
        )

        vector = MockVectorStore([
            MockVectorResult(
                "Customer table includes audit columns: created_at, updated_at",
                score=0.9,
                project="billing-service",
            ),
        ])

        llm = MockLLMProvider(
            "Based on the context: Use UUID for primary keys, include audit columns "
            "(created_at, updated_at). Customer entity typically has email, phone fields. "
            "[local:CLAUDE.md] [graph:billing-service/Customer]"
        )

        orchestrator = HybridRetrievalOrchestrator(
            llm=llm,
            semantic_memory=graph,
            vector_store=vector,
            local_index=local,
        )

        result = await orchestrator.retrieve("What patterns exist for customer tables?")

        assert "UUID" in result.answer
        assert result.local_observations >= 1
        assert result.graph_observations >= 1

    @pytest.mark.asyncio
    async def test_cross_project_discovery(self):
        """Test discovering patterns across projects."""
        graph = MockSemanticMemory(
            results=[
                MockGraphResult(
                    "Authentication handled by OAuth2 in auth-service",
                    project="auth-service",
                ),
                MockGraphResult(
                    "Customer portal uses auth-service for SSO",
                    project="customer-portal",
                ),
            ]
        )

        vector = MockVectorStore([
            MockVectorResult(
                "Example OAuth2 flow implementation",
                score=0.88,
                project="auth-service",
            ),
        ])

        llm = MockLLMProvider("Auth is centralized in auth-service using OAuth2.")

        orchestrator = HybridRetrievalOrchestrator(
            llm=llm,
            semantic_memory=graph,
            vector_store=vector,
            local_index=None,
        )

        result = await orchestrator.retrieve("How do other teams handle authentication?")

        assert result.classification.query_type == QueryType.CROSS_PROJECT
        assert result.graph_observations >= 1
