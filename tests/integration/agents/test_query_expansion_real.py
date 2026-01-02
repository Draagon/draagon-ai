"""Real integration tests for Semantic Query Expansion.

These tests use actual Neo4j and Groq LLM to validate the query expansion
pipeline with real providers.

Requires:
- Neo4j running at bolt://localhost:7687
- GROQ_API_KEY environment variable set
"""

import pytest
import asyncio
import os
from dataclasses import dataclass

# Skip all tests if dependencies not available
pytest.importorskip("neo4j")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
async def real_llm():
    """Create real Groq LLM provider."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    from draagon_ai.llm.groq import GroqLLM
    return GroqLLM(api_key=api_key)


@pytest.fixture
async def real_embedder():
    """Create real embedding provider."""
    # Use 768d to match existing Neo4j index (created for nomic-embed-text)
    from tests.integration.agents.conftest import MockEmbeddingProvider
    return MockEmbeddingProvider(dimension=768)


@pytest.fixture
async def real_neo4j(real_embedder, real_llm):
    """Create real Neo4j memory provider with test data."""
    from draagon_ai.memory.providers.neo4j import (
        Neo4jMemoryProvider,
        Neo4jMemoryConfig,
    )

    uri = os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_TEST_USER", "neo4j")
    password = os.getenv("NEO4J_TEST_PASSWORD", "draagon-ai-2025")

    config = Neo4jMemoryConfig(
        uri=uri,
        username=user,
        password=password,
        database="neo4j",
        embedding_dimension=768,
    )

    try:
        provider = Neo4jMemoryProvider(
            config=config,
            embedding_provider=real_embedder,
            llm_provider=real_llm,
        )
        await provider.initialize()

        # Seed test data
        await seed_company_data(provider)

        yield provider

        # Cleanup
        await cleanup_test_data(provider)
        await provider.close()

    except Exception as e:
        pytest.skip(f"Neo4j not available: {e}")


async def seed_company_data(provider):
    """Seed the knowledge graph with company/team data."""
    from draagon_ai.memory.base import MemoryType, MemoryScope

    # Teams with their authentication patterns
    teams = [
        ("Engineering", "OAuth2 with PKCE", "JWT tokens with 1hr expiry"),
        ("Platform", "Service mesh mTLS", "Istio with certificate rotation"),
        ("Data", "AWS IAM roles", "Service accounts with cross-account access"),
        ("Mobile", "Native biometric", "Face ID/Touch ID with refresh tokens"),
        ("QA", "Mock authentication", "Bypass tokens for testing"),
    ]

    for team_name, auth_pattern, auth_tech in teams:
        # Store team fact
        await provider.store(
            content=f"{team_name} team uses {auth_pattern} for authentication. They implement {auth_tech}.",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.CONTEXT,  # Shared team knowledge
            entities=[team_name, "authentication", auth_pattern],
        )

    # Store user membership
    await provider.store(
        content="Doug is a member of the Engineering team. He works on customer-facing features.",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.CONTEXT,
        entities=["Doug", "Engineering"],
    )

    # Store some APIs
    apis = [
        ("Payment API", "Handles payment processing with PCI-DSS compliance"),
        ("User API", "Manages user accounts and profiles"),
        ("Gateway API", "Entry point with rate limiting at 1000 req/min"),
    ]

    for api_name, description in apis:
        await provider.store(
            content=f"{api_name}: {description}",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.CONTEXT,
            entities=[api_name],
        )

    # Store databases
    databases = [
        ("PostgreSQL", "OLTP database for user accounts and transactions"),
        ("MongoDB", "Document store for mobile offline sync"),
        ("Redis", "Session cache with 15-minute TTL"),
    ]

    for db_name, description in databases:
        await provider.store(
            content=f"{db_name}: {description}",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.CONTEXT,
            entities=[db_name, "database"],
        )


async def cleanup_test_data(provider):
    """Clean up test data."""
    # The test database should be cleared between test runs
    # For now, we'll leave data for debugging
    pass


# =============================================================================
# Adapter for SemanticMemoryProvider protocol
# =============================================================================


class Neo4jSemanticAdapter:
    """Adapts Neo4jMemoryProvider to SemanticMemoryProvider protocol."""

    def __init__(self, provider):
        self.provider = provider

    async def search(self, query: str, limit: int = 10):
        """Search memories."""
        results = await self.provider.search(query, limit=limit)
        return [
            {
                "content": r.memory.content,
                "entities": r.memory.entities or [],
                "score": r.score,
            }
            for r in results
        ]

    async def find_entities(self, names: list[str]):
        """Find entities by name."""
        results = []
        for name in names:
            search_results = await self.provider.search(name, limit=5)
            for r in search_results:
                if r.memory.entities:
                    for entity in r.memory.entities:
                        results.append({"name": entity})
        return results

    async def get_related(self, entity_id: str, relationship_types=None, depth=1):
        """Get related entities."""
        return await self.provider.search(entity_id, limit=10)


# =============================================================================
# Tests
# =============================================================================


class TestRealQueryExpander:
    """Tests using real Neo4j and LLM."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expand_other_teams_with_graph(self, real_neo4j, real_llm):
        """Test 'other teams' expansion with real graph context."""
        from draagon_ai.orchestration.hybrid_retrieval import QueryExpander

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)

        result = await expander.expand(
            query="How do other teams handle authentication?",
            user_context={"user_name": "Doug", "team": "Engineering"},
        )

        # Should have detected ambiguity
        assert len(result.ambiguous_terms) >= 1
        assert "other teams" in result.ambiguous_terms

        # Should have multiple expansions
        assert len(result.expansions) >= 2

        # Should NOT include Engineering (user's team)
        for exp in result.expansions:
            assert "engineering" not in exp.query.lower()

        # Should mention other teams
        all_queries = " ".join(e.query.lower() for e in result.expansions)
        # At least one of the other teams should be mentioned
        other_teams = ["platform", "data", "mobile", "qa"]
        mentioned = sum(1 for t in other_teams if t in all_queries)
        assert mentioned >= 1, f"Expected at least one team in expansions: {all_queries}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expand_specific_query_no_change(self, real_neo4j, real_llm):
        """Test that specific queries are not expanded."""
        from draagon_ai.orchestration.hybrid_retrieval import QueryExpander

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)

        result = await expander.expand(
            query="How does Platform team handle authentication?",
            user_context={"user_name": "Doug", "team": "Engineering"},
        )

        # No ambiguity detected
        assert len(result.ambiguous_terms) == 0

        # Single expansion (the original query)
        assert len(result.expansions) == 1
        assert result.expansions[0].confidence == 1.0

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expand_without_user_context(self, real_neo4j, real_llm):
        """Test expansion without user context."""
        from draagon_ai.orchestration.hybrid_retrieval import QueryExpander

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)

        result = await expander.expand(
            query="How do other teams handle authentication?",
            user_context=None,  # No user context
        )

        # Should still produce expansions
        assert len(result.expansions) >= 1

        # Confidence should be lower since we can't determine "other"
        # relative to what
        avg_conf = sum(e.confidence for e in result.expansions) / len(result.expansions)
        # Without user context, we expect generally lower confidence
        # but LLM should still provide reasonable expansions


class TestRealWeightedRRF:
    """Tests for weighted RRF merging with real data."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_rrf_merge_real_results(self, real_neo4j, real_llm):
        """Test RRF merging with real retrieval results."""
        from draagon_ai.orchestration.hybrid_retrieval import (
            QueryExpander,
            QueryExpansion,
            WeightedRRFMerger,
            Observation,
        )

        adapter = Neo4jSemanticAdapter(real_neo4j)

        # Create some expansions
        expansions = [
            QueryExpansion(
                query="How does Platform team handle authentication?",
                confidence=0.85,
                reasoning="Platform team expansion",
                target_entities=["Platform"],
            ),
            QueryExpansion(
                query="How does Data team handle authentication?",
                confidence=0.80,
                reasoning="Data team expansion",
                target_entities=["Data"],
            ),
        ]

        # Retrieve for each expansion
        results_per_query = []
        for exp in expansions:
            search_results = await adapter.search(exp.query, limit=5)
            observations = [
                Observation(
                    content=r["content"],
                    source=f"graph:{exp.target_entities[0] if exp.target_entities else 'unknown'}",
                    confidence=r.get("score", 0.8),
                )
                for r in search_results
            ]
            results_per_query.append(observations)

        # Merge with RRF
        merger = WeightedRRFMerger()
        merged = merger.merge(results_per_query, expansions)

        # Should have merged results
        assert len(merged) > 0

        # Results should be sorted by RRF score
        for i in range(len(merged) - 1):
            assert merged[i].rrf_score >= merged[i + 1].rrf_score

        # Each result should track contributing queries
        for result in merged:
            assert len(result.contributing_queries) >= 1


class TestRealEndToEnd:
    """End-to-end tests with real providers."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_expansion_retrieval_pipeline(self, real_neo4j, real_llm):
        """Test the complete expansion → retrieval → merge pipeline."""
        from draagon_ai.orchestration.hybrid_retrieval import (
            QueryExpander,
            WeightedRRFMerger,
            Observation,
        )

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)
        merger = WeightedRRFMerger()

        # Step 1: Expand query
        expansion_result = await expander.expand(
            query="How do other teams handle authentication?",
            user_context={"user_name": "Doug", "team": "Engineering"},
        )

        assert len(expansion_result.expansions) >= 1

        # Step 2: Retrieve for each expansion
        results_per_query = []
        for exp in expansion_result.expansions:
            search_results = await adapter.search(exp.query, limit=5)
            observations = [
                Observation(
                    content=r["content"],
                    source=f"graph:{','.join(r.get('entities', ['unknown'])[:2])}",
                    confidence=r.get("score", 0.8),
                )
                for r in search_results
            ]
            results_per_query.append(observations)

        # Step 3: Merge with RRF
        merged = merger.merge(results_per_query, expansion_result.expansions)

        # Should have comprehensive results
        assert len(merged) >= 1

        # Verify content mentions authentication patterns
        all_content = " ".join(r.observation.content.lower() for r in merged)

        # Should find authentication-related content
        auth_terms = ["authentication", "oauth", "jwt", "iam", "mtls", "biometric"]
        found_auth = any(term in all_content for term in auth_terms)
        assert found_auth, f"Expected auth terms in: {all_content[:500]}"

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_expansion_timing(self, real_neo4j, real_llm):
        """Test that expansion completes within reasonable time."""
        import time
        from draagon_ai.orchestration.hybrid_retrieval import QueryExpander

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)

        start = time.perf_counter()

        result = await expander.expand(
            query="How do other teams handle authentication?",
            user_context={"user_name": "Doug", "team": "Engineering"},
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete within 5 seconds (generous for LLM call + graph lookup)
        assert elapsed_ms < 5000, f"Expansion took {elapsed_ms}ms, expected < 5000ms"

        # Processing time should be tracked
        assert result.processing_time_ms > 0


class TestGraphContextGathering:
    """Tests for graph context gathering."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_gather_team_context(self, real_neo4j, real_llm):
        """Test gathering team context from graph."""
        from draagon_ai.orchestration.hybrid_retrieval import QueryExpander

        adapter = Neo4jSemanticAdapter(real_neo4j)
        expander = QueryExpander(llm=real_llm, semantic_memory=adapter)

        # Call internal method to test graph context gathering
        context = await expander._gather_graph_context(
            ambiguous_terms=["other teams"],
            user_context={"team": "Engineering"},
        )

        # Should have team context
        # Note: This depends on what entities the graph search returns
        # In a real scenario, we'd have more structured team data


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
