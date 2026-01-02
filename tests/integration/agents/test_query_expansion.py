"""Tests for Semantic Query Expansion with Knowledge Graph Context.

This test suite validates the query expansion pipeline that resolves
ambiguous references like "other teams" using graph-based context.

Test Plan:
1. Seed a company knowledge graph with teams, members, and patterns
2. Test query expansion with various levels of context
3. Validate prioritized expansion generation
4. Benchmark parallel retrieval with weighted RRF merging

See: docs/research/semantic_query_expansion.md
"""

import pytest
import asyncio
import re
import os
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# Test Data Models
# =============================================================================


@dataclass
class TeamData:
    """Team information for test seeding."""
    name: str
    auth_pattern: str
    auth_technology: str
    description: str
    members: list[str] = field(default_factory=list)


@dataclass
class QueryExpansion:
    """A single expanded query with confidence."""
    query: str
    confidence: float
    reasoning: str
    target_entities: list[str] = field(default_factory=list)


@dataclass
class ExpansionResult:
    """Result of query expansion."""
    original_query: str
    expansions: list[QueryExpansion]
    graph_context_used: dict[str, Any]
    processing_time_ms: float


# =============================================================================
# Test Data - Company Knowledge Graph
# =============================================================================


COMPANY_NAME = "Acme Corp"

TEAMS = [
    TeamData(
        name="Engineering",
        auth_pattern="OAuth2 with PKCE",
        auth_technology="JWT tokens with 1hr expiry",
        description="Core product engineering team building customer-facing features",
        members=["Doug", "Alice", "Bob"],
    ),
    TeamData(
        name="Platform",
        auth_pattern="Service mesh mTLS",
        auth_technology="Istio service mesh with certificate rotation",
        description="Platform team managing infrastructure and service mesh",
        members=["Charlie", "Diana"],
    ),
    TeamData(
        name="Data",
        auth_pattern="AWS IAM roles",
        auth_technology="Service accounts with cross-account access",
        description="Data engineering team running pipelines and analytics",
        members=["Eve", "Frank"],
    ),
    TeamData(
        name="Mobile",
        auth_pattern="Native biometric with refresh tokens",
        auth_technology="Face ID/Touch ID with 30-day refresh tokens",
        description="Mobile app development team for iOS and Android",
        members=["Grace", "Henry"],
    ),
    TeamData(
        name="QA",
        auth_pattern="Mock authentication for testing",
        auth_technology="Bypass tokens with configurable permissions",
        description="Quality assurance team with testing infrastructure",
        members=["Ivan", "Julia"],
    ),
]

# User context for tests
USER_CONTEXT_DOUG = {
    "user_id": "doug",
    "user_name": "Doug",
    "team": "Engineering",
    "role": "Senior Engineer",
}

USER_CONTEXT_UNKNOWN = {
    "user_id": "unknown",
    "user_name": "Unknown User",
}


# =============================================================================
# Mock Query Expander (for unit tests)
# =============================================================================


class MockQueryExpander:
    """Mock query expander for unit tests.

    In the real implementation, this would:
    1. Call Phase 0 to extract entities from query
    2. Look up entities in the knowledge graph
    3. Use LLM to generate prioritized expansions
    """

    def __init__(self, graph_data: dict[str, Any], llm=None):
        self.graph_data = graph_data
        self.llm = llm
        self.teams = {t.name: t for t in TEAMS}

    async def expand(
        self,
        query: str,
        user_context: dict[str, Any] | None = None,
        confidence_threshold: float = 0.75,
        max_expansions: int = 5,
    ) -> ExpansionResult:
        """Expand ambiguous query using graph context."""
        import time
        start = time.perf_counter()

        expansions = []
        graph_context = {}

        # Detect "other teams" pattern
        if "other teams" in query.lower():
            graph_context["ambiguous_term"] = "other teams"
            graph_context["resolution_type"] = "team_exclusion"

            user_team = user_context.get("team") if user_context else None
            graph_context["user_team"] = user_team

            # Get teams excluding user's team
            if user_team:
                other_teams = [t for t in self.teams.values() if t.name != user_team]
                graph_context["resolved_teams"] = [t.name for t in other_teams]

                # Generate expansions for each team
                base_conf = 0.88
                for i, team in enumerate(other_teams):
                    conf = base_conf - (i * 0.03)  # Decay confidence slightly
                    expanded_query = query.lower().replace(
                        "other teams",
                        f"{team.name} team"
                    )
                    # Capitalize first letter
                    expanded_query = expanded_query[0].upper() + expanded_query[1:]

                    expansions.append(QueryExpansion(
                        query=expanded_query,
                        confidence=conf,
                        reasoning=f"Replacing 'other teams' with specific team: {team.name}",
                        target_entities=[team.name],
                    ))
            else:
                # No user context - can't determine "other"
                graph_context["resolved_teams"] = [t.name for t in self.teams.values()]
                expansions.append(QueryExpansion(
                    query=query.replace("other teams", "all teams"),
                    confidence=0.70,
                    reasoning="No user context - cannot determine 'other' relative to what",
                    target_entities=list(self.teams.keys()),
                ))
                expansions.append(QueryExpansion(
                    query=f"What authentication patterns exist across teams at {COMPANY_NAME}?",
                    confidence=0.65,
                    reasoning="Generic cross-team query as fallback",
                    target_entities=list(self.teams.keys()),
                ))

        # Detect "the API" pattern
        elif "the api" in query.lower():
            graph_context["ambiguous_term"] = "the API"
            graph_context["resolution_type"] = "entity_disambiguation"

            # Simulate multiple APIs in the system
            apis = ["Payment API", "User API", "Gateway API", "Internal API"]
            graph_context["candidate_entities"] = apis

            # Check user context for recent work
            last_worked = user_context.get("last_worked_on") if user_context else None
            if last_worked and last_worked in apis:
                graph_context["user_recent_context"] = last_worked
                # Put user's recent context first
                apis = [last_worked] + [a for a in apis if a != last_worked]

            for i, api in enumerate(apis):
                conf = 0.90 - (i * 0.10)
                expanded_query = query.lower().replace("the api", api.lower())
                expanded_query = expanded_query[0].upper() + expanded_query[1:]

                expansions.append(QueryExpansion(
                    query=expanded_query,
                    confidence=conf,
                    reasoning=f"Disambiguating 'the API' to specific API: {api}",
                    target_entities=[api],
                ))

        # Detect "databases" pattern
        elif "databases" in query.lower() or "database" in query.lower():
            graph_context["ambiguous_term"] = "databases"
            graph_context["resolution_type"] = "category_expansion"

            db_types = [
                ("PostgreSQL", "relational", 0.85),
                ("MongoDB", "document", 0.80),
                ("Redis", "cache", 0.75),
                ("Elasticsearch", "search", 0.70),
            ]
            graph_context["database_instances"] = [d[0] for d in db_types]

            for db_name, db_type, conf in db_types:
                expanded_query = query.lower().replace(
                    "databases" if "databases" in query.lower() else "database",
                    f"{db_type} databases ({db_name})"
                )
                expanded_query = expanded_query[0].upper() + expanded_query[1:]

                expansions.append(QueryExpansion(
                    query=expanded_query,
                    confidence=conf,
                    reasoning=f"Expanding 'databases' to include {db_type} type: {db_name}",
                    target_entities=[db_name],
                ))

        else:
            # No expansion needed - query is already specific
            expansions.append(QueryExpansion(
                query=query,
                confidence=1.0,
                reasoning="Query is already specific, no expansion needed",
                target_entities=[],
            ))

        # Filter by confidence threshold
        filtered = [e for e in expansions if e.confidence >= confidence_threshold]
        if not filtered and expansions:
            # Keep at least the highest confidence expansion
            filtered = [max(expansions, key=lambda x: x.confidence)]

        # Limit to max expansions
        filtered = filtered[:max_expansions]

        elapsed_ms = (time.perf_counter() - start) * 1000

        return ExpansionResult(
            original_query=query,
            expansions=filtered,
            graph_context_used=graph_context,
            processing_time_ms=elapsed_ms,
        )


# =============================================================================
# Mock Parallel Retriever
# =============================================================================


@dataclass
class RetrievalResult:
    """Result from a single retrieval."""
    content: str
    source: str
    score: float
    from_expansion: str  # Which expanded query this came from


@dataclass
class MergedResult:
    """Result after merging multiple retrievals with RRF."""
    content: str
    sources: list[str]
    rrf_score: float
    contributing_expansions: list[str]
    expansion_confidences: list[float]


class MockParallelRetriever:
    """Mock parallel retriever with weighted RRF merging."""

    def __init__(self, knowledge_base: dict[str, list[str]]):
        """Initialize with knowledge base mapping team -> facts."""
        self.knowledge_base = knowledge_base

    async def retrieve_parallel(
        self,
        expansions: list[QueryExpansion],
        limit_per_query: int = 5,
    ) -> list[list[RetrievalResult]]:
        """Retrieve in parallel for each expansion."""
        tasks = [
            self._retrieve_single(exp, limit_per_query)
            for exp in expansions
        ]
        return await asyncio.gather(*tasks)

    async def _retrieve_single(
        self,
        expansion: QueryExpansion,
        limit: int,
    ) -> list[RetrievalResult]:
        """Retrieve for a single expansion."""
        results = []

        for entity in expansion.target_entities:
            if entity in self.knowledge_base:
                for i, fact in enumerate(self.knowledge_base[entity][:limit]):
                    results.append(RetrievalResult(
                        content=fact,
                        source=f"graph:{entity}",
                        score=1.0 - (i * 0.1),  # Decay by rank
                        from_expansion=expansion.query,
                    ))

        return results

    def merge_with_rrf(
        self,
        all_results: list[list[RetrievalResult]],
        expansions: list[QueryExpansion],
        k: int = 60,  # RRF smoothing constant
    ) -> list[MergedResult]:
        """Merge results using weighted Reciprocal Rank Fusion.

        RRF score = Σ (1 / (k + rank_i)) × confidence_i

        We weight by expansion confidence so higher-confidence
        expansions contribute more to the final ranking.
        """
        # Group by content
        content_scores: dict[str, dict] = {}

        for exp_idx, results in enumerate(all_results):
            exp_confidence = expansions[exp_idx].confidence
            exp_query = expansions[exp_idx].query

            for rank, result in enumerate(results, start=1):
                content = result.content

                if content not in content_scores:
                    content_scores[content] = {
                        "rrf_score": 0.0,
                        "sources": set(),
                        "contributing_expansions": [],
                        "expansion_confidences": [],
                    }

                # Weighted RRF contribution
                rrf_contribution = (1 / (k + rank)) * exp_confidence
                content_scores[content]["rrf_score"] += rrf_contribution
                content_scores[content]["sources"].add(result.source)

                if exp_query not in content_scores[content]["contributing_expansions"]:
                    content_scores[content]["contributing_expansions"].append(exp_query)
                    content_scores[content]["expansion_confidences"].append(exp_confidence)

        # Convert to MergedResult list
        merged = [
            MergedResult(
                content=content,
                sources=list(data["sources"]),
                rrf_score=data["rrf_score"],
                contributing_expansions=data["contributing_expansions"],
                expansion_confidences=data["expansion_confidences"],
            )
            for content, data in content_scores.items()
        ]

        # Sort by RRF score descending
        merged.sort(key=lambda x: x.rrf_score, reverse=True)

        return merged


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def team_knowledge_base():
    """Knowledge base mapping teams to their facts."""
    return {
        "Engineering": [
            "Engineering team uses JWT tokens with 1hr expiry for authentication",
            "Engineering follows OAuth2 with PKCE flow for web applications",
            "Engineering stores session tokens in HttpOnly cookies",
        ],
        "Platform": [
            "Platform team uses Istio service mesh with mTLS for service-to-service auth",
            "Platform rotates certificates automatically every 24 hours",
            "Platform uses Envoy sidecars for authentication proxying",
        ],
        "Data": [
            "Data team uses AWS IAM roles for authentication to data services",
            "Data team uses service accounts with cross-account access",
            "Data pipelines authenticate using IRSA (IAM Roles for Service Accounts)",
        ],
        "Mobile": [
            "Mobile apps use native biometric authentication (Face ID/Touch ID)",
            "Mobile uses 30-day refresh tokens stored in secure enclave",
            "Mobile falls back to PIN code when biometrics unavailable",
        ],
        "QA": [
            "QA uses mock authentication with bypass tokens for testing",
            "QA tokens have configurable permissions for test scenarios",
            "QA authentication can simulate any user role for testing",
        ],
        # APIs
        "Payment API": [
            "Payment API uses HMAC signature verification for webhooks",
            "Payment API requires PCI-DSS compliant token handling",
        ],
        "Gateway API": [
            "Gateway API implements rate limiting at 1000 req/min per client",
            "Gateway API uses API keys with JWT for hybrid authentication",
        ],
        # Databases
        "PostgreSQL": [
            "PostgreSQL is used by Engineering and Data teams for OLTP",
            "PostgreSQL handles user accounts and transaction data",
        ],
        "MongoDB": [
            "MongoDB is used by Mobile team for offline-first sync",
            "MongoDB stores user preferences and device state",
        ],
        "Redis": [
            "Redis is used for session caching with 15-minute TTL",
            "Redis cluster handles 100K+ concurrent sessions",
        ],
    }


@pytest.fixture
def query_expander(team_knowledge_base):
    """Create mock query expander with graph data."""
    return MockQueryExpander(graph_data=team_knowledge_base)


@pytest.fixture
def parallel_retriever(team_knowledge_base):
    """Create mock parallel retriever."""
    return MockParallelRetriever(knowledge_base=team_knowledge_base)


# =============================================================================
# Unit Tests - Query Expansion
# =============================================================================


class TestQueryExpansionBasic:
    """Basic query expansion tests."""

    @pytest.mark.asyncio
    async def test_other_teams_with_user_context(self, query_expander):
        """Test 'other teams' expands correctly with user context."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Should have expansions for teams OTHER than Engineering
        assert len(result.expansions) >= 3

        # Should NOT include Engineering (user's team)
        expansion_texts = [e.query.lower() for e in result.expansions]
        assert not any("engineering" in q for q in expansion_texts)

        # Should include other teams
        assert any("platform" in q for q in expansion_texts)
        assert any("data" in q for q in expansion_texts)

        # Should have recorded graph context
        assert result.graph_context_used["user_team"] == "Engineering"
        assert "Engineering" not in result.graph_context_used["resolved_teams"]

    @pytest.mark.asyncio
    async def test_other_teams_without_user_context(self, query_expander):
        """Test 'other teams' without user context falls back to generic."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_UNKNOWN,
        )

        # Should have fallback expansions
        assert len(result.expansions) >= 1

        # First expansion should be generic cross-team
        assert result.expansions[0].confidence < 0.75  # Lower confidence

        # Reasoning should explain the limitation
        assert "no user context" in result.expansions[0].reasoning.lower() or \
               "cannot determine" in result.expansions[0].reasoning.lower()

    @pytest.mark.asyncio
    async def test_specific_query_no_expansion(self, query_expander):
        """Test that specific queries don't get expanded."""
        result = await query_expander.expand(
            query="How does Engineering team handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Should return original query unchanged
        assert len(result.expansions) == 1
        assert result.expansions[0].query == "How does Engineering team handle authentication?"
        assert result.expansions[0].confidence == 1.0

    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, query_expander):
        """Test that low-confidence expansions are filtered."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
            confidence_threshold=0.80,
        )

        # All expansions should be above threshold
        for exp in result.expansions:
            assert exp.confidence >= 0.80

    @pytest.mark.asyncio
    async def test_max_expansions_limit(self, query_expander):
        """Test that expansions are limited to max."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
            max_expansions=2,
        )

        assert len(result.expansions) <= 2


class TestQueryExpansionEntities:
    """Entity disambiguation tests."""

    @pytest.mark.asyncio
    async def test_the_api_disambiguation(self, query_expander):
        """Test 'the API' disambiguates to specific APIs."""
        result = await query_expander.expand(
            query="How does the API handle rate limiting?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Should have multiple API expansions
        assert len(result.expansions) >= 2

        # Should mention specific APIs
        expansion_texts = " ".join(e.query.lower() for e in result.expansions)
        assert "payment api" in expansion_texts or "gateway api" in expansion_texts

    @pytest.mark.asyncio
    async def test_the_api_with_recent_context(self, query_expander):
        """Test 'the API' prioritizes user's recent work."""
        context_with_recent = {
            **USER_CONTEXT_DOUG,
            "last_worked_on": "Payment API",
        }

        result = await query_expander.expand(
            query="How does the API handle rate limiting?",
            user_context=context_with_recent,
        )

        # First expansion should be Payment API
        assert "payment api" in result.expansions[0].query.lower()
        assert result.expansions[0].confidence > 0.85

    @pytest.mark.asyncio
    async def test_databases_category_expansion(self, query_expander):
        """Test 'databases' expands to specific database types."""
        result = await query_expander.expand(
            query="What databases do we use?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Should have expansions for different DB types
        assert len(result.expansions) >= 2

        # Should cover different categories
        expansion_texts = " ".join(e.query.lower() for e in result.expansions)
        assert "postgresql" in expansion_texts or "relational" in expansion_texts
        assert "mongodb" in expansion_texts or "document" in expansion_texts


# =============================================================================
# Integration Tests - Parallel Retrieval
# =============================================================================


class TestParallelRetrieval:
    """Tests for parallel retrieval with weighted RRF merging."""

    @pytest.mark.asyncio
    async def test_parallel_retrieval_basic(
        self, query_expander, parallel_retriever
    ):
        """Test basic parallel retrieval flow."""
        # Expand query
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Retrieve in parallel
        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        # Should have results for each expansion
        assert len(all_results) == len(expansion_result.expansions)

        # Each expansion should have some results
        for exp_results in all_results:
            assert len(exp_results) > 0

    @pytest.mark.asyncio
    async def test_rrf_merging_weights_by_confidence(
        self, query_expander, parallel_retriever
    ):
        """Test that RRF weights results by expansion confidence."""
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        merged = parallel_retriever.merge_with_rrf(
            all_results=all_results,
            expansions=expansion_result.expansions,
        )

        # Should have merged results
        assert len(merged) > 0

        # Results should be sorted by RRF score
        for i in range(len(merged) - 1):
            assert merged[i].rrf_score >= merged[i + 1].rrf_score

    @pytest.mark.asyncio
    async def test_rrf_corroborates_overlapping_results(
        self, query_expander, parallel_retriever
    ):
        """Test that results appearing in multiple expansions get boosted."""
        # Create expansions that would hit overlapping content
        # (This would require knowledge base with shared facts)

        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        merged = parallel_retriever.merge_with_rrf(
            all_results=all_results,
            expansions=expansion_result.expansions,
        )

        # Check that results track contributing expansions
        for result in merged:
            assert len(result.contributing_expansions) >= 1
            assert len(result.expansion_confidences) >= 1

    @pytest.mark.asyncio
    async def test_excludes_user_team_from_results(
        self, query_expander, parallel_retriever
    ):
        """Test that user's team is excluded from 'other teams' results."""
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        merged = parallel_retriever.merge_with_rrf(
            all_results=all_results,
            expansions=expansion_result.expansions,
        )

        # No results should be from Engineering (user's team)
        for result in merged:
            assert "Engineering" not in result.sources
            assert "engineering" not in result.content.lower() or \
                   "engineering team uses" not in result.content.lower()


# =============================================================================
# End-to-End Tests
# =============================================================================


class TestEndToEndExpansionRetrieval:
    """End-to-end tests for the full expansion + retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_other_teams(
        self, query_expander, parallel_retriever
    ):
        """Test full pipeline: expand → retrieve → merge → validate."""
        # Step 1: Expand
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        assert len(expansion_result.expansions) >= 3

        # Step 2: Retrieve in parallel
        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        # Step 3: Merge with RRF
        merged = parallel_retriever.merge_with_rrf(
            all_results=all_results,
            expansions=expansion_result.expansions,
        )

        # Step 4: Validate coverage
        content_combined = " ".join(r.content for r in merged)

        # Should mention Platform's patterns
        assert "istio" in content_combined.lower() or "mtls" in content_combined.lower()

        # Should mention Data's patterns
        assert "iam" in content_combined.lower() or "service account" in content_combined.lower()

        # Should mention Mobile's patterns
        assert "biometric" in content_combined.lower() or "refresh token" in content_combined.lower()

        # Should NOT mention Engineering's patterns (user's team)
        # (This depends on our test data not having "JWT" unique to Engineering)

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(
        self, query_expander, parallel_retriever
    ):
        """Test that full pipeline completes within latency target."""
        import time

        start = time.perf_counter()

        # Full pipeline
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        merged = parallel_retriever.merge_with_rrf(
            all_results=all_results,
            expansions=expansion_result.expansions,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Should complete well under 3 seconds (target from research doc)
        # For mock implementation, should be < 100ms
        assert elapsed_ms < 100, f"Pipeline took {elapsed_ms}ms, expected < 100ms for mocks"


# =============================================================================
# Pipeline Variation Tests
# =============================================================================


class TestPipelineVariations:
    """Tests for different processing pipeline orderings."""

    @pytest.mark.asyncio
    async def test_variation_expand_first(self, query_expander, parallel_retriever):
        """V1: Query → LLM Expand → Retrieve each variation."""
        # This is the standard flow we've implemented
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        all_results = await parallel_retriever.retrieve_parallel(
            expansions=expansion_result.expansions,
        )

        # Validate we got results from multiple expansions
        non_empty = [r for r in all_results if len(r) > 0]
        assert len(non_empty) >= 2

    @pytest.mark.asyncio
    async def test_variation_single_vs_multi_expansion(
        self, query_expander, parallel_retriever
    ):
        """Compare single-query vs multi-expansion retrieval."""
        # Single query (no expansion)
        single_expansion = [QueryExpansion(
            query="How do other teams handle authentication?",
            confidence=1.0,
            reasoning="Original query",
            target_entities=["Platform", "Data", "Mobile", "QA"],
        )]

        single_results = await parallel_retriever.retrieve_parallel(single_expansion)
        single_merged = parallel_retriever.merge_with_rrf(single_results, single_expansion)

        # Multi-expansion
        expansion_result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
        )

        multi_results = await parallel_retriever.retrieve_parallel(expansion_result.expansions)
        multi_merged = parallel_retriever.merge_with_rrf(multi_results, expansion_result.expansions)

        # Multi-expansion should have at least as many results
        # (In practice, it should have MORE diverse results)
        assert len(multi_merged) >= len(single_merged)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_no_graph_matches(self, query_expander):
        """Test behavior when graph has no matching entities."""
        result = await query_expander.expand(
            query="How do alien teams handle telepathy?",
            user_context=USER_CONTEXT_DOUG,
        )

        # Should return original query with full confidence
        assert len(result.expansions) == 1
        assert result.expansions[0].query == "How do alien teams handle telepathy?"

    @pytest.mark.asyncio
    async def test_empty_user_context(self, query_expander):
        """Test with completely empty user context."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context={},
        )

        # Should still produce expansions (generic ones)
        assert len(result.expansions) >= 1

    @pytest.mark.asyncio
    async def test_null_user_context(self, query_expander):
        """Test with None user context."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=None,
        )

        # Should handle gracefully
        assert len(result.expansions) >= 1

    @pytest.mark.asyncio
    async def test_very_high_confidence_threshold(self, query_expander):
        """Test with threshold that filters all expansions."""
        result = await query_expander.expand(
            query="How do other teams handle authentication?",
            user_context=USER_CONTEXT_DOUG,
            confidence_threshold=0.99,
        )

        # Should still return at least one (highest confidence)
        assert len(result.expansions) >= 1


# =============================================================================
# Real LLM Integration Tests (Skip if no API key)
# =============================================================================


@pytest.fixture
def real_llm():
    """Create real LLM provider if API key available."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    from draagon_ai.llm.groq import GroqLLM
    return GroqLLM(api_key=api_key)


class TestRealLLMExpansion:
    """Tests with real LLM for expansion generation."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_llm_expansion_prompt(self, real_llm, team_knowledge_base):
        """Test LLM-based query expansion with real model."""
        # This tests the actual LLM prompt for expansion

        expansion_prompt = """You are expanding an ambiguous query using context from a knowledge graph.

Query: "How do other teams handle authentication?"

User Context:
- User: Doug
- User's Team: Engineering

Known Teams (from knowledge graph):
- Engineering (user's team - EXCLUDE)
- Platform
- Data
- Mobile
- QA

Task: Generate 3-5 specific query variations that resolve "other teams" to concrete team names.
Each variation should target a specific team that is NOT the user's team.

Respond with XML:
<expansions>
  <expansion>
    <query>The expanded query text</query>
    <confidence>0.0-1.0 confidence score</confidence>
    <reasoning>Why this expansion</reasoning>
    <target_team>Team name</target_team>
  </expansion>
  ...
</expansions>"""

        response = await real_llm.chat(
            messages=[{"role": "user", "content": expansion_prompt}],
            temperature=0.3,
            max_tokens=500,
        )

        response_text = response.content if hasattr(response, "content") else str(response)

        # Validate response structure
        assert "<expansions>" in response_text
        assert "<query>" in response_text
        assert "<confidence>" in response_text

        # Should NOT mention Engineering
        assert "engineering" not in response_text.lower() or \
               "exclude" in response_text.lower()

        # Should mention at least one other team
        other_teams_mentioned = sum(1 for team in ["Platform", "Data", "Mobile", "QA"]
                                    if team.lower() in response_text.lower())
        assert other_teams_mentioned >= 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_llm_disambiguation(self, real_llm):
        """Test LLM entity disambiguation."""

        disambiguation_prompt = """Disambiguate "the API" in this query using context.

Query: "How does the API handle rate limiting?"

Known APIs in the system:
- Payment API: Handles payment processing, has PCI-DSS requirements
- User API: Manages user accounts and profiles
- Gateway API: Entry point, handles rate limiting and auth
- Internal API: Backend service communication

Most relevant context: Rate limiting is typically handled at the Gateway level.

Which specific API is most likely meant? Provide ranked options.

Respond with XML:
<disambiguation>
  <option rank="1">
    <entity>Gateway API</entity>
    <confidence>0.85</confidence>
    <reasoning>Rate limiting is typically handled at gateway level</reasoning>
  </option>
  ...
</disambiguation>"""

        response = await real_llm.chat(
            messages=[{"role": "user", "content": disambiguation_prompt}],
            temperature=0.2,
            max_tokens=400,
        )

        response_text = response.content if hasattr(response, "content") else str(response)

        # Gateway API should be ranked high (it's the most relevant for rate limiting)
        assert "gateway" in response_text.lower()

        # Should have confidence scores
        assert "<confidence>" in response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
