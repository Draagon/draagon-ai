#!/usr/bin/env python3
"""Demo: Full Query Expansion Pipeline with Real Systems.

This script demonstrates the complete query expansion pipeline:
1. Seed company data into Neo4j knowledge graph
2. Run ambiguous queries through the QueryExpander
3. Show how expansions are generated with confidence scores
4. Demonstrate parallel retrieval with weighted RRF merging
5. Show the full HybridRetrievalOrchestrator in action

IMPORTANT: Uses REAL embeddings per CONSTITUTION.md Section 1.7.
Mock embeddings would break semantic search completely.

Run with:
    source .env && python tests/integration/agents/demo_query_expansion.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


async def main():
    """Run the full demo."""
    print("\n" + "=" * 70)
    print("SEMANTIC QUERY EXPANSION DEMO - Real Systems")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set. Run: source .env")
        return

    # Import after path setup
    from draagon_ai.llm.groq import GroqLLM
    from draagon_ai.memory.providers.neo4j import Neo4jMemoryProvider, Neo4jMemoryConfig
    from draagon_ai.memory.base import MemoryType, MemoryScope
    from draagon_ai.orchestration.hybrid_retrieval import (
        QueryExpander,
        WeightedRRFMerger,
        QueryExpansion,
        Observation,
        HybridRetrievalOrchestrator,
        HybridRetrievalConfig,
    )
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    # Initialize providers
    print("\n[1] Initializing providers...")
    llm = GroqLLM(api_key=api_key)

    # Use REAL embeddings via Ollama - per CONSTITUTION.md Section 1.7
    # Mock embeddings are FORBIDDEN for semantic search
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.168.200:11434")
    print(f"    Connecting to Ollama at {ollama_url}...")
    embedder = OllamaEmbeddingProvider(
        base_url=ollama_url,
        model="nomic-embed-text",
        dimension=768,
    )

    # Test embedding connection
    try:
        test_vec = await embedder.embed("test")
        print(f"    ✓ Ollama embeddings ready ({len(test_vec)} dimensions)")
    except Exception as e:
        print(f"    ✗ Ollama not available: {e}")
        print("    Make sure Ollama is running with nomic-embed-text model")
        return

    config = Neo4jMemoryConfig(
        uri=os.getenv("NEO4J_TEST_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_TEST_USER", "neo4j"),
        password=os.getenv("NEO4J_TEST_PASSWORD", "draagon-ai-2025"),
        database="neo4j",
        embedding_dimension=768,  # nomic-embed-text dimension
    )

    try:
        provider = Neo4jMemoryProvider(config, embedder, llm)
        await provider.initialize()
        print("    ✓ Neo4j connected")
        print("    ✓ Groq LLM ready")
    except Exception as e:
        print(f"    ✗ Failed to connect: {e}")
        return

    # Seed test data
    print("\n[2] Seeding company knowledge graph...")
    teams = [
        ("Engineering", "OAuth2 with PKCE", "JWT tokens with 1hr expiry"),
        ("Platform", "Service mesh mTLS", "Istio with certificate rotation"),
        ("Data", "AWS IAM roles", "Service accounts with cross-account access"),
        ("Mobile", "Native biometric", "Face ID/Touch ID with refresh tokens"),
        ("QA", "Mock authentication", "Bypass tokens for testing"),
    ]

    for team_name, auth_pattern, auth_tech in teams:
        await provider.store(
            content=f"{team_name} team uses {auth_pattern} for authentication. They implement {auth_tech}.",
            memory_type=MemoryType.FACT,
            scope=MemoryScope.CONTEXT,
            entities=[team_name, "authentication", auth_pattern],
        )
        print(f"    ✓ Stored: {team_name} team auth pattern")

    # Store user membership
    await provider.store(
        content="Doug is a member of the Engineering team. He works on customer-facing features.",
        memory_type=MemoryType.FACT,
        scope=MemoryScope.CONTEXT,
        entities=["Doug", "Engineering"],
    )
    print("    ✓ Stored: Doug's team membership")

    # Create adapter for QueryExpander
    class Neo4jSemanticAdapter:
        def __init__(self, neo4j_provider):
            self.provider = neo4j_provider

        async def search(self, query: str, limit: int = 10):
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
            results = []
            for name in names:
                search_results = await self.provider.search(name, limit=5)
                for r in search_results:
                    if r.memory.entities:
                        for entity in r.memory.entities:
                            results.append({"name": entity})
            return results

    adapter = Neo4jSemanticAdapter(provider)

    # Debug: Check what the adapter returns for "team" search
    print("\n[DEBUG] Testing adapter search for 'team':")
    debug_results = await adapter.search("team", limit=10)
    for i, r in enumerate(debug_results):
        print(f"    {i+1}. entities={r.get('entities', [])}, content={r.get('content', '')[:60]}...")

    # Debug: Try more specific search terms
    print("\n[DEBUG] Testing adapter search for 'Engineering team':")
    debug_results2 = await adapter.search("Engineering team", limit=5)
    for i, r in enumerate(debug_results2):
        print(f"    {i+1}. entities={r.get('entities', [])}, content={r.get('content', '')[:60]}...")

    print("\n[DEBUG] Testing adapter search for 'authentication':")
    debug_results3 = await adapter.search("authentication", limit=5)
    for i, r in enumerate(debug_results3):
        print(f"    {i+1}. entities={r.get('entities', [])}, content={r.get('content', '')[:60]}...")

    # Demo 1: Query Expansion
    print("\n" + "-" * 70)
    print("[3] DEMO: Query Expansion")
    print("-" * 70)

    expander = QueryExpander(llm=llm, semantic_memory=adapter)

    query = "How do other teams handle authentication?"
    user_context = {"user_name": "Doug", "team": "Engineering"}

    print(f"\n    Query: \"{query}\"")
    print(f"    User Context: {user_context}")

    import time
    start = time.perf_counter()
    result = await expander.expand(
        query=query,
        user_context=user_context,
        confidence_threshold=0.70,
        max_expansions=5,
    )
    elapsed = (time.perf_counter() - start) * 1000

    # Debug: Show graph context gathered
    print(f"\n    Graph context gathered: {result.graph_context}")

    print(f"\n    Ambiguous terms detected: {result.ambiguous_terms}")
    print(f"    Processing time: {elapsed:.0f}ms")
    print(f"\n    Generated {len(result.expansions)} expansions:")

    for i, exp in enumerate(result.expansions, 1):
        print(f"\n    {i}. \"{exp.query}\"")
        print(f"       Confidence: {exp.confidence:.2f}")
        print(f"       Target entities: {exp.target_entities}")
        print(f"       Reasoning: {exp.reasoning[:80]}...")

    # Demo 2: Parallel Retrieval + RRF Merging
    print("\n" + "-" * 70)
    print("[4] DEMO: Parallel Retrieval with Weighted RRF")
    print("-" * 70)

    # Retrieve for each expansion
    results_per_query = []
    for exp in result.expansions:
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
        print(f"\n    Retrieved {len(observations)} results for: \"{exp.query[:50]}...\"")

    # Merge with RRF
    merger = WeightedRRFMerger()
    merged = merger.merge(results_per_query, result.expansions)

    print(f"\n    Merged to {len(merged)} unique results with RRF scores:")
    for i, r in enumerate(merged[:5], 1):
        print(f"\n    {i}. RRF Score: {r.rrf_score:.4f}")
        print(f"       Content: {r.observation.content[:80]}...")
        print(f"       Contributing queries: {len(r.contributing_queries)}")

    # Demo 3: Full Orchestrator
    print("\n" + "-" * 70)
    print("[5] DEMO: Full HybridRetrievalOrchestrator")
    print("-" * 70)

    orchestrator_config = HybridRetrievalConfig(
        enable_query_expansion=True,
        expansion_confidence_threshold=0.70,
        max_query_expansions=4,
        use_weighted_rrf=True,
        enable_local=False,  # No local index for this demo
        enable_graph=True,
        enable_vector=False,  # Using graph only
    )

    orchestrator = HybridRetrievalOrchestrator(
        llm=llm,
        semantic_memory=adapter,
        config=orchestrator_config,
    )

    print(f"\n    Running orchestrator with query: \"{query}\"")
    start = time.perf_counter()
    orchestrator_result = await orchestrator.retrieve(
        query=query,
        user_context=user_context,
    )
    total_elapsed = (time.perf_counter() - start) * 1000

    print(f"\n    RESULTS:")
    print(f"    --------")
    print(f"    Total time: {orchestrator_result.total_time_ms:.0f}ms")
    print(f"    Expansion time: {orchestrator_result.expansion_time_ms:.0f}ms")
    print(f"    Graph time: {orchestrator_result.graph_time_ms:.0f}ms")
    print(f"\n    Ambiguous terms resolved: {orchestrator_result.ambiguous_terms_resolved}")
    print(f"    Expansions used: {len(orchestrator_result.expansions_used)}")
    for i, (exp, conf) in enumerate(zip(orchestrator_result.expansions_used, orchestrator_result.expansion_confidences), 1):
        print(f"      {i}. ({conf:.2f}) {exp[:60]}...")

    print(f"\n    Observations retrieved: {len(orchestrator_result.observations)}")
    print(f"    Graph observations: {orchestrator_result.graph_observations}")

    print(f"\n    SYNTHESIZED ANSWER:")
    print(f"    " + "-" * 50)
    # Word wrap the answer
    answer = orchestrator_result.answer
    for line in [answer[i:i+66] for i in range(0, len(answer), 66)]:
        print(f"    {line}")

    # Demo 4: Compare with/without expansion
    print("\n" + "-" * 70)
    print("[6] DEMO: A/B Comparison (With vs Without Expansion)")
    print("-" * 70)

    # Without expansion
    no_expand_config = HybridRetrievalConfig(
        enable_query_expansion=False,
        enable_local=False,
        enable_graph=True,
        enable_vector=False,
    )
    no_expand_orchestrator = HybridRetrievalOrchestrator(
        llm=llm,
        semantic_memory=adapter,
        config=no_expand_config,
    )

    print(f"\n    Query: \"{query}\"")

    # With expansion (already ran)
    print(f"\n    WITH EXPANSION:")
    print(f"      - Expansions: {len(orchestrator_result.expansions_used)}")
    print(f"      - Observations: {len(orchestrator_result.observations)}")
    print(f"      - Time: {orchestrator_result.total_time_ms:.0f}ms")

    # Without expansion
    start = time.perf_counter()
    no_expand_result = await no_expand_orchestrator.retrieve(query=query)

    print(f"\n    WITHOUT EXPANSION:")
    print(f"      - Expansions: 0 (query used literally)")
    print(f"      - Observations: {len(no_expand_result.observations)}")
    print(f"      - Time: {no_expand_result.total_time_ms:.0f}ms")

    # Cleanup
    await provider.close()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
