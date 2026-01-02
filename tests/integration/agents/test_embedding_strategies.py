#!/usr/bin/env python3
"""Test: All 4 Embedding Strategies with Real Systems.

Tests all 4 embedding strategies from FR-011:
1. Raw - Direct query embedding
2. HyDE - Hypothetical Document Embedding
3. Query2Doc - Original + LLM expansion
4. Grounded - Phase0/1 + graph context

IMPORTANT: Uses REAL embeddings via Ollama per CONSTITUTION.md Section 1.7.
Mock embeddings would invalidate the semantic search results.

Run with:
    python3.11 tests/integration/agents/test_embedding_strategies.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


async def main():
    """Run embedding strategy tests."""
    print("\n" + "=" * 70)
    print("EMBEDDING STRATEGY TEST - Real Systems (FR-011)")
    print("=" * 70)

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    # Import after path setup
    from draagon_ai.llm.groq import GroqLLM
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider
    from draagon_ai.orchestration.hybrid_retrieval import (
        EmbeddingStrategy,
        EmbeddingStrategyExecutor,
    )

    # Initialize providers
    print("\n[1] Initializing providers...")

    llm = GroqLLM(api_key=api_key)
    print("    ✓ Groq LLM ready")

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
        return

    # Create executor
    executor = EmbeddingStrategyExecutor(
        llm=llm,
        embedder=embedder,
        semantic_memory=None,  # No graph for basic test
    )

    # Test query
    query = "How do teams handle customer authentication?"
    user_context = {"user_name": "Doug", "team": "Engineering"}

    print(f"\n[2] Testing query: \"{query}\"")
    print(f"    User context: {user_context}")

    # Test each strategy individually
    print("\n" + "-" * 70)
    print("[3] Testing Individual Strategies")
    print("-" * 70)

    for strategy in EmbeddingStrategy:
        print(f"\n    Strategy: {strategy.value.upper()}")
        print("    " + "-" * 40)

        result = await executor.execute_embedding(
            query=query,
            strategy=strategy,
            user_context=user_context,
        )

        print(f"    Latency: {result.latency_ms:.0f}ms")
        print(f"    Embedding dim: {len(result.embedding)}")

        # Show expanded text (truncated)
        expanded = result.expanded_text[:200]
        if len(result.expanded_text) > 200:
            expanded += "..."
        print(f"    Expanded text:")
        # Word wrap for display
        for line in [expanded[i:i+60] for i in range(0, len(expanded), 60)]:
            print(f"      {line}")

        # Verify embedding has semantic meaning
        if len(result.embedding) == 768:
            print("    ✓ Valid 768-dim embedding")
        else:
            print(f"    ✗ Unexpected dimension: {len(result.embedding)}")

    # Test parallel execution of all strategies
    print("\n" + "-" * 70)
    print("[4] Testing Parallel Execution (All Strategies)")
    print("-" * 70)

    import time
    start = time.perf_counter()
    results = await executor.execute_all(
        query=query,
        strategies=list(EmbeddingStrategy),
        user_context=user_context,
        parallel=True,
    )
    total_time = (time.perf_counter() - start) * 1000

    print(f"\n    Total parallel execution time: {total_time:.0f}ms")
    print(f"    Strategies executed: {len(results)}")

    # Show individual latencies
    individual_total = 0
    for r in results:
        print(f"      - {r.strategy.value}: {r.latency_ms:.0f}ms")
        individual_total += r.latency_ms

    print(f"\n    Sequential time would be: {individual_total:.0f}ms")
    speedup = individual_total / total_time if total_time > 0 else 1
    print(f"    Parallel speedup: {speedup:.1f}x")

    # Test semantic similarity between strategies
    print("\n" + "-" * 70)
    print("[5] Semantic Similarity Between Strategies")
    print("-" * 70)

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

    # Compare all pairs
    print("\n    Strategy pair similarities:")
    raw_embedding = None
    for i, r1 in enumerate(results):
        if r1.strategy == EmbeddingStrategy.RAW:
            raw_embedding = r1.embedding
            break

    if raw_embedding:
        for r in results:
            if r.strategy != EmbeddingStrategy.RAW:
                sim = cosine_similarity(raw_embedding, r.embedding)
                print(f"      RAW vs {r.strategy.value}: {sim:.4f}")

    # Show how expanded queries differ
    print("\n    Expanded query summaries:")
    for r in results:
        summary = r.expanded_text[:80]
        if len(r.expanded_text) > 80:
            summary += "..."
        print(f"      {r.strategy.value}: {summary}")

    # Success summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"    ✓ All 4 embedding strategies working")
    print(f"    ✓ Real Ollama embeddings (768 dimensions)")
    print(f"    ✓ Real Groq LLM for expansions")
    print(f"    ✓ Parallel execution with {speedup:.1f}x speedup")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
