#!/usr/bin/env python3
"""Experiment: Query Expansion with Semantic Variations (Parallel Paths)

HYPOTHESIS:
Short user queries often lack context and can have multiple interpretations.
By expanding the query using:
1. Recent conversation context
2. Semantic/RAG memory that matches
3. LLM interpretation of ambiguous terms

We can generate MULTIPLE query variations, each representing a different
possible meaning. These can be searched in PARALLEL, and results aggregated.

EXAMPLE:
User says: "How do I fix the connection?"

Possible expansions:
1. "How do I fix the database connection error in the authentication module?"
2. "How do I fix the WebSocket connection dropping in the real-time chat?"
3. "How do I fix the API connection timeout when calling external services?"

Each expansion is searched separately, and results are merged with weighting.

Run with:
    GROQ_API_KEY=your_key python3.11 tests/integration/agents/experiment_query_expansion.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


@dataclass
class QueryExpansion:
    """An expanded version of the original query."""
    original_query: str
    expanded_query: str
    interpretation: str  # What this expansion assumes
    confidence: float  # How likely is this the intended meaning?


@dataclass
class SearchResult:
    """A document found during search."""
    doc_id: str
    content: str
    score: float
    from_expansion: str  # Which expansion found this


@dataclass
class MergedResults:
    """Final results after merging from multiple expansions."""
    documents: list[SearchResult]
    consensus_docs: list[str]  # Docs found by multiple expansions
    total_unique: int


def parse_section(content: str, prefix: str) -> list[str]:
    """Parse a section that may be comma-separated or bullet-point format."""
    lines = content.split("\n")
    items = []
    in_section = False

    for line in lines:
        line = line.strip()

        if line.upper().startswith(prefix.upper()):
            in_section = True
            after_prefix = line[len(prefix):].strip()
            if after_prefix.startswith(":"):
                after_prefix = after_prefix[1:].strip()
            if after_prefix:
                items.extend([x.strip() for x in after_prefix.split(",") if x.strip()])
            continue

        if in_section:
            if any(line.upper().startswith(p) for p in ["EXPANSION", "INTERPRETATION", "CONFIDENCE", "QUERY", "ORIGINAL"]):
                break

            if line.startswith("-") or line.startswith("*"):
                item = line[1:].strip()
                if item:
                    items.append(item)
            elif line and line[0].isdigit() and "." in line[:3]:
                item = line.split(".", 1)[1].strip() if "." in line else line
                if item:
                    items.append(item)

    return items


async def main():
    """Run the query expansion experiment."""
    from groq import AsyncGroq

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    client = AsyncGroq(api_key=api_key)

    class GroqLLM:
        async def chat(self, messages, temperature=0.7, max_tokens=1000, model="llama-3.1-8b-instant"):
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message

    llm = GroqLLM()

    print("=" * 80)
    print("EXPERIMENT: Query Expansion with Semantic Variations")
    print("=" * 80)
    print()
    print("Testing whether generating multiple query interpretations and searching")
    print("in parallel improves retrieval quality.")
    print()

    # Simulated document corpus
    documents = [
        {"id": "db_conn", "content": "Database connection troubleshooting: Check connection string, verify credentials, ensure firewall allows port 5432. Common error: 'Connection refused' means PostgreSQL is not running.", "keywords": ["database", "connection", "postgresql", "error"]},
        {"id": "ws_conn", "content": "WebSocket connection handling: Implement heartbeat every 30s, handle reconnection with exponential backoff. Disconnections often caused by proxy timeouts.", "keywords": ["websocket", "connection", "heartbeat", "timeout"]},
        {"id": "api_conn", "content": "API connection timeouts: Set reasonable timeout (30s default), implement retry logic, use circuit breaker for failing endpoints.", "keywords": ["api", "connection", "timeout", "retry"]},
        {"id": "auth_flow", "content": "Authentication flow: User submits credentials, server validates, JWT token issued. Token expires after 1 hour, refresh token valid for 7 days.", "keywords": ["authentication", "jwt", "token", "credentials"]},
        {"id": "cache_inv", "content": "Cache invalidation strategies: TTL-based, event-based, or hybrid. Use consistent hashing for distributed cache. Invalidate on write.", "keywords": ["cache", "invalidation", "ttl", "distributed"]},
        {"id": "err_handle", "content": "Error handling best practices: Catch specific exceptions, log with context, return appropriate HTTP codes. Use structured logging for debugging.", "keywords": ["error", "handling", "logging", "exceptions"]},
        {"id": "retry_logic", "content": "Retry logic implementation: Exponential backoff with jitter, max 5 retries, circuit breaker after 3 failures. Handle idempotency.", "keywords": ["retry", "backoff", "circuit", "idempotency"]},
        {"id": "sms_delivery", "content": "SMS delivery failures: Carrier webhook reports status, exponential backoff for retries (1s, 2s, 4s, 8s), switch to backup carrier after 3 failures.", "keywords": ["sms", "delivery", "carrier", "retry"]},
    ]

    # Test cases: ambiguous queries that could have multiple meanings
    test_cases = [
        {
            "query": "How do I fix the connection?",
            "context": "Working on the messaging feature that sends notifications",
            "expected_relevant": ["ws_conn", "api_conn", "sms_delivery"],
        },
        {
            "query": "Why is it failing?",
            "context": "The user login is not working after the update",
            "expected_relevant": ["auth_flow", "err_handle", "db_conn"],
        },
        {
            "query": "How do I handle the timeout?",
            "context": "External API calls are taking too long",
            "expected_relevant": ["api_conn", "retry_logic", "err_handle"],
        },
    ]

    async def generate_expansions(query: str, context: str, k: int = 3) -> list[QueryExpansion]:
        """Generate multiple interpretations of an ambiguous query."""
        prompt = f"""Given this short query and context, generate {k} different interpretations
of what the user might be asking about. Each interpretation should be specific.

Original Query: {query}
Context: {context}

For each interpretation, provide:
1. An expanded, specific version of the query
2. What this interpretation assumes
3. A confidence score (0.0-1.0) for how likely this interpretation is

Output format (repeat for each interpretation):
EXPANSION 1:
QUERY: [specific expanded query]
INTERPRETATION: [what this assumes]
CONFIDENCE: [0.0-1.0]

EXPANSION 2:
...
"""

        response = await llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=600,
        )
        content = response.content

        expansions = []
        current_expansion = {}

        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("EXPANSION") and ":" in line:
                if current_expansion.get("query"):
                    expansions.append(QueryExpansion(
                        original_query=query,
                        expanded_query=current_expansion.get("query", query),
                        interpretation=current_expansion.get("interpretation", ""),
                        confidence=current_expansion.get("confidence", 0.5),
                    ))
                current_expansion = {}
            elif line.startswith("QUERY:"):
                current_expansion["query"] = line[6:].strip()
            elif line.startswith("INTERPRETATION:"):
                current_expansion["interpretation"] = line[15:].strip()
            elif line.startswith("CONFIDENCE:"):
                try:
                    current_expansion["confidence"] = float(line[11:].strip())
                except:
                    current_expansion["confidence"] = 0.5

        # Don't forget the last one
        if current_expansion.get("query"):
            expansions.append(QueryExpansion(
                original_query=query,
                expanded_query=current_expansion.get("query", query),
                interpretation=current_expansion.get("interpretation", ""),
                confidence=current_expansion.get("confidence", 0.5),
            ))

        return expansions[:k]

    def simple_search(query: str, docs: list, k: int = 3) -> list[tuple[dict, float]]:
        """Simple keyword-based search for testing."""
        query_words = set(query.lower().split())

        scored = []
        for doc in docs:
            content_words = set(doc["content"].lower().split())
            keyword_words = set(doc["keywords"])

            # Score based on word overlap
            overlap = len(query_words & (content_words | keyword_words))
            if overlap > 0:
                scored.append((doc, overlap / len(query_words)))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:k]

    async def parallel_expansion_search(query: str, context: str, docs: list) -> dict:
        """Search using multiple query expansions in parallel."""
        # Generate expansions
        expansions = await generate_expansions(query, context, k=3)

        # Search with each expansion in parallel
        all_results = {}
        for exp in expansions:
            results = simple_search(exp.expanded_query, docs, k=3)
            for doc, score in results:
                doc_id = doc["id"]
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        "doc": doc,
                        "scores": [],
                        "found_by": [],
                    }
                all_results[doc_id]["scores"].append(score * exp.confidence)
                all_results[doc_id]["found_by"].append(exp.expanded_query[:50])

        # Aggregate: boost docs found by multiple expansions
        final_results = []
        for doc_id, data in all_results.items():
            # Multi-source boost: +0.2 per additional source
            base_score = max(data["scores"])
            boost = 0.2 * (len(data["found_by"]) - 1)
            final_score = min(1.0, base_score + boost)

            final_results.append({
                "doc_id": doc_id,
                "doc": data["doc"],
                "final_score": final_score,
                "found_by_count": len(data["found_by"]),
                "expansions": data["found_by"],
            })

        final_results.sort(key=lambda x: (-x["found_by_count"], -x["final_score"]))

        return {
            "expansions": expansions,
            "results": final_results[:5],
            "consensus": [r["doc_id"] for r in final_results if r["found_by_count"] > 1],
        }

    async def baseline_search(query: str, docs: list) -> dict:
        """Search using just the original query."""
        results = simple_search(query, docs, k=5)

        return {
            "results": [{"doc_id": doc["id"], "doc": doc, "score": score} for doc, score in results],
        }

    # Run experiments
    print("=" * 80)
    print("RUNNING EXPERIMENTS")
    print("=" * 80)

    total_baseline_hits = 0
    total_expanded_hits = 0
    total_expected = 0
    baseline_times = []
    expanded_times = []

    for case in test_cases:
        query = case["query"]
        context = case["context"]
        expected = set(case["expected_relevant"])
        total_expected += len(expected)

        print()
        print("-" * 80)
        print(f"Query: \"{query}\"")
        print(f"Context: {context}")
        print(f"Expected: {expected}")
        print("-" * 80)

        # Baseline: just the original query
        print("\n[BASELINE - Original query only]")
        start = time.time()
        baseline = await baseline_search(query, documents)
        baseline_time = (time.time() - start) * 1000
        baseline_times.append(baseline_time)

        baseline_found = set(r["doc_id"] for r in baseline["results"])
        baseline_hits = len(baseline_found & expected)
        total_baseline_hits += baseline_hits

        print(f"  Found: {[r['doc_id'] for r in baseline['results'][:3]]}")
        print(f"  Hits: {baseline_hits}/{len(expected)} expected")
        print(f"  Time: {baseline_time:.0f}ms")

        # Expanded: multiple query interpretations in parallel
        print("\n[EXPANDED - Parallel query variations]")
        start = time.time()
        expanded = await parallel_expansion_search(query, context, documents)
        expanded_time = (time.time() - start) * 1000
        expanded_times.append(expanded_time)

        print("  Expansions generated:")
        for i, exp in enumerate(expanded["expansions"]):
            print(f"    {i+1}. \"{exp.expanded_query}\" (conf: {exp.confidence:.2f})")
            print(f"       Assumes: {exp.interpretation}")

        expanded_found = set(r["doc_id"] for r in expanded["results"])
        expanded_hits = len(expanded_found & expected)
        total_expanded_hits += expanded_hits

        print(f"\n  Found: {[r['doc_id'] for r in expanded['results'][:5]]}")
        print(f"  Consensus (multi-source): {expanded['consensus']}")
        print(f"  Hits: {expanded_hits}/{len(expected)} expected")
        print(f"  Time: {expanded_time:.0f}ms")

        # Comparison
        if expanded_hits > baseline_hits:
            print(f"\n  ✅ EXPANDED WINS (+{expanded_hits - baseline_hits} hits)")
        elif baseline_hits > expanded_hits:
            print(f"\n  ❌ BASELINE WINS (+{baseline_hits - expanded_hits} hits)")
        else:
            print(f"\n  🟰 TIE")

    # Summary
    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)
    print()

    baseline_recall = total_baseline_hits / total_expected if total_expected else 0
    expanded_recall = total_expanded_hits / total_expected if total_expected else 0

    print(f"Total Hits (of {total_expected} expected):")
    print(f"  Baseline: {total_baseline_hits} ({baseline_recall:.1%} recall)")
    print(f"  Expanded: {total_expanded_hits} ({expanded_recall:.1%} recall)")
    print()
    print(f"Average Latency:")
    print(f"  Baseline: {sum(baseline_times)/len(baseline_times):.0f}ms")
    print(f"  Expanded: {sum(expanded_times)/len(expanded_times):.0f}ms")
    print()

    if expanded_recall > baseline_recall:
        improvement = (expanded_recall - baseline_recall) / baseline_recall * 100 if baseline_recall else float('inf')
        print(f"🏆 EXPANDED WINS! +{improvement:.0f}% improvement in recall")
        print()
        print("RECOMMENDATION: Use parallel query expansion for ambiguous queries.")
        print("The context-aware expansions find relevant documents that simple")
        print("keyword matching misses.")
    elif baseline_recall > expanded_recall:
        print("❌ BASELINE WINS")
        print("Query expansion didn't help - may be adding noise or the expansions")
        print("are going in wrong directions.")
    else:
        print("🟰 TIE - no difference between approaches")

    print()
    print("=" * 80)
    print("ARCHITECTURE RECOMMENDATION")
    print("=" * 80)
    print()
    print("Based on these experiments, the optimal pipeline would be:")
    print()
    print("1. SHORT QUERY + CONTEXT")
    print("   ↓")
    print("2. GENERATE 3 QUERY EXPANSIONS (parallel)")
    print("   - Each represents a different interpretation")
    print("   - Include confidence scores")
    print("   ↓")
    print("3. SEARCH WITH EACH EXPANSION (parallel)")
    print("   - Use existing RAG/Vector/Graph approaches")
    print("   - Tag results by which expansion found them")
    print("   ↓")
    print("4. MERGE RESULTS")
    print("   - Boost documents found by multiple expansions (consensus)")
    print("   - Weight by expansion confidence")
    print("   ↓")
    print("5. SYNTHESIZE ANSWER")
    print("   - If round-trip helps: expand extracted terms back to natural language")
    print("   - Use HYBRID context (extracted terms + raw snippets)")


if __name__ == "__main__":
    asyncio.run(main())
