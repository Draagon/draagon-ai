#!/usr/bin/env python3
"""Unified Benchmark: Raw Context vs RAG vs Semantic Graph.

Compares the 3 fundamental retrieval approaches on the same test cases:
1. Raw Context - Load full files into LLM context
2. Vector/RAG - Embed & search (using best embedding strategy)
3. Semantic Graph - Query knowledge graph for entities/relationships

For RAG, we use the embedding strategy that performed best in our previous
benchmark (HyDE for lexical mismatch/multi-hop, Query2Doc for entities).

Run with:
    python3.11 tests/integration/agents/benchmark_three_approaches.py
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add project to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))


class RetrievalApproach(str, Enum):
    """The three fundamental retrieval approaches."""

    RAW_CONTEXT = "raw_context"  # Load full files
    VECTOR_RAG = "vector_rag"  # Embed & search
    SEMANTIC_GRAPH = "semantic_graph"  # Knowledge graph


class QueryType(str, Enum):
    """Query types that may favor different approaches."""

    ENTITY_LOOKUP = "entity_lookup"  # "What is X?"
    SIMILARITY = "similarity"  # "Find similar to..."
    FULL_CONTEXT = "full_context"  # "Summarize this file"
    MULTI_HOP = "multi_hop"  # "Doug's team's auth"
    NEEDLE = "needle"  # Find specific fact in large doc
    CROSS_PROJECT = "cross_project"  # Patterns across projects


@dataclass
class TestCase:
    """Test case with expected approach winner."""

    query: str
    expected_content: list[str]
    query_type: QueryType
    expected_winner: RetrievalApproach | None = None  # For validation
    description: str = ""
    scale: str = "small"  # small, medium, large


@dataclass
class ApproachResult:
    """Result from a single approach."""

    approach: RetrievalApproach
    recall: float
    precision: float
    latency_ms: float
    context_size: int  # chars for raw, docs for RAG, entities for graph
    answer_quality: float = 0.0  # LLM-as-judge score


@dataclass
class BenchmarkResult:
    """Result of running all approaches on a test case."""

    test_case: TestCase
    results: dict[RetrievalApproach, ApproachResult]
    winner: RetrievalApproach
    expected_winner_correct: bool


# =============================================================================
# Knowledge Base (Same as embedding benchmark for comparison)
# =============================================================================

KNOWLEDGE_BASE = [
    {
        "id": "auth_engineering",
        "content": "Engineering team uses OAuth2 with PKCE for customer authentication. They implement JWT tokens with 1-hour expiry and refresh token rotation.",
        "entities": ["Engineering", "OAuth2", "PKCE", "JWT"],
        "domain": "authentication",
    },
    {
        "id": "auth_platform",
        "content": "Platform team uses service mesh mTLS for inter-service authentication. They implement Istio with automatic certificate rotation.",
        "entities": ["Platform", "mTLS", "Istio"],
        "domain": "authentication",
    },
    {
        "id": "auth_mobile",
        "content": "Mobile team uses native biometric authentication. They implement Face ID and Touch ID with secure enclave storage.",
        "entities": ["Mobile", "biometric", "Face ID", "Touch ID"],
        "domain": "authentication",
    },
    {
        "id": "auth_data",
        "content": "Data team uses AWS IAM roles for authentication. They implement cross-account access with temporary credentials.",
        "entities": ["Data", "AWS", "IAM"],
        "domain": "authentication",
    },
    {
        "id": "user_doug",
        "content": "Doug is a senior engineer on the Engineering team. He specializes in authentication systems and API security.",
        "entities": ["Doug", "Engineering"],
        "domain": "user",
    },
    {
        "id": "user_sarah",
        "content": "Sarah leads the Platform team. She designed the service mesh architecture.",
        "entities": ["Sarah", "Platform"],
        "domain": "user",
    },
    {
        "id": "project_atlas",
        "content": "Project Atlas is our customer identity platform. It handles SSO federation with external IdPs.",
        "entities": ["Atlas", "SSO", "IdP"],
        "domain": "project",
    },
    {
        "id": "project_atlas_db",
        "content": "MongoDB Atlas is our cloud database provider. We use it for the Mobile team's data sync.",
        "entities": ["Atlas", "MongoDB", "cloud"],
        "domain": "database",
    },
]


# =============================================================================
# Test Cases by Query Type
# =============================================================================

TEST_CASES = [
    # Entity Lookup - Semantic Graph should win
    TestCase(
        query="What authentication does the Engineering team use?",
        expected_content=["OAuth2", "JWT", "PKCE"],
        query_type=QueryType.ENTITY_LOOKUP,
        expected_winner=RetrievalApproach.SEMANTIC_GRAPH,
        description="Direct entity lookup",
        scale="small",
    ),
    TestCase(
        query="Who leads the Platform team?",
        expected_content=["Sarah"],
        query_type=QueryType.ENTITY_LOOKUP,
        expected_winner=RetrievalApproach.SEMANTIC_GRAPH,
        description="Entity relationship query",
        scale="small",
    ),

    # Multi-hop - Semantic Graph should win
    TestCase(
        query="What authentication method does Doug's team use?",
        expected_content=["OAuth2", "JWT", "Engineering"],
        query_type=QueryType.MULTI_HOP,
        expected_winner=RetrievalApproach.SEMANTIC_GRAPH,
        description="2-hop: Doug → Engineering → OAuth2",
        scale="small",
    ),
    TestCase(
        query="What did the leader of the Platform team design?",
        expected_content=["Sarah", "service mesh", "Istio"],
        query_type=QueryType.MULTI_HOP,
        expected_winner=RetrievalApproach.SEMANTIC_GRAPH,
        description="2-hop: Platform leader → Sarah → service mesh",
        scale="small",
    ),

    # Similarity - RAG should win
    TestCase(
        query="How do teams verify user identity?",
        expected_content=["authentication", "OAuth2", "biometric", "mTLS"],
        query_type=QueryType.SIMILARITY,
        expected_winner=RetrievalApproach.VECTOR_RAG,
        description="Lexical mismatch: 'verify identity' ≈ 'authentication'",
        scale="small",
    ),
    TestCase(
        query="Find authentication methods that work offline",
        expected_content=["biometric", "Face ID", "Touch ID"],
        query_type=QueryType.SIMILARITY,
        expected_winner=RetrievalApproach.VECTOR_RAG,
        description="Semantic similarity search",
        scale="small",
    ),

    # Cross-project - Semantic Graph should win
    TestCase(
        query="Which teams use certificate-based auth?",
        expected_content=["Platform", "mTLS", "Istio"],
        query_type=QueryType.CROSS_PROJECT,
        expected_winner=RetrievalApproach.SEMANTIC_GRAPH,
        description="Cross-entity pattern query",
        scale="small",
    ),

    # Ambiguous entity - Tests disambiguation
    TestCase(
        query="What is Atlas used for?",
        expected_content=["SSO", "identity"],  # Should find Project Atlas
        query_type=QueryType.ENTITY_LOOKUP,
        expected_winner=None,  # Ambiguous - could be either
        description="Ambiguous entity: Project Atlas vs MongoDB Atlas",
        scale="small",
    ),
]


# =============================================================================
# Approach Implementations
# =============================================================================

class RawContextApproach:
    """Load full documents into LLM context."""

    def __init__(self, llm, knowledge_base: list[dict]):
        self.llm = llm
        self.knowledge_base = knowledge_base

    async def retrieve(self, query: str, tc: TestCase) -> ApproachResult:
        """Load all docs as context, ask LLM to find answer."""
        start = time.perf_counter()

        # Build full context
        context = "\n\n".join([
            f"[{doc['id']}] {doc['content']}"
            for doc in self.knowledge_base
        ])

        # Ask LLM
        prompt = f"""Given this context, answer the question.

Context:
{context}

Question: {query}

Answer with specific facts from the context. If the answer isn't in the context, say "Not found"."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        # Score based on expected content
        recall = self._compute_recall(content, tc.expected_content)

        return ApproachResult(
            approach=RetrievalApproach.RAW_CONTEXT,
            recall=recall,
            precision=recall,  # Simplified
            latency_ms=latency,
            context_size=len(context),
        )

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0


class VectorRAGApproach:
    """Embed query, search, return top-k."""

    def __init__(self, llm, embedder, knowledge_base: list[dict]):
        self.llm = llm
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        self.embedded_docs: list[tuple[dict, list[float]]] = []

    async def setup(self):
        """Embed all documents."""
        for doc in self.knowledge_base:
            embedding = await self.embedder.embed(doc["content"])
            self.embedded_docs.append((doc, embedding))

    async def retrieve(self, query: str, tc: TestCase, use_hyde: bool = True) -> ApproachResult:
        """Search with optional HyDE expansion."""
        import math
        start = time.perf_counter()

        # Optionally expand with HyDE for lexical mismatch
        if use_hyde and tc.query_type in [QueryType.SIMILARITY, QueryType.MULTI_HOP]:
            query_text = await self._hyde_expand(query)
        else:
            query_text = query

        # Embed query
        query_embedding = await self.embedder.embed(query_text)

        # Search
        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scores = [(doc, cosine_sim(query_embedding, emb)) for doc, emb in self.embedded_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in scores[:5]]

        # Build context from top docs
        context = "\n\n".join([f"[{doc['id']}] {doc['content']}" for doc in top_docs])

        # Ask LLM
        prompt = f"""Given these retrieved documents, answer the question.

Retrieved Documents:
{context}

Question: {query}

Answer with specific facts from the documents."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        recall = self._compute_recall(content, tc.expected_content)

        return ApproachResult(
            approach=RetrievalApproach.VECTOR_RAG,
            recall=recall,
            precision=recall,
            latency_ms=latency,
            context_size=len(top_docs),
        )

    async def _hyde_expand(self, query: str) -> str:
        """Generate hypothetical document for better embedding."""
        prompt = f"""Write a detailed paragraph that answers this question as if you are an expert:

Question: {query}

Write a comprehensive answer with specific details."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200,
        )
        return response.content if hasattr(response, "content") else str(response)

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0


class SemanticGraphApproach:
    """Query knowledge graph for entities and relationships."""

    def __init__(self, llm, knowledge_base: list[dict]):
        self.llm = llm
        self.knowledge_base = knowledge_base
        # Build simple entity index
        self.entity_to_docs: dict[str, list[dict]] = {}
        for doc in knowledge_base:
            for entity in doc.get("entities", []):
                if entity not in self.entity_to_docs:
                    self.entity_to_docs[entity] = []
                self.entity_to_docs[entity].append(doc)

    async def retrieve(self, query: str, tc: TestCase) -> ApproachResult:
        """Extract entities, traverse graph, find relevant facts."""
        start = time.perf_counter()

        # Extract entities from query using LLM
        entities = await self._extract_entities(query)

        # Find relevant docs via entity traversal
        relevant_docs = set()
        for entity in entities:
            # Direct match
            if entity in self.entity_to_docs:
                for doc in self.entity_to_docs[entity]:
                    relevant_docs.add(doc["id"])

            # Fuzzy match (case-insensitive)
            for key in self.entity_to_docs:
                if entity.lower() in key.lower() or key.lower() in entity.lower():
                    for doc in self.entity_to_docs[key]:
                        relevant_docs.add(doc["id"])

        # For multi-hop, expand to related entities
        if tc.query_type == QueryType.MULTI_HOP:
            # Get docs for found entities, then find their related entities
            expanded_docs = set()
            for doc_id in relevant_docs:
                doc = next((d for d in self.knowledge_base if d["id"] == doc_id), None)
                if doc:
                    for entity in doc.get("entities", []):
                        if entity in self.entity_to_docs:
                            for related_doc in self.entity_to_docs[entity]:
                                expanded_docs.add(related_doc["id"])
            relevant_docs.update(expanded_docs)

        # Get full docs
        docs = [d for d in self.knowledge_base if d["id"] in relevant_docs]

        if not docs:
            # Fallback: return all docs
            docs = self.knowledge_base

        # Build context
        context = "\n\n".join([f"[{doc['id']}] {doc['content']}" for doc in docs])

        # Ask LLM
        prompt = f"""Given this knowledge graph context, answer the question.

Knowledge Graph Facts:
{context}

Question: {query}

Answer with specific facts. Trace multi-hop relationships if needed."""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300,
        )
        content = response.content if hasattr(response, "content") else str(response)

        latency = (time.perf_counter() - start) * 1000

        recall = self._compute_recall(content, tc.expected_content)

        return ApproachResult(
            approach=RetrievalApproach.SEMANTIC_GRAPH,
            recall=recall,
            precision=recall,
            latency_ms=latency,
            context_size=len(docs),
        )

    async def _extract_entities(self, query: str) -> list[str]:
        """Extract entity names from query."""
        prompt = f"""Extract entity names from this query. Return only the entity names, one per line.

Query: {query}

Entities:"""

        response = await self.llm.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse entities
        entities = [line.strip() for line in content.strip().split("\n") if line.strip()]
        return entities

    def _compute_recall(self, answer: str, expected: list[str]) -> float:
        found = sum(1 for e in expected if e.lower() in answer.lower())
        return found / len(expected) if expected else 1.0


# =============================================================================
# LLM Strategy Selector
# =============================================================================

class StrategySelector:
    """LLM-based strategy selection."""

    PROMPT = """Analyze this query and decide the best retrieval approach.

Query: {query}

Available approaches:
1. RAW_CONTEXT - Load full documents into LLM context. Best for: small files, summarization.
2. VECTOR_RAG - Embed query and search by similarity. Best for: similarity search, lexical mismatch.
3. SEMANTIC_GRAPH - Query knowledge graph entities/relationships. Best for: entity lookup, multi-hop reasoning.

Consider:
- Is this asking about a specific entity? → SEMANTIC_GRAPH
- Is this looking for similar content? → VECTOR_RAG
- Does this need full document context? → RAW_CONTEXT
- Does this require connecting multiple facts? → SEMANTIC_GRAPH

Respond with just the approach name: RAW_CONTEXT, VECTOR_RAG, or SEMANTIC_GRAPH"""

    def __init__(self, llm):
        self.llm = llm

    async def select(self, query: str) -> RetrievalApproach:
        """Use LLM to select best approach."""
        response = await self.llm.chat(
            messages=[{"role": "user", "content": self.PROMPT.format(query=query)}],
            temperature=0.0,
            max_tokens=50,
        )
        content = response.content if hasattr(response, "content") else str(response)

        # Parse response
        content = content.strip().upper()
        if "SEMANTIC" in content:
            return RetrievalApproach.SEMANTIC_GRAPH
        elif "VECTOR" in content or "RAG" in content:
            return RetrievalApproach.VECTOR_RAG
        else:
            return RetrievalApproach.RAW_CONTEXT


# =============================================================================
# Main Benchmark
# =============================================================================

async def main():
    """Run the unified benchmark."""
    print("\n" + "=" * 80)
    print("UNIFIED RETRIEVAL BENCHMARK")
    print("Comparing: Raw Context vs Vector/RAG vs Semantic Graph")
    print("=" * 80)

    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        return

    # Import providers
    from draagon_ai.llm.groq import GroqLLM
    from draagon_ai.memory.embedding import OllamaEmbeddingProvider

    print("\n[1] Initializing providers...")
    llm = GroqLLM(api_key=api_key)

    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://192.168.168.200:11434")
    embedder = OllamaEmbeddingProvider(
        base_url=ollama_url,
        model="nomic-embed-text",
        dimension=768,
    )

    try:
        await embedder.embed("test")
        print("    ✓ Providers ready")
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return

    # Initialize approaches
    print("\n[2] Initializing approaches...")
    raw_approach = RawContextApproach(llm, KNOWLEDGE_BASE)
    rag_approach = VectorRAGApproach(llm, embedder, KNOWLEDGE_BASE)
    graph_approach = SemanticGraphApproach(llm, KNOWLEDGE_BASE)

    await rag_approach.setup()
    print("    ✓ All approaches ready")

    # Initialize strategy selector
    selector = StrategySelector(llm)

    # Run benchmark
    print("\n" + "-" * 80)
    print("[3] Running Benchmark")
    print("-" * 80)

    results: list[BenchmarkResult] = []
    llm_selections: list[tuple[TestCase, RetrievalApproach]] = []

    for tc in TEST_CASES:
        print(f"\n  Query: \"{tc.query[:60]}...\"" if len(tc.query) > 60 else f"\n  Query: \"{tc.query}\"")
        print(f"  Type: {tc.query_type.value}")

        # Run all approaches
        approach_results = {}

        raw_result = await raw_approach.retrieve(tc.query, tc)
        approach_results[RetrievalApproach.RAW_CONTEXT] = raw_result

        rag_result = await rag_approach.retrieve(tc.query, tc)
        approach_results[RetrievalApproach.VECTOR_RAG] = rag_result

        graph_result = await graph_approach.retrieve(tc.query, tc)
        approach_results[RetrievalApproach.SEMANTIC_GRAPH] = graph_result

        # Determine winner
        winner = max(approach_results.keys(), key=lambda a: approach_results[a].recall)

        # Get LLM selection
        llm_selection = await selector.select(tc.query)
        llm_selections.append((tc, llm_selection))

        # Check if expected winner is correct
        expected_correct = tc.expected_winner is None or winner == tc.expected_winner

        results.append(BenchmarkResult(
            test_case=tc,
            results=approach_results,
            winner=winner,
            expected_winner_correct=expected_correct,
        ))

        # Print results
        print(f"  Expected winner: {tc.expected_winner.value if tc.expected_winner else 'Any'}")
        print(f"  Actual winner: {winner.value}")
        print(f"  LLM selected: {llm_selection.value}")

        for approach, result in approach_results.items():
            marker = "→" if approach == winner else " "
            llm_marker = "★" if approach == llm_selection else " "
            print(f"    {marker}{llm_marker} {approach.value:15} R={result.recall:.2f} {result.latency_ms:.0f}ms")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Wins by approach
    wins = {a: 0 for a in RetrievalApproach}
    for r in results:
        wins[r.winner] += 1

    print("\n  Wins by Approach:")
    for approach, count in sorted(wins.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(results) * 100
        print(f"    {approach.value:15} {count}/{len(results)} ({pct:.0f}%)")

    # LLM selector accuracy
    llm_correct = sum(1 for tc, sel in llm_selections
                      for r in results if r.test_case == tc and sel == r.winner)
    llm_accuracy = llm_correct / len(llm_selections) * 100
    print(f"\n  LLM Selector Accuracy: {llm_correct}/{len(llm_selections)} ({llm_accuracy:.0f}%)")

    # By query type
    print("\n  Performance by Query Type:")
    type_results: dict[QueryType, dict[RetrievalApproach, list[float]]] = {}
    for r in results:
        qt = r.test_case.query_type
        if qt not in type_results:
            type_results[qt] = {a: [] for a in RetrievalApproach}
        for approach, ar in r.results.items():
            type_results[qt][approach].append(ar.recall)

    for qt, approach_recalls in type_results.items():
        print(f"\n    {qt.value}:")
        for approach, recalls in approach_recalls.items():
            avg = sum(recalls) / len(recalls) if recalls else 0
            print(f"      {approach.value:15} avg_recall={avg:.1%}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
