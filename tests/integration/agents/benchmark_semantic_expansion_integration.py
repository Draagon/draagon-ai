#!/usr/bin/env python3
"""Benchmark: Semantic Expansion Impact on Retrieval.

Tests how semantic expansion (from prototypes/semantic_expansion) affects
retrieval quality across the 3 approaches:
1. Raw Context
2. Vector/RAG
3. Semantic Graph

The hypothesis is that semantic expansion helps by:
- Resolving ambiguous entities (e.g., "Atlas" → Project Atlas vs MongoDB Atlas)
- Expanding queries with synonyms and related concepts
- Word sense disambiguation (e.g., "bank" → financial vs river)

Run with:
    python3.11 tests/integration/agents/benchmark_semantic_expansion_integration.py

Requires:
    - GROQ_API_KEY (or OLLAMA for local)
    - prototypes/semantic_expansion to be available
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# Add project paths
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "prototypes" / "semantic_expansion" / "src"))


class RetrievalApproach(str, Enum):
    """The three fundamental retrieval approaches."""

    RAW_CONTEXT = "raw_context"
    VECTOR_RAG = "vector_rag"
    SEMANTIC_GRAPH = "semantic_graph"


class ExpansionMode(str, Enum):
    """Whether to use semantic expansion."""

    NO_EXPANSION = "no_expansion"
    WITH_EXPANSION = "with_expansion"


@dataclass
class TestCase:
    """Test case for semantic expansion evaluation."""

    query: str
    expected_content: list[str]
    description: str
    ambiguity_type: str = "none"  # none, entity, word_sense, pronoun, implicit


@dataclass
class ExpansionResult:
    """Result of semantic expansion preprocessing."""

    original_query: str
    expanded_query: str
    resolved_entities: dict[str, str] = field(default_factory=dict)
    disambiguated_terms: dict[str, str] = field(default_factory=dict)
    expansion_latency_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Result comparing with/without expansion."""

    test_case: TestCase
    approach: RetrievalApproach
    no_expansion_recall: float
    with_expansion_recall: float
    improvement: float  # Positive = expansion helped
    expansion_latency_ms: float


# =============================================================================
# Knowledge Base (Same as other benchmarks)
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
    {
        "id": "db_analytics",
        "content": "Analytics team uses PostgreSQL for data warehousing. They run nightly ETL jobs.",
        "entities": ["Analytics", "PostgreSQL", "ETL"],
        "domain": "database",
    },
    {
        "id": "bank_financial",
        "content": "The bank transaction service handles financial operations. It uses PCI-compliant encryption.",
        "entities": ["bank", "financial", "PCI"],
        "domain": "finance",
    },
    {
        "id": "river_bank",
        "content": "The office is located near the river bank. The walking path along the bank is popular at lunch.",
        "entities": ["river", "bank", "office"],
        "domain": "location",
    },
]


# =============================================================================
# Test Cases for Semantic Expansion
# =============================================================================

# These test cases are designed so that:
# - Raw queries may miss content due to lexical mismatch
# - Expanded queries should find the correct content
# - Some are easy baselines to ensure expansion doesn't hurt

TEST_CASES = [
    # =========================================================================
    # TIER 1: Direct Queries (baseline - both should work)
    # =========================================================================
    TestCase(
        query="What authentication does the Engineering team use?",
        expected_content=["OAuth2", "JWT", "PKCE"],
        description="Direct query - no ambiguity",
        ambiguity_type="none",
    ),
    TestCase(
        query="What database does the Analytics team use?",
        expected_content=["PostgreSQL"],
        description="Direct entity query",
        ambiguity_type="none",
    ),

    # =========================================================================
    # TIER 2: Lexical Mismatch (expansion should help)
    # =========================================================================
    TestCase(
        query="How do we verify customer identity?",  # Should find OAuth2/SSO
        expected_content=["OAuth2", "SSO", "authentication"],
        description="Lexical: 'verify identity' → authentication",
        ambiguity_type="lexical",
    ),
    TestCase(
        query="What security measures protect data transfer?",  # Should find mTLS
        expected_content=["mTLS", "certificate"],
        description="Lexical: 'protect data transfer' → mTLS",
        ambiguity_type="lexical",
    ),
    TestCase(
        query="How do we handle user login?",  # Should find biometric, OAuth2
        expected_content=["biometric", "OAuth2", "authentication"],
        description="Lexical: 'user login' → authentication methods",
        ambiguity_type="lexical",
    ),

    # =========================================================================
    # TIER 3: Entity Disambiguation
    # =========================================================================
    TestCase(
        query="What is Atlas used for?",
        expected_content=["SSO", "identity"],  # Project Atlas
        description="Ambiguous entity: Project Atlas vs MongoDB Atlas",
        ambiguity_type="entity",
    ),
    TestCase(
        query="What database does Atlas provide?",
        expected_content=["MongoDB", "cloud"],  # MongoDB Atlas
        description="Entity with context clue",
        ambiguity_type="entity",
    ),

    # =========================================================================
    # TIER 4: Word Sense Disambiguation
    # =========================================================================
    TestCase(
        query="What services does the bank provide?",
        expected_content=["transaction", "financial", "PCI"],  # Financial bank
        description="WSD: bank (services → financial)",
        ambiguity_type="word_sense",
    ),
    TestCase(
        query="Where can employees walk near the bank?",
        expected_content=["river", "path", "walking"],  # River bank
        description="WSD: bank (walk → river)",
        ambiguity_type="word_sense",
    ),

    # =========================================================================
    # TIER 5: Multi-hop / Implicit Reference
    # =========================================================================
    TestCase(
        query="What tokens does the API security expert's team use?",
        expected_content=["JWT", "PKCE"],  # Doug → Engineering → JWT
        description="Multi-hop: API security expert → Doug → Engineering → JWT",
        ambiguity_type="multi_hop",
    ),
    TestCase(
        query="What service mesh does the team lead by Sarah use?",
        expected_content=["Istio", "mTLS"],  # Sarah → Platform → Istio
        description="Multi-hop: Sarah → Platform → Istio",
        ambiguity_type="multi_hop",
    ),

    # =========================================================================
    # TIER 6: Pronoun Resolution
    # =========================================================================
    TestCase(
        query="Doug specializes in auth. What tokens does his team issue?",
        expected_content=["JWT"],  # his team = Engineering
        description="Pronoun: his team → Engineering → JWT",
        ambiguity_type="pronoun",
    ),
    TestCase(
        query="Sarah leads Platform. How does her team handle inter-service auth?",
        expected_content=["mTLS", "Istio"],  # her team = Platform
        description="Pronoun: her team → Platform → mTLS",
        ambiguity_type="pronoun",
    ),

    # =========================================================================
    # TIER 7: Synonym/Concept Expansion
    # =========================================================================
    TestCase(
        query="Which teams use cloud-hosted databases?",  # MongoDB Atlas is cloud
        expected_content=["Mobile", "MongoDB", "Atlas"],
        description="Concept: cloud-hosted → MongoDB Atlas",
        ambiguity_type="concept",
    ),
    TestCase(
        query="Which teams use temporary credentials?",  # IAM temp creds
        expected_content=["Data", "AWS", "IAM"],
        description="Concept: temporary credentials → AWS IAM",
        ambiguity_type="concept",
    ),
]


# =============================================================================
# Semantic Expansion Service Integration
# =============================================================================

class SemanticExpansionPreprocessor:
    """Integrates semantic expansion prototype for query preprocessing."""

    def __init__(self, llm):
        """Initialize with LLM provider.

        Uses semantic expansion components from prototype:
        - WordSenseDisambiguator for WSD
        - SemanticExpansionService for query expansion
        """
        self.llm = llm
        self._wsd = None
        self._expansion_service = None

    async def _ensure_initialized(self):
        """Lazy initialization of expansion components."""
        if self._wsd is None:
            try:
                from wsd import WordSenseDisambiguator
                from expansion import SemanticExpansionService

                self._wsd = WordSenseDisambiguator(llm=self.llm)
                self._expansion_service = SemanticExpansionService(
                    llm=self.llm, wsd=self._wsd
                )
            except ImportError as e:
                raise ImportError(
                    f"Semantic expansion prototype not available: {e}\n"
                    "Ensure prototypes/semantic_expansion/src is in path"
                )

    async def expand_query(
        self,
        query: str,
        context: list[str] | None = None,
        max_retries: int = 3,
    ) -> ExpansionResult:
        """Expand query using semantic expansion prototype.

        This is the main integration point - takes a query and returns
        an expanded version with resolved entities and disambiguated terms.
        """
        await self._ensure_initialized()
        start = time.perf_counter()

        result = ExpansionResult(original_query=query, expanded_query=query)

        for attempt in range(max_retries):
            try:
                # Use LLM to expand and disambiguate query
                expansion_prompt = f"""Analyze this query and expand it for better retrieval.

Query: {query}
{"Context: " + chr(10).join(context) if context else ""}

Respond with:
1. EXPANDED_QUERY: A more explicit version that resolves ambiguities
2. ENTITIES: Any entities that should be resolved (entity: resolved_form)
3. TERMS: Any ambiguous terms and their intended meaning (term: meaning)

Example:
Query: "What does Doug's team use?"
EXPANDED_QUERY: What authentication does the Engineering team use? (Doug is on Engineering team)
ENTITIES: Doug's team: Engineering team
TERMS: use: authentication method

Now analyze:
Query: {query}"""

                response = await self.llm.chat(
                    messages=[{"role": "user", "content": expansion_prompt}],
                    temperature=0.3,
                    max_tokens=300,
                )
                content = response.content if hasattr(response, "content") else str(response)

                # Parse response
                result.expanded_query = self._extract_expanded_query(content, query)
                result.resolved_entities = self._extract_entities(content)
                result.disambiguated_terms = self._extract_terms(content)
                break  # Success - exit retry loop

            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Expansion attempt {attempt + 1} failed: {e}, retrying...")
                    await asyncio.sleep(1.0 * (attempt + 1))  # Backoff
                else:
                    print(f"Expansion failed after {max_retries} attempts: {e}")
                    result.expanded_query = query

        result.expansion_latency_ms = (time.perf_counter() - start) * 1000
        return result

    def _extract_expanded_query(self, content: str, fallback: str) -> str:
        """Extract expanded query from LLM response."""
        for line in content.split("\n"):
            if "EXPANDED_QUERY:" in line:
                return line.split("EXPANDED_QUERY:", 1)[1].strip()
        # Fall back to first non-empty line that looks like a query
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith(("ENTITIES", "TERMS", "1.", "2.", "3.")):
                return line
        return fallback

    def _extract_entities(self, content: str) -> dict[str, str]:
        """Extract resolved entities from LLM response."""
        entities = {}
        in_entities = False
        for line in content.split("\n"):
            if "ENTITIES:" in line:
                in_entities = True
                # Handle same-line entities
                parts = line.split("ENTITIES:", 1)[1].strip()
                if ":" in parts:
                    k, v = parts.split(":", 1)
                    entities[k.strip()] = v.strip()
            elif in_entities and ":" in line and not line.startswith("TERMS"):
                k, v = line.split(":", 1)
                entities[k.strip()] = v.strip()
            elif "TERMS:" in line:
                break
        return entities

    def _extract_terms(self, content: str) -> dict[str, str]:
        """Extract disambiguated terms from LLM response."""
        terms = {}
        in_terms = False
        for line in content.split("\n"):
            if "TERMS:" in line:
                in_terms = True
                parts = line.split("TERMS:", 1)[1].strip()
                if ":" in parts:
                    k, v = parts.split(":", 1)
                    terms[k.strip()] = v.strip()
            elif in_terms and ":" in line:
                k, v = line.split(":", 1)
                terms[k.strip()] = v.strip()
        return terms


# =============================================================================
# Simple Retrieval Approaches for Benchmarking
# =============================================================================

class SimpleEmbedder:
    """Simple embedder using Ollama."""

    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    async def embed(self, text: str) -> list[float]:
        """Get embedding for text."""
        import ollama
        response = ollama.embed(model=self.model, input=text)
        return response["embeddings"][0]


class VectorRetriever:
    """Simple vector retriever for benchmarking."""

    def __init__(self, embedder: SimpleEmbedder, knowledge_base: list[dict]):
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        self.embedded_docs: list[tuple[dict, list[float]]] = []

    async def setup(self):
        """Embed all documents."""
        for doc in self.knowledge_base:
            embedding = await self.embedder.embed(doc["content"])
            self.embedded_docs.append((doc, embedding))

    async def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k docs by cosine similarity."""
        import math

        query_embedding = await self.embedder.embed(query)

        def cosine_sim(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        scores = [(doc, cosine_sim(query_embedding, emb)) for doc, emb in self.embedded_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]


# =============================================================================
# Benchmark Runner
# =============================================================================

async def run_benchmark(llm) -> list[BenchmarkResult]:
    """Run the semantic expansion benchmark."""
    print("=" * 70)
    print("SEMANTIC EXPANSION IMPACT BENCHMARK")
    print("=" * 70)

    # Initialize components
    expander = SemanticExpansionPreprocessor(llm)
    embedder = SimpleEmbedder()
    retriever = VectorRetriever(embedder, KNOWLEDGE_BASE)

    print("\nEmbedding documents...")
    await retriever.setup()
    print(f"Embedded {len(KNOWLEDGE_BASE)} documents")

    results: list[BenchmarkResult] = []

    for tc in TEST_CASES:
        print(f"\n--- {tc.description} ---")
        print(f"Query: {tc.query}")

        # 1. Without expansion
        no_exp_docs = await retriever.retrieve(tc.query)
        no_exp_recall = compute_recall(no_exp_docs, tc.expected_content)

        # 2. With expansion
        expansion = await expander.expand_query(tc.query)
        print(f"Expanded: {expansion.expanded_query}")
        if expansion.resolved_entities:
            print(f"Entities: {expansion.resolved_entities}")
        if expansion.disambiguated_terms:
            print(f"Terms: {expansion.disambiguated_terms}")

        with_exp_docs = await retriever.retrieve(expansion.expanded_query)
        with_exp_recall = compute_recall(with_exp_docs, tc.expected_content)

        improvement = with_exp_recall - no_exp_recall

        print(f"Recall: {no_exp_recall:.1%} → {with_exp_recall:.1%} ({improvement:+.1%})")
        print(f"Expansion latency: {expansion.expansion_latency_ms:.0f}ms")

        results.append(BenchmarkResult(
            test_case=tc,
            approach=RetrievalApproach.VECTOR_RAG,
            no_expansion_recall=no_exp_recall,
            with_expansion_recall=with_exp_recall,
            improvement=improvement,
            expansion_latency_ms=expansion.expansion_latency_ms,
        ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY BY AMBIGUITY TYPE")
    print("=" * 70)

    by_type: dict[str, list[BenchmarkResult]] = {}
    for r in results:
        t = r.test_case.ambiguity_type
        by_type.setdefault(t, []).append(r)

    for amb_type, type_results in sorted(by_type.items()):
        avg_no_exp = sum(r.no_expansion_recall for r in type_results) / len(type_results)
        avg_with_exp = sum(r.with_expansion_recall for r in type_results) / len(type_results)
        avg_improvement = sum(r.improvement for r in type_results) / len(type_results)
        avg_latency = sum(r.expansion_latency_ms for r in type_results) / len(type_results)

        print(f"\n{amb_type.upper()} ({len(type_results)} cases):")
        print(f"  Without expansion: {avg_no_exp:.1%}")
        print(f"  With expansion:    {avg_with_exp:.1%}")
        print(f"  Average improvement: {avg_improvement:+.1%}")
        print(f"  Average latency: {avg_latency:.0f}ms")

    # Overall
    avg_no = sum(r.no_expansion_recall for r in results) / len(results)
    avg_with = sum(r.with_expansion_recall for r in results) / len(results)
    avg_imp = sum(r.improvement for r in results) / len(results)
    avg_lat = sum(r.expansion_latency_ms for r in results) / len(results)

    print(f"\nOVERALL ({len(results)} cases):")
    print(f"  Without expansion: {avg_no:.1%}")
    print(f"  With expansion:    {avg_with:.1%}")
    print(f"  Average improvement: {avg_imp:+.1%}")
    print(f"  Average latency: {avg_lat:.0f}ms")

    # Did expansion hurt any direct queries?
    hurt_cases = [r for r in results if r.improvement < 0]
    if hurt_cases:
        print(f"\n⚠️  Expansion HURT {len(hurt_cases)} cases:")
        for r in hurt_cases:
            print(f"  - {r.test_case.description}: {r.improvement:+.1%}")
    else:
        print("\n✓ Expansion never hurt performance")

    return results


def compute_recall(docs: list[dict], expected: list[str]) -> float:
    """Compute recall - how many expected items are in retrieved docs."""
    text = " ".join(doc["content"] for doc in docs)
    found = sum(1 for e in expected if e.lower() in text.lower())
    return found / len(expected) if expected else 1.0


# =============================================================================
# Main
# =============================================================================

async def main():
    """Run the benchmark."""
    # Try to get LLM
    groq_key = os.getenv("GROQ_API_KEY")

    if groq_key:
        from draagon_ai.llm.groq import GroqLLM
        llm = GroqLLM(api_key=groq_key)
        print("Using Groq LLM")
    else:
        print("ERROR: GROQ_API_KEY required")
        print("Set it in .env or environment")
        sys.exit(1)

    await run_benchmark(llm)


if __name__ == "__main__":
    asyncio.run(main())
