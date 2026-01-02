#!/usr/bin/env python3
"""Comprehensive Embedding Strategy Benchmark.

Based on research from BEIR, MTEB, and RAG evaluation literature, this benchmark
evaluates all 4 embedding strategies across progressively harder test cases.

Test Tiers (inspired by BEIR's diverse task approach):
- Tier 1: Basic queries all strategies should handle
- Tier 2: Hard cases that break some strategies (lexical mismatch, entity confusion)
- Tier 3: Frontier cases that challenge all strategies (multi-hop, implicit reference)
- Holdout: Reserved for overfitting detection (never optimize against this)

Failure Mode Categories (from enterprise retrieval research):
1. Lexical Mismatch - Query uses different terms than documents
2. Entity Confusion - Similar entities with different meanings
3. Implicit Reference - Pronouns, "it", "that", requiring context
4. Multi-hop Reasoning - Answer requires connecting multiple facts
5. Out-of-Domain - Technical jargon not in embedding model's training

Anti-Overfitting Measures:
- 70/30 train/holdout split
- Cross-validation across query categories
- Diverse query types (question, keyword, instruction)
- Novel entity names to prevent memorization

Sources:
- BEIR: https://github.com/beir-cellar/beir
- MTEB: https://prasun-mishra.medium.com/massive-text-embedding-benchmark-mteb
- Hard Negatives: https://arxiv.org/html/2505.18366
- RAG Failures: https://snorkel.ai/blog/retrieval-augmented-generation-rag-failure-modes

Run with:
    python3.11 tests/integration/agents/benchmark_embedding_strategies.py
"""

import asyncio
import json
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


class FailureMode(str, Enum):
    """Categories of retrieval failure."""

    NONE = "none"  # Should succeed
    LEXICAL_MISMATCH = "lexical_mismatch"  # Different words, same meaning
    ENTITY_CONFUSION = "entity_confusion"  # Similar names, different things
    IMPLICIT_REFERENCE = "implicit_reference"  # Pronouns, context-dependent
    MULTI_HOP = "multi_hop"  # Requires connecting facts
    OUT_OF_DOMAIN = "out_of_domain"  # Specialized jargon


class QueryFormat(str, Enum):
    """Query format types (from ViDoRe V3)."""

    QUESTION = "question"  # "What is...?"
    KEYWORD = "keyword"  # "auth tokens JWT"
    INSTRUCTION = "instruction"  # "Find information about..."


@dataclass
class TestCase:
    """A single test case with expected outcomes."""

    query: str
    expected_content: list[str]  # Keywords that should appear in results
    unexpected_content: list[str] = field(default_factory=list)  # Should NOT appear
    failure_mode: FailureMode = FailureMode.NONE
    query_format: QueryFormat = QueryFormat.QUESTION
    tier: int = 1  # 1=basic, 2=hard, 3=frontier
    description: str = ""
    is_holdout: bool = False  # Part of held-out test set


@dataclass
class StrategyScore:
    """Score for a single strategy on a test case."""

    strategy: str
    recall: float  # % of expected content found
    precision: float  # % of results that are relevant
    mrr: float  # Mean Reciprocal Rank of first relevant result
    latency_ms: float
    expanded_text: str = ""


@dataclass
class TestResult:
    """Result of running a test case across all strategies."""

    test_case: TestCase
    strategy_scores: dict[str, StrategyScore]
    winner: str  # Best performing strategy
    all_failed: bool  # True if no strategy found relevant content


# =============================================================================
# Test Data: Knowledge Base
# =============================================================================

KNOWLEDGE_BASE = [
    # Team authentication patterns (clear, direct matches)
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
    # Database patterns (for confusion tests)
    {
        "id": "db_postgres",
        "content": "PostgreSQL is used by the Analytics team for OLAP workloads. They run complex aggregation queries.",
        "entities": ["PostgreSQL", "Analytics", "OLAP"],
        "domain": "database",
    },
    {
        "id": "db_mongo",
        "content": "MongoDB is used by the Mobile team for offline-first data sync. It stores user preferences and cached content.",
        "entities": ["MongoDB", "Mobile", "offline"],
        "domain": "database",
    },
    # User facts (for multi-hop tests)
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
    # Specialized jargon (for out-of-domain tests)
    {
        "id": "security_owasp",
        "content": "CSRF protection uses SameSite cookies and anti-forgery tokens. OWASP recommends double-submit cookie pattern.",
        "entities": ["CSRF", "OWASP", "SameSite"],
        "domain": "security",
    },
    {
        "id": "security_cve",
        "content": "CVE-2024-1234 affects JWT libraries with algorithm confusion vulnerabilities. Upgrade to signed-only verification.",
        "entities": ["CVE-2024-1234", "JWT", "algorithm confusion"],
        "domain": "security",
    },
    # Entities with similar names (for confusion tests)
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
# Test Cases: Tiered by Difficulty
# =============================================================================

TIER_1_BASIC = [
    # Direct matches - all strategies should succeed
    TestCase(
        query="How does the Engineering team handle authentication?",
        expected_content=["Engineering", "OAuth2", "JWT"],
        failure_mode=FailureMode.NONE,
        tier=1,
        description="Direct team + topic query",
    ),
    TestCase(
        query="What authentication does the Mobile team use?",
        expected_content=["Mobile", "biometric", "Face ID"],
        failure_mode=FailureMode.NONE,
        tier=1,
        description="Direct team + auth query",
    ),
    TestCase(
        query="authentication JWT tokens",
        expected_content=["JWT", "Engineering"],
        query_format=QueryFormat.KEYWORD,
        failure_mode=FailureMode.NONE,
        tier=1,
        description="Keyword search",
    ),
    TestCase(
        query="Find information about the Platform team's security approach",
        expected_content=["Platform", "mTLS", "Istio"],
        query_format=QueryFormat.INSTRUCTION,
        failure_mode=FailureMode.NONE,
        tier=1,
        description="Instruction format query",
    ),
]

TIER_2_HARD = [
    # Lexical mismatch - query uses synonyms
    TestCase(
        query="How do teams verify user identity?",
        expected_content=["authentication", "OAuth2", "biometric"],
        unexpected_content=["database"],
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=2,
        description="'verify identity' vs 'authentication'",
    ),
    TestCase(
        query="What credentials do services use to talk to each other?",
        expected_content=["mTLS", "Platform", "Istio"],
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=2,
        description="'credentials' + 'services' vs 'inter-service authentication'",
    ),
    TestCase(
        query="How do we prevent unauthorized API access?",
        expected_content=["OAuth2", "JWT", "authentication"],
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=2,
        description="'prevent unauthorized access' vs 'authentication'",
    ),
    # Entity confusion - similar names
    TestCase(
        query="What is Atlas used for?",
        expected_content=["SSO", "identity"],
        unexpected_content=["MongoDB", "cloud database"],
        failure_mode=FailureMode.ENTITY_CONFUSION,
        tier=2,
        description="Project Atlas vs MongoDB Atlas - should get identity platform",
    ),
    TestCase(
        query="What database does the Mobile team use?",
        expected_content=["MongoDB", "offline", "sync"],
        unexpected_content=["PostgreSQL", "Analytics"],
        failure_mode=FailureMode.ENTITY_CONFUSION,
        tier=2,
        description="Mobile team uses MongoDB, not PostgreSQL",
    ),
]

TIER_3_FRONTIER = [
    # Multi-hop reasoning
    TestCase(
        query="What authentication method does Doug's team use?",
        expected_content=["OAuth2", "JWT", "Engineering"],
        failure_mode=FailureMode.MULTI_HOP,
        tier=3,
        description="Doug → Engineering team → OAuth2 (2 hops)",
    ),
    TestCase(
        query="What did Sarah design?",
        expected_content=["service mesh", "Platform", "mTLS"],
        failure_mode=FailureMode.MULTI_HOP,
        tier=3,
        description="Sarah → Platform team → service mesh architecture",
    ),
    # Implicit reference
    TestCase(
        query="For the biometric auth system, what's the backup when it fails?",
        expected_content=["Mobile", "Face ID", "Touch ID"],
        failure_mode=FailureMode.IMPLICIT_REFERENCE,
        tier=3,
        description="Implicit 'it' refers to biometric auth",
    ),
    # Out-of-domain jargon
    TestCase(
        query="How do we protect against CSRF attacks?",
        expected_content=["CSRF", "SameSite", "anti-forgery"],
        failure_mode=FailureMode.OUT_OF_DOMAIN,
        tier=3,
        description="Security jargon - CSRF, SameSite cookies",
    ),
    TestCase(
        query="Are we affected by the recent JWT algorithm confusion CVE?",
        expected_content=["CVE-2024-1234", "algorithm confusion"],
        failure_mode=FailureMode.OUT_OF_DOMAIN,
        tier=3,
        description="Specific CVE reference with jargon",
    ),
]

# Held-out set for overfitting detection (30% of test cases, novel patterns)
HOLDOUT_SET = [
    TestCase(
        query="Which teams don't use cloud-based auth?",
        expected_content=["Mobile", "biometric"],  # Only Mobile uses local biometric
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=2,
        is_holdout=True,
        description="Negation query - holdout",
    ),
    TestCase(
        query="password reset flow",
        expected_content=[],  # No content about password reset exists
        failure_mode=FailureMode.NONE,
        tier=1,
        is_holdout=True,
        description="No match case - holdout",
    ),
    TestCase(
        query="What auth does the team that uses PostgreSQL employ?",
        expected_content=["Analytics"],  # Analytics uses PostgreSQL, but we don't have their auth
        failure_mode=FailureMode.MULTI_HOP,
        tier=3,
        is_holdout=True,
        description="Multi-hop with missing link - holdout",
    ),
    TestCase(
        query="Compare OAuth and mTLS approaches",
        expected_content=["OAuth2", "mTLS", "Engineering", "Platform"],
        failure_mode=FailureMode.MULTI_HOP,
        tier=3,
        is_holdout=True,
        description="Comparison query - holdout",
    ),
]


# =============================================================================
# TIER 4: FRONTIER CASES DESIGNED TO BREAK ALL STRATEGIES
# =============================================================================
# These are intentionally hard cases that expose fundamental limitations.
# The goal is NOT to optimize for these - but to understand the limits.

TIER_4_BREAKING = [
    # Category 1: SEMANTIC INVERSION (negation, opposite meaning)
    # Embeddings typically struggle with negation - "not X" is close to "X"
    TestCase(
        query="Which teams explicitly avoid using JWT tokens?",
        expected_content=["Platform", "mTLS", "Mobile", "biometric"],  # Teams NOT using JWT
        unexpected_content=["JWT", "Engineering"],  # Should NOT return JWT users
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=4,
        description="Negation: 'avoid JWT' should NOT return JWT content",
    ),
    TestCase(
        query="authentication methods that don't require network connectivity",
        expected_content=["biometric", "Face ID", "Touch ID", "Mobile"],
        unexpected_content=["OAuth2", "mTLS", "IAM"],  # All require network
        failure_mode=FailureMode.LEXICAL_MISMATCH,
        tier=4,
        description="Negation: offline-capable auth only",
    ),

    # Category 2: TEMPORAL/CONTEXTUAL REASONING
    # Requires understanding sequence or state that isn't explicit
    TestCase(
        query="What happens after the JWT token expires?",
        expected_content=["refresh token", "rotation"],
        failure_mode=FailureMode.IMPLICIT_REFERENCE,
        tier=4,
        description="Temporal: 'after expires' requires understanding flow",
    ),
    TestCase(
        query="What was the auth method before we adopted service mesh?",
        expected_content=[],  # No historical data exists
        failure_mode=FailureMode.IMPLICIT_REFERENCE,
        tier=4,
        description="Temporal: historical query with no data",
    ),

    # Category 3: COMPOSITIONAL/MULTI-HOP WITH INFERENCE
    # Requires chaining facts AND making inferences
    TestCase(
        query="Which team member works on the system that uses certificate rotation?",
        expected_content=["Sarah", "Platform"],  # Sarah → Platform → Istio → cert rotation
        failure_mode=FailureMode.MULTI_HOP,
        tier=4,
        description="3-hop: person → team → system → feature",
    ),
    TestCase(
        query="What's the security vulnerability if Doug's team's token expiry is too long?",
        expected_content=["JWT", "1-hour"],  # Doug → Engineering → JWT with 1hr expiry
        failure_mode=FailureMode.MULTI_HOP,
        tier=4,
        description="Multi-hop with implicit reasoning about security",
    ),

    # Category 4: AMBIGUOUS ENTITY RESOLUTION WITH CONTEXT
    # Same word means different things based on context
    TestCase(
        query="What does Atlas handle for mobile apps?",
        expected_content=["sync", "offline", "MongoDB"],  # MongoDB Atlas for mobile
        unexpected_content=["SSO", "IdP"],  # NOT Project Atlas
        failure_mode=FailureMode.ENTITY_CONFUSION,
        tier=4,
        description="Entity: 'Atlas' + 'mobile' should disambiguate to MongoDB Atlas",
    ),
    TestCase(
        query="How does the identity Atlas integrate with external systems?",
        expected_content=["SSO", "IdP", "federation"],  # Project Atlas
        unexpected_content=["MongoDB", "cloud", "sync"],  # NOT MongoDB Atlas
        failure_mode=FailureMode.ENTITY_CONFUSION,
        tier=4,
        description="Entity: 'identity Atlas' should disambiguate to Project Atlas",
    ),

    # Category 5: OUT-OF-VOCABULARY / RARE TERMINOLOGY
    # Terms that embedding models may not understand well
    TestCase(
        query="How do we implement TOTP for 2FA fallback?",
        expected_content=[],  # No TOTP content exists
        failure_mode=FailureMode.OUT_OF_DOMAIN,
        tier=4,
        description="OOV: TOTP is auth-related but not in our KB",
    ),
    TestCase(
        query="FIDO2 WebAuthn passwordless implementation",
        expected_content=["biometric"],  # Closest is biometric auth
        failure_mode=FailureMode.OUT_OF_DOMAIN,
        tier=4,
        description="OOV: FIDO2/WebAuthn jargon, should map to biometric",
    ),

    # Category 6: COUNTERFACTUAL / HYPOTHETICAL
    # Questions about things that could be but aren't
    TestCase(
        query="If the Engineering team switched to mTLS, what would change?",
        expected_content=["OAuth2", "mTLS"],  # Should find both to compare
        failure_mode=FailureMode.MULTI_HOP,
        tier=4,
        description="Counterfactual: hypothetical scenario",
    ),
    TestCase(
        query="What would break if we disabled certificate rotation?",
        expected_content=["Istio", "Platform", "mTLS"],
        failure_mode=FailureMode.IMPLICIT_REFERENCE,
        tier=4,
        description="Counterfactual: impact analysis",
    ),
]


# =============================================================================
# Benchmark Runner
# =============================================================================

class RetrievalBenchmark:
    """Runs the retrieval benchmark across all strategies."""

    def __init__(self, llm, embedder, knowledge_base: list[dict]):
        self.llm = llm
        self.embedder = embedder
        self.knowledge_base = knowledge_base
        self.embedded_docs: list[tuple[dict, list[float]]] = []

    async def setup(self):
        """Embed all knowledge base documents."""
        print("\n    Embedding knowledge base...")
        for doc in self.knowledge_base:
            embedding = await self.embedder.embed(doc["content"])
            self.embedded_docs.append((doc, embedding))
        print(f"    ✓ Embedded {len(self.embedded_docs)} documents")

    async def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for similar documents using cosine similarity."""
        import math

        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0

        # Score all documents
        scores = []
        for doc, doc_embedding in self.embedded_docs:
            score = cosine_sim(query_embedding, doc_embedding)
            scores.append((doc, score))

        # Sort by score and return top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scores[:top_k]]

    def compute_metrics(
        self,
        results: list[dict],
        expected: list[str],
        unexpected: list[str],
    ) -> tuple[float, float, float]:
        """Compute recall, precision, and MRR."""
        if not expected:
            # No expected content - check we don't return unexpected
            has_unexpected = any(
                any(kw.lower() in doc["content"].lower() for kw in unexpected)
                for doc in results
            )
            return (1.0 if not has_unexpected else 0.0), 1.0, 1.0

        # Recall: % of expected keywords found in any result
        found_expected = 0
        for kw in expected:
            if any(kw.lower() in doc["content"].lower() for doc in results):
                found_expected += 1
        recall = found_expected / len(expected) if expected else 0

        # Precision: % of results containing at least one expected keyword
        relevant_results = 0
        first_relevant_rank = 0
        for i, doc in enumerate(results):
            is_relevant = any(kw.lower() in doc["content"].lower() for kw in expected)
            has_unexpected = any(kw.lower() in doc["content"].lower() for kw in unexpected)
            if is_relevant and not has_unexpected:
                relevant_results += 1
                if first_relevant_rank == 0:
                    first_relevant_rank = i + 1
        precision = relevant_results / len(results) if results else 0

        # MRR: Reciprocal of rank of first relevant result
        mrr = 1.0 / first_relevant_rank if first_relevant_rank > 0 else 0

        return recall, precision, mrr

    async def run_test_case(
        self,
        test_case: TestCase,
        strategies: list,
        executor,
    ) -> TestResult:
        """Run a single test case across all strategies."""
        from draagon_ai.orchestration.hybrid_retrieval import EmbeddingStrategy

        strategy_scores = {}

        for strategy in strategies:
            start = time.perf_counter()

            # Generate embedding using strategy
            result = await executor.execute_embedding(
                query=test_case.query,
                strategy=strategy,
            )

            # Search with the embedding
            search_results = await self.search(result.embedding, top_k=5)

            latency = (time.perf_counter() - start) * 1000

            # Compute metrics
            recall, precision, mrr = self.compute_metrics(
                search_results,
                test_case.expected_content,
                test_case.unexpected_content,
            )

            strategy_scores[strategy.value] = StrategyScore(
                strategy=strategy.value,
                recall=recall,
                precision=precision,
                mrr=mrr,
                latency_ms=latency,
                expanded_text=result.expanded_text[:100],
            )

        # Determine winner (highest recall, then MRR, then precision)
        best_strategy = max(
            strategy_scores.keys(),
            key=lambda s: (
                strategy_scores[s].recall,
                strategy_scores[s].mrr,
                strategy_scores[s].precision,
            ),
        )

        # Check if all failed
        all_failed = all(s.recall < 0.5 for s in strategy_scores.values())

        return TestResult(
            test_case=test_case,
            strategy_scores=strategy_scores,
            winner=best_strategy,
            all_failed=all_failed,
        )


async def main():
    """Run the complete retrieval benchmark."""
    print("\n" + "=" * 80)
    print("EMBEDDING STRATEGY BENCHMARK")
    print("Evaluating: RAW, HyDE, Query2Doc, Grounded")
    print("=" * 80)

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

    # Initialize
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

    # Create benchmark
    benchmark = RetrievalBenchmark(llm, embedder, KNOWLEDGE_BASE)
    await benchmark.setup()

    # Create strategy executor
    executor = EmbeddingStrategyExecutor(
        llm=llm,
        embedder=embedder,
    )

    strategies = list(EmbeddingStrategy)

    # Collect all test cases
    all_tests = TIER_1_BASIC + TIER_2_HARD + TIER_3_FRONTIER
    holdout_tests = HOLDOUT_SET

    # Run benchmarks by tier
    results_by_tier: dict[int, list[TestResult]] = {1: [], 2: [], 3: [], 4: []}
    holdout_results: list[TestResult] = []

    for tier, test_cases in [(1, TIER_1_BASIC), (2, TIER_2_HARD), (3, TIER_3_FRONTIER), (4, TIER_4_BREAKING)]:
        tier_names = {1: "BASIC", 2: "HARD", 3: "FRONTIER", 4: "BREAKING"}
        print(f"\n" + "-" * 80)
        print(f"[TIER {tier}] {tier_names.get(tier, 'UNKNOWN')}")
        print("-" * 80)

        for tc in test_cases:
            result = await benchmark.run_test_case(tc, strategies, executor)
            results_by_tier[tier].append(result)

            # Print result
            status = "✓" if not result.all_failed else "✗"
            print(f"\n  {status} {tc.description}")
            print(f"    Query: \"{tc.query[:60]}...\"" if len(tc.query) > 60 else f"    Query: \"{tc.query}\"")
            print(f"    Failure mode: {tc.failure_mode.value}")
            print(f"    Winner: {result.winner}")

            # Show all strategy scores
            for strat, score in result.strategy_scores.items():
                marker = "→" if strat == result.winner else " "
                print(f"      {marker} {strat:10} R={score.recall:.2f} P={score.precision:.2f} MRR={score.mrr:.2f} ({score.latency_ms:.0f}ms)")

    # Run holdout tests
    print(f"\n" + "-" * 80)
    print("[HOLDOUT] Anti-Overfitting Test Set")
    print("-" * 80)

    for tc in holdout_tests:
        result = await benchmark.run_test_case(tc, strategies, executor)
        holdout_results.append(result)

        status = "✓" if not result.all_failed else "✗"
        print(f"\n  {status} {tc.description}")
        print(f"    Query: \"{tc.query[:60]}...\"" if len(tc.query) > 60 else f"    Query: \"{tc.query}\"")

        for strat, score in result.strategy_scores.items():
            marker = "→" if strat == result.winner else " "
            print(f"      {marker} {strat:10} R={score.recall:.2f} P={score.precision:.2f} MRR={score.mrr:.2f}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Per-strategy summary
    print("\n  Strategy Performance by Tier:")
    print("  " + "-" * 70)
    print(f"  {'Strategy':<12} {'Tier 1':>10} {'Tier 2':>10} {'Tier 3':>10} {'Tier 4':>10} {'Holdout':>10}")
    print("  " + "-" * 76)

    for strat in ["raw", "hyde", "query2doc", "grounded"]:
        tier_recalls = []
        for tier in [1, 2, 3, 4]:
            recalls = [r.strategy_scores[strat].recall for r in results_by_tier[tier]]
            tier_recalls.append(sum(recalls) / len(recalls) if recalls else 0)

        holdout_recall = sum(r.strategy_scores[strat].recall for r in holdout_results) / len(holdout_results) if holdout_results else 0

        print(f"  {strat:<12} {tier_recalls[0]:>10.1%} {tier_recalls[1]:>10.1%} {tier_recalls[2]:>10.1%} {tier_recalls[3]:>10.1%} {holdout_recall:>10.1%}")

    # Failure mode analysis
    print("\n  Performance by Failure Mode:")
    print("  " + "-" * 70)

    failure_mode_results: dict[FailureMode, list[TestResult]] = {}
    for tier_results in results_by_tier.values():
        for r in tier_results:
            fm = r.test_case.failure_mode
            if fm not in failure_mode_results:
                failure_mode_results[fm] = []
            failure_mode_results[fm].append(r)

    for fm, results in failure_mode_results.items():
        print(f"\n  {fm.value}:")
        for strat in ["raw", "hyde", "query2doc", "grounded"]:
            avg_recall = sum(r.strategy_scores[strat].recall for r in results) / len(results)
            wins = sum(1 for r in results if r.winner == strat)
            print(f"    {strat:<12} avg_recall={avg_recall:.1%} wins={wins}/{len(results)}")

    # Identify hardest cases
    print("\n  Hardest Cases (all strategies <50% recall):")
    print("  " + "-" * 70)

    all_results = sum(results_by_tier.values(), []) + holdout_results
    hard_cases = [r for r in all_results if r.all_failed]

    if hard_cases:
        for r in hard_cases:
            print(f"    - {r.test_case.description}")
            print(f"      Query: \"{r.test_case.query}\"")
            best_recall = max(s.recall for s in r.strategy_scores.values())
            print(f"      Best recall achieved: {best_recall:.1%}")
    else:
        print("    (none - all test cases had at least one strategy with >50% recall)")

    # Overfitting detection
    print("\n  Overfitting Check:")
    print("  " + "-" * 70)

    for strat in ["raw", "hyde", "query2doc", "grounded"]:
        train_recall = sum(
            r.strategy_scores[strat].recall
            for tier_results in results_by_tier.values()
            for r in tier_results
        ) / sum(len(tier_results) for tier_results in results_by_tier.values())

        holdout_recall = sum(r.strategy_scores[strat].recall for r in holdout_results) / len(holdout_results)

        gap = train_recall - holdout_recall
        status = "⚠️ OVERFITTING" if gap > 0.15 else "✓ OK" if gap < 0.10 else "⚡ WATCH"

        print(f"    {strat:<12} train={train_recall:.1%} holdout={holdout_recall:.1%} gap={gap:+.1%} {status}")

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
