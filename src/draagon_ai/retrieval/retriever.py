"""Hybrid retrieval system with multi-stage pipeline and quality validation.

This module implements a sophisticated retrieval pipeline that combines:
1. Broad recall - Get many candidates from multiple sources
2. Re-ranking - Score relevance using metadata boosts and optional LLM
3. Quality validation - Assess if results are good enough
4. Fallback - Query expansion if quality too low

The design is based on Corrective RAG (CRAG) and Self-RAG patterns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .protocols import MemorySearchProvider, LLMProvider

logger = logging.getLogger(__name__)


class CRAGGrade(str, Enum):
    """CRAG-style document grades based on relevance."""

    RELEVANT = "relevant"      # High relevance (>= 0.55)
    AMBIGUOUS = "ambiguous"    # Medium relevance (0.4 - 0.55)
    IRRELEVANT = "irrelevant"  # Low relevance (< 0.4)


@dataclass
class RetrievalConfig:
    """Configuration for hybrid retrieval.

    Attributes:
        recall_multiplier: How many more candidates to fetch for broad recall
        min_relevance: Minimum average relevance for results to be "sufficient"
        relevant_threshold: Score threshold for CRAG "relevant" grade
        ambiguous_threshold: Score threshold for CRAG "ambiguous" grade
        enable_query_expansion: Whether to try query expansion on low quality
        memory_type_boosts: Boost factors for different memory types
        keyword_match_boost: Boost per matching keyword
        exact_phrase_boost: Boost for exact phrase match
        enable_llm_rerank: Whether to use LLM for re-ranking (slower but more accurate)
    """

    recall_multiplier: int = 10
    max_recall: int = 100
    min_relevance: float = 0.5
    relevant_threshold: float = 0.55
    ambiguous_threshold: float = 0.4
    enable_query_expansion: bool = True
    memory_type_boosts: dict[str, float] = field(default_factory=lambda: {
        "knowledge": 1.2,  # System knowledge is curated/verified
        "skill": 1.1,      # Skills are actionable
        "instruction": 1.15,  # User instructions are important
    })
    scope_boosts: dict[str, float] = field(default_factory=lambda: {
        "private": 1.15,  # User memories get slight boost
        "shared": 1.15,
    })
    keyword_match_boost: float = 0.1
    exact_phrase_boost: float = 1.3
    enable_llm_rerank: bool = False


@dataclass
class RetrievalResult:
    """Result of retrieval process with quality metrics.

    Attributes:
        documents: List of retrieved documents with scores
        relevance_score: Average relevance of top results
        sufficient: Whether retrieval quality meets threshold
        strategy_used: Which retrieval strategy worked ("standard", "expanded", "none")
        candidates_count: How many candidates were considered
        grading: CRAG-style grading counts
    """

    documents: list[dict[str, Any]]
    relevance_score: float
    sufficient: bool
    strategy_used: str
    candidates_count: int
    grading: dict[str, int] = field(default_factory=lambda: {
        "relevant": 0,
        "irrelevant": 0,
        "ambiguous": 0,
    })

    @property
    def has_relevant(self) -> bool:
        """Check if any documents were graded as relevant."""
        return self.grading.get("relevant", 0) > 0

    @property
    def mostly_irrelevant(self) -> bool:
        """Check if most documents are irrelevant."""
        total = sum(self.grading.values())
        if total == 0:
            return True
        return self.grading.get("irrelevant", 0) / total > 0.7


class HybridRetriever:
    """Multi-stage retrieval with quality validation.

    Improves on basic semantic search by:
    - Getting more candidates (broad recall)
    - Re-ranking with metadata boosts (more accurate)
    - Query expansion fallback (handles paraphrasing)

    Expected improvement: 2/10 relevant → 7/10 relevant
    """

    def __init__(
        self,
        memory_provider: MemorySearchProvider,
        llm_provider: LLMProvider | None = None,
        config: RetrievalConfig | None = None,
    ):
        """Initialize hybrid retriever.

        Args:
            memory_provider: Provider for memory/knowledge search
            llm_provider: Optional LLM provider for query expansion and re-ranking
            config: Configuration options
        """
        self.memory = memory_provider
        self.llm = llm_provider
        self.config = config or RetrievalConfig()

    async def retrieve(
        self,
        query: str,
        user_id: str,
        k: int = 10,
        min_relevance: float | None = None,
        include_knowledge: bool = True,
        context_id: str | None = None,
        memory_types: list[Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve documents using hybrid multi-stage approach.

        Stages:
        1. Broad recall (semantic search with lower threshold)
        2. Re-ranking with metadata boosts
        3. Quality validation
        4. Fallback strategies if quality low

        Args:
            query: Search query
            user_id: User ID for scoping
            k: Number of documents to return
            min_relevance: Minimum acceptable relevance score
            include_knowledge: Include system knowledge base
            context_id: Optional context (e.g., household ID)
            memory_types: Optional filter by memory types

        Returns:
            RetrievalResult with documents and quality metrics
        """
        min_relevance = min_relevance or self.config.min_relevance
        logger.info(f"[HYBRID RETRIEVAL] Starting for query: {query[:50]}...")

        # Stage 1: Broad recall
        recall_k = min(k * self.config.recall_multiplier, self.config.max_recall)
        candidates = await self._recall_stage(
            query=query,
            user_id=user_id,
            k=recall_k,
            include_knowledge=include_knowledge,
            context_id=context_id,
            memory_types=memory_types,
        )

        logger.info(f"[HYBRID RETRIEVAL] Recall stage: {len(candidates)} candidates")

        if not candidates:
            return RetrievalResult(
                documents=[],
                relevance_score=0.0,
                sufficient=False,
                strategy_used="none",
                candidates_count=0,
                grading={"relevant": 0, "irrelevant": 0, "ambiguous": 0},
            )

        # Stage 2: Re-rank with metadata boosts and CRAG grading
        reranked, grading = await self._rerank_stage(query, candidates, k=k)
        logger.info(f"[HYBRID RETRIEVAL] Re-rank stage: {len(reranked)} selected")

        # Stage 3: Validate relevance
        avg_relevance = self._calculate_relevance(reranked)
        logger.info(f"[HYBRID RETRIEVAL] Average relevance: {avg_relevance:.2f}")

        # Stage 4: Fallback if quality too low
        strategy = "standard"
        if avg_relevance < min_relevance and self.config.enable_query_expansion and self.llm:
            logger.warning(
                f"[HYBRID RETRIEVAL] Low quality ({avg_relevance:.2f}), trying query expansion"
            )

            expanded = await self._expand_query(query)
            logger.info(f"[HYBRID RETRIEVAL] Expanded query: {expanded}")

            if expanded != query:
                expanded_candidates = await self._recall_stage(
                    query=expanded,
                    user_id=user_id,
                    k=recall_k,
                    include_knowledge=include_knowledge,
                    context_id=context_id,
                    memory_types=memory_types,
                )

                if expanded_candidates:
                    reranked, grading = await self._rerank_stage(expanded, expanded_candidates, k=k)
                    avg_relevance = self._calculate_relevance(reranked)
                    strategy = "expanded"
                    logger.info(f"[HYBRID RETRIEVAL] Expanded relevance: {avg_relevance:.2f}")

        return RetrievalResult(
            documents=reranked,
            relevance_score=avg_relevance,
            sufficient=avg_relevance >= min_relevance,
            strategy_used=strategy,
            candidates_count=len(candidates),
            grading=grading,
        )

    async def _recall_stage(
        self,
        query: str,
        user_id: str,
        k: int,
        include_knowledge: bool,
        context_id: str | None = None,
        memory_types: list[Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Broad recall stage: get candidates from memory provider.

        Uses lower threshold to maximize recall at the cost of precision.
        Precision is recovered in re-ranking stage.
        """
        results = await self.memory.search(
            query=query,
            user_id=user_id,
            limit=k,
            min_score=0.3,  # Lower threshold for broad recall
            memory_types=memory_types,
            context_id=context_id,
        )

        # Convert results to dict format
        documents = []
        for result in results:
            if hasattr(result, "to_dict"):
                doc = result.to_dict()
            elif hasattr(result, "memory"):
                # Handle SearchResult wrapper from LayeredMemoryProvider
                memory = result.memory
                doc = {
                    "id": memory.id,
                    "content": memory.content,
                    "score": result.score,
                    "memory_type": memory.memory_type.value if hasattr(memory.memory_type, "value") else str(memory.memory_type),
                    "importance": memory.importance,
                    "user_id": memory.user_id,
                    "scope": memory.scope.value if hasattr(memory.scope, "value") else str(memory.scope),
                    "entities": memory.entities or [],
                    "created_at": memory.created_at.isoformat() if memory.created_at else None,
                }
            elif isinstance(result, dict):
                doc = result
            else:
                # Try to access common attributes
                doc = {
                    "id": getattr(result, "id", ""),
                    "content": getattr(result, "content", ""),
                    "score": getattr(result, "score", 0.5),
                    "memory_type": getattr(result, "memory_type", "fact"),
                    "importance": getattr(result, "importance", 0.5),
                }
            documents.append(doc)

        return documents

    async def _rerank_stage(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        """Re-rank candidates using metadata boosts.

        Uses:
        1. Base embedding similarity score
        2. Memory type preference
        3. Scope preference
        4. Keyword matching bonus
        5. Exact phrase matching bonus

        Returns:
            Tuple of (reranked documents, grading dict)
        """
        if not candidates:
            return [], {"relevant": 0, "irrelevant": 0, "ambiguous": 0}

        query_lower = query.lower()
        query_words = [w for w in query_lower.split() if len(w) > 3]

        scored = []
        for doc in candidates:
            # Start with weighted score if available, fall back to raw score
            base_score = doc.get("weighted_score") or doc.get("score", 0.5)
            score = base_score

            # Boost 1: Memory type preference
            mem_type = doc.get("memory_type", "")
            if mem_type in self.config.memory_type_boosts:
                score *= self.config.memory_type_boosts[mem_type]

            # Boost 2: Scope preference
            scope = doc.get("scope", "")
            if scope in self.config.scope_boosts:
                score *= self.config.scope_boosts[scope]

            # Boost 3: Keyword matching
            content = doc.get("content", "").lower()
            keyword_matches = sum(1 for word in query_words if word in content)
            if keyword_matches > 0:
                score *= (1.0 + self.config.keyword_match_boost * keyword_matches)

            # Boost 4: Exact phrase matching
            if query_lower in content:
                score *= self.config.exact_phrase_boost

            # Store boosted score
            doc["boosted_score"] = score
            scored.append((score, doc))

        # Sort by score and take top k
        scored.sort(reverse=True, key=lambda x: x[0])

        logger.debug(f"[HYBRID RETRIEVAL] Top 3 scores: {[s for s, _ in scored[:3]]}")

        # CRAG-style grading
        grading = {"relevant": 0, "irrelevant": 0, "ambiguous": 0}
        for score, _ in scored:
            if score >= self.config.relevant_threshold:
                grading["relevant"] += 1
            elif score >= self.config.ambiguous_threshold:
                grading["ambiguous"] += 1
            else:
                grading["irrelevant"] += 1

        logger.info(
            f"[HYBRID RETRIEVAL] CRAG grading: R:{grading['relevant']} "
            f"A:{grading['ambiguous']} I:{grading['irrelevant']}"
        )

        return [doc for _, doc in scored[:k]], grading

    def _calculate_relevance(self, documents: list[dict[str, Any]]) -> float:
        """Calculate average relevance of top documents.

        Uses boosted scores with position weighting.
        """
        if not documents:
            return 0.0

        # Average with position weighting
        weighted_scores = []
        for i, doc in enumerate(documents[:5]):
            position_weight = 1.0 - (i * 0.15)  # Top result = 1.0, 5th = 0.4
            score = doc.get("boosted_score", doc.get("score", 0.5))
            weighted_scores.append(score * position_weight)

        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

    async def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and paraphrasing.

        Helps when user's phrasing doesn't match stored documents.
        """
        if not self.llm:
            return query

        prompt = """Rephrase this query to be more likely to match documentation.
Add synonyms and expand abbreviations.

Original: {query}

Return ONLY the expanded query, nothing else.""".format(query=query)

        try:
            result = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=50,
            )

            expanded = result.get("content", query).strip()
            return expanded if expanded else query
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
            return query

    async def search_with_self_rag(
        self,
        query: str,
        user_id: str,
        k: int = 5,
        relevance_threshold: float = 0.6,
        context_id: str | None = None,
    ) -> RetrievalResult:
        """Search with self-RAG pattern (relevance assessment before use).

        This is a convenience method that:
        1. Retrieves documents with the hybrid pipeline
        2. Filters to only high-relevance results
        3. Returns structured result with quality metrics

        Args:
            query: Search query
            user_id: User ID for scoping
            k: Number of documents to return
            relevance_threshold: Minimum score for inclusion
            context_id: Optional context for scoping

        Returns:
            RetrievalResult with only high-relevance documents
        """
        result = await self.retrieve(
            query=query,
            user_id=user_id,
            k=k * 2,  # Fetch more to filter
            min_relevance=relevance_threshold,
            context_id=context_id,
        )

        # Filter to high-relevance only
        high_relevance = [
            doc for doc in result.documents
            if doc.get("boosted_score", doc.get("score", 0)) >= relevance_threshold
        ]

        return RetrievalResult(
            documents=high_relevance[:k],
            relevance_score=result.relevance_score,
            sufficient=len(high_relevance) > 0,
            strategy_used=result.strategy_used,
            candidates_count=result.candidates_count,
            grading=result.grading,
        )

    async def search_cross_chat(
        self,
        query: str,
        user_id: str,
        current_conversation_id: str | None = None,
        limit: int = 3,
        score_threshold: float = 0.6,
        context_id: str | None = None,
    ) -> dict[str, Any]:
        """Search for relevant context from past conversations.

        This is useful for cross-chat RAG where past conversation
        context may be relevant to the current query.

        Args:
            query: Search query
            user_id: User ID
            current_conversation_id: ID of current conversation to exclude
            limit: Maximum results
            score_threshold: Minimum relevance score
            context_id: Optional context for scoping

        Returns:
            Dict with results, count, and formatted contexts
        """
        # Search episodic memories specifically
        from ..memory import MemoryType

        try:
            result = await self.retrieve(
                query=query,
                user_id=user_id,
                k=limit * 2,
                min_relevance=score_threshold,
                memory_types=[MemoryType.EPISODIC],
                context_id=context_id,
            )
        except Exception:
            # If MemoryType import fails, search without type filter
            result = await self.retrieve(
                query=query,
                user_id=user_id,
                k=limit * 2,
                min_relevance=score_threshold,
                context_id=context_id,
            )

        # Filter out current conversation
        results = result.documents
        if current_conversation_id:
            results = [
                r for r in results
                if r.get("metadata", {}).get("conversation_id") != current_conversation_id
                and r.get("conversation_id") != current_conversation_id
            ]

        results = results[:limit]

        return {
            "results": results,
            "count": len(results),
            "contexts": [
                f"[Past conversation ({r.get('created_at', 'unknown')})]: {r['content']}"
                for r in results
            ],
        }
