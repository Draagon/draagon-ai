"""Retrieval module for draagon-ai.

This module provides sophisticated retrieval patterns for RAG-based AI agents:

- **HybridRetriever**: Multi-stage retrieval with broad recall, re-ranking, and quality validation
- **QueryContextualizer**: Pre-RAG query rewriting for multi-turn conversations
- **CRAG (Corrective RAG)**: Document grading with relevance thresholds
- **Self-RAG**: Relevance assessment before using retrieved context

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Retrieval Pipeline                            │
    │                                                                  │
    │  Query → [Contextualize] → [Broad Recall] → [Re-rank] → [Grade] │
    │           (multi-turn)      (many candidates)  (top-k)   (CRAG)  │
    │                                                                  │
    │  If quality low: [Query Expansion] → retry                      │
    └─────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.retrieval import (
        HybridRetriever,
        QueryContextualizer,
        RetrievalResult,
        ContextualizedQuery,
    )

    # Create retriever with memory and LLM providers
    retriever = HybridRetriever(memory_provider, llm_provider)

    # Optional: contextualize query for multi-turn conversations
    contextualizer = QueryContextualizer(llm_provider)
    context_result = await contextualizer.contextualize(query, history)

    # Retrieve with hybrid approach
    result = await retriever.retrieve(
        query=context_result.standalone_query,
        user_id="user123",
        k=10,
    )

    # Check quality
    if result.sufficient:
        # Use result.documents
        pass
    else:
        # May need web search fallback
        pass
"""

from .retriever import (
    HybridRetriever,
    RetrievalResult,
    RetrievalConfig,
    CRAGGrade,
)
from .contextualizer import (
    QueryContextualizer,
    ContextualizedQuery,
    ContextualizerConfig,
)
from .protocols import (
    MemorySearchProvider,
    EmbeddingProvider,
    LLMProvider,
)

__all__ = [
    # Core retriever
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalConfig",
    "CRAGGrade",
    # Query contextualization
    "QueryContextualizer",
    "ContextualizedQuery",
    "ContextualizerConfig",
    # Protocols
    "MemorySearchProvider",
    "EmbeddingProvider",
    "LLMProvider",
]
