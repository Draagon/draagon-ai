"""Production-grade retrieval benchmark infrastructure.

This module provides tools for building and running retrieval benchmarks
following industry standards (BEIR, RAGAS, MTEB, HotpotQA).

Per CONSTITUTION.md: All benchmarks use REAL providers, not mocks.
"""

from .corpus import (
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
    DocumentCorpus,
    CorpusMetadata,
)
from .downloaders import (
    LocalDocumentScanner,
    OnlineDocumentFetcher,
    OnlineSource,
    FetchResult,
    TECHNICAL_SOURCES,
    NARRATIVE_SOURCES,
    ACADEMIC_SOURCES,
    DistractorGenerator,
    DistractorConfig,
    SimilarityLevel,
)

__all__ = [
    "BenchmarkDocument",
    "DocumentCategory",
    "DocumentSource",
    "DocumentCorpus",
    "CorpusMetadata",
    "LocalDocumentScanner",
    "OnlineDocumentFetcher",
    "OnlineSource",
    "FetchResult",
    "TECHNICAL_SOURCES",
    "NARRATIVE_SOURCES",
    "ACADEMIC_SOURCES",
    "DistractorGenerator",
    "DistractorConfig",
    "SimilarityLevel",
]
