"""Document downloaders and scanners for benchmark corpus assembly.

Provides tools for collecting documents from various sources:
- LocalDocumentScanner: Scan local filesystem
- OnlineDocumentFetcher: Fetch from web documentation
- DistractorGenerator: Generate synthetic distractors (future)
"""

from .local_scanner import LocalDocumentScanner
from .online_fetcher import (
    OnlineDocumentFetcher,
    OnlineSource,
    FetchResult,
    TECHNICAL_SOURCES,
    NARRATIVE_SOURCES,
    ACADEMIC_SOURCES,
)

__all__ = [
    "LocalDocumentScanner",
    "OnlineDocumentFetcher",
    "OnlineSource",
    "FetchResult",
    "TECHNICAL_SOURCES",
    "NARRATIVE_SOURCES",
    "ACADEMIC_SOURCES",
]
