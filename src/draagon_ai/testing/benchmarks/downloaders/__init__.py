"""Document downloaders and scanners for benchmark corpus assembly.

Provides tools for collecting documents from various sources:
- LocalDocumentScanner: Scan local filesystem
- OnlineDocumentationFetcher: Fetch from web (future)
- DistractorGenerator: Generate synthetic distractors (future)
"""

from .local_scanner import LocalDocumentScanner

__all__ = [
    "LocalDocumentScanner",
]
