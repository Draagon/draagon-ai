"""Document downloaders and scanners for benchmark corpus assembly.

Provides tools for collecting documents from various sources:
- LocalDocumentScanner: Scan local filesystem
- OnlineDocumentFetcher: Fetch from web documentation
- DistractorGenerator: Generate synthetic distractor documents
- LegalDocumentFetcher: Fetch legal documents (licenses, ToS, privacy policies)
- CorpusBuilder: Orchestrate corpus assembly from all sources
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
from .distractor_generator import (
    DistractorGenerator,
    DistractorConfig,
    SimilarityLevel,
)
from .legal_fetcher import (
    LegalDocumentFetcher,
    OPENSOURCE_LICENSES,
    TOS_URLS,
    PRIVACY_POLICY_URLS,
)
from .corpus_builder import (
    CorpusBuilder,
    CorpusBuilderConfig,
    SourceConfig,
    BuildProgress,
    build_default_corpus,
)

__all__ = [
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
    "LegalDocumentFetcher",
    "OPENSOURCE_LICENSES",
    "TOS_URLS",
    "PRIVACY_POLICY_URLS",
    "CorpusBuilder",
    "CorpusBuilderConfig",
    "SourceConfig",
    "BuildProgress",
    "build_default_corpus",
]
