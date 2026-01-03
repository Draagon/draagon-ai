"""Corpus builder orchestrator for benchmark infrastructure.

Provides a unified interface to assemble benchmark corpora from
multiple sources (local, online, synthetic) with configurable
diversity requirements.

Per FR-012, the corpus builder ensures:
- 500+ documents minimum
- 8 distinct content categories
- 30-50% synthetic distractors
- Proper distribution across sources

Example:
    builder = CorpusBuilder(
        local_paths=[Path.home() / "Development"],
        distractor_ratio=0.4,
    )
    corpus = await builder.build(min_docs=500)
    corpus.save("benchmark_corpus.json")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..corpus import (
    BenchmarkDocument,
    DocumentCategory,
    DocumentCorpus,
    DocumentSource,
    CorpusMetadata,
)
from .local_scanner import LocalDocumentScanner
from .online_fetcher import (
    OnlineDocumentFetcher,
    OnlineSource,
    TECHNICAL_SOURCES,
    NARRATIVE_SOURCES,
    ACADEMIC_SOURCES,
)
from .distractor_generator import DistractorGenerator, SimilarityLevel
from .legal_fetcher import LegalDocumentFetcher

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers used by distractor generation."""

    async def chat(self, messages: list[dict], **kwargs) -> dict: ...


@dataclass
class SourceConfig:
    """Configuration for a document source.

    Attributes:
        enabled: Whether to use this source
        max_docs: Maximum documents from this source
        weight: Relative weight for distribution (ignored if max_docs set)
    """

    enabled: bool = True
    max_docs: int | None = None
    weight: float = 1.0


@dataclass
class CorpusBuilderConfig:
    """Configuration for corpus assembly.

    Attributes:
        min_documents: Minimum target document count
        distractor_ratio: Target ratio of distractors (0.3-0.5 recommended)
        local: Configuration for local document scanning
        online: Configuration for online document fetching
        legal: Configuration for legal document fetching
        distractors: Configuration for synthetic distractor generation
        cache_dir: Directory for caching fetched documents
        output_path: Optional path to save corpus after build
    """

    min_documents: int = 500
    distractor_ratio: float = 0.4
    local: SourceConfig = field(default_factory=SourceConfig)
    online: SourceConfig = field(default_factory=SourceConfig)
    legal: SourceConfig = field(default_factory=SourceConfig)
    distractors: SourceConfig = field(default_factory=SourceConfig)
    cache_dir: Path | None = None
    output_path: Path | None = None


@dataclass
class BuildProgress:
    """Progress tracking for corpus build.

    Attributes:
        local_docs: Documents from local sources
        online_docs: Documents from online sources
        legal_docs: Documents from legal sources
        distractor_docs: Synthetic distractor documents
        total: Total documents collected
        duplicates_removed: Count of duplicates removed
    """

    local_docs: int = 0
    online_docs: int = 0
    legal_docs: int = 0
    distractor_docs: int = 0
    total: int = 0
    duplicates_removed: int = 0

    def __str__(self) -> str:
        return (
            f"BuildProgress(local={self.local_docs}, online={self.online_docs}, "
            f"legal={self.legal_docs}, distractors={self.distractor_docs}, "
            f"total={self.total}, deduped={self.duplicates_removed})"
        )


class CorpusBuilder:
    """Orchestrates corpus assembly from multiple sources.

    Coordinates local scanning, online fetching, legal documents,
    and synthetic distractor generation to build diverse benchmark
    corpora meeting FR-012 requirements.

    Example:
        builder = CorpusBuilder(
            local_paths=[Path.home() / "Development"],
            config=CorpusBuilderConfig(
                min_documents=500,
                distractor_ratio=0.4,
            ),
        )

        # Build corpus with progress logging
        corpus = await builder.build()

        # Or build with custom online sources
        corpus = await builder.build(
            online_sources=TECHNICAL_SOURCES + NARRATIVE_SOURCES,
        )
    """

    def __init__(
        self,
        local_paths: list[Path] | None = None,
        config: CorpusBuilderConfig | None = None,
        llm_provider: LLMProvider | None = None,
    ) -> None:
        """Initialize corpus builder.

        Args:
            local_paths: Directories to scan for local documents
            config: Builder configuration
            llm_provider: Optional LLM for advanced distractor generation
        """
        self.local_paths = local_paths or []
        self.config = config or CorpusBuilderConfig()
        self.llm_provider = llm_provider

        # Initialize component scanners/fetchers
        self._init_components()

    def _init_components(self) -> None:
        """Initialize scanner and fetcher components."""
        self._local_scanners: list[LocalDocumentScanner] = []

        for path in self.local_paths:
            try:
                scanner = LocalDocumentScanner(
                    root_path=path,
                    patterns=[
                        "**/*.md",
                        "**/*.py",
                        "**/*.ts",
                        "**/*.tsx",
                        "**/*.rst",
                        "**/*.pdf",  # PDF support
                        "**/*.txt",
                    ],
                    size_range=(500, 500_000),  # 500 bytes to 500KB (PDFs can be larger)
                )
                self._local_scanners.append(scanner)
            except ValueError as e:
                logger.warning(f"Could not create scanner for {path}: {e}")

        self._online_fetcher = OnlineDocumentFetcher(
            cache_dir=self.config.cache_dir,
            rate_limit=1.0,  # 1 request per second
            timeout=30.0,
        )

        self._legal_fetcher = LegalDocumentFetcher(
            cache_dir=self.config.cache_dir,
            rate_limit=1.0,
            timeout=30.0,
        )

        self._distractor_generator = DistractorGenerator(
            llm_provider=self.llm_provider,
            seed=42,  # Reproducible builds
        )

    async def build(
        self,
        online_sources: list[OnlineSource] | None = None,
        version: str = "1.0.0",
        description: str = "",
    ) -> DocumentCorpus:
        """Build complete benchmark corpus.

        Assembles documents from all configured sources, deduplicates,
        and generates synthetic distractors to meet target ratios.

        Args:
            online_sources: Override default online sources
            version: Corpus version string
            description: Human-readable description

        Returns:
            Complete DocumentCorpus ready for benchmarking
        """
        documents: list[BenchmarkDocument] = []
        progress = BuildProgress()

        # Phase 1: Local documents
        if self.config.local.enabled:
            local_docs = await self._collect_local_documents()
            documents.extend(local_docs)
            progress.local_docs = len(local_docs)
            logger.info(f"Collected {len(local_docs)} local documents")

        # Phase 2: Online documents
        if self.config.online.enabled:
            sources = online_sources or self._default_online_sources()
            online_docs = await self._collect_online_documents(sources)
            documents.extend(online_docs)
            progress.online_docs = len(online_docs)
            logger.info(f"Collected {len(online_docs)} online documents")

        # Phase 3: Legal documents
        if self.config.legal.enabled:
            legal_docs = await self._collect_legal_documents()
            documents.extend(legal_docs)
            progress.legal_docs = len(legal_docs)
            logger.info(f"Collected {len(legal_docs)} legal documents")

        # Deduplicate before adding distractors
        seen_hashes: set[str] = set()
        unique_docs: list[BenchmarkDocument] = []
        for doc in documents:
            if doc.content_hash not in seen_hashes:
                seen_hashes.add(doc.content_hash)
                unique_docs.append(doc)

        progress.duplicates_removed = len(documents) - len(unique_docs)
        documents = unique_docs
        logger.info(f"Removed {progress.duplicates_removed} duplicate documents")

        # Phase 4: Calculate and generate distractors
        if self.config.distractors.enabled:
            distractor_docs = await self._generate_distractors(
                existing_count=len(documents),
                reference_docs=documents[:50],  # Use first 50 as reference
            )
            documents.extend(distractor_docs)
            progress.distractor_docs = len(distractor_docs)
            logger.info(f"Generated {len(distractor_docs)} distractor documents")

        progress.total = len(documents)
        logger.info(f"Final corpus: {progress}")

        # Create corpus
        corpus = DocumentCorpus(
            documents=documents,
            version=version,
            description=description or f"Benchmark corpus with {len(documents)} documents",
        )

        # Optionally save
        if self.config.output_path:
            corpus.save(self.config.output_path)
            logger.info(f"Saved corpus to {self.config.output_path}")

        return corpus

    async def _collect_local_documents(self) -> list[BenchmarkDocument]:
        """Collect documents from local filesystem sources.

        Distributes the max_docs limit evenly across all scanners to ensure
        each path contributes to the corpus.
        """
        documents: list[BenchmarkDocument] = []
        max_docs = self.config.local.max_docs
        num_scanners = len(self._local_scanners)

        # Distribute max_docs evenly across scanners
        per_scanner_limit = None
        if max_docs is not None and num_scanners > 0:
            per_scanner_limit = max_docs // num_scanners

        for scanner in self._local_scanners:
            limit = per_scanner_limit
            # Last scanner can take any remaining capacity
            if max_docs is not None:
                remaining = max_docs - len(documents)
                if remaining <= 0:
                    break
                if limit is None or limit > remaining:
                    limit = remaining

            docs = scanner.scan(max_docs=limit)
            documents.extend(docs)

        return documents

    async def _collect_online_documents(
        self, sources: list[OnlineSource]
    ) -> list[BenchmarkDocument]:
        """Collect documents from online sources."""
        max_docs = self.config.online.max_docs
        return await self._online_fetcher.fetch_from_sources(sources, max_docs=max_docs)

    async def _collect_legal_documents(self) -> list[BenchmarkDocument]:
        """Collect legal documents."""
        max_docs = self.config.legal.max_docs
        return await self._legal_fetcher.fetch_all(
            max_docs=max_docs,
            include_templates=True,
        )

    async def _generate_distractors(
        self,
        existing_count: int,
        reference_docs: list[BenchmarkDocument],
    ) -> list[BenchmarkDocument]:
        """Generate synthetic distractor documents.

        Calculates required distractor count based on target ratio
        and generates documents with appropriate similarity distribution.

        Args:
            existing_count: Current document count
            reference_docs: Reference documents for context

        Returns:
            List of generated distractor documents
        """
        # Calculate target distractor count
        # If ratio is 0.4, we want distractors = 0.4 * total
        # distractors = 0.4 * (existing + distractors)
        # distractors = 0.4 * existing + 0.4 * distractors
        # 0.6 * distractors = 0.4 * existing
        # distractors = (0.4 / 0.6) * existing = (ratio / (1-ratio)) * existing
        ratio = self.config.distractor_ratio
        if ratio <= 0 or ratio >= 1:
            return []

        target_count = int((ratio / (1 - ratio)) * existing_count)

        # Apply max_docs limit if configured
        if self.config.distractors.max_docs is not None:
            target_count = min(target_count, self.config.distractors.max_docs)

        if target_count <= 0:
            return []

        # Ensure we meet minimum document requirement
        total_target = self.config.min_documents
        if existing_count + target_count < total_target:
            # Need more distractors to meet minimum
            target_count = total_target - existing_count

        # Use LLM generation if available for better quality
        if self.llm_provider:
            try:
                return await self._distractor_generator.generate_with_llm(
                    count=target_count,
                    reference_docs=reference_docs,
                )
            except Exception as e:
                logger.warning(f"LLM distractor generation failed: {e}, falling back to templates")

        # Template-based generation
        return self._distractor_generator.generate(
            count=target_count,
            # Use diverse categories from existing docs
            categories=self._extract_categories(reference_docs),
            # FR-012 distribution: 50% very different, 30% somewhat similar, 20% very similar
            similarity_distribution={
                "very_different": 0.5,
                "somewhat_similar": 0.3,
                "very_similar": 0.2,
            },
        )

    def _default_online_sources(self) -> list[OnlineSource]:
        """Get default online sources for fetching."""
        return TECHNICAL_SOURCES + NARRATIVE_SOURCES + ACADEMIC_SOURCES

    def _extract_categories(
        self, documents: list[BenchmarkDocument]
    ) -> list[DocumentCategory]:
        """Extract unique categories from documents."""
        categories: set[DocumentCategory] = set()
        for doc in documents:
            if doc.category != DocumentCategory.SYNTHETIC:
                categories.add(doc.category)

        if not categories:
            # Default categories if none found
            return [
                DocumentCategory.TECHNICAL,
                DocumentCategory.NARRATIVE,
                DocumentCategory.KNOWLEDGE_BASE,
            ]

        return list(categories)

    def get_category_coverage(self, corpus: DocumentCorpus) -> dict[str, bool]:
        """Check which categories are represented in corpus.

        Args:
            corpus: Corpus to check

        Returns:
            Dict mapping category name to presence boolean
        """
        present = {cat.value: False for cat in DocumentCategory}

        for doc in corpus.documents:
            present[doc.category.value] = True

        return present

    def validate_corpus(self, corpus: DocumentCorpus) -> list[str]:
        """Validate corpus meets FR-012 requirements.

        Args:
            corpus: Corpus to validate

        Returns:
            List of validation issues (empty if valid)
        """
        issues: list[str] = []

        # Check minimum document count
        if len(corpus) < self.config.min_documents:
            issues.append(
                f"Document count {len(corpus)} below minimum {self.config.min_documents}"
            )

        # Check category coverage
        coverage = self.get_category_coverage(corpus)
        missing = [cat for cat, present in coverage.items() if not present]
        if len(missing) > 4:  # Allow some missing categories
            issues.append(f"Too many missing categories: {missing}")

        # Check distractor ratio
        distractor_count = len(corpus.get_distractors())
        actual_ratio = distractor_count / len(corpus) if len(corpus) > 0 else 0
        target_ratio = self.config.distractor_ratio

        if abs(actual_ratio - target_ratio) > 0.1:  # Allow 10% variance
            issues.append(
                f"Distractor ratio {actual_ratio:.2f} differs from target {target_ratio:.2f}"
            )

        return issues


async def build_default_corpus(
    local_paths: list[Path] | None = None,
    cache_dir: Path | None = None,
    output_path: Path | None = None,
    min_documents: int = 500,
) -> DocumentCorpus:
    """Convenience function to build a default benchmark corpus.

    Args:
        local_paths: Directories to scan (default: ~/Development)
        cache_dir: Cache directory for online documents
        output_path: Optional path to save corpus
        min_documents: Minimum document count

    Returns:
        Built DocumentCorpus
    """
    if local_paths is None:
        dev_path = Path.home() / "Development"
        local_paths = [dev_path] if dev_path.exists() else []

    config = CorpusBuilderConfig(
        min_documents=min_documents,
        distractor_ratio=0.4,
        cache_dir=cache_dir,
        output_path=output_path,
    )

    builder = CorpusBuilder(
        local_paths=local_paths,
        config=config,
    )

    return await builder.build(
        version="1.0.0",
        description=f"Default benchmark corpus targeting {min_documents}+ documents",
    )
