"""Document corpus data models for benchmark infrastructure.

Provides core data structures for building diverse benchmark corpora
following FR-012 requirements for production-grade retrieval validation.

Key Classes:
    - BenchmarkDocument: Single document with metadata, content hash
    - DocumentCorpus: Collection with save/load, filtering, querying
    - CorpusMetadata: Statistics about corpus composition
"""

from __future__ import annotations

import hashlib
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class DocumentCategory(str, Enum):
    """Content category for corpus diversity tracking.

    Per FR-012.1, the corpus must cover 8 distinct content categories
    to validate retrieval across diverse content types.
    """

    TECHNICAL = "technical"  # Code, API docs, READMEs
    NARRATIVE = "narrative"  # Stories, fiction, Wikipedia articles
    KNOWLEDGE_BASE = "knowledge_base"  # FAQs, Stack Overflow, how-to
    LEGAL = "legal"  # ToS, contracts, court opinions, regulations
    CONVERSATIONAL = "conversational"  # Chat, email, support tickets
    ACADEMIC = "academic"  # arXiv, research papers, scientific articles
    NEWS_BLOG = "news_blog"  # Blog posts, news articles
    SYNTHETIC = "synthetic"  # LLM-generated distractors


class DocumentSource(str, Enum):
    """Origin of document content."""

    LOCAL = "local"  # Local filesystem (~/Development)
    ONLINE = "online"  # Fetched from web (docs, Wikipedia, etc.)
    SYNTHETIC = "synthetic"  # LLM-generated content


@dataclass
class BenchmarkDocument:
    """Single document in benchmark corpus.

    Stores document content with metadata for retrieval evaluation.
    Content hash enables deduplication across sources.

    Attributes:
        doc_id: Unique identifier for this document
        source: Origin (local, online, synthetic)
        category: Content type (technical, legal, narrative, etc.)
        domain: Specific topic (python, contract_law, fiction, etc.)
        file_path: Original file path or URL
        content: Full document text
        chunk_ids: IDs of chunks created from this document
        metadata: Source-specific metadata (author, date, etc.)
        is_distractor: True if synthetic distractor document
        semantic_tags: Tags for relevance judgment
        content_hash: SHA256 hash of content (first 16 chars)
        size_bytes: Content size in bytes
        created_at: When document was added to corpus
    """

    doc_id: str
    source: DocumentSource
    category: DocumentCategory
    domain: str
    file_path: str
    content: str
    chunk_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    is_distractor: bool = False
    semantic_tags: list[str] = field(default_factory=list)
    content_hash: str = field(default="", init=False)
    size_bytes: int = field(default=0, init=False)
    created_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """Calculate content hash and size after initialization."""
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        self.size_bytes = len(self.content.encode())

    def to_dict(self) -> dict[str, Any]:
        """Serialize document to dictionary for JSON export."""
        return {
            "doc_id": self.doc_id,
            "source": self.source.value,
            "category": self.category.value,
            "domain": self.domain,
            "file_path": self.file_path,
            "content": self.content,
            "chunk_ids": self.chunk_ids,
            "metadata": self.metadata,
            "is_distractor": self.is_distractor,
            "semantic_tags": self.semantic_tags,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkDocument:
        """Deserialize document from dictionary."""
        doc = cls(
            doc_id=data["doc_id"],
            source=DocumentSource(data["source"]),
            category=DocumentCategory(data["category"]),
            domain=data["domain"],
            file_path=data["file_path"],
            content=data["content"],
            chunk_ids=data.get("chunk_ids", []),
            metadata=data.get("metadata", {}),
            is_distractor=data.get("is_distractor", False),
            semantic_tags=data.get("semantic_tags", []),
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
        )
        # Restore computed fields (they're recalculated in __post_init__
        # but should match the serialized values)
        return doc


@dataclass
class CorpusMetadata:
    """Statistics about benchmark corpus composition.

    Tracks distribution of documents across sources, categories, and domains
    to validate corpus diversity requirements.

    Attributes:
        total_documents: Total document count
        source_distribution: Count per source (local, online, synthetic)
        category_distribution: Count per category (technical, legal, etc.)
        domain_distribution: Count per domain (python, contract_law, etc.)
        distractor_count: Number of distractor documents
        distractor_ratio: Ratio of distractors to total
        size_stats: Min/max/mean/median document sizes
        created_at: When corpus metadata was calculated
    """

    total_documents: int
    source_distribution: dict[str, int]
    category_distribution: dict[str, int]
    domain_distribution: dict[str, int]
    distractor_count: int
    distractor_ratio: float
    size_stats: dict[str, float]
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata to dictionary for JSON export."""
        return {
            "total_documents": self.total_documents,
            "source_distribution": self.source_distribution,
            "category_distribution": self.category_distribution,
            "domain_distribution": self.domain_distribution,
            "distractor_count": self.distractor_count,
            "distractor_ratio": self.distractor_ratio,
            "size_stats": self.size_stats,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorpusMetadata:
        """Deserialize metadata from dictionary."""
        return cls(
            total_documents=data["total_documents"],
            source_distribution=data["source_distribution"],
            category_distribution=data["category_distribution"],
            domain_distribution=data["domain_distribution"],
            distractor_count=data["distractor_count"],
            distractor_ratio=data["distractor_ratio"],
            size_stats=data["size_stats"],
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(),
        )

    @classmethod
    def calculate(cls, documents: list[BenchmarkDocument]) -> CorpusMetadata:
        """Calculate metadata from list of documents.

        Args:
            documents: List of documents to analyze

        Returns:
            CorpusMetadata with calculated statistics
        """
        if not documents:
            return cls(
                total_documents=0,
                source_distribution={},
                category_distribution={},
                domain_distribution={},
                distractor_count=0,
                distractor_ratio=0.0,
                size_stats={"min": 0, "max": 0, "mean": 0, "median": 0},
            )

        # Count distributions
        source_dist: dict[str, int] = {}
        category_dist: dict[str, int] = {}
        domain_dist: dict[str, int] = {}
        distractor_count = 0
        sizes: list[int] = []

        for doc in documents:
            source_key = doc.source.value
            source_dist[source_key] = source_dist.get(source_key, 0) + 1

            category_key = doc.category.value
            category_dist[category_key] = category_dist.get(category_key, 0) + 1

            domain_dist[doc.domain] = domain_dist.get(doc.domain, 0) + 1

            if doc.is_distractor:
                distractor_count += 1

            sizes.append(doc.size_bytes)

        total = len(documents)

        return cls(
            total_documents=total,
            source_distribution=source_dist,
            category_distribution=category_dist,
            domain_distribution=domain_dist,
            distractor_count=distractor_count,
            distractor_ratio=distractor_count / total if total > 0 else 0.0,
            size_stats={
                "min": min(sizes),
                "max": max(sizes),
                "mean": statistics.mean(sizes),
                "median": statistics.median(sizes),
            },
        )


@dataclass
class DocumentCorpus:
    """Collection of benchmark documents with querying and persistence.

    Provides save/load functionality, filtering by domain/category,
    and distractor selection for benchmark execution.

    Attributes:
        documents: List of all documents in corpus
        metadata: Calculated statistics about corpus
        version: Corpus version string
        description: Human-readable description
    """

    documents: list[BenchmarkDocument]
    metadata: CorpusMetadata = field(default=None)  # type: ignore
    version: str = "1.0.0"
    description: str = ""

    def __post_init__(self) -> None:
        """Calculate metadata if not provided."""
        if self.metadata is None:
            self.metadata = CorpusMetadata.calculate(self.documents)

    def __len__(self) -> int:
        """Return document count."""
        return len(self.documents)

    def save(self, path: str | Path) -> None:
        """Save corpus to JSON file.

        Args:
            path: File path for JSON output
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "version": self.version,
            "description": self.description,
            "metadata": self.metadata.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> DocumentCorpus:
        """Load corpus from JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded DocumentCorpus
        """
        path = Path(path)

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        documents = [BenchmarkDocument.from_dict(d) for d in data["documents"]]
        metadata = CorpusMetadata.from_dict(data["metadata"])

        return cls(
            documents=documents,
            metadata=metadata,
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
        )

    def get_document(self, doc_id: str) -> BenchmarkDocument | None:
        """Retrieve document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document if found, None otherwise
        """
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_by_category(self, category: DocumentCategory) -> list[BenchmarkDocument]:
        """Filter documents by content category.

        Args:
            category: Category to filter by

        Returns:
            List of matching documents
        """
        return [doc for doc in self.documents if doc.category == category]

    def get_by_domain(self, domain: str) -> list[BenchmarkDocument]:
        """Filter documents by domain.

        Args:
            domain: Domain to filter by (e.g., "python", "contract_law")

        Returns:
            List of matching documents
        """
        return [doc for doc in self.documents if doc.domain == domain]

    def get_by_source(self, source: DocumentSource) -> list[BenchmarkDocument]:
        """Filter documents by source.

        Args:
            source: Source to filter by

        Returns:
            List of matching documents
        """
        return [doc for doc in self.documents if doc.source == source]

    def get_distractors(
        self, count: int | None = None, domain: str | None = None
    ) -> list[BenchmarkDocument]:
        """Get distractor documents, optionally filtered by domain.

        Args:
            count: Maximum number to return (None = all)
            domain: Optional domain filter

        Returns:
            List of distractor documents
        """
        distractors = [doc for doc in self.documents if doc.is_distractor]

        if domain:
            distractors = [d for d in distractors if d.domain == domain]

        if count is not None:
            distractors = distractors[:count]

        return distractors

    def get_non_distractors(self) -> list[BenchmarkDocument]:
        """Get all non-distractor documents.

        Returns:
            List of real (non-synthetic distractor) documents
        """
        return [doc for doc in self.documents if not doc.is_distractor]

    def deduplicate(self) -> int:
        """Remove duplicate documents based on content hash.

        Returns:
            Number of duplicates removed
        """
        seen_hashes: set[str] = set()
        unique_docs: list[BenchmarkDocument] = []

        for doc in self.documents:
            if doc.content_hash not in seen_hashes:
                seen_hashes.add(doc.content_hash)
                unique_docs.append(doc)

        removed = len(self.documents) - len(unique_docs)
        self.documents = unique_docs
        self.metadata = CorpusMetadata.calculate(self.documents)

        return removed

    def add_document(self, doc: BenchmarkDocument) -> bool:
        """Add document to corpus if not duplicate.

        Args:
            doc: Document to add

        Returns:
            True if added, False if duplicate
        """
        for existing in self.documents:
            if existing.content_hash == doc.content_hash:
                return False

        self.documents.append(doc)
        self.metadata = CorpusMetadata.calculate(self.documents)
        return True

    def get_category_distribution(self) -> dict[DocumentCategory, int]:
        """Get document count by category.

        Returns:
            Dict mapping category to count
        """
        dist: dict[DocumentCategory, int] = {}
        for doc in self.documents:
            dist[doc.category] = dist.get(doc.category, 0) + 1
        return dist
