# FR-012: Production-Grade Retrieval Benchmark - Technical Implementation Plan

**Status:** Ready for Implementation
**Priority:** Critical
**Created:** 2026-01-02
**Estimated Effort:** 4 weeks (1 engineer)
**Dependencies:** FR-009 (Integration Testing Framework), RAGAS library, Ollama

---

## Overview

This document provides detailed technical implementation guidance for FR-012 (Production-Grade Retrieval Pipeline Benchmark). It breaks down the 8-phase implementation into specific tasks with code patterns, architectural decisions, and testing strategies.

**Goal:** Build a benchmark suite that proves draagon-ai's retrieval pipeline works at production scale (500+ docs, multi-hop queries, RAGAS metrics) and leads the industry.

---

## Architecture Overview

```
src/draagon_ai/testing/benchmarks/
├── __init__.py                    # Public API exports
├── corpus.py                      # DocumentCorpus, CorpusBuilder
├── queries.py                     # Query types (MultiHop, ZeroResult, Adversarial)
├── evaluation.py                  # RAGASEvaluator, IndustryComparison
├── runner.py                      # BenchmarkRunner, StatisticalValidator
├── statistics.py                  # Statistical analysis, p-values, CI
├── reporting.py                   # Markdown/CSV report generation
├── downloaders/
│   ├── __init__.py
│   ├── local_scanner.py           # Scan ~/Development
│   ├── online_fetcher.py          # Download docs from URLs
│   └── distractor_generator.py   # LLM-generated synthetic docs
└── integration/
    ├── __init__.py
    ├── embedding_validator.py     # Validate MTEB-benchmarked models
    └── ci_smoke_tests.py          # Fast 50-query smoke tests

tests/benchmarks/
├── conftest.py                    # Fixtures for benchmark tests
├── test_corpus_builder.py         # Test corpus assembly
├── test_query_suite.py            # Test query generation
├── test_ragas_evaluation.py       # Test RAGAS metrics
├── test_retrieval_smoke.py        # CI smoke tests (5 min)
└── test_retrieval_full.py         # Full benchmark (30 min)

.specify/benchmarks/
├── corpus/
│   ├── corpus_v1.json             # Cached 500+ document corpus
│   └── metadata.json              # Corpus statistics
├── queries/
│   ├── multi_hop_v1.json          # 50+ multi-hop queries
│   ├── zero_result_v1.json        # 25+ zero-result queries
│   └── adversarial_v1.json        # 40+ adversarial queries
└── results/
    ├── run_2026-01-02.md          # Markdown report
    ├── run_2026-01-02.csv         # CSV export
    └── history.json               # Historical tracking
```

---

## Phase 1: Corpus Assembly Infrastructure (Week 1, Days 1-3)

### Task 1.1: Document Data Models

**File:** `src/draagon_ai/testing/benchmarks/corpus.py`

**Implementation:**

```python
"""Benchmark corpus data models and builder."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
import hashlib
import json


class DocumentSource(str, Enum):
    """Source type for benchmark documents."""
    LOCAL = "local"
    ONLINE = "online"
    SYNTHETIC = "synthetic"


class DocumentDomain(str, Enum):
    """Domain categorization for documents."""
    PYTHON = "python"
    JAVA = "java"
    TYPESCRIPT = "typescript"
    AI_FRAMEWORK = "ai_framework"
    GAME_DESIGN = "game_design"
    WEB_FRAMEWORK = "web_framework"
    DATABASE = "database"
    DEVOPS = "devops"
    FRONTEND = "frontend"
    UNKNOWN = "unknown"


@dataclass
class BenchmarkDocument:
    """Document in benchmark corpus.

    Represents a single document with metadata for retrieval evaluation.
    Each document can be chunked for semantic search.
    """

    doc_id: str
    source: DocumentSource
    domain: DocumentDomain
    file_path: str  # Original path or URL
    content: str
    chunk_ids: list[str] = field(default_factory=list)
    is_distractor: bool = False
    semantic_tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    content_hash: str = field(init=False)
    size_bytes: int = field(init=False)

    def __post_init__(self):
        """Calculate content hash and size."""
        self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        self.size_bytes = len(self.content.encode())

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        return {
            "doc_id": self.doc_id,
            "source": self.source.value,
            "domain": self.domain.value,
            "file_path": self.file_path,
            "content": self.content,
            "chunk_ids": self.chunk_ids,
            "is_distractor": self.is_distractor,
            "semantic_tags": self.semantic_tags,
            "metadata": self.metadata,
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BenchmarkDocument":
        """Deserialize from dict."""
        return cls(
            doc_id=data["doc_id"],
            source=DocumentSource(data["source"]),
            domain=DocumentDomain(data["domain"]),
            file_path=data["file_path"],
            content=data["content"],
            chunk_ids=data.get("chunk_ids", []),
            is_distractor=data.get("is_distractor", False),
            semantic_tags=data.get("semantic_tags", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class CorpusMetadata:
    """Metadata about the benchmark corpus."""

    version: str
    created_at: str
    total_documents: int
    source_distribution: dict[DocumentSource, int]
    domain_distribution: dict[DocumentDomain, int]
    size_distribution: dict[str, int]  # "min", "max", "mean", "median"
    distractor_count: int
    distractor_ratio: float

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "version": self.version,
            "created_at": self.created_at,
            "total_documents": self.total_documents,
            "source_distribution": {k.value: v for k, v in self.source_distribution.items()},
            "domain_distribution": {k.value: v for k, v in self.domain_distribution.items()},
            "size_distribution": self.size_distribution,
            "distractor_count": self.distractor_count,
            "distractor_ratio": self.distractor_ratio,
        }


@dataclass
class DocumentCorpus:
    """Collection of benchmark documents.

    Provides methods for loading, saving, filtering, and querying the corpus.
    """

    documents: list[BenchmarkDocument]
    metadata: CorpusMetadata
    version: str = "v1"

    def __len__(self) -> int:
        return len(self.documents)

    def get_document(self, doc_id: str) -> Optional[BenchmarkDocument]:
        """Retrieve document by ID."""
        for doc in self.documents:
            if doc.doc_id == doc_id:
                return doc
        return None

    def get_by_domain(self, domain: DocumentDomain) -> list[BenchmarkDocument]:
        """Filter documents by domain."""
        return [doc for doc in self.documents if doc.domain == domain]

    def get_distractors(self, domain: Optional[DocumentDomain] = None, count: int = 10) -> list[BenchmarkDocument]:
        """Get distractor documents, optionally filtered by domain."""
        distractors = [doc for doc in self.documents if doc.is_distractor]
        if domain:
            distractors = [doc for doc in distractors if doc.domain == domain]
        return distractors[:count]

    def save(self, path: Path) -> None:
        """Save corpus to JSON file."""
        data = {
            "version": self.version,
            "metadata": self.metadata.to_dict(),
            "documents": [doc.to_dict() for doc in self.documents],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "DocumentCorpus":
        """Load corpus from JSON file."""
        with open(path) as f:
            data = json.load(f)

        # Reconstruct metadata
        meta_dict = data["metadata"]
        metadata = CorpusMetadata(
            version=meta_dict["version"],
            created_at=meta_dict["created_at"],
            total_documents=meta_dict["total_documents"],
            source_distribution={
                DocumentSource(k): v for k, v in meta_dict["source_distribution"].items()
            },
            domain_distribution={
                DocumentDomain(k): v for k, v in meta_dict["domain_distribution"].items()
            },
            size_distribution=meta_dict["size_distribution"],
            distractor_count=meta_dict["distractor_count"],
            distractor_ratio=meta_dict["distractor_ratio"],
        )

        # Reconstruct documents
        documents = [BenchmarkDocument.from_dict(d) for d in data["documents"]]

        return cls(documents=documents, metadata=metadata, version=data["version"])
```

**Tests:** `tests/benchmarks/test_corpus_builder.py`

```python
import pytest
from draagon_ai.testing.benchmarks.corpus import (
    BenchmarkDocument,
    DocumentCorpus,
    DocumentSource,
    DocumentDomain,
)


def test_document_content_hash():
    """Test content hash is deterministic."""
    doc1 = BenchmarkDocument(
        doc_id="doc1",
        source=DocumentSource.LOCAL,
        domain=DocumentDomain.PYTHON,
        file_path="/test/file.py",
        content="print('hello')",
    )
    doc2 = BenchmarkDocument(
        doc_id="doc2",
        source=DocumentSource.LOCAL,
        domain=DocumentDomain.PYTHON,
        file_path="/test/file2.py",
        content="print('hello')",
    )
    assert doc1.content_hash == doc2.content_hash  # Same content


def test_corpus_save_load(tmp_path):
    """Test corpus serialization round-trip."""
    from datetime import datetime

    doc = BenchmarkDocument(
        doc_id="doc1",
        source=DocumentSource.LOCAL,
        domain=DocumentDomain.PYTHON,
        file_path="/test.py",
        content="test content",
        semantic_tags=["python", "test"],
    )

    from draagon_ai.testing.benchmarks.corpus import CorpusMetadata

    metadata = CorpusMetadata(
        version="v1",
        created_at=datetime.now().isoformat(),
        total_documents=1,
        source_distribution={DocumentSource.LOCAL: 1},
        domain_distribution={DocumentDomain.PYTHON: 1},
        size_distribution={"min": 100, "max": 100, "mean": 100, "median": 100},
        distractor_count=0,
        distractor_ratio=0.0,
    )

    corpus = DocumentCorpus(documents=[doc], metadata=metadata)

    # Save and load
    path = tmp_path / "corpus.json"
    corpus.save(path)
    loaded = DocumentCorpus.load(path)

    assert len(loaded) == 1
    assert loaded.documents[0].doc_id == "doc1"
    assert loaded.documents[0].content == "test content"
```

**Acceptance Criteria:**
- [ ] `BenchmarkDocument` model implemented with content hashing
- [ ] `DocumentCorpus` supports save/load with JSON
- [ ] Corpus filtering by domain, source, distractor status
- [ ] Metadata tracks source/domain distributions
- [ ] Round-trip serialization preserves all fields

---

### Task 1.2: Local Document Scanner

**File:** `src/draagon_ai/testing/benchmarks/downloaders/local_scanner.py`

**Implementation:**

```python
"""Scan local filesystem for benchmark documents."""

import logging
from pathlib import Path
from typing import Optional
import re

from draagon_ai.testing.benchmarks.corpus import (
    BenchmarkDocument,
    DocumentSource,
    DocumentDomain,
)

logger = logging.getLogger(__name__)


class LocalDocumentScanner:
    """Scan local filesystem for documents matching criteria.

    Filters by:
    - File patterns (*.md, *.py, *.java, etc.)
    - Size range (1KB-500KB)
    - Exclusion patterns (node_modules, target, .archive)
    """

    def __init__(
        self,
        root_path: Path,
        patterns: list[str],
        size_range: tuple[int, int] = (1024, 500_000),
        exclude_patterns: Optional[list[str]] = None,
    ):
        self.root_path = Path(root_path).expanduser()
        self.patterns = patterns
        self.min_size, self.max_size = size_range
        self.exclude_patterns = exclude_patterns or [
            "**/target/**",
            "**/node_modules/**",
            "**/.archive/**",
            "**/build/**",
            "**/__pycache__/**",
        ]

    async def scan(self, max_docs: int = 300) -> list[BenchmarkDocument]:
        """Scan filesystem and return matching documents.

        Args:
            max_docs: Maximum documents to collect

        Returns:
            List of BenchmarkDocument instances
        """
        logger.info(f"Scanning {self.root_path} for documents (max={max_docs})")

        documents = []
        seen_hashes = set()

        for pattern in self.patterns:
            for file_path in self.root_path.glob(pattern):
                if len(documents) >= max_docs:
                    break

                # Check exclusions
                if self._is_excluded(file_path):
                    continue

                # Check size
                size = file_path.stat().st_size
                if not (self.min_size <= size <= self.max_size):
                    continue

                # Read content
                try:
                    content = file_path.read_text(encoding="utf-8")
                except (UnicodeDecodeError, PermissionError) as e:
                    logger.warning(f"Skipping {file_path}: {e}")
                    continue

                # Deduplicate by content hash
                import hashlib

                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                if content_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate: {file_path}")
                    continue
                seen_hashes.add(content_hash)

                # Create document
                domain = self._infer_domain(file_path)
                doc_id = self._generate_doc_id(file_path)

                doc = BenchmarkDocument(
                    doc_id=doc_id,
                    source=DocumentSource.LOCAL,
                    domain=domain,
                    file_path=str(file_path),
                    content=content,
                    is_distractor=False,  # Local docs are NOT distractors
                    semantic_tags=self._extract_tags(file_path, content),
                    metadata={"size_bytes": size, "file_type": file_path.suffix},
                )

                documents.append(doc)
                logger.debug(f"Added document: {doc_id} ({domain.value}, {size} bytes)")

        logger.info(f"Scanned {len(documents)} documents from local filesystem")
        return documents

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if file matches exclusion patterns."""
        for pattern in self.exclude_patterns:
            if file_path.match(pattern):
                return True
        return False

    def _infer_domain(self, file_path: Path) -> DocumentDomain:
        """Infer domain from file path heuristics."""
        path_str = str(file_path).lower()

        if "draagon" in path_str or "cognitive" in path_str or "agent" in path_str:
            return DocumentDomain.AI_FRAMEWORK
        elif "party-lore" in path_str or "game" in path_str:
            return DocumentDomain.GAME_DESIGN
        elif "metaobjects" in path_str and ".java" in path_str:
            return DocumentDomain.JAVA
        elif ".py" in path_str:
            return DocumentDomain.PYTHON
        elif ".ts" in path_str or ".tsx" in path_str or "react" in path_str:
            return DocumentDomain.TYPESCRIPT
        else:
            return DocumentDomain.UNKNOWN

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path."""
        # Use relative path from root, replace slashes with underscores
        rel_path = file_path.relative_to(self.root_path)
        doc_id = str(rel_path).replace("/", "_").replace("\\", "_")
        # Remove extension
        doc_id = doc_id.rsplit(".", 1)[0]
        return doc_id

    def _extract_tags(self, file_path: Path, content: str) -> list[str]:
        """Extract semantic tags from file content."""
        tags = []

        # Add file type
        tags.append(file_path.suffix[1:])  # Remove leading dot

        # Add common keywords from content (first 500 chars)
        snippet = content[:500].lower()
        keywords = ["async", "class", "function", "interface", "test", "api", "database"]
        for keyword in keywords:
            if keyword in snippet:
                tags.append(keyword)

        return list(set(tags))  # Deduplicate
```

**Tests:** Add to `tests/benchmarks/test_corpus_builder.py`

```python
import pytest
from pathlib import Path
from draagon_ai.testing.benchmarks.downloaders.local_scanner import LocalDocumentScanner


@pytest.mark.asyncio
async def test_local_scanner_finds_documents(tmp_path):
    """Test scanner finds matching documents."""
    # Create test files
    (tmp_path / "test1.md").write_text("# Test Document 1\n\nThis is a test.")
    (tmp_path / "test2.py").write_text("def test(): pass")
    (tmp_path / "too_small.md").write_text("x")  # Too small
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "excluded.md").write_text("Should be excluded")

    scanner = LocalDocumentScanner(
        root_path=tmp_path,
        patterns=["**/*.md", "**/*.py"],
        size_range=(10, 500_000),
    )

    docs = await scanner.scan(max_docs=100)

    assert len(docs) == 2  # test1.md, test2.py
    assert all(doc.source.value == "local" for doc in docs)
    assert not any("node_modules" in doc.file_path for doc in docs)


@pytest.mark.asyncio
async def test_local_scanner_deduplicates(tmp_path):
    """Test scanner removes duplicate content."""
    (tmp_path / "test1.md").write_text("Same content")
    (tmp_path / "test2.md").write_text("Same content")

    scanner = LocalDocumentScanner(
        root_path=tmp_path,
        patterns=["**/*.md"],
    )

    docs = await scanner.scan(max_docs=100)

    assert len(docs) == 1  # Deduplicated
```

**Acceptance Criteria:**
- [ ] Scans local filesystem with glob patterns
- [ ] Filters by size range (1KB-500KB)
- [ ] Excludes patterns (node_modules, target, .archive)
- [ ] Deduplicates by content hash
- [ ] Infers domain from file path heuristics
- [ ] Extracts semantic tags from content
- [ ] Handles Unicode errors gracefully

---

### Task 1.3: Online Documentation Downloader

**File:** `src/draagon_ai/testing/benchmarks/downloaders/online_fetcher.py`

**Implementation:**

```python
"""Download documentation from online sources."""

import logging
from pathlib import Path
from typing import Optional
import asyncio
import hashlib
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

from draagon_ai.testing.benchmarks.corpus import (
    BenchmarkDocument,
    DocumentSource,
    DocumentDomain,
)

logger = logging.getLogger(__name__)


class OnlineDocumentationFetcher:
    """Download documentation pages from online sources.

    Crawls documentation sites with depth limit, caches locally,
    converts HTML to markdown format.
    """

    def __init__(
        self,
        cache_dir: Path,
        max_depth: int = 2,
        timeout_seconds: int = 30,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_depth = max_depth
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)

    async def fetch_documentation(
        self,
        domain: DocumentDomain,
        base_url: str,
        max_docs: int = 50,
    ) -> list[BenchmarkDocument]:
        """Fetch documentation from a base URL.

        Args:
            domain: Document domain classification
            base_url: Starting URL for crawl
            max_docs: Maximum documents to fetch

        Returns:
            List of BenchmarkDocument instances
        """
        logger.info(f"Fetching documentation from {base_url} (max={max_docs})")

        documents = []
        visited = set()
        to_visit = [(base_url, 0)]  # (url, depth)

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            while to_visit and len(documents) < max_docs:
                url, depth = to_visit.pop(0)

                if url in visited or depth > self.max_depth:
                    continue
                visited.add(url)

                # Check cache first
                cached_doc = self._load_from_cache(url, domain)
                if cached_doc:
                    documents.append(cached_doc)
                    logger.debug(f"Loaded from cache: {url}")
                    continue

                # Fetch URL
                try:
                    doc = await self._fetch_url(session, url, domain)
                    if doc:
                        documents.append(doc)
                        self._save_to_cache(url, doc)

                        # Extract links for next depth
                        if depth < self.max_depth:
                            links = self._extract_links(doc.content, base_url)
                            for link in links:
                                if link not in visited:
                                    to_visit.append((link, depth + 1))

                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {e}")
                    continue

        logger.info(f"Fetched {len(documents)} documents from {base_url}")
        return documents

    async def _fetch_url(
        self,
        session: aiohttp.ClientSession,
        url: str,
        domain: DocumentDomain,
    ) -> Optional[BenchmarkDocument]:
        """Fetch single URL and convert to document."""
        async with session.get(url) as response:
            if response.status != 200:
                logger.warning(f"HTTP {response.status} for {url}")
                return None

            html = await response.text()

        # Convert HTML to markdown-like text
        soup = BeautifulSoup(html, "html.parser")

        # Remove script, style, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Basic cleanup
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        content = "\n\n".join(lines)

        if len(content) < 100:  # Too short
            return None

        doc_id = self._url_to_doc_id(url)

        return BenchmarkDocument(
            doc_id=doc_id,
            source=DocumentSource.ONLINE,
            domain=domain,
            file_path=url,
            content=content,
            is_distractor=False,
            semantic_tags=[domain.value, "documentation"],
            metadata={"url": url, "size_bytes": len(content)},
        )

    def _extract_links(self, content: str, base_url: str) -> list[str]:
        """Extract links from HTML content (placeholder - would need real implementation)."""
        # In production, use BeautifulSoup to extract <a> tags
        # For now, return empty to keep crawl simple
        return []

    def _url_to_doc_id(self, url: str) -> str:
        """Convert URL to document ID."""
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        return f"online_{parsed.netloc}_{path}" if path else f"online_{parsed.netloc}_index"

    def _get_cache_path(self, url: str) -> Path:
        """Get cache file path for URL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.cache_dir / f"{url_hash}.json"

    def _load_from_cache(self, url: str, domain: DocumentDomain) -> Optional[BenchmarkDocument]:
        """Load document from cache if exists."""
        cache_path = self._get_cache_path(url)
        if not cache_path.exists():
            return None

        try:
            import json

            with open(cache_path) as f:
                data = json.load(f)
            return BenchmarkDocument.from_dict(data)
        except Exception as e:
            logger.warning(f"Failed to load cache for {url}: {e}")
            return None

    def _save_to_cache(self, url: str, doc: BenchmarkDocument) -> None:
        """Save document to cache."""
        cache_path = self._get_cache_path(url)
        try:
            import json

            with open(cache_path, "w") as f:
                json.dump(doc.to_dict(), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache for {url}: {e}")
```

**Note:** For simplicity, this implementation doesn't fully crawl (link extraction is placeholder). In production, you'd use BeautifulSoup to find `<a>` tags and filter same-domain links.

**Acceptance Criteria:**
- [ ] Downloads documentation from URLs
- [ ] Converts HTML to plain text
- [ ] Caches downloaded docs locally (avoid re-fetching)
- [ ] Respects max_depth and max_docs limits
- [ ] Handles HTTP errors gracefully
- [ ] Generates unique doc IDs from URLs

---

### Task 1.4: Synthetic Distractor Generator

**File:** `src/draagon_ai/testing/benchmarks/downloaders/distractor_generator.py`

**Implementation:**

```python
"""Generate synthetic distractor documents using LLM."""

import logging
from typing import Optional
import asyncio

from draagon_ai.llm.base import LLMProvider
from draagon_ai.testing.benchmarks.corpus import (
    BenchmarkDocument,
    DocumentSource,
    DocumentDomain,
)

logger = logging.getLogger(__name__)


class DistractorGenerator:
    """Generate synthetic distractor documents using LLM.

    Creates technical documentation on topics NOT in the corpus
    to test retrieval robustness against semantic similarity.
    """

    def __init__(self, llm_provider: LLMProvider):
        self.llm = llm_provider

    async def generate(
        self,
        topics: list[str],
        count: int = 100,
        similarity_distribution: Optional[dict[str, float]] = None,
    ) -> list[BenchmarkDocument]:
        """Generate distractor documents.

        Args:
            topics: Topics to generate docs about (e.g., "kubernetes", "react")
            count: Total number of distractors to generate
            similarity_distribution: Ratio of "very_different", "somewhat_similar", "very_similar"

        Returns:
            List of synthetic BenchmarkDocument instances
        """
        if similarity_distribution is None:
            similarity_distribution = {
                "very_different": 0.5,
                "somewhat_similar": 0.3,
                "very_similar": 0.2,
            }

        # Calculate counts for each similarity level
        counts = {
            level: int(count * ratio) for level, ratio in similarity_distribution.items()
        }

        logger.info(f"Generating {count} distractor documents across {len(topics)} topics")

        documents = []
        for level, level_count in counts.items():
            docs_per_topic = level_count // len(topics)

            for topic in topics:
                for i in range(docs_per_topic):
                    doc = await self._generate_single(topic, level, i)
                    if doc:
                        documents.append(doc)

                    if len(documents) >= count:
                        break
                if len(documents) >= count:
                    break

        logger.info(f"Generated {len(documents)} distractor documents")
        return documents

    async def _generate_single(
        self, topic: str, similarity_level: str, index: int
    ) -> Optional[BenchmarkDocument]:
        """Generate single distractor document."""
        prompt = self._build_prompt(topic, similarity_level)

        try:
            content = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model_tier="fast",
                temperature=0.8,  # More variety
                max_tokens=2000,
            )

            if len(content) < 200:
                logger.warning(f"Generated content too short for {topic}")
                return None

            doc_id = f"distractor_{topic}_{similarity_level}_{index}"
            domain = self._infer_domain(topic)

            return BenchmarkDocument(
                doc_id=doc_id,
                source=DocumentSource.SYNTHETIC,
                domain=domain,
                file_path=f"synthetic://{topic}/{index}",
                content=content,
                is_distractor=True,
                semantic_tags=[topic, similarity_level, "synthetic"],
                metadata={"topic": topic, "similarity_level": similarity_level},
            )

        except Exception as e:
            logger.error(f"Failed to generate distractor for {topic}: {e}")
            return None

    def _build_prompt(self, topic: str, similarity_level: str) -> str:
        """Build LLM prompt for distractor generation."""
        if similarity_level == "very_different":
            return f"""Write technical documentation about {topic} for beginners.
Include setup instructions, basic concepts, and simple examples.
Length: 500-1000 words.
Style: Clear, tutorial-like."""

        elif similarity_level == "somewhat_similar":
            return f"""Write advanced technical documentation about {topic} internals.
Include architecture details, design patterns, and performance considerations.
Use terms like: memory, optimization, architecture, system, design.
Length: 500-1000 words.
Style: Technical, detailed."""

        else:  # very_similar
            return f"""Write documentation about {topic} with heavy use of AI/agent terminology.
Include: cognitive architecture, decision making, memory systems, learning, agents.
Even though it's about {topic}, use AI/agent vocabulary throughout.
Length: 500-1000 words.
Style: Technical, AI-focused."""

    def _infer_domain(self, topic: str) -> DocumentDomain:
        """Infer domain from topic."""
        topic_lower = topic.lower()
        if topic_lower in ["kubernetes", "docker", "terraform"]:
            return DocumentDomain.DEVOPS
        elif topic_lower in ["react", "vue", "angular"]:
            return DocumentDomain.FRONTEND
        elif topic_lower in ["postgresql", "redis", "mongodb"]:
            return DocumentDomain.DATABASE
        else:
            return DocumentDomain.UNKNOWN
```

**Acceptance Criteria:**
- [ ] Generates synthetic docs using LLM
- [ ] Supports similarity distribution (very different, somewhat similar, very similar)
- [ ] Marks all generated docs as `is_distractor=True`
- [ ] Infers domain from topic
- [ ] Handles LLM failures gracefully
- [ ] Generates 500-1000 word documents

---

(Continuing in next message due to length...)

### Task 1.5: Corpus Builder Orchestrator

**File:** `src/draagon_ai/testing/benchmarks/corpus.py` (add to existing)

**Implementation:**

```python
class CorpusBuilder:
    """Orchestrates corpus assembly from multiple sources.

    Coordinates local scanning, online fetching, and distractor generation
    to build a balanced benchmark corpus.
    """

    def __init__(self, target_size: int, cache_dir: Path):
        self.target_size = target_size
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.documents: list[BenchmarkDocument] = []

    async def add_local_documents(
        self,
        root_path: Path,
        patterns: list[str],
        size_range: tuple[int, int] = (1024, 500_000),
        exclude_patterns: Optional[list[str]] = None,
        max_docs: int = 300,
    ) -> None:
        """Add local filesystem documents."""
        from draagon_ai.testing.benchmarks.downloaders.local_scanner import LocalDocumentScanner

        scanner = LocalDocumentScanner(
            root_path=root_path,
            patterns=patterns,
            size_range=size_range,
            exclude_patterns=exclude_patterns,
        )

        docs = await scanner.scan(max_docs=max_docs)
        self.documents.extend(docs)
        logger.info(f"Added {len(docs)} local documents (total: {len(self.documents)})")

    async def add_online_documentation(
        self,
        sources: list[tuple[str, str, int]],  # (domain, url, max_depth)
        max_docs: int = 200,
    ) -> None:
        """Add online documentation sources."""
        from draagon_ai.testing.benchmarks.downloaders.online_fetcher import (
            OnlineDocumentationFetcher,
        )

        fetcher = OnlineDocumentationFetcher(
            cache_dir=self.cache_dir / "online_cache",
            max_depth=2,
        )

        all_docs = []
        for domain_str, url, max_depth in sources:
            domain = DocumentDomain(domain_str)
            docs = await fetcher.fetch_documentation(
                domain=domain, base_url=url, max_docs=max_docs // len(sources)
            )
            all_docs.extend(docs)

        self.documents.extend(all_docs[:max_docs])
        logger.info(f"Added {len(all_docs[:max_docs])} online documents (total: {len(self.documents)})")

    async def add_synthetic_distractors(
        self,
        llm_provider: LLMProvider,
        topics: list[str],
        count: int = 100,
        similarity_distribution: Optional[dict[str, float]] = None,
    ) -> None:
        """Add synthetic distractor documents."""
        from draagon_ai.testing.benchmarks.downloaders.distractor_generator import (
            DistractorGenerator,
        )

        generator = DistractorGenerator(llm_provider)
        docs = await generator.generate(
            topics=topics, count=count, similarity_distribution=similarity_distribution
        )

        self.documents.extend(docs)
        logger.info(f"Added {len(docs)} synthetic distractors (total: {len(self.documents)})")

    async def build(self) -> DocumentCorpus:
        """Build final corpus with metadata."""
        from datetime import datetime
        from collections import Counter

        # Calculate metadata
        source_counts = Counter(doc.source for doc in self.documents)
        domain_counts = Counter(doc.domain for doc in self.documents)
        distractor_count = sum(1 for doc in self.documents if doc.is_distractor)

        sizes = [doc.size_bytes for doc in self.documents]
        size_distribution = {
            "min": min(sizes) if sizes else 0,
            "max": max(sizes) if sizes else 0,
            "mean": sum(sizes) // len(sizes) if sizes else 0,
            "median": sorted(sizes)[len(sizes) // 2] if sizes else 0,
        }

        metadata = CorpusMetadata(
            version="v1",
            created_at=datetime.now().isoformat(),
            total_documents=len(self.documents),
            source_distribution=dict(source_counts),
            domain_distribution=dict(domain_counts),
            size_distribution=size_distribution,
            distractor_count=distractor_count,
            distractor_ratio=distractor_count / len(self.documents) if self.documents else 0.0,
        )

        corpus = DocumentCorpus(documents=self.documents, metadata=metadata)

        logger.info(f"Built corpus: {len(self.documents)} documents, {distractor_count} distractors")
        return corpus
```

**Acceptance Criteria:**
- [ ] Orchestrates all 3 document sources (local, online, synthetic)
- [ ] Tracks total document count during build
- [ ] Calculates corpus metadata (source/domain distribution, size stats)
- [ ] Returns `DocumentCorpus` with full metadata
- [ ] Logs progress at each stage

---

## Phase 2: Query Suite Creation (Week 1-2, Days 4-7)

### Task 2.1: Query Data Models

**File:** `src/draagon_ai/testing/benchmarks/queries.py`

**Implementation:**

```python
"""Benchmark query data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryType(str, Enum):
    """Type of query for categorization."""
    STANDARD = "standard"
    MULTI_HOP_BRIDGE = "multi_hop_bridge"
    MULTI_HOP_COMPARISON = "multi_hop_comparison"
    MULTI_HOP_AGGREGATION = "multi_hop_aggregation"
    MULTI_HOP_TEMPORAL = "multi_hop_temporal"
    MULTI_HOP_NEGATION = "multi_hop_negation"
    ZERO_RESULT = "zero_result"
    ADVERSARIAL = "adversarial"


class QueryDifficulty(str, Enum):
    """Query difficulty classification."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


class ZeroResultCategory(str, Enum):
    """Category of zero-result query."""
    OUT_OF_DOMAIN = "out_of_domain"
    TEMPORALLY_INVALID = "temporally_invalid"
    NONSENSICAL = "nonsensical"
    CONTRADICTORY_PREMISE = "contradictory_premise"


class AttackVector(str, Enum):
    """Type of adversarial attack."""
    KEYWORD_STUFFING = "keyword_stuffing"
    SEMANTIC_PARAPHRASING = "semantic_paraphrasing"
    CONTRADICTORY_SOURCES = "contradictory_sources"
    MISLEADING_CONTEXT = "misleading_context"


@dataclass
class HopDescription:
    """Description of a reasoning hop in multi-hop query."""
    step: int
    reasoning: str
    required_document_ids: list[str]
    required_facts: list[str]


@dataclass
class BenchmarkQuery:
    """Base query for benchmark evaluation."""
    query_id: str
    question: str
    query_type: QueryType
    difficulty: QueryDifficulty
    expected_answer_contains: list[str]  # Keywords/phrases expected in answer
    ground_truth_document_ids: list[str]  # Relevant document IDs
    metadata: dict = field(default_factory=dict)


@dataclass
class MultiHopQuery(BenchmarkQuery):
    """Multi-hop reasoning query."""
    hops: list[HopDescription] = field(default_factory=list)
    minimum_documents_required: int = 2
    maximum_documents_sufficient: int = 5


@dataclass
class ZeroResultQuery(BenchmarkQuery):
    """Query that should return no answer."""
    category: ZeroResultCategory = ZeroResultCategory.OUT_OF_DOMAIN
    acceptable_responses: list[str] = field(default_factory=list)
    unacceptable_responses: list[str] = field(default_factory=list)
    max_confidence_threshold: float = 0.3


@dataclass
class DistractorDocument:
    """Distractor document for adversarial queries."""
    doc_id: str
    content: str
    keyword_overlap: float  # 0.0-1.0
    semantic_relevance: float  # 0.0-1.0
    is_hard_negative: bool


@dataclass
class AdversarialQuery(BenchmarkQuery):
    """Adversarial query with distractors."""
    attack_vector: AttackVector = AttackVector.KEYWORD_STUFFING
    distractor_documents: list[DistractorDocument] = field(default_factory=list)
    correct_document_ids: list[str] = field(default_factory=list)
    expected_retrieved_docs: list[str] = field(default_factory=list)
    max_acceptable_false_positives: int = 0
```

**Acceptance Criteria:**
- [ ] All query types modeled (Standard, MultiHop, ZeroResult, Adversarial)
- [ ] Multi-hop queries track hops with reasoning steps
- [ ] Zero-result queries specify acceptable/unacceptable responses
- [ ] Adversarial queries track distractor documents
- [ ] All queries have difficulty classification

---

(Due to length, I'll summarize the remaining phases. Would you like me to continue with the full detail for phases 3-8?)

## Summary of Remaining Phases

**Phase 3: RAGAS Integration (Week 2)** - Implement RAGAS metrics (faithfulness, answer relevancy, context precision/recall), integrate RAGAS library, build LLM-as-judge harness.

**Phase 4: Embedding Quality (Week 2-3)** - Integrate Ollama mxbai-embed-large, add SentenceTransformer fallback, validate embedding quality with semantic similarity tests.

**Phase 5: Statistical Framework (Week 3)** - Implement multiple-run harness, statistical reporting (mean/std/CI), variance analysis, p-value calculations for comparisons.

**Phase 6: Benchmark Infrastructure (Week 3-4)** - Build BenchmarkRunner orchestrator, checkpointing, progress tracking, markdown report generation, CSV export.

**Phase 7: Validation and Tuning (Week 4)** - Run full benchmark suite, compare to industry baselines, identify failure modes, iterate and optimize.

**Phase 8: CI/CD Integration (Week 4)** - Build smoke test suite (50 queries, 5 min), GitHub Actions workflow, nightly benchmarks, regression detection.

---

Would you like me to expand any of these phases with full implementation detail?
