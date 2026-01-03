"""Online documentation fetcher for benchmark corpus.

Fetches documentation from online sources like:
- Technical: Python docs, FastAPI, Neo4j, Anthropic
- Academic: arXiv abstracts, Wikipedia science articles
- Narrative: Wikipedia articles, Project Gutenberg excerpts
- Knowledge Base: Stack Overflow, help centers
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin, urlparse

import aiohttp

from ..corpus import BenchmarkDocument, DocumentCategory, DocumentSource

logger = logging.getLogger(__name__)


# Rate limiting: requests per second per domain
DEFAULT_RATE_LIMIT = 1.0
# Request timeout in seconds
DEFAULT_TIMEOUT = 30
# User agent for requests
USER_AGENT = "draagon-ai-benchmark/1.0 (documentation-crawler)"


@dataclass
class FetchResult:
    """Result of fetching a document."""

    url: str
    content: str | None
    title: str | None
    success: bool
    error: str | None = None
    content_type: str | None = None
    size_bytes: int = 0


@dataclass
class OnlineSource:
    """Configuration for an online documentation source."""

    name: str
    base_url: str
    category: DocumentCategory
    domain: str
    patterns: list[str] = field(default_factory=list)
    max_docs: int = 50
    rate_limit: float = DEFAULT_RATE_LIMIT
    content_selector: str | None = None  # CSS selector for main content


# Pre-configured sources for common documentation sites
TECHNICAL_SOURCES = [
    OnlineSource(
        name="python_docs",
        base_url="https://docs.python.org/3.11/library/",
        category=DocumentCategory.TECHNICAL,
        domain="python",
        patterns=["asyncio", "collections", "dataclasses", "functools", "typing"],
    ),
    OnlineSource(
        name="fastapi_docs",
        base_url="https://fastapi.tiangolo.com/",
        category=DocumentCategory.TECHNICAL,
        domain="fastapi",
        patterns=["tutorial/", "advanced/"],
    ),
]

NARRATIVE_SOURCES = [
    OnlineSource(
        name="wikipedia",
        base_url="https://en.wikipedia.org/wiki/",
        category=DocumentCategory.NARRATIVE,
        domain="wikipedia",
        patterns=["History_of", "Science_of", "Art_of"],
    ),
    OnlineSource(
        name="gutenberg",
        base_url="https://www.gutenberg.org/cache/epub/",
        category=DocumentCategory.NARRATIVE,
        domain="literature",
        patterns=[],  # Will use specific book IDs
    ),
]

ACADEMIC_SOURCES = [
    OnlineSource(
        name="arxiv",
        base_url="https://arxiv.org/abs/",
        category=DocumentCategory.ACADEMIC,
        domain="research",
        patterns=["cs.AI", "cs.CL", "cs.LG"],
    ),
]


class OnlineDocumentFetcher:
    """Fetches documents from online documentation sources.

    Supports rate limiting, caching, and content extraction from HTML.

    Example:
        fetcher = OnlineDocumentFetcher(
            cache_dir=Path("/tmp/doc_cache"),
            rate_limit=1.0,  # 1 request per second
        )

        # Fetch from configured sources
        docs = await fetcher.fetch_from_sources(TECHNICAL_SOURCES, max_docs=100)

        # Fetch specific URLs
        docs = await fetcher.fetch_urls(
            urls=["https://docs.python.org/3.11/library/asyncio.html"],
            category=DocumentCategory.TECHNICAL,
            domain="python",
        )
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        rate_limit: float = DEFAULT_RATE_LIMIT,
        timeout: float = DEFAULT_TIMEOUT,
        content_extractor: Callable[[str, str], tuple[str, str | None]] | None = None,
    ) -> None:
        """Initialize fetcher.

        Args:
            cache_dir: Directory to cache fetched documents
            rate_limit: Maximum requests per second per domain
            timeout: Request timeout in seconds
            content_extractor: Custom function to extract content and title from HTML
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rate_limit = rate_limit
        self.timeout = timeout
        self.content_extractor = content_extractor or self._default_content_extractor

        # Rate limiting state
        self._domain_last_request: dict[str, float] = {}
        self._rate_lock = asyncio.Lock()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_from_sources(
        self,
        sources: list[OnlineSource],
        max_docs: int | None = None,
    ) -> list[BenchmarkDocument]:
        """Fetch documents from multiple online sources.

        Args:
            sources: List of OnlineSource configurations
            max_docs: Maximum total documents to fetch

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []
        seen_hashes: set[str] = set()

        for source in sources:
            if max_docs and len(documents) >= max_docs:
                break

            logger.info(f"Fetching from source: {source.name}")
            source_docs = await self._fetch_source(source, seen_hashes)

            for doc in source_docs:
                if max_docs and len(documents) >= max_docs:
                    break
                documents.append(doc)
                seen_hashes.add(doc.content_hash)

        logger.info(f"Fetched {len(documents)} documents from {len(sources)} sources")
        return documents

    async def fetch_urls(
        self,
        urls: list[str],
        category: DocumentCategory,
        domain: str,
        max_docs: int | None = None,
    ) -> list[BenchmarkDocument]:
        """Fetch documents from specific URLs.

        Args:
            urls: List of URLs to fetch
            category: Document category for all URLs
            domain: Domain for all documents
            max_docs: Maximum documents to fetch

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []
        seen_hashes: set[str] = set()

        for url in urls:
            if max_docs and len(documents) >= max_docs:
                break

            doc = await self._fetch_and_create_document(
                url=url,
                category=category,
                domain=domain,
                seen_hashes=seen_hashes,
            )

            if doc:
                documents.append(doc)
                seen_hashes.add(doc.content_hash)

        return documents

    async def _fetch_source(
        self,
        source: OnlineSource,
        seen_hashes: set[str],
    ) -> list[BenchmarkDocument]:
        """Fetch documents from a single source.

        Args:
            source: Source configuration
            seen_hashes: Set of content hashes already seen

        Returns:
            List of documents from this source
        """
        documents: list[BenchmarkDocument] = []

        # Build URLs from patterns
        urls = self._build_urls(source)

        for url in urls:
            if len(documents) >= source.max_docs:
                break

            doc = await self._fetch_and_create_document(
                url=url,
                category=source.category,
                domain=source.domain,
                seen_hashes=seen_hashes,
            )

            if doc:
                documents.append(doc)
                seen_hashes.add(doc.content_hash)

        return documents

    def _build_urls(self, source: OnlineSource) -> list[str]:
        """Build URLs from source patterns.

        Args:
            source: Source configuration

        Returns:
            List of URLs to fetch
        """
        if not source.patterns:
            return [source.base_url]

        urls = []
        for pattern in source.patterns:
            url = urljoin(source.base_url, pattern)
            urls.append(url)

        return urls

    async def _fetch_and_create_document(
        self,
        url: str,
        category: DocumentCategory,
        domain: str,
        seen_hashes: set[str],
    ) -> BenchmarkDocument | None:
        """Fetch URL and create document.

        Args:
            url: URL to fetch
            category: Document category
            domain: Document domain
            seen_hashes: Already seen content hashes

        Returns:
            BenchmarkDocument or None if fetch failed or duplicate
        """
        # Check cache first
        cached = self._get_cached(url)
        if cached:
            logger.debug(f"Cache hit: {url}")
            content, title = cached
        else:
            # Fetch from network
            result = await self._fetch_url(url)
            if not result.success or not result.content:
                logger.warning(f"Failed to fetch {url}: {result.error}")
                return None

            # Extract content
            content, title = self.content_extractor(result.content, url)

            # Cache result
            self._cache_result(url, content, title)

        # Check for duplicate content
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash in seen_hashes:
            logger.debug(f"Duplicate content: {url}")
            return None

        # Create document
        doc_id = self._generate_doc_id(url)
        tags = self._extract_tags(url, content)

        return BenchmarkDocument(
            doc_id=doc_id,
            source=DocumentSource.ONLINE,
            category=category,
            domain=domain,
            file_path=url,
            content=content,
            semantic_tags=tags,
            metadata={
                "url": url,
                "title": title or doc_id,
            },
        )

    async def _fetch_url(self, url: str) -> FetchResult:
        """Fetch a URL with rate limiting.

        Args:
            url: URL to fetch

        Returns:
            FetchResult with content or error
        """
        # Rate limiting
        await self._rate_limit(url)

        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {"User-Agent": USER_AGENT}

            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status != 200:
                        return FetchResult(
                            url=url,
                            content=None,
                            title=None,
                            success=False,
                            error=f"HTTP {response.status}",
                        )

                    content_type = response.headers.get("Content-Type", "")
                    content = await response.text()

                    return FetchResult(
                        url=url,
                        content=content,
                        title=None,  # Will be extracted from content
                        success=True,
                        content_type=content_type,
                        size_bytes=len(content.encode()),
                    )

        except asyncio.TimeoutError:
            return FetchResult(
                url=url,
                content=None,
                title=None,
                success=False,
                error="Timeout",
            )
        except aiohttp.ClientError as e:
            return FetchResult(
                url=url,
                content=None,
                title=None,
                success=False,
                error=str(e),
            )

    async def _rate_limit(self, url: str) -> None:
        """Apply rate limiting for domain.

        Args:
            url: URL being fetched
        """
        domain = urlparse(url).netloc

        async with self._rate_lock:
            now = asyncio.get_event_loop().time()
            last_request = self._domain_last_request.get(domain, 0)
            min_interval = 1.0 / self.rate_limit

            if now - last_request < min_interval:
                wait_time = min_interval - (now - last_request)
                await asyncio.sleep(wait_time)

            self._domain_last_request[domain] = asyncio.get_event_loop().time()

    def _default_content_extractor(
        self, html: str, url: str
    ) -> tuple[str, str | None]:
        """Extract main content and title from HTML.

        Simple extraction using regex - for production, use BeautifulSoup.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Tuple of (content, title)
        """
        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else None

        # Remove script and style tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Decode HTML entities
        content = re.sub(r"&nbsp;", " ", content)
        content = re.sub(r"&amp;", "&", content)
        content = re.sub(r"&lt;", "<", content)
        content = re.sub(r"&gt;", ">", content)
        content = re.sub(r"&quot;", '"', content)

        # Normalize whitespace
        content = re.sub(r"\s+", " ", content).strip()

        return content, title

    def _generate_doc_id(self, url: str) -> str:
        """Generate unique document ID from URL.

        Args:
            url: Source URL

        Returns:
            Document ID string
        """
        parsed = urlparse(url)
        path = parsed.path.strip("/").replace("/", "_")
        domain = parsed.netloc.replace(".", "_")

        # Combine and sanitize
        doc_id = f"{domain}_{path}" if path else domain
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)

        # Truncate if too long
        if len(doc_id) > 100:
            hash_suffix = hashlib.sha256(url.encode()).hexdigest()[:8]
            doc_id = doc_id[:91] + "_" + hash_suffix

        return f"online_{doc_id}"

    def _extract_tags(self, url: str, content: str) -> list[str]:
        """Extract semantic tags from URL and content.

        Args:
            url: Source URL
            content: Document content

        Returns:
            List of tag strings
        """
        tags: set[str] = set()

        # Add domain as tag
        domain = urlparse(url).netloc
        if "python" in domain:
            tags.add("python")
        elif "fastapi" in domain:
            tags.add("fastapi")
        elif "wikipedia" in domain:
            tags.add("wikipedia")
        elif "arxiv" in domain:
            tags.add("arxiv")
        elif "gutenberg" in domain:
            tags.add("literature")

        # Check content for common keywords (first 500 chars)
        content_lower = content[:500].lower()
        keywords = [
            "async", "api", "database", "function", "class",
            "python", "javascript", "typescript",
            "tutorial", "guide", "documentation",
            "research", "paper", "study",
        ]
        for keyword in keywords:
            if keyword in content_lower:
                tags.add(keyword)

        return sorted(tags)

    def _get_cached(self, url: str) -> tuple[str, str | None] | None:
        """Get cached document if available.

        Args:
            url: Source URL

        Returns:
            Tuple of (content, title) or None if not cached
        """
        if not self.cache_dir:
            return None

        cache_key = hashlib.sha256(url.encode()).hexdigest()
        content_path = self.cache_dir / f"{cache_key}.txt"
        title_path = self.cache_dir / f"{cache_key}.title"

        if not content_path.exists():
            return None

        try:
            content = content_path.read_text(encoding="utf-8")
            title = title_path.read_text(encoding="utf-8") if title_path.exists() else None
            return content, title
        except OSError:
            return None

    def _cache_result(self, url: str, content: str, title: str | None) -> None:
        """Cache fetched document.

        Args:
            url: Source URL
            content: Document content
            title: Document title
        """
        if not self.cache_dir:
            return

        cache_key = hashlib.sha256(url.encode()).hexdigest()
        content_path = self.cache_dir / f"{cache_key}.txt"
        title_path = self.cache_dir / f"{cache_key}.title"

        try:
            content_path.write_text(content, encoding="utf-8")
            if title:
                title_path.write_text(title, encoding="utf-8")
        except OSError as e:
            logger.warning(f"Failed to cache {url}: {e}")
