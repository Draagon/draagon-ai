"""Tests for OnlineDocumentFetcher.

Tests online documentation fetching including rate limiting,
caching, content extraction, and deduplication.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from draagon_ai.testing.benchmarks import (
    OnlineDocumentFetcher,
    OnlineSource,
    FetchResult,
    DocumentCategory,
    DocumentSource,
    TECHNICAL_SOURCES,
    NARRATIVE_SOURCES,
    ACADEMIC_SOURCES,
)


class TestOnlineDocumentFetcher:
    """Tests for OnlineDocumentFetcher class."""

    def test_fetcher_initialization(self, tmp_path: Path) -> None:
        """Fetcher initializes with cache directory."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        assert fetcher.cache_dir == tmp_path
        assert fetcher.rate_limit == 1.0
        assert fetcher.timeout == 30

    def test_fetcher_creates_cache_dir(self, tmp_path: Path) -> None:
        """Fetcher creates cache directory if it doesn't exist."""
        cache_dir = tmp_path / "new_cache"
        fetcher = OnlineDocumentFetcher(cache_dir=cache_dir)
        assert cache_dir.exists()

    def test_fetcher_no_cache(self) -> None:
        """Fetcher works without cache directory."""
        fetcher = OnlineDocumentFetcher(cache_dir=None)
        assert fetcher.cache_dir is None


class TestOnlineSource:
    """Tests for OnlineSource configuration."""

    def test_source_configuration(self) -> None:
        """OnlineSource holds configuration correctly."""
        source = OnlineSource(
            name="test_source",
            base_url="https://example.com/docs/",
            category=DocumentCategory.TECHNICAL,
            domain="testing",
            patterns=["api/", "guide/"],
            max_docs=25,
            rate_limit=0.5,
        )

        assert source.name == "test_source"
        assert source.base_url == "https://example.com/docs/"
        assert source.category == DocumentCategory.TECHNICAL
        assert source.domain == "testing"
        assert source.patterns == ["api/", "guide/"]
        assert source.max_docs == 25
        assert source.rate_limit == 0.5

    def test_predefined_sources_exist(self) -> None:
        """Predefined source lists are available."""
        assert len(TECHNICAL_SOURCES) > 0
        assert len(NARRATIVE_SOURCES) > 0
        assert len(ACADEMIC_SOURCES) > 0

    def test_technical_sources_valid(self) -> None:
        """Technical sources have correct category."""
        for source in TECHNICAL_SOURCES:
            assert source.category == DocumentCategory.TECHNICAL

    def test_narrative_sources_valid(self) -> None:
        """Narrative sources have correct category."""
        for source in NARRATIVE_SOURCES:
            assert source.category == DocumentCategory.NARRATIVE

    def test_academic_sources_valid(self) -> None:
        """Academic sources have correct category."""
        for source in ACADEMIC_SOURCES:
            assert source.category == DocumentCategory.ACADEMIC


class TestFetchResult:
    """Tests for FetchResult dataclass."""

    def test_successful_result(self) -> None:
        """FetchResult represents success correctly."""
        result = FetchResult(
            url="https://example.com/doc",
            content="Document content here",
            title="Example Document",
            success=True,
            content_type="text/html",
            size_bytes=100,
        )

        assert result.success is True
        assert result.content == "Document content here"
        assert result.title == "Example Document"
        assert result.error is None

    def test_failed_result(self) -> None:
        """FetchResult represents failure correctly."""
        result = FetchResult(
            url="https://example.com/missing",
            content=None,
            title=None,
            success=False,
            error="HTTP 404",
        )

        assert result.success is False
        assert result.content is None
        assert result.error == "HTTP 404"


class TestContentExtraction:
    """Tests for HTML content extraction."""

    def test_extract_title(self, tmp_path: Path) -> None:
        """Extracts title from HTML."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        html = "<html><head><title>Test Title</title></head><body>Content</body></html>"

        content, title = fetcher._default_content_extractor(html, "https://example.com")

        assert title == "Test Title"

    def test_extract_content_removes_tags(self, tmp_path: Path) -> None:
        """Removes HTML tags from content."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        html = "<html><body><p>Paragraph one.</p><p>Paragraph two.</p></body></html>"

        content, _ = fetcher._default_content_extractor(html, "https://example.com")

        assert "<p>" not in content
        assert "Paragraph one" in content
        assert "Paragraph two" in content

    def test_extract_content_removes_scripts(self, tmp_path: Path) -> None:
        """Removes script tags from content."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        html = "<html><body><script>alert('evil');</script><p>Real content.</p></body></html>"

        content, _ = fetcher._default_content_extractor(html, "https://example.com")

        assert "alert" not in content
        assert "Real content" in content

    def test_extract_content_removes_styles(self, tmp_path: Path) -> None:
        """Removes style tags from content."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        html = "<html><head><style>body { color: red; }</style></head><body>Content</body></html>"

        content, _ = fetcher._default_content_extractor(html, "https://example.com")

        assert "color: red" not in content
        assert "Content" in content

    def test_decode_html_entities(self, tmp_path: Path) -> None:
        """Decodes common HTML entities."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        html = "<html><body>A &amp; B &lt;compare&gt; C</body></html>"

        content, _ = fetcher._default_content_extractor(html, "https://example.com")

        assert "A & B" in content
        assert "<compare>" in content


class TestDocumentIdGeneration:
    """Tests for document ID generation."""

    def test_generates_unique_ids(self, tmp_path: Path) -> None:
        """Generates unique IDs from different URLs."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        id1 = fetcher._generate_doc_id("https://example.com/doc1")
        id2 = fetcher._generate_doc_id("https://example.com/doc2")
        id3 = fetcher._generate_doc_id("https://other.com/doc1")

        assert id1 != id2
        assert id1 != id3
        assert id2 != id3

    def test_id_starts_with_online(self, tmp_path: Path) -> None:
        """Document IDs start with 'online_' prefix."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        doc_id = fetcher._generate_doc_id("https://example.com/docs/api")

        assert doc_id.startswith("online_")

    def test_id_sanitized(self, tmp_path: Path) -> None:
        """Document IDs are sanitized for special characters."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        doc_id = fetcher._generate_doc_id("https://example.com/docs/api?v=1&foo=bar")

        # Should not contain problematic characters
        assert "?" not in doc_id
        assert "&" not in doc_id
        assert "=" not in doc_id

    def test_long_url_truncated(self, tmp_path: Path) -> None:
        """Very long URLs are truncated with hash suffix."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        long_path = "/".join(["segment"] * 50)
        doc_id = fetcher._generate_doc_id(f"https://example.com/{long_path}")

        assert len(doc_id) <= 110  # online_ prefix + max 100 chars


class TestTagExtraction:
    """Tests for semantic tag extraction."""

    def test_python_domain_tag(self, tmp_path: Path) -> None:
        """Adds python tag for Python docs URLs."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        tags = fetcher._extract_tags("https://docs.python.org/3.11/library/asyncio.html", "async def foo(): pass")

        assert "python" in tags

    def test_content_keyword_tags(self, tmp_path: Path) -> None:
        """Detects keywords in content."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        tags = fetcher._extract_tags(
            "https://example.com/doc",
            "This tutorial explains async functions and API design"
        )

        assert "async" in tags
        assert "api" in tags
        assert "tutorial" in tags

    def test_wikipedia_domain_tag(self, tmp_path: Path) -> None:
        """Adds wikipedia tag for Wikipedia URLs."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        tags = fetcher._extract_tags("https://en.wikipedia.org/wiki/Python", "Content about Python")

        assert "wikipedia" in tags


class TestCaching:
    """Tests for document caching."""

    def test_cache_stores_content(self, tmp_path: Path) -> None:
        """Cache stores content and title."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        fetcher._cache_result(
            url="https://example.com/doc",
            content="Cached content here",
            title="Cached Title"
        )

        # Verify cache files exist
        cached = fetcher._get_cached("https://example.com/doc")
        assert cached is not None
        content, title = cached
        assert content == "Cached content here"
        assert title == "Cached Title"

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """Cache miss returns None."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        cached = fetcher._get_cached("https://example.com/nonexistent")

        assert cached is None

    def test_cache_disabled_when_no_dir(self) -> None:
        """No caching when cache_dir is None."""
        fetcher = OnlineDocumentFetcher(cache_dir=None)

        # Should not raise
        fetcher._cache_result("https://example.com/doc", "content", "title")
        cached = fetcher._get_cached("https://example.com/doc")

        assert cached is None


class TestUrlBuilding:
    """Tests for URL building from patterns."""

    def test_builds_urls_from_patterns(self, tmp_path: Path) -> None:
        """Builds URLs by combining base URL with patterns."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        source = OnlineSource(
            name="test",
            base_url="https://docs.example.com/",
            category=DocumentCategory.TECHNICAL,
            domain="test",
            patterns=["api/v1", "guide/intro"],
        )

        urls = fetcher._build_urls(source)

        assert "https://docs.example.com/api/v1" in urls
        assert "https://docs.example.com/guide/intro" in urls

    def test_returns_base_url_when_no_patterns(self, tmp_path: Path) -> None:
        """Returns base URL when no patterns specified."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)
        source = OnlineSource(
            name="test",
            base_url="https://docs.example.com/index.html",
            category=DocumentCategory.TECHNICAL,
            domain="test",
            patterns=[],
        )

        urls = fetcher._build_urls(source)

        assert urls == ["https://docs.example.com/index.html"]


class TestFetchUrls:
    """Tests for fetching specific URLs."""

    @pytest.mark.asyncio
    async def test_fetch_urls_creates_documents(self, tmp_path: Path) -> None:
        """fetch_urls creates BenchmarkDocument instances."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        # Mock the network fetch - return unique content per URL
        async def mock_fetch(url):
            return FetchResult(
                url=url,
                content=f"<html><head><title>Test</title></head><body>Content for {url}</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_urls(
            urls=["https://example.com/doc1", "https://example.com/doc2"],
            category=DocumentCategory.TECHNICAL,
            domain="test",
        )

        assert len(docs) == 2
        assert all(doc.source == DocumentSource.ONLINE for doc in docs)
        assert all(doc.category == DocumentCategory.TECHNICAL for doc in docs)

    @pytest.mark.asyncio
    async def test_fetch_urls_respects_max_docs(self, tmp_path: Path) -> None:
        """fetch_urls stops at max_docs limit."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        async def mock_fetch(url):
            return FetchResult(
                url=url,
                content=f"<html><body>{url} content</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_urls(
            urls=[f"https://example.com/doc{i}" for i in range(10)],
            category=DocumentCategory.TECHNICAL,
            domain="test",
            max_docs=3,
        )

        assert len(docs) == 3

    @pytest.mark.asyncio
    async def test_fetch_urls_deduplicates(self, tmp_path: Path) -> None:
        """fetch_urls removes duplicate content."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        async def mock_fetch(url):
            # Return identical content for all URLs
            return FetchResult(
                url=url,
                content="<html><body>Same content everywhere</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_urls(
            urls=["https://example.com/doc1", "https://example.com/doc2", "https://example.com/doc3"],
            category=DocumentCategory.TECHNICAL,
            domain="test",
        )

        # Should only have one document due to deduplication
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_fetch_urls_handles_failures(self, tmp_path: Path) -> None:
        """fetch_urls continues after individual failures."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        call_count = 0

        async def mock_fetch(url):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                return FetchResult(url=url, content=None, title=None, success=False, error="HTTP 404")
            return FetchResult(
                url=url,
                content=f"<html><body>Content {call_count}</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_urls(
            urls=["https://example.com/doc1", "https://example.com/doc2", "https://example.com/doc3"],
            category=DocumentCategory.TECHNICAL,
            domain="test",
        )

        # Should have 2 documents (1 failed)
        assert len(docs) == 2


class TestFetchFromSources:
    """Tests for fetching from configured sources."""

    @pytest.mark.asyncio
    async def test_fetch_from_sources(self, tmp_path: Path) -> None:
        """fetch_from_sources processes multiple sources."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        source1 = OnlineSource(
            name="source1",
            base_url="https://example1.com/",
            category=DocumentCategory.TECHNICAL,
            domain="tech1",
            patterns=["doc"],
            max_docs=2,
        )
        source2 = OnlineSource(
            name="source2",
            base_url="https://example2.com/",
            category=DocumentCategory.NARRATIVE,
            domain="narrative1",
            patterns=["story"],
            max_docs=2,
        )

        async def mock_fetch(url):
            return FetchResult(
                url=url,
                content=f"<html><body>Content from {url}</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_from_sources([source1, source2])

        assert len(docs) == 2
        # One from each source
        categories = {doc.category for doc in docs}
        assert DocumentCategory.TECHNICAL in categories
        assert DocumentCategory.NARRATIVE in categories

    @pytest.mark.asyncio
    async def test_fetch_from_sources_respects_max_docs(self, tmp_path: Path) -> None:
        """fetch_from_sources stops at total max_docs."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path)

        sources = [
            OnlineSource(
                name=f"source{i}",
                base_url=f"https://example{i}.com/",
                category=DocumentCategory.TECHNICAL,
                domain="tech",
                patterns=["doc1", "doc2", "doc3"],
                max_docs=10,
            )
            for i in range(5)
        ]

        call_count = 0

        async def mock_fetch(url):
            nonlocal call_count
            call_count += 1
            return FetchResult(
                url=url,
                content=f"<html><body>Unique content {call_count}</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_from_sources(sources, max_docs=5)

        assert len(docs) == 5


class TestRateLimiting:
    """Tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_delays_requests(self, tmp_path: Path) -> None:
        """Rate limiter delays requests to same domain."""
        # Use high rate limit to make test fast but still verify delay happens
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path, rate_limit=100.0)  # 100 req/s = 10ms delay

        start = asyncio.get_event_loop().time()

        # Make two requests to same domain
        await fetcher._rate_limit("https://example.com/doc1")
        await fetcher._rate_limit("https://example.com/doc2")

        elapsed = asyncio.get_event_loop().time() - start

        # Should have some delay (at least ~10ms)
        assert elapsed >= 0.005  # Allow some margin

    @pytest.mark.asyncio
    async def test_different_domains_not_limited(self, tmp_path: Path) -> None:
        """Different domains are not rate limited together."""
        fetcher = OnlineDocumentFetcher(cache_dir=tmp_path, rate_limit=1.0)  # 1 req/s

        start = asyncio.get_event_loop().time()

        # Make requests to different domains
        await fetcher._rate_limit("https://example1.com/doc")
        await fetcher._rate_limit("https://example2.com/doc")
        await fetcher._rate_limit("https://example3.com/doc")

        elapsed = asyncio.get_event_loop().time() - start

        # Should be nearly instant since different domains
        assert elapsed < 0.1


class TestCustomContentExtractor:
    """Tests for custom content extraction."""

    @pytest.mark.asyncio
    async def test_custom_extractor_used(self, tmp_path: Path) -> None:
        """Custom content extractor is called."""
        call_count = 0

        def custom_extractor(html: str, url: str) -> tuple[str, str | None]:
            nonlocal call_count
            call_count += 1
            return f"Custom: {html[:20]}", "Custom Title"

        fetcher = OnlineDocumentFetcher(
            cache_dir=tmp_path,
            content_extractor=custom_extractor,
        )

        async def mock_fetch(url):
            return FetchResult(
                url=url,
                content="<html><body>Original content</body></html>",
                title=None,
                success=True,
            )

        fetcher._fetch_url = mock_fetch

        docs = await fetcher.fetch_urls(
            urls=["https://example.com/doc"],
            category=DocumentCategory.TECHNICAL,
            domain="test",
        )

        assert call_count == 1
        assert docs[0].content.startswith("Custom:")
        assert docs[0].metadata["title"] == "Custom Title"
