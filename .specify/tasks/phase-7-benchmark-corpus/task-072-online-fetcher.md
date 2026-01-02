# TASK-072: Online Documentation Fetcher

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Needed for Technical, Academic, Knowledge Base categories)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-070 (Document data models)

---

## Description

Implement web scraper to download documentation from online sources:
- Crawl documentation sites with configurable depth
- Convert HTML to clean markdown/text
- Cache downloaded content locally
- Support multiple source types (docs sites, Wikipedia, arXiv)

**Location:** `src/draagon_ai/testing/benchmarks/downloaders/online_fetcher.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `OnlineDocumentationFetcher` class with cache_dir, max_depth, timeout
- [ ] `fetch_documentation(category, base_url, max_docs)` returns list of BenchmarkDocument
- [ ] HTML to text conversion (strip scripts, styles, nav, footer)
- [ ] Local caching with URL-based hash keys
- [ ] Configurable crawl depth (default: 2)

### Source-Specific Handlers
- [ ] Generic HTML scraper for documentation sites
- [ ] Wikipedia article fetcher (via API or HTML)
- [ ] arXiv abstract fetcher (cs.AI, cs.CL, cs.LG categories)
- [ ] Stack Overflow answer fetcher (top answers by tag)

### Caching
- [ ] Cache documents to avoid re-fetching
- [ ] URL hash as cache key (SHA256, 16 chars)
- [ ] Load from cache if exists, skip network
- [ ] Cache invalidation by age (optional)

### Error Handling
- [ ] HTTP errors logged and skipped (don't crash)
- [ ] Timeout handling (30s default)
- [ ] Rate limiting respect (1 req/sec default)
- [ ] Invalid HTML handled gracefully

---

## Technical Notes

### HTML to Text Conversion
```python
from bs4 import BeautifulSoup

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content elements
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n\n".join(lines)
```

### Caching Strategy
```python
def _get_cache_path(self, url: str) -> Path:
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    return self.cache_dir / f"{url_hash}.json"
```

### Rate Limiting
```python
import asyncio

async def _fetch_with_rate_limit(self, session, url):
    await asyncio.sleep(self.rate_limit_seconds)  # Default 1.0
    async with session.get(url) as response:
        return await response.text()
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_html_to_text():
    """HTML conversion strips scripts, keeps content."""
    html = "<html><script>bad</script><p>Good content</p></html>"
    text = html_to_text(html)
    assert "bad" not in text
    assert "Good content" in text

@pytest.mark.asyncio
async def test_caching(tmp_path):
    """Fetcher uses cache on second request."""
    fetcher = OnlineDocumentationFetcher(cache_dir=tmp_path)
    # First fetch (network)
    docs1 = await fetcher.fetch_documentation(...)
    # Second fetch (cache)
    docs2 = await fetcher.fetch_documentation(...)
    assert docs1[0].content == docs2[0].content
```

### Integration Tests (requires network)
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_fetch_real_documentation():
    """Fetch real Python docs page."""
    fetcher = OnlineDocumentationFetcher(...)
    docs = await fetcher.fetch_documentation(
        category=DocumentCategory.TECHNICAL,
        base_url="https://docs.python.org/3.11/library/asyncio.html",
        max_docs=1,
    )
    assert len(docs) == 1
    assert "asyncio" in docs[0].content.lower()
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/downloaders/online_fetcher.py`
- Add tests to `tests/benchmarks/test_corpus_builder.py`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing
- [ ] Integration test validates real URL fetch
- [ ] Caching verified (second fetch uses cache)
- [ ] Rate limiting prevents hammering servers
- [ ] Error handling covers common HTTP failures
