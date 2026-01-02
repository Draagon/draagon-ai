# TASK-074: Legal Document Fetcher

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Legal docs are key differentiator for benchmark quality)
**Effort**: 1.5 days
**Status**: Pending
**Dependencies**: TASK-070 (Document data models), TASK-072 (Online fetcher base)

---

## Description

Implement specialized fetcher for legal documents from public sources:
- Terms of Service from major tech companies
- Open-source licenses (full text)
- Court opinions from CourtListener
- SEC filings from EDGAR
- EU regulations from EUR-Lex

**Location:** `src/draagon_ai/testing/benchmarks/downloaders/legal_fetcher.py`

---

## Acceptance Criteria

### Terms of Service (ToS)
- [ ] Fetch ToS from: Apple, Google, GitHub, Microsoft, Amazon, Meta
- [ ] Extract main content (skip navigation, footers)
- [ ] Handle JavaScript-rendered pages (if needed)
- [ ] Tag with domain: `tos`, `privacy_policy`

### Open-Source Licenses
- [ ] Fetch full text: MIT, Apache-2.0, GPL-3.0, BSD-3-Clause, MPL-2.0
- [ ] Source from SPDX or OSI official repositories
- [ ] Clean formatting (no HTML artifacts)
- [ ] Tag with domain: `license`

### Court Opinions (CourtListener)
- [ ] Use CourtListener REST API (free, no auth required for public data)
- [ ] Fetch US federal court opinions
- [ ] Filter by: recent, relevance, jurisdiction
- [ ] Extract opinion text, case metadata
- [ ] Tag with domain: `case_law`, `federal_court`

### SEC Filings (EDGAR)
- [ ] Fetch 10-K filings (annual reports) - risk factors section
- [ ] Use SEC EDGAR API or direct HTML scraping
- [ ] Extract text from HTML filings
- [ ] Tag with domain: `sec_filing`, `10k`

### EU Regulations (EUR-Lex)
- [ ] Fetch GDPR full text (Regulation 2016/679)
- [ ] Fetch AI Act (when available)
- [ ] Handle multi-language (English version)
- [ ] Tag with domain: `eu_regulation`, `gdpr`

---

## Technical Notes

### CourtListener API

```python
import aiohttp

COURTLISTENER_BASE = "https://www.courtlistener.com/api/rest/v3"

async def fetch_court_opinions(self, max_docs: int = 15) -> list[BenchmarkDocument]:
    """Fetch recent federal court opinions."""
    async with aiohttp.ClientSession() as session:
        # Search for opinions
        async with session.get(
            f"{COURTLISTENER_BASE}/opinions/",
            params={
                "court": "scotus",  # Supreme Court, or "ca1" for 1st Circuit, etc.
                "order_by": "-date_filed",
                "page_size": max_docs,
            }
        ) as resp:
            data = await resp.json()

        documents = []
        for opinion in data["results"]:
            # Fetch full opinion text
            async with session.get(opinion["absolute_url"]) as resp:
                html = await resp.text()
                content = self._extract_opinion_text(html)

            doc = BenchmarkDocument(
                doc_id=f"courtlistener_{opinion['id']}",
                source=DocumentSource.ONLINE,
                category=DocumentCategory.LEGAL,
                domain="case_law",
                file_path=opinion["absolute_url"],
                content=content,
                metadata={
                    "court": opinion["court"],
                    "date_filed": opinion["date_filed"],
                    "case_name": opinion["case_name"],
                },
            )
            documents.append(doc)

        return documents
```

### SEC EDGAR

```python
SEC_EDGAR_BASE = "https://www.sec.gov/cgi-bin/browse-edgar"

async def fetch_10k_filings(self, companies: list[str], max_docs: int = 10):
    """Fetch 10-K risk factors from EDGAR."""
    # Use SEC EDGAR full-text search or company CIK lookup
    # Extract "Risk Factors" section (Item 1A)
    pass
```

### ToS Fetching Strategy

For JavaScript-heavy sites:
1. Try direct HTML fetch first
2. If content is minimal, use pre-cached versions
3. Fallback: Use archived versions from web.archive.org

```python
TOS_URLS = {
    "apple": "https://www.apple.com/legal/internet-services/terms/site.html",
    "google": "https://policies.google.com/terms",
    "github": "https://docs.github.com/en/site-policy/github-terms/github-terms-of-service",
    "microsoft": "https://www.microsoft.com/en-us/servicesagreement",
}
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_license_fetch():
    """Fetch open-source license text."""
    fetcher = LegalDocumentFetcher(cache_dir=tmp_path)
    docs = await fetcher.fetch_licenses(["MIT", "Apache-2.0"])

    assert len(docs) == 2
    assert all(doc.category == DocumentCategory.LEGAL for doc in docs)
    assert "permission" in docs[0].content.lower()  # Common license term

@pytest.mark.asyncio
async def test_tos_extraction():
    """ToS extraction gets main content."""
    content = fetcher._extract_tos_content(sample_html)
    assert len(content) > 1000  # Substantial content
    assert "navigation" not in content.lower()  # No nav elements
```

### Integration Tests
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_courtlistener_api():
    """Fetch real court opinions."""
    fetcher = LegalDocumentFetcher(cache_dir=tmp_path)
    docs = await fetcher.fetch_court_opinions(max_docs=2)

    assert len(docs) == 2
    assert all("court" in doc.metadata for doc in docs)
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/downloaders/legal_fetcher.py`
- Add tests to `tests/benchmarks/test_corpus_builder.py`

---

## Why Legal Docs Matter

Legal documents are the **hardest test** for retrieval because:
1. **Precise language**: "shall" vs "may" changes meaning entirely
2. **Cross-references**: "Subject to Section 4.2(b)" requires multi-hop
3. **Negation**: "not liable except..." requires understanding exceptions
4. **Long sentences**: 100+ word sentences test chunking
5. **Domain terminology**: "indemnification", "severability", "force majeure"

If the benchmark passes with legal docs, it proves **real semantic understanding**.

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] 50 legal documents fetchable
- [ ] ToS from 5+ major tech companies
- [ ] Court opinions from CourtListener API
- [ ] SEC 10-K excerpts from EDGAR
- [ ] EU GDPR full text
- [ ] Open-source licenses (5+ types)
- [ ] Integration tests passing
- [ ] Caching prevents re-fetching
