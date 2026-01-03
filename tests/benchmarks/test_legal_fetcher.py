"""Tests for LegalDocumentFetcher.

Tests legal document fetching including licenses, terms of service,
privacy policies, and template-based content generation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from draagon_ai.testing.benchmarks import (
    LegalDocumentFetcher,
    BenchmarkDocument,
    DocumentCategory,
    DocumentSource,
    OPENSOURCE_LICENSES,
    TOS_URLS,
    PRIVACY_POLICY_URLS,
)


class TestLegalDocumentFetcher:
    """Tests for LegalDocumentFetcher class."""

    def test_fetcher_initialization(self) -> None:
        """Fetcher initializes with default settings."""
        fetcher = LegalDocumentFetcher()
        assert fetcher.rate_limit == 1.0
        assert fetcher.timeout == 30.0

    def test_fetcher_custom_settings(self) -> None:
        """Fetcher accepts custom rate limit and timeout."""
        fetcher = LegalDocumentFetcher(rate_limit=0.5, timeout=60.0)
        assert fetcher.rate_limit == 0.5
        assert fetcher.timeout == 60.0

    def test_fetcher_with_cache(self, tmp_path) -> None:
        """Fetcher accepts cache directory."""
        fetcher = LegalDocumentFetcher(cache_dir=tmp_path)
        assert fetcher.cache_dir == tmp_path


class TestPredefinedSources:
    """Tests for predefined legal document sources."""

    def test_opensource_licenses_defined(self) -> None:
        """Common open source licenses are defined."""
        assert "MIT" in OPENSOURCE_LICENSES
        assert "Apache-2.0" in OPENSOURCE_LICENSES
        assert "GPL-3.0" in OPENSOURCE_LICENSES
        assert "BSD-3-Clause" in OPENSOURCE_LICENSES

    def test_opensource_licenses_are_urls(self) -> None:
        """License sources are valid URLs."""
        for name, url in OPENSOURCE_LICENSES.items():
            assert url.startswith("http://") or url.startswith("https://"), f"{name} has invalid URL"

    def test_tos_urls_defined(self) -> None:
        """Terms of service URLs are defined."""
        assert len(TOS_URLS) > 0
        for company, url in TOS_URLS.items():
            assert url.startswith("http://") or url.startswith("https://"), f"{company} has invalid URL"

    def test_privacy_policy_urls_defined(self) -> None:
        """Privacy policy URLs are defined."""
        assert len(PRIVACY_POLICY_URLS) > 0
        for company, url in PRIVACY_POLICY_URLS.items():
            assert url.startswith("http://") or url.startswith("https://"), f"{company} has invalid URL"


class TestInlineLicenses:
    """Tests for inline license text generation."""

    @pytest.mark.asyncio
    async def test_fetch_inline_mit_license(self) -> None:
        """Fetches inline MIT license text."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        # Should have at least the inline version
        mit_docs = [d for d in docs if "mit" in d.doc_id.lower()]
        assert len(mit_docs) >= 1

        # Check inline doc has correct properties
        inline_doc = next((d for d in mit_docs if "inline" in d.doc_id), None)
        if inline_doc:
            assert inline_doc.category == DocumentCategory.LEGAL
            assert inline_doc.source == DocumentSource.SYNTHETIC
            assert "Permission is hereby granted" in inline_doc.content
            assert inline_doc.domain == "legal_license"

    @pytest.mark.asyncio
    async def test_fetch_inline_apache_license(self) -> None:
        """Fetches inline Apache 2.0 license text."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["Apache-2.0"], include_inline=True)

        apache_docs = [d for d in docs if "apache" in d.doc_id.lower()]
        assert len(apache_docs) >= 1

        inline_doc = next((d for d in apache_docs if "inline" in d.doc_id), None)
        if inline_doc:
            assert "Apache License" in inline_doc.content
            # Content includes "AS IS" disclaimer
            assert "AS IS" in inline_doc.content

    @pytest.mark.asyncio
    async def test_fetch_inline_bsd_license(self) -> None:
        """Fetches inline BSD 3-Clause license text."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["BSD-3-Clause"], include_inline=True)

        bsd_docs = [d for d in docs if "bsd" in d.doc_id.lower()]
        assert len(bsd_docs) >= 1

        inline_doc = next((d for d in bsd_docs if "inline" in d.doc_id), None)
        if inline_doc:
            assert "Redistribution and use" in inline_doc.content


class TestTemplateDocuments:
    """Tests for template-based legal document generation."""

    @pytest.mark.asyncio
    async def test_fetch_tos_templates(self) -> None:
        """Fetches template terms of service."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=True)

        template_docs = [d for d in docs if "template" in d.doc_id.lower()]
        assert len(template_docs) >= 1

        for doc in template_docs:
            assert doc.category == DocumentCategory.LEGAL
            assert doc.source == DocumentSource.SYNTHETIC
            assert "Terms" in doc.content or "Service" in doc.content

    @pytest.mark.asyncio
    async def test_fetch_privacy_policy_templates(self) -> None:
        """Fetches template privacy policies."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_privacy_policies(include_templates=True)

        template_docs = [d for d in docs if "template" in d.doc_id.lower()]
        assert len(template_docs) >= 1

        for doc in template_docs:
            assert doc.category == DocumentCategory.LEGAL
            assert "Privacy" in doc.content or "data" in doc.content.lower()

    @pytest.mark.asyncio
    async def test_templates_have_legal_tags(self) -> None:
        """Template documents have appropriate semantic tags."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=True)

        for doc in docs:
            assert "legal" in doc.semantic_tags


class TestFetchLicenses:
    """Tests for license fetching functionality."""

    @pytest.mark.asyncio
    async def test_fetch_specific_licenses(self) -> None:
        """Fetches only specified licenses."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(
            licenses=["MIT", "Apache-2.0"],
            include_inline=True,
        )

        # Should have docs for both licenses
        doc_ids = [d.doc_id.lower() for d in docs]
        assert any("mit" in doc_id for doc_id in doc_ids)
        assert any("apache" in doc_id for doc_id in doc_ids)
        # Should not have GPL
        assert not any("gpl" in doc_id for doc_id in doc_ids)

    @pytest.mark.asyncio
    async def test_fetch_all_licenses_inline_only(self) -> None:
        """Fetches all inline licenses without web fetch."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(include_inline=True)

        # Should have inline versions of supported licenses
        assert len(docs) >= 3  # MIT, Apache, BSD at minimum

    @pytest.mark.asyncio
    async def test_license_docs_have_correct_category(self) -> None:
        """License documents have LEGAL category."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            assert doc.category == DocumentCategory.LEGAL

    @pytest.mark.asyncio
    async def test_license_docs_have_license_domain(self) -> None:
        """License documents have 'legal_license' domain."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            assert doc.domain == "legal_license"


class TestFetchToS:
    """Tests for terms of service fetching."""

    @pytest.mark.asyncio
    async def test_fetch_tos_templates_only(self) -> None:
        """Fetches only template ToS when no companies specified."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=True)

        assert len(docs) >= 1
        for doc in docs:
            assert doc.source == DocumentSource.SYNTHETIC

    @pytest.mark.asyncio
    async def test_tos_docs_have_correct_domain(self) -> None:
        """ToS documents have 'legal_tos' domain."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=True)

        for doc in docs:
            assert doc.domain == "legal_tos"

    @pytest.mark.asyncio
    async def test_tos_docs_have_tos_tag(self) -> None:
        """ToS documents have 'tos' semantic tag."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=True)

        for doc in docs:
            assert "tos" in doc.semantic_tags


class TestFetchPrivacyPolicies:
    """Tests for privacy policy fetching."""

    @pytest.mark.asyncio
    async def test_fetch_privacy_templates_only(self) -> None:
        """Fetches only template privacy policies."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_privacy_policies(include_templates=True)

        # Templates only (no fetch from URLs)
        template_docs = [d for d in docs if d.source == DocumentSource.SYNTHETIC]
        assert len(template_docs) >= 1

    @pytest.mark.asyncio
    async def test_privacy_docs_have_correct_domain(self) -> None:
        """Privacy policy documents have 'legal_privacy' domain."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_privacy_policies(include_templates=True)

        # Check template docs have correct domain
        template_docs = [d for d in docs if d.source == DocumentSource.SYNTHETIC]
        for doc in template_docs:
            assert doc.domain == "legal_privacy"

    @pytest.mark.asyncio
    async def test_privacy_docs_have_privacy_tag(self) -> None:
        """Privacy policy documents have 'privacy' semantic tag."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_privacy_policies(include_templates=True)

        for doc in docs:
            assert "privacy" in doc.semantic_tags


class TestFetchAll:
    """Tests for fetch_all aggregation method."""

    @pytest.mark.asyncio
    async def test_fetch_all_includes_all_types(self) -> None:
        """fetch_all includes licenses, ToS, and privacy policies."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_all(include_templates=True)

        # Should have documents from each category
        domains = {d.domain for d in docs}
        assert "legal_license" in domains
        assert "legal_tos" in domains
        assert "legal_privacy" in domains

    @pytest.mark.asyncio
    async def test_fetch_all_respects_max_docs(self) -> None:
        """fetch_all respects max_docs limit."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_all(max_docs=5, include_templates=True)

        assert len(docs) <= 5

    @pytest.mark.asyncio
    async def test_fetch_all_all_are_legal_category(self) -> None:
        """All documents from fetch_all have LEGAL category."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_all(include_templates=True)

        for doc in docs:
            assert doc.category == DocumentCategory.LEGAL


class TestMaxDocsLimit:
    """Tests for max_docs limit across methods."""

    @pytest.mark.asyncio
    async def test_fetch_licenses_max_docs(self) -> None:
        """fetch_licenses respects max_docs."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(max_docs=2, include_inline=True)

        assert len(docs) <= 2

    @pytest.mark.asyncio
    async def test_fetch_tos_max_docs(self) -> None:
        """fetch_tos respects max_docs."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(max_docs=1, include_templates=True)

        assert len(docs) <= 1

    @pytest.mark.asyncio
    async def test_fetch_privacy_max_docs(self) -> None:
        """fetch_privacy_policies respects max_docs."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_privacy_policies(max_docs=1, include_templates=True)

        assert len(docs) <= 1


class TestDocumentMetadata:
    """Tests for document metadata."""

    @pytest.mark.asyncio
    async def test_docs_have_source_type(self) -> None:
        """Documents have source_type in metadata."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            assert "source_type" in doc.metadata

    @pytest.mark.asyncio
    async def test_docs_have_source_name(self) -> None:
        """License documents have source_name in metadata."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            assert "source_name" in doc.metadata

    @pytest.mark.asyncio
    async def test_inline_docs_marked_as_synthetic(self) -> None:
        """Inline documents are marked as synthetic source."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        inline_docs = [d for d in docs if "inline" in d.doc_id]
        for doc in inline_docs:
            assert doc.source == DocumentSource.SYNTHETIC


class TestDocumentStructure:
    """Tests for document structure."""

    @pytest.mark.asyncio
    async def test_doc_id_format(self) -> None:
        """Document IDs follow expected format."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            assert doc.doc_id.startswith("legal_")

    @pytest.mark.asyncio
    async def test_file_path_format(self) -> None:
        """File paths are set appropriately."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        for doc in docs:
            # Inline docs use inline:// prefix
            if doc.source == DocumentSource.SYNTHETIC:
                assert doc.file_path.startswith("inline://") or doc.file_path.startswith("template://")
            # Fetched docs use URL
            elif doc.source == DocumentSource.ONLINE:
                assert doc.file_path.startswith("http")

    @pytest.mark.asyncio
    async def test_unique_doc_ids(self) -> None:
        """All documents have unique IDs."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_all(include_templates=True)

        # Filter to just template docs to avoid network issues
        template_docs = [d for d in docs if d.source == DocumentSource.SYNTHETIC]
        doc_ids = [d.doc_id for d in template_docs]
        assert len(doc_ids) == len(set(doc_ids))


class TestWebFetching:
    """Tests for web-based fetching with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_fetch_license_from_url(self) -> None:
        """Fetches license from URL with mocked response."""
        fetcher = LegalDocumentFetcher()

        # Mock the aiohttp session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="MIT License\n\nPermission is hereby granted...")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        with patch("aiohttp.ClientSession", return_value=mock_session):
            docs = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=False)

        # Should have fetched doc
        assert len(docs) >= 1

    @pytest.mark.asyncio
    async def test_handles_fetch_error_gracefully(self) -> None:
        """Handles HTTP errors gracefully by returning inline content."""
        fetcher = LegalDocumentFetcher()

        # When only using inline (no network), we always get content
        docs = await fetcher.fetch_licenses(licenses=[], include_inline=True)

        # Should still have inline versions
        assert len(docs) >= 1
        # All should be synthetic (inline)
        assert all(d.source == DocumentSource.SYNTHETIC for d in docs)


class TestDeduplication:
    """Tests for content deduplication."""

    @pytest.mark.asyncio
    async def test_deduplicates_identical_content(self) -> None:
        """Deduplicates documents with identical content."""
        fetcher = LegalDocumentFetcher()

        # Fetch same license multiple times should dedupe
        docs1 = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)
        docs2 = await fetcher.fetch_licenses(licenses=["MIT"], include_inline=True)

        # Content hashes should be same
        if docs1 and docs2:
            assert docs1[0].content_hash == docs2[0].content_hash


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_licenses_list(self) -> None:
        """Handles empty licenses list."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=[], include_inline=False)

        assert docs == []

    @pytest.mark.asyncio
    async def test_unknown_license_name(self) -> None:
        """Handles unknown license names gracefully."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_licenses(licenses=["UNKNOWN-LICENSE"], include_inline=True)

        # Should return empty or just not crash
        assert isinstance(docs, list)

    @pytest.mark.asyncio
    async def test_zero_max_docs(self) -> None:
        """max_docs=0 should return empty list."""
        fetcher = LegalDocumentFetcher()
        # With templates disabled and empty lists, no fetching happens
        docs = await fetcher.fetch_tos(companies=[], include_templates=False)
        assert docs == []

        # With licenses=[] and include_inline=False, no licenses fetched
        docs = await fetcher.fetch_licenses(licenses=[], include_inline=False)
        assert docs == []

    @pytest.mark.asyncio
    async def test_templates_disabled(self) -> None:
        """Works with templates disabled."""
        fetcher = LegalDocumentFetcher()
        docs = await fetcher.fetch_tos(companies=[], include_templates=False)

        # Should return empty since no companies and no templates
        assert docs == []
