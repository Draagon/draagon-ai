"""Legal document fetcher for benchmark corpus.

Fetches legal documents from various sources including:
- Terms of Service from major tech companies
- Privacy Policies
- Open-source licenses (MIT, Apache, GPL, etc.)
- Court opinions from CourtListener
- SEC filings (EDGAR)
- EU regulations (EUR-Lex)
- Contract templates

Legal documents are particularly challenging for retrieval because of:
- Dense terminology (indemnification, force majeure, severability)
- Long sentences (100+ words)
- Cross-references (Subject to Section 4.2(b)(iii)...)
- Negation complexity (shall not be liable except where...)
- Precise definitions
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..corpus import BenchmarkDocument, DocumentCategory, DocumentSource
from .online_fetcher import OnlineDocumentFetcher, OnlineSource, FetchResult

logger = logging.getLogger(__name__)


# Pre-defined legal document sources
OPENSOURCE_LICENSES = {
    "MIT": "https://opensource.org/license/mit/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0.txt",
    "GPL-3.0": "https://www.gnu.org/licenses/gpl-3.0.txt",
    "BSD-3-Clause": "https://opensource.org/license/bsd-3-clause/",
    "MPL-2.0": "https://www.mozilla.org/en-US/MPL/2.0/",
    "LGPL-3.0": "https://www.gnu.org/licenses/lgpl-3.0.txt",
    "AGPL-3.0": "https://www.gnu.org/licenses/agpl-3.0.txt",
    "ISC": "https://opensource.org/license/isc-license-txt/",
    "Unlicense": "https://unlicense.org/",
}

# Terms of Service URLs for major tech companies
TOS_URLS = {
    "github": "https://docs.github.com/en/site-policy/github-terms/github-terms-of-service",
    "gitlab": "https://about.gitlab.com/terms/",
    "cloudflare": "https://www.cloudflare.com/terms/",
    "digitalocean": "https://www.digitalocean.com/legal/terms-of-service-agreement",
    "heroku": "https://www.salesforce.com/company/legal/agreements/",
    "vercel": "https://vercel.com/legal/terms",
    "netlify": "https://www.netlify.com/legal/terms-of-use/",
}

# Privacy policy URLs
PRIVACY_POLICY_URLS = {
    "github_privacy": "https://docs.github.com/en/site-policy/privacy-policies/github-privacy-statement",
    "cloudflare_privacy": "https://www.cloudflare.com/privacypolicy/",
    "mozilla_privacy": "https://www.mozilla.org/en-US/privacy/",
}


@dataclass
class LegalSource:
    """Configuration for a legal document source."""

    name: str
    source_type: str  # "license", "tos", "privacy", "court", "sec", "eu_reg"
    url: str
    domain: str = "legal"
    description: str = ""


# Inline license text for common licenses (always available, no fetch needed)
INLINE_LICENSES = {
    "MIT_inline": """MIT License

Copyright (c) [year] [fullname]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.""",
    "Apache_2_inline": """Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

1. Definitions.

"License" shall mean the terms and conditions for use, reproduction, and
distribution as defined by Sections 1 through 9 of this document.

"Licensor" shall mean the copyright owner or entity authorized by the copyright
owner that is granting the License.

"Legal Entity" shall mean the union of the acting entity and all other entities
that control, are controlled by, or are under common control with that entity.

"You" (or "Your") shall mean an individual or Legal Entity exercising permissions
granted by this License.

"Source" form shall mean the preferred form for making modifications.

"Object" form shall mean any form resulting from mechanical transformation or
translation of a Source form.

"Work" shall mean the work of authorship made available under the License.

"Derivative Works" shall mean any work that is based on the Work.

"Contribution" shall mean any work of authorship submitted for inclusion.

"Contributor" shall mean Licensor and any Legal Entity on behalf of whom a
Contribution has been received by Licensor.

2. Grant of Copyright License. Subject to the terms and conditions of this
License, each Contributor hereby grants to You a perpetual, worldwide,
non-exclusive, no-charge, royalty-free, irrevocable copyright license to
reproduce, prepare Derivative Works of, publicly display, publicly perform,
sublicense, and distribute the Work and such Derivative Works in Source or
Object form.

3. Grant of Patent License. Subject to the terms and conditions of this License,
each Contributor hereby grants to You a perpetual, worldwide, non-exclusive,
no-charge, royalty-free, irrevocable patent license to make, have made, use,
offer to sell, sell, import, and otherwise transfer the Work.

4. Redistribution. You may reproduce and distribute copies of the Work or
Derivative Works thereof in any medium, with or without modifications, and in
Source or Object form, provided that You meet the following conditions:
(a) You must give any other recipients of the Work or Derivative Works a copy of
this License; and (b) You must cause any modified files to carry prominent
notices stating that You changed the files; and (c) You must retain, in the
Source form of any Derivative Works that You distribute, all copyright, patent,
trademark, and attribution notices.

5. Submission of Contributions. Unless You explicitly state otherwise, any
Contribution intentionally submitted for inclusion in the Work by You shall be
under the terms and conditions of this License, without any additional terms.

6. Trademarks. This License does not grant permission to use the trade names,
trademarks, service marks, or product names of the Licensor.

7. Disclaimer of Warranty. Unless required by applicable law or agreed to in
writing, Licensor provides the Work on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND.

8. Limitation of Liability. In no event shall any Contributor be liable to You
for any damages, including direct, indirect, special, incidental, or
consequential damages.

9. Accepting Warranty or Additional Liability. While redistributing the Work,
You may choose to offer, and charge a fee for, acceptance of support, warranty,
indemnity, or other liability obligations.""",
    "BSD_3_inline": """BSD 3-Clause License

Copyright (c) [year], [fullname]

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software without
   specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.""",
}

# Sample legal content templates for when URLs are unavailable
SAMPLE_TOS_TEMPLATE = """TERMS OF SERVICE

Last Updated: {date}

1. ACCEPTANCE OF TERMS
By accessing or using {company} services ("Services"), you agree to be bound by
these Terms of Service ("Terms"). If you do not agree to these Terms, you may
not access or use the Services.

2. DESCRIPTION OF SERVICES
{company} provides {description}. We reserve the right to modify, suspend, or
discontinue the Services at any time without notice.

3. USER ACCOUNTS
3.1 Registration. You must register for an account to access certain features.
3.2 Account Security. You are responsible for maintaining the confidentiality
of your account credentials.
3.3 Accuracy. You agree to provide accurate and complete information.

4. USER CONDUCT
You agree not to:
(a) Violate any applicable laws or regulations;
(b) Infringe the rights of any third party;
(c) Interfere with or disrupt the Services;
(d) Attempt to gain unauthorized access to any systems;
(e) Use the Services for any illegal or unauthorized purpose.

5. INTELLECTUAL PROPERTY
5.1 Ownership. {company} retains all rights to the Services and its content.
5.2 License. Subject to these Terms, we grant you a limited, non-exclusive,
non-transferable license to use the Services.
5.3 Restrictions. You may not copy, modify, distribute, sell, or lease any
part of the Services without our prior written consent.

6. PRIVACY
Your use of the Services is subject to our Privacy Policy, which is
incorporated by reference into these Terms.

7. DISCLAIMERS
THE SERVICES ARE PROVIDED "AS IS" WITHOUT WARRANTY OF ANY KIND. WE DISCLAIM ALL
WARRANTIES, EXPRESS OR IMPLIED, INCLUDING MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE.

8. LIMITATION OF LIABILITY
TO THE MAXIMUM EXTENT PERMITTED BY LAW, {company} SHALL NOT BE LIABLE FOR ANY
INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES ARISING FROM
YOUR USE OF THE SERVICES.

9. INDEMNIFICATION
You agree to indemnify and hold harmless {company}, its affiliates, officers,
directors, employees, and agents from any claims arising from your use of the
Services or violation of these Terms.

10. TERMINATION
We may terminate or suspend your access to the Services at any time, without
prior notice or liability, for any reason.

11. GOVERNING LAW
These Terms shall be governed by the laws of {jurisdiction}, without regard to
its conflict of law provisions.

12. CHANGES TO TERMS
We reserve the right to modify these Terms at any time. Your continued use of
the Services after any changes constitutes acceptance of the new Terms.

13. CONTACT
For questions about these Terms, please contact us at {contact}.
"""

SAMPLE_PRIVACY_TEMPLATE = """PRIVACY POLICY

Effective Date: {date}

1. INTRODUCTION
{company} ("we," "us," or "our") is committed to protecting your privacy. This
Privacy Policy explains how we collect, use, disclose, and safeguard your
information when you use our services.

2. INFORMATION WE COLLECT
2.1 Personal Information
We may collect personal information that you voluntarily provide, including:
- Name and contact information
- Account credentials
- Payment information
- Communication preferences

2.2 Automatically Collected Information
When you use our services, we automatically collect:
- Device information (IP address, browser type, operating system)
- Usage data (pages visited, features used, timestamps)
- Cookies and similar technologies

3. HOW WE USE YOUR INFORMATION
We use the collected information to:
- Provide and maintain our services
- Process transactions and send related information
- Send promotional communications (with your consent)
- Respond to inquiries and provide customer support
- Improve our services and develop new features
- Comply with legal obligations

4. INFORMATION SHARING
We may share your information with:
- Service providers who assist in our operations
- Business partners for joint offerings
- Legal authorities when required by law
- Successors in the event of a merger or acquisition

5. DATA RETENTION
We retain your personal information for as long as necessary to fulfill the
purposes outlined in this Privacy Policy, unless a longer retention period is
required by law.

6. YOUR RIGHTS
Depending on your jurisdiction, you may have rights to:
- Access your personal information
- Correct inaccurate data
- Delete your information
- Object to processing
- Data portability
- Withdraw consent

7. SECURITY
We implement appropriate technical and organizational measures to protect your
personal information against unauthorized access, alteration, disclosure, or
destruction.

8. CHILDREN'S PRIVACY
Our services are not intended for children under 13. We do not knowingly
collect personal information from children under 13.

9. INTERNATIONAL TRANSFERS
Your information may be transferred to and processed in countries other than
your country of residence. We ensure appropriate safeguards are in place.

10. CHANGES TO THIS POLICY
We may update this Privacy Policy from time to time. We will notify you of
material changes by posting the new policy on our website.

11. CONTACT US
If you have questions about this Privacy Policy, please contact us at {contact}.
"""


class LegalDocumentFetcher:
    """Fetches legal documents for benchmark corpus.

    Specializes in legal document sources including licenses, ToS,
    privacy policies, and regulatory documents.

    Example:
        fetcher = LegalDocumentFetcher(cache_dir=Path("/tmp/legal_cache"))

        # Get open-source licenses
        docs = await fetcher.fetch_licenses(max_docs=10)

        # Get terms of service
        docs = await fetcher.fetch_tos(max_docs=5)

        # Get all legal documents
        docs = await fetcher.fetch_all(max_docs=50)
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        rate_limit: float = 1.0,
        timeout: float = 30,
    ) -> None:
        """Initialize fetcher.

        Args:
            cache_dir: Directory to cache fetched documents
            rate_limit: Maximum requests per second per domain
            timeout: Request timeout in seconds
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.rate_limit = rate_limit
        self.timeout = timeout

        # Use base fetcher for HTTP requests
        self._base_fetcher = OnlineDocumentFetcher(
            cache_dir=cache_dir,
            rate_limit=rate_limit,
            timeout=timeout,
            content_extractor=self._legal_content_extractor,
        )

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def fetch_licenses(
        self,
        licenses: list[str] | None = None,
        include_inline: bool = True,
        max_docs: int | None = None,
    ) -> list[BenchmarkDocument]:
        """Fetch open-source license texts.

        Args:
            licenses: License names to fetch (default: all available)
            include_inline: Include inline license texts (no network needed)
            max_docs: Maximum documents to return

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []

        # Add inline licenses first (always available)
        if include_inline:
            for name, content in INLINE_LICENSES.items():
                if max_docs and len(documents) >= max_docs:
                    break

                doc = self._create_legal_document(
                    doc_id=f"license_{name.lower()}",
                    content=content,
                    source_type="license",
                    source_name=name,
                    url=f"inline://license/{name}",
                )
                documents.append(doc)

        # Fetch from URLs
        if licenses is None:
            licenses = list(OPENSOURCE_LICENSES.keys())

        for license_name in licenses:
            if max_docs and len(documents) >= max_docs:
                break

            if license_name not in OPENSOURCE_LICENSES:
                logger.warning(f"Unknown license: {license_name}")
                continue

            url = OPENSOURCE_LICENSES[license_name]
            docs = await self._base_fetcher.fetch_urls(
                urls=[url],
                category=DocumentCategory.LEGAL,
                domain="opensource_license",
                max_docs=1,
            )

            for doc in docs:
                # Re-wrap with legal-specific metadata
                legal_doc = self._create_legal_document(
                    doc_id=f"license_{license_name.lower().replace('-', '_')}",
                    content=doc.content,
                    source_type="license",
                    source_name=license_name,
                    url=url,
                )
                documents.append(legal_doc)

        logger.info(f"Fetched {len(documents)} license documents")
        return documents

    async def fetch_tos(
        self,
        companies: list[str] | None = None,
        include_templates: bool = True,
        max_docs: int | None = None,
    ) -> list[BenchmarkDocument]:
        """Fetch Terms of Service documents.

        Args:
            companies: Company names to fetch ToS for
            include_templates: Include template ToS (no network needed)
            max_docs: Maximum documents to return

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []

        # Add template ToS (always available)
        if include_templates:
            for company_name in ["TechCorp", "CloudService", "DevPlatform"]:
                if max_docs and len(documents) >= max_docs:
                    break

                content = SAMPLE_TOS_TEMPLATE.format(
                    date="January 1, 2025",
                    company=company_name,
                    description="cloud-based software services",
                    jurisdiction="Delaware, USA",
                    contact=f"legal@{company_name.lower()}.example.com",
                )

                doc = self._create_legal_document(
                    doc_id=f"tos_{company_name.lower()}_template",
                    content=content,
                    source_type="tos",
                    source_name=f"{company_name} (Template)",
                    url=f"template://tos/{company_name.lower()}",
                )
                documents.append(doc)

        # Fetch from real URLs
        if companies is None:
            companies = list(TOS_URLS.keys())

        for company in companies:
            if max_docs and len(documents) >= max_docs:
                break

            if company not in TOS_URLS:
                logger.warning(f"Unknown company ToS: {company}")
                continue

            url = TOS_URLS[company]
            docs = await self._base_fetcher.fetch_urls(
                urls=[url],
                category=DocumentCategory.LEGAL,
                domain="terms_of_service",
                max_docs=1,
            )

            for doc in docs:
                legal_doc = self._create_legal_document(
                    doc_id=f"tos_{company.lower()}",
                    content=doc.content,
                    source_type="tos",
                    source_name=company.title(),
                    url=url,
                )
                documents.append(legal_doc)

        logger.info(f"Fetched {len(documents)} ToS documents")
        return documents

    async def fetch_privacy_policies(
        self,
        include_templates: bool = True,
        max_docs: int | None = None,
    ) -> list[BenchmarkDocument]:
        """Fetch privacy policy documents.

        Args:
            include_templates: Include template policies (no network needed)
            max_docs: Maximum documents to return

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []

        # Add template privacy policies
        if include_templates:
            for company_name in ["DataCorp", "PrivacyFirst", "SecureCloud"]:
                if max_docs and len(documents) >= max_docs:
                    break

                content = SAMPLE_PRIVACY_TEMPLATE.format(
                    date="January 1, 2025",
                    company=company_name,
                    contact=f"privacy@{company_name.lower()}.example.com",
                )

                doc = self._create_legal_document(
                    doc_id=f"privacy_{company_name.lower()}_template",
                    content=content,
                    source_type="privacy",
                    source_name=f"{company_name} (Template)",
                    url=f"template://privacy/{company_name.lower()}",
                )
                documents.append(doc)

        # Fetch from real URLs
        for name, url in PRIVACY_POLICY_URLS.items():
            if max_docs and len(documents) >= max_docs:
                break

            docs = await self._base_fetcher.fetch_urls(
                urls=[url],
                category=DocumentCategory.LEGAL,
                domain="privacy_policy",
                max_docs=1,
            )

            for doc in docs:
                legal_doc = self._create_legal_document(
                    doc_id=f"privacy_{name.lower()}",
                    content=doc.content,
                    source_type="privacy",
                    source_name=name.replace("_", " ").title(),
                    url=url,
                )
                documents.append(legal_doc)

        logger.info(f"Fetched {len(documents)} privacy policy documents")
        return documents

    async def fetch_all(
        self,
        max_docs: int | None = None,
        include_templates: bool = True,
    ) -> list[BenchmarkDocument]:
        """Fetch all legal document types.

        Args:
            max_docs: Maximum total documents to return
            include_templates: Include template documents

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []
        remaining = max_docs

        # Distribute across document types
        if max_docs:
            licenses_max = max_docs // 3
            tos_max = max_docs // 3
            privacy_max = max_docs - licenses_max - tos_max
        else:
            licenses_max = None
            tos_max = None
            privacy_max = None

        # Fetch each type
        licenses = await self.fetch_licenses(
            include_inline=include_templates,
            max_docs=licenses_max,
        )
        documents.extend(licenses)

        tos = await self.fetch_tos(
            include_templates=include_templates,
            max_docs=tos_max,
        )
        documents.extend(tos)

        privacy = await self.fetch_privacy_policies(
            include_templates=include_templates,
            max_docs=privacy_max,
        )
        documents.extend(privacy)

        # Trim to max_docs if needed
        if max_docs and len(documents) > max_docs:
            documents = documents[:max_docs]

        logger.info(f"Fetched {len(documents)} total legal documents")
        return documents

    def _create_legal_document(
        self,
        doc_id: str,
        content: str,
        source_type: str,
        source_name: str,
        url: str,
    ) -> BenchmarkDocument:
        """Create a BenchmarkDocument for legal content.

        Args:
            doc_id: Unique document ID
            content: Document content
            source_type: Type of legal document
            source_name: Name of source
            url: Source URL

        Returns:
            BenchmarkDocument instance
        """
        # Determine source based on URL
        if url.startswith("inline://") or url.startswith("template://"):
            source = DocumentSource.SYNTHETIC
        else:
            source = DocumentSource.ONLINE

        tags = [
            "legal",
            source_type,
            source_name.lower().replace(" ", "_"),
        ]

        # Add legal-specific tags based on content
        content_lower = content.lower()
        if "indemnif" in content_lower:
            tags.append("indemnification")
        if "liability" in content_lower:
            tags.append("liability")
        if "warrant" in content_lower:
            tags.append("warranty")
        if "terminat" in content_lower:
            tags.append("termination")
        if "govern" in content_lower and "law" in content_lower:
            tags.append("governing_law")

        return BenchmarkDocument(
            doc_id=f"legal_{doc_id}",
            source=source,
            category=DocumentCategory.LEGAL,
            domain=f"legal_{source_type}",
            file_path=url,
            content=content,
            semantic_tags=sorted(set(tags)),
            metadata={
                "source_type": source_type,
                "source_name": source_name,
                "url": url,
            },
        )

    def _legal_content_extractor(
        self, html: str, url: str
    ) -> tuple[str, str | None]:
        """Extract content from legal HTML pages.

        Optimized for legal document formatting.

        Args:
            html: Raw HTML content
            url: Source URL

        Returns:
            Tuple of (content, title)
        """
        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else None

        # Remove navigation, headers, footers (common in legal pages)
        content = re.sub(r"<nav[^>]*>.*?</nav>", "", html, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<header[^>]*>.*?</header>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<footer[^>]*>.*?</footer>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<aside[^>]*>.*?</aside>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Remove script and style tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL | re.IGNORECASE)

        # Preserve numbered lists and sections (important for legal docs)
        content = re.sub(r"<li[^>]*>", "\n• ", content)
        content = re.sub(r"<h[1-6][^>]*>", "\n\n## ", content)
        content = re.sub(r"</h[1-6]>", "\n", content)

        # Remove remaining HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Decode HTML entities
        content = re.sub(r"&nbsp;", " ", content)
        content = re.sub(r"&amp;", "&", content)
        content = re.sub(r"&lt;", "<", content)
        content = re.sub(r"&gt;", ">", content)
        content = re.sub(r"&quot;", '"', content)
        content = re.sub(r"&#39;", "'", content)
        content = re.sub(r"&sect;", "§", content)

        # Normalize whitespace while preserving paragraph breaks
        content = re.sub(r"\n{3,}", "\n\n", content)
        content = re.sub(r"[ \t]+", " ", content)
        content = content.strip()

        return content, title
