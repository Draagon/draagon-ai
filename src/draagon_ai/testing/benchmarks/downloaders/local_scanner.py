"""Local filesystem document scanner for benchmark corpus.

Scans local directories with configurable glob patterns, size filtering,
and exclusion patterns to collect documents for benchmark corpus.

Supports text files (.md, .py, .txt, etc.) and PDF files with text extraction.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Callable

from ..corpus import BenchmarkDocument, DocumentCategory, DocumentSource

# Optional PDF support
try:
    from pypdf import PdfReader

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

logger = logging.getLogger(__name__)


def extract_pdf_text(file_path: Path) -> str | None:
    """Extract text content from a PDF file.

    Args:
        file_path: Path to PDF file

    Returns:
        Extracted text or None if extraction fails
    """
    if not PDF_SUPPORT:
        logger.warning("PDF support not available (install pypdf)")
        return None

    try:
        reader = PdfReader(file_path)
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

        if not text_parts:
            logger.debug(f"No text extracted from PDF: {file_path}")
            return None

        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF {file_path}: {e}")
        return None


# Default exclusion patterns for common build/dependency directories
DEFAULT_EXCLUDE_PATTERNS = [
    "**/node_modules/**",
    "**/target/**",
    "**/.archive/**",
    "**/__pycache__/**",
    "**/.git/**",
    "**/venv/**",
    "**/.venv/**",
    "**/build/**",
    "**/dist/**",
    "**/.tox/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/coverage/**",
    "**/*.egg-info/**",
]

# Keywords to detect in content for semantic tagging
CONTENT_KEYWORDS = [
    "async",
    "await",
    "class",
    "function",
    "def",
    "test",
    "api",
    "http",
    "database",
    "memory",
    "agent",
    "llm",
    "embedding",
    "vector",
    "graph",
    "query",
    "retrieval",
    "benchmark",
    "config",
    "install",
    "usage",
    "example",
]


class LocalDocumentScanner:
    """Scans local filesystem to collect benchmark documents.

    Supports glob patterns, size filtering, exclusion patterns, and
    automatic deduplication via content hashing.

    Example:
        scanner = LocalDocumentScanner(
            root_path=Path.home() / "Development",
            patterns=["**/*.md", "**/*.py"],
            size_range=(1024, 500_000),
        )
        docs = scanner.scan(max_docs=100)
    """

    def __init__(
        self,
        root_path: str | Path,
        patterns: list[str] | None = None,
        size_range: tuple[int, int] = (1024, 500_000),
        exclude_patterns: list[str] | None = None,
        category_inferrer: Callable[[Path, str], DocumentCategory] | None = None,
    ) -> None:
        """Initialize scanner.

        Args:
            root_path: Root directory to scan
            patterns: Glob patterns to match (default: ["**/*.md", "**/*.py"])
            size_range: (min_bytes, max_bytes) for file size filtering
            exclude_patterns: Patterns to exclude (default: node_modules, etc.)
            category_inferrer: Optional custom function to infer category
        """
        self.root_path = Path(root_path).expanduser().resolve()
        self.patterns = patterns or ["**/*.md", "**/*.py"]
        self.min_size, self.max_size = size_range
        self.exclude_patterns = exclude_patterns or DEFAULT_EXCLUDE_PATTERNS
        self.category_inferrer = category_inferrer or self._infer_category

        if not self.root_path.exists():
            raise ValueError(f"Root path does not exist: {self.root_path}")
        if not self.root_path.is_dir():
            raise ValueError(f"Root path is not a directory: {self.root_path}")

    def scan(self, max_docs: int | None = None) -> list[BenchmarkDocument]:
        """Scan filesystem and return documents.

        Args:
            max_docs: Maximum documents to return (None = unlimited)

        Returns:
            List of BenchmarkDocument instances
        """
        documents: list[BenchmarkDocument] = []
        seen_hashes: set[str] = set()
        files_scanned = 0
        files_skipped_size = 0
        files_skipped_excluded = 0
        files_skipped_duplicate = 0
        files_skipped_error = 0

        logger.info(f"Starting scan of {self.root_path} with patterns {self.patterns}")

        for pattern in self.patterns:
            for file_path in self.root_path.glob(pattern):
                # Check if we've reached max_docs
                if max_docs is not None and len(documents) >= max_docs:
                    logger.info(f"Reached max_docs limit ({max_docs})")
                    break

                # Skip non-files
                if not file_path.is_file():
                    continue

                files_scanned += 1

                # Log progress every 100 files
                if files_scanned % 100 == 0:
                    logger.info(f"Scanned {files_scanned} files, found {len(documents)} documents")

                # Check exclusion patterns
                if self._is_excluded(file_path):
                    files_skipped_excluded += 1
                    continue

                # Check file size
                try:
                    file_size = file_path.stat().st_size
                except OSError as e:
                    logger.warning(f"Cannot stat {file_path}: {e}")
                    files_skipped_error += 1
                    continue

                if file_size < self.min_size or file_size > self.max_size:
                    files_skipped_size += 1
                    continue

                # Read content (handle PDFs differently)
                try:
                    if file_path.suffix.lower() == ".pdf":
                        content = extract_pdf_text(file_path)
                        if content is None:
                            files_skipped_error += 1
                            continue
                    else:
                        content = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError as e:
                    logger.warning(f"Unicode decode error in {file_path}: {e}")
                    files_skipped_error += 1
                    continue
                except PermissionError as e:
                    logger.warning(f"Permission denied for {file_path}: {e}")
                    files_skipped_error += 1
                    continue
                except OSError as e:
                    logger.warning(f"Cannot read {file_path}: {e}")
                    files_skipped_error += 1
                    continue

                # Check for duplicate content
                content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
                if content_hash in seen_hashes:
                    logger.debug(f"Skipping duplicate: {file_path}")
                    files_skipped_duplicate += 1
                    continue
                seen_hashes.add(content_hash)

                # Create document
                doc = self._create_document(file_path, content)
                documents.append(doc)

            # Check again after pattern loop
            if max_docs is not None and len(documents) >= max_docs:
                break

        logger.info(
            f"Scan complete: {len(documents)} documents from {files_scanned} files. "
            f"Skipped: {files_skipped_size} size, {files_skipped_excluded} excluded, "
            f"{files_skipped_duplicate} duplicate, {files_skipped_error} errors"
        )

        return documents

    def _is_excluded(self, file_path: Path) -> bool:
        """Check if file matches any exclusion pattern.

        Args:
            file_path: Path to check

        Returns:
            True if file should be excluded
        """
        # Get path parts for directory-level matching
        path_parts = file_path.parts

        for pattern in self.exclude_patterns:
            # Handle ** patterns by checking if directory appears in path parts
            if "**" in pattern:
                # Extract the key directory (e.g., "node_modules" from "**/node_modules/**")
                key_dir = pattern.replace("**", "").strip("/").strip("*")
                if key_dir:
                    # Check if key_dir is an actual path component, not just a substring
                    if key_dir in path_parts:
                        return True
            elif file_path.match(pattern):
                return True
        return False

    def _create_document(self, file_path: Path, content: str) -> BenchmarkDocument:
        """Create BenchmarkDocument from file.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            BenchmarkDocument instance
        """
        doc_id = self._generate_doc_id(file_path)
        category = self.category_inferrer(file_path, content)
        domain = self._infer_domain(file_path, content)
        tags = self._extract_tags(file_path, content)

        return BenchmarkDocument(
            doc_id=doc_id,
            source=DocumentSource.LOCAL,
            category=category,
            domain=domain,
            file_path=str(file_path),
            content=content,
            semantic_tags=tags,
            metadata={
                "relative_path": str(file_path.relative_to(self.root_path)),
                "extension": file_path.suffix.lower(),
            },
        )

    def _generate_doc_id(self, file_path: Path) -> str:
        """Generate unique document ID from file path.

        Args:
            file_path: Path to file

        Returns:
            Document ID string
        """
        try:
            rel_path = file_path.relative_to(self.root_path)
        except ValueError:
            rel_path = file_path

        # Convert path to ID: replace separators, remove extension
        doc_id = str(rel_path).replace("/", "_").replace("\\", "_")
        if "." in doc_id:
            doc_id = doc_id.rsplit(".", 1)[0]

        # Sanitize for use as ID
        doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)

        return f"local_{doc_id}"

    def _infer_category(self, file_path: Path, content: str) -> DocumentCategory:
        """Infer document category from path and content.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            DocumentCategory enum value
        """
        path_str = str(file_path).lower()
        content_lower = content[:2000].lower()  # Check first 2000 chars

        # Check for narrative/creative content
        if "party-lore" in path_str or "story" in path_str or "fiction" in path_str:
            return DocumentCategory.NARRATIVE

        # Check for legal content
        if any(term in content_lower for term in [
            "terms of service", "privacy policy", "license agreement",
            "indemnification", "liability", "hereby grants"
        ]):
            return DocumentCategory.LEGAL

        # Check for academic/research content
        if any(term in content_lower for term in [
            "abstract:", "references:", "arxiv", "doi:", "et al."
        ]):
            return DocumentCategory.ACADEMIC

        # Check for knowledge base / FAQ
        if any(term in content_lower for term in [
            "frequently asked", "faq", "how to", "step 1:", "q:", "a:"
        ]):
            return DocumentCategory.KNOWLEDGE_BASE

        # Check for conversational
        if any(term in content_lower for term in [
            "hey ", "hi ", "thanks!", "cheers", "from:", "to:", "subject:"
        ]):
            return DocumentCategory.CONVERSATIONAL

        # Default to technical for code and docs
        return DocumentCategory.TECHNICAL

    def _infer_domain(self, file_path: Path, content: str) -> str:
        """Infer domain from file path.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            Domain string (e.g., "python", "ai_framework")
        """
        path_str = str(file_path).lower()
        extension = file_path.suffix.lower()

        # Check for known project domains
        if "draagon" in path_str:
            return "ai_framework"
        if "party-lore" in path_str:
            return "game_fiction"
        if "metaobjects" in path_str:
            return "java_framework"
        if "roxy" in path_str:
            return "voice_assistant"

        # Infer from extension
        extension_domains = {
            ".py": "python",
            ".java": "java",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".rs": "rust",
            ".go": "golang",
            ".rb": "ruby",
            ".md": "documentation",
            ".rst": "documentation",
            ".txt": "text",
            ".json": "config",
            ".yaml": "config",
            ".yml": "config",
            ".toml": "config",
        }

        return extension_domains.get(extension, "unknown")

    def _extract_tags(self, file_path: Path, content: str) -> list[str]:
        """Extract semantic tags from file.

        Args:
            file_path: Path to file
            content: File content

        Returns:
            List of tag strings
        """
        tags: set[str] = set()

        # Add extension as tag
        ext = file_path.suffix.lower().lstrip(".")
        if ext:
            tags.add(ext)

        # Add filename keywords
        stem = file_path.stem.lower()
        if "test" in stem:
            tags.add("test")
        if "readme" in stem:
            tags.add("readme")
        if "config" in stem:
            tags.add("config")
        if "claude" in stem:
            tags.add("claude")

        # Check content for keywords (first 500 chars for performance)
        content_lower = content[:500].lower()
        for keyword in CONTENT_KEYWORDS:
            if keyword in content_lower:
                tags.add(keyword)

        return sorted(tags)
