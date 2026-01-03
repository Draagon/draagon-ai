"""Tests for LocalDocumentScanner.

Tests filesystem scanning functionality including glob patterns,
size filtering, exclusion patterns, and deduplication.
"""

import tempfile
from pathlib import Path

import pytest

from draagon_ai.testing.benchmarks import (
    LocalDocumentScanner,
    DocumentCategory,
    DocumentSource,
)


class TestLocalDocumentScanner:
    """Tests for LocalDocumentScanner class."""

    def test_scanner_initialization(self, tmp_path: Path) -> None:
        """Scanner initializes with valid path."""
        scanner = LocalDocumentScanner(root_path=tmp_path)
        assert scanner.root_path == tmp_path.resolve()

    def test_scanner_invalid_path_raises(self) -> None:
        """Scanner raises on non-existent path."""
        with pytest.raises(ValueError, match="does not exist"):
            LocalDocumentScanner(root_path="/nonexistent/path")

    def test_scanner_file_path_raises(self, tmp_path: Path) -> None:
        """Scanner raises when given a file instead of directory."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("content")
        with pytest.raises(ValueError, match="not a directory"):
            LocalDocumentScanner(root_path=file_path)

    def test_scan_finds_markdown_files(self, tmp_path: Path) -> None:
        """Scanner finds matching markdown files."""
        (tmp_path / "test1.md").write_text("# Test Document 1\n\nThis is content.")
        (tmp_path / "test2.md").write_text("# Test Document 2\n\nMore content here.")
        (tmp_path / "ignored.txt").write_text("This should be ignored")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 2
        assert all(doc.source == DocumentSource.LOCAL for doc in docs)

    def test_scan_finds_python_files(self, tmp_path: Path) -> None:
        """Scanner finds matching Python files."""
        (tmp_path / "module.py").write_text("def hello():\n    return 'world'")
        (tmp_path / "test_module.py").write_text("def test_hello():\n    assert True")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 2

    def test_scan_multiple_patterns(self, tmp_path: Path) -> None:
        """Scanner handles multiple glob patterns."""
        (tmp_path / "readme.md").write_text("# README with enough content")
        (tmp_path / "module.py").write_text("def func(): return True")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md", "**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 2

    def test_scan_nested_directories(self, tmp_path: Path) -> None:
        """Scanner finds files in nested directories."""
        nested = tmp_path / "level1" / "level2" / "level3"
        nested.mkdir(parents=True)
        (nested / "deep.md").write_text("# Deep nested file with content")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert "level3" in docs[0].file_path


class TestSizeFiltering:
    """Tests for file size filtering."""

    def test_filters_too_small(self, tmp_path: Path) -> None:
        """Scanner excludes files below min_size."""
        (tmp_path / "too_small.md").write_text("x")  # 1 byte
        (tmp_path / "just_right.md").write_text("x" * 100)  # 100 bytes

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(50, 500_000),  # min 50 bytes
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert "just_right" in docs[0].doc_id

    def test_filters_too_large(self, tmp_path: Path) -> None:
        """Scanner excludes files above max_size."""
        (tmp_path / "too_large.md").write_text("x" * 1000)  # 1000 bytes
        (tmp_path / "just_right.md").write_text("x" * 100)  # 100 bytes

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500),  # max 500 bytes
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert "just_right" in docs[0].doc_id


class TestExclusionPatterns:
    """Tests for exclusion pattern filtering."""

    def test_excludes_node_modules(self, tmp_path: Path) -> None:
        """Scanner excludes node_modules directory."""
        (tmp_path / "good.md").write_text("# Good file with content here")
        node_modules = tmp_path / "node_modules"
        node_modules.mkdir()
        (node_modules / "bad.md").write_text("# Should be excluded from scan")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        # Verify the good file was found, not the excluded one
        assert docs[0].doc_id.endswith("good")

    def test_excludes_pycache(self, tmp_path: Path) -> None:
        """Scanner excludes __pycache__ directory."""
        (tmp_path / "good.py").write_text("def good(): return True")
        pycache = tmp_path / "__pycache__"
        pycache.mkdir()
        (pycache / "bad.py").write_text("# cached bytecode should be excluded")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert "__pycache__" not in docs[0].file_path

    def test_excludes_target(self, tmp_path: Path) -> None:
        """Scanner excludes target directory (Maven/Cargo builds)."""
        (tmp_path / "good.md").write_text("# Source file with content")
        target = tmp_path / "target"
        target.mkdir()
        (target / "bad.md").write_text("# Build output should be excluded")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        # Verify the good file was found, not the excluded one
        assert docs[0].doc_id.endswith("good")

    def test_excludes_git(self, tmp_path: Path) -> None:
        """Scanner excludes .git directory."""
        (tmp_path / "good.md").write_text("# Source file with content")
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("[core]\n  repositoryformatversion = 0")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert ".git" not in docs[0].file_path

    def test_custom_exclude_patterns(self, tmp_path: Path) -> None:
        """Scanner uses custom exclusion patterns."""
        (tmp_path / "good.md").write_text("# Good file with content")
        custom_dir = tmp_path / "custom_exclude"
        custom_dir.mkdir()
        (custom_dir / "bad.md").write_text("# Should be excluded with custom pattern")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
            exclude_patterns=["**/custom_exclude/**"],
        )

        docs = scanner.scan()
        assert len(docs) == 1
        # Verify the good file was found, not the excluded one
        assert docs[0].doc_id.endswith("good")


class TestDeduplication:
    """Tests for content deduplication."""

    def test_deduplicates_identical_content(self, tmp_path: Path) -> None:
        """Scanner removes files with identical content."""
        content = "# Same content in multiple files\n\nThis is duplicated."
        (tmp_path / "file1.md").write_text(content)
        (tmp_path / "file2.md").write_text(content)
        (tmp_path / "file3.md").write_text(content)

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1  # Only one kept

    def test_keeps_different_content(self, tmp_path: Path) -> None:
        """Scanner keeps files with different content."""
        (tmp_path / "file1.md").write_text("# Content A\n\nFirst file.")
        (tmp_path / "file2.md").write_text("# Content B\n\nSecond file.")
        (tmp_path / "file3.md").write_text("# Content C\n\nThird file.")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 3


class TestMaxDocs:
    """Tests for max_docs limit."""

    def test_respects_max_docs(self, tmp_path: Path) -> None:
        """Scanner stops at max_docs limit."""
        for i in range(10):
            (tmp_path / f"file{i}.md").write_text(f"# Document {i}\n\nUnique content {i}.")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan(max_docs=5)
        assert len(docs) == 5

    def test_unlimited_when_none(self, tmp_path: Path) -> None:
        """Scanner returns all docs when max_docs is None."""
        for i in range(10):
            (tmp_path / f"file{i}.md").write_text(f"# Document {i}\n\nUnique content {i}.")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan(max_docs=None)
        assert len(docs) == 10


class TestDomainInference:
    """Tests for domain inference from file paths."""

    def test_infers_python_domain(self, tmp_path: Path) -> None:
        """Infers python domain for .py files."""
        (tmp_path / "module.py").write_text("def function(): return True")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].domain == "python"

    def test_infers_typescript_domain(self, tmp_path: Path) -> None:
        """Infers typescript domain for .ts files."""
        (tmp_path / "component.ts").write_text("export function test(): void {}")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.ts"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].domain == "typescript"

    def test_infers_ai_framework_for_draagon(self, tmp_path: Path) -> None:
        """Infers ai_framework domain for draagon paths."""
        draagon = tmp_path / "draagon-ai"
        draagon.mkdir()
        (draagon / "readme.md").write_text("# draagon-ai Framework\n\nContent here.")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].domain == "ai_framework"

    def test_infers_game_fiction_for_party_lore(self, tmp_path: Path) -> None:
        """Infers game_fiction domain for party-lore paths."""
        party_lore = tmp_path / "party-lore"
        party_lore.mkdir()
        (party_lore / "story.md").write_text("# Adventure Story\n\nOnce upon a time...")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].domain == "game_fiction"


class TestCategoryInference:
    """Tests for category inference from content."""

    def test_infers_technical_by_default(self, tmp_path: Path) -> None:
        """Defaults to technical category."""
        (tmp_path / "code.py").write_text("def regular_function(): pass")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].category == DocumentCategory.TECHNICAL

    def test_infers_narrative_for_story(self, tmp_path: Path) -> None:
        """Infers narrative for story paths."""
        party_lore = tmp_path / "party-lore"
        party_lore.mkdir()
        (party_lore / "tale.md").write_text("# The Great Tale\n\nIn a land far away...")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].category == DocumentCategory.NARRATIVE

    def test_infers_legal_for_tos(self, tmp_path: Path) -> None:
        """Infers legal for terms of service content."""
        (tmp_path / "tos.md").write_text(
            "# Terms of Service\n\nBy using this service, you agree to these terms."
        )

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert docs[0].category == DocumentCategory.LEGAL


class TestSemanticTagging:
    """Tests for semantic tag extraction."""

    def test_adds_extension_tag(self, tmp_path: Path) -> None:
        """Adds file extension as tag."""
        (tmp_path / "module.py").write_text("def function(): return True")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert "py" in docs[0].semantic_tags

    def test_detects_async_keyword(self, tmp_path: Path) -> None:
        """Detects async keyword in content."""
        (tmp_path / "async_module.py").write_text("async def handler(): await something()")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert "async" in docs[0].semantic_tags

    def test_detects_test_in_filename(self, tmp_path: Path) -> None:
        """Detects 'test' in filename."""
        (tmp_path / "test_module.py").write_text("def test_something(): assert True")

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.py"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert "test" in docs[0].semantic_tags


class TestErrorHandling:
    """Tests for error handling."""

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Handles empty directory gracefully."""
        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
        )

        docs = scanner.scan()
        assert len(docs) == 0

    def test_unicode_content(self, tmp_path: Path) -> None:
        """Handles unicode content correctly."""
        content = "# Unicode Test 🎉\n\n日本語テキスト with emoji 🔥💯"
        (tmp_path / "unicode.md").write_text(content)

        scanner = LocalDocumentScanner(
            root_path=tmp_path,
            patterns=["**/*.md"],
            size_range=(10, 500_000),
        )

        docs = scanner.scan()
        assert len(docs) == 1
        assert "🎉" in docs[0].content


class TestRealFilesystem:
    """Integration tests with real filesystem (optional)."""

    @pytest.mark.integration
    def test_scan_real_development_directory(self) -> None:
        """Scan real ~/Development directory."""
        dev_path = Path.home() / "Development"
        if not dev_path.exists():
            pytest.skip("~/Development does not exist")

        scanner = LocalDocumentScanner(
            root_path=dev_path,
            patterns=["**/*.md"],
            size_range=(1024, 100_000),  # 1KB - 100KB
        )

        docs = scanner.scan(max_docs=50)

        # Should find some documents
        assert len(docs) > 0
        assert all(doc.source == DocumentSource.LOCAL for doc in docs)
        assert all(1024 <= doc.size_bytes <= 100_000 for doc in docs)
