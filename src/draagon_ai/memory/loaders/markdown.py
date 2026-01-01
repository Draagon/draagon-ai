"""Markdown document loader.

Loads markdown files with structure-aware parsing that preserves
heading hierarchy, code blocks, and other semantic elements.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import Document, DocumentLoader, LoaderConfig
from draagon_ai.memory.base import MemoryType, MemoryScope


@dataclass
class MarkdownSection:
    """A section of a markdown document defined by its heading."""

    heading: str
    level: int  # 1-6 for #-######
    content: str
    heading_path: list[str]  # e.g., ["Installation", "Requirements"]
    start_line: int
    end_line: int
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkdownLoader(DocumentLoader):
    """Loader for Markdown files.

    Features:
    - Preserves heading hierarchy
    - Extracts frontmatter (YAML between ---)
    - Handles code blocks
    - Can chunk by section or by size

    Example:
        loader = MarkdownLoader()
        docs = await loader.load("README.md")

        # With section-based chunking
        config = LoaderConfig(preserve_structure=True)
        loader = MarkdownLoader(config)
        docs = await loader.load("docs/architecture.md")
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".md", ".markdown", ".mdown"]

    async def load(self, path: str | Path) -> list[Document]:
        """Load a markdown file.

        Args:
            path: Path to the markdown file

        Returns:
            List of Document objects (one per section if preserve_structure=True)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        file_meta = self._get_file_metadata(path)

        # Extract frontmatter if present
        frontmatter, content = self._extract_frontmatter(content)

        documents = []

        if self.config.preserve_structure:
            # Parse into sections based on headings
            sections = self._parse_sections(content)

            for section in sections:
                if not section.content.strip():
                    continue

                # Merge frontmatter into section metadata
                meta = {**frontmatter, **section.metadata}

                doc = Document(
                    content=section.content,
                    source_path=str(path),
                    memory_type=self._infer_memory_type(section.content, meta),
                    scope=self.config.default_scope,
                    title=section.heading or path.stem,
                    section=" > ".join(section.heading_path) if section.heading_path else None,
                    file_type="markdown",
                    metadata=meta,
                    **file_meta,
                )
                documents.append(doc)
        else:
            # Load as single document or chunk by size
            if self.config.chunk_size > 0:
                chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
                for chunk_text, idx in chunks:
                    doc = Document(
                        content=chunk_text,
                        source_path=str(path),
                        memory_type=self._infer_memory_type(chunk_text, frontmatter),
                        scope=self.config.default_scope,
                        title=frontmatter.get("title", path.stem),
                        file_type="markdown",
                        chunk_index=idx,
                        total_chunks=len(chunks),
                        metadata=frontmatter,
                        **file_meta,
                    )
                    documents.append(doc)
            else:
                # Single document
                doc = Document(
                    content=content,
                    source_path=str(path),
                    memory_type=self._infer_memory_type(content, frontmatter),
                    scope=self.config.default_scope,
                    title=frontmatter.get("title", path.stem),
                    file_type="markdown",
                    metadata=frontmatter,
                    **file_meta,
                )
                documents.append(doc)

        return documents

    def _extract_frontmatter(self, content: str) -> tuple[dict[str, Any], str]:
        """Extract YAML frontmatter from markdown.

        Args:
            content: Full markdown content

        Returns:
            Tuple of (frontmatter_dict, remaining_content)
        """
        frontmatter: dict[str, Any] = {}

        # Check for YAML frontmatter
        if content.startswith("---"):
            parts = content.split("---", 2)
            if len(parts) >= 3:
                yaml_content = parts[1].strip()
                try:
                    import yaml
                    frontmatter = yaml.safe_load(yaml_content) or {}
                except ImportError:
                    # Parse simple key: value pairs
                    for line in yaml_content.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            frontmatter[key.strip()] = value.strip()
                except Exception:
                    pass

                content = parts[2]

        return frontmatter, content

    def _parse_sections(self, content: str) -> list[MarkdownSection]:
        """Parse markdown into sections based on headings.

        Args:
            content: Markdown content (without frontmatter)

        Returns:
            List of MarkdownSection objects
        """
        sections: list[MarkdownSection] = []
        lines = content.split("\n")

        # Track heading hierarchy
        heading_stack: list[tuple[int, str]] = []

        current_section = MarkdownSection(
            heading="",
            level=0,
            content="",
            heading_path=[],
            start_line=0,
            end_line=0,
        )

        in_code_block = False

        for i, line in enumerate(lines):
            # Track code blocks to avoid treating # in code as headings
            if line.strip().startswith("```"):
                in_code_block = not in_code_block
                current_section.content += line + "\n"
                continue

            if in_code_block:
                current_section.content += line + "\n"
                continue

            # Check for heading
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                # Save current section
                if current_section.content.strip():
                    current_section.end_line = i - 1
                    sections.append(current_section)

                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()

                # Update heading stack
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, heading))

                # Build heading path
                heading_path = [h[1] for h in heading_stack]

                current_section = MarkdownSection(
                    heading=heading,
                    level=level,
                    content="",
                    heading_path=heading_path,
                    start_line=i,
                    end_line=0,
                )
            else:
                current_section.content += line + "\n"

        # Don't forget the last section
        if current_section.content.strip():
            current_section.end_line = len(lines) - 1
            sections.append(current_section)

        return sections

    def _infer_memory_type(self, content: str, metadata: dict[str, Any]) -> MemoryType:
        """Infer memory type from markdown content and metadata.

        Args:
            content: Section content
            metadata: Document/section metadata

        Returns:
            Inferred MemoryType
        """
        # Check frontmatter type
        if "type" in metadata:
            type_str = str(metadata["type"]).lower()
            type_map = {
                "fact": MemoryType.FACT,
                "instruction": MemoryType.INSTRUCTION,
                "skill": MemoryType.SKILL,
                "preference": MemoryType.PREFERENCE,
                "knowledge": MemoryType.KNOWLEDGE,
                "insight": MemoryType.INSIGHT,
            }
            if type_str in type_map:
                return type_map[type_str]

        # Infer from content patterns
        content_lower = content.lower()

        # Code-heavy content is likely a skill/how-to
        code_block_count = content.count("```")
        if code_block_count >= 2:
            return MemoryType.SKILL

        # API docs are knowledge
        if "## api" in content_lower or "### parameters" in content_lower:
            return MemoryType.KNOWLEDGE

        # Configuration is preference-like
        if "config" in content_lower or "settings" in content_lower:
            return MemoryType.PREFERENCE

        return super()._infer_memory_type(content, metadata)
