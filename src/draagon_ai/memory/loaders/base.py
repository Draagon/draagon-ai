"""Base classes for document loaders.

Provides the common interface and data structures for loading
documents into the semantic graph memory system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from draagon_ai.memory.base import MemoryType, MemoryScope


@dataclass
class LoaderConfig:
    """Configuration for document loaders.

    Attributes:
        chunk_size: Maximum characters per chunk (0 = no chunking)
        chunk_overlap: Characters to overlap between chunks
        default_memory_type: Default type for loaded memories
        default_scope: Default scope for loaded memories
        extract_metadata: Whether to extract file metadata
        preserve_structure: Whether to preserve document structure (headings, etc.)
    """

    chunk_size: int = 0  # 0 = no chunking, load as single document
    chunk_overlap: int = 200
    default_memory_type: MemoryType = MemoryType.KNOWLEDGE
    default_scope: MemoryScope = MemoryScope.AGENT
    extract_metadata: bool = True
    preserve_structure: bool = True


@dataclass
class Document:
    """A loaded document ready for memory storage.

    Represents a single piece of content extracted from a file,
    with associated metadata for proper memory classification.
    """

    # Content
    content: str
    source_path: str

    # Memory classification
    memory_type: MemoryType = MemoryType.KNOWLEDGE
    scope: MemoryScope = MemoryScope.AGENT

    # Metadata
    title: str | None = None
    section: str | None = None  # e.g., heading path: "Installation > Requirements"
    file_type: str | None = None  # e.g., "markdown", "text", "json"
    chunk_index: int | None = None  # If document was chunked
    total_chunks: int | None = None

    # Source info
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime | None = None
    file_size: int | None = None

    # Additional metadata (extensible)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Extracted entities (if pre-extracted)
    entities: list[str] = field(default_factory=list)

    @property
    def filename(self) -> str:
        """Get the filename from source path."""
        return Path(self.source_path).name

    @property
    def extension(self) -> str:
        """Get the file extension."""
        return Path(self.source_path).suffix.lower()


class DocumentLoader(ABC):
    """Abstract base class for document loaders.

    Subclasses implement loading logic for specific file formats.
    All loaders produce Document objects that can be stored in memory.
    """

    def __init__(self, config: LoaderConfig | None = None):
        """Initialize the loader.

        Args:
            config: Loader configuration
        """
        self.config = config or LoaderConfig()

    @property
    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Get list of supported file extensions (e.g., ['.md', '.markdown'])."""
        ...

    @abstractmethod
    async def load(self, path: str | Path) -> list[Document]:
        """Load documents from a file.

        Args:
            path: Path to the file

        Returns:
            List of Document objects
        """
        ...

    def supports(self, path: str | Path) -> bool:
        """Check if this loader supports the given file.

        Args:
            path: Path to check

        Returns:
            True if the file extension is supported
        """
        ext = Path(path).suffix.lower()
        return ext in self.supported_extensions

    def _get_file_metadata(self, path: Path) -> dict[str, Any]:
        """Extract metadata from file stats.

        Args:
            path: Path to the file

        Returns:
            Metadata dict
        """
        if not path.exists():
            return {}

        stat = path.stat()
        return {
            "file_size": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
            "created_at": datetime.fromtimestamp(stat.st_ctime),
        }

    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int,
    ) -> list[tuple[str, int]]:
        """Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Maximum chunk size
            overlap: Overlap between chunks

        Returns:
            List of (chunk_text, chunk_index) tuples
        """
        if chunk_size <= 0 or len(text) <= chunk_size:
            return [(text, 0)]

        chunks = []
        start = 0
        chunk_idx = 0

        while start < len(text):
            # Find end of chunk
            end = min(start + chunk_size, len(text))

            # Try to break at sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + chunk_size // 2:
                    end = para_break + 2

                # Or sentence break
                elif text.rfind(". ", start, end) > start + chunk_size // 2:
                    end = text.rfind(". ", start, end) + 2

            chunk = text[start:end].strip()
            if chunk:
                chunks.append((chunk, chunk_idx))
                chunk_idx += 1

            # Move start with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def _infer_memory_type(self, content: str, metadata: dict[str, Any]) -> MemoryType:
        """Infer memory type from content and metadata.

        Subclasses can override for format-specific inference.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Inferred MemoryType
        """
        # Check for explicit type in metadata
        if "memory_type" in metadata:
            try:
                return MemoryType(metadata["memory_type"])
            except ValueError:
                pass

        # Check for patterns suggesting specific types
        content_lower = content.lower()

        # Instructions/directives
        if any(marker in content_lower for marker in [
            "always ", "never ", "must ", "should always",
            "remember to ", "don't forget",
        ]):
            return MemoryType.INSTRUCTION

        # Procedural/how-to
        if any(marker in content_lower for marker in [
            "how to ", "step 1", "step 2", "# steps",
            "installation", "usage:", "example:",
        ]):
            return MemoryType.SKILL

        # Preferences
        if any(marker in content_lower for marker in [
            "prefer", "i like", "favorite", "setting:",
        ]):
            return MemoryType.PREFERENCE

        return self.config.default_memory_type
