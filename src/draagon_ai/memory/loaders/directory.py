"""Directory document loader.

Recursively loads documents from directories, delegating to
appropriate loaders based on file extension.
"""

import logging
from pathlib import Path
from typing import Any

from .base import Document, DocumentLoader, LoaderConfig
from .markdown import MarkdownLoader
from .text import TextLoader

logger = logging.getLogger(__name__)


class DirectoryLoader(DocumentLoader):
    """Loader for directories of documents.

    Recursively walks directories and loads files using the
    appropriate loader for each file type.

    Features:
    - Recursive directory traversal
    - Extension filtering
    - Exclusion patterns
    - Parallel loading (optional)

    Example:
        loader = DirectoryLoader(extensions=[".md", ".txt"])
        docs = await loader.load("path/to/docs")

        # With exclusions
        loader = DirectoryLoader(
            extensions=[".md"],
            exclude_patterns=["node_modules", ".git", "__pycache__"],
        )
        docs = await loader.load("project/")
    """

    def __init__(
        self,
        config: LoaderConfig | None = None,
        extensions: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        loaders: dict[str, DocumentLoader] | None = None,
    ):
        """Initialize the directory loader.

        Args:
            config: Loader configuration
            extensions: File extensions to include (e.g., [".md", ".txt"])
            exclude_patterns: Directory/file patterns to exclude
            loaders: Custom loaders for specific extensions
        """
        super().__init__(config)

        self.extensions = extensions or [".md", ".markdown", ".txt"]
        self.exclude_patterns = exclude_patterns or [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            ".mypy_cache",
            "__MACOSX",
            ".DS_Store",
        ]

        # Build loader registry
        self._loaders: dict[str, DocumentLoader] = {}

        # Default loaders
        md_loader = MarkdownLoader(config)
        for ext in md_loader.supported_extensions:
            self._loaders[ext] = md_loader

        txt_loader = TextLoader(config)
        for ext in txt_loader.supported_extensions:
            self._loaders[ext] = txt_loader

        # Custom loaders override defaults
        if loaders:
            self._loaders.update(loaders)

    @property
    def supported_extensions(self) -> list[str]:
        return self.extensions

    async def load(self, path: str | Path) -> list[Document]:
        """Load all documents from a directory.

        Args:
            path: Path to the directory

        Returns:
            List of Document objects from all files
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        if not path.is_dir():
            # If it's a file, load it directly
            return await self._load_file(path)

        documents = []
        files = list(self._walk_directory(path))

        logger.info(f"Loading {len(files)} files from {path}")

        for file_path in files:
            try:
                docs = await self._load_file(file_path)
                documents.extend(docs)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Loaded {len(documents)} documents from {len(files)} files")
        return documents

    async def _load_file(self, path: Path) -> list[Document]:
        """Load a single file.

        Args:
            path: Path to the file

        Returns:
            List of Document objects
        """
        ext = path.suffix.lower()
        loader = self._loaders.get(ext)

        if not loader:
            logger.debug(f"No loader for extension {ext}, skipping {path}")
            return []

        return await loader.load(path)

    def _walk_directory(self, root: Path):
        """Walk directory and yield matching files.

        Args:
            root: Root directory

        Yields:
            Path objects for matching files
        """
        for item in root.iterdir():
            # Check exclusions
            if self._should_exclude(item):
                continue

            if item.is_dir():
                yield from self._walk_directory(item)
            elif item.is_file():
                if item.suffix.lower() in self.extensions:
                    yield item

    def _should_exclude(self, path: Path) -> bool:
        """Check if a path should be excluded.

        Args:
            path: Path to check

        Returns:
            True if path matches exclusion pattern
        """
        name = path.name

        # Check direct matches
        if name in self.exclude_patterns:
            return True

        # Check pattern matches
        for pattern in self.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True

        return False

    def add_loader(self, extension: str, loader: DocumentLoader) -> None:
        """Add a custom loader for an extension.

        Args:
            extension: File extension (e.g., ".json")
            loader: Loader to use for this extension
        """
        if not extension.startswith("."):
            extension = f".{extension}"
        self._loaders[extension] = loader
        if extension not in self.extensions:
            self.extensions.append(extension)


# =============================================================================
# Convenience Functions
# =============================================================================


async def load_directory(
    path: str | Path,
    extensions: list[str] | None = None,
    chunk_size: int = 0,
) -> list[Document]:
    """Convenience function to load documents from a directory.

    Args:
        path: Path to directory
        extensions: File extensions to include
        chunk_size: Chunk size (0 for no chunking)

    Returns:
        List of Document objects
    """
    config = LoaderConfig(chunk_size=chunk_size)
    loader = DirectoryLoader(config=config, extensions=extensions)
    return await loader.load(path)


async def ingest_to_memory(
    path: str | Path,
    provider,  # Neo4jMemoryProvider
    *,
    extensions: list[str] | None = None,
    user_id: str | None = None,
    agent_id: str | None = None,
    batch_size: int = 10,
) -> int:
    """Convenience function to ingest documents directly into memory.

    Args:
        path: Path to file or directory
        provider: Memory provider to use
        extensions: File extensions to include
        user_id: User ID for memories
        agent_id: Agent ID for memories
        batch_size: Number of documents to process in parallel

    Returns:
        Number of documents ingested
    """
    # Load documents
    if Path(path).is_dir():
        loader = DirectoryLoader(extensions=extensions)
    else:
        loader = DirectoryLoader(extensions=[Path(path).suffix])

    documents = await loader.load(path)

    # Store in memory
    count = 0
    for doc in documents:
        try:
            await provider.store(
                content=doc.content,
                memory_type=doc.memory_type,
                scope=doc.scope,
                user_id=user_id,
                agent_id=agent_id,
                metadata={
                    "source_path": doc.source_path,
                    "title": doc.title,
                    "section": doc.section,
                    "file_type": doc.file_type,
                    **doc.metadata,
                },
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to store document from {doc.source_path}: {e}")

    logger.info(f"Ingested {count}/{len(documents)} documents into memory")
    return count
