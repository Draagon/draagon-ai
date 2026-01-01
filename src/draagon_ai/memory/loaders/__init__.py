"""Document loaders for draagon-ai memory system.

Document loaders handle ingestion of various file formats into the
semantic graph memory system. All content flows through Phase 0/1
extraction pipeline before being stored in Neo4j.

Available loaders:
- DocumentLoader: Base class for all loaders
- MarkdownLoader: Markdown files (.md)
- TextLoader: Plain text files (.txt)
- DirectoryLoader: Recursively load from directories

Example:
    from draagon_ai.memory.loaders import MarkdownLoader, DirectoryLoader
    from draagon_ai.memory.providers.neo4j import Neo4jMemoryProvider

    # Load single file
    loader = MarkdownLoader()
    documents = await loader.load("path/to/file.md")

    # Load directory
    dir_loader = DirectoryLoader(extensions=[".md", ".txt"])
    documents = await dir_loader.load("path/to/docs")

    # Ingest into memory
    provider = Neo4jMemoryProvider(config, embedder, llm)
    for doc in documents:
        await provider.store(
            content=doc.content,
            memory_type=doc.memory_type,
            scope=doc.scope,
            metadata=doc.metadata,
        )
"""

from .base import (
    Document,
    DocumentLoader,
    LoaderConfig,
)
from .markdown import MarkdownLoader
from .text import TextLoader
from .directory import DirectoryLoader

__all__ = [
    "Document",
    "DocumentLoader",
    "LoaderConfig",
    "MarkdownLoader",
    "TextLoader",
    "DirectoryLoader",
]
