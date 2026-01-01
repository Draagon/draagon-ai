"""Plain text document loader.

Loads plain text files with optional paragraph-based chunking.
"""

from pathlib import Path
from typing import Any

from .base import Document, DocumentLoader, LoaderConfig
from draagon_ai.memory.base import MemoryType


class TextLoader(DocumentLoader):
    """Loader for plain text files.

    Features:
    - Paragraph-based chunking
    - Configurable chunk size
    - Simple and fast

    Example:
        loader = TextLoader()
        docs = await loader.load("notes.txt")
    """

    @property
    def supported_extensions(self) -> list[str]:
        return [".txt", ".text"]

    async def load(self, path: str | Path) -> list[Document]:
        """Load a text file.

        Args:
            path: Path to the text file

        Returns:
            List of Document objects
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        content = path.read_text(encoding="utf-8")
        file_meta = self._get_file_metadata(path)

        documents = []

        if self.config.chunk_size > 0:
            # Chunk by size
            chunks = self._chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
            for chunk_text, idx in chunks:
                doc = Document(
                    content=chunk_text,
                    source_path=str(path),
                    memory_type=self._infer_memory_type(chunk_text, {}),
                    scope=self.config.default_scope,
                    title=path.stem,
                    file_type="text",
                    chunk_index=idx,
                    total_chunks=len(chunks),
                    **file_meta,
                )
                documents.append(doc)
        elif self.config.preserve_structure:
            # Split by paragraphs
            paragraphs = self._split_paragraphs(content)
            for i, para in enumerate(paragraphs):
                if not para.strip():
                    continue
                doc = Document(
                    content=para,
                    source_path=str(path),
                    memory_type=self._infer_memory_type(para, {}),
                    scope=self.config.default_scope,
                    title=path.stem,
                    file_type="text",
                    chunk_index=i,
                    total_chunks=len(paragraphs),
                    **file_meta,
                )
                documents.append(doc)
        else:
            # Single document
            doc = Document(
                content=content,
                source_path=str(path),
                memory_type=self._infer_memory_type(content, {}),
                scope=self.config.default_scope,
                title=path.stem,
                file_type="text",
                **file_meta,
            )
            documents.append(doc)

        return documents

    def _split_paragraphs(self, content: str) -> list[str]:
        """Split text into paragraphs.

        Args:
            content: Full text content

        Returns:
            List of paragraphs
        """
        # Split on double newlines
        paragraphs = content.split("\n\n")

        # Clean up
        result = []
        for para in paragraphs:
            cleaned = para.strip()
            if cleaned:
                result.append(cleaned)

        return result
