"""Document Ingestion Orchestrator - End-to-end document processing pipeline.

This module connects:
1. Document Loaders (Markdown, Text, Code, Config)
2. Phase 0/1 Decomposition (semantic extraction)
3. Memory Integration (entity/fact/relationship storage)
4. Cross-Document Linking (shared entity resolution)

The result is an interconnected semantic web where entities mentioned across
multiple documents are linked together.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    DocumentIngestionOrchestrator                         │
    │                                                                          │
    │  ┌─────────────┐    ┌─────────────────┐    ┌────────────────────────┐  │
    │  │   Loaders   │───>│  Decomposition  │───>│  Memory Integration    │  │
    │  │ (md,py,yaml)│    │  (Phase 0/1)    │    │  (entities, facts)     │  │
    │  └─────────────┘    └─────────────────┘    └────────────────────────┘  │
    │         │                   │                         │                  │
    │         │                   │                         ▼                  │
    │         │                   │              ┌────────────────────────┐   │
    │         │                   └─────────────>│  Entity Resolution     │   │
    │         │                                  │  (cross-doc linking)   │   │
    │         │                                  └────────────────────────┘   │
    │         │                                              │                 │
    │         ▼                                              ▼                 │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                    Semantic Knowledge Graph                      │   │
    │  │              (interconnected entities/facts/rels)                │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.orchestration import DocumentIngestionOrchestrator

    orchestrator = DocumentIngestionOrchestrator(
        llm=my_llm,
        semantic_memory=my_semantic,
    )

    # Ingest a single file
    result = await orchestrator.ingest_file("docs/CLAUDE.md")

    # Ingest a project directory
    result = await orchestrator.ingest_directory(
        "my-project/",
        patterns=["*.md", "*.py", "*.yaml"],
    )

    # Check interconnections
    print(f"Created {result.total_entities} entities")
    print(f"Cross-document links: {result.cross_doc_links}")
"""

from __future__ import annotations

import ast
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class LLMProvider(Protocol):
    """LLM provider protocol."""

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        ...


class SemanticMemoryProvider(Protocol):
    """Semantic memory provider protocol."""

    async def create_entity(
        self,
        name: str,
        entity_type: str,
        *,
        scope_id: str = "agent:default",
        aliases: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...

    async def resolve_entity(
        self,
        name: str,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> list[Any]:
        ...

    async def add_fact(
        self,
        content: str,
        *,
        scope_id: str = "agent:default",
        subject_entity_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> Any:
        ...

    async def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        *,
        scope_id: str = "agent:default",
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> Any | None:
        ...


# =============================================================================
# Configuration
# =============================================================================


class ContentType(str, Enum):
    """Type of content being processed."""

    MARKDOWN = "markdown"
    TEXT = "text"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    JAVASCRIPT = "javascript"
    YAML = "yaml"
    JSON = "json"
    TOML = "toml"
    UNKNOWN = "unknown"


@dataclass
class IngestionConfig:
    """Configuration for document ingestion."""

    # File patterns to include
    include_patterns: list[str] = field(
        default_factory=lambda: ["*.md", "*.txt", "*.py", "*.ts", "*.js", "*.yaml", "*.yml", "*.json", "*.toml"]
    )

    # Patterns to exclude
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "node_modules",
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            ".mypy_cache",
            "*.pyc",
            ".DS_Store",
            "package-lock.json",
            "yarn.lock",
        ]
    )

    # Decomposition settings
    decompose_markdown: bool = True
    decompose_code: bool = True  # Extract docstrings, comments, function names
    decompose_config: bool = True  # Extract key-value pairs as facts

    # Entity resolution
    cross_document_linking: bool = True
    entity_match_threshold: float = 0.85

    # Chunking for large files
    max_chunk_size: int = 4000  # Characters
    chunk_overlap: int = 200

    # Processing limits
    max_files: int = 1000
    max_file_size_kb: int = 500  # Skip files larger than this

    # Scope for stored items
    default_scope_id: str = "agent:default"


@dataclass
class DocumentChunk:
    """A chunk of document content for processing."""

    content: str
    source_path: str
    content_type: ContentType
    chunk_index: int = 0
    total_chunks: int = 1
    section: str | None = None  # e.g., "# Installation > ## Requirements"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedKnowledge:
    """Knowledge extracted from a document chunk."""

    source_path: str
    chunk_index: int

    # Extracted items
    entities: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    relationships: list[dict[str, Any]] = field(default_factory=list)

    # Code-specific
    functions: list[dict[str, Any]] = field(default_factory=list)
    classes: list[dict[str, Any]] = field(default_factory=list)
    imports: list[str] = field(default_factory=list)

    # Config-specific
    config_entries: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class IngestionResult:
    """Result of ingesting documents."""

    # Files processed
    files_processed: int = 0
    files_skipped: int = 0
    files_failed: int = 0

    # Knowledge created
    total_entities: int = 0
    total_facts: int = 0
    total_relationships: int = 0

    # Cross-document linking
    entities_merged: int = 0  # Linked to existing
    cross_doc_links: int = 0  # New cross-doc relationships

    # Entity mapping for the session
    entity_mapping: dict[str, str] = field(default_factory=dict)  # canonical -> entity_id

    # Timing
    processing_time_ms: float = 0.0

    # Errors
    errors: list[str] = field(default_factory=list)


# =============================================================================
# Content Extractors
# =============================================================================


class PythonExtractor:
    """Extracts semantic knowledge from Python source code.

    Uses AST parsing to extract:
    - Class names and docstrings
    - Function names, docstrings, and parameters
    - Import statements
    - Module-level docstrings
    """

    def extract(self, content: str, source_path: str) -> ExtractedKnowledge:
        """Extract knowledge from Python code."""
        result = ExtractedKnowledge(source_path=source_path, chunk_index=0)

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python file {source_path}: {e}")
            return result

        # Extract module docstring
        if ast.get_docstring(tree):
            result.facts.append({
                "content": f"Module {Path(source_path).stem} docstring: {ast.get_docstring(tree)[:200]}",
                "subject": Path(source_path).stem,
                "predicate": "has_docstring",
                "confidence": 1.0,
            })

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                result.classes.append(self._extract_class(node, source_path))
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                # Only top-level functions
                if not isinstance(node, ast.FunctionDef) or node.col_offset == 0:
                    result.functions.append(self._extract_function(node, source_path))
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    result.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    result.imports.append(node.module)

        # Create entities from classes and functions
        for cls in result.classes:
            result.entities.append({
                "text": cls["name"],
                "canonical_name": cls["name"],
                "entity_type": "class",
                "properties": {
                    "source_file": source_path,
                    "docstring": cls.get("docstring", "")[:200],
                    "methods": cls.get("methods", []),
                },
            })

        for func in result.functions:
            result.entities.append({
                "text": func["name"],
                "canonical_name": func["name"],
                "entity_type": "function",
                "properties": {
                    "source_file": source_path,
                    "docstring": func.get("docstring", "")[:200],
                    "parameters": func.get("parameters", []),
                },
            })

        return result

    def _extract_class(self, node: ast.ClassDef, source_path: str) -> dict[str, Any]:
        """Extract class information."""
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                methods.append(item.name)

        return {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "methods": methods,
            "base_classes": [
                ast.unparse(base) if hasattr(ast, "unparse") else str(base)
                for base in node.bases
            ],
            "source_file": source_path,
            "line_number": node.lineno,
        }

    def _extract_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, source_path: str
    ) -> dict[str, Any]:
        """Extract function information."""
        params = []
        for arg in node.args.args:
            param = {"name": arg.arg}
            if arg.annotation:
                param["type"] = (
                    ast.unparse(arg.annotation)
                    if hasattr(ast, "unparse")
                    else str(arg.annotation)
                )
            params.append(param)

        return {
            "name": node.name,
            "docstring": ast.get_docstring(node) or "",
            "parameters": params,
            "is_async": isinstance(node, ast.AsyncFunctionDef),
            "source_file": source_path,
            "line_number": node.lineno,
        }


class ConfigExtractor:
    """Extracts semantic knowledge from config files (YAML, JSON, TOML)."""

    def extract(
        self, content: str, source_path: str, content_type: ContentType
    ) -> ExtractedKnowledge:
        """Extract knowledge from config content."""
        result = ExtractedKnowledge(source_path=source_path, chunk_index=0)

        try:
            if content_type == ContentType.YAML:
                data = self._parse_yaml(content)
            elif content_type == ContentType.JSON:
                data = json.loads(content)
            elif content_type == ContentType.TOML:
                data = self._parse_toml(content)
            else:
                return result

            # Flatten config to key-value pairs
            entries = self._flatten_config(data)
            result.config_entries = entries

            # Create facts from config entries
            for entry in entries:
                result.facts.append({
                    "content": f"{entry['key']}: {entry['value']}",
                    "subject": Path(source_path).name,
                    "predicate": "has_config",
                    "object_value": str(entry["value"]),
                    "confidence": 1.0,
                    "metadata": {"config_key": entry["key"]},
                })

        except Exception as e:
            logger.warning(f"Failed to parse config file {source_path}: {e}")

        return result

    def _parse_yaml(self, content: str) -> dict[str, Any]:
        """Parse YAML content."""
        try:
            import yaml

            return yaml.safe_load(content) or {}
        except ImportError:
            logger.warning("PyYAML not installed, using basic parsing")
            return {}

    def _parse_toml(self, content: str) -> dict[str, Any]:
        """Parse TOML content."""
        try:
            import tomllib

            return tomllib.loads(content)
        except ImportError:
            try:
                import tomli

                return tomli.loads(content)
            except ImportError:
                logger.warning("tomllib/tomli not available, skipping TOML parsing")
                return {}

    def _flatten_config(
        self, data: dict[str, Any], prefix: str = ""
    ) -> list[dict[str, Any]]:
        """Flatten nested config to list of key-value pairs."""
        entries = []

        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                entries.extend(self._flatten_config(value, full_key))
            elif isinstance(value, list):
                # Store list as single entry
                entries.append({
                    "key": full_key,
                    "value": value,
                    "type": "list",
                })
            else:
                entries.append({
                    "key": full_key,
                    "value": value,
                    "type": type(value).__name__,
                })

        return entries


class MarkdownExtractor:
    """Extracts semantic knowledge from Markdown documents using LLM."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def extract(self, content: str, source_path: str) -> ExtractedKnowledge:
        """Extract knowledge from markdown using LLM decomposition."""
        result = ExtractedKnowledge(source_path=source_path, chunk_index=0)

        prompt = f"""Extract semantic knowledge from this markdown document.

DOCUMENT (from {source_path}):
{content[:4000]}

Output XML with entities, facts, and relationships found:

<knowledge>
  <entities>
    <entity type="concept|person|tool|system">Name</entity>
  </entities>
  <facts>
    <fact subject="Entity" predicate="relationship" object="value">Full fact statement</fact>
  </facts>
  <relationships>
    <rel source="Entity1" type="uses|depends_on|is_part_of|etc" target="Entity2"/>
  </relationships>
</knowledge>

Focus on:
- Key concepts, tools, and systems mentioned
- Factual statements about how things work
- Relationships between components
- Configuration patterns and conventions"""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )

            # Parse entities
            entity_pattern = r'<entity type="([^"]+)">([^<]+)</entity>'
            for match in re.finditer(entity_pattern, response):
                etype, name = match.groups()
                result.entities.append({
                    "text": name.strip(),
                    "canonical_name": name.strip(),
                    "entity_type": etype,
                })

            # Parse facts
            fact_pattern = r'<fact subject="([^"]+)" predicate="([^"]+)" object="([^"]+)">([^<]+)</fact>'
            for match in re.finditer(fact_pattern, response):
                subject, predicate, obj, content = match.groups()
                result.facts.append({
                    "content": content.strip(),
                    "subject": subject.strip(),
                    "predicate": predicate.strip(),
                    "object_value": obj.strip(),
                    "confidence": 0.9,
                })

            # Parse relationships
            rel_pattern = r'<rel source="([^"]+)" type="([^"]+)" target="([^"]+)"/>'
            for match in re.finditer(rel_pattern, response):
                source, rel_type, target = match.groups()
                result.relationships.append({
                    "source": source.strip(),
                    "target": target.strip(),
                    "type": rel_type.strip(),
                })

        except Exception as e:
            logger.warning(f"LLM extraction failed for {source_path}: {e}")

        return result


# =============================================================================
# Document Ingestion Orchestrator
# =============================================================================


class DocumentIngestionOrchestrator:
    """Orchestrates end-to-end document ingestion into semantic memory.

    This is the main entry point for processing documents and storing
    them as interconnected semantic knowledge.

    Features:
    - Multi-format support (Markdown, Python, YAML, JSON, etc.)
    - Phase 0/1 semantic decomposition for prose
    - AST-based extraction for code
    - Cross-document entity resolution for interconnection
    - Chunking for large files
    """

    def __init__(
        self,
        llm: LLMProvider,
        semantic_memory: SemanticMemoryProvider,
        config: IngestionConfig | None = None,
    ):
        """Initialize the orchestrator.

        Args:
            llm: LLM provider for semantic extraction
            semantic_memory: Memory layer for storage
            config: Ingestion configuration
        """
        self.llm = llm
        self.semantic = semantic_memory
        self.config = config or IngestionConfig()

        # Extractors
        self._python_extractor = PythonExtractor()
        self._config_extractor = ConfigExtractor()
        self._markdown_extractor = MarkdownExtractor(llm)

        # Session state for cross-document linking
        self._entity_cache: dict[str, str] = {}  # canonical -> entity_id

    async def ingest_file(
        self,
        path: str | Path,
        scope_id: str | None = None,
    ) -> IngestionResult:
        """Ingest a single file.

        Args:
            path: Path to the file
            scope_id: Scope for stored items

        Returns:
            IngestionResult with statistics
        """
        path = Path(path)
        scope_id = scope_id or self.config.default_scope_id

        result = IngestionResult()
        start_time = time.perf_counter()

        if not path.exists():
            result.errors.append(f"File not found: {path}")
            result.files_failed = 1
            return result

        try:
            await self._process_file(path, scope_id, result)
            result.files_processed = 1
        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            result.errors.append(f"{path}: {e}")
            result.files_failed = 1

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.entity_mapping = dict(self._entity_cache)

        return result

    async def ingest_directory(
        self,
        path: str | Path,
        scope_id: str | None = None,
        patterns: list[str] | None = None,
    ) -> IngestionResult:
        """Ingest all matching files from a directory.

        Args:
            path: Path to the directory
            scope_id: Scope for stored items
            patterns: Override include patterns

        Returns:
            IngestionResult with statistics
        """
        path = Path(path)
        scope_id = scope_id or self.config.default_scope_id
        patterns = patterns or self.config.include_patterns

        result = IngestionResult()
        start_time = time.perf_counter()

        if not path.exists():
            result.errors.append(f"Directory not found: {path}")
            return result

        if not path.is_dir():
            # Single file
            return await self.ingest_file(path, scope_id)

        # Walk directory and collect files
        files = list(self._walk_directory(path, patterns))
        logger.info(f"Found {len(files)} files to process in {path}")

        # Process files
        for file_path in files[: self.config.max_files]:
            try:
                await self._process_file(file_path, scope_id, result)
                result.files_processed += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                result.errors.append(f"{file_path}: {e}")
                result.files_failed += 1

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        result.entity_mapping = dict(self._entity_cache)

        logger.info(
            f"Ingested {result.files_processed} files: "
            f"{result.total_entities} entities, "
            f"{result.total_facts} facts, "
            f"{result.total_relationships} relationships"
        )

        return result

    async def _process_file(
        self,
        path: Path,
        scope_id: str,
        result: IngestionResult,
    ) -> None:
        """Process a single file."""
        # Check file size
        if path.stat().st_size > self.config.max_file_size_kb * 1024:
            logger.warning(f"Skipping large file: {path}")
            result.files_skipped += 1
            return

        # Determine content type
        content_type = self._get_content_type(path)
        if content_type == ContentType.UNKNOWN:
            logger.debug(f"Unknown content type, skipping: {path}")
            result.files_skipped += 1
            return

        # Read content
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            result.files_skipped += 1
            return

        # Extract knowledge based on content type
        if content_type == ContentType.PYTHON:
            knowledge = self._python_extractor.extract(content, str(path))
        elif content_type in (ContentType.YAML, ContentType.JSON, ContentType.TOML):
            knowledge = self._config_extractor.extract(content, str(path), content_type)
        elif content_type == ContentType.MARKDOWN:
            knowledge = await self._markdown_extractor.extract(content, str(path))
        else:
            # Text files - treat as markdown
            knowledge = await self._markdown_extractor.extract(content, str(path))

        # Store extracted knowledge
        await self._store_knowledge(knowledge, scope_id, result)

    async def _store_knowledge(
        self,
        knowledge: ExtractedKnowledge,
        scope_id: str,
        result: IngestionResult,
    ) -> None:
        """Store extracted knowledge in semantic memory."""
        # Process entities first (for linking)
        for entity in knowledge.entities:
            entity_id = await self._resolve_or_create_entity(
                name=entity["canonical_name"],
                entity_type=entity.get("entity_type", "concept"),
                properties=entity.get("properties", {}),
                scope_id=scope_id,
                result=result,
            )

            # Cache for cross-document linking
            self._entity_cache[entity["canonical_name"].lower()] = entity_id

        # Process facts
        for fact in knowledge.facts:
            # Try to link to subject entity
            subject_id = self._entity_cache.get(fact.get("subject", "").lower())

            await self.semantic.add_fact(
                content=fact["content"],
                scope_id=scope_id,
                subject_entity_id=subject_id,
                confidence=fact.get("confidence", 1.0),
                metadata={
                    "source_file": knowledge.source_path,
                    "predicate": fact.get("predicate", ""),
                    "object_value": fact.get("object_value", ""),
                },
            )
            result.total_facts += 1

        # Process relationships
        for rel in knowledge.relationships:
            source_id = self._entity_cache.get(rel["source"].lower())
            target_id = self._entity_cache.get(rel["target"].lower())

            if source_id and target_id:
                stored = await self.semantic.add_relationship(
                    source_entity_id=source_id,
                    target_entity_id=target_id,
                    relationship_type=rel["type"],
                    scope_id=scope_id,
                    properties={"source_file": knowledge.source_path},
                )
                if stored:
                    result.total_relationships += 1

                    # Track cross-document links
                    # (relationship between entities from different files)
                    result.cross_doc_links += 1

    async def _resolve_or_create_entity(
        self,
        name: str,
        entity_type: str,
        properties: dict[str, Any],
        scope_id: str,
        result: IngestionResult,
    ) -> str:
        """Resolve to existing entity or create new one."""
        # Check local cache first
        cached_id = self._entity_cache.get(name.lower())
        if cached_id:
            result.entities_merged += 1
            return cached_id

        # Try to resolve in memory (cross-document linking)
        if self.config.cross_document_linking:
            try:
                matches = await self.semantic.resolve_entity(
                    name,
                    min_score=self.config.entity_match_threshold,
                )
                if matches:
                    existing_id = matches[0].entity.node_id
                    self._entity_cache[name.lower()] = existing_id
                    result.entities_merged += 1
                    return existing_id
            except Exception as e:
                logger.debug(f"Entity resolution failed for '{name}': {e}")

        # Create new entity
        new_entity = await self.semantic.create_entity(
            name=name,
            entity_type=entity_type,
            scope_id=scope_id,
            properties=properties,
        )

        entity_id = new_entity.node_id
        self._entity_cache[name.lower()] = entity_id
        result.total_entities += 1

        return entity_id

    def _walk_directory(self, root: Path, patterns: list[str]):
        """Walk directory yielding matching files."""
        for item in root.iterdir():
            # Check exclusions
            if self._should_exclude(item):
                continue

            if item.is_dir():
                yield from self._walk_directory(item, patterns)
            elif item.is_file():
                if self._matches_pattern(item, patterns):
                    yield item

    def _should_exclude(self, path: Path) -> bool:
        """Check if path should be excluded."""
        name = path.name

        for pattern in self.config.exclude_patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            elif pattern.endswith("*"):
                if name.startswith(pattern[:-1]):
                    return True
            elif name == pattern:
                return True

        return False

    def _matches_pattern(self, path: Path, patterns: list[str]) -> bool:
        """Check if file matches any include pattern."""
        name = path.name

        for pattern in patterns:
            if pattern.startswith("*"):
                if name.endswith(pattern[1:]):
                    return True
            else:
                if name.endswith(pattern.lstrip("*")):
                    return True

        return False

    def _get_content_type(self, path: Path) -> ContentType:
        """Determine content type from file extension."""
        ext = path.suffix.lower()

        mapping = {
            ".md": ContentType.MARKDOWN,
            ".markdown": ContentType.MARKDOWN,
            ".txt": ContentType.TEXT,
            ".py": ContentType.PYTHON,
            ".ts": ContentType.TYPESCRIPT,
            ".tsx": ContentType.TYPESCRIPT,
            ".js": ContentType.JAVASCRIPT,
            ".jsx": ContentType.JAVASCRIPT,
            ".yaml": ContentType.YAML,
            ".yml": ContentType.YAML,
            ".json": ContentType.JSON,
            ".toml": ContentType.TOML,
        }

        return mapping.get(ext, ContentType.UNKNOWN)

    def clear_cache(self) -> None:
        """Clear the entity cache between sessions."""
        self._entity_cache.clear()


# =============================================================================
# Convenience Functions
# =============================================================================


async def ingest_project(
    path: str | Path,
    llm: LLMProvider,
    semantic_memory: SemanticMemoryProvider,
    patterns: list[str] | None = None,
    scope_id: str = "agent:default",
) -> IngestionResult:
    """Convenience function to ingest a project directory.

    Args:
        path: Path to project root
        llm: LLM provider
        semantic_memory: Semantic memory layer
        patterns: File patterns to include
        scope_id: Scope for stored items

    Returns:
        IngestionResult with statistics
    """
    orchestrator = DocumentIngestionOrchestrator(
        llm=llm,
        semantic_memory=semantic_memory,
    )

    return await orchestrator.ingest_directory(
        path,
        scope_id=scope_id,
        patterns=patterns,
    )


async def ingest_claude_md(
    path: str | Path,
    llm: LLMProvider,
    semantic_memory: SemanticMemoryProvider,
    scope_id: str = "agent:default",
) -> IngestionResult:
    """Convenience function to ingest a CLAUDE.md file.

    This is optimized for processing project instruction files.

    Args:
        path: Path to CLAUDE.md
        llm: LLM provider
        semantic_memory: Semantic memory layer
        scope_id: Scope for stored items

    Returns:
        IngestionResult with statistics
    """
    orchestrator = DocumentIngestionOrchestrator(
        llm=llm,
        semantic_memory=semantic_memory,
    )

    return await orchestrator.ingest_file(path, scope_id)
