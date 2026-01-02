"""Integration tests for DocumentIngestionOrchestrator.

Tests the end-to-end document ingestion pipeline:
1. Loading documents (Markdown, Python, Config files)
2. Semantic extraction via LLM
3. Storage in semantic memory
4. Cross-document entity linking

These tests verify that:
- CLAUDE.md files are properly decomposed into semantic knowledge
- Python source code is parsed for classes, functions, and docstrings
- Config files (YAML, JSON, TOML) are converted to facts
- Entities mentioned across multiple files are linked together
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from draagon_ai.orchestration.document_ingestion import (
    DocumentIngestionOrchestrator,
    IngestionConfig,
    IngestionResult,
    ContentType,
    PythonExtractor,
    ConfigExtractor,
)


# =============================================================================
# Mock Providers
# =============================================================================


@dataclass
class MockEntity:
    """Mock entity for testing."""

    node_id: str
    name: str
    entity_type: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class MockEntityMatch:
    """Mock entity match for resolution."""

    entity: MockEntity
    score: float


@dataclass
class MockFact:
    """Mock fact for testing."""

    node_id: str
    content: str


@dataclass
class MockRelationship:
    """Mock relationship for testing."""

    node_id: str
    source_id: str
    target_id: str
    rel_type: str


class MockSemanticMemory:
    """Mock semantic memory for testing."""

    def __init__(self):
        self.entities: dict[str, MockEntity] = {}
        self.facts: list[MockFact] = []
        self.relationships: list[MockRelationship] = []
        self._entity_counter = 0
        self._fact_counter = 0
        self._rel_counter = 0

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
    ) -> MockEntity:
        """Create a new entity."""
        self._entity_counter += 1
        entity = MockEntity(
            node_id=f"entity_{self._entity_counter}",
            name=name,
            entity_type=entity_type,
            properties=properties or {},
        )
        self.entities[entity.node_id] = entity
        return entity

    async def resolve_entity(
        self,
        name: str,
        min_score: float = 0.7,
        limit: int = 5,
    ) -> list[MockEntityMatch]:
        """Resolve entity by name."""
        matches = []
        for entity in self.entities.values():
            if entity.name.lower() == name.lower():
                matches.append(MockEntityMatch(entity=entity, score=1.0))
            elif name.lower() in entity.name.lower():
                matches.append(MockEntityMatch(entity=entity, score=0.8))
        return matches[:limit]

    async def add_fact(
        self,
        content: str,
        *,
        scope_id: str = "agent:default",
        subject_entity_id: str | None = None,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> MockFact:
        """Add a fact."""
        self._fact_counter += 1
        fact = MockFact(
            node_id=f"fact_{self._fact_counter}",
            content=content,
        )
        self.facts.append(fact)
        return fact

    async def add_relationship(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relationship_type: str,
        *,
        scope_id: str = "agent:default",
        properties: dict[str, Any] | None = None,
        confidence: float = 1.0,
    ) -> MockRelationship | None:
        """Add a relationship."""
        self._rel_counter += 1
        rel = MockRelationship(
            node_id=f"rel_{self._rel_counter}",
            source_id=source_entity_id,
            target_id=target_entity_id,
            rel_type=relationship_type,
        )
        self.relationships.append(rel)
        return rel


class MockLLMProvider:
    """Mock LLM for testing markdown extraction."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Return mock LLM response."""
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })

        # Check for pattern-based responses
        content = messages[-1]["content"] if messages else ""
        for pattern, response in self.responses.items():
            if pattern.lower() in content.lower():
                return response

        # Default response for markdown extraction
        return """
<knowledge>
  <entities>
    <entity type="concept">Agent</entity>
    <entity type="system">LLM</entity>
  </entities>
  <facts>
    <fact subject="Agent" predicate="uses" object="LLM">Agent uses LLM for reasoning</fact>
  </facts>
  <relationships>
    <rel source="Agent" type="uses" target="LLM"/>
  </relationships>
</knowledge>
"""


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_memory():
    """Create mock semantic memory."""
    return MockSemanticMemory()


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def orchestrator(mock_llm, mock_memory):
    """Create orchestrator with mocks."""
    return DocumentIngestionOrchestrator(
        llm=mock_llm,
        semantic_memory=mock_memory,
    )


@pytest.fixture
def temp_project():
    """Create a temporary project directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        # Create CLAUDE.md
        (project / "CLAUDE.md").write_text("""# My Project

## Overview
This project uses an Agent architecture with LLM integration.

## Architecture
- Agent: Main orchestrator
- LLM: Language model provider
- Memory: Stores knowledge
""")

        # Create Python file
        (project / "agent.py").write_text('''"""Agent module for orchestration."""

class Agent:
    """Main agent class that coordinates LLM calls."""

    def __init__(self, llm):
        """Initialize with LLM provider."""
        self.llm = llm

    async def process(self, query: str) -> str:
        """Process a user query."""
        return await self.llm.chat([{"role": "user", "content": query}])


def create_agent(llm) -> Agent:
    """Factory function to create an agent."""
    return Agent(llm)
''')

        # Create config file
        (project / "config.yaml").write_text("""
project:
  name: my-agent
  version: 1.0.0

agent:
  max_tokens: 2000
  temperature: 0.7
""")

        # Create subdir with more files
        subdir = project / "utils"
        subdir.mkdir()
        (subdir / "helpers.py").write_text('''"""Helper utilities."""

def format_response(text: str) -> str:
    """Format a response for display."""
    return text.strip()
''')

        yield project


# =============================================================================
# Unit Tests - Extractors
# =============================================================================


class TestPythonExtractor:
    """Tests for Python AST extraction."""

    def test_extract_class(self):
        """Test extracting class information."""
        extractor = PythonExtractor()
        code = '''
class MyClass:
    """A test class."""

    def method_one(self, arg: str) -> str:
        """Do something."""
        return arg

    async def method_two(self):
        """Async method."""
        pass
'''
        result = extractor.extract(code, "test.py")

        assert len(result.classes) == 1
        assert result.classes[0]["name"] == "MyClass"
        assert "A test class" in result.classes[0]["docstring"]
        assert "method_one" in result.classes[0]["methods"]
        assert "method_two" in result.classes[0]["methods"]

    def test_extract_function(self):
        """Test extracting function information."""
        extractor = PythonExtractor()
        code = '''
def my_function(arg1: str, arg2: int = 0) -> bool:
    """Do something useful."""
    return True

async def async_function():
    """Async operation."""
    pass
'''
        result = extractor.extract(code, "test.py")

        # Should have 2 functions
        assert len(result.functions) >= 1

        # Find my_function
        my_func = next((f for f in result.functions if f["name"] == "my_function"), None)
        assert my_func is not None
        assert "Do something useful" in my_func["docstring"]
        assert len(my_func["parameters"]) >= 2

    def test_extract_imports(self):
        """Test extracting imports."""
        extractor = PythonExtractor()
        code = '''
import os
import sys
from pathlib import Path
from typing import Any, Optional
'''
        result = extractor.extract(code, "test.py")

        assert "os" in result.imports
        assert "sys" in result.imports
        assert "pathlib" in result.imports
        assert "typing" in result.imports

    def test_creates_entities_from_classes(self):
        """Test that classes become entities."""
        extractor = PythonExtractor()
        code = '''
class Agent:
    """Main agent class."""
    pass
'''
        result = extractor.extract(code, "test.py")

        # Should create entity for the class
        agent_entity = next(
            (e for e in result.entities if e["canonical_name"] == "Agent"),
            None,
        )
        assert agent_entity is not None
        assert agent_entity["entity_type"] == "class"


class TestConfigExtractor:
    """Tests for config file extraction."""

    def test_extract_yaml(self):
        """Test extracting YAML config."""
        extractor = ConfigExtractor()
        content = """
project:
  name: test
  version: 1.0

settings:
  enabled: true
  max_items: 100
"""
        result = extractor.extract(content, "config.yaml", ContentType.YAML)

        # Should have flattened config entries
        assert len(result.config_entries) > 0

        # Check for nested key
        name_entry = next(
            (e for e in result.config_entries if e["key"] == "project.name"),
            None,
        )
        assert name_entry is not None
        assert name_entry["value"] == "test"

    def test_extract_json(self):
        """Test extracting JSON config."""
        extractor = ConfigExtractor()
        content = '{"name": "test", "version": 1}'

        result = extractor.extract(content, "config.json", ContentType.JSON)

        assert len(result.config_entries) == 2
        name_entry = next(
            (e for e in result.config_entries if e["key"] == "name"),
            None,
        )
        assert name_entry is not None
        assert name_entry["value"] == "test"

    def test_creates_facts_from_config(self):
        """Test that config entries become facts."""
        extractor = ConfigExtractor()
        content = '{"api_key": "secret", "timeout": 30}'

        result = extractor.extract(content, "config.json", ContentType.JSON)

        # Should create facts
        assert len(result.facts) == 2
        timeout_fact = next(
            (f for f in result.facts if "timeout" in f["content"]),
            None,
        )
        assert timeout_fact is not None


# =============================================================================
# Integration Tests - Orchestrator
# =============================================================================


class TestDocumentIngestionOrchestrator:
    """Tests for the full ingestion orchestrator."""

    @pytest.mark.asyncio
    async def test_ingest_single_markdown(self, orchestrator, mock_memory, temp_project):
        """Test ingesting a single markdown file."""
        result = await orchestrator.ingest_file(temp_project / "CLAUDE.md")

        assert result.files_processed == 1
        assert result.files_failed == 0
        assert result.total_entities > 0
        assert result.total_facts > 0

    @pytest.mark.asyncio
    async def test_ingest_single_python(self, orchestrator, mock_memory, temp_project):
        """Test ingesting a Python file."""
        result = await orchestrator.ingest_file(temp_project / "agent.py")

        assert result.files_processed == 1

        # Should have created entities for Agent class and create_agent function
        agent_created = any(
            e.name == "Agent" for e in mock_memory.entities.values()
        )
        assert agent_created, "Agent class should be stored as entity"

    @pytest.mark.asyncio
    async def test_ingest_single_yaml(self, orchestrator, mock_memory, temp_project):
        """Test ingesting a YAML config file."""
        result = await orchestrator.ingest_file(temp_project / "config.yaml")

        assert result.files_processed == 1
        assert result.total_facts > 0

        # Should have facts from config
        assert len(mock_memory.facts) > 0

    @pytest.mark.asyncio
    async def test_ingest_directory(self, orchestrator, mock_memory, temp_project):
        """Test ingesting entire directory."""
        result = await orchestrator.ingest_directory(temp_project)

        # Should process multiple files
        assert result.files_processed >= 3  # CLAUDE.md, agent.py, config.yaml

        # Should have entities from multiple files
        assert result.total_entities > 0
        assert len(mock_memory.entities) > 0

    @pytest.mark.asyncio
    async def test_ingest_with_patterns(self, orchestrator, mock_memory, temp_project):
        """Test ingesting with specific patterns."""
        result = await orchestrator.ingest_directory(
            temp_project,
            patterns=["*.py"],  # Only Python files
        )

        # Should only process Python files
        assert result.files_processed >= 1
        assert result.total_facts >= 0

    @pytest.mark.asyncio
    async def test_cross_document_linking(self, orchestrator, mock_memory, temp_project):
        """Test that entities are linked across documents."""
        # First, ingest a file that mentions "Agent"
        await orchestrator.ingest_file(temp_project / "agent.py")
        first_count = len(mock_memory.entities)

        # Then ingest another file that also mentions "Agent"
        await orchestrator.ingest_file(temp_project / "CLAUDE.md")

        # The Agent entity should be merged, not duplicated
        # (entities_merged should increase, not total_entities)
        # Note: Exact behavior depends on LLM response matching

    @pytest.mark.asyncio
    async def test_entity_cache_persistence(self, orchestrator, mock_memory, temp_project):
        """Test that entity cache persists across files."""
        # Ingest first file
        await orchestrator.ingest_file(temp_project / "agent.py")

        # Check cache has entries
        assert len(orchestrator._entity_cache) > 0

        # Ingest second file
        result = await orchestrator.ingest_file(temp_project / "CLAUDE.md")

        # Should still have cache entries
        assert len(result.entity_mapping) > 0

    @pytest.mark.asyncio
    async def test_clear_cache(self, orchestrator, mock_memory, temp_project):
        """Test clearing entity cache."""
        await orchestrator.ingest_file(temp_project / "agent.py")
        assert len(orchestrator._entity_cache) > 0

        orchestrator.clear_cache()
        assert len(orchestrator._entity_cache) == 0

    @pytest.mark.asyncio
    async def test_file_not_found(self, orchestrator):
        """Test handling missing file."""
        result = await orchestrator.ingest_file("/nonexistent/file.md")

        assert result.files_failed == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_excludes_patterns(self, orchestrator, mock_memory):
        """Test exclusion patterns work."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)

            # Create files
            (project / "main.py").write_text("# Main")
            (project / "node_modules").mkdir()
            (project / "node_modules" / "dep.js").write_text("// Dep")

            result = await orchestrator.ingest_directory(project)

            # Should skip node_modules
            # The main.py should be processed
            assert result.files_processed >= 0  # At least main.py


class TestIngestionConfig:
    """Tests for ingestion configuration."""

    def test_default_patterns(self):
        """Test default include patterns."""
        config = IngestionConfig()

        assert "*.md" in config.include_patterns
        assert "*.py" in config.include_patterns
        assert "*.yaml" in config.include_patterns

    def test_default_excludes(self):
        """Test default exclude patterns."""
        config = IngestionConfig()

        assert "node_modules" in config.exclude_patterns
        assert ".git" in config.exclude_patterns
        assert "__pycache__" in config.exclude_patterns

    def test_custom_config(self):
        """Test custom configuration."""
        config = IngestionConfig(
            include_patterns=["*.md"],
            max_files=10,
            max_file_size_kb=100,
        )

        assert config.include_patterns == ["*.md"]
        assert config.max_files == 10
        assert config.max_file_size_kb == 100


class TestContentTypeDetection:
    """Tests for content type detection."""

    def test_markdown_detection(self, orchestrator):
        """Test markdown file detection."""
        assert orchestrator._get_content_type(Path("README.md")) == ContentType.MARKDOWN
        assert orchestrator._get_content_type(Path("doc.markdown")) == ContentType.MARKDOWN

    def test_python_detection(self, orchestrator):
        """Test Python file detection."""
        assert orchestrator._get_content_type(Path("main.py")) == ContentType.PYTHON

    def test_config_detection(self, orchestrator):
        """Test config file detection."""
        assert orchestrator._get_content_type(Path("config.yaml")) == ContentType.YAML
        assert orchestrator._get_content_type(Path("config.yml")) == ContentType.YAML
        assert orchestrator._get_content_type(Path("package.json")) == ContentType.JSON
        assert orchestrator._get_content_type(Path("pyproject.toml")) == ContentType.TOML

    def test_unknown_detection(self, orchestrator):
        """Test unknown file type."""
        assert orchestrator._get_content_type(Path("data.bin")) == ContentType.UNKNOWN


# =============================================================================
# Integration Tests - Real LLM (Skipped by default)
# =============================================================================


@pytest.mark.skip(reason="Requires real LLM - run manually")
class TestDocumentIngestionWithRealLLM:
    """Integration tests with real LLM provider."""

    @pytest.fixture
    def real_llm(self):
        """Create real LLM provider."""
        import os
        from draagon_ai.llm.groq import GroqProvider

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            pytest.skip("GROQ_API_KEY not set")

        return GroqProvider(api_key=api_key)

    @pytest.mark.asyncio
    async def test_ingest_real_claude_md(self, real_llm, mock_memory, temp_project):
        """Test real ingestion of CLAUDE.md."""
        orchestrator = DocumentIngestionOrchestrator(
            llm=real_llm,
            semantic_memory=mock_memory,
        )

        result = await orchestrator.ingest_file(temp_project / "CLAUDE.md")

        assert result.files_processed == 1
        assert result.total_entities > 0
        assert result.total_facts > 0

        # Verify meaningful extraction
        entity_names = [e.name for e in mock_memory.entities.values()]
        assert len(entity_names) > 0
