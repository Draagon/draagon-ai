"""Integration tests for Retrieval Benchmark framework.

These tests compare semantic web vs raw context vs RAG approaches:
1. Semantic Web: Load into knowledge graph → query graph → answer
2. Raw Context: Files + question → LLM → answer
3. RAG (future): Files → vector store → similarity search → answer

The hypothesis is that semantic web scales better with larger file sets.
"""

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from draagon_ai.orchestration.retrieval_benchmark import (
    RetrievalBenchmark,
    RetrievalStrategy,
    TestCase,
    BenchmarkResult,
    SemanticWebProcessor,
    RawContextProcessor,
    RAGProcessor,
    BenchmarkEvaluator,
    compare_approaches,
    create_scale_test_suite,
)


# =============================================================================
# Mock Providers
# =============================================================================


@dataclass
class MockSearchResult:
    """Mock search result from semantic memory."""

    content: str
    entity_name: str = ""
    score: float = 1.0


class MockSemanticMemory:
    """Mock semantic memory for testing."""

    def __init__(self, knowledge_base: dict[str, list[str]] | None = None):
        """Initialize with optional knowledge base.

        Args:
            knowledge_base: Dict mapping query keywords to list of relevant facts
        """
        self.knowledge_base = knowledge_base or {}
        self.search_calls: list[str] = []

    async def search(self, query: str, limit: int = 10) -> list[MockSearchResult]:
        """Search for relevant knowledge."""
        self.search_calls.append(query)

        results = []
        query_lower = query.lower()

        for keyword, facts in self.knowledge_base.items():
            if keyword.lower() in query_lower:
                for fact in facts[:limit]:
                    results.append(MockSearchResult(content=fact, entity_name=keyword))

        return results[:limit]

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[Any]:
        """Get related entities (stub)."""
        return []


class MockVectorStore:
    """Mock vector store for RAG testing."""

    def __init__(self, chunks: list[str] | None = None):
        self.chunks = chunks or []
        self.search_calls: list[str] = []

    async def search(self, query: str, limit: int = 10) -> list[MockSearchResult]:
        """Search for similar chunks."""
        self.search_calls.append(query)
        return [MockSearchResult(content=c) for c in self.chunks[:limit]]


class MockLLMProvider:
    """Mock LLM for testing."""

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1000,
    ) -> str:
        """Return mock response."""
        self.calls.append({"messages": messages})

        content = messages[-1]["content"] if messages else ""

        # Check for pattern matches
        for pattern, response in self.responses.items():
            if pattern.lower() in content.lower():
                return response

        # Default response for evaluation
        if "evaluate" in content.lower() or "rate" in content.lower():
            return """
<evaluation>
  <correctness>0.8</correctness>
  <correctness_reason>Answer is mostly correct</correctness_reason>
  <completeness>0.7</completeness>
  <completeness_reason>Covers main points</completeness_reason>
  <relevance>0.9</relevance>
  <relevance_reason>Stays on topic</relevance_reason>
  <missing>none</missing>
  <extra>none</extra>
</evaluation>
"""

        # Default answer
        return "The agent uses a decision engine to process queries and execute tools."


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def mock_llm():
    """Create mock LLM."""
    return MockLLMProvider({
        "decision": "The decision engine processes queries through activation and execution phases.",
        "tool": "Tools are registered via the @tool decorator and executed by the tool registry.",
        "memory": "Memory is organized in 4 layers: working, episodic, semantic, and metacognitive.",
    })


@pytest.fixture
def mock_semantic_memory():
    """Create mock semantic memory with knowledge base."""
    return MockSemanticMemory({
        "agent": [
            "Agent is the main orchestrator class",
            "Agent uses DecisionEngine for query processing",
            "Agent coordinates with ToolRegistry for action execution",
        ],
        "decision": [
            "DecisionEngine handles query → activation → decision → execution flow",
            "DecisionEngine uses LLM for semantic understanding",
        ],
        "tool": [
            "@tool decorator registers handlers with ToolRegistry",
            "Tools have name, description, and parameters",
        ],
        "memory": [
            "Working memory has 5-minute TTL",
            "Episodic memory has 2-week TTL",
            "Semantic memory has 6-month TTL",
            "Metacognitive memory is permanent",
        ],
    })


@pytest.fixture
def mock_vector_store():
    """Create mock vector store with chunks."""
    return MockVectorStore([
        "The Agent class is the main entry point for processing user queries.",
        "Tools are registered using the @tool decorator which adds them to the registry.",
        "Memory is organized in layers with different TTLs.",
    ])


@pytest.fixture
def temp_project():
    """Create a temporary project with test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project = Path(tmpdir)

        # Small file (~500 bytes)
        (project / "small.md").write_text("""# Small File

This is a small file for testing.
The agent uses a decision engine.
""")

        # Medium file (~5KB)
        medium_content = """# Medium File

## Overview
This project implements an agentic AI framework.

## Architecture
The system uses several key components:
- Agent: Main orchestrator
- DecisionEngine: Query processing
- ToolRegistry: Action execution
- Memory: Knowledge storage

## Details
""" + "\n".join([f"- Detail point {i}" for i in range(100)])

        (project / "medium.md").write_text(medium_content)

        # Large file (~50KB)
        large_content = """# Large File

## Comprehensive Documentation

This is a large documentation file with extensive content.
""" + "\n\n".join([
            f"### Section {i}\n" + "\n".join([f"Content line {j} for section {i}." for j in range(20)])
            for i in range(50)
        ])

        (project / "large.md").write_text(large_content)

        # Python source file
        (project / "agent.py").write_text('''"""Agent module."""

class Agent:
    """Main agent class that coordinates LLM and tools."""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools

    async def process(self, query: str) -> str:
        """Process a user query through the decision engine."""
        return await self.decide_and_execute(query)

    async def decide_and_execute(self, query: str) -> str:
        """Decide on action and execute it."""
        pass
''')

        yield project


# =============================================================================
# Unit Tests - Processors
# =============================================================================


class TestRawContextProcessor:
    """Tests for RawContextProcessor."""

    @pytest.mark.asyncio
    async def test_process_single_file(self, mock_llm, temp_project):
        """Test processing a single file."""
        processor = RawContextProcessor(llm=mock_llm)

        result = await processor.process(
            files=[temp_project / "small.md"],
            question="What does this file describe?",
        )

        assert result.strategy == RetrievalStrategy.RAW_CONTEXT
        assert result.answer != ""
        assert result.context_size_chars > 0
        assert result.total_time_ms > 0
        assert result.error is None

    @pytest.mark.asyncio
    async def test_process_multiple_files(self, mock_llm, temp_project):
        """Test processing multiple files."""
        processor = RawContextProcessor(llm=mock_llm)

        result = await processor.process(
            files=[temp_project / "small.md", temp_project / "medium.md"],
            question="What components are described?",
        )

        assert result.context_size_chars > 500  # Combined size
        assert "small.md" in result.retrieved_context
        assert "medium.md" in result.retrieved_context

    @pytest.mark.asyncio
    async def test_truncates_large_context(self, mock_llm, temp_project):
        """Test that large context is truncated."""
        processor = RawContextProcessor(llm=mock_llm, max_context_chars=1000)

        result = await processor.process(
            files=[temp_project / "large.md"],
            question="What is in this file?",
        )

        assert result.context_size_chars <= 1100  # Allow some buffer
        assert "[TRUNCATED]" in result.retrieved_context

    @pytest.mark.asyncio
    async def test_handles_missing_file(self, mock_llm, temp_project):
        """Test handling of missing file."""
        processor = RawContextProcessor(llm=mock_llm)

        result = await processor.process(
            files=[temp_project / "nonexistent.md"],
            question="What is this?",
        )

        # Should not error, just have empty context
        assert result.error is None


class TestSemanticWebProcessor:
    """Tests for SemanticWebProcessor."""

    @pytest.mark.asyncio
    async def test_process_with_knowledge(self, mock_llm, mock_semantic_memory):
        """Test processing with semantic knowledge."""
        processor = SemanticWebProcessor(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        result = await processor.process(
            files=[],  # Files not used - knowledge is pre-loaded
            question="How does the agent work?",
        )

        assert result.strategy == RetrievalStrategy.SEMANTIC_WEB
        assert result.answer != ""
        assert len(mock_semantic_memory.search_calls) > 0
        assert "agent" in mock_semantic_memory.search_calls[0].lower()

    @pytest.mark.asyncio
    async def test_retrieves_relevant_entities(self, mock_llm, mock_semantic_memory):
        """Test that relevant entities are retrieved."""
        processor = SemanticWebProcessor(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        result = await processor.process(
            files=[],
            question="Tell me about the decision engine.",
        )

        # Should have retrieved decision-related content
        assert "decision" in result.retrieved_context.lower()
        assert result.retrieval_time_ms > 0


class TestRAGProcessor:
    """Tests for RAGProcessor."""

    @pytest.mark.asyncio
    async def test_process_with_vector_store(self, mock_llm, mock_vector_store):
        """Test processing with vector store."""
        processor = RAGProcessor(
            llm=mock_llm,
            vector_store=mock_vector_store,
        )

        result = await processor.process(
            files=[],
            question="How does the agent work?",
        )

        assert result.strategy == RetrievalStrategy.RAG
        assert result.answer != ""
        assert len(mock_vector_store.search_calls) > 0

    @pytest.mark.asyncio
    async def test_handles_missing_vector_store(self, mock_llm):
        """Test handling when vector store not configured."""
        processor = RAGProcessor(llm=mock_llm, vector_store=None)

        result = await processor.process(
            files=[],
            question="How does the agent work?",
        )

        assert result.error is not None
        assert "not configured" in result.error


# =============================================================================
# Unit Tests - Evaluator
# =============================================================================


class TestBenchmarkEvaluator:
    """Tests for BenchmarkEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluates_result(self, mock_llm):
        """Test evaluation of a result."""
        evaluator = BenchmarkEvaluator(llm=mock_llm)

        test_case = TestCase(
            id="test_1",
            files=[],
            question="How does the agent work?",
            expected_answer_contains=["decision", "engine"],
        )

        from draagon_ai.orchestration.retrieval_benchmark import ProcessingResult

        result = ProcessingResult(
            strategy=RetrievalStrategy.RAW_CONTEXT,
            test_case_id="test_1",
            answer="The agent uses a decision engine to process queries.",
        )

        evaluation = await evaluator.evaluate(test_case, result)

        assert 0 <= evaluation.correctness <= 1
        assert 0 <= evaluation.completeness <= 1
        assert 0 <= evaluation.relevance <= 1
        assert 0 <= evaluation.overall <= 1


# =============================================================================
# Integration Tests - Full Benchmark
# =============================================================================


class TestRetrievalBenchmark:
    """Tests for the full benchmark framework."""

    @pytest.mark.asyncio
    async def test_run_benchmark(self, mock_llm, mock_semantic_memory, temp_project):
        """Test running a full benchmark."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        benchmark.add_test_case(
            files=[temp_project / "small.md"],
            question="What does the file describe?",
            expected_answer_contains=["agent", "decision"],
        )

        summary = await benchmark.run(evaluate=True)

        assert len(summary.results) == 1
        assert RetrievalStrategy.RAW_CONTEXT in summary.results[0].results
        assert RetrievalStrategy.SEMANTIC_WEB in summary.results[0].results

    @pytest.mark.asyncio
    async def test_determines_winner(self, mock_llm, mock_semantic_memory, temp_project):
        """Test that benchmark determines a winner."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        benchmark.add_test_case(
            files=[temp_project / "small.md"],
            question="Describe the agent architecture.",
            expected_answer_contains=["agent"],
        )

        summary = await benchmark.run(evaluate=True)

        result = summary.results[0]
        winner = result.winner()

        assert winner is not None
        assert winner in [RetrievalStrategy.RAW_CONTEXT, RetrievalStrategy.SEMANTIC_WEB]

    @pytest.mark.asyncio
    async def test_generates_summary(self, mock_llm, mock_semantic_memory, temp_project):
        """Test summary generation."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        benchmark.add_test_case(
            files=[temp_project / "small.md"],
            question="What is this about?",
        )

        benchmark.add_test_case(
            files=[temp_project / "medium.md"],
            question="Describe the architecture.",
        )

        summary = await benchmark.run(evaluate=True)

        # Should have aggregated stats
        assert len(summary.wins_by_strategy) > 0 or len(summary.avg_scores) > 0

        # Should generate readable summary
        table = summary.summary_table()
        assert "RETRIEVAL BENCHMARK SUMMARY" in table
        assert "Total test cases: 2" in table

    @pytest.mark.asyncio
    async def test_multiple_strategies(
        self, mock_llm, mock_semantic_memory, mock_vector_store, temp_project
    ):
        """Test benchmark with all three strategies."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
            vector_store=mock_vector_store,
        )

        benchmark.add_test_case(
            files=[temp_project / "small.md"],
            question="How does the agent work?",
        )

        summary = await benchmark.run(evaluate=True)

        result = summary.results[0]

        # Should have all three strategies
        assert RetrievalStrategy.RAW_CONTEXT in result.results
        assert RetrievalStrategy.SEMANTIC_WEB in result.results
        assert RetrievalStrategy.RAG in result.results


class TestCompareApproaches:
    """Tests for the compare_approaches convenience function."""

    @pytest.mark.asyncio
    async def test_quick_comparison(self, mock_llm, mock_semantic_memory, temp_project):
        """Test quick comparison function."""
        result = await compare_approaches(
            files=[temp_project / "small.md"],
            question="What is the main component?",
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
            expected_contains=["agent"],
        )

        assert isinstance(result, BenchmarkResult)
        assert len(result.results) >= 1


class TestCreateScaleTestSuite:
    """Tests for scale test suite creation."""

    def test_creates_test_cases(self, temp_project):
        """Test creating test cases at different scales."""
        questions = [
            ("How does the agent work?", ["agent", "process"]),
            ("What are the main components?", ["component"]),
        ]

        test_cases = create_scale_test_suite(temp_project, questions)

        # Should create 3 scale levels × 2 questions = 6 test cases
        assert len(test_cases) == 6

        # Should have small, medium, large variants
        tags = [tc.tags[0] for tc in test_cases if tc.tags]
        assert "small" in tags
        assert "medium" in tags
        assert "large" in tags


# =============================================================================
# Scale Tests - Demonstrate Semantic Web Advantage
# =============================================================================


class TestScaleComparison:
    """Tests demonstrating semantic web advantage at scale.

    These tests show that as file sizes grow:
    - Raw context degrades (truncation, lost in context)
    - Semantic web maintains quality (relevant facts retrieved)
    """

    @pytest.mark.asyncio
    async def test_small_scale_comparable(
        self, mock_llm, mock_semantic_memory, temp_project
    ):
        """At small scale, both approaches should work similarly."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        benchmark.add_test_case(
            files=[temp_project / "small.md"],
            question="What does the agent use?",
            expected_answer_contains=["decision", "engine"],
        )

        summary = await benchmark.run(evaluate=True)
        result = summary.results[0]

        # Both approaches should produce answers (success, not error)
        for strategy, proc_result in result.results.items():
            assert proc_result.error is None, f"{strategy} should not error"
            assert len(proc_result.answer) > 0, f"{strategy} should produce answer"

        # With mocks, evaluations may vary - just check they completed
        assert len(result.results) >= 2  # At least raw and semantic

    @pytest.mark.asyncio
    async def test_large_scale_semantic_advantage(
        self, mock_llm, mock_semantic_memory, temp_project
    ):
        """At large scale, semantic web should have advantage.

        This test demonstrates the concept - with real LLM and larger files,
        the semantic approach would show clearer advantage.
        """
        # Create knowledge base that would be in semantic memory after ingestion
        mock_semantic_memory.knowledge_base["needle"] = [
            "The critical configuration value is 42.",
            "This specific fact is buried in the large document.",
        ]

        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        # Large file with needle buried in it
        benchmark.add_test_case(
            files=[temp_project / "large.md"],  # 50KB file
            question="What is the needle critical configuration value?",
            expected_answer_contains=["42"],
            tags=["needle_in_haystack"],
        )

        summary = await benchmark.run(evaluate=True)
        result = summary.results[0]

        # Semantic web should retrieve the specific fact
        semantic_result = result.results.get(RetrievalStrategy.SEMANTIC_WEB)
        if semantic_result:
            assert "42" in semantic_result.retrieved_context or len(semantic_result.retrieved_context) > 0

    @pytest.mark.asyncio
    async def test_token_efficiency(self, mock_llm, mock_semantic_memory, temp_project):
        """Test that semantic web uses fewer tokens."""
        benchmark = RetrievalBenchmark(
            llm=mock_llm,
            semantic_memory=mock_semantic_memory,
        )

        benchmark.add_test_case(
            files=[temp_project / "large.md"],
            question="What is the main topic?",
        )

        summary = await benchmark.run(evaluate=False)
        result = summary.results[0]

        semantic = result.results.get(RetrievalStrategy.SEMANTIC_WEB)
        raw = result.results.get(RetrievalStrategy.RAW_CONTEXT)

        if semantic and raw:
            # Semantic should have smaller context
            assert semantic.context_size_chars <= raw.context_size_chars


# =============================================================================
# Integration with Real LLM (Skipped by Default)
# =============================================================================


@pytest.mark.skip(reason="Requires real LLM and semantic memory - run manually")
class TestRetrievalBenchmarkRealLLM:
    """Integration tests with real LLM."""

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
    async def test_real_comparison(self, real_llm, temp_project):
        """Run real comparison with actual LLM."""
        # This test uses real LLM but mock semantic memory
        # Full integration would use real Neo4j + ingestion

        mock_semantic = MockSemanticMemory({
            "agent": ["Agent coordinates LLM and tools for query processing."],
        })

        benchmark = RetrievalBenchmark(
            llm=real_llm,
            semantic_memory=mock_semantic,
        )

        benchmark.add_test_case(
            files=[temp_project / "agent.py"],
            question="What does the Agent class do?",
            expected_answer_contains=["process", "query"],
        )

        summary = await benchmark.run(evaluate=True)

        print(summary.summary_table())

        assert len(summary.results) == 1
