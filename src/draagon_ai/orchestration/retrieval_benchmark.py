"""Retrieval Benchmark - Compare semantic web vs raw context vs RAG approaches.

This module provides a framework for benchmarking different retrieval strategies:

1. **Semantic Web** - Load files into knowledge graph, query graph for context
2. **Raw Context** - Inject full file content directly into LLM context
3. **RAG (future)** - Use vector similarity search (Qdrant) for retrieval

The hypothesis is that semantic web approaches scale better as files get larger,
because they extract and index the relevant knowledge rather than relying on
the LLM to find needles in haystacks.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                         RetrievalBenchmark                               │
    │                                                                          │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
    │  │  Semantic Web   │  │  Raw Context    │  │  RAG (Qdrant)          │ │
    │  │  Processor      │  │  Processor      │  │  Processor (future)    │ │
    │  └────────┬────────┘  └────────┬────────┘  └────────────┬───────────┘ │
    │           │                    │                         │             │
    │           ▼                    ▼                         ▼             │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                      Evaluator (LLM-as-judge)                    │  │
    │  │  - Correctness: Does answer match expected?                      │  │
    │  │  - Completeness: All relevant info included?                     │  │
    │  │  - Relevance: No irrelevant information?                         │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    │                                                                          │
    │  ┌─────────────────────────────────────────────────────────────────┐  │
    │  │                      Metrics Collector                           │  │
    │  │  - Token usage (input/output)                                    │  │
    │  │  - Latency (retrieval + generation)                              │  │
    │  │  - Accuracy at scale                                             │  │
    │  └─────────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────────┘

Usage:
    from draagon_ai.orchestration import RetrievalBenchmark

    benchmark = RetrievalBenchmark(llm=my_llm, semantic_memory=my_memory)

    # Add test cases
    benchmark.add_test_case(
        files=["docs/CLAUDE.md", "src/agent.py"],
        question="How does the agent process queries?",
        expected_answer_contains=["decision engine", "tool execution"],
    )

    # Run comparison
    results = await benchmark.run()

    # Results show which approach is better at each scale
    print(results.summary())
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
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
    """Semantic memory for knowledge graph queries."""

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Any]:
        ...

    async def get_related_entities(
        self,
        entity_id: str,
        relationship_types: list[str] | None = None,
        depth: int = 1,
    ) -> list[Any]:
        ...


class VectorStoreProvider(Protocol):
    """Vector store for RAG (future - Qdrant)."""

    async def search(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Any]:
        ...


# =============================================================================
# Data Types
# =============================================================================


class RetrievalStrategy(str, Enum):
    """Available retrieval strategies."""

    SEMANTIC_WEB = "semantic_web"  # Query knowledge graph
    RAW_CONTEXT = "raw_context"  # Inject full file content
    RAG = "rag"  # Vector similarity search (future)


@dataclass
class BenchmarkTestCase:
    """A single benchmark test case."""

    id: str
    files: list[str | Path]
    question: str

    # Expected outcomes
    expected_answer_contains: list[str] = field(default_factory=list)
    expected_entities: list[str] = field(default_factory=list)
    expected_facts: list[str] = field(default_factory=list)

    # Metadata
    category: str = "general"
    difficulty: str = "medium"  # easy, medium, hard
    tags: list[str] = field(default_factory=list)


# Alias for backwards compatibility
TestCase = BenchmarkTestCase


@dataclass
class ProcessingResult:
    """Result from a single processing attempt."""

    strategy: RetrievalStrategy
    test_case_id: str
    answer: str

    # Metrics
    retrieval_time_ms: float = 0.0
    generation_time_ms: float = 0.0
    total_time_ms: float = 0.0

    # Token usage (if available)
    input_tokens: int = 0
    output_tokens: int = 0
    context_size_chars: int = 0

    # Retrieved context
    retrieved_context: str = ""
    retrieved_entities: list[str] = field(default_factory=list)
    retrieved_facts: list[str] = field(default_factory=list)

    # Errors
    error: str | None = None


@dataclass
class EvaluationResult:
    """Evaluation of a processing result."""

    # Scores (0.0 - 1.0)
    correctness: float = 0.0
    completeness: float = 0.0
    relevance: float = 0.0
    overall: float = 0.0

    # Details
    reasoning: str = ""
    missing_info: list[str] = field(default_factory=list)
    extra_info: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""

    test_case: TestCase
    results: dict[RetrievalStrategy, ProcessingResult] = field(default_factory=dict)
    evaluations: dict[RetrievalStrategy, EvaluationResult] = field(default_factory=dict)

    def winner(self) -> RetrievalStrategy | None:
        """Return the strategy with highest overall score."""
        if not self.evaluations:
            return None
        return max(self.evaluations, key=lambda s: self.evaluations[s].overall)


@dataclass
class BenchmarkSummary:
    """Summary of all benchmark results."""

    results: list[BenchmarkResult] = field(default_factory=list)

    # Aggregated wins
    wins_by_strategy: dict[RetrievalStrategy, int] = field(default_factory=dict)

    # Aggregated metrics
    avg_scores: dict[RetrievalStrategy, float] = field(default_factory=dict)
    avg_latency_ms: dict[RetrievalStrategy, float] = field(default_factory=dict)
    avg_tokens: dict[RetrievalStrategy, int] = field(default_factory=dict)

    # By file size category
    scores_by_size: dict[str, dict[RetrievalStrategy, float]] = field(
        default_factory=dict
    )

    def summary_table(self) -> str:
        """Generate a summary table."""
        lines = [
            "=" * 70,
            "RETRIEVAL BENCHMARK SUMMARY",
            "=" * 70,
            "",
            f"Total test cases: {len(self.results)}",
            "",
            "WINS BY STRATEGY:",
        ]

        for strategy, wins in sorted(
            self.wins_by_strategy.items(), key=lambda x: -x[1]
        ):
            lines.append(f"  {strategy.value}: {wins} wins")

        lines.extend([
            "",
            "AVERAGE SCORES:",
        ])

        for strategy, score in sorted(self.avg_scores.items(), key=lambda x: -x[1]):
            lines.append(f"  {strategy.value}: {score:.2f}")

        lines.extend([
            "",
            "AVERAGE LATENCY (ms):",
        ])

        for strategy, latency in sorted(
            self.avg_latency_ms.items(), key=lambda x: x[1]
        ):
            lines.append(f"  {strategy.value}: {latency:.1f}ms")

        if self.scores_by_size:
            lines.extend([
                "",
                "SCORES BY FILE SIZE:",
            ])
            for size, scores in self.scores_by_size.items():
                lines.append(f"  {size}:")
                for strategy, score in scores.items():
                    lines.append(f"    {strategy.value}: {score:.2f}")

        lines.append("=" * 70)
        return "\n".join(lines)


# =============================================================================
# Retrieval Processors
# =============================================================================


class RetrievalProcessor(ABC):
    """Base class for retrieval processors."""

    @property
    @abstractmethod
    def strategy(self) -> RetrievalStrategy:
        """Return the strategy this processor implements."""
        ...

    @abstractmethod
    async def process(
        self,
        files: list[Path],
        question: str,
    ) -> ProcessingResult:
        """Process files and answer question."""
        ...


class SemanticWebProcessor(RetrievalProcessor):
    """Process using semantic knowledge graph.

    Flow:
    1. Files are pre-loaded into semantic memory (entities, facts, relationships)
    2. Question is used to query the graph for relevant context
    3. Retrieved context is passed to LLM for answer generation
    """

    def __init__(
        self,
        llm: LLMProvider,
        semantic_memory: SemanticMemoryProvider,
        max_context_items: int = 20,
    ):
        self.llm = llm
        self.semantic = semantic_memory
        self.max_context_items = max_context_items

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.SEMANTIC_WEB

    async def process(
        self,
        files: list[Path],
        question: str,
    ) -> ProcessingResult:
        """Query semantic graph and generate answer."""
        result = ProcessingResult(
            strategy=self.strategy,
            test_case_id="",
            answer="",
        )

        start_time = time.perf_counter()

        try:
            # Step 1: Query semantic memory
            retrieval_start = time.perf_counter()
            search_results = await self.semantic.search(
                question, limit=self.max_context_items
            )
            result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

            # Build context from retrieved items
            context_parts = []
            for item in search_results:
                if hasattr(item, "content"):
                    context_parts.append(item.content)
                    if hasattr(item, "entity_name"):
                        result.retrieved_entities.append(item.entity_name)
                elif isinstance(item, dict):
                    context_parts.append(str(item.get("content", item)))

            result.retrieved_context = "\n".join(context_parts)
            result.context_size_chars = len(result.retrieved_context)

            # Step 2: Generate answer
            generation_start = time.perf_counter()

            prompt = f"""You are answering a question based on knowledge from a semantic graph.

RETRIEVED KNOWLEDGE:
{result.retrieved_context}

QUESTION: {question}

Answer based ONLY on the retrieved knowledge. If the knowledge doesn't contain the answer, say so."""

            result.answer = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result.generation_time_ms = (time.perf_counter() - generation_start) * 1000

        except Exception as e:
            result.error = str(e)
            logger.error(f"SemanticWebProcessor error: {e}")

        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        return result


class RawContextProcessor(RetrievalProcessor):
    """Process by injecting full file content into LLM context.

    Flow:
    1. Read all file contents
    2. Concatenate and inject into LLM prompt
    3. LLM finds relevant info and generates answer

    This is the baseline - how LLMs typically handle context.
    At small scales this works well, but degrades as files grow.
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_context_chars: int = 100000,  # ~25k tokens
    ):
        self.llm = llm
        self.max_context_chars = max_context_chars

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.RAW_CONTEXT

    async def process(
        self,
        files: list[Path],
        question: str,
    ) -> ProcessingResult:
        """Inject raw file content and generate answer."""
        result = ProcessingResult(
            strategy=self.strategy,
            test_case_id="",
            answer="",
        )

        start_time = time.perf_counter()

        try:
            # Step 1: Read all files
            retrieval_start = time.perf_counter()
            context_parts = []
            total_chars = 0

            for file_path in files:
                if not file_path.exists():
                    continue

                try:
                    content = file_path.read_text(encoding="utf-8")

                    # Check if we'd exceed limit
                    if total_chars + len(content) > self.max_context_chars:
                        # Truncate this file
                        remaining = self.max_context_chars - total_chars
                        content = content[:remaining] + "\n[TRUNCATED]"

                    context_parts.append(f"=== {file_path.name} ===\n{content}")
                    total_chars += len(content)

                    if total_chars >= self.max_context_chars:
                        break

                except Exception as e:
                    logger.warning(f"Failed to read {file_path}: {e}")

            result.retrieved_context = "\n\n".join(context_parts)
            result.context_size_chars = len(result.retrieved_context)
            result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

            # Step 2: Generate answer
            generation_start = time.perf_counter()

            prompt = f"""You are answering a question based on the following files.

FILES:
{result.retrieved_context}

QUESTION: {question}

Answer based on the file contents. Be specific and cite relevant sections."""

            result.answer = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result.generation_time_ms = (time.perf_counter() - generation_start) * 1000

        except Exception as e:
            result.error = str(e)
            logger.error(f"RawContextProcessor error: {e}")

        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        return result


class RAGProcessor(RetrievalProcessor):
    """Process using vector similarity search (RAG).

    Flow:
    1. Files are pre-chunked and embedded in vector store (Qdrant)
    2. Question is embedded and similar chunks retrieved
    3. Retrieved chunks passed to LLM for answer generation

    This is the traditional RAG approach - better than raw context at scale,
    but may miss connections that semantic graphs capture.
    """

    def __init__(
        self,
        llm: LLMProvider,
        vector_store: VectorStoreProvider | None = None,
        max_chunks: int = 10,
    ):
        self.llm = llm
        self.vector_store = vector_store
        self.max_chunks = max_chunks

    @property
    def strategy(self) -> RetrievalStrategy:
        return RetrievalStrategy.RAG

    async def process(
        self,
        files: list[Path],
        question: str,
    ) -> ProcessingResult:
        """Query vector store and generate answer."""
        result = ProcessingResult(
            strategy=self.strategy,
            test_case_id="",
            answer="",
        )

        start_time = time.perf_counter()

        if not self.vector_store:
            result.error = "RAG processor not configured - vector store required"
            result.answer = "[RAG not available - vector store not configured]"
            result.total_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        try:
            # Step 1: Query vector store
            retrieval_start = time.perf_counter()
            search_results = await self.vector_store.search(
                question, limit=self.max_chunks
            )
            result.retrieval_time_ms = (time.perf_counter() - retrieval_start) * 1000

            # Build context from chunks
            context_parts = []
            for chunk in search_results:
                if hasattr(chunk, "content"):
                    context_parts.append(chunk.content)
                elif isinstance(chunk, dict):
                    context_parts.append(str(chunk.get("content", chunk)))

            result.retrieved_context = "\n---\n".join(context_parts)
            result.context_size_chars = len(result.retrieved_context)

            # Step 2: Generate answer
            generation_start = time.perf_counter()

            prompt = f"""You are answering a question based on retrieved document chunks.

RETRIEVED CHUNKS:
{result.retrieved_context}

QUESTION: {question}

Answer based on the retrieved chunks. If the information is incomplete, note what's missing."""

            result.answer = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1000,
            )

            result.generation_time_ms = (time.perf_counter() - generation_start) * 1000

        except Exception as e:
            result.error = str(e)
            logger.error(f"RAGProcessor error: {e}")

        result.total_time_ms = (time.perf_counter() - start_time) * 1000
        return result


# =============================================================================
# Evaluator
# =============================================================================


class BenchmarkEvaluator:
    """Evaluates processing results using LLM-as-judge."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def evaluate(
        self,
        test_case: TestCase,
        result: ProcessingResult,
    ) -> EvaluationResult:
        """Evaluate a processing result against expected outcomes."""
        evaluation = EvaluationResult()

        prompt = f"""Evaluate this answer against the expected criteria.

QUESTION: {test_case.question}

EXPECTED TO CONTAIN: {', '.join(test_case.expected_answer_contains) or 'N/A'}
EXPECTED ENTITIES: {', '.join(test_case.expected_entities) or 'N/A'}
EXPECTED FACTS: {', '.join(test_case.expected_facts) or 'N/A'}

ACTUAL ANSWER:
{result.answer}

Rate the answer on these criteria (0.0 to 1.0):

<evaluation>
  <correctness>0.0-1.0</correctness>
  <correctness_reason>Why this score</correctness_reason>
  <completeness>0.0-1.0</completeness>
  <completeness_reason>Why this score</completeness_reason>
  <relevance>0.0-1.0</relevance>
  <relevance_reason>Why this score</relevance_reason>
  <missing>Comma-separated list of missing info, or "none"</missing>
  <extra>Comma-separated list of irrelevant info, or "none"</extra>
</evaluation>

Be strict but fair. An answer can be correct even if worded differently."""

        try:
            response = await self.llm.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=500,
            )

            # Parse scores
            import re

            correctness_match = re.search(
                r"<correctness>([0-9.]+)</correctness>", response
            )
            completeness_match = re.search(
                r"<completeness>([0-9.]+)</completeness>", response
            )
            relevance_match = re.search(r"<relevance>([0-9.]+)</relevance>", response)
            missing_match = re.search(r"<missing>([^<]+)</missing>", response)
            extra_match = re.search(r"<extra>([^<]+)</extra>", response)

            if correctness_match:
                evaluation.correctness = float(correctness_match.group(1))
            if completeness_match:
                evaluation.completeness = float(completeness_match.group(1))
            if relevance_match:
                evaluation.relevance = float(relevance_match.group(1))

            evaluation.overall = (
                evaluation.correctness * 0.5
                + evaluation.completeness * 0.3
                + evaluation.relevance * 0.2
            )

            if missing_match and missing_match.group(1).strip().lower() != "none":
                evaluation.missing_info = [
                    m.strip() for m in missing_match.group(1).split(",")
                ]

            if extra_match and extra_match.group(1).strip().lower() != "none":
                evaluation.extra_info = [
                    e.strip() for e in extra_match.group(1).split(",")
                ]

            evaluation.reasoning = response

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            evaluation.reasoning = f"Evaluation error: {e}"

        return evaluation


# =============================================================================
# Main Benchmark Class
# =============================================================================


class RetrievalBenchmark:
    """Main benchmark runner comparing retrieval strategies."""

    def __init__(
        self,
        llm: LLMProvider,
        semantic_memory: SemanticMemoryProvider | None = None,
        vector_store: VectorStoreProvider | None = None,
        strategies: list[RetrievalStrategy] | None = None,
    ):
        """Initialize benchmark.

        Args:
            llm: LLM provider for generation and evaluation
            semantic_memory: Semantic memory for knowledge graph queries
            vector_store: Vector store for RAG queries (optional)
            strategies: Which strategies to benchmark (default: all available)
        """
        self.llm = llm
        self.semantic_memory = semantic_memory
        self.vector_store = vector_store

        # Build processors
        self.processors: dict[RetrievalStrategy, RetrievalProcessor] = {}

        if semantic_memory:
            self.processors[RetrievalStrategy.SEMANTIC_WEB] = SemanticWebProcessor(
                llm=llm, semantic_memory=semantic_memory
            )

        self.processors[RetrievalStrategy.RAW_CONTEXT] = RawContextProcessor(llm=llm)

        if vector_store:
            self.processors[RetrievalStrategy.RAG] = RAGProcessor(
                llm=llm, vector_store=vector_store
            )

        # Filter to requested strategies
        if strategies:
            self.processors = {
                s: p for s, p in self.processors.items() if s in strategies
            }

        self.evaluator = BenchmarkEvaluator(llm)
        self.test_cases: list[TestCase] = []

    def add_test_case(
        self,
        files: list[str | Path],
        question: str,
        expected_answer_contains: list[str] | None = None,
        expected_entities: list[str] | None = None,
        expected_facts: list[str] | None = None,
        category: str = "general",
        difficulty: str = "medium",
        tags: list[str] | None = None,
    ) -> None:
        """Add a test case to the benchmark."""
        test_case = TestCase(
            id=f"tc_{len(self.test_cases) + 1}",
            files=[Path(f) for f in files],
            question=question,
            expected_answer_contains=expected_answer_contains or [],
            expected_entities=expected_entities or [],
            expected_facts=expected_facts or [],
            category=category,
            difficulty=difficulty,
            tags=tags or [],
        )
        self.test_cases.append(test_case)

    async def run(
        self,
        evaluate: bool = True,
    ) -> BenchmarkSummary:
        """Run the benchmark on all test cases.

        Args:
            evaluate: Whether to evaluate results (slower but more informative)

        Returns:
            BenchmarkSummary with all results
        """
        all_results: list[BenchmarkResult] = []

        for test_case in self.test_cases:
            logger.info(f"Running test case: {test_case.id} - {test_case.question[:50]}")

            benchmark_result = BenchmarkResult(test_case=test_case)

            # Run each strategy
            for strategy, processor in self.processors.items():
                logger.debug(f"  Strategy: {strategy.value}")

                result = await processor.process(test_case.files, test_case.question)
                result.test_case_id = test_case.id
                benchmark_result.results[strategy] = result

                # Evaluate if requested
                if evaluate and not result.error:
                    evaluation = await self.evaluator.evaluate(test_case, result)
                    benchmark_result.evaluations[strategy] = evaluation
                    logger.debug(f"    Score: {evaluation.overall:.2f}")

            all_results.append(benchmark_result)

        # Build summary
        return self._build_summary(all_results)

    async def run_single(
        self,
        test_case: TestCase,
        evaluate: bool = True,
    ) -> BenchmarkResult:
        """Run benchmark on a single test case."""
        benchmark_result = BenchmarkResult(test_case=test_case)

        for strategy, processor in self.processors.items():
            result = await processor.process(test_case.files, test_case.question)
            result.test_case_id = test_case.id
            benchmark_result.results[strategy] = result

            if evaluate and not result.error:
                evaluation = await self.evaluator.evaluate(test_case, result)
                benchmark_result.evaluations[strategy] = evaluation

        return benchmark_result

    def _build_summary(self, results: list[BenchmarkResult]) -> BenchmarkSummary:
        """Build summary from results."""
        summary = BenchmarkSummary(results=results)

        # Count wins
        for result in results:
            winner = result.winner()
            if winner:
                summary.wins_by_strategy[winner] = (
                    summary.wins_by_strategy.get(winner, 0) + 1
                )

        # Calculate averages
        strategy_scores: dict[RetrievalStrategy, list[float]] = {}
        strategy_latencies: dict[RetrievalStrategy, list[float]] = {}
        strategy_tokens: dict[RetrievalStrategy, list[int]] = {}

        for result in results:
            for strategy, evaluation in result.evaluations.items():
                if strategy not in strategy_scores:
                    strategy_scores[strategy] = []
                strategy_scores[strategy].append(evaluation.overall)

            for strategy, proc_result in result.results.items():
                if strategy not in strategy_latencies:
                    strategy_latencies[strategy] = []
                    strategy_tokens[strategy] = []
                strategy_latencies[strategy].append(proc_result.total_time_ms)
                strategy_tokens[strategy].append(
                    proc_result.input_tokens + proc_result.output_tokens
                )

        for strategy, scores in strategy_scores.items():
            summary.avg_scores[strategy] = sum(scores) / len(scores) if scores else 0

        for strategy, latencies in strategy_latencies.items():
            summary.avg_latency_ms[strategy] = (
                sum(latencies) / len(latencies) if latencies else 0
            )

        for strategy, tokens in strategy_tokens.items():
            summary.avg_tokens[strategy] = (
                int(sum(tokens) / len(tokens)) if tokens else 0
            )

        # Group by file size
        size_buckets = {"small": [], "medium": [], "large": []}
        for result in results:
            total_size = sum(
                f.stat().st_size for f in result.test_case.files if f.exists()
            )
            if total_size < 10000:  # <10KB
                bucket = "small"
            elif total_size < 100000:  # <100KB
                bucket = "medium"
            else:
                bucket = "large"

            for strategy, evaluation in result.evaluations.items():
                size_buckets[bucket].append((strategy, evaluation.overall))

        for size, items in size_buckets.items():
            if items:
                by_strategy: dict[RetrievalStrategy, list[float]] = {}
                for strategy, score in items:
                    if strategy not in by_strategy:
                        by_strategy[strategy] = []
                    by_strategy[strategy].append(score)

                summary.scores_by_size[size] = {
                    s: sum(scores) / len(scores) for s, scores in by_strategy.items()
                }

        return summary


# =============================================================================
# Convenience Functions
# =============================================================================


async def compare_approaches(
    files: list[str | Path],
    question: str,
    llm: LLMProvider,
    semantic_memory: SemanticMemoryProvider | None = None,
    expected_contains: list[str] | None = None,
) -> BenchmarkResult:
    """Quick comparison of approaches for a single question.

    Args:
        files: Files to process
        question: Question to answer
        llm: LLM provider
        semantic_memory: Optional semantic memory for knowledge graph
        expected_contains: Expected keywords in answer

    Returns:
        BenchmarkResult comparing approaches
    """
    benchmark = RetrievalBenchmark(
        llm=llm,
        semantic_memory=semantic_memory,
    )

    benchmark.add_test_case(
        files=files,
        question=question,
        expected_answer_contains=expected_contains or [],
    )

    summary = await benchmark.run()
    return summary.results[0] if summary.results else BenchmarkResult(
        test_case=TestCase(id="", files=[], question=question)
    )


def create_scale_test_suite(
    base_path: Path,
    questions: list[tuple[str, list[str]]],  # (question, expected_contains)
) -> list[TestCase]:
    """Create test cases at different file size scales.

    Args:
        base_path: Path to project directory
        questions: List of (question, expected_keywords) tuples

    Returns:
        List of TestCase objects at various scales
    """
    test_cases = []

    # Find files at different scales
    all_files = list(base_path.rglob("*.md")) + list(base_path.rglob("*.py"))

    # Sort by size
    files_by_size = sorted(all_files, key=lambda f: f.stat().st_size)

    # Create buckets
    small_files = files_by_size[:5] if len(files_by_size) >= 5 else files_by_size
    medium_files = files_by_size[:20] if len(files_by_size) >= 20 else files_by_size
    large_files = files_by_size  # All files

    for i, (question, expected) in enumerate(questions):
        # Small scale test
        test_cases.append(
            TestCase(
                id=f"scale_small_{i}",
                files=small_files,
                question=question,
                expected_answer_contains=expected,
                category="scale_test",
                difficulty="easy",
                tags=["small"],
            )
        )

        # Medium scale test
        test_cases.append(
            TestCase(
                id=f"scale_medium_{i}",
                files=medium_files,
                question=question,
                expected_answer_contains=expected,
                category="scale_test",
                difficulty="medium",
                tags=["medium"],
            )
        )

        # Large scale test
        test_cases.append(
            TestCase(
                id=f"scale_large_{i}",
                files=large_files,
                question=question,
                expected_answer_contains=expected,
                category="scale_test",
                difficulty="hard",
                tags=["large"],
            )
        )

    return test_cases
