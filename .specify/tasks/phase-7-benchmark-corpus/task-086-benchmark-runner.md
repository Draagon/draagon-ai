# TASK-086: BenchmarkRunner Orchestrator

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Central orchestration)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-075 (Corpus), TASK-078 (RAGAS), TASK-083 (Harness)

---

## Description

Implement the main orchestrator that coordinates all benchmark components:
- Load corpus and query suite
- Configure retrieval pipeline
- Execute benchmark with multiple runs
- Generate RAGAS evaluations
- Produce reports and save results

This is the "main" entry point for running benchmarks.

**Location:** `src/draagon_ai/testing/benchmarks/runner.py`

---

## Acceptance Criteria

### Configuration
- [ ] `BenchmarkConfig` dataclass with all settings
- [ ] Load config from YAML/JSON file
- [ ] CLI argument overrides
- [ ] Sensible defaults for all options

### Orchestration
- [ ] `BenchmarkRunner.run()` executes full benchmark
- [ ] Coordinates corpus loading, query execution, evaluation
- [ ] Handles errors gracefully (partial results saved)
- [ ] Progress reporting during execution

### Pipeline Integration
- [ ] Pluggable retrieval pipeline interface
- [ ] Support for multiple retriever configurations
- [ ] A/B testing between retrievers
- [ ] Baseline comparison runs

### Output
- [ ] Results saved to output directory
- [ ] Reports generated (markdown, CSV, JSON)
- [ ] Run metadata (config, seeds, timestamps)
- [ ] Checkpoints for resumption

---

## Technical Notes

### Configuration

```python
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

@dataclass
class BenchmarkConfig:
    # Corpus
    corpus_path: Path
    queries_path: Path

    # Execution
    num_runs: int = 5
    base_seed: int = 42
    concurrency: int = 10
    timeout_per_query: float = 30.0

    # Evaluation
    evaluate_faithfulness: bool = True
    evaluate_relevancy: bool = True
    evaluate_precision: bool = True
    evaluate_recall: bool = True

    # Output
    output_dir: Path = Path("benchmark_results")
    save_checkpoints: bool = True
    checkpoint_interval: int = 50  # Queries between checkpoints

    # Baselines
    run_baselines: bool = True
    baseline_retrievers: list[str] = field(default_factory=lambda: ["bm25", "contriever"])

    # Advanced
    filter_query_types: Optional[list[str]] = None
    filter_difficulties: Optional[list[str]] = None
    max_queries: Optional[int] = None

    @classmethod
    def from_yaml(cls, path: Path) -> "BenchmarkConfig":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def from_cli(cls, args) -> "BenchmarkConfig":
        """Create config from argparse namespace."""
        return cls(
            corpus_path=Path(args.corpus),
            queries_path=Path(args.queries),
            num_runs=args.runs or 5,
            output_dir=Path(args.output or "benchmark_results"),
            run_baselines=not args.skip_baselines,
        )
```

### Main Runner

```python
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    harness_result: HarnessResult
    baseline_results: dict[str, HarnessResult]
    timestamp: datetime
    duration_seconds: float


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        retriever: BaselineRetriever,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
    ):
        self.config = config
        self.retriever = retriever
        self.llm = llm_provider
        self.embedder = embedding_provider

        self.corpus: Optional[DocumentCorpus] = None
        self.queries: Optional[QuerySuite] = None
        self.evaluator: Optional[RAGASEvaluator] = None

    async def run(self) -> BenchmarkResult:
        """Execute full benchmark suite."""
        start_time = datetime.now()
        logger.info(f"Starting benchmark with config: {self.config}")

        try:
            # Phase 1: Load data
            await self._load_data()

            # Phase 2: Run our retriever
            harness = MultiRunHarness(
                num_runs=self.config.num_runs,
                base_seed=self.config.base_seed,
            )

            our_result = await harness.run(
                self._run_single_benchmark,
                retriever=self.retriever,
            )

            # Phase 3: Run baselines
            baseline_results = {}
            if self.config.run_baselines:
                baseline_results = await self._run_baselines()

            # Phase 4: Generate reports
            await self._generate_reports(our_result, baseline_results)

            duration = (datetime.now() - start_time).total_seconds()

            return BenchmarkResult(
                config=self.config,
                harness_result=our_result,
                baseline_results=baseline_results,
                timestamp=start_time,
                duration_seconds=duration,
            )

        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            # Save partial results
            await self._save_partial_results()
            raise

    async def _load_data(self):
        """Load corpus and queries."""
        logger.info("Loading corpus...")
        self.corpus = DocumentCorpus.load(self.config.corpus_path)
        logger.info(f"Loaded {len(self.corpus)} documents")

        logger.info("Loading queries...")
        self.queries = QuerySuite.load(self.config.queries_path)
        logger.info(f"Loaded {len(self.queries.queries)} queries")

        # Apply filters
        if self.config.filter_query_types:
            self.queries = self.queries.filter_by_types(self.config.filter_query_types)

        if self.config.filter_difficulties:
            self.queries = self.queries.filter_by_difficulties(self.config.filter_difficulties)

        if self.config.max_queries:
            self.queries = self.queries.limit(self.config.max_queries)

        # Initialize evaluator
        self.evaluator = RAGASEvaluator(
            llm_provider=self.llm,
            embedding_provider=self.embedder,
            concurrency=self.config.concurrency,
        )

    async def _run_single_benchmark(
        self,
        retriever: BaselineRetriever,
    ) -> dict[str, float]:
        """Run single benchmark iteration, return metrics."""
        # Index corpus
        await retriever.index(self.corpus.documents)

        # Run all queries
        results = []
        for i, query in enumerate(self.queries.queries):
            if i > 0 and i % 50 == 0:
                logger.info(f"Progress: {i}/{len(self.queries.queries)} queries")

            try:
                result = await asyncio.wait_for(
                    self._run_single_query(retriever, query),
                    timeout=self.config.timeout_per_query,
                )
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning(f"Query {query.query_id} timed out")
                results.append(self._timeout_result(query))

            # Checkpoint
            if self.config.save_checkpoints and i % self.config.checkpoint_interval == 0:
                await self._save_checkpoint(results, i)

        # Evaluate all results
        evaluation = await self.evaluator.evaluate_batch(results, self.queries.queries)

        return {
            "faithfulness": evaluation.mean_faithfulness,
            "answer_relevancy": evaluation.mean_relevancy,
            "context_precision": evaluation.mean_precision,
            "context_recall": evaluation.mean_recall,
            "aggregate_score": evaluation.aggregate_score,
        }

    async def _run_single_query(
        self,
        retriever: BaselineRetriever,
        query: BenchmarkQuery,
    ) -> RetrievalResult:
        """Execute single query and get answer."""
        # Retrieve documents
        retrieval = await retriever.retrieve(query.question, k=10)

        # Generate answer from retrieved context
        context = [
            self.corpus.get_document(doc_id).content
            for doc_id in retrieval.doc_ids
            if self.corpus.get_document(doc_id)
        ]

        answer = await self._generate_answer(query.question, context)

        return RetrievalResult(
            query_id=query.query_id,
            retrieved_doc_ids=retrieval.doc_ids,
            scores=retrieval.scores,
            context=context,
            answer=answer,
        )

    async def _generate_answer(
        self,
        question: str,
        context: list[str],
    ) -> str:
        """Generate answer using LLM with retrieved context."""
        prompt = f"""Answer the question based on the provided context.

Context:
{chr(10).join(context[:5])}

Question: {question}

Answer:"""

        response = await self.llm.chat([{"role": "user", "content": prompt}])
        return response

    async def _run_baselines(self) -> dict[str, HarnessResult]:
        """Run baseline retrievers for comparison."""
        results = {}

        for baseline_name in self.config.baseline_retrievers:
            logger.info(f"Running baseline: {baseline_name}")

            if baseline_name == "bm25":
                retriever = BM25Baseline()
            elif baseline_name == "contriever":
                retriever = ContrieverBaseline()
            else:
                logger.warning(f"Unknown baseline: {baseline_name}")
                continue

            harness = MultiRunHarness(
                num_runs=self.config.num_runs,
                base_seed=self.config.base_seed,
            )

            results[baseline_name] = await harness.run(
                self._run_single_benchmark,
                retriever=retriever,
            )

        return results

    async def _generate_reports(
        self,
        our_result: HarnessResult,
        baseline_results: dict[str, HarnessResult],
    ):
        """Generate all report formats."""
        from .reporting import save_reports

        output_dir = self.config.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        save_reports(our_result, output_dir, baseline_results)

        logger.info(f"Reports saved to {output_dir}")
```

### CLI Entry Point

```python
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run retrieval benchmark")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSON")
    parser.add_argument("--queries", required=True, help="Path to queries JSON")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--output", help="Output directory")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--config", help="Path to config YAML")

    args = parser.parse_args()

    if args.config:
        config = BenchmarkConfig.from_yaml(Path(args.config))
    else:
        config = BenchmarkConfig.from_cli(args)

    # Initialize components
    retriever = HybridRetriever(...)  # Our retriever
    llm = GroqProvider(...)
    embedder = OllamaEmbeddingProvider()

    runner = BenchmarkRunner(config, retriever, llm, embedder)
    result = asyncio.run(runner.run())

    print(result.harness_result.to_report())


if __name__ == "__main__":
    main()
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_runner_loads_data(tmp_path, mock_corpus, mock_queries):
    """Runner loads corpus and queries."""
    config = BenchmarkConfig(
        corpus_path=tmp_path / "corpus.json",
        queries_path=tmp_path / "queries.json",
    )

    runner = BenchmarkRunner(config, mock_retriever, mock_llm, mock_embedder)
    await runner._load_data()

    assert len(runner.corpus) > 0
    assert len(runner.queries.queries) > 0

@pytest.mark.asyncio
async def test_runner_generates_reports(tmp_path):
    """Runner generates all report formats."""
    config = BenchmarkConfig(
        output_dir=tmp_path,
        ...
    )

    runner = BenchmarkRunner(config, ...)
    result = await runner.run()

    assert (tmp_path / "summary.md").exists()
    assert (tmp_path / "results.json").exists()
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_benchmark_run():
    """Run complete benchmark with real components."""
    config = BenchmarkConfig(
        corpus_path=Path("test_corpus.json"),
        queries_path=Path("test_queries.json"),
        num_runs=2,
        max_queries=10,
    )

    runner = BenchmarkRunner(
        config,
        HybridRetriever(...),
        GroqProvider(...),
        OllamaEmbeddingProvider(),
    )

    result = await runner.run()

    assert result.harness_result.num_runs == 2
    assert "faithfulness" in result.harness_result.aggregates
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/runner.py`
- `src/draagon_ai/testing/benchmarks/config.py`
- `src/draagon_ai/testing/benchmarks/__main__.py` (CLI entry)
- Add tests to `tests/benchmarks/test_runner.py`

---

## Definition of Done

- [ ] BenchmarkConfig with all options
- [ ] YAML/JSON config loading
- [ ] CLI argument parsing
- [ ] BenchmarkRunner orchestrates all phases
- [ ] Progress logging
- [ ] Error handling with partial saves
- [ ] Baseline comparison runs
- [ ] Report generation
- [ ] Integration test passing
