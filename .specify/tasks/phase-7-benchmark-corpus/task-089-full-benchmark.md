# TASK-089: Full Benchmark Execution

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Validates everything works)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: All previous tasks (TASK-070 through TASK-088)

---

## Description

Execute the complete benchmark suite end-to-end:
- 500+ documents across 8 categories
- 250+ queries (standard, multi-hop, adversarial, zero-result)
- 5 runs for statistical validity
- RAGAS evaluation on all queries
- Baseline comparison (BM25, Contriever)
- Full report generation

This is the integration test that proves everything works together.

**Location:** Script in `scripts/run_full_benchmark.py`

---

## Acceptance Criteria

### Corpus Validation
- [ ] 500+ documents loaded
- [ ] All 8 categories represented
- [ ] Category distribution within 10% of targets
- [ ] No duplicate documents (by content hash)

### Query Validation
- [ ] 250+ queries loaded
- [ ] 135+ standard queries
- [ ] 50+ multi-hop queries
- [ ] 25+ zero-result queries
- [ ] 40+ adversarial queries
- [ ] Ground truth document IDs valid

### Execution Requirements
- [ ] All 5 runs complete without error
- [ ] All queries processed (no timeouts on >95%)
- [ ] Checkpointing works (can resume after interrupt)
- [ ] Progress tracking accurate

### Evaluation Requirements
- [ ] Faithfulness evaluated for all queries
- [ ] Context recall calculated correctly
- [ ] Ground truth matching accurate
- [ ] Multi-hop queries evaluated with hop tracking

### Performance Requirements
- [ ] Total runtime ≤ 30 minutes
- [ ] Memory usage ≤ 8GB
- [ ] No memory leaks over 5 runs

### Report Requirements
- [ ] Executive summary generated
- [ ] Detailed report generated
- [ ] Per-query CSV generated
- [ ] Baseline comparison generated
- [ ] History updated

---

## Technical Notes

### Execution Script

```python
#!/usr/bin/env python3
"""Run full production benchmark."""

import asyncio
import logging
from pathlib import Path
from datetime import datetime

from draagon_ai.testing.benchmarks import (
    BenchmarkConfig,
    BenchmarkRunner,
    DocumentCorpus,
    QuerySuite,
    OllamaEmbeddingProvider,
    HybridRetriever,
)
from draagon_ai.llm import GroqProvider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    # Configuration
    config = BenchmarkConfig(
        corpus_path=Path("data/benchmark_corpus.json"),
        queries_path=Path("data/benchmark_queries.json"),
        num_runs=5,
        base_seed=42,
        concurrency=10,
        timeout_per_query=30.0,
        output_dir=Path("benchmark_results"),
        run_baselines=True,
        save_checkpoints=True,
    )

    # Validate corpus
    logger.info("Validating corpus...")
    corpus = DocumentCorpus.load(config.corpus_path)
    validate_corpus(corpus)

    # Validate queries
    logger.info("Validating queries...")
    queries = QuerySuite.load(config.queries_path)
    validate_queries(queries, corpus)

    # Initialize components
    logger.info("Initializing components...")
    embedding_provider = OllamaEmbeddingProvider()

    if not await embedding_provider.health_check():
        raise RuntimeError("Ollama not available - required for benchmark")

    llm_provider = GroqProvider()
    retriever = HybridRetriever(
        embedding_provider=embedding_provider,
        # ... other config
    )

    # Run benchmark
    logger.info("Starting benchmark...")
    runner = BenchmarkRunner(
        config=config,
        retriever=retriever,
        llm_provider=llm_provider,
        embedding_provider=embedding_provider,
    )

    start_time = datetime.now()
    result = await runner.run()
    duration = (datetime.now() - start_time).total_seconds()

    # Validate results
    logger.info("Validating results...")
    validate_results(result)

    # Summary
    logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║                    BENCHMARK COMPLETE                        ║
╠══════════════════════════════════════════════════════════════╣
║  Duration: {duration/60:>6.1f} minutes                                  ║
║  Queries:  {len(queries.queries):>6d}                                           ║
║  Runs:     {result.harness_result.num_runs:>6d}                                           ║
╠══════════════════════════════════════════════════════════════╣
║  Faithfulness:    {result.harness_result.aggregates['faithfulness'].mean:>6.3f}                                  ║
║  Context Recall:  {result.harness_result.aggregates['context_recall'].mean:>6.3f}                                  ║
║  Aggregate Score: {result.harness_result.aggregates['aggregate_score'].mean:>6.3f}                                  ║
╠══════════════════════════════════════════════════════════════╣
║  vs BM25:       +{(result.harness_result.aggregates['aggregate_score'].mean - result.baseline_results.get('bm25', {}).get('aggregate_score', 0)) * 100:>5.1f}%                                    ║
║  vs Contriever: +{(result.harness_result.aggregates['aggregate_score'].mean - result.baseline_results.get('contriever', {}).get('aggregate_score', 0)) * 100:>5.1f}%                                    ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Return exit code based on thresholds
    if result.harness_result.aggregates['faithfulness'].mean < 0.80:
        logger.error("❌ Faithfulness below threshold (0.80)")
        return 1

    if result.harness_result.aggregates['context_recall'].mean < 0.80:
        logger.error("❌ Context Recall below threshold (0.80)")
        return 1

    logger.info("✅ All thresholds passed!")
    return 0


def validate_corpus(corpus: DocumentCorpus):
    """Validate corpus meets requirements."""
    assert len(corpus) >= 500, f"Corpus too small: {len(corpus)} < 500"

    # Check categories
    from collections import Counter
    categories = Counter(doc.category for doc in corpus.documents)

    for category in DocumentCategory:
        count = categories.get(category, 0)
        assert count > 0, f"Missing category: {category}"

    # Check for duplicates
    hashes = [doc.content_hash for doc in corpus.documents]
    assert len(hashes) == len(set(hashes)), "Duplicate documents found"

    logger.info(f"✓ Corpus valid: {len(corpus)} documents")


def validate_queries(queries: QuerySuite, corpus: DocumentCorpus):
    """Validate queries meet requirements."""
    assert len(queries.queries) >= 250, f"Too few queries: {len(queries.queries)} < 250"

    # Check distribution
    from collections import Counter
    types = Counter(q.query_type for q in queries.queries)

    standard_count = sum(v for k, v in types.items() if "STANDARD" in k.value.upper())
    assert standard_count >= 135, f"Too few standard queries: {standard_count}"

    multi_hop_count = sum(v for k, v in types.items() if "MULTI_HOP" in k.value.upper())
    assert multi_hop_count >= 50, f"Too few multi-hop queries: {multi_hop_count}"

    # Validate ground truth references
    corpus_doc_ids = {doc.doc_id for doc in corpus.documents}
    for query in queries.queries:
        for doc_id in query.ground_truth_document_ids:
            assert doc_id in corpus_doc_ids, f"Invalid ground truth: {doc_id}"

    logger.info(f"✓ Queries valid: {len(queries.queries)} queries")


def validate_results(result: BenchmarkResult):
    """Validate results are complete and reasonable."""
    # All runs completed
    assert result.harness_result.num_runs == 5

    # All metrics present
    required_metrics = ["faithfulness", "context_recall", "aggregate_score"]
    for metric in required_metrics:
        assert metric in result.harness_result.aggregates

    # Metrics in valid range
    for name, agg in result.harness_result.aggregates.items():
        assert 0 <= agg.mean <= 1, f"Invalid metric range: {name} = {agg.mean}"

    # Low variance (stable results)
    for name, agg in result.harness_result.aggregates.items():
        assert agg.coefficient_of_variation < 0.2, f"High variance: {name} CV = {agg.coefficient_of_variation}"

    logger.info("✓ Results valid")


if __name__ == "__main__":
    exit(asyncio.run(main()))
```

### Pre-Benchmark Checklist

```markdown
## Pre-Benchmark Checklist

Before running full benchmark, verify:

### Environment
- [ ] Ollama running with mxbai-embed-large model
- [ ] GROQ_API_KEY set
- [ ] 8GB+ RAM available
- [ ] 30+ minutes of uninterrupted time

### Data
- [ ] Corpus JSON exists and is valid
- [ ] Query suite JSON exists and is valid
- [ ] Output directory is writable

### Configuration
- [ ] num_runs = 5
- [ ] base_seed = 42
- [ ] All evaluations enabled

### Commands
```bash
# Check Ollama
ollama list | grep mxbai

# Check API key
echo $GROQ_API_KEY | head -c 10

# Run benchmark
python scripts/run_full_benchmark.py
```
```

---

## Testing Requirements

### Pre-Flight Tests
```python
@pytest.mark.asyncio
async def test_corpus_loadable():
    """Corpus loads without error."""
    corpus = DocumentCorpus.load(Path("data/benchmark_corpus.json"))
    assert len(corpus) >= 500

@pytest.mark.asyncio
async def test_queries_loadable():
    """Query suite loads without error."""
    queries = QuerySuite.load(Path("data/benchmark_queries.json"))
    assert len(queries.queries) >= 250

@pytest.mark.asyncio
async def test_ollama_available():
    """Ollama embedding model available."""
    provider = OllamaEmbeddingProvider()
    assert await provider.health_check()

@pytest.mark.asyncio
async def test_groq_available():
    """Groq API accessible."""
    provider = GroqProvider()
    response = await provider.chat([{"role": "user", "content": "Test"}])
    assert len(response) > 0
```

### Mini-Benchmark Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_mini_benchmark():
    """Run benchmark with subset to verify pipeline."""
    config = BenchmarkConfig(
        corpus_path=Path("data/benchmark_corpus.json"),
        queries_path=Path("data/benchmark_queries.json"),
        num_runs=1,
        max_queries=10,
        run_baselines=False,
    )

    runner = BenchmarkRunner(config, ...)
    result = await runner.run()

    assert result.harness_result.num_runs == 1
    assert "faithfulness" in result.harness_result.aggregates
```

---

## Files to Create/Modify

- `scripts/run_full_benchmark.py`
- `scripts/validate_benchmark_data.py`
- Add tests to `tests/integration/test_full_benchmark.py`

---

## Definition of Done

- [ ] Full benchmark runs to completion
- [ ] 500+ documents processed
- [ ] 250+ queries evaluated
- [ ] 5 runs with different seeds
- [ ] All RAGAS metrics calculated
- [ ] Baseline comparisons generated
- [ ] Reports written successfully
- [ ] Runtime ≤ 30 minutes
- [ ] All thresholds passed (or documented why not)
- [ ] Repeatable (same seeds = same results)
