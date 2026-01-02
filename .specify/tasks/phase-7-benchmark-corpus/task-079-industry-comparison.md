# TASK-079: Industry Comparison Framework

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P1 (Validates we're competitive)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-078 (RAGAS Integration)

---

## Description

Implement framework for comparing draagon-ai retrieval against industry baselines:
- BM25 (lexical baseline)
- Contriever (neural retrieval)
- ColBERT (late interaction)
- OpenAI ada-002 (commercial embedding)

This provides context for our results - we need to know if we're better than simple baselines.

**Location:** `src/draagon_ai/testing/benchmarks/baselines.py`

---

## Acceptance Criteria

### Baseline Implementations
- [ ] `BM25Baseline` using rank-bm25 library
- [ ] `ContrieverBaseline` using HuggingFace model
- [ ] `AdaBaseline` using OpenAI embeddings (optional, requires API key)
- [ ] Common `BaselineRetriever` interface

### Comparison Framework
- [ ] `BaselineComparison` runs same queries on all retrievers
- [ ] Side-by-side metric comparison (nDCG@10, Recall@k, MRR)
- [ ] Statistical significance testing (paired t-test, Wilcoxon)
- [ ] Improvement percentages vs each baseline

### Metrics
- [ ] nDCG@10: Normalized Discounted Cumulative Gain at 10
- [ ] Recall@1, @5, @10: Documents retrieved at each cutoff
- [ ] MRR: Mean Reciprocal Rank
- [ ] P@1: Precision at 1 (first result correct?)

---

## Technical Notes

### Baseline Interface

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    doc_ids: list[str]
    scores: list[float]
    query: str


class BaselineRetriever(ABC):
    @abstractmethod
    async def index(self, documents: list[BenchmarkDocument]) -> None:
        """Index documents for retrieval."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        """Retrieve top-k documents for query."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for reports."""
        pass
```

### BM25 Baseline

```python
from rank_bm25 import BM25Okapi

class BM25Baseline(BaselineRetriever):
    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

    async def index(self, documents: list[BenchmarkDocument]) -> None:
        self.documents = documents
        self.doc_ids = [doc.doc_id for doc in documents]

        # Tokenize documents
        tokenized = [doc.content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized)

    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_k_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]

        return RetrievalResult(
            doc_ids=[self.doc_ids[i] for i in top_k_indices],
            scores=[scores[i] for i in top_k_indices],
            query=query,
        )

    @property
    def name(self) -> str:
        return "BM25"
```

### Contriever Baseline

```python
from sentence_transformers import SentenceTransformer
import numpy as np

class ContrieverBaseline(BaselineRetriever):
    def __init__(self, model_name: str = "facebook/contriever"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.doc_ids = []

    async def index(self, documents: list[BenchmarkDocument]) -> None:
        self.doc_ids = [doc.doc_id for doc in documents]
        texts = [doc.content for doc in documents]

        # Batch encode
        self.embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
        )

    async def retrieve(self, query: str, k: int = 10) -> RetrievalResult:
        query_embedding = self.model.encode(query)

        # Cosine similarity
        scores = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        top_k_indices = np.argsort(scores)[::-1][:k]

        return RetrievalResult(
            doc_ids=[self.doc_ids[i] for i in top_k_indices],
            scores=[float(scores[i]) for i in top_k_indices],
            query=query,
        )

    @property
    def name(self) -> str:
        return "Contriever"
```

### Comparison Runner

```python
@dataclass
class ComparisonResult:
    retriever_name: str
    ndcg_at_10: float
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    mrr: float
    p_at_1: float


class BaselineComparison:
    def __init__(
        self,
        corpus: DocumentCorpus,
        queries: QuerySuite,
        retrievers: list[BaselineRetriever],
    ):
        self.corpus = corpus
        self.queries = queries
        self.retrievers = retrievers

    async def run(self) -> dict[str, ComparisonResult]:
        results = {}

        for retriever in self.retrievers:
            # Index corpus
            await retriever.index(self.corpus.documents)

            # Run all queries
            query_results = []
            for query in self.queries.queries:
                result = await retriever.retrieve(query.question, k=10)
                query_results.append((query, result))

            # Calculate metrics
            metrics = self._calculate_metrics(query_results)
            results[retriever.name] = metrics

        return results

    def _calculate_ndcg_at_k(
        self,
        retrieved: list[str],
        ground_truth: list[str],
        k: int = 10,
    ) -> float:
        """Normalized Discounted Cumulative Gain at k."""
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            if doc_id in ground_truth:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed

        # Ideal DCG
        idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(ground_truth), k)))

        return dcg / idcg if idcg > 0 else 0.0
```

### Statistical Significance

```python
from scipy import stats

def compare_significance(
    scores_a: list[float],
    scores_b: list[float],
    test: str = "wilcoxon",
) -> tuple[float, bool]:
    """Test if improvement is statistically significant."""
    if test == "wilcoxon":
        statistic, p_value = stats.wilcoxon(scores_a, scores_b)
    elif test == "ttest":
        statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    significant = p_value < 0.05
    return p_value, significant
```

---

## Testing Requirements

### Unit Tests
```python
def test_bm25_baseline_retrieval():
    """BM25 retrieves relevant documents."""
    baseline = BM25Baseline()
    await baseline.index([
        BenchmarkDocument(doc_id="1", content="Python programming language"),
        BenchmarkDocument(doc_id="2", content="Java programming language"),
    ])

    result = await baseline.retrieve("Python", k=1)
    assert result.doc_ids[0] == "1"

def test_ndcg_calculation():
    """nDCG calculated correctly."""
    comparison = BaselineComparison(...)
    ndcg = comparison._calculate_ndcg_at_k(
        retrieved=["doc_1", "doc_3", "doc_2"],
        ground_truth=["doc_1", "doc_2"],
        k=10,
    )
    # doc_1 at position 1, doc_2 at position 3
    assert 0.5 < ndcg < 1.0
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_baseline_comparison():
    """Compare all baselines on test corpus."""
    comparison = BaselineComparison(
        corpus=test_corpus,
        queries=test_queries,
        retrievers=[BM25Baseline(), ContrieverBaseline()],
    )

    results = await comparison.run()

    assert "BM25" in results
    assert "Contriever" in results
    assert results["Contriever"].ndcg_at_10 >= results["BM25"].ndcg_at_10
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/baselines.py`
- `src/draagon_ai/testing/benchmarks/metrics.py` (nDCG, MRR, etc.)
- Add tests to `tests/benchmarks/test_baselines.py`

---

## Definition of Done

- [ ] BM25 baseline implemented
- [ ] Contriever baseline implemented
- [ ] OpenAI ada baseline (optional)
- [ ] nDCG@10 calculation correct
- [ ] Recall@k calculations correct
- [ ] MRR calculation correct
- [ ] Statistical significance testing
- [ ] Side-by-side comparison reports
- [ ] Unit tests for each baseline
- [ ] Integration test comparing baselines
