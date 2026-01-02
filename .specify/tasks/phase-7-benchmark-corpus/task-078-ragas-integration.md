# TASK-078: RAGAS Metrics Integration

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Core evaluation framework)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-075 (Corpus), TASK-076 (Query Models)

---

## Description

Integrate RAGAS (Retrieval Augmented Generation Assessment) metrics for evaluating RAG pipeline quality:
- Faithfulness: Does the answer match the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Context Precision: Are retrieved documents ranked correctly?
- Context Recall: Are all relevant documents retrieved?

**Location:** `src/draagon_ai/testing/benchmarks/ragas_evaluator.py`

---

## Acceptance Criteria

### Core Metrics
- [ ] `Faithfulness` score (0-1): Claims in answer supported by context
- [ ] `AnswerRelevancy` score (0-1): Answer addresses the question
- [ ] `ContextPrecision` score (0-1): Ranking quality of retrieved docs
- [ ] `ContextRecall` score (0-1): Coverage of ground truth documents

### Evaluator Interface
- [ ] `RAGASEvaluator` class with LLM provider injection
- [ ] `evaluate_single(query, context, answer, ground_truth)` for one query
- [ ] `evaluate_batch(results)` for multiple queries
- [ ] `EvaluationResult` dataclass with all metrics + reasoning

### LLM-Based Evaluation
- [ ] Uses LLM to judge faithfulness (claim extraction + verification)
- [ ] Uses LLM to judge answer relevancy (question-answer alignment)
- [ ] Embedding-based context precision (semantic similarity)
- [ ] Ground truth document matching for context recall

### Performance
- [ ] Batch processing with configurable concurrency
- [ ] Progress tracking for long evaluations
- [ ] Caching of intermediate results

---

## Technical Notes

### RAGAS Metric Formulas

```python
# Faithfulness: What fraction of claims are supported?
faithfulness = supported_claims / total_claims

# Answer Relevancy: Semantic similarity between question and answer
answer_relevancy = cosine_similarity(embed(question), embed(answer))

# Context Precision: Are relevant docs ranked higher?
context_precision = sum(precision_at_k * is_relevant_k) / total_relevant

# Context Recall: Are all ground truth docs retrieved?
context_recall = len(retrieved ∩ ground_truth) / len(ground_truth)
```

### Evaluator Implementation

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class EvaluationResult:
    query_id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    # Detailed reasoning
    faithfulness_details: dict  # claims, supported, unsupported
    relevancy_details: dict     # question-answer alignment
    precision_details: dict     # per-position relevance
    recall_details: dict        # ground truth coverage

    @property
    def aggregate_score(self) -> float:
        """Weighted average of all metrics."""
        return (
            self.faithfulness * 0.3 +
            self.answer_relevancy * 0.2 +
            self.context_precision * 0.25 +
            self.context_recall * 0.25
        )


class RAGASEvaluator:
    def __init__(
        self,
        llm_provider: LLMProvider,
        embedding_provider: EmbeddingProvider,
        concurrency: int = 5,
    ):
        self.llm = llm_provider
        self.embedder = embedding_provider
        self.concurrency = concurrency

    async def evaluate_faithfulness(
        self,
        answer: str,
        context: list[str],
    ) -> tuple[float, dict]:
        """Extract claims from answer and verify against context."""

        # Step 1: Extract claims from answer
        extract_prompt = f"""Extract all factual claims from this answer.

Answer: {answer}

List each claim on a separate line:
<claims>
<claim>First factual claim</claim>
<claim>Second factual claim</claim>
</claims>"""

        claims_response = await self.llm.chat([{"role": "user", "content": extract_prompt}])
        claims = self._parse_claims(claims_response)

        # Step 2: Verify each claim against context
        supported = 0
        claim_results = []

        for claim in claims:
            verify_prompt = f"""Is this claim supported by the context?

Claim: {claim}

Context:
{chr(10).join(context)}

Respond with:
<verification>
    <supported>true|false</supported>
    <evidence>Quote from context if supported, or "not found"</evidence>
</verification>"""

            verify_response = await self.llm.chat([{"role": "user", "content": verify_prompt}])
            is_supported = self._parse_verification(verify_response)

            if is_supported:
                supported += 1
            claim_results.append({"claim": claim, "supported": is_supported})

        score = supported / len(claims) if claims else 1.0
        return score, {"claims": claim_results, "total": len(claims), "supported": supported}

    async def evaluate_context_recall(
        self,
        retrieved_doc_ids: list[str],
        ground_truth_doc_ids: list[str],
    ) -> tuple[float, dict]:
        """Calculate what fraction of ground truth docs were retrieved."""
        retrieved_set = set(retrieved_doc_ids)
        ground_truth_set = set(ground_truth_doc_ids)

        intersection = retrieved_set & ground_truth_set
        recall = len(intersection) / len(ground_truth_set) if ground_truth_set else 1.0

        return recall, {
            "retrieved": list(retrieved_set),
            "ground_truth": list(ground_truth_set),
            "found": list(intersection),
            "missed": list(ground_truth_set - intersection),
        }
```

### Batch Evaluation

```python
async def evaluate_batch(
    self,
    results: list[RetrievalResult],
    queries: list[BenchmarkQuery],
) -> BatchEvaluationResult:
    """Evaluate multiple query results with progress tracking."""

    semaphore = asyncio.Semaphore(self.concurrency)
    evaluations = []

    async def evaluate_one(result, query):
        async with semaphore:
            return await self.evaluate_single(
                query=query.question,
                context=[doc.content for doc in result.retrieved_documents],
                answer=result.answer,
                ground_truth_doc_ids=query.ground_truth_document_ids,
            )

    tasks = [
        evaluate_one(result, query)
        for result, query in zip(results, queries)
    ]

    evaluations = await asyncio.gather(*tasks)

    return BatchEvaluationResult(
        evaluations=evaluations,
        mean_faithfulness=statistics.mean(e.faithfulness for e in evaluations),
        mean_relevancy=statistics.mean(e.answer_relevancy for e in evaluations),
        mean_precision=statistics.mean(e.context_precision for e in evaluations),
        mean_recall=statistics.mean(e.context_recall for e in evaluations),
    )
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_faithfulness_scoring(mock_llm):
    """Faithfulness correctly identifies supported claims."""
    evaluator = RAGASEvaluator(llm_provider=mock_llm, embedding_provider=mock_embedder)

    score, details = await evaluator.evaluate_faithfulness(
        answer="The cat is black and weighs 10 pounds.",
        context=["The cat is black.", "The cat is 3 years old."],
    )

    # "black" is supported, "10 pounds" is not
    assert 0.4 <= score <= 0.6  # Approximately 50%
    assert details["total"] == 2
    assert details["supported"] == 1

@pytest.mark.asyncio
async def test_context_recall():
    """Context recall calculates ground truth coverage."""
    evaluator = RAGASEvaluator(...)

    score, details = await evaluator.evaluate_context_recall(
        retrieved_doc_ids=["doc_1", "doc_2", "doc_3"],
        ground_truth_doc_ids=["doc_1", "doc_4"],
    )

    assert score == 0.5  # 1 of 2 ground truth docs found
    assert details["found"] == ["doc_1"]
    assert details["missed"] == ["doc_4"]
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_ragas_evaluation():
    """End-to-end RAGAS evaluation with real LLM."""
    evaluator = RAGASEvaluator(
        llm_provider=real_llm,
        embedding_provider=real_embedder,
    )

    result = await evaluator.evaluate_single(
        query="What is the capital of France?",
        context=["Paris is the capital of France.", "France is in Europe."],
        answer="The capital of France is Paris.",
        ground_truth_doc_ids=["doc_france"],
    )

    assert result.faithfulness >= 0.9  # Answer matches context
    assert result.answer_relevancy >= 0.8  # Answer addresses question
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/ragas_evaluator.py`
- `src/draagon_ai/testing/benchmarks/evaluation.py` (result dataclasses)
- Add tests to `tests/benchmarks/test_ragas.py`

---

## Definition of Done

- [ ] All 4 RAGAS metrics implemented
- [ ] LLM-based faithfulness evaluation working
- [ ] Embedding-based answer relevancy working
- [ ] Context precision with ranking awareness
- [ ] Context recall with ground truth matching
- [ ] Batch evaluation with concurrency
- [ ] Unit tests for each metric
- [ ] Integration test with real LLM
