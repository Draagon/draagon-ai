# TASK-082: Embedding Quality Validation

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Ensures embeddings actually work)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-080, TASK-081

---

## Description

Validate that embedding providers produce semantically meaningful vectors:
- Semantic similarity tests (paraphrases should be similar)
- Retrieval accuracy tests (can we find related documents?)
- Cross-provider consistency (results should be comparable)
- MTEB-style evaluation subset

This catches issues like hash-based "embeddings" that don't capture meaning.

**Location:** `src/draagon_ai/testing/benchmarks/embedding_validation.py`

---

## Acceptance Criteria

### Semantic Similarity Validation
- [ ] Paraphrase pairs have similarity > 0.8
- [ ] Unrelated pairs have similarity < 0.5
- [ ] Negations have lower similarity than affirmations
- [ ] Uses curated test pairs (not random)

### Retrieval Validation
- [ ] Given query, correct document in top-3
- [ ] Distractor documents ranked lower
- [ ] Multi-hop queries retrieve related docs
- [ ] Zero-result queries don't return high confidence

### Cross-Provider Validation
- [ ] Ollama and SentenceTransformer agree on rankings
- [ ] Correlation coefficient > 0.7 between providers
- [ ] No provider produces degenerate embeddings

### MTEB Subset
- [ ] Run on 100 examples from STS Benchmark
- [ ] Spearman correlation > 0.7 with human labels
- [ ] Report MTEB-compatible scores

---

## Technical Notes

### Semantic Similarity Test Pairs

```python
# Curated test pairs for validation
PARAPHRASE_PAIRS = [
    ("The cat sat on the mat.", "A cat was sitting on a mat."),
    ("I love programming.", "I enjoy coding."),
    ("The weather is nice today.", "It's a beautiful day outside."),
    ("She runs fast.", "She is a quick runner."),
    ("The restaurant was expensive.", "The meal cost a lot of money."),
]

UNRELATED_PAIRS = [
    ("The cat sat on the mat.", "The stock market crashed."),
    ("I love programming.", "The pizza was delicious."),
    ("The weather is nice.", "My car needs an oil change."),
]

NEGATION_PAIRS = [
    ("I love coffee.", "I hate coffee."),
    ("The movie was great.", "The movie was terrible."),
    ("She is happy.", "She is sad."),
]
```

### Validation Suite

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class ValidationResult:
    provider_name: str
    paraphrase_accuracy: float  # % of pairs with sim > 0.8
    unrelated_accuracy: float   # % of pairs with sim < 0.5
    negation_accuracy: float    # % where negation < affirmation
    retrieval_mrr: float        # Mean reciprocal rank
    sts_correlation: float      # Spearman on STS subset
    passed: bool                # Overall pass/fail

    def __str__(self) -> str:
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        return f"""
{self.provider_name} Embedding Validation: {status}
  Paraphrase Accuracy: {self.paraphrase_accuracy:.1%}
  Unrelated Accuracy:  {self.unrelated_accuracy:.1%}
  Negation Accuracy:   {self.negation_accuracy:.1%}
  Retrieval MRR:       {self.retrieval_mrr:.3f}
  STS Correlation:     {self.sts_correlation:.3f}
"""


class EmbeddingValidator:
    def __init__(
        self,
        provider: EmbeddingProvider,
        sts_data: list[tuple[str, str, float]] = None,
    ):
        self.provider = provider
        self.sts_data = sts_data or self._load_sts_subset()

    async def validate(self) -> ValidationResult:
        """Run full validation suite."""
        paraphrase_acc = await self._validate_paraphrases()
        unrelated_acc = await self._validate_unrelated()
        negation_acc = await self._validate_negations()
        retrieval_mrr = await self._validate_retrieval()
        sts_corr = await self._validate_sts()

        # Pass criteria
        passed = (
            paraphrase_acc >= 0.8 and
            unrelated_acc >= 0.8 and
            negation_acc >= 0.6 and
            retrieval_mrr >= 0.5 and
            sts_corr >= 0.6
        )

        return ValidationResult(
            provider_name=self.provider.__class__.__name__,
            paraphrase_accuracy=paraphrase_acc,
            unrelated_accuracy=unrelated_acc,
            negation_accuracy=negation_acc,
            retrieval_mrr=retrieval_mrr,
            sts_correlation=sts_corr,
            passed=passed,
        )

    async def _validate_paraphrases(self) -> float:
        """Check that paraphrases have high similarity."""
        correct = 0
        for text1, text2 in PARAPHRASE_PAIRS:
            emb1 = await self.provider.embed(text1)
            emb2 = await self.provider.embed(text2)
            sim = cosine_similarity(emb1, emb2)
            if sim > 0.8:
                correct += 1
        return correct / len(PARAPHRASE_PAIRS)

    async def _validate_unrelated(self) -> float:
        """Check that unrelated pairs have low similarity."""
        correct = 0
        for text1, text2 in UNRELATED_PAIRS:
            emb1 = await self.provider.embed(text1)
            emb2 = await self.provider.embed(text2)
            sim = cosine_similarity(emb1, emb2)
            if sim < 0.5:
                correct += 1
        return correct / len(UNRELATED_PAIRS)

    async def _validate_negations(self) -> float:
        """Check that negations are less similar than the original."""
        correct = 0
        for affirmative, negation in NEGATION_PAIRS:
            emb_aff = await self.provider.embed(affirmative)
            emb_neg = await self.provider.embed(negation)
            sim = cosine_similarity(emb_aff, emb_neg)
            # Negations should be < 0.9 similar (not identical)
            if sim < 0.9:
                correct += 1
        return correct / len(NEGATION_PAIRS)

    async def _validate_sts(self) -> float:
        """Validate against STS Benchmark subset."""
        predictions = []
        labels = []

        for text1, text2, label in self.sts_data:
            emb1 = await self.provider.embed(text1)
            emb2 = await self.provider.embed(text2)
            sim = cosine_similarity(emb1, emb2)
            predictions.append(sim)
            labels.append(label)

        # Spearman correlation
        from scipy.stats import spearmanr
        correlation, _ = spearmanr(predictions, labels)
        return correlation
```

### Cross-Provider Validation

```python
async def validate_cross_provider_consistency(
    provider_a: EmbeddingProvider,
    provider_b: EmbeddingProvider,
    test_texts: list[str],
) -> float:
    """Check that providers produce consistent rankings."""
    # Get pairwise similarities from both providers
    sims_a = []
    sims_b = []

    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i < j:
                emb1_a = await provider_a.embed(text1)
                emb2_a = await provider_a.embed(text2)
                sims_a.append(cosine_similarity(emb1_a, emb2_a))

                emb1_b = await provider_b.embed(text1)
                emb2_b = await provider_b.embed(text2)
                sims_b.append(cosine_similarity(emb1_b, emb2_b))

    # Correlation between similarity judgments
    from scipy.stats import pearsonr
    correlation, _ = pearsonr(sims_a, sims_b)
    return correlation
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_paraphrase_validation():
    """Paraphrases detected correctly."""
    provider = SentenceTransformerProvider()
    validator = EmbeddingValidator(provider)

    accuracy = await validator._validate_paraphrases()
    assert accuracy >= 0.8

@pytest.mark.asyncio
async def test_unrelated_validation():
    """Unrelated texts have low similarity."""
    provider = SentenceTransformerProvider()
    validator = EmbeddingValidator(provider)

    accuracy = await validator._validate_unrelated()
    assert accuracy >= 0.8

@pytest.mark.asyncio
async def test_full_validation_passes():
    """Full validation suite passes for real embeddings."""
    provider = SentenceTransformerProvider()
    validator = EmbeddingValidator(provider)

    result = await validator.validate()
    assert result.passed
```

### Mock Embedding Detection
```python
@pytest.mark.asyncio
async def test_hash_embeddings_fail_validation():
    """Hash-based fake embeddings fail validation."""

    class HashEmbeddingProvider:
        async def embed(self, text: str) -> list[float]:
            # Fake embedding based on hash (no semantic meaning)
            import hashlib
            h = hashlib.md5(text.encode()).hexdigest()
            return [int(c, 16) / 16.0 for c in h[:384]]

    provider = HashEmbeddingProvider()
    validator = EmbeddingValidator(provider)

    result = await validator.validate()
    assert not result.passed  # Should fail!
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_ollama_validation():
    """Ollama embeddings pass validation."""
    provider = OllamaEmbeddingProvider()

    if not await provider.health_check():
        pytest.skip("Ollama not available")

    validator = EmbeddingValidator(provider)
    result = await validator.validate()

    print(result)
    assert result.passed
    assert result.sts_correlation > 0.7
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/embedding_validation.py`
- `src/draagon_ai/testing/benchmarks/test_pairs.py` (curated test data)
- Add tests to `tests/benchmarks/test_embedding_validation.py`

---

## Definition of Done

- [ ] Paraphrase similarity validation
- [ ] Unrelated pair validation
- [ ] Negation detection validation
- [ ] Retrieval MRR validation
- [ ] STS Benchmark subset correlation
- [ ] Cross-provider consistency check
- [ ] Hash/mock embedding detection (fails validation)
- [ ] All real providers pass validation
- [ ] Validation report generated
