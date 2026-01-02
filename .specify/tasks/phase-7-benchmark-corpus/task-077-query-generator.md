# TASK-077: Multi-Hop & Adversarial Query Generator

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Need 250+ queries for benchmark)
**Effort**: 2 days
**Status**: Pending
**Dependencies**: TASK-075 (Corpus), TASK-076 (Query models)

---

## Description

Generate benchmark queries across all categories and difficulty levels:
- 135 standard queries (one per category minimum)
- 50 multi-hop queries (bridge, comparison, aggregation)
- 25 zero-result queries (out-of-domain, contradictory)
- 40 adversarial queries (keyword stuffing, paraphrasing)

Queries should test each document category (technical, legal, narrative, etc.)

**Location:** `src/draagon_ai/testing/benchmarks/query_generator.py`

---

## Acceptance Criteria

### Query Distribution

| Type | Count | % | Notes |
|------|-------|---|-------|
| Standard | 135 | 54% | ~17 per category |
| Multi-Hop | 50 | 20% | Bridge, comparison, aggregation |
| Zero-Result | 25 | 10% | Out-of-domain, nonsensical |
| Adversarial | 40 | 16% | Keyword stuffing, paraphrasing |
| **Total** | **250** | **100%** | |

### Difficulty Distribution

| Difficulty | % | Characteristics |
|------------|---|-----------------|
| EASY | 20% | Keywords present, explicit relationship |
| MEDIUM | 50% | Paraphrased, implicit relationship |
| HARD | 25% | No keyword overlap, inference required |
| EXPERT | 5% | Contradictory sources, deep understanding |

### Category Coverage
- [ ] Each category has minimum 15 queries
- [ ] Legal category has extra multi-hop (cross-reference testing)
- [ ] Conversational has informal language queries
- [ ] Narrative has story comprehension queries

### Query Generation Methods
- [ ] `generate_standard_queries(corpus, count)` - LLM generates from documents
- [ ] `generate_multi_hop_queries(corpus, count)` - Identify document pairs/chains
- [ ] `generate_zero_result_queries(corpus, count)` - Out-of-scope topics
- [ ] `generate_adversarial_queries(corpus, count)` - Create distractors

---

## Technical Notes

### Standard Query Generation

```python
async def generate_standard_queries(
    self,
    corpus: DocumentCorpus,
    count_per_category: int = 17,
) -> list[BenchmarkQuery]:
    """Generate standard queries from corpus documents."""
    queries = []

    for category in DocumentCategory:
        docs = corpus.get_by_category(category)

        for doc in docs[:count_per_category]:
            prompt = f"""Given this document, generate a question that can be
answered from its content. Also provide the expected answer.

Document ({category.value}):
{doc.content[:2000]}

Respond in XML:
<query>
    <question>The question</question>
    <expected_answer>Key phrases that should appear in answer</expected_answer>
    <difficulty>easy|medium|hard</difficulty>
</query>"""

            response = await self.llm.chat([{"role": "user", "content": prompt}])
            query = self._parse_query_response(response, doc, category)
            queries.append(query)

    return queries
```

### Multi-Hop Query Generation

```python
async def generate_multi_hop_queries(
    self,
    corpus: DocumentCorpus,
    count: int = 50,
) -> list[MultiHopQuery]:
    """Generate queries requiring multiple documents."""
    queries = []

    # Find document pairs with semantic overlap
    pairs = self._find_related_document_pairs(corpus)

    for doc1, doc2, relationship in pairs[:count]:
        prompt = f"""Create a question that requires information from BOTH documents.

Document 1 ({doc1.category.value}):
{doc1.content[:1500]}

Document 2 ({doc2.category.value}):
{doc2.content[:1500]}

Relationship type: {relationship}

Respond in XML:
<multi_hop_query>
    <question>Question requiring both documents</question>
    <hop1>
        <reasoning>Why doc1 is needed</reasoning>
        <fact>Key fact from doc1</fact>
    </hop1>
    <hop2>
        <reasoning>Why doc2 is needed</reasoning>
        <fact>Key fact from doc2</fact>
    </hop2>
    <expected_answer>Combined answer</expected_answer>
</multi_hop_query>"""

        response = await self.llm.chat([{"role": "user", "content": prompt}])
        query = self._parse_multi_hop_response(response, doc1, doc2)
        queries.append(query)

    return queries
```

### Category-Specific Query Templates

```python
CATEGORY_QUERY_TEMPLATES = {
    DocumentCategory.LEGAL: [
        "What liability protections does {doc} provide?",
        "Under what conditions can {party} terminate the agreement?",
        "What are the indemnification requirements in {doc}?",
    ],
    DocumentCategory.NARRATIVE: [
        "What is {character}'s motivation in the story?",
        "How does the setting affect the plot?",
        "What conflict does {character} face?",
    ],
    DocumentCategory.CONVERSATIONAL: [
        "What issue is the customer reporting?",
        "How was the problem resolved?",
        "What was the customer's sentiment?",
    ],
    DocumentCategory.ACADEMIC: [
        "What methodology does the paper use?",
        "What are the key findings?",
        "How does this relate to prior work?",
    ],
}
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_standard_query_generation(mock_llm, sample_corpus):
    """Generate standard queries from corpus."""
    generator = QueryGenerator(llm_provider=mock_llm)
    queries = await generator.generate_standard_queries(
        corpus=sample_corpus,
        count_per_category=2,
    )

    assert len(queries) >= 16  # 2 per 8 categories
    assert all(q.ground_truth_document_ids for q in queries)

@pytest.mark.asyncio
async def test_multi_hop_query_has_multiple_docs():
    """Multi-hop queries require 2+ documents."""
    queries = await generator.generate_multi_hop_queries(corpus, count=5)

    for query in queries:
        assert len(query.ground_truth_document_ids) >= 2
        assert len(query.hops) >= 2

@pytest.mark.asyncio
async def test_difficulty_distribution():
    """Queries follow difficulty distribution."""
    suite = await generator.generate_full_suite(corpus)

    easy = len([q for q in suite.queries if q.difficulty == QueryDifficulty.EASY])
    medium = len([q for q in suite.queries if q.difficulty == QueryDifficulty.MEDIUM])

    # Check ratios (with tolerance)
    total = len(suite.queries)
    assert 0.15 <= easy/total <= 0.25  # 20% ± 5%
    assert 0.45 <= medium/total <= 0.55  # 50% ± 5%
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_real_query_generation():
    """Generate queries with real LLM."""
    generator = QueryGenerator(llm_provider=real_llm)
    corpus = DocumentCorpus.load("test_corpus.json")

    queries = await generator.generate_standard_queries(corpus, count_per_category=2)

    assert len(queries) >= 10
    # Queries should be answerable from their linked documents
    for query in queries:
        doc = corpus.get_document(query.ground_truth_document_ids[0])
        # At least one expected keyword should appear in doc
        assert any(kw.lower() in doc.content.lower()
                   for kw in query.expected_answer_contains)
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/query_generator.py`
- Add tests to `tests/benchmarks/test_query_suite.py`

---

## Definition of Done

- [ ] 250+ queries generated
- [ ] All 8 categories covered (15+ queries each)
- [ ] 50 multi-hop queries with reasoning chains
- [ ] 25 zero-result queries
- [ ] 40 adversarial queries
- [ ] Difficulty distribution: 20/50/25/5
- [ ] Ground truth document IDs linked
- [ ] QuerySuite saved to JSON
- [ ] Human validation of sample queries
