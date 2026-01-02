# TASK-076: Query Data Models

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Foundation for all query types)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None (can parallel with corpus tasks)

---

## Description

Implement data models for benchmark queries:
- Standard queries with expected answers
- Multi-hop queries with reasoning chain
- Zero-result queries (no valid answer exists)
- Adversarial queries with distractors
- Query suite aggregation and serialization

**Location:** `src/draagon_ai/testing/benchmarks/queries.py`

---

## Acceptance Criteria

### Core Query Model
- [ ] `BenchmarkQuery` base dataclass with query_id, question, expected_answer
- [ ] `QueryType` enum: STANDARD, MULTI_HOP, ZERO_RESULT, ADVERSARIAL
- [ ] `QueryDifficulty` enum: EASY, MEDIUM, HARD, EXPERT
- [ ] `ground_truth_document_ids` links to relevant corpus documents
- [ ] `target_category` specifies which document category query tests

### Multi-Hop Queries
- [ ] `MultiHopQuery` with list of `HopDescription`
- [ ] Each hop: step number, reasoning, required_document_ids, required_facts
- [ ] `minimum_documents_required` (default: 2)
- [ ] Support for: BRIDGE, COMPARISON, AGGREGATION, TEMPORAL, NEGATION

### Zero-Result Queries
- [ ] `ZeroResultQuery` for queries with no valid answer
- [ ] Categories: OUT_OF_DOMAIN, TEMPORALLY_INVALID, NONSENSICAL, CONTRADICTORY
- [ ] `acceptable_responses` (e.g., "I don't know")
- [ ] `unacceptable_responses` (hallucinations)
- [ ] `max_confidence_threshold` (should be < 0.3)

### Adversarial Queries
- [ ] `AdversarialQuery` with attack vector type
- [ ] `DistractorDocument` with keyword_overlap, semantic_relevance scores
- [ ] Attack vectors: KEYWORD_STUFFING, SEMANTIC_PARAPHRASING, CONTRADICTORY_SOURCES
- [ ] `max_acceptable_false_positives`

### Query Suite
- [ ] `QuerySuite` aggregates all query types
- [ ] Save/load to JSON
- [ ] Filter by difficulty, type, category
- [ ] Statistics: count by type, difficulty distribution

---

## Technical Notes

### Query Type Enums

```python
class QueryType(str, Enum):
    STANDARD = "standard"
    MULTI_HOP_BRIDGE = "multi_hop_bridge"
    MULTI_HOP_COMPARISON = "multi_hop_comparison"
    MULTI_HOP_AGGREGATION = "multi_hop_aggregation"
    MULTI_HOP_TEMPORAL = "multi_hop_temporal"
    MULTI_HOP_NEGATION = "multi_hop_negation"
    ZERO_RESULT = "zero_result"
    ADVERSARIAL = "adversarial"

class QueryDifficulty(str, Enum):
    EASY = "easy"      # 20% - Keywords present, explicit
    MEDIUM = "medium"  # 50% - Paraphrased, implicit
    HARD = "hard"      # 25% - No keyword overlap, inference
    EXPERT = "expert"  # 5% - Contradictory sources, deep understanding
```

### Multi-Hop Query Example

```python
@dataclass
class MultiHopQuery(BenchmarkQuery):
    hops: list[HopDescription]
    minimum_documents_required: int = 2
    maximum_documents_sufficient: int = 5

# Example query
MultiHopQuery(
    query_id="mh_legal_001",
    question="What liability protections does the MIT license provide that the GPL does not?",
    query_type=QueryType.MULTI_HOP_COMPARISON,
    difficulty=QueryDifficulty.HARD,
    target_category=DocumentCategory.LEGAL,
    expected_answer_contains=["limitation of liability", "warranty disclaimer"],
    ground_truth_document_ids=["license_mit", "license_gpl3"],
    hops=[
        HopDescription(
            step=1,
            reasoning="Find MIT license liability clause",
            required_document_ids=["license_mit"],
            required_facts=["MIT disclaims all warranties"],
        ),
        HopDescription(
            step=2,
            reasoning="Find GPL license liability clause",
            required_document_ids=["license_gpl3"],
            required_facts=["GPL has similar warranty disclaimer"],
        ),
        HopDescription(
            step=3,
            reasoning="Compare the two",
            required_document_ids=["license_mit", "license_gpl3"],
            required_facts=["Both disclaim warranties, but GPL has additional provisions"],
        ),
    ],
)
```

### Zero-Result Query Example

```python
ZeroResultQuery(
    query_id="zr_001",
    question="What does the GDPR say about quantum computing regulations?",
    query_type=QueryType.ZERO_RESULT,
    category=ZeroResultCategory.OUT_OF_DOMAIN,
    target_category=DocumentCategory.LEGAL,
    acceptable_responses=[
        "The GDPR does not address quantum computing",
        "I don't have information about quantum computing in GDPR",
    ],
    unacceptable_responses=[
        "The GDPR requires quantum-safe encryption",  # Hallucination
        "Article 32 covers quantum computing",  # False claim
    ],
    max_confidence_threshold=0.3,
)
```

---

## Testing Requirements

### Unit Tests
```python
def test_multi_hop_query_validation():
    """Multi-hop query must have at least 2 hops."""
    with pytest.raises(ValueError):
        MultiHopQuery(
            ...,
            hops=[HopDescription(step=1, ...)],  # Only 1 hop
            minimum_documents_required=2,
        )

def test_query_suite_serialization(tmp_path):
    """Query suite saves and loads correctly."""
    suite = QuerySuite(queries=[...])
    path = tmp_path / "queries.json"
    suite.save(path)
    loaded = QuerySuite.load(path)
    assert len(loaded.queries) == len(suite.queries)

def test_query_filtering():
    """Filter queries by type and difficulty."""
    suite = QuerySuite(queries=[...])
    hard_queries = suite.get_by_difficulty(QueryDifficulty.HARD)
    legal_queries = suite.get_by_category(DocumentCategory.LEGAL)
    multi_hop = suite.get_by_type(QueryType.MULTI_HOP_BRIDGE)
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/queries.py`
- Add tests to `tests/benchmarks/test_query_suite.py`

---

## Definition of Done

- [ ] All query types modeled
- [ ] Multi-hop queries track reasoning chain
- [ ] Zero-result queries specify acceptable/unacceptable responses
- [ ] Adversarial queries track distractors
- [ ] QuerySuite save/load works
- [ ] Filtering by type, difficulty, category
- [ ] Unit tests cover all query types
