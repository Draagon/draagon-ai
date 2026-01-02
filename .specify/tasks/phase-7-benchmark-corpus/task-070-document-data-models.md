# TASK-070: Document Data Models for Benchmark Corpus

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Critical path - blocks all corpus assembly)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None (foundational task)

---

## Description

Implement core data models for benchmark document corpus:
- `BenchmarkDocument`: Single document with metadata, content hash, semantic tags
- `DocumentCorpus`: Collection of documents with save/load, filtering, querying
- `CorpusMetadata`: Statistics about corpus (source/domain distribution, size stats)
- Enums: `DocumentSource` (local/online/synthetic), `DocumentDomain` (python/java/ai/etc.)

**Location:** `src/draagon_ai/testing/benchmarks/corpus.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `BenchmarkDocument` dataclass with all fields (doc_id, source, domain, content, etc.)
- [ ] Content hash automatically calculated in `__post_init__` (SHA256, 16 chars)
- [ ] Size in bytes automatically calculated
- [ ] `to_dict()` and `from_dict()` for JSON serialization
- [ ] Round-trip serialization preserves all fields exactly

### DocumentCorpus
- [ ] `DocumentCorpus` holds list of documents + metadata
- [ ] `save(path)` writes corpus to JSON file
- [ ] `load(path)` reads corpus from JSON file
- [ ] `get_document(doc_id)` retrieves by ID
- [ ] `get_by_domain(domain)` filters by domain
- [ ] `get_distractors(domain, count)` returns distractor subset
- [ ] `__len__()` returns document count

### CorpusMetadata
- [ ] Tracks total documents, source/domain distributions
- [ ] Calculates size distribution (min/max/mean/median)
- [ ] Tracks distractor count and ratio
- [ ] `to_dict()` serialization for JSON export

### Data Integrity
- [ ] Content hash is deterministic (same content = same hash)
- [ ] Deduplication works via content hash comparison
- [ ] Empty corpus handles gracefully (no division by zero)
- [ ] Large documents (500KB) serialize without truncation

---

## Technical Notes

### Content Hashing Strategy
```python
import hashlib

content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
```
- SHA256 for collision resistance
- First 16 hex chars for brevity
- Deterministic for deduplication

### JSON Serialization
- Enums serialize as `.value` (string form)
- Enums deserialize via `DocumentSource(data["source"])`
- `metadata` field is flexible dict (source-specific data)

### Document Categories (NEW)

The corpus now tracks 8 content categories for diversity:

```python
class DocumentCategory(str, Enum):
    TECHNICAL = "technical"        # Code, API docs, READMEs
    NARRATIVE = "narrative"        # Stories, fiction, Wikipedia articles
    KNOWLEDGE_BASE = "knowledge_base"  # FAQs, Stack Overflow, how-to
    LEGAL = "legal"                # ToS, contracts, court opinions, regulations
    CONVERSATIONAL = "conversational"  # Chat, email, support tickets
    ACADEMIC = "academic"          # arXiv, research papers, scientific articles
    NEWS_BLOG = "news_blog"        # Blog posts, news articles
    SYNTHETIC = "synthetic"        # LLM-generated distractors
```

### Category + Domain Inference

Two-level classification:
1. **Category**: Broad content type (legal, narrative, technical, etc.)
2. **Domain**: Specific topic within category (contract_law, fiction, python, etc.)

Heuristics for initial implementation:
- `"draagon"` in path → TECHNICAL, domain=`ai_framework`
- `"party-lore"` in path → NARRATIVE, domain=`game_fiction`
- `"contract"` or `"agreement"` in content → LEGAL, domain=`contracts`
- `"terms of service"` in content → LEGAL, domain=`tos`
- `.py` extension → TECHNICAL, domain=`python`
- `.java` extension → TECHNICAL, domain=`java`
- arXiv URL → ACADEMIC, domain=`research`
- Court opinion structure → LEGAL, domain=`case_law`

Can be refined with LLM classification later.

---

## Testing Requirements

### Unit Tests (`tests/benchmarks/test_corpus_builder.py`)

```python
def test_document_content_hash():
    """Content hash is deterministic for same content."""
    doc1 = BenchmarkDocument(...)
    doc2 = BenchmarkDocument(...)  # Same content
    assert doc1.content_hash == doc2.content_hash

def test_corpus_save_load_roundtrip(tmp_path):
    """Corpus serialization preserves all fields."""
    corpus = DocumentCorpus(documents=[...], metadata=...)
    path = tmp_path / "corpus.json"
    corpus.save(path)
    loaded = DocumentCorpus.load(path)
    assert loaded.documents[0].content == original_content

def test_corpus_filtering():
    """Filter documents by domain and distractor status."""
    corpus = DocumentCorpus(...)
    python_docs = corpus.get_by_domain(DocumentDomain.PYTHON)
    distractors = corpus.get_distractors(count=10)
    assert all(doc.is_distractor for doc in distractors)

def test_empty_corpus_metadata():
    """Empty corpus doesn't crash on metadata calculation."""
    metadata = CorpusMetadata(
        ...,
        distractor_ratio=0.0 if not documents else distractor_count / len(documents)
    )
```

### Edge Cases to Test
- [ ] Empty corpus (0 documents)
- [ ] Single document corpus
- [ ] Very large document (500KB+)
- [ ] Unicode content (emoji, non-ASCII)
- [ ] Duplicate content (same hash)
- [ ] Missing optional fields (chunk_ids, semantic_tags)

---

## Files to Create/Modify

### New Files
- `src/draagon_ai/testing/benchmarks/__init__.py` (module init, exports)
- `src/draagon_ai/testing/benchmarks/corpus.py` (main implementation)

### Test Files
- `tests/benchmarks/__init__.py`
- `tests/benchmarks/conftest.py` (fixtures)
- `tests/benchmarks/test_corpus_builder.py` (unit tests)

---

## Integration Points

**None** - This is a foundational task. Future tasks depend on these models:
- TASK-071: LocalDocumentScanner produces `BenchmarkDocument` instances
- TASK-072: OnlineDocumentationFetcher produces `BenchmarkDocument` instances
- TASK-073: DistractorGenerator produces `BenchmarkDocument` instances
- TASK-074: CorpusBuilder aggregates into `DocumentCorpus`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (100% coverage on new code)
- [ ] Round-trip serialization verified
- [ ] Edge cases tested (empty, large, unicode)
- [ ] Code reviewed for data integrity
- [ ] Documentation complete (docstrings, type hints)
