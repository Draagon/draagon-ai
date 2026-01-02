# TASK-075: CorpusBuilder Orchestrator

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Critical - assembles final corpus)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-070, TASK-071, TASK-072, TASK-073, TASK-074

---

## Description

Implement the orchestrator that combines all document sources into a unified corpus:
- Coordinate local scanner, online fetcher, legal fetcher, distractor generator
- Enforce category distribution targets
- Calculate and validate corpus metadata
- Save/load corpus with versioning

**Location:** `src/draagon_ai/testing/benchmarks/corpus.py` (add to existing)

---

## Acceptance Criteria

### Orchestration
- [ ] `CorpusBuilder` class with target_size, cache_dir
- [ ] Methods for each source: `add_local_documents()`, `add_online_content()`, etc.
- [ ] `build()` returns `DocumentCorpus` with all documents and metadata
- [ ] Category tracking during build process

### Category Distribution Enforcement
- [ ] Track documents per category during build
- [ ] Warn if category is under/over target
- [ ] Final validation: all 8 categories represented
- [ ] Tolerance: ±10% of target per category

### Metadata Calculation
- [ ] `CorpusMetadata` with category distribution
- [ ] Size statistics (min, max, mean, median)
- [ ] Source distribution (local, online, synthetic)
- [ ] Distractor count and ratio
- [ ] Creation timestamp and version

### Persistence
- [ ] `corpus.save(path)` writes JSON
- [ ] `DocumentCorpus.load(path)` reads JSON
- [ ] Version tracking (v1, v2, etc.)
- [ ] Incremental updates (add to existing corpus)

---

## Technical Notes

### Target Distribution

```python
CATEGORY_TARGETS = {
    DocumentCategory.TECHNICAL: 125,
    DocumentCategory.NARRATIVE: 75,
    DocumentCategory.KNOWLEDGE_BASE: 75,
    DocumentCategory.LEGAL: 50,
    DocumentCategory.CONVERSATIONAL: 50,
    DocumentCategory.ACADEMIC: 50,
    DocumentCategory.NEWS_BLOG: 50,
    DocumentCategory.SYNTHETIC: 25,
}
```

### Build Process

```python
class CorpusBuilder:
    def __init__(self, target_size: int = 500, cache_dir: Path):
        self.target_size = target_size
        self.cache_dir = cache_dir
        self.documents: list[BenchmarkDocument] = []
        self._category_counts: dict[DocumentCategory, int] = defaultdict(int)

    async def add_local_documents(self, category: DocumentCategory, ...):
        scanner = LocalDocumentScanner(...)
        docs = await scanner.scan(max_docs=...)
        for doc in docs:
            doc.category = category
            self.documents.append(doc)
            self._category_counts[category] += 1

    async def build(self) -> DocumentCorpus:
        # Validate distribution
        self._validate_distribution()

        # Calculate metadata
        metadata = self._calculate_metadata()

        return DocumentCorpus(
            documents=self.documents,
            metadata=metadata,
            version="v1",
        )

    def _validate_distribution(self):
        for category, target in CATEGORY_TARGETS.items():
            actual = self._category_counts[category]
            tolerance = target * 0.1  # 10%
            if abs(actual - target) > tolerance:
                logger.warning(
                    f"{category.value}: {actual} docs (target: {target}, diff: {actual-target})"
                )
```

### Corpus Metadata

```python
def _calculate_metadata(self) -> CorpusMetadata:
    from datetime import datetime
    from statistics import median

    sizes = [doc.size_bytes for doc in self.documents]

    return CorpusMetadata(
        version="v1",
        created_at=datetime.now().isoformat(),
        total_documents=len(self.documents),
        category_distribution=dict(self._category_counts),
        source_distribution=self._count_by_source(),
        size_distribution={
            "min": min(sizes),
            "max": max(sizes),
            "mean": sum(sizes) // len(sizes),
            "median": int(median(sizes)),
        },
        distractor_count=sum(1 for d in self.documents if d.is_distractor),
        distractor_ratio=...,
    )
```

---

## Testing Requirements

### Unit Tests
```python
@pytest.mark.asyncio
async def test_corpus_builder_integration(tmp_path):
    """Build corpus from multiple sources."""
    builder = CorpusBuilder(target_size=10, cache_dir=tmp_path)

    # Add mock documents
    await builder.add_documents([
        BenchmarkDocument(..., category=DocumentCategory.TECHNICAL),
        BenchmarkDocument(..., category=DocumentCategory.LEGAL),
    ])

    corpus = await builder.build()

    assert len(corpus) == 2
    assert corpus.metadata.total_documents == 2

@pytest.mark.asyncio
async def test_category_distribution_warning(tmp_path, caplog):
    """Warn when category is off-target."""
    builder = CorpusBuilder(target_size=500, cache_dir=tmp_path)

    # Add only technical docs (missing other categories)
    await builder.add_documents([
        BenchmarkDocument(..., category=DocumentCategory.TECHNICAL)
        for _ in range(50)
    ])

    corpus = await builder.build()

    assert "LEGAL" in caplog.text  # Warning about missing legal docs
```

### Integration Test
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_corpus_build():
    """Build real corpus with all sources."""
    builder = CorpusBuilder(target_size=50, cache_dir=Path("/tmp/test_corpus"))

    await builder.add_local_documents(
        root_path=Path.home() / "Development",
        category=DocumentCategory.TECHNICAL,
        max_docs=20,
    )

    await builder.add_synthetic_distractors(
        categories=[DocumentCategory.TECHNICAL],
        count=5,
    )

    corpus = await builder.build()

    assert len(corpus) >= 20
    assert corpus.metadata.distractor_count == 5
```

---

## Files to Create/Modify

- `src/draagon_ai/testing/benchmarks/corpus.py` (add CorpusBuilder class)
- Add tests to `tests/benchmarks/test_corpus_builder.py`

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] CorpusBuilder orchestrates all sources
- [ ] Category distribution tracked and validated
- [ ] Metadata calculated correctly
- [ ] Save/load round-trip works
- [ ] Integration test builds real corpus
- [ ] Warnings for off-target categories
