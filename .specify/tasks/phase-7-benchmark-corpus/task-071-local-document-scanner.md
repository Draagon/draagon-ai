# TASK-071: Local Document Scanner

**Phase**: 7 (Production-Grade Retrieval Benchmark)
**Priority**: P0 (Critical - needed for 300+ corpus documents)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: TASK-070 (Document data models)

---

## Description

Implement filesystem scanner to collect 300+ documents from `~/Development`:
- Scan with glob patterns (`**/*.md`, `**/*.py`, `**/*.java`, `**/*.ts`)
- Filter by size range (1KB-500KB)
- Exclude patterns (node_modules, target, .archive, __pycache__)
- Deduplicate by content hash
- Infer domain from file path heuristics
- Extract semantic tags from content

**Location:** `src/draagon_ai/testing/benchmarks/downloaders/local_scanner.py`

---

## Acceptance Criteria

### Core Functionality
- [ ] `LocalDocumentScanner` class initialized with root_path, patterns, size_range, exclude_patterns
- [ ] `scan(max_docs)` returns list of `BenchmarkDocument` instances
- [ ] Glob patterns work recursively (`**/*.md` finds nested files)
- [ ] Size filtering: Only files between min_size and max_size bytes
- [ ] Exclusion patterns filter out node_modules, target, .archive, build, __pycache__
- [ ] Deduplication: Same content (by hash) appears only once

### Domain Inference
- [ ] `_infer_domain(file_path)` returns `DocumentDomain` enum
- [ ] Heuristics: "draagon" → AI_FRAMEWORK, "party-lore" → GAME_DESIGN
- [ ] Extension mapping: `.py` → PYTHON, `.java` → JAVA, `.ts/.tsx` → TYPESCRIPT
- [ ] Default: UNKNOWN for unrecognized paths

### Semantic Tagging
- [ ] `_extract_tags(file_path, content)` returns list of keywords
- [ ] File extension added as tag (e.g., "md", "py")
- [ ] Common keywords detected in first 500 chars (async, class, function, test, api)
- [ ] Tags deduplicated (set conversion)

### Error Handling
- [ ] Unicode decode errors skip file with warning log
- [ ] Permission errors skip file with warning log
- [ ] Invalid paths handled gracefully
- [ ] Empty directories don't crash scanner

### Performance
- [ ] Processes 1000 files in < 30 seconds (on SSD)
- [ ] Logs progress every 100 files scanned
- [ ] Early termination when max_docs reached

---

## Technical Notes

### Glob Pattern Matching
```python
for pattern in self.patterns:
    for file_path in self.root_path.glob(pattern):
        # Process file...
```

Use `pathlib.Path.glob()` for cross-platform compatibility.

### Exclusion Pattern Matching
```python
def _is_excluded(self, file_path: Path) -> bool:
    for pattern in self.exclude_patterns:
        if file_path.match(pattern):
            return True
    return False
```

### Document ID Generation
```python
def _generate_doc_id(self, file_path: Path) -> str:
    rel_path = file_path.relative_to(self.root_path)
    doc_id = str(rel_path).replace("/", "_").replace("\\", "_")
    doc_id = doc_id.rsplit(".", 1)[0]  # Remove extension
    return doc_id
```

Example: `~/Development/draagon-ai/CLAUDE.md` → `draagon-ai_CLAUDE`

### Content Hash Deduplication
```python
seen_hashes = set()
content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
if content_hash in seen_hashes:
    logger.debug(f"Skipping duplicate: {file_path}")
    continue
seen_hashes.add(content_hash)
```

---

## Testing Requirements

### Unit Tests (`tests/benchmarks/test_corpus_builder.py`)

```python
@pytest.mark.asyncio
async def test_local_scanner_finds_documents(tmp_path):
    """Scanner finds matching documents."""
    (tmp_path / "test1.md").write_text("# Test")
    (tmp_path / "test2.py").write_text("def test(): pass")
    (tmp_path / "too_small.md").write_text("x")  # Too small

    scanner = LocalDocumentScanner(
        root_path=tmp_path,
        patterns=["**/*.md", "**/*.py"],
        size_range=(10, 500_000),
    )

    docs = await scanner.scan(max_docs=100)
    assert len(docs) == 2  # test1.md, test2.py

@pytest.mark.asyncio
async def test_local_scanner_excludes_patterns(tmp_path):
    """Scanner excludes node_modules, target, etc."""
    (tmp_path / "good.md").write_text("# Good")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "bad.md").write_text("# Should exclude")

    scanner = LocalDocumentScanner(root_path=tmp_path, patterns=["**/*.md"])
    docs = await scanner.scan(max_docs=100)

    assert len(docs) == 1
    assert not any("node_modules" in doc.file_path for doc in docs)

@pytest.mark.asyncio
async def test_local_scanner_deduplicates(tmp_path):
    """Scanner removes duplicate content."""
    (tmp_path / "test1.md").write_text("Same content")
    (tmp_path / "test2.md").write_text("Same content")

    scanner = LocalDocumentScanner(root_path=tmp_path, patterns=["**/*.md"])
    docs = await scanner.scan(max_docs=100)

    assert len(docs) == 1  # Deduplicated

@pytest.mark.asyncio
async def test_domain_inference(tmp_path):
    """Infer domain from file path."""
    (tmp_path / "draagon-ai").mkdir()
    (tmp_path / "draagon-ai" / "test.py").write_text("# AI code")

    scanner = LocalDocumentScanner(root_path=tmp_path, patterns=["**/*.py"])
    docs = await scanner.scan(max_docs=100)

    assert docs[0].domain == DocumentDomain.AI_FRAMEWORK
```

### Integration Test (Real Filesystem)
```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_scan_real_development_directory():
    """Scan real ~/Development directory (slow test)."""
    scanner = LocalDocumentScanner(
        root_path=Path.home() / "Development",
        patterns=["**/*.md"],
        size_range=(1024, 500_000),
    )

    docs = await scanner.scan(max_docs=50)

    assert len(docs) > 0
    assert all(doc.source == DocumentSource.LOCAL for doc in docs)
    assert all(1024 <= doc.size_bytes <= 500_000 for doc in docs)
```

---

## Files to Create/Modify

### New Files
- `src/draagon_ai/testing/benchmarks/downloaders/__init__.py`
- `src/draagon_ai/testing/benchmarks/downloaders/local_scanner.py`

### Test Files
- Add tests to `tests/benchmarks/test_corpus_builder.py`

---

## Integration Points

**Upstream:**
- TASK-070: Uses `BenchmarkDocument`, `DocumentSource`, `DocumentDomain` models

**Downstream:**
- TASK-074: CorpusBuilder calls `LocalDocumentScanner.scan()` to add local docs

---

## Validation Checklist

### Before Committing
- [ ] Scans ~/Development and finds 100+ markdown files
- [ ] Excludes node_modules, target, .archive automatically
- [ ] Deduplicates identical files (e.g., copied READMEs)
- [ ] Infers domain correctly for draagon-ai, party-lore, metaobjects-core
- [ ] Handles Unicode content (emoji, non-ASCII) without errors
- [ ] Logs progress and statistics

### Performance Validation
- [ ] Scan 1000 files in < 30 seconds
- [ ] Memory usage stays reasonable (< 500MB for 1000 docs)
- [ ] Early termination works when max_docs reached

---

## Definition of Done

- [ ] All acceptance criteria met
- [ ] Unit tests passing (90%+ coverage)
- [ ] Integration test validates real ~/Development scan
- [ ] Edge cases tested (empty dirs, unicode, permissions)
- [ ] Performance validated (1000 files < 30s)
- [ ] Code reviewed for filesystem safety
- [ ] Documentation complete (docstrings, examples)
