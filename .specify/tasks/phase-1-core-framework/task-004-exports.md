# TASK-004: Update Testing Module Exports

**Phase**: 1 (Core Framework)
**Priority**: P1 (Required for discoverability)
**Effort**: 0.25 days
**Status**: Pending
**Dependencies**: TASK-001, TASK-002, TASK-003

## Description

Update `src/draagon_ai/testing/__init__.py` to export all new FR-009 classes and fixtures for easy imports.

## Acceptance Criteria

- [ ] All new classes exported from `__init__.py`
- [ ] `__all__` list updated
- [ ] Import test: can import all classes from `draagon_ai.testing`
- [ ] No circular import issues

## Technical Notes

**Exports to Add:**

```python
# src/draagon_ai/testing/__init__.py

# Existing exports (keep these)
from .cache import UnifiedCacheManager, CacheMode, CacheConfig
from .mocks import LLMMock, HTTPMock, EmbeddingMock, ServiceMock
from .modes import TestMode, TestContext, test_mode, get_test_context
from .fixtures import *  # Existing fixtures

# NEW: FR-009 exports
from .seeds import (
    SeedItem,
    SeedFactory,
    SeedSet,
    CircularDependencyError,
)
from .database import (
    TestDatabase,
    Neo4jMemoryConfig,
)

__all__ = [
    # Existing...
    "UnifiedCacheManager", "CacheMode", "CacheConfig",
    "LLMMock", "HTTPMock", "EmbeddingMock", "ServiceMock",
    "TestMode", "TestContext", "test_mode", "get_test_context",

    # NEW: FR-009
    "SeedItem", "SeedFactory", "SeedSet", "CircularDependencyError",
    "TestDatabase", "Neo4jMemoryConfig",
]
```

## Testing Requirements

### Import Test (`tests/framework/test_imports.py`)

```python
def test_can_import_seed_classes():
    from draagon_ai.testing import (
        SeedItem,
        SeedFactory,
        SeedSet,
        CircularDependencyError,
    )
    assert SeedItem is not None

def test_can_import_database_classes():
    from draagon_ai.testing import TestDatabase, Neo4jMemoryConfig
    assert TestDatabase is not None

def test_all_exports_defined():
    import draagon_ai.testing as testing
    # Verify all __all__ items are actually defined
    for name in testing.__all__:
        assert hasattr(testing, name)
```

## Files to Modify

- `src/draagon_ai/testing/__init__.py` - EXTEND
  - Add FR-009 imports
  - Update `__all__` list

- `tests/framework/test_imports.py` - NEW
  - Test all imports work
  - Test no circular imports

## Implementation Sequence

1. Add FR-009 imports to `__init__.py`
2. Update `__all__` list
3. Write import tests
4. Run tests to verify no circular imports
5. Verify IDE autocomplete works

## Cognitive Testing Requirements

N/A - This is infrastructure only, no cognitive components.

## Success Criteria

- Can import all FR-009 classes from `draagon_ai.testing`
- No circular import errors
- IDE autocomplete shows new classes
- Import tests pass
