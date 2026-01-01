# TASK-001: Implement Seed Factory and Base Classes

**Phase**: 1 (Core Framework)
**Priority**: P0 (Blocking for all other tasks)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None

## Description

Implement the core seed factory system with declarative seed data creation, dependency resolution via topological sort, and parallel-safe instance-based registry.

This is the foundation for all test data creation in FR-009.

## Acceptance Criteria

- [ ] `SeedItem` abstract base class created with `create()` method
- [ ] `SeedFactory` class with instance-based registry (parallel-safe)
- [ ] `@SeedFactory.register()` decorator for global seed registration
- [ ] `SeedSet` dataclass for grouping seeds
- [ ] Topological sort implementation for dependency resolution
- [ ] `CircularDependencyError` exception for cyclic dependencies
- [ ] Rollback mechanism when seed creation fails mid-process
- [ ] Unit tests: dependency resolution, circular dependency detection
- [ ] Unit tests: instance isolation (parallel safety)

## Technical Notes

**Critical Design Decision: Instance-Based Registry**

```python
class SeedFactory:
    # Global registry for decorator-registered classes
    _global_seed_classes: ClassVar[dict[str, type[SeedItem]]] = {}

    def __init__(self):
        # Instance registry - each test gets its own
        self._registry: dict[str, SeedItem] = {}
        # Copy global seeds
        for seed_id, seed_class in self._global_seed_classes.items():
            self._registry[seed_id] = seed_class()
```

**Why?** Parallel test execution requires isolated registries to avoid seed ID collisions and state leakage.

**Key Methods:**
- `@classmethod register(seed_id)` - Decorator for global registration
- `register_instance(seed_id, seed)` - Per-test registration
- `_topological_sort(seed_ids)` - Dependency ordering (Kahn's algorithm)
- `create_all(seed_ids, provider)` - Execute seeds with rollback

**Rollback Logic:**
```python
# On failure, delete all created memories in reverse order
except Exception as e:
    for memory_id in reversed(created_memory_ids):
        await provider.delete(memory_id)
    raise
```

## Testing Requirements

### Unit Tests (`tests/framework/test_seeds.py`)

1. **Dependency Resolution**
   ```python
   async def test_dependency_resolution():
       factory = SeedFactory()
       # Register parent → child dependency
       # Verify sorted_ids = ["parent", "child"]
   ```

2. **Circular Dependency Detection**
   ```python
   async def test_circular_dependency_error():
       # A depends on B, B depends on A
       # Verify CircularDependencyError raised
   ```

3. **Parallel Safety**
   ```python
   async def test_instance_isolation():
       factory1 = SeedFactory()
       factory2 = SeedFactory()
       factory1.register_instance("test", TestSeed())
       # Verify factory2 doesn't see "test"
   ```

4. **Rollback on Failure**
   ```python
   async def test_rollback_on_failure(memory_provider):
       # Create 2 seeds, 3rd fails
       # Verify first 2 are deleted (rolled back)
   ```

## Files to Create

- `src/draagon_ai/testing/seeds.py` - NEW
  - `SeedItem` (ABC)
  - `SeedFactory`
  - `SeedSet`
  - `CircularDependencyError`

- `tests/framework/test_seeds.py` - NEW
  - Test dependency resolution
  - Test circular dependencies
  - Test instance isolation
  - Test rollback

## Implementation Sequence

1. Define `CircularDependencyError` exception
2. Implement `SeedItem` ABC with `create()` signature
3. Implement `SeedSet` dataclass
4. Implement `SeedFactory`:
   - Instance-based registry
   - Global class registry
   - `@register()` decorator
   - `_topological_sort()` (Kahn's algorithm)
   - `create_all()` with rollback
5. Write unit tests (no real MemoryProvider needed - use mocks)
6. Verify parallel safety with concurrent test runs

## Cognitive Testing Requirements

N/A - This is infrastructure only, no cognitive components.

## Success Criteria

- All unit tests pass
- Can register seeds globally and per-instance
- Dependency graph resolution works for complex cases
- Circular dependencies are detected and rejected
- Rollback works when creation fails
- Two factory instances don't interfere (parallel-safe)
