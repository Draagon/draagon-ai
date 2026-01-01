# TASK-003: Implement Seed Applicator Fixture

**Phase**: 1 (Core Framework)
**Priority**: P0 (Required for writing tests)
**Effort**: 0.5 days
**Status**: Pending
**Dependencies**: TASK-001, TASK-002

## Description

Implement the `seed` fixture that combines `SeedFactory` and `MemoryProvider` for easy test data application. This is the primary API for applying seeds in tests.

## Acceptance Criteria

- [ ] `seed_factory` fixture creates fresh `SeedFactory` per test
- [ ] `seed` fixture combines factory + provider
- [ ] `SeedApplicator` helper class for clean API
- [ ] Seed applicator supports optional provider override
- [ ] Unit test: seed fixture applies seeds correctly
- [ ] Unit test: multiple tests get isolated factories

## Technical Notes

**SeedApplicator Pattern:**

```python
@pytest.fixture
def seed_factory() -> SeedFactory:
    """Create fresh seed factory for each test.

    Each test gets its own factory instance with copies of global seeds.
    This enables parallel test execution without registry conflicts.
    """
    return SeedFactory()  # Fresh instance, copies global seeds


@pytest.fixture
async def seed(memory_provider: MemoryProvider, seed_factory: SeedFactory):
    """Seed applicator with REAL MemoryProvider.

    Usage in test:
        async def test_recall_pets(seed, memory_provider):
            await seed.apply(USER_WITH_PETS, memory_provider)
            # Test continues with seeded data
    """
    class SeedApplicator:
        def __init__(self, provider: MemoryProvider, factory: SeedFactory):
            self.provider = provider
            self.factory = factory

        async def apply(self, seed_set, provider: MemoryProvider | None = None):
            # Use provided provider or default
            p = provider or self.provider
            return await seed_set.apply(p)

    return SeedApplicator(memory_provider, seed_factory)
```

**Why SeedApplicator?**
- Provides clean API: `await seed.apply(SEED_SET)`
- Encapsulates provider management
- Allows provider override for advanced cases

## Testing Requirements

### Unit Tests (`tests/framework/test_fixtures.py`)

1. **Fresh Factory Per Test**
   ```python
   def test_seed_factory_is_fresh(seed_factory):
       seed_factory.register_instance("test1", TestSeed())
       # In another test, "test1" should not exist

   def test_seed_factory_has_global_seeds(seed_factory):
       # Verify global seeds are copied to instance
   ```

2. **Seed Applicator Works**
   ```python
   async def test_seed_apply(seed, memory_provider):
       @SeedFactory.register("test_seed")
       class TestSeed(SeedItem):
           async def create(self, provider):
               return await provider.store(...)

       SEED_SET = SeedSet("test", ["test_seed"])
       results = await seed.apply(SEED_SET)
       assert "test_seed" in results
   ```

3. **Provider Override**
   ```python
   async def test_seed_apply_with_override(seed):
       other_provider = Mock(MemoryProvider)
       await seed.apply(SEED_SET, provider=other_provider)
       # Verify other_provider was used, not default
   ```

## Files to Modify

- `src/draagon_ai/testing/fixtures.py` - EXTEND
  - Add `seed_factory` fixture
  - Add `seed` fixture with `SeedApplicator`

- `tests/framework/test_fixtures.py` - EXTEND
  - Test factory isolation
  - Test applicator functionality

## Implementation Sequence

1. Add `seed_factory` fixture to `fixtures.py`
2. Implement `SeedApplicator` helper class
3. Add `seed` fixture with applicator
4. Write unit tests for isolation
5. Write unit tests for applicator
6. Test with real seeds from TASK-001

## Cognitive Testing Requirements

N/A - This is infrastructure only, no cognitive components.

## Success Criteria

- Each test gets isolated `SeedFactory` instance
- `seed.apply(SEED_SET)` works cleanly in tests
- Provider can be overridden if needed
- All unit tests pass
- Ready for use in integration tests
