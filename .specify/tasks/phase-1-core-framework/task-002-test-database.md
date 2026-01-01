# TASK-002: Implement Neo4j Test Database Lifecycle Manager

**Phase**: 1 (Core Framework)
**Priority**: P0 (Blocking for all integration tests)
**Effort**: 1 day
**Status**: Pending
**Dependencies**: None

## Description

Implement lifecycle-only test database manager for Neo4j isolation. Does NOT wrap MemoryProvider - only manages connection, cleanup, and configuration.

**Critical Design:** TestDatabase is lifecycle-only. Tests use REAL MemoryProvider directly.

## Acceptance Criteria

- [ ] `TestDatabase` class with lifecycle methods only
- [ ] `initialize()` - Create test database if needed
- [ ] `clear()` - Delete all nodes/relationships
- [ ] `close()` - Close driver connection
- [ ] `get_config()` - Return config for creating real providers
- [ ] `verify_connection()` - Health check with helpful errors
- [ ] `Neo4jMemoryConfig` dataclass for provider configuration
- [ ] Session-scoped fixture `test_database`
- [ ] Function-scoped fixture `clean_database` (clear before, not after)
- [ ] Function-scoped fixture `memory_provider` (real provider)
- [ ] Unit tests: database creation, cleanup, config
- [ ] Integration test: verify connection error messages

## Technical Notes

**Lifecycle-Only Design:**

```python
@dataclass
class TestDatabase:
    """Neo4j test database lifecycle manager ONLY.

    Does NOT wrap or provide helper methods.
    Tests use the REAL MemoryProvider directly.
    """

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "draagon-ai-2025"
    database_name: str = "neo4j_test"

    async def initialize(self) -> None: ...
    async def clear(self) -> None: ...
    async def close(self) -> None: ...
    def get_config(self) -> Neo4jMemoryConfig: ...
    async def verify_connection(self) -> None: ...
```

**Why lifecycle-only?**
1. No leaky abstractions
2. Tests validate production APIs
3. Real bugs surface in tests

**Cleanup Policy:**
- Clear BEFORE each test (clean slate)
- Do NOT clear AFTER test (preserve for debugging)
- Next test clears before running anyway

**Error Handling:**
```python
async def verify_connection(self):
    try:
        await session.run("RETURN 1")
    except Exception as e:
        raise RuntimeError(
            f"Neo4j connection failed.\n"
            f"URI: {self.neo4j_uri}\n"
            f"Database: {self.database_name}\n"
            f"Is Neo4j running? Try: docker ps | grep neo4j\n"
            f"Error: {e}"
        )
```

## Testing Requirements

### Unit Tests (`tests/framework/test_database.py`)

1. **Database Creation**
   ```python
   async def test_initialize_creates_database():
       db = TestDatabase()
       await db.initialize()
       # Verify database exists in Neo4j system
   ```

2. **Cleanup**
   ```python
   async def test_clear_removes_all_data(test_database):
       # Add some nodes
       # Call clear()
       # Verify no nodes remain
   ```

3. **Config Returns Correct Values**
   ```python
   def test_get_config():
       db = TestDatabase()
       config = db.get_config()
       assert config.uri == db.neo4j_uri
       assert config.database == db.database_name
   ```

4. **Helpful Error Messages**
   ```python
   async def test_verify_connection_error_message():
       db = TestDatabase(neo4j_uri="bolt://invalid:9999")
       await db.initialize()
       with pytest.raises(RuntimeError, match="Is Neo4j running"):
           await db.verify_connection()
   ```

### Fixture Tests (`tests/framework/test_fixtures.py`)

1. **Session Scope**
   ```python
   def test_test_database_is_session_scoped():
       # Verify fixture scope
   ```

2. **Clean Database Clears Before**
   ```python
   async def test_clean_database_clears_before(clean_database):
       # Verify starts empty
   ```

3. **Memory Provider Uses Real API**
   ```python
   async def test_memory_provider_is_real(memory_provider):
       memory_id = await memory_provider.store(...)
       assert memory_id  # Real ID from Neo4j
   ```

## Files to Create

- `src/draagon_ai/testing/database.py` - NEW
  - `Neo4jMemoryConfig` dataclass
  - `TestDatabase` class

- `src/draagon_ai/testing/fixtures.py` - EXTEND
  - `test_database` fixture (session)
  - `clean_database` fixture (function)
  - `memory_provider` fixture (function)
  - `preserved_database` fixture (debugging)

- `tests/framework/test_database.py` - NEW
  - Test lifecycle methods
  - Test error handling

- `tests/framework/test_fixtures.py` - NEW
  - Test fixture scopes
  - Test cleanup policy

## Implementation Sequence

1. Define `Neo4jMemoryConfig` dataclass
2. Implement `TestDatabase.__init__()` with defaults
3. Implement `initialize()` - create database if needed
4. Implement `clear()` - MATCH (n) DETACH DELETE n
5. Implement `close()` - close driver
6. Implement `get_config()` - return config dict
7. Implement `verify_connection()` - health check with helpful errors
8. Add fixtures to `fixtures.py`:
   - `test_database` (session-scoped)
   - `clean_database` (function-scoped, clears before)
   - `memory_provider` (function-scoped, real provider)
   - `preserved_database` (debugging only)
9. Write unit tests
10. Test cleanup policy manually

## Cognitive Testing Requirements

N/A - This is infrastructure only, no cognitive components.

## Success Criteria

- TestDatabase only manages lifecycle (no helper methods)
- Tests get REAL MemoryProvider via fixture
- Clear before test, preserve after for debugging
- Connection errors are helpful ("Is Neo4j running?")
- All unit tests pass
- Integration with real Neo4j works
