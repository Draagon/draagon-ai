# Draagon AI Testing Framework

A comprehensive testing infrastructure for AI agent applications built on draagon-ai.

## Overview

The testing framework provides:
- **Multi-layer caching** - LLM responses, HTTP requests, embeddings
- **Service virtualization** - Mock external services in test mode
- **Test mode detection** - Automatic behavior switching based on context
- **Reusable fixtures** - Common pytest fixtures for all apps

## Quick Start

### 1. Configure Your App's `conftest.py`

```python
# tests/conftest.py
import pytest
from draagon_ai.testing import (
    TestMode, test_mode, CacheConfig, CacheMode,
    ServiceMock, get_mock, is_test_mode
)
from draagon_ai.testing.fixtures import (
    cache_config,
    configure_cache,
    llm_mock,
    embedding_mock,
    pytest_configure,
    pytest_collection_modifyitems,
)

# Re-export framework fixtures
__all__ = [
    "cache_config",
    "configure_cache",
    "llm_mock",
    "embedding_mock",
]

# Create your app-specific mocks
@pytest.fixture
def api_mock():
    mock = ServiceMock()
    mock.add_response({"status": "ok"}, pattern=r"/api/status")
    return mock

@pytest.fixture
def unit_test_mode(llm_mock, embedding_mock, api_mock):
    """Full unit test mode with all mocks."""
    with test_mode(
        mode=TestMode.UNIT,
        mocks={
            "llm": llm_mock,
            "embedding": embedding_mock,
            "api": api_mock,
        }
    ) as ctx:
        yield ctx
```

### 2. Integrate Mocks Into Your Services

```python
# src/my_app/services/api_client.py
from draagon_ai.testing import get_mock

class APIClient:
    async def get_status(self) -> dict:
        # Check for mock first
        mock = get_mock("api")
        if mock:
            return mock.get_response("/api/status")

        # Real implementation
        return await self._real_get_status()
```

### 3. Write Tests

```python
# tests/test_api.py
import pytest

@pytest.mark.unit
async def test_api_status(unit_test_mode):
    """Unit test - uses mocks, runs fast."""
    client = APIClient()
    result = await client.get_status()
    assert result["status"] == "ok"

@pytest.mark.integration
async def test_api_real():
    """Integration test - uses real API."""
    client = APIClient()
    result = await client.get_status()
    assert "status" in result
```

### 4. Run Tests

```bash
# Fast unit tests (all mocked)
pytest tests/ -m unit

# Integration tests (real LLM, mocked services)
pytest tests/ -m integration

# Only failed tests
pytest tests/ --lf

# Record fixtures
TEST_CACHE_MODE=record pytest tests/

# Replay only (CI mode)
TEST_CACHE_MODE=replay pytest tests/
```

## Test Modes

| Mode | Description | LLM | External Services |
|------|-------------|-----|-------------------|
| `UNIT` | Fast, isolated tests | Mocked | All mocked |
| `INTEGRATION` | Real AI, mocked services | Real | Mocked |
| `E2E` | Full system tests | Real | Real |
| `RECORD` | Capture responses | Real | Real, cached |
| `REPLAY` | Fail on cache miss | Cached | Cached |

## Cache Configuration

### Environment Variables

```bash
# Cache mode
TEST_CACHE_MODE=read_write    # Use and update cache (default)
TEST_CACHE_MODE=read_only     # Use cache, don't write
TEST_CACHE_MODE=record        # Always call API, save responses
TEST_CACHE_MODE=replay        # Fail on cache miss
TEST_CACHE_MODE=disabled      # No caching

# Cache settings
TEST_CACHE_TTL=3600           # TTL in seconds
TEST_CACHE_PATH=.test_cache   # Storage directory

# Per-service overrides
TEST_CACHE_LLM_MODE=replay    # Override just for LLM
TEST_CACHE_HTTP_MODE=record   # Override just for HTTP
```

### Programmatic Configuration

```python
from draagon_ai.testing import CacheConfig, CacheMode, UnifiedCacheManager

config = CacheConfig(
    mode=CacheMode.READ_WRITE,
    ttl_seconds=3600,
    storage_path=".test_cache",
    llm_mode=CacheMode.REPLAY,  # Override for LLM
)

UnifiedCacheManager.configure(config)
```

## Creating Service Mocks

### Basic Mock

```python
from draagon_ai.testing import ServiceMock

class DatabaseMock(ServiceMock):
    def __init__(self):
        super().__init__()
        self.data: dict[str, dict] = {}

    def get(self, key: str) -> dict | None:
        self.record_call("get", key=key)
        return self.data.get(key) or self.get_response(key)

    def put(self, key: str, value: dict) -> bool:
        self.record_call("put", key=key, value=value)
        self.data[key] = value
        return True
```

### Mock with Pattern Matching

```python
mock = ServiceMock()

# Pattern-based responses
mock.add_response({"temp": 72}, pattern=r"temperature|weather")
mock.add_response({"time": "3:00 PM"}, pattern=r"time|clock")

# Custom matcher
mock.add_response(
    {"status": "error"},
    matcher=lambda r: "invalid" in str(r).lower()
)

# Default fallback
mock.set_default({"status": "unknown"})
```

### Mock with Assertions

```python
@pytest.mark.unit
async def test_service_calls_database(db_mock, unit_test_mode):
    service = MyService()
    await service.fetch_user("user123")

    # Verify mock was called
    db_mock.assert_called("get", times=1)
    db_mock.assert_called_with("get", key="user123")
```

## Cache Integration Pattern

Add caching to your services for test optimization:

```python
from draagon_ai.testing import UnifiedCacheManager, is_test_mode

class LLMService:
    async def chat(self, prompt: str) -> str:
        cache = UnifiedCacheManager.get_instance()

        return await cache.get_or_call(
            service="llm",
            key_parts=(self.model, prompt),
            call_fn=lambda: self._real_chat(prompt),
            ttl=3600,  # 1 hour
        )
```

## pytest.ini Configuration

```ini
[tool.pytest.ini_options]
markers = [
    "unit: Fast unit tests with all mocks",
    "integration: Integration tests with real LLM",
    "e2e: End-to-end tests with real services",
    "slow: Tests that take > 5 seconds",
]

filterwarnings = ["ignore::DeprecationWarning"]

# Default to unit tests
addopts = "-m 'not slow'"
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    env:
      TEST_CACHE_MODE: replay
      TEST_MODE: unit
    steps:
      - uses: actions/checkout@v4
      - uses: actions/cache@v4
        with:
          path: .test_cache
          key: test-cache-${{ hashFiles('tests/fixtures/**') }}
      - run: pytest tests/ -m unit

  integration-tests:
    runs-on: ubuntu-latest
    env:
      TEST_CACHE_MODE: read_write
      TEST_MODE: integration
    steps:
      - run: pytest tests/ -m integration
```

## Best Practices

### 1. Use Markers Consistently

```python
@pytest.mark.unit
def test_fast_isolated():
    """Runs with all mocks, <100ms."""
    pass

@pytest.mark.integration
def test_with_real_llm():
    """Uses real LLM, mocked external services."""
    pass

@pytest.mark.e2e
def test_full_system():
    """Uses all real services."""
    pass
```

### 2. Record Fixtures for Deterministic Tests

```bash
# First, record real responses
TEST_CACHE_MODE=record pytest tests/test_api.py -m integration

# Then replay in CI
TEST_CACHE_MODE=replay pytest tests/test_api.py -m integration
```

### 3. Scope Mocks Appropriately

```python
# Per-test mock (most common)
@pytest.fixture
def api_mock():
    return APIMock()

# Session-wide mock (for expensive setup)
@pytest.fixture(scope="session")
def expensive_mock():
    mock = ExpensiveMock()
    mock.setup()  # One-time setup
    return mock
```

### 4. Use Factory Functions for Complex Mocks

```python
from draagon_ai.testing.fixtures import create_mock_fixture

# Create fixture with defaults
user_api_mock = create_mock_fixture(
    UserAPIMock,
    default_response={"id": "user123", "name": "Test User"},
    responses={
        r"/users/\d+": {"id": "user456", "name": "Found User"},
    }
)
```

### 5. Run Failed Tests First

```bash
# Only re-run failed tests (huge time saver)
pytest --lf

# Failed first, then others
pytest --ff
```

## Troubleshooting

### Cache Miss in Replay Mode

```
CacheMissError: Cache miss in replay mode: service=llm, key=abc123...
```

**Solution:** Run in record mode first to capture responses:
```bash
TEST_CACHE_MODE=record pytest tests/test_problem.py
```

### Mock Not Being Used

Check that:
1. Mock is registered with correct service name
2. `get_mock()` is being called in the service
3. Test is using the right fixture

```python
# Debug: Check if mock is active
from draagon_ai.testing import is_test_mode, get_mock

print(f"Test mode: {is_test_mode()}")
print(f"API mock: {get_mock('api')}")
```

### Nested Test Contexts

The framework supports nesting - inner contexts shadow outer ones:

```python
with test_mode(mocks={"api": mock1}):
    # mock1 is active
    with test_mode(mocks={"api": mock2}):
        # mock2 is active (shadows mock1)
    # mock1 is active again
```

## API Reference

### `draagon_ai.testing`

| Export | Type | Description |
|--------|------|-------------|
| `CacheMode` | Enum | Cache operation modes |
| `CacheConfig` | Class | Cache configuration |
| `UnifiedCacheManager` | Class | Central cache manager |
| `ServiceMock` | Class | Base class for mocks |
| `MockResponse` | Class | Response with matching |
| `TestMode` | Enum | Test execution modes |
| `TestContext` | Class | Current test context |
| `test_mode()` | Context manager | Enter test mode |
| `get_mock()` | Function | Get registered mock |
| `is_test_mode()` | Function | Check if in test mode |
| `with_mocks()` | Decorator | Set up mocks for function |

### `draagon_ai.testing.fixtures`

| Export | Type | Description |
|--------|------|-------------|
| `cache_config` | Fixture | Session cache config |
| `configure_cache` | Fixture | Set up cache manager |
| `llm_mock` | Fixture | Basic LLM mock |
| `embedding_mock` | Fixture | Embedding mock |
| `unit_test_context` | Fixture | Unit test mode context |
| `integration_test_context` | Fixture | Integration mode context |
| `pytest_configure` | Hook | Register markers |
| `pytest_collection_modifyitems` | Hook | Auto-apply fixtures |
