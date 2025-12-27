"""
Draagon AI Testing Framework.

A comprehensive testing infrastructure for AI agent applications built on draagon-ai.
Provides caching, mocking, and test mode management for efficient and deterministic testing.

Usage for App Developers
------------------------

1. Configure test modes in your conftest.py:

    from draagon_ai.testing import (
        TestMode, test_mode, CacheConfig, CacheMode,
        ServiceMock, MockResponse
    )

    @pytest.fixture
    def unit_test_mode(my_service_mock):
        with test_mode(TestMode.UNIT, mocks={"my_service": my_service_mock}):
            yield

2. Create service-specific mocks by extending ServiceMock:

    from draagon_ai.testing import ServiceMock

    class MyServiceMock(ServiceMock):
        def my_method(self, arg):
            return self.get_response(arg)

3. Use cache manager for external API caching:

    from draagon_ai.testing import UnifiedCacheManager, CacheMode

    cache = UnifiedCacheManager.get_instance()
    result = await cache.get_or_call("llm", (prompt,), call_fn)

4. Run tests with different modes:

    # Fast unit tests (all mocked)
    pytest tests/ -m unit

    # Integration tests (real LLM, mocked services)
    pytest tests/ -m integration

    # Record fixtures
    TEST_CACHE_MODE=record pytest tests/

    # Replay only (CI)
    TEST_CACHE_MODE=replay pytest tests/
"""

from draagon_ai.testing.cache import (
    CacheMode,
    CacheConfig,
    UnifiedCacheManager,
    CacheMissError,
)

from draagon_ai.testing.mocks import (
    MockResponse,
    ServiceMock,
    MockNotFoundError,
)

from draagon_ai.testing.modes import (
    TestMode,
    TestContext,
    get_test_context,
    is_test_mode,
    get_mock,
    test_mode,
    with_mocks,
)

__all__ = [
    # Cache
    "CacheMode",
    "CacheConfig",
    "UnifiedCacheManager",
    "CacheMissError",
    # Mocks
    "MockResponse",
    "ServiceMock",
    "MockNotFoundError",
    # Modes
    "TestMode",
    "TestContext",
    "get_test_context",
    "is_test_mode",
    "get_mock",
    "test_mode",
    "with_mocks",
]
