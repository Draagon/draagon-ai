"""
Pytest Fixtures for Draagon AI Testing.

Provides reusable pytest fixtures that apps can import and use in their conftest.py.

Usage in app's conftest.py:
    from draagon_ai.testing.fixtures import (
        cache_config,
        configure_cache,
        llm_mock,
        embedding_mock,
        unit_test_mode,
    )

    # Re-export the fixtures
    __all__ = [
        "cache_config",
        "configure_cache",
        "llm_mock",
        "embedding_mock",
        "unit_test_mode",
    ]

    # Add app-specific fixtures
    @pytest.fixture
    def my_service_mock():
        return MyServiceMock()
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Generator

import pytest

from draagon_ai.testing.cache import CacheConfig, CacheMode, UnifiedCacheManager
from draagon_ai.testing.mocks import EmbeddingMock, HTTPMock, LLMMock, ServiceMock
from draagon_ai.testing.modes import TestContext, TestMode, test_mode

if TYPE_CHECKING:
    from _pytest.config import Config
    from _pytest.nodes import Item


# ============================================================================
# Cache Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def cache_config() -> CacheConfig:
    """
    Configure caching for test session.

    Reads from environment variables:
    - TEST_CACHE_MODE: Cache mode (read_write, replay, etc.)
    - TEST_CACHE_TTL: TTL in seconds
    - TEST_CACHE_PATH: Path to cache storage

    Returns:
        CacheConfig instance
    """
    mode_str = os.environ.get("TEST_CACHE_MODE", "read_write")
    try:
        mode = CacheMode(mode_str)
    except ValueError:
        mode = CacheMode.READ_WRITE

    return CacheConfig(
        mode=mode,
        ttl_seconds=int(os.environ.get("TEST_CACHE_TTL", "3600")),
        storage_path=os.environ.get("TEST_CACHE_PATH", ".test_cache"),
    )


@pytest.fixture(scope="session")
def configure_cache(cache_config: CacheConfig) -> Generator[UnifiedCacheManager, None, None]:
    """
    Set up cache manager for test session.

    Configures the global cache manager and persists cache on teardown.

    Yields:
        Configured UnifiedCacheManager instance
    """
    manager = UnifiedCacheManager.configure(cache_config)
    yield manager

    # Print stats on teardown
    stats = manager.get_stats()
    if stats["hits"] + stats["misses"] > 0:
        print(f"\nCache stats: {stats['hit_rate']} hit rate, "
              f"{stats['total_entries']} entries across {stats['services']}")

    # Reset for clean state
    UnifiedCacheManager.reset()


# ============================================================================
# Mock Fixtures
# ============================================================================


@pytest.fixture
def llm_mock() -> LLMMock:
    """
    Basic LLM mock.

    Apps should extend this with their own responses.

    Returns:
        LLMMock instance with default responses
    """
    mock = LLMMock()

    # Add some default responses
    mock.add_response("Hello! How can I help you?", pattern=r"hello|hi|hey")
    mock.add_response("I don't know the answer to that.", pattern=r".*")

    return mock


@pytest.fixture
def embedding_mock() -> EmbeddingMock:
    """
    Embedding mock with deterministic outputs.

    Returns:
        EmbeddingMock instance
    """
    return EmbeddingMock(dimensions=768)


@pytest.fixture
def http_mock() -> HTTPMock:
    """
    HTTP client mock.

    Returns:
        HTTPMock instance
    """
    mock = HTTPMock()
    mock.set_default({"status": "ok"})
    return mock


# ============================================================================
# Test Mode Fixtures
# ============================================================================


@pytest.fixture
def unit_test_context(
    llm_mock: LLMMock,
    embedding_mock: EmbeddingMock,
) -> Generator[TestContext, None, None]:
    """
    Context for fast unit tests with all mocks.

    Yields:
        TestContext with LLM and embedding mocks
    """
    with test_mode(
        mode=TestMode.UNIT,
        mocks={
            "llm": llm_mock,
            "embedding": embedding_mock,
        },
    ) as ctx:
        yield ctx


@pytest.fixture
def integration_test_context() -> Generator[TestContext, None, None]:
    """
    Context for integration tests (real LLM, mocked external services).

    Yields:
        TestContext in integration mode
    """
    with test_mode(mode=TestMode.INTEGRATION) as ctx:
        yield ctx


@pytest.fixture
def e2e_test_context() -> Generator[TestContext, None, None]:
    """
    Context for end-to-end tests with real services.

    Yields:
        TestContext in e2e mode
    """
    with test_mode(mode=TestMode.E2E) as ctx:
        yield ctx


# ============================================================================
# Pytest Hooks
# ============================================================================


def pytest_configure(config: "Config") -> None:
    """
    Register custom pytest markers.

    Called by pytest during startup.
    """
    config.addinivalue_line("markers", "unit: Fast unit tests with all mocks")
    config.addinivalue_line("markers", "integration: Integration tests with real LLM")
    config.addinivalue_line("markers", "e2e: End-to-end tests with real services")
    config.addinivalue_line("markers", "slow: Tests that take > 5 seconds")


def pytest_collection_modifyitems(items: list["Item"]) -> None:
    """
    Auto-apply test mode fixtures based on markers.

    Tests marked with @pytest.mark.unit get the unit_test_context fixture.
    Tests marked with @pytest.mark.integration get the integration_test_context fixture.
    """
    for item in items:
        if "unit" in item.keywords and "unit_test_context" not in item.fixturenames:
            item.fixturenames.append("unit_test_context")
        elif "integration" in item.keywords and "integration_test_context" not in item.fixturenames:
            item.fixturenames.append("integration_test_context")
        elif "e2e" in item.keywords and "e2e_test_context" not in item.fixturenames:
            item.fixturenames.append("e2e_test_context")


# ============================================================================
# Helper Functions for Apps
# ============================================================================


def create_mock_fixture(mock_class: type[ServiceMock], **defaults):
    """
    Factory function to create mock fixtures.

    Usage in app's conftest.py:
        from draagon_ai.testing.fixtures import create_mock_fixture
        from my_app.testing import MyServiceMock

        my_service_mock = create_mock_fixture(
            MyServiceMock,
            default_response={"status": "ok"}
        )

    Args:
        mock_class: The mock class to instantiate
        **defaults: Default configuration for the mock

    Returns:
        A pytest fixture function
    """

    @pytest.fixture
    def mock_fixture():
        mock = mock_class()
        if "default_response" in defaults:
            mock.set_default(defaults["default_response"])
        for pattern, response in defaults.get("responses", {}).items():
            mock.add_response(response, pattern=pattern)
        return mock

    return mock_fixture


def create_test_mode_fixture(
    mode: TestMode,
    mock_names: list[str],
):
    """
    Factory function to create test mode fixtures with specific mocks.

    Usage in app's conftest.py:
        from draagon_ai.testing.fixtures import create_test_mode_fixture, TestMode

        api_test_mode = create_test_mode_fixture(
            mode=TestMode.INTEGRATION,
            mock_names=["database", "cache"]
        )

    Args:
        mode: The test mode to use
        mock_names: List of mock fixture names to include

    Returns:
        A pytest fixture function
    """

    @pytest.fixture
    def mode_fixture(request):
        mocks = {}
        for name in mock_names:
            if name in request.fixturenames:
                mocks[name] = request.getfixturevalue(name)

        with test_mode(mode=mode, mocks=mocks) as ctx:
            yield ctx

    return mode_fixture
