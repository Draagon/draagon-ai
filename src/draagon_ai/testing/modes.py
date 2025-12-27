"""
Test Mode Registry.

Provides test mode detection and context management for switching between
different testing configurations (unit, integration, e2e, etc.).

Example:
    from draagon_ai.testing import test_mode, TestMode, get_mock

    # In test
    with test_mode(TestMode.UNIT, mocks={"db": db_mock}):
        result = await my_service.query("test")  # Uses mock

    # In service code
    from draagon_ai.testing import get_mock, is_test_mode

    async def query(self, sql: str):
        mock = get_mock("db")
        if mock:
            return mock.query(sql)
        return await self._real_query(sql)
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, TypeVar

if TYPE_CHECKING:
    from draagon_ai.testing.cache import CacheConfig

T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class TestMode(str, Enum):
    """Test execution modes."""

    PRODUCTION = "production"  # Real services, no mocking
    INTEGRATION = "integration"  # Real LLM, mocked external services
    UNIT = "unit"  # All mocked, fast tests
    RECORD = "record"  # Record real responses for fixtures
    REPLAY = "replay"  # Replay only, fail on missing fixtures
    E2E = "e2e"  # End-to-end with real services


@dataclass
class TestContext:
    """
    Current test execution context.

    Holds the active test mode, registered mocks, and cache configuration.
    """

    mode: TestMode = TestMode.PRODUCTION
    mocks: dict[str, Any] = field(default_factory=dict)
    cache_config: "CacheConfig | None" = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_mock(self, service: str) -> Any | None:
        """Get mock for a service."""
        return self.mocks.get(service)

    def is_mocked(self, service: str) -> bool:
        """Check if a service is mocked."""
        return service in self.mocks

    def add_mock(self, service: str, mock: Any) -> None:
        """Register a mock for a service."""
        self.mocks[service] = mock

    @property
    def is_test(self) -> bool:
        """Check if in any test mode (not production)."""
        return self.mode != TestMode.PRODUCTION


# Global test context stack (supports nesting)
_context_stack: list[TestContext] = []


def get_test_context() -> TestContext | None:
    """
    Get current test context, if any.

    Returns:
        Current TestContext or None if not in test mode
    """
    if _context_stack:
        return _context_stack[-1]
    return None


def is_test_mode() -> bool:
    """
    Check if running in test mode.

    Returns:
        True if in any test mode (not production)
    """
    ctx = get_test_context()
    return ctx is not None and ctx.is_test


def get_current_mode() -> TestMode:
    """
    Get the current test mode.

    Returns:
        Current TestMode, defaults to PRODUCTION if not in test
    """
    ctx = get_test_context()
    return ctx.mode if ctx else TestMode.PRODUCTION


def get_mock(service: str) -> Any | None:
    """
    Get mock for a service, if in test mode.

    This is the primary way services check for mocks.

    Args:
        service: Service identifier (e.g., "homeassistant", "llm", "search")

    Returns:
        The mock object if registered, None otherwise

    Example:
        async def call_api(self, endpoint: str):
            mock = get_mock("api")
            if mock:
                return mock.call(endpoint)
            return await self._real_call(endpoint)
    """
    ctx = get_test_context()
    if ctx:
        return ctx.get_mock(service)
    return None


def register_mock(service: str, mock: Any) -> None:
    """
    Register a mock for the current test context.

    Args:
        service: Service identifier
        mock: Mock object to register

    Raises:
        RuntimeError: If not in a test context
    """
    ctx = get_test_context()
    if ctx is None:
        raise RuntimeError("Cannot register mock outside of test context")
    ctx.add_mock(service, mock)


@contextmanager
def test_mode(
    mode: TestMode = TestMode.UNIT,
    mocks: dict[str, Any] | None = None,
    cache_config: "CacheConfig | None" = None,
    **metadata: Any,
):
    """
    Context manager for entering test mode.

    Creates a new test context that is active within the context.
    Supports nesting - inner contexts shadow outer ones.

    Args:
        mode: The test mode to use
        mocks: Dictionary of service name to mock object
        cache_config: Optional cache configuration
        **metadata: Additional metadata for the context

    Yields:
        The active TestContext

    Example:
        with test_mode(TestMode.UNIT, mocks={"db": db_mock}):
            # db_mock is now active
            result = await service.query()

        # Outside context, no mock active
    """
    ctx = TestContext(
        mode=mode,
        mocks=mocks or {},
        cache_config=cache_config,
        metadata=metadata,
    )

    _context_stack.append(ctx)
    try:
        yield ctx
    finally:
        _context_stack.pop()


def with_mocks(**mocks: Any) -> Callable[[F], F]:
    """
    Decorator to set up mocks for a test function.

    Works with both sync and async functions.

    Args:
        **mocks: Service name to mock object mappings

    Returns:
        Decorated function that runs in test mode with mocks

    Example:
        @with_mocks(db=DatabaseMock(), api=APIMock())
        async def test_integration():
            result = await service.fetch_data()
            assert result == expected
    """

    def decorator(func: F) -> F:
        import asyncio

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with test_mode(mode=TestMode.UNIT, mocks=mocks):
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                with test_mode(mode=TestMode.UNIT, mocks=mocks):
                    return func(*args, **kwargs)

            return sync_wrapper  # type: ignore

    return decorator


def detect_mode_from_env() -> TestMode:
    """
    Detect test mode from environment variables.

    Checks TEST_MODE env var and pytest markers.

    Returns:
        Detected TestMode
    """
    # Check explicit env var
    mode_str = os.environ.get("TEST_MODE", "").lower()
    mode_map = {
        "unit": TestMode.UNIT,
        "integration": TestMode.INTEGRATION,
        "e2e": TestMode.E2E,
        "record": TestMode.RECORD,
        "replay": TestMode.REPLAY,
        "production": TestMode.PRODUCTION,
    }
    if mode_str in mode_map:
        return mode_map[mode_str]

    # Check if running under pytest
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return TestMode.INTEGRATION  # Default for pytest

    return TestMode.PRODUCTION


# Pytest integration helpers


def pytest_configure_modes():
    """
    Register pytest markers for test modes.

    Call this from conftest.py:

        def pytest_configure(config):
            from draagon_ai.testing import pytest_configure_modes
            pytest_configure_modes()
    """
    import pytest

    pytest.mark.unit = pytest.mark.unit
    pytest.mark.integration = pytest.mark.integration
    pytest.mark.e2e = pytest.mark.e2e


def get_mode_from_markers(item: Any) -> TestMode | None:
    """
    Get test mode from pytest markers.

    Args:
        item: pytest test item

    Returns:
        TestMode based on markers, or None if no marker
    """
    if hasattr(item, "get_closest_marker"):
        if item.get_closest_marker("unit"):
            return TestMode.UNIT
        if item.get_closest_marker("integration"):
            return TestMode.INTEGRATION
        if item.get_closest_marker("e2e"):
            return TestMode.E2E
    return None
