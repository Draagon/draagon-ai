"""
Service Mocking Framework.

Provides base classes for creating service mocks that can be used in tests.
App developers extend these classes to create their own service-specific mocks.

Example:
    from draagon_ai.testing import ServiceMock, MockResponse

    class MyAPIMock(ServiceMock):
        def __init__(self):
            super().__init__()
            self.data: dict[str, Any] = {}

        def get_item(self, item_id: str) -> dict | None:
            self.record_call("get_item", item_id=item_id)
            return self.data.get(item_id) or self.get_response(item_id)

        def add_item(self, item_id: str, data: dict):
            self.data[item_id] = data
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class MockNotFoundError(Exception):
    """Raised when no mock response matches the request."""

    pass


@dataclass
class MockResponse(Generic[T]):
    """
    A mock response with optional matching conditions.

    Attributes:
        response: The value to return when matched
        match_pattern: Optional regex pattern to match against request
        match_fn: Optional custom matching function
        delay_ms: Simulated latency in milliseconds
        call_count: Number of times this response was used
        max_uses: Maximum number of times this response can be used (None = unlimited)
    """

    response: T
    match_pattern: str | None = None
    match_fn: Callable[[Any], bool] | None = None
    delay_ms: int = 0
    call_count: int = 0
    max_uses: int | None = None

    def matches(self, request: Any) -> bool:
        """Check if this response matches the given request."""
        if self.max_uses is not None and self.call_count >= self.max_uses:
            return False

        if self.match_pattern:
            request_str = str(request)
            if re.search(self.match_pattern, request_str, re.IGNORECASE):
                return True
            return False

        if self.match_fn:
            return self.match_fn(request)

        # No conditions means it matches everything (default response)
        return True

    async def apply_delay(self) -> None:
        """Apply simulated delay if configured."""
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms / 1000)


@dataclass
class CallRecord:
    """Record of a call to a mocked method."""

    method: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    response: Any = None
    error: Exception | None = None


@dataclass
class ServiceMock:
    """
    Base class for service mocks.

    Provides common functionality for:
    - Adding mock responses
    - Recording calls for verification
    - Response matching with patterns

    Subclasses should implement service-specific methods that call
    get_response() or record_call().

    Example:
        class DatabaseMock(ServiceMock):
            def query(self, sql: str) -> list[dict]:
                self.record_call("query", sql=sql)
                return self.get_response(sql)
    """

    enabled: bool = True
    responses: list[MockResponse] = field(default_factory=list)
    default_response: Any = None
    calls: list[CallRecord] = field(default_factory=list)
    raise_on_no_match: bool = True

    def add_response(
        self,
        response: Any,
        pattern: str | None = None,
        matcher: Callable[[Any], bool] | None = None,
        delay_ms: int = 0,
        max_uses: int | None = None,
    ) -> "ServiceMock":
        """
        Add a mock response.

        Args:
            response: The value to return
            pattern: Optional regex pattern to match requests
            matcher: Optional custom matching function
            delay_ms: Simulated latency
            max_uses: Maximum times this response can be used

        Returns:
            Self for chaining
        """
        self.responses.append(
            MockResponse(
                response=response,
                match_pattern=pattern,
                match_fn=matcher,
                delay_ms=delay_ms,
                max_uses=max_uses,
            )
        )
        return self

    def set_default(self, response: Any) -> "ServiceMock":
        """Set the default response when no patterns match."""
        self.default_response = response
        return self

    def get_response(self, request: Any = None) -> Any:
        """
        Get matching response for a request.

        Searches through responses in order, returning the first match.

        Args:
            request: The request to match against

        Returns:
            The matched response value

        Raises:
            MockNotFoundError: If no response matches and raise_on_no_match is True
        """
        for mock_resp in self.responses:
            if mock_resp.matches(request):
                mock_resp.call_count += 1
                return mock_resp.response

        if self.default_response is not None:
            return self.default_response

        if self.raise_on_no_match:
            raise MockNotFoundError(f"No mock response found for request: {request}")

        return None

    async def get_response_async(self, request: Any = None) -> Any:
        """
        Async version of get_response with delay simulation.

        Args:
            request: The request to match against

        Returns:
            The matched response value
        """
        for mock_resp in self.responses:
            if mock_resp.matches(request):
                mock_resp.call_count += 1
                await mock_resp.apply_delay()
                return mock_resp.response

        if self.default_response is not None:
            return self.default_response

        if self.raise_on_no_match:
            raise MockNotFoundError(f"No mock response found for request: {request}")

        return None

    def record_call(
        self,
        method: str,
        *args: Any,
        response: Any = None,
        error: Exception | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Record a method call for later verification.

        Args:
            method: Name of the method called
            *args: Positional arguments
            response: Response that was returned
            error: Exception that was raised (if any)
            **kwargs: Keyword arguments
        """
        self.calls.append(
            CallRecord(
                method=method,
                args=args,
                kwargs=kwargs,
                response=response,
                error=error,
            )
        )

    def get_calls(self, method: str | None = None) -> list[CallRecord]:
        """
        Get recorded calls, optionally filtered by method.

        Args:
            method: Filter to calls of this method, or None for all

        Returns:
            List of matching call records
        """
        if method is None:
            return self.calls
        return [c for c in self.calls if c.method == method]

    def assert_called(self, method: str, times: int | None = None) -> None:
        """
        Assert that a method was called.

        Args:
            method: Method name to check
            times: Expected number of calls, or None to just check it was called

        Raises:
            AssertionError: If assertion fails
        """
        calls = self.get_calls(method)
        if times is not None:
            assert len(calls) == times, (
                f"Expected {method} to be called {times} times, "
                f"but was called {len(calls)} times"
            )
        else:
            assert len(calls) > 0, f"Expected {method} to be called, but was not"

    def assert_called_with(self, method: str, **expected_kwargs: Any) -> None:
        """
        Assert that a method was called with specific arguments.

        Args:
            method: Method name to check
            **expected_kwargs: Expected keyword arguments

        Raises:
            AssertionError: If no matching call found
        """
        calls = self.get_calls(method)
        for call in calls:
            if all(call.kwargs.get(k) == v for k, v in expected_kwargs.items()):
                return
        raise AssertionError(
            f"Expected {method} to be called with {expected_kwargs}, "
            f"but calls were: {[c.kwargs for c in calls]}"
        )

    def reset(self) -> None:
        """Reset all recorded calls and response counts."""
        self.calls = []
        for resp in self.responses:
            resp.call_count = 0


# Common mock implementations that apps can extend


class LLMMock(ServiceMock):
    """
    Mock for LLM services.

    Example:
        mock = LLMMock()
        mock.add_response("Hello!", pattern=r"greet")
        mock.add_response("The time is 3 PM", pattern=r"time|clock")

        response = mock.chat("What time is it?")  # Returns "The time is 3 PM"
    """

    def __init__(self):
        super().__init__()
        self.model = "mock-model"

    def chat(self, prompt: str, **kwargs: Any) -> str:
        """Mock chat completion."""
        self.record_call("chat", prompt=prompt, **kwargs)
        return self.get_response(prompt)

    async def chat_async(self, prompt: str, **kwargs: Any) -> str:
        """Mock async chat completion."""
        self.record_call("chat", prompt=prompt, **kwargs)
        return await self.get_response_async(prompt)


class HTTPMock(ServiceMock):
    """
    Mock for HTTP services.

    Example:
        mock = HTTPMock()
        mock.add_response({"status": "ok"}, pattern=r"/api/status")
        mock.add_response({"error": "not found"}, pattern=r"/api/missing")

        response = await mock.get("/api/status")  # Returns {"status": "ok"}
    """

    def __init__(self):
        super().__init__()
        self.base_url = "http://mock"

    async def get(self, path: str, **kwargs: Any) -> Any:
        """Mock HTTP GET request."""
        self.record_call("get", path=path, **kwargs)
        return await self.get_response_async(path)

    async def post(self, path: str, data: Any = None, **kwargs: Any) -> Any:
        """Mock HTTP POST request."""
        self.record_call("post", path=path, data=data, **kwargs)
        return await self.get_response_async(f"POST:{path}:{data}")


class EmbeddingMock(ServiceMock):
    """
    Mock for embedding services.

    Returns deterministic embeddings based on input text hash.
    """

    def __init__(self, dimensions: int = 768):
        super().__init__()
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        """Generate deterministic mock embedding."""
        self.record_call("embed", text=text)

        # Generate deterministic embedding from text hash
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # Convert bytes to floats in range [-1, 1]
        embedding = []
        for i in range(self.dimensions):
            byte_val = h[i % len(h)]
            embedding.append((byte_val / 127.5) - 1)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate mock embeddings for batch."""
        return [self.embed(t) for t in texts]
