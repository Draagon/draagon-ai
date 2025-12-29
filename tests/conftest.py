"""Pytest configuration for draagon-ai tests."""

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest


def _load_env_file(path: Path) -> None:
    """Load environment variables from a .env file."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


# Load environment from .env files
# 1. Try local draagon-ai .env
_load_env_file(Path(__file__).parent.parent / ".env")
# 2. Try roxy-voice-assistant .env (for GROQ_API_KEY, etc.)
_load_env_file(Path(__file__).parent.parent.parent / "roxy-voice-assistant" / ".env")


@pytest.fixture
def anyio_backend():
    """Use asyncio for async tests."""
    return "asyncio"


# =============================================================================
# Mock LLM Fixtures
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for tests that need LLM responses without real API calls.

    Usage:
        @pytest.fixture
        def mock_llm():
            return MockLLMProvider(default_response={"answer": "42"})

        async def test_something(mock_llm):
            result = await mock_llm.chat([{"role": "user", "content": "Hello"}])
            assert result["content"] == '{"answer": "42"}'
    """

    def __init__(
        self,
        default_response: dict[str, Any] | str | None = None,
        responses: list[dict[str, Any] | str] | None = None,
    ):
        """Initialize mock LLM.

        Args:
            default_response: Default response dict (will be JSON-serialized) or string.
            responses: List of responses to return in sequence (each JSON-serialized if dict).
        """
        self.default_response = default_response or {"response": "ok"}
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """Mock chat completion."""
        self.call_count += 1
        self.calls.append({"messages": messages, "model": model, "kwargs": kwargs})

        if self.responses:
            response = self.responses.pop(0)
        else:
            response = self.default_response

        if isinstance(response, dict):
            content = json.dumps(response)
        else:
            content = str(response)

        return {"content": content}

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Mock chat completion that returns parsed JSON."""
        result = await self.chat(messages, model, **kwargs)
        return json.loads(result["content"])

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Mock embedding - returns deterministic fake embeddings."""
        if isinstance(text, str):
            text = [text]
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for t in text:
            # 768-dim embedding based on hash
            seed = hash(t) % 10000
            embeddings.append([float((seed + i) % 1000) / 1000 for i in range(768)])
        return embeddings

    def reset(self) -> None:
        """Reset call tracking."""
        self.call_count = 0
        self.calls = []


@pytest.fixture
def mock_llm() -> MockLLMProvider:
    """Provide a mock LLM provider for tests."""
    return MockLLMProvider()


@pytest.fixture
def mock_llm_with_responses():
    """Factory fixture for creating mock LLM with specific responses."""

    def _create(responses: list[dict[str, Any] | str]) -> MockLLMProvider:
        return MockLLMProvider(responses=responses)

    return _create


# =============================================================================
# Mock Memory Fixtures
# =============================================================================


class MockMemoryProvider:
    """Mock memory provider for tests.

    Stores memories in-memory without Qdrant.
    """

    def __init__(self):
        self.memories: dict[str, dict[str, Any]] = {}
        self.call_log: list[tuple[str, dict]] = []

    async def store(
        self,
        content: str,
        user_id: str = "test_user",
        memory_type: str = "fact",
        importance: float = 0.5,
        entities: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Store a memory and return its ID."""
        memory_id = f"mem_{len(self.memories)}"
        self.memories[memory_id] = {
            "id": memory_id,
            "content": content,
            "user_id": user_id,
            "memory_type": memory_type,
            "importance": importance,
            "entities": entities or [],
            **kwargs,
        }
        self.call_log.append(("store", {"content": content, "user_id": user_id}))
        return memory_id

    async def search(
        self,
        query: str,
        user_id: str = "test_user",
        limit: int = 5,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search memories by simple substring match."""
        self.call_log.append(("search", {"query": query, "user_id": user_id}))
        results = []
        query_lower = query.lower()
        for mem in self.memories.values():
            if mem["user_id"] == user_id and query_lower in mem["content"].lower():
                results.append({**mem, "score": 0.9})
        return results[:limit]

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        self.call_log.append(("delete", {"memory_id": memory_id}))
        if memory_id in self.memories:
            del self.memories[memory_id]
            return True
        return False

    async def get(self, memory_id: str) -> dict[str, Any] | None:
        """Get a memory by ID."""
        self.call_log.append(("get", {"memory_id": memory_id}))
        return self.memories.get(memory_id)

    def reset(self) -> None:
        """Reset all memories and call log."""
        self.memories.clear()
        self.call_log.clear()


@pytest.fixture
def mock_memory() -> MockMemoryProvider:
    """Provide a mock memory provider for tests."""
    return MockMemoryProvider()


# =============================================================================
# Mock Tool Fixtures
# =============================================================================


@pytest.fixture
def mock_tool():
    """Create a mock tool for orchestration tests."""

    def _create(
        name: str = "test_tool",
        description: str = "A test tool",
        result: Any = "tool result",
    ):
        tool = MagicMock()
        tool.name = name
        tool.description = description
        tool.execute = AsyncMock(return_value=result)
        return tool

    return _create


# =============================================================================
# Test User/Conversation Fixtures
# =============================================================================


@pytest.fixture
def test_user_id() -> str:
    """Provide a consistent test user ID."""
    return "test_user_123"


@pytest.fixture
def test_conversation_id() -> str:
    """Provide a unique test conversation ID."""
    import uuid

    return f"test_conv_{uuid.uuid4().hex[:8]}"
