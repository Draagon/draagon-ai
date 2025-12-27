"""Pytest configuration for draagon-ai tests."""

import os
from pathlib import Path

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
