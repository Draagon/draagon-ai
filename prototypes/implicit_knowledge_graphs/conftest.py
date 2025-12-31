"""Pytest configuration for the implicit_knowledge_graphs prototype.

This adds src/ to the Python path for imports.
"""

import sys
from pathlib import Path

# Add src to path for imports BEFORE pytest runs test collection
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also expose fixture
import pytest

@pytest.fixture(autouse=True)
def ensure_path():
    """Ensure src is in path for all tests."""
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
