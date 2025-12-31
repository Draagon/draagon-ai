"""Test configuration for semantic expansion prototype.

This conftest.py sets up the Python path so tests can import from
the prototype's src folder.
"""

import sys
from pathlib import Path

# Add the prototype src folder to the path
prototype_root = Path(__file__).parent.parent
src_path = prototype_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also ensure the main draagon-ai package is available for base types
project_root = prototype_root.parent.parent
draagon_src = project_root / "src"
if str(draagon_src) not in sys.path:
    sys.path.insert(0, str(draagon_src))
