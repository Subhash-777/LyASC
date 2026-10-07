"""conftest.py — Pytest configuration for LyASC/base_model tests.

Adds the workspace root to sys.path so that `import src.*` works from
test files without needing to install the package.
"""

import sys
from pathlib import Path

# Add the base_model workspace root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
