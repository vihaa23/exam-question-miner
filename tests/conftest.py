"""Pytest configuration and shared fixtures."""

import sys
from pathlib import Path

# Ensure project root is in sys.path so tests find the package
sys.path.insert(0, str(Path(__file__).parent.parent))
