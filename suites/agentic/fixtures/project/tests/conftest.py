"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src/ to sys.path so tests can import from src
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))
