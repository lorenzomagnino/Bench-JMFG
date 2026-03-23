"""Root conftest: add src/ to sys.path for the src-layout package structure."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))
