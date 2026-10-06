"""
Color Feature Extraction and Matching Engine.
Re-exports the working implementation from the root module.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from color_features import extract_color_features, compare_color_features

__all__ = ["extract_color_features", "compare_color_features"]
