"""
Unified Database Management Module.
Re-exports from src.database.brand_database for full backward compatibility.
"""

import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.database.brand_database import BrandDatabase

__all__ = ["BrandDatabase"]
