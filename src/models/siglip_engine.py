"""
SigLIP 2 Feature Extraction and Multi-View Ensembling Engine.
Re-exports the working implementation from the root module.
"""
# Import the complete working implementation
import sys
from pathlib import Path

# Ensure project root is on path for existing imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from siglip_engine import SigLIPEngine

__all__ = ["SigLIPEngine"]
