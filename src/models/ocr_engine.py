"""
OCR Engine — Text Extraction and Fuzzy Matching.
Re-exports the working implementation from the root module.
"""
import sys
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ocr_engine import OCREngine

__all__ = ["OCREngine"]
