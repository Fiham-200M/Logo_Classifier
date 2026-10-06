"""
Deterministic Image Preprocessing Module.
Re-exports the working implementation from the root preprocessing module.
"""

import sys
from pathlib import Path

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from preprocessing import (
    load_image_safe,
    composite_on_background,
    pad_to_square,
    resize_preserve_aspect,
    center_crop_view,
    expand_padding_view,
    upscale_for_ocr,
    generate_preprocessing_views,
)

__all__ = [
    "load_image_safe",
    "composite_on_background",
    "pad_to_square",
    "resize_preserve_aspect",
    "center_crop_view",
    "expand_padding_view",
    "upscale_for_ocr",
    "generate_preprocessing_views",
]
