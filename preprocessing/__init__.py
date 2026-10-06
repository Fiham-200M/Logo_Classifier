"""
Preprocessing package and module interface.
Exposes both legacy deterministic preprocessing utilities and the new MediaNormalizer.
"""

from preprocessing.media_normalizer import MediaNormalizer
from preprocessing.gif_processor import GIFProcessor, process_gif

# Import legacy helpers to preserve backward compatibility for existing code (e.g. ocr_engine.py)
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent

# Re-export from root preprocessing.py if needed
try:
    import importlib.util
    legacy_path = _root / "preprocessing.py"
    if legacy_path.exists():
        spec = importlib.util.spec_from_file_location("_legacy_preprocessing", str(legacy_path))
        legacy_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy_mod)
        load_image_safe = legacy_mod.load_image_safe
        composite_on_background = legacy_mod.composite_on_background
        pad_to_square = legacy_mod.pad_to_square
        resize_preserve_aspect = legacy_mod.resize_preserve_aspect
        center_crop_view = legacy_mod.center_crop_view
        expand_padding_view = legacy_mod.expand_padding_view
        upscale_for_ocr = legacy_mod.upscale_for_ocr
        generate_preprocessing_views = legacy_mod.generate_preprocessing_views
except Exception as e:
    pass

__all__ = [
    "MediaNormalizer",
    "GIFProcessor",
    "process_gif",
    "load_image_safe",
    "composite_on_background",
    "pad_to_square",
    "resize_preserve_aspect",
    "center_crop_view",
    "expand_padding_view",
    "upscale_for_ocr",
    "generate_preprocessing_views",
]
