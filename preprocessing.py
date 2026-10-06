"""
Deterministic Image Preprocessing Module.
Generates multi-view representations of input images for SigLIP embedding ensembling,
color analysis, and OCR robustness.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageOps, ImageSequence


def load_image_safe(image_source: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """
    Safely load an image from path, PIL Image, or numpy array.
    Handles animated GIFs / WebPs by extracting the first valid frame.
    Preserves RGBA if transparency is present; otherwise RGB.
    """
    if isinstance(image_source, Image.Image):
        img = image_source
    elif isinstance(image_source, (str, Path)):
        img = Image.open(str(image_source))
    elif isinstance(image_source, np.ndarray):
        img = Image.fromarray(image_source)
    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    # If animated GIF/WebP, extract the first frame
    if getattr(img, "is_animated", False):
        img.seek(0)
        img = img.copy()

    # Ensure in RGB or RGBA mode
    if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
        return img.convert("RGBA")
    return img.convert("RGB")


def composite_on_background(img: Image.Image, bg_color: Tuple[int, int, int]) -> Image.Image:
    """
    Composite an RGBA image onto a solid background color (e.g. white or black).
    If the image has no alpha, returns an RGB copy.
    """
    if img.mode != "RGBA":
        return img.convert("RGB")

    bg = Image.new("RGB", img.size, bg_color)
    bg.paste(img, mask=img.split()[3])  # 3 is the alpha channel
    return bg


def pad_to_square(img: Image.Image, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Pad the image to a square preserving aspect ratio.
    """
    rgb = composite_on_background(img, bg_color)
    w, h = rgb.size
    max_dim = max(w, h)
    square = Image.new("RGB", (max_dim, max_dim), bg_color)
    offset_x = (max_dim - w) // 2
    offset_y = (max_dim - h) // 2
    square.paste(rgb, (offset_x, offset_y))
    return square


def resize_preserve_aspect(img: Image.Image, target_size: int = 224, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Resize preserving aspect ratio and fit into target_size x target_size with padding.
    """
    rgb = composite_on_background(img, bg_color)
    w, h = rgb.size
    scale = target_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (target_size, target_size), bg_color)
    offset_x = (target_size - new_w) // 2
    offset_y = (target_size - new_h) // 2
    canvas.paste(resized, (offset_x, offset_y))
    return canvas


def center_crop_view(img: Image.Image, crop_ratio: float = 0.85) -> Image.Image:
    """
    Extract a center crop of the image (default 85% area) to test focus on core logo shape.
    """
    rgb = composite_on_background(img, (255, 255, 255))
    w, h = rgb.size
    crop_w = max(1, int(w * crop_ratio))
    crop_h = max(1, int(h * crop_ratio))
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    return rgb.crop((left, top, left + crop_w, top + crop_h))


def expand_padding_view(img: Image.Image, pad_pct: float = 0.15, bg_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
    """
    Add external padding around the image.
    """
    rgb = composite_on_background(img, bg_color)
    w, h = rgb.size
    pad_x = max(2, int(w * pad_pct))
    pad_y = max(2, int(h * pad_pct))
    new_w = w + 2 * pad_x
    new_h = h + 2 * pad_y
    padded = Image.new("RGB", (new_w, new_h), bg_color)
    padded.paste(rgb, (pad_x, pad_y))
    return padded


def upscale_for_ocr(img: Image.Image, min_height: int = 64, max_scale: int = 4) -> Image.Image:
    """
    High-quality upscale for OCR if the input image is small.
    """
    w, h = img.size
    if h >= min_height:
        return img
    scale = min(max_scale, max(2, int(np.ceil(min_height / max(1, h)))))
    return img.resize((w * scale, h * scale), Image.Resampling.LANCZOS)


def generate_preprocessing_views(image_source: Union[str, Path, Image.Image, np.ndarray]) -> Dict[str, Image.Image]:
    """
    Generate the complete dictionary of deterministic views for an input image.
    Returns:
        Dict[view_name, PIL.Image in RGB]
    """
    raw_img = load_image_safe(image_source)

    # Base RGB versions with alpha resolved
    white_bg = composite_on_background(raw_img, (255, 255, 255))
    black_bg = composite_on_background(raw_img, (0, 0, 0))

    # Standard original view (white background if transparent)
    original_view = white_bg

    # Contrast enhancement (moderate factor 1.3)
    contrast_enhancer = ImageEnhance.Contrast(white_bg)
    contrast_view = contrast_enhancer.enhance(1.3)

    # Subtle sharpening (factor 1.35, avoids creating fake ringing artifacts)
    sharp_enhancer = ImageEnhance.Sharpness(white_bg)
    sharp_view = sharp_enhancer.enhance(1.35)

    # Grayscale view
    gray_view = ImageOps.grayscale(white_bg).convert("RGB")

    # Geometric views
    square_view = pad_to_square(raw_img, (255, 255, 255))
    resized_view = resize_preserve_aspect(raw_img, 224, (255, 255, 255))
    crop_view = center_crop_view(raw_img, 0.85)
    expanded_view = expand_padding_view(raw_img, 0.15, (255, 255, 255))

    if isinstance(image_source, (str, Path)):
        temp = Image.open(image_source)
        if getattr(temp, "is_animated", False):
            temp.seek(0)
        raw_rgb = temp.convert("RGB")
    else:
        raw_rgb = raw_img.convert("RGB")

    return {
        "raw_rgb": raw_rgb,
        "original": original_view,
        "white_bg": white_bg,
        "black_bg": black_bg,
        "square_padded": square_view,
        "resized": resized_view,
        "center_crop": crop_view,
        "expanded_crop": expanded_view,
        "contrast_enhanced": contrast_view,
        "sharpened": sharp_view,
        "grayscale": gray_view,
    }
