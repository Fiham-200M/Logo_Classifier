"""
Media Normalizer Module.
Handles multi-format media ingestion (PNG, JPEG, WebP, GIF, animated GIF).
Generates standardized forensic canvases:
  - Canvas A: White-background render (255, 255, 255)
  - Canvas B: Black-background render (0, 0, 0)
  - Canvas C: Alpha / Transparency mask
  - Canvas D: Normalized RGB image
"""

from typing import Dict, Any, Union, Optional, Tuple
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np

from preprocessing.gif_processor import process_gif


class MediaNormalizer:
    @staticmethod
    def normalize_media(
        image_input: Union[str, Path, Image.Image]
    ) -> Dict[str, Any]:
        """
        Ingests image/media and outputs standardized forensic canvases and metadata.

        Returns:
            Dict containing:
                - canvas_white: PIL.Image (RGB, white background)
                - canvas_black: PIL.Image (RGB, black background)
                - alpha_mask: PIL.Image (L mode, 0=transparent, 255=opaque)
                - normalized_rgb: PIL.Image (RGB canonical)
                - original: PIL.Image (original unaltered image)
                - metadata: Dict[str, Any]
        """
        file_format = "MEMORY"
        original_img = None

        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.exists():
                raise FileNotFoundError(f"Media file not found: {path}")
            original_img = Image.open(path)
            file_format = (original_img.format or path.suffix.strip(".").upper())
        elif isinstance(image_input, Image.Image):
            original_img = image_input
            file_format = original_img.format or "PNG"
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        is_animated = False
        frames_processed = 1
        duration_ms = 0

        key_frames: List[Image.Image] = []
        # Handle GIF / Animated media
        if file_format.upper() in ("GIF",) or getattr(original_img, "is_animated", False):
            gif_data = process_gif(original_img)
            is_animated = gif_data["is_animated"]
            frames_processed = gif_data["frame_count"]
            duration_ms = gif_data["total_duration_ms"]
            rgba_image = gif_data["composite_rgba"]
            raw_frames = gif_data.get("frames", [])
            if len(raw_frames) > 1:
                step = max(1, len(raw_frames) // 8)
                for idx in range(0, len(raw_frames), step):
                    frame_rgb = raw_frames[idx].convert("RGB")
                    # Filter out blank/uniform transition frames (e.g. all-black transition frames)
                    if float(np.array(frame_rgb).std()) > 5.0:
                        key_frames.append(frame_rgb)
        else:
            rgba_image = original_img.convert("RGBA")

        width, height = rgba_image.size

        # Extract Alpha Mask
        r, g, b, alpha = rgba_image.split()
        alpha_arr = np.array(alpha)
        has_alpha = bool(np.any(alpha_arr < 255))

        # Canvas A: White background render (255, 255, 255)
        white_bg = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        white_bg.paste(rgba_image, (0, 0), rgba_image)
        canvas_white = white_bg.convert("RGB")

        # Canvas B: Black background render (0, 0, 0)
        black_bg = Image.new("RGBA", (width, height), (0, 0, 0, 255))
        black_bg.paste(rgba_image, (0, 0), rgba_image)
        canvas_black = black_bg.convert("RGB")

        # Canvas C: Alpha / transparency mask
        alpha_mask = alpha.convert("L")

        # Canvas D: Normalized RGB image
        # If image has transparency, use dark-neutral render for canonical display;
        # otherwise, use high-fidelity RGB convert
        if has_alpha:
            canvas_d = canvas_black.copy()
        else:
            canvas_d = original_img.convert("RGB")

        metadata = {
            "file_format": file_format.upper(),
            "is_animated": is_animated,
            "frames_processed": frames_processed,
            "duration_ms": duration_ms,
            "alpha_channel_detected": has_alpha,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(height, 1), 4),
        }

        return {
            "canvas_white": canvas_white,
            "canvas_black": canvas_black,
            "alpha_mask": alpha_mask,
            "normalized_rgb": canvas_d,
            "original": original_img,
            "key_frames": key_frames,
            "metadata": metadata,
        }
