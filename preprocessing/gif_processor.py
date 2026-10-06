"""
GIF Processing Module.
Extracts and synthesizes all meaningful frames from static and animated GIFs.
Performs alpha-aware compositing across frames to generate a unified visual representation.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
from PIL import Image, ImageSequence
import numpy as np


class GIFProcessor:
    @staticmethod
    def process_gif(gif_path_or_image: Any) -> Dict[str, Any]:
        return process_gif(gif_path_or_image)


def process_gif(gif_path_or_image: Any) -> Dict[str, Any]:
    """
    Processes an animated or static GIF.
    Inspects all frames, extracts timing and transparency, and produces
    an alpha-aware multi-frame composite image representing all active elements.

    Returns:
        Dict with:
            - is_animated: bool
            - frame_count: int
            - durations: List[int]
            - total_duration_ms: int
            - frames: List[Image.Image] (RGBA)
            - composite_rgba: Image.Image (RGBA representation of all frames)
            - width: int
            - height: int
            - has_transparency: bool
    """
    if isinstance(gif_path_or_image, (str, Path)):
        img = Image.open(gif_path_or_image)
    else:
        img = gif_path_or_image

    is_animated = getattr(img, "is_animated", False)
    frame_count = getattr(img, "n_frames", 1)

    frames: List[Image.Image] = []
    durations: List[int] = []
    has_transparency = False

    width, height = img.size

    for frame in ImageSequence.Iterator(img):
        # Extract duration
        duration = frame.info.get("duration", 100)
        durations.append(duration)

        # Convert to RGBA
        frame_rgba = frame.convert("RGBA")
        frames.append(frame_rgba)

        # Check alpha channel
        alpha = np.array(frame_rgba.split()[-1])
        if np.any(alpha < 255):
            has_transparency = True

    # Multi-frame alpha-aware compositing:
    # If animated, merge frames so all persistent or moving logo elements are represented
    if len(frames) == 1:
        composite_rgba = frames[0]
    else:
        # Create base canvas with transparency
        base = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for f in frames:
            base.paste(f, (0, 0), f)
        composite_rgba = base

    return {
        "is_animated": is_animated,
        "frame_count": len(frames),
        "durations": durations,
        "total_duration_ms": sum(durations),
        "frames": frames,
        "composite_rgba": composite_rgba,
        "width": width,
        "height": height,
        "has_transparency": has_transparency,
    }
