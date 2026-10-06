"""
Color Feature Extraction and Matching Module.
Directly wraps and exposes the validated 238-dimensional color feature system
from validate_color_database.py:
- HSV histogram (52D)
- RGB histogram (24D)
- Color statistics (18D)
- Spatial grid color distribution (112D)
- Dominant color clusters (32D)
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image

import validate_color_database as vcd

COLOR_GROUP_WEIGHTS = vcd.WEIGHTS


def extract_color_features(img_or_path: Union[str, Path, Image.Image]) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract exact 238-dim color features matching reference_embeddings/color_database.pt.
    Accepts file path or PIL Image.
    """
    try:
        if isinstance(img_or_path, (str, Path)):
            return vcd.calculate_color_features(str(img_or_path))

        elif isinstance(img_or_path, Image.Image):
            frames = []
            if getattr(img_or_path, "is_animated", False):
                from PIL import ImageSequence
                for f in ImageSequence.Iterator(img_or_path):
                    frames.append(f.convert("RGBA"))
            else:
                frames.append(img_or_path.convert("RGBA"))

            features = []
            for frame in frames:
                res = vcd.calculate_frame_features(frame)
                if res is not None:
                    features.append(res)

            if not features:
                return None

            output = {}
            for key in ["feature", "hsv", "rgb", "stats", "spatial", "dominant"]:
                output[key] = np.mean(
                    np.stack([x[key] for x in features]),
                    axis=0
                ).astype(np.float32)
            return output
    except Exception as e:
        return None
    return None


def compare_color_features(query_feat: Dict[str, np.ndarray], ref_feat: Dict[str, np.ndarray]) -> float:
    """
    Compare query and reference color features across all 5 groups using established weights.
    Returns similarity in [0, 1].
    """
    if query_feat is None or ref_feat is None:
        return 0.50
    score, _ = vcd.color_similarity(query_feat, ref_feat)
    return float(np.clip(score, 0.0, 1.0))
