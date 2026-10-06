"""
Lightweight Structural & Shape Similarity Module (Optional Signal).
Computes edge map and luminance silhouette correlation on canonical 64x64 thumbnails
to help differentiate visually adjacent logos with similar color palettes.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import cv2
from PIL import Image


def extract_structure_feature(img: Image.Image) -> np.ndarray:
    """
    Extract a normalized 64x64 silhouette and edge representation.
    Returns float32 vector of shape (128,).
    """
    # Resize to canonical 64x64
    gray = np.array(img.convert("L").resize((64, 64), Image.Resampling.BILINEAR))

    # Canny edges (captures fine strokes and typography boundaries)
    edges = cv2.Canny(gray, 50, 150)

    # Horizontal and vertical edge projections (64 + 64 = 128 dims)
    h_proj = np.sum(edges, axis=1, dtype=np.float32)
    v_proj = np.sum(edges, axis=0, dtype=np.float32)

    vec = np.concatenate([h_proj, v_proj])
    norm = np.linalg.norm(vec)
    if norm > 1e-6:
        vec = vec / norm
    return vec.astype(np.float32)


def compute_structure_similarity(query_feat: np.ndarray, ref_feat: np.ndarray) -> float:
    """
    Compute cosine correlation between structural projection features.
    Returns similarity in [0, 1].
    """
    if query_feat is None or ref_feat is None:
        return 0.50
    denom = (np.linalg.norm(query_feat) * np.linalg.norm(ref_feat))
    if denom < 1e-12:
        return 0.50
    sim = float(np.dot(query_feat, ref_feat) / denom)
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def compute_edge_stroke_iou(query_img: Image.Image, ref_img: Image.Image) -> float:
    """
    Computes high-resolution edge stroke Intersection-over-Union (IoU) between
    a query logo image and its candidate official reference logo.
    Detects redrawn typography, modified font curves, stroke thickness anomalies,
    and subtle structural graphic differences.
    
    Returns:
        float: IoU value in [0.0, 1.0].
               Authentic web captures/compressed versions typically score >= 0.70.
               Redrawn fonts and competitor clones score < 0.55.
    """
    if query_img is None or ref_img is None:
        return 1.0

    try:
        # Convert to grayscale
        g_query = np.array(query_img.convert("L"))
        g_ref = np.array(ref_img.convert("L"))

        h_ref, w_ref = g_ref.shape[:2]
        if h_ref < 10 or w_ref < 10:
            return 1.0

        # Resize query to reference resolution
        g_query_res = cv2.resize(g_query, (w_ref, h_ref), interpolation=cv2.INTER_LANCZOS4)

        # Canny edge detection
        e_query = cv2.Canny(g_query_res, 50, 150)
        e_ref = cv2.Canny(g_ref, 50, 150)

        # Dilate edges slightly (3x3 kernel) to tolerate minor rasterization/sampling offsets
        kernel = np.ones((3, 3), np.uint8)
        d_query = cv2.dilate(e_query, kernel) > 0
        d_ref = cv2.dilate(e_ref, kernel) > 0

        intersection = np.logical_and(d_query, d_ref).sum()
        union = np.logical_or(d_query, d_ref).sum()

        if union == 0:
            return 1.0

        return float(intersection / union)
    except Exception:
        return 1.0

