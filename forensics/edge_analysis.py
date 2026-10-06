"""
Edge and Structural Forensics Module.
Performs sub-pixel edge extraction, alignment, morphological dilation,
and edge stroke Intersection-over-Union (IoU) calculation.
Catches redrawn typography, modified curves, altered stroke thickness,
and subtle structural graphic differences.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from skimage.metrics import structural_similarity as ssim


class EdgeAnalyzer:
    @staticmethod
    def extract_edges(
        image: Image.Image,
        low_thresh: int = 50,
        high_thresh: int = 150
    ) -> np.ndarray:
        """
        Extracts clean Canny edge map from an image.
        Returns uint8 array of shape (H, W).
        """
        gray = np.array(image.convert("L"))
        # Bilateral blur to preserve stroke edges while suppressing JPEG noise
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)
        edges = cv2.Canny(blurred, low_thresh, high_thresh)
        return edges

    @staticmethod
    def crop_content(img: Image.Image, margin: int = 5) -> Image.Image:
        """Crops away uniform border padding to ensure scale-invariant stroke alignment."""
        arr = np.array(img.convert("RGB"))
        corners = [
            arr[0, 0].astype(np.float32),
            arr[0, -1].astype(np.float32),
            arr[-1, 0].astype(np.float32),
            arr[-1, -1].astype(np.float32),
        ]
        c_med = np.median(corners, axis=0)
        max_diff = float(np.max([np.linalg.norm(c - c_med) for c in corners]))
        if max_diff < 25.0:
            diff = np.linalg.norm(arr.astype(np.float32) - c_med, axis=-1)
            mask = diff > 20.0
            y_indices, x_indices = np.where(mask)
            if len(y_indices) > 50:
                x1 = max(0, int(np.min(x_indices)) - margin)
                y1 = max(0, int(np.min(y_indices)) - margin)
                x2 = min(img.width, int(np.max(x_indices)) + margin)
                y2 = min(img.height, int(np.max(y_indices)) + margin)
                if (x2 - x1) > 20 and (y2 - y1) > 20:
                    return img.crop((x1, y1, x2, y2))
        return img

    @staticmethod
    def _compare_single(
        query_img: Image.Image,
        ref_img: Image.Image,
        kernel_size: int = 3,
    ) -> Dict[str, Any]:
        g_ref = np.array(ref_img.convert("L"))
        h_ref, w_ref = g_ref.shape[:2]

        g_query = np.array(query_img.convert("L"))

        # Resize query to reference resolution using high-quality Lanczos4 interpolation
        g_query_res = cv2.resize(g_query, (w_ref, h_ref), interpolation=cv2.INTER_LANCZOS4)

        # Check for slight phase correlation / translation alignment
        try:
            shift, _ = cv2.phaseCorrelate(
                np.float32(g_query_res), np.float32(g_ref)
            )
            dx, dy = shift
            if abs(dx) < 15 and abs(dy) < 15:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                g_query_res = cv2.warpAffine(
                    g_query_res, M, (w_ref, h_ref), borderMode=cv2.BORDER_REPLICATE
                )
        except Exception:
            pass

        e_query = cv2.Canny(g_query_res, 50, 150)
        e_ref = cv2.Canny(g_ref, 50, 150)

        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        d_query = cv2.dilate(e_query, kernel) > 0
        d_ref = cv2.dilate(e_ref, kernel) > 0

        intersection = np.logical_and(d_query, d_ref).sum()
        union = np.logical_or(d_query, d_ref).sum()
        edge_iou = float(intersection / union) if union > 0 else 1.0

        new_edges = np.logical_and(d_query, ~d_ref).sum()
        injected_edge_ratio = float(new_edges / (d_ref.sum() + 1e-6))

        try:
            ssim_val, _ = ssim(g_query_res, g_ref, full=True)
            ssim_score = float(ssim_val)
        except Exception:
            ssim_score = 0.50

        diff_mag = np.abs(g_query_res.astype(np.float32) - g_ref.astype(np.float32))
        heatmap = cv2.applyColorMap(np.uint8(np.clip(diff_mag * 3, 0, 255)), cv2.COLORMAP_JET)

        return {
            "edge_stroke_iou": round(edge_iou, 4),
            "injected_edge_ratio": round(injected_edge_ratio, 4),
            "ssim_score": round(ssim_score, 4),
            "edge_query": e_query,
            "edge_ref": e_ref,
            "aligned_query": g_query_res,
            "diff_heatmap": heatmap,
        }

    @staticmethod
    def align_and_compare(
        query_img: Image.Image,
        ref_img: Image.Image,
        kernel_size: int = 3,
    ) -> Dict[str, Any]:
        """
        Aligns query to reference dimensions and computes edge stroke IoU.
        Evaluates both raw and content-cropped alignment to remain invariant
        to padding variations while strictly detecting redraws and mutations.
        """
        raw_res = EdgeAnalyzer._compare_single(query_img, ref_img, kernel_size=kernel_size)
        if raw_res["edge_stroke_iou"] >= 0.70:
            return raw_res

        # Try content-cropped comparison for padded variants
        c_query = EdgeAnalyzer.crop_content(query_img)
        c_ref = EdgeAnalyzer.crop_content(ref_img)
        if c_query.size != query_img.size or c_ref.size != ref_img.size:
            cropped_res = EdgeAnalyzer._compare_single(c_query, c_ref, kernel_size=kernel_size)
            if cropped_res["edge_stroke_iou"] > raw_res["edge_stroke_iou"]:
                return cropped_res

        return raw_res
