"""
Color Forensics Module.
Performs non-transparent pixel color analysis:
  - Dominant color extraction
  - HSV, RGB, and CIE LAB statistics
  - 3D HSV Color Histogram intersection
  - CIE LAB Delta E 2000 color distance
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2
from PIL import Image
from skimage.color import rgb2lab, deltaE_ciede2000


class ColorAnalyzer:
    @staticmethod
    def extract_color_profile(
        image: Image.Image,
        alpha_mask: Optional[Image.Image] = None,
        num_dominant: int = 5,
    ) -> Dict[str, Any]:
        """
        Extracts comprehensive color statistics from non-transparent foreground pixels.
        """
        rgb_arr = np.array(image.convert("RGB"))
        h, w, _ = rgb_arr.shape

        # Filter out transparent or uniform canvas background pixels
        has_real_alpha = False
        if alpha_mask is not None:
            a_arr = np.array(alpha_mask.convert("L"))
            if np.any(a_arr < 250):
                fg_mask = a_arr > 30
                has_real_alpha = True

        if not has_real_alpha:
            # Check corners for solid canvas background (black, white, dark gray, light gray, etc.)
            corners = [
                rgb_arr[0, 0].astype(np.float32),
                rgb_arr[0, -1].astype(np.float32),
                rgb_arr[-1, 0].astype(np.float32),
                rgb_arr[-1, -1].astype(np.float32)
            ]
            c_med = np.median(corners, axis=0)
            max_c_diff = float(np.max([np.linalg.norm(c - c_med) for c in corners]))
            if max_c_diff < 25.0:
                # Uniform background detected — exclude pixels close to background color
                dist = np.linalg.norm(rgb_arr.astype(np.float32) - c_med, axis=-1)
                fg_mask = dist > 20.0
            else:
                fg_mask = np.ones((h, w), dtype=bool)

        if np.sum(fg_mask) < 30:
            fg_mask = np.ones((h, w), dtype=bool)

        fg_rgb = rgb_arr[fg_mask]  # Shape: (N, 3)

        # 1. RGB statistics
        rgb_mean = np.mean(fg_rgb, axis=0).tolist()
        rgb_std = np.std(fg_rgb, axis=0).tolist()

        # 2. HSV statistics and 3D histogram (8x8x8 = 512 bins)
        hsv_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2HSV)
        fg_hsv = hsv_arr[fg_mask]
        hsv_mean = np.mean(fg_hsv, axis=0).tolist()

        # 3D HSV histogram normalized
        hsv_flat = fg_hsv.astype(np.float32)
        hist, _ = np.histogramdd(
            hsv_flat,
            bins=(8, 8, 8),
            range=((0, 180), (0, 256), (0, 256))
        )
        hist_sum = hist.sum()
        hist_norm = (hist / (hist_sum + 1e-12)).astype(np.float32).flatten()

        # 3. CIE LAB conversion & statistics
        fg_rgb_unit = (fg_rgb / 255.0).astype(np.float32).reshape(-1, 1, 3)
        fg_lab = rgb2lab(fg_rgb_unit).reshape(-1, 3)
        lab_mean = np.mean(fg_lab, axis=0).tolist()

        # 4. Dominant Colors via K-Means in RGB
        dominant_colors_rgb = []
        dominant_colors_lab = []
        if len(fg_rgb) >= num_dominant:
            # Subsample for speed if image is large using fixed deterministic seed
            if len(fg_rgb) > 5000:
                rng = np.random.RandomState(42)
                indices = rng.choice(len(fg_rgb), 5000, replace=False)
                sample_pts = fg_rgb[indices].astype(np.float32)
            else:
                sample_pts = fg_rgb.astype(np.float32)

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, centers = cv2.kmeans(
                sample_pts, num_dominant, None, criteria, 3, cv2.KMEANS_PP_CENTERS
            )
            # Count cluster weights
            counts = np.bincount(labels.flatten())
            sorted_idx = np.argsort(counts)[::-1]
            centers_sorted = centers[sorted_idx]

            dominant_colors_rgb = centers_sorted.tolist()
            # Convert dominant to LAB
            dom_unit = (centers_sorted / 255.0).astype(np.float32).reshape(-1, 1, 3)
            dominant_colors_lab = rgb2lab(dom_unit).reshape(-1, 3).tolist()

        return {
            "rgb_mean": rgb_mean,
            "rgb_std": rgb_std,
            "hsv_mean": hsv_mean,
            "hsv_histogram": hist_norm,
            "lab_mean": lab_mean,
            "dominant_rgb": dominant_colors_rgb,
            "dominant_lab": dominant_colors_lab,
        }

    @staticmethod
    def compare_color_profiles(
        query_profile: Dict[str, Any],
        ref_profile: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Compares two color profiles.
        Calculates:
          - histogram_similarity: [0.0, 1.0] (Intersection of 3D HSV histograms)
          - cielab_delta_e: CIE LAB Delta E 2000 difference using palette Chamfer distance and mean LAB
        """
        # 1. 3D HSV Histogram Intersection
        q_hist = query_profile.get("hsv_histogram")
        r_hist = ref_profile.get("hsv_histogram")
        if q_hist is not None and r_hist is not None:
            hist_sim = float(np.sum(np.minimum(q_hist, r_hist)))
            hist_sim = max(0.0, min(1.0, hist_sim))
        else:
            hist_sim = 0.50

        # 2. CIE LAB Delta E 2000
        q_m = np.array(query_profile.get("lab_mean", [50, 0, 0]), dtype=np.float32).reshape(1, 1, 3)
        r_m = np.array(ref_profile.get("lab_mean", [50, 0, 0]), dtype=np.float32).reshape(1, 1, 3)
        de_mean = float(deltaE_ciede2000(q_m, r_m)[0, 0])

        q_dom = query_profile.get("dominant_lab", [])
        r_dom = ref_profile.get("dominant_lab", [])

        if q_dom and r_dom:
            # Multi-color nearest-neighbor Chamfer distance across dominant palettes
            dists = []
            for q in q_dom:
                q_arr = np.array(q, dtype=np.float32).reshape(1, 1, 3)
                min_d = min(float(deltaE_ciede2000(q_arr, np.array(r, dtype=np.float32).reshape(1, 1, 3))[0, 0]) for r in r_dom)
                dists.append(min_d)
            for r in r_dom:
                r_arr = np.array(r, dtype=np.float32).reshape(1, 1, 3)
                min_d = min(float(deltaE_ciede2000(r_arr, np.array(q, dtype=np.float32).reshape(1, 1, 3))[0, 0]) for q in q_dom)
                dists.append(min_d)
            de_palette = float(np.mean(dists))
            delta_e = 0.5 * de_mean + 0.5 * de_palette
        else:
            delta_e = de_mean

        # If 3D color histogram is essentially identical (>= 0.95), bounded by de_mean
        if hist_sim >= 0.95:
            delta_e = min(delta_e, de_mean + 0.5)

        return {
            "histogram_similarity": round(hist_sim, 4),
            "cielab_delta_e": round(delta_e, 2),
        }
