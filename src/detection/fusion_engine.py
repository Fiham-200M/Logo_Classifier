"""
Score Fusion Engine.
Dynamically combines SigLIP, Color, OCR, VLM, and optional Structure signals into
a composite score, automatically re-normalizing weights based on available modalities.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


class ScoreFusion:
    def __init__(
        self,
        siglip_weight: float = config.SIGLIP_WEIGHT,
        color_weight: float = config.COLOR_WEIGHT,
        ocr_weight: float = config.OCR_WEIGHT,
        classifier_weight: float = getattr(config, "CLASSIFIER_WEIGHT", 0.0),
        vlm_weight: float = getattr(config, "VLM_WEIGHT", 0.0),
        structure_weight: float = config.STRUCTURE_WEIGHT,
    ):
        self.base_siglip_weight = siglip_weight
        self.base_color_weight = color_weight
        self.base_ocr_weight = ocr_weight
        self.base_classifier_weight = classifier_weight
        self.base_vlm_weight = vlm_weight
        self.base_structure_weight = structure_weight

    def fuse(
        self,
        siglip_scores: np.ndarray,
        color_scores: np.ndarray,
        ocr_scores: np.ndarray,
        classifier_scores: Optional[np.ndarray] = None,
        vlm_scores: Optional[np.ndarray] = None,
        structure_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute weighted composite score across candidate brands.
        Dynamically adjusts weights based on whether Classifier, VLM, and Structure signals are provided.
        """
        w_siglip = self.base_siglip_weight
        w_color = self.base_color_weight
        w_ocr = self.base_ocr_weight
        w_classifier = (
            self.base_classifier_weight
            if (classifier_scores is not None and np.any(classifier_scores > 0))
            else 0.0
        )
        w_vlm = self.base_vlm_weight if (vlm_scores is not None and np.any(vlm_scores > 0)) else 0.0
        w_struct = self.base_structure_weight if structure_scores is not None else 0.0

        total_weight = w_siglip + w_color + w_ocr + w_classifier + w_vlm + w_struct
        if total_weight <= 0:
            total_weight = 1.0

        fused = (
            (w_siglip / total_weight) * siglip_scores
            + (w_color / total_weight) * color_scores
            + (w_ocr / total_weight) * ocr_scores
        )

        if classifier_scores is not None and w_classifier > 0:
            fused += (w_classifier / total_weight) * classifier_scores

        if vlm_scores is not None and w_vlm > 0:
            fused += (w_vlm / total_weight) * vlm_scores

        if structure_scores is not None and w_struct > 0:
            fused += (w_struct / total_weight) * structure_scores

        return np.clip(fused, 0.0, 1.0).astype(np.float32)
