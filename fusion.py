"""
Score Fusion and Decision Engine.
Re-exports from src.detection while maintaining 100% backward compatibility for existing callers.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.detection.fusion_engine import ScoreFusion as BaseScoreFusion
from src.detection.decision_engine import EvidenceDecisionEngine


class ScoreFusion:
    def __init__(
        self,
        siglip_weight: float = config.SIGLIP_WEIGHT,
        color_weight: float = config.COLOR_WEIGHT,
        ocr_weight: float = config.OCR_WEIGHT,
        structure_weight: float = config.STRUCTURE_WEIGHT,
    ):
        self._fusion = BaseScoreFusion(
            siglip_weight=siglip_weight,
            color_weight=color_weight,
            ocr_weight=ocr_weight,
            structure_weight=structure_weight,
        )
        self._decision_engine = EvidenceDecisionEngine()
        self.siglip_weight = self._fusion.base_siglip_weight
        self.color_weight = self._fusion.base_color_weight
        self.ocr_weight = self._fusion.base_ocr_weight
        self.structure_weight = self._fusion.base_structure_weight

    def fuse(
        self,
        siglip_scores: np.ndarray,
        color_scores: np.ndarray,
        ocr_scores: np.ndarray,
        structure_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return self._fusion.fuse(
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            structure_scores=structure_scores,
        )

    def evaluate_decision(
        self,
        brand_names: List[str],
        final_scores: np.ndarray,
        siglip_scores: np.ndarray,
        color_scores: np.ndarray,
        ocr_scores: np.ndarray,
        detected_ocr_texts: List[Dict[str, Any]],
        view_names_used: List[str],
        vlm_result: Optional[Dict[str, Any]] = None,
        vlm_texts: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return self._decision_engine.evaluate_decision(
            brand_names=brand_names,
            final_scores=final_scores,
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            detected_ocr_texts=detected_ocr_texts,
            view_names_used=view_names_used,
            vlm_result=vlm_result,
            vlm_texts=vlm_texts,
        )


__all__ = ["ScoreFusion", "EvidenceDecisionEngine"]
