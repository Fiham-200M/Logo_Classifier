"""
Multi-Stage Anti-Phishing Brand Detection Pipeline.
Executes efficient staged inference:
  Stage 1: Fast Visual Screening (SigLIP multi-view + Color matching)
           -> Fast-path exit if visual match is overwhelming.
  Stage 2: Text Verification (PaddleOCR / EasyOCR fuzzy candidate matching)
  Stage 3: Vision-Language Model Verification (Conditional on ambiguity / conflict / protected)
  Stage 4: Evidence-Aware Decision & Explainability
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
from PIL import Image

import sys
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.preprocessing.image_preprocessor import generate_preprocessing_views, load_image_safe
from src.models.siglip_engine import SigLIPEngine
from src.models.color_engine import extract_color_features, compare_color_features
from src.models.ocr_engine import OCREngine
from src.models.vlm_engine import query_vlm_analysis, query_vlm_fallback, query_vlm_text_extraction
from src.models.classifier_engine import ClassifierEngine
from src.database.brand_database import BrandDatabase
from src.detection.fusion_engine import ScoreFusion
from src.detection.decision_engine import EvidenceDecisionEngine
from structure_features import compute_edge_stroke_iou


class MultiStagePipeline:
    def __init__(
        self,
        db: Optional[BrandDatabase] = None,
        siglip: Optional[SigLIPEngine] = None,
        ocr: Optional[OCREngine] = None,
        classifier: Optional[ClassifierEngine] = None,
        use_vlm: bool = False,
        use_classifier: bool = True,
    ):
        self.use_vlm = use_vlm or getattr(config, "VLM_ENABLED", False)
        self.use_classifier = use_classifier and getattr(config, "CLASSIFIER_ENABLED", True)
        self.db = db or BrandDatabase()
        self.siglip = siglip or SigLIPEngine()
        self.ocr = ocr or OCREngine()
        self.classifier = classifier or (ClassifierEngine() if self.use_classifier else None)
        self.fusion = ScoreFusion()
        self.decision_engine = EvidenceDecisionEngine()

        self.fast_path_score = getattr(config, "PIPELINE_FAST_PATH_SCORE", 0.92)
        self.fast_path_margin = getattr(config, "PIPELINE_FAST_PATH_MARGIN", 0.08)

    def process(
        self,
        image_source: Union[str, Path, Image.Image],
        force_vlm: bool = False,
        skip_ocr: bool = False,
        disable_fast_path: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes the staged brand detection pipeline on an image.
        """
        # ============================================================
        # STAGE 1: FAST VISUAL SCREENING (Always runs)
        # ============================================================
        img = load_image_safe(image_source)
        views = generate_preprocessing_views(img)
        view_names = list(views.keys())
        num_brands = len(self.db.brand_names)

        # 1a. SigLIP multi-view extraction & multi-reference scoring
        view_embeddings = self.siglip.extract_multi_view_embeddings(views)
        if hasattr(self.db, "score_siglip_multi_ref"):
            siglip_scores, view_breakdown = self.db.score_siglip_multi_ref(
                view_embeddings, strategy=config.SIGLIP_ENSEMBLE_STRATEGY
            )
        else:
            siglip_scores, view_breakdown = self.siglip.score_against_references(
                views, self.db.reference_embeddings, strategy=config.SIGLIP_ENSEMBLE_STRATEGY
            )

        # 1b. Color feature extraction & scoring
        query_color = extract_color_features(img)
        if hasattr(self.db, "score_color_multi_ref"):
            color_scores = self.db.score_color_multi_ref(query_color)
        elif query_color is not None:
            color_scores = np.zeros(num_brands, dtype=np.float32)
            for i, brand in enumerate(self.db.brand_names):
                ref_color = self.db.color_database.get(brand)
                if ref_color is not None:
                    color_scores[i] = compare_color_features(query_color, ref_color)
                else:
                    color_scores[i] = 0.50
        else:
            color_scores = np.full(num_brands, 0.50, dtype=np.float32)


        # 1c. Brand Classifier scoring (Phase 9)
        classifier_pred = None
        classifier_scores = None
        if self.use_classifier and self.classifier and self.classifier.loaded:
            prim_emb = view_embeddings.get("raw_rgb", next(iter(view_embeddings.values())))
            classifier_pred = self.classifier.predict(prim_emb)
            classifier_scores = self.classifier.score_candidates(prim_emb, self.db.brand_names)

        # Initial visual-only composite
        vis_weights = self.fusion.base_siglip_weight + self.fusion.base_color_weight
        w_sig = self.fusion.base_siglip_weight / (vis_weights + 1e-12)
        w_col = self.fusion.base_color_weight / (vis_weights + 1e-12)
        stage1_scores = (w_sig * siglip_scores + w_col * color_scores).astype(np.float32)

        sorted_s1 = np.argsort(stage1_scores)[::-1]
        top1_s1_idx = sorted_s1[0]
        top2_s1_idx = sorted_s1[1] if num_brands > 1 else top1_s1_idx
        top1_s1_score = float(stage1_scores[top1_s1_idx])
        s1_margin = float(top1_s1_score - stage1_scores[top2_s1_idx])

        # Check fast-path criteria
        can_fast_path = (
            not disable_fast_path
            and not force_vlm
            and not self.use_vlm
            and top1_s1_score >= self.fast_path_score
            and s1_margin >= self.fast_path_margin
        )
        # When VLM is always-on, fast-path only skips OCR, not VLM
        can_skip_ocr_fast = can_fast_path

        stage_executed = ["STAGE_1_VISUAL"]

        # ============================================================
        # STAGE 2: TEXT VERIFICATION (OCR)
        # ============================================================
        ocr_scores = np.full(num_brands, config.OCR_NEUTRAL_SCORE, dtype=np.float32)
        detected_texts: List[Dict[str, Any]] = []
        ocr_found_text = False

        if not can_skip_ocr_fast and not skip_ocr:
            stage_executed.append("STAGE_2_OCR")
            ocr_scores, detected_texts = self.ocr.score_candidates(
                views, self.db.brand_names, self.db.ocr_database
            )
            ocr_found_text = len(detected_texts) > 0

        # Interim score fusion (SigLIP + Color + OCR + Classifier)
        interim_scores = self.fusion.fuse(
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            classifier_scores=classifier_scores,
        )

        # ============================================================
        # STAGE 3: VLM VERIFICATION (Conditional)
        # ============================================================
        vlm_result = None
        vlm_texts = []
        vlm_scores = np.zeros(num_brands, dtype=np.float32)

        needs_vlm, vlm_trigger_reason = self.decision_engine.check_needs_vlm(
            final_scores=interim_scores,
            siglip_scores=siglip_scores,
            ocr_scores=ocr_scores,
            brand_names=self.db.brand_names,
            ocr_found_text=ocr_found_text,
        )

        # When VLM is globally enabled (use_vlm=True), ALWAYS run VLM for every image.
        # Only conditionally skip when VLM is off and not explicitly forced.
        run_vlm = self.use_vlm or force_vlm or needs_vlm
        if run_vlm:
            stage_executed.append("STAGE_3_VLM")
            # Candidates for VLM context
            sorted_interim = np.argsort(interim_scores)[::-1]
            top_k_candidates = [
                self.db.brand_names[i]
                for i in sorted_interim[:getattr(config, "VLM_TOP_CANDIDATES", 4)]
            ]

            top_cand = top_k_candidates[0] if top_k_candidates else None
            ref_img = None
            if top_cand and hasattr(self.db, "brand_to_filename") and top_cand in self.db.brand_to_filename:
                ref_fname = self.db.brand_to_filename[top_cand]
                ref_path = config.LOGOS_DIR / ref_fname
                if not ref_path.exists():
                    for ext in [".png", ".webp", ".jpg", ".jpeg"]:
                        alt = config.LOGOS_DIR / f"{Path(ref_fname).stem}{ext}"
                        if alt.exists():
                            ref_path = alt
                            break
                if ref_path.exists():
                    try:
                        ref_img = Image.open(ref_path).convert("RGB")
                    except Exception:
                        pass

            try:
                # 3a. Disambiguation & side-by-side reference verification
                vlm_res = query_vlm_fallback(
                    img,
                    top_k_candidates,
                    reference_image=ref_img,
                    reference_brand=top_cand,
                )
                if vlm_res:
                    vlm_result = vlm_res
                    cand = vlm_res.get("candidate")
                    conf = float(vlm_res.get("confidence", 0.8))
                    if cand and cand in self.db.brand_to_idx:
                        vlm_scores[self.db.brand_to_idx[cand]] = conf

                # 3b. Extract independent VLM text reading
                vlm_texts = query_vlm_text_extraction(img) or []
            except Exception as e:
                print(f"[MultiStagePipeline] VLM evaluation warning: {e}")

        # ============================================================
        # STAGE 4: FINAL FUSION & EVIDENCE DECISION
        # ============================================================
        stage_executed.append("STAGE_4_FUSION")
        final_scores = self.fusion.fuse(
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            classifier_scores=classifier_scores,
            vlm_scores=vlm_scores if vlm_result is not None else None,
        )

        # High-resolution edge stroke consistency verification against top candidate reference
        edge_stroke_iou = None
        top_cand_final = self.db.brand_names[int(np.argmax(final_scores))]
        if top_cand_final and hasattr(self.db, "brand_to_filename") and top_cand_final in self.db.brand_to_filename:
            ref_fname = self.db.brand_to_filename[top_cand_final]
            ref_path = config.LOGOS_DIR / ref_fname
            if not ref_path.exists():
                for ext in [".png", ".webp", ".jpg", ".jpeg"]:
                    alt = config.LOGOS_DIR / f"{Path(ref_fname).stem}{ext}"
                    if alt.exists():
                        ref_path = alt
                        break
            if ref_path.exists():
                try:
                    ref_img_eval = Image.open(ref_path).convert("RGB")
                    edge_stroke_iou = compute_edge_stroke_iou(img, ref_img_eval)
                except Exception:
                    pass

        decision_payload = self.decision_engine.evaluate_decision(
            brand_names=self.db.brand_names,
            final_scores=final_scores,
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            classifier_scores=classifier_scores,
            classifier_result=classifier_pred,
            detected_ocr_texts=detected_texts,
            view_names_used=view_names,
            vlm_result=vlm_result,
            vlm_texts=vlm_texts,
            edge_stroke_iou=edge_stroke_iou,
        )

        decision_payload["stages_executed"] = stage_executed
        decision_payload["fast_path_taken"] = can_fast_path
        decision_payload["vlm_trigger_reason"] = vlm_trigger_reason if needs_vlm else None
        if isinstance(image_source, (str, Path)):
            decision_payload["image_path"] = str(image_source)

        return decision_payload
