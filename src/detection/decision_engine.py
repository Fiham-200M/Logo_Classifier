"""
Evidence-Aware Decision Engine.
Evaluates multi-modal evidence across visual (SigLIP), color, OCR, and VLM signals
to produce robust anti-phishing brand identification decisions with minimal false negatives.

Core principles:
- OCR absence/failure alone NEVER rejects a brand (many logos contain no text).
- Color mismatch alone NEVER rejects a brand (flags for REVIEW as possible color spoof).
- Conflicting signals or ambiguous margins trigger VLM verification.
- Protected brands receive lower thresholds and extra sensitivity.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

import sys
from pathlib import Path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


class EvidenceDecisionEngine:
    def __init__(
        self,
        match_score_threshold: float = config.MATCH_SCORE_THRESHOLD,
        match_margin_threshold: float = config.MATCH_MARGIN_THRESHOLD,
        review_score_threshold: float = config.REVIEW_SCORE_THRESHOLD,
        review_margin_threshold: float = config.REVIEW_MARGIN_THRESHOLD,
        strong_visual_threshold: float = getattr(config, "STRONG_VISUAL_THRESHOLD", 0.85),
        strong_color_threshold: float = getattr(config, "STRONG_COLOR_THRESHOLD", 0.80),
        strong_ocr_threshold: float = getattr(config, "STRONG_OCR_THRESHOLD", 0.80),
        protected_brands: List[str] = config.PROTECTED_BRANDS,
        protected_score_threshold: float = config.PROTECTED_SCORE_THRESHOLD,
    ):
        self.match_score_threshold = match_score_threshold
        self.match_margin_threshold = match_margin_threshold
        self.review_score_threshold = review_score_threshold
        self.review_margin_threshold = review_margin_threshold
        self.strong_visual_threshold = strong_visual_threshold
        self.strong_color_threshold = strong_color_threshold
        self.strong_ocr_threshold = strong_ocr_threshold
        self.protected_brands = set(protected_brands)
        self.protected_score_threshold = protected_score_threshold

    def check_needs_vlm(
        self,
        final_scores: np.ndarray,
        siglip_scores: np.ndarray,
        ocr_scores: np.ndarray,
        brand_names: List[str],
        ocr_found_text: bool,
    ) -> Tuple[bool, str]:
        """
        Determines if Stage 3 (VLM verification) should be invoked.
        Returns (needs_vlm, reason_for_trigger).
        """
        if len(final_scores) == 0:
            return False, ""

        sorted_indices = np.argsort(final_scores)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1] if len(sorted_indices) > 1 else top1_idx

        top1_score = float(final_scores[top1_idx])
        top2_score = float(final_scores[top2_idx])
        margin = top1_score - top2_score
        top1_brand = brand_names[top1_idx]

        # 1. Protected brand in top-2 candidates with plausible score
        for idx in sorted_indices[:3]:
            cand = brand_names[idx]
            cand_score = float(final_scores[idx])
            if cand in self.protected_brands and cand_score >= self.protected_score_threshold:
                if margin < getattr(config, "PROTECTED_VLM_TRIGGER_MARGIN", 0.040):
                    return True, f"Protected brand '{cand}' in top candidates with close margin ({margin:.4f})"

        # 2. Competitor Logo Hijacking / Text-Visual Anomaly:
        # Visual shape matches our brand, but OCR detected distinct competitor text (e.g. NAGA188)
        top1_siglip = float(siglip_scores[top1_idx])
        top1_ocr = float(ocr_scores[top1_idx])
        if ocr_found_text and top1_siglip >= 0.60 and top1_ocr <= 0.55:
            return True, f"Visual-Text conflict: Visual resembles '{top1_brand}' ({top1_siglip:.3f}), but detected text does not match"

        # 3. Color Alteration / B&W Spoof:
        # Visual structure matches, but color features deviate significantly (color shift or grayscale)
        top1_color = float(ocr_scores[top1_idx])  # will also check color in evaluate_decision
        if top1_siglip >= 0.70 and float(final_scores[top1_idx]) >= self.review_score_threshold:
            pass

        # 4. Ambiguity: top 2 candidates are very close
        ambiguity_margin = getattr(config, "VLM_AMBIGUITY_MARGIN", 0.025)
        if top1_score >= self.review_score_threshold and margin < ambiguity_margin:
            top2_brand = brand_names[top2_idx]
            return True, f"Score ambiguity between '{top1_brand}' ({top1_score:.3f}) and '{top2_brand}' ({top2_score:.3f})"

        # 5. Disagreement: Visual strongly points to one brand, but OCR detected text strongly matches another
        if ocr_found_text:
            vis_top_idx = int(np.argmax(siglip_scores))
            ocr_top_idx = int(np.argmax(ocr_scores))
            if vis_top_idx != ocr_top_idx:
                vis_score = float(siglip_scores[vis_top_idx])
                ocr_score = float(ocr_scores[ocr_top_idx])
                if vis_score >= 0.70 and ocr_score >= 0.70:
                    return True, f"Disagreement: Visual favors '{brand_names[vis_top_idx]}' ({vis_score:.2f}) while OCR favors '{brand_names[ocr_top_idx]}' ({ocr_score:.2f})"

        # 6. Moderate evidence across all signals
        if 0.65 <= top1_score < self.match_score_threshold and margin < self.match_margin_threshold:
            return True, f"Moderate confidence ({top1_score:.3f}) requires secondary visual-language confirmation"

        return False, ""

    def evaluate_decision(
        self,
        brand_names: List[str],
        final_scores: np.ndarray,
        siglip_scores: np.ndarray,
        color_scores: np.ndarray,
        ocr_scores: np.ndarray,
        classifier_scores: Optional[np.ndarray] = None,
        classifier_result: Optional[Dict[str, Any]] = None,
        detected_ocr_texts: Optional[List[Dict[str, Any]]] = None,
        view_names_used: Optional[List[str]] = None,
        vlm_result: Optional[Dict[str, Any]] = None,
        vlm_texts: Optional[List[str]] = None,
        edge_stroke_iou: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate full multi-modal evidence to reach final decision.
        """
        detected_ocr_texts = detected_ocr_texts or []
        view_names_used = view_names_used or []

        num_brands = len(brand_names)
        if num_brands == 0:
            return {
                "prediction": "unknown",
                "decision": "UNKNOWN",
                "final_score": 0.0,
                "margin": 0.0,
                "reason": "Empty brand database",
                "signals": {},
                "top_candidates": [],
            }

        sorted_indices = np.argsort(final_scores)[::-1]
        top1_idx = sorted_indices[0]
        top2_idx = sorted_indices[1] if num_brands > 1 else top1_idx

        top1_brand = brand_names[top1_idx]
        top2_brand = brand_names[top2_idx]

        top1_score = float(final_scores[top1_idx])
        top2_score = float(final_scores[top2_idx])
        margin = float(top1_score - top2_score)

        top1_siglip = float(siglip_scores[top1_idx])
        top1_color = float(color_scores[top1_idx])
        top1_ocr = float(ocr_scores[top1_idx])
        top1_cls = float(classifier_scores[top1_idx]) if classifier_scores is not None else 0.0

        is_protected = top1_brand in self.protected_brands
        ocr_detected_any = len(detected_ocr_texts) > 0

        decision = "UNKNOWN"
        reason = ""

        # --- Rule 0: VLM Disambiguation / Verification Override ---
        if vlm_result and (vlm_result.get("detected_brand") or vlm_result.get("candidate")):
            vlm_cand = vlm_result.get("detected_brand") or vlm_result.get("candidate")
            vlm_conf = float(vlm_result.get("brand_confidence") or vlm_result.get("confidence") or 0.0)
            vlm_reason = vlm_result.get("reasoning") or vlm_result.get("reason") or ""

            if vlm_cand in brand_names:
                vlm_idx = brand_names.index(vlm_cand)
                vlm_ocr_score = float(ocr_scores[vlm_idx])

                # Cross-check: If OCR detected text but strongly disagrees with VLM's candidate,
                # this is likely a number-squatter / competitor (e.g., "SURGA555" vs "surga55").
                # OCR score <= 0.35 means detected text does NOT match VLM's candidate.
                if ocr_detected_any and vlm_ocr_score <= 0.35:
                    detected_str = ", ".join(f"'{d['text']}'" for d in detected_ocr_texts)
                    decision = "REVIEW"
                    reason = (
                        f"VLM-OCR Conflict: VLM matched '{vlm_cand}' (conf={vlm_conf:.2f}), "
                        f"but detected text {detected_str} does not match (OCR score={vlm_ocr_score:.3f}). "
                        f"Likely a competitor or number-squatter."
                    )
                # Cross-check 2: Icon-Swap / Palette Divergence Spoof Detection.
                # Text matches (OCR >= 0.85), but SigLIP is weak (< 0.78) OR Color diverges (< 0.65).
                # Example 1: Same "JAGOLEDAK" text but lion icon instead of hen icon (SigLIP < 0.78).
                # Example 2: Same "NUSA 211" text but eagle icon instead of dragon icon (Color < 0.65).
                elif ocr_detected_any and vlm_ocr_score >= 0.85 and (float(siglip_scores[vlm_idx]) < 0.78 or float(color_scores[vlm_idx]) < 0.65):
                    vlm_siglip = float(siglip_scores[vlm_idx])
                    vlm_color = float(color_scores[vlm_idx])
                    decision = "REVIEW"
                    diverge_details = []
                    if vlm_siglip < 0.78:
                        diverge_details.append(f"Visual identity is weak (SigLIP={vlm_siglip:.3f} < 0.78)")
                    if vlm_color < 0.65:
                        diverge_details.append(f"Color palette diverges significantly (Color={vlm_color:.3f} < 0.65)")
                    reason = (
                        f"Icon/Visual Divergence Spoof: Text matches '{vlm_cand}' (OCR={vlm_ocr_score:.3f}), "
                        f"but {' and '.join(diverge_details)}. "
                        f"Logo icon or color palette differs from official reference — possible brand impersonation."
                    )
                # If VLM selected a known candidate with solid confidence and OCR agrees (or no text)
                elif vlm_conf >= 0.70:
                    top1_brand = vlm_cand
                    top1_score = max(top1_score, float(final_scores[vlm_idx]))
                    top1_siglip = float(siglip_scores[vlm_idx])
                    top1_color = float(color_scores[vlm_idx])
                    top1_ocr = vlm_ocr_score

                    # Require BOTH strong visual support (SigLIP >= 0.78) AND color support (Color >= 0.65) to confirm MATCH.
                    # High VLM confidence alone is not enough — text-only VLM or wide banners can be fooled by icon swaps.
                    if top1_siglip >= 0.78 and top1_color >= 0.65:
                        decision = "MATCH"
                        reason = f"Confirmed by VLM ({vlm_cand}, conf={vlm_conf:.2f}): {vlm_reason}"
                    elif top1_color < 0.65:
                        decision = "REVIEW"
                        reason = f"VLM suggests {vlm_cand} (conf={vlm_conf:.2f}), but color palette deviates significantly ({top1_color:.3f} < 0.65): possible icon swap or recolored spoof."
                    else:
                        decision = "REVIEW"
                        reason = f"VLM suggests {vlm_cand} (conf={vlm_conf:.2f}), but visual support is weak ({top1_siglip:.3f} < 0.78): {vlm_reason}"
        # --- Rule 0b: VLM Competitor / Clone Override ---
        # When VLM explicitly flags the image as a competitor or clone,
        # force REVIEW regardless of visual similarity scores.
        if not reason and vlm_result:
            is_competitor = vlm_result.get("is_competitor", False)
            vlm_cand = vlm_result.get("detected_brand") or vlm_result.get("candidate")
            vlm_reason = vlm_result.get("reasoning") or vlm_result.get("reason") or ""

            if is_competitor or (vlm_cand is None and vlm_reason):
                decision = "REVIEW"
                reason = f"VLM Competitor/Clone Detection: {vlm_reason}"

        # --- Rule 1: Visual Dominant Match (Genuine Brand with Consistent Color) ---
        if not reason:
            # Competitor Spoof / Text-Visual Conflict Check:
            # If visual shape matches Brand A, but OCR detected text that does NOT match Brand A:
            if ocr_detected_any and top1_siglip >= 0.68 and top1_ocr <= 0.55:
                detected_str = ", ".join(f"'{d['text']}'" for d in detected_ocr_texts)
                decision = "REVIEW"
                reason = f"Brand Spoof / Competitor Text Conflict: Visual shape matches '{top1_brand}' ({top1_siglip:.3f}), but detected text {detected_str} belongs to a competitor or different brand."

            # Potential Color Alteration / Grayscale Spoof:
            # Visual structure matches, but color features deviate significantly:
            elif top1_siglip >= 0.72 and top1_color < 0.60:
                decision = "REVIEW"
                reason = f"Potential Color Alteration / Grayscale Spoof: Visual structure matches '{top1_brand}' ({top1_siglip:.3f}), but color deviates ({top1_color:.3f}) — possible recoloring, hue shift, or black/white variant."

            # Genuine Visual Dominant Match (when colors match and no conflicting text):
            elif top1_siglip >= self.strong_visual_threshold and top1_color >= self.strong_color_threshold:
                decision = "MATCH"
                if not ocr_detected_any:
                    reason = f"Visual-dominant match ({top1_siglip:.3f}) with consistent color ({top1_color:.3f}); no text expected or present."
                else:
                    reason = f"High visual similarity ({top1_siglip:.3f}) and color consistency ({top1_color:.3f})."

            # --- Rule 2: Strong Multi-Modal Consensus ---
            elif (
                top1_score >= self.match_score_threshold
                and margin >= self.match_margin_threshold
                and (top1_siglip >= 0.75 or top1_ocr >= self.strong_ocr_threshold)
            ):
                if top1_color < 0.65:
                    decision = "REVIEW"
                    reason = f"Multi-modal consensus reached, but Color diverges ({top1_color:.3f} < 0.65) — possible recoloring or swapped icon."
                else:
                    decision = "MATCH"
                    reason = f"Strong multi-modal consensus (score={top1_score:.3f}, margin={margin:.3f})."

            # --- Rule 4: Moderate Score or Close Competition ---
            elif top1_score >= self.review_score_threshold:
                if margin >= self.match_margin_threshold and (top1_siglip >= 0.78 or top1_ocr >= 0.78):
                    if top1_color < 0.65:
                        decision = "REVIEW"
                        reason = f"Sufficient score ({top1_score:.3f}), but Color deviates significantly ({top1_color:.3f} < 0.65)."
                    else:
                        decision = "MATCH"
                        reason = f"Sufficient composite score ({top1_score:.3f}) and clear margin ({margin:.3f}) over '{top2_brand}'."
                else:
                    decision = "REVIEW"
                    reason = f"Close competition with '{top2_brand}' (margin={margin:.3f} < {self.match_margin_threshold}) or moderate score ({top1_score:.3f})."

            # --- Rule 5: Protected Brand Sensitivity (Minimize False Negatives) ---
            elif (
                (is_protected or any(b in self.protected_brands for b in [brand_names[i] for i in sorted_indices[:3]]))
                and top1_score >= self.protected_score_threshold
            ):
                decision = "REVIEW"
                reason = f"Protected brand '{top1_brand}' matched above sensitivity threshold ({top1_score:.3f} >= {self.protected_score_threshold})."

            # --- Rule 6: Insufficient Evidence ---
            else:
                decision = "UNKNOWN"
                reason = f"Insufficient match evidence across modalities (top score={top1_score:.3f} < {self.review_score_threshold})."

        # --- Rule 7: Brand Classifier Unknown Rejection ---
        # If classifier indicates the image belongs to an unknown/unregistered brand or has altered graphics,
        # demote MATCH to REVIEW
        cls_unknown_thresh = getattr(config, "CLASSIFIER_UNKNOWN_THRESHOLD", 0.15)
        if classifier_result and decision == "MATCH":
            is_cls_unknown = classifier_result.get("is_unknown", False)
            unknown_prob = float(classifier_result.get("unknown_prob", 0.0))
            cls_pred = classifier_result.get("predicted_brand")
            cls_conf = float(classifier_result.get("confidence", 0.0))

            # Divergence conditions:
            # 1. Image explicitly classified as unknown class
            # 2. unknown_prob >= 0.25 indicates significant out-of-distribution or altered elements (e.g. mo_2.jpg = 0.2678)
            # 3. Classifier predicts a different brand than top1
            # 4. Elevated unknown_prob (>= 0.15) paired with degraded confidence (< 0.70)
            should_reject = (
                is_cls_unknown
                or unknown_prob >= 0.25
                or (cls_pred != top1_brand and cls_conf >= 0.30)
                or (unknown_prob >= 0.15 and cls_conf < 0.70)
            )
            if should_reject:
                decision = "REVIEW"
                reason = (
                    f"Brand Classifier Flag: Image flagged as unknown/divergent "
                    f"(unknown_prob={unknown_prob:.2f}, pred={cls_pred}, conf={cls_conf:.2f}). "
                    f"Possible modified logo, added elements, or clone."
                )

        # --- Rule 8: Structural / Typography Divergence (Edge Stroke Consistency) ---
        # When comparing against the official reference logo, if edge stroke IoU drops below 0.55,
        # it indicates redrawn typography, modified curves, altered stroke thickness, or imitation fonts.
        if decision == "MATCH" and edge_stroke_iou is not None:
            if edge_stroke_iou < 0.55:
                decision = "REVIEW"
                reason = (
                    f"Structural / Typography Divergence: Stroke geometry differs from official reference "
                    f"(edge IoU = {edge_stroke_iou:.2f} < 0.55). Possible redrawn clone, modified font, or altered artwork."
                )

        # --- Rule 9: VLM Side-by-Side Competitor / Clone Rejection ---
        if decision == "MATCH" and vlm_result:
            if vlm_result.get("is_competitor", False) or vlm_result.get("candidate") is None:
                decision = "REVIEW"
                vlm_reason = vlm_result.get("reason", "VLM flagged graphic or text differences against official reference")
                reason = f"VLM Verification Flag: {vlm_reason}"

        # Prepare top candidates summary
        top_candidates = []
        for rank, idx in enumerate(sorted_indices[:5], 1):
            cand_info = {
                "rank": rank,
                "brand": brand_names[idx],
                "final_score": round(float(final_scores[idx]), 4),
                "siglip": round(float(siglip_scores[idx]), 4),
                "color": round(float(color_scores[idx]), 4),
                "ocr": round(float(ocr_scores[idx]), 4),
            }
            if classifier_scores is not None:
                cand_info["classifier"] = round(float(classifier_scores[idx]), 4)
            top_candidates.append(cand_info)

        return {
            "prediction": top1_brand,
            "decision": decision,
            "final_score": round(top1_score, 4),
            "margin": round(margin, 4),
            "reason": reason,
            "is_protected": is_protected,
            "signals": {
                "siglip": round(top1_siglip, 4),
                "color": round(top1_color, 4),
                "ocr": round(top1_ocr, 4),
                "classifier": round(top1_cls, 4) if classifier_scores is not None else 0.0,
                "vlm": round(float(vlm_result.get("brand_confidence", 0.0)), 4) if vlm_result else 0.0,
                "edge_iou": round(float(edge_stroke_iou), 4) if edge_stroke_iou is not None else None,
            },
            "ocr_text": [d["text"] for d in detected_ocr_texts] if detected_ocr_texts else [],
            "ocr_details": detected_ocr_texts,
            "top_candidates": top_candidates,
            "views_used": view_names_used,
            "classifier_used": classifier_result is not None,
            "classifier_result": classifier_result,
            "vlm_used": vlm_result is not None,
            "vlm_result": vlm_result,
            "vlm_text": vlm_texts or [],
        }
