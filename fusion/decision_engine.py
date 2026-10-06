"""
Forensic Evidence Fusion and Decision Engine Module.
Synthesizes multi-modal evidence across:
  - Semantic Vision (SigLIP 2)
  - Geometric Structure (DINOv2)
  - Color Profiles & Delta E (HSV / CIE LAB)
  - Sub-pixel Edge Stroke Consistency (IoU)
  - Optical Character Recognition (OCR & Number Collision)
  - Microscopic Forensic VLM Inspection

Core Philosophy:
  HIGH SIMILARITY + MICRO DIFFERENCE = REVIEW (Not MATCH)

Threat Taxonomy:
  EXACT_REPLICA, COLOR_SPOOF, VECTOR_REDRAW, ELEMENT_INJECTION,
  NUMBER_COLLISION, TEXT_MODIFICATION, LAYOUT_MODIFICATION,
  UNRELATED, UNKNOWN
"""

from typing import Dict, Any, List, Optional, Tuple
import yaml
from pathlib import Path


class ForensicDecisionEngine:
    def __init__(self, thresholds_path: Optional[Path] = None):
        if thresholds_path is None:
            thresholds_path = Path(__file__).resolve().parent.parent / "config" / "thresholds.yaml"

        self.thresholds = self._load_thresholds(thresholds_path)

    @staticmethod
    def _load_thresholds(path: Path) -> Dict[str, Any]:
        """Loads YAML thresholds with safe defaults."""
        defaults = {
            "siglip": {"strong_match": 0.88, "suspicious": 0.80, "unknown": 0.50},
            "dinov2": {"strong_match": 0.88, "suspicious": 0.80, "geometry_drift_threshold": 0.08},
            "color": {"delta_e_match": 3.5, "delta_e_review": 7.0, "histogram_match": 0.70, "histogram_review": 0.55},
            "edge": {"strong_match": 0.72, "suspicious": 0.55, "critical_mismatch": 0.40},
            "ocr": {"fuzzy_match_threshold": 0.82},
            "screening": {"phash_match_threshold": 4, "dhash_match_threshold": 6},
        }
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        defaults.update(loaded)
            except Exception:
                pass
        return defaults

    def evaluate(
        self,
        brand_id: str,
        forensic_metrics: Dict[str, Any],
        ocr_evidence: Dict[str, Any],
        vlm_evidence: Optional[Dict[str, Any]] = None,
        candidate_margin: float = 0.20,
    ) -> Dict[str, Any]:
        """
        Fuses multi-modal forensic evidence and returns verdict, threat type,
        confidence score, micro-differences list, and human-readable reason.
        """
        sig_t = self.thresholds["siglip"]
        dino_t = self.thresholds["dinov2"]
        col_t = self.thresholds["color"]
        edge_t = self.thresholds["edge"]
        ocr_t = self.thresholds["ocr"]

        siglip_score = float(forensic_metrics.get("siglip2_semantic_score", 0.0))
        dinov2_score = float(forensic_metrics.get("dinov2_geometry_score", 0.0))
        delta_e = float(forensic_metrics.get("cielab_delta_e", 0.0))
        color_sim = float(forensic_metrics.get("color_histogram_similarity", 0.5))
        edge_iou = float(forensic_metrics.get("edge_stroke_iou", 0.5))
        ocr_match = bool(forensic_metrics.get("ocr_text_match", False))
        phash_dist = int(forensic_metrics.get("phash_hamming_distance", 64))

        micro_diffs: List[str] = []
        threat_type = "UNKNOWN"
        verdict = "UNKNOWN"
        action_reason = ""

        # =============================================================
        # 1. SCREENING OUT UNRELATED / UNKNOWN MEDIA
        # =============================================================
        is_fav = bool(forensic_metrics.get("is_favicon", False))

        # Favicons: if no OCR and structural geometry is not solidly matching the official favicon, it is a competitor favicon
        if is_fav and not ocr_match:
            if (
                dinov2_score < 0.65
                or edge_iou < 0.35
                or (phash_dist > 25 and dinov2_score < 0.75)
                or (siglip_score < 0.82 and dinov2_score < 0.70)
            ):
                return {
                    "verdict": "UNKNOWN",
                    "confidence_score": round(max(siglip_score, dinov2_score), 4),
                    "threat_type": "UNRELATED",
                    "micro_differences_detected": ["No structural or visual consensus with protected official favicons"],
                    "action_reason": "Competitor favicon / unrelated emblem: No match with our protected brands or official favicons.",
                }

        classifier_is_unknown = bool(forensic_metrics.get("classifier_is_unknown", False))
        classifier_unknown_prob = float(forensic_metrics.get("classifier_unknown_prob", 0.0))

        # Classifier Unknown Defense: Trained on competitor dataset
        if classifier_is_unknown and classifier_unknown_prob >= 0.40 and not ocr_match and siglip_score < 0.82:
            return {
                "verdict": "UNKNOWN",
                "confidence_score": round(classifier_unknown_prob, 4),
                "threat_type": "UNRELATED",
                "micro_differences_detected": [f"Brand classifier identified as unknown/competitor (prob={classifier_unknown_prob:.2f})"],
                "action_reason": (
                    "Competitor favicon / unrelated emblem: Rejected by brand classifier."
                    if is_fav
                    else "Competitor logo / unrelated image: Rejected by brand classifier as unknown/competitor."
                ),
            }

        # General Logos: An image with low semantic and geometric similarity is unrelated
        if (
            (siglip_score < 0.72 and dinov2_score < 0.72)
            or (siglip_score < 0.76 and phash_dist > 20 and not ocr_match)
            or (dinov2_score < 0.58 and edge_iou < 0.30 and phash_dist > 25 and not ocr_match)
        ):
            return {
                "verdict": "UNKNOWN",
                "confidence_score": round(max(siglip_score, dinov2_score), 4),
                "threat_type": "UNRELATED",
                "micro_differences_detected": ["No semantic or geometric relationship with protected brand"],
                "action_reason": (
                    "Competitor favicon / unrelated emblem: No match with our protected brands or official favicons."
                    if is_fav
                    else "Competitor logo / unrelated image: No match with our protected brands or official logos."
                ),
            }

        # =============================================================
        # 2. FORENSIC ANALYSIS OF MICRO-DIFFERENCES
        # =============================================================
        import re
        from ocr_engine import normalize_text, normalize_ocr_confusions

        cand_text = ocr_evidence.get("normalized_text", "").upper()
        ref_text = str(forensic_metrics.get("ref_ocr_text", "")).upper()
        c_norm = normalize_text(cand_text)
        r_norm = normalize_text(ref_text)
        b_norm = normalize_text(brand_id)

        # A. Alphanumeric / Brand Number Collision Check
        has_number_collision = False
        cand_tokens = ocr_evidence.get("texts", [])
        if not cand_tokens and cand_text:
            cand_tokens = cand_text.split()

        # Token-based collision check (e.g. A500M vs A200M, SURGA500 vs SURGA5000, TIKET100 vs TIKET200)
        b_clean = re.sub(r'[^A-Z0-9]+', '', brand_id.upper())
        b_digits = re.findall(r'\d+', b_clean)

        # If the candidate OCR text matches the official reference OCR text, there is NO collision:
        # both candidate and official reference produced the identical OCR reading.
        ocr_texts_identical = bool(c_norm and r_norm and (c_norm == r_norm))
        # If candidate visual match is essentially perfect (near-zero hash, near-1.0 DINO/SigLIP, high edge IoU)
        is_visual_identical = (phash_dist <= 2 and dinov2_score >= 0.95 and siglip_score >= 0.98 and edge_iou >= 0.80)

        if not ocr_texts_identical:
            for tok in cand_tokens:
                t_clean = re.sub(r'[^A-Z0-9]+', '', tok.upper())
                if not t_clean or t_clean == b_clean:
                    continue

                # If this token is already present in the official reference OCR text, it is an expected official glyph
                if r_norm and (t_clean.lower() in r_norm or normalize_text(t_clean) in r_norm):
                    continue

                # If character substitution confusion matches brand or reference:
                if normalize_ocr_confusions(t_clean) == normalize_ocr_confusions(b_clean):
                    continue
                if r_norm and normalize_ocr_confusions(t_clean) in normalize_ocr_confusions(r_norm):
                    continue

                # Visual identical bypass (e.g. OCR single-glyph misread on stylized font when candidate is an exact visual replica)
                if is_visual_identical:
                    continue

                t_digits = re.findall(r'\d+', t_clean)
                b_letters = re.sub(r'\d+', '', b_clean)
                t_letters = re.sub(r'\d+', '', t_clean)

                # Case 1: Identical letter prefix/suffix but different digits (e.g. A500M vs A200M)
                if b_digits and t_digits and b_letters == t_letters and b_digits != t_digits:
                    has_number_collision = True
                    micro_diffs.append(f"Alphanumeric token collision: Detected '{t_clean}' vs official '{b_clean}'")
                    break

                # Case 2: Fuzzy similarity >= 0.70 with conflicting digits
                from ocr_engine import levenshtein_similarity
                if b_digits and t_digits and b_digits != t_digits and levenshtein_similarity(t_clean, b_clean) >= 0.70:
                    has_number_collision = True
                    micro_diffs.append(f"Fuzzy alphanumeric collision: Detected '{t_clean}' vs official '{b_clean}'")
                    break

        # B. Color Drift / Color Spoof Check (Requires structural/geometric correspondence)
        is_color_spoof = False
        has_structural_match = (dinov2_score >= 0.70) or (edge_iou >= 0.50) or ocr_match
        if siglip_score >= 0.78 and has_structural_match:
            if delta_e > col_t["delta_e_review"] or color_sim < col_t["histogram_review"]:
                is_color_spoof = True
                micro_diffs.append(f"Color palette shift: CIE LAB Delta E = {delta_e:.1f} (Histogram overlap = {color_sim:.2f})")

        # C. Vector / Font Redraw Check (Requires high semantic resemblance and structural anchor)
        geometry_drift = siglip_score - dinov2_score
        is_vector_redraw = False
        if siglip_score >= 0.78 and (dinov2_score >= 0.65 or ocr_match):
            if edge_iou < edge_t["suspicious"]:
                is_vector_redraw = True
                micro_diffs.append(
                    f"Typography / stroke geometry divergence: Edge stroke overlap ({edge_iou:.2f}) < {edge_t['suspicious']:.2f}"
                )
            elif geometry_drift > dino_t["geometry_drift_threshold"] and edge_iou < edge_t["strong_match"]:
                is_vector_redraw = True
                micro_diffs.append(
                    f"Spatial geometry drift: DINOv2 structural score ({dinov2_score:.2f}) lags SigLIP ({siglip_score:.2f})"
                )

        # D. Injected Elements / Secondary Badges / VLM Forensic Check
        is_element_injection = False
        if vlm_evidence:
            if vlm_evidence.get("icon_difference"):
                is_element_injection = True
                micro_diffs.append(f"VLM forensic alert: Graphic icon/mascot modification ({vlm_evidence.get('explanation')})")
            elif vlm_evidence.get("micro_modification_detected"):
                micro_diffs.append(f"VLM forensic alert: {vlm_evidence.get('explanation')}")

        injected_edge_ratio = float(forensic_metrics.get("injected_edge_ratio", 0.0))
        if not is_element_injection and siglip_score >= 0.85 and injected_edge_ratio > 0.12:
            is_element_injection = True
            micro_diffs.append(f"Structural edge injection: {injected_edge_ratio:.1%} new graphic strokes detected")

        # =============================================================
        # 3. DECISION SYNTHESIS
        # =============================================================
        composite_score = (
            0.35 * siglip_score
            + 0.25 * dinov2_score
            + 0.15 * color_sim
            + 0.15 * edge_iou
            + 0.10 * (1.0 if ocr_match else (0.50 if not cand_text else 0.20))
        )

        asset_type = forensic_metrics.get("asset_type", "logo")
        is_fav = forensic_metrics.get("is_favicon", False)

        # Prioritize specific threat classifications
        if has_number_collision:
            verdict = "REVIEW"
            threat_type = "NUMBER_COLLISION"
            action_reason = (
                f"Candidate visual identity matches '{brand_id}', but alphanumeric text differs "
                f"('{cand_text}' vs '{ref_text}'). Suspicious brand squatter/clone."
            )

        elif is_color_spoof:
            verdict = "REVIEW"
            threat_type = "COLOR_SPOOF"
            action_reason = (
                f"Candidate logo structure matches '{brand_id}', but exhibits significant color drift "
                f"(Delta E={delta_e:.1f}). Possible recolored spoof."
            )

        elif is_element_injection:
            verdict = "REVIEW"
            threat_type = "ELEMENT_INJECTION"
            action_reason = (
                f"Candidate matches '{brand_id}' overall, but auxiliary graphic/icon modifications "
                f"were detected. Possible altered spoof."
            )

        elif is_vector_redraw:
            verdict = "REVIEW"
            threat_type = "VECTOR_REDRAW"
            action_reason = (
                f"Candidate is semantically similar to '{brand_id}' (SigLIP={siglip_score:.3f}), "
                f"but typography stroke geometry diverges (edge IoU={edge_iou:.2f}). Possible redrawn font or clone."
            )

        elif len(micro_diffs) > 0 and siglip_score >= sig_t["suspicious"]:
            verdict = "REVIEW"
            threat_type = "TEXT_MODIFICATION" if (cand_text and not ocr_match) else "LAYOUT_MODIFICATION"
            action_reason = f"Candidate is related to '{brand_id}' but micro-modifications were detected: {'; '.join(micro_diffs)}."

        elif (
            siglip_score >= sig_t["strong_match"]
            and dinov2_score >= 0.80
            and edge_iou >= 0.68
            and (
                delta_e <= col_t["delta_e_match"]
                or (delta_e <= col_t["delta_e_review"] and dinov2_score >= 0.90 and edge_iou >= 0.78)
                or (phash_dist <= 2 and dinov2_score >= 0.95 and delta_e <= col_t["delta_e_review"])
            )
            and (ocr_match or not cand_text)
            and len(micro_diffs) == 0
        ):
            verdict = "MATCH"
            threat_type = "EXACT_REPLICA"
            action_reason = (
                f"High-confidence multi-modal consensus confirming authentic official {asset_type} '{brand_id}' "
                f"(SigLIP={siglip_score:.3f}, DINOv2={dinov2_score:.3f}, Edge IoU={edge_iou:.3f}, Delta E={delta_e:.1f})."
            )

        elif siglip_score >= sig_t["suspicious"]:
            if edge_iou >= 0.75 and dinov2_score >= 0.75:
                threat_type = "AUTHENTIC_VARIANT" if len(micro_diffs) == 0 else "LAYOUT_MODIFICATION"
                if len(micro_diffs) == 0 and dinov2_score >= 0.90 and siglip_score >= sig_t["strong_match"] and delta_e <= col_t["delta_e_review"]:
                    verdict = "MATCH"
                    action_reason = (
                        f"Verified authentic visual variant of official brand '{brand_id}' "
                        f"(SigLIP={siglip_score:.3f}, DINOv2={dinov2_score:.3f}, Edge IoU={edge_iou:.3f}, Delta E={delta_e:.1f})."
                    )
                else:
                    verdict = "REVIEW"
                    action_reason = (
                        f"Moderate similarity with '{brand_id}' (score={composite_score:.3f}), "
                        f"requires manual forensic review to confirm authenticity."
                    )
            else:
                verdict = "REVIEW"
                threat_type = "LAYOUT_MODIFICATION"
                action_reason = (
                    f"Moderate similarity with '{brand_id}' (score={composite_score:.3f}), "
                    f"requires manual forensic review to confirm authenticity."
                )

        else:
            verdict = "UNKNOWN"
            threat_type = "UNRELATED"
            if is_fav:
                action_reason = "Competitor favicon / unrelated emblem: No match with our protected brands or official favicons."
            else:
                action_reason = "Competitor logo / unrelated image: No match with our protected brands or official logos."

        return {
            "verdict": verdict,
            "confidence_score": round(composite_score, 4),
            "threat_type": threat_type,
            "micro_differences_detected": micro_diffs if micro_diffs else ["None detected; consistent with official reference"],
            "action_reason": action_reason,
        }
