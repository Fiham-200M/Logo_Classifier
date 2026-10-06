#!/usr/bin/env python3
"""
Multi-Modal Anti-Spoofing Forensics Engine CLI & API Entry Point.

Usage:
    python verify_logo.py logos/a200m.png
    python verify_logo.py --debug logos/a200m.png
    python verify_logo.py --batch logos/
"""

import sys
import os
from pathlib import Path

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import argparse
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
from PIL import Image

import config
from preprocessing.media_normalizer import MediaNormalizer
from forensics.perceptual_hash import PerceptualHasher
from forensics.color_analysis import ColorAnalyzer
from forensics.edge_analysis import EdgeAnalyzer
from models.siglip_model import SigLIPModel
from models.dinov2_model import DINOv2Model
from models.ocr_model import OCRModel
from models.vlm_model import VLMModel
from reference_database.reference_store import ReferenceStore
from fusion.decision_engine import ForensicDecisionEngine
from src.models.classifier_engine import ClassifierEngine


# ============================================================
# OCR Canvas Renderer — bounding polygons + HUD ledger
# ============================================================

# Distinct colors per token (BGR order for OpenCV)
_TOKEN_COLORS_BGR = [
    (0, 255, 136),   # bright green
    (255, 229, 0),   # cyan-ish
    (0, 165, 255),   # orange
    (255, 0, 128),   # magenta
    (0, 255, 255),   # yellow
    (255, 128, 0),   # light blue
    (128, 0, 255),   # pink-red
    (0, 200, 200),   # gold
]


def render_ocr_canvas(pil_image: Image.Image, raw_detections: list) -> np.ndarray:
    """
    Render a professional OCR visualization canvas:
      1. Top half: the logo image with numbered bounding polygons drawn on detected text regions.
      2. Bottom half: a dark HUD ledger panel listing each token on its OWN separate row
         with index badge, confidence score, and the detected text — zero overlap guaranteed.

    Args:
        pil_image: The normalized RGB PIL Image of the logo.
        raw_detections: List of dicts from PaddleOCR, each with keys:
            'text' (str), 'confidence' (float), optionally 'box' (list of 4 [x,y] points).

    Returns:
        numpy array (RGB) of the composite canvas.
    """
    img = np.array(pil_image.copy())
    img_h, img_w = img.shape[:2]

    # Ensure minimum canvas width for the ledger text
    min_canvas_w = max(img_w, 480)

    detections = raw_detections or []

    # --- Part 1: Draw bounding polygons + index badges on the logo ---
    overlay = img.copy()
    for idx, det in enumerate(detections):
        color_bgr = _TOKEN_COLORS_BGR[idx % len(_TOKEN_COLORS_BGR)]
        # Convert to RGB for our RGB image
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        box = det.get("box")
        if box and len(box) >= 4:
            pts = np.array(box, dtype=np.int32).reshape((-1, 1, 2))
            # Draw filled polygon with transparency
            cv2.fillPoly(overlay, [pts.reshape(-1, 2)], color_rgb)
            # Draw polygon outline
            cv2.polylines(img, [pts], isClosed=True, color=color_rgb, thickness=2)

            # Draw corner markers
            for pt in box:
                cx, cy = int(pt[0]), int(pt[1])
                cv2.circle(img, (cx, cy), 4, color_rgb, -1)
                cv2.circle(img, (cx, cy), 4, (255, 255, 255), 1)

            # Draw index badge near top-left of the polygon
            badge_x = int(box[0][0])
            badge_y = int(box[0][1]) - 8
            badge_label = f"#{idx + 1}"
            (tw, th), _ = cv2.getTextSize(badge_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bx1 = max(badge_x - 2, 0)
            by1 = max(badge_y - th - 4, 0)
            bx2 = bx1 + tw + 6
            by2 = by1 + th + 6
            cv2.rectangle(img, (bx1, by1), (bx2, by2), color_rgb, -1)
            cv2.putText(img, badge_label, (bx1 + 3, by2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            # No box — just draw index in a small region
            fy = 25 + idx * 28
            badge_label = f"#{idx + 1}"
            cv2.putText(img, badge_label, (8, fy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1, cv2.LINE_AA)

    # Blend polygon fill at 25% opacity
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)

    # --- Part 2: Build HUD ledger panel below the image ---
    row_height = 32
    header_height = 36
    padding_top = 10
    padding_bottom = 10
    num_tokens = max(len(detections), 1)  # at least 1 row for "No text detected"
    ledger_height = header_height + padding_top + (num_tokens * row_height) + padding_bottom

    # Dark background panel
    ledger = np.full((ledger_height, min_canvas_w, 3), (11, 15, 25), dtype=np.uint8)  # #0b0f19

    # Draw header bar
    cv2.rectangle(ledger, (0, 0), (min_canvas_w, header_height), (18, 24, 38), -1)
    cv2.putText(ledger, "OCR DETECTED TOKENS", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 136), 1, cv2.LINE_AA)

    if not detections:
        cv2.putText(ledger, "No text detected", (20, header_height + padding_top + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 120, 140), 1, cv2.LINE_AA)
    else:
        for idx, det in enumerate(detections):
            color_bgr = _TOKEN_COLORS_BGR[idx % len(_TOKEN_COLORS_BGR)]
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            txt = det.get("text", "")
            conf = det.get("confidence", 0.0)
            y = header_height + padding_top + (idx * row_height) + 22

            # Subtle row stripe for readability
            if idx % 2 == 0:
                ry = header_height + padding_top + (idx * row_height)
                cv2.rectangle(ledger, (0, ry), (min_canvas_w, ry + row_height), (16, 20, 32), -1)

            # Index badge
            badge = f"#{idx + 1}"
            cv2.putText(ledger, badge, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 1, cv2.LINE_AA)

            # Confidence badge
            conf_str = f"{conf:.0%}"
            conf_color = (0, 255, 136) if conf >= 0.85 else (0, 200, 255) if conf >= 0.6 else (0, 100, 255)
            cv2.putText(ledger, conf_str, (55, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, conf_color, 1, cv2.LINE_AA)

            # Detected text (truncate if too wide)
            max_text_w = min_canvas_w - 140
            display_txt = txt
            while True:
                (dtw, _), _ = cv2.getTextSize(display_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                if dtw <= max_text_w or len(display_txt) <= 3:
                    break
                display_txt = display_txt[:-4] + "..."
            cv2.putText(ledger, display_txt, (115, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 225, 240), 1, cv2.LINE_AA)

    # --- Part 3: Compose final canvas (image on top, ledger below) ---
    # Resize image to match min_canvas_w if needed
    if img_w < min_canvas_w:
        pad_left = (min_canvas_w - img_w) // 2
        pad_right = min_canvas_w - img_w - pad_left
        img_padded = cv2.copyMakeBorder(img, 0, 0, pad_left, pad_right,
                                         cv2.BORDER_CONSTANT, value=(11, 15, 25))
    elif img_w > min_canvas_w:
        # Scale ledger width to match image
        ledger_resized = np.full((ledger_height, img_w, 3), (11, 15, 25), dtype=np.uint8)
        ledger_resized[:, :min(min_canvas_w, img_w), :] = ledger[:, :min(min_canvas_w, img_w), :]
        ledger = ledger_resized
        img_padded = img
    else:
        img_padded = img

    canvas = np.vstack([img_padded, ledger])
    return canvas

class LogoForensicsEngine:
    def __init__(self):
        print("[LogoForensicsEngine] Initializing models and reference database...")
        self.ref_store = ReferenceStore()
        self.siglip = SigLIPModel()
        self.dinov2 = DINOv2Model()
        self.ocr = OCRModel()
        self.vlm = VLMModel()
        self.classifier = ClassifierEngine()
        self.decision_engine = ForensicDecisionEngine()
        print(f"[LogoForensicsEngine] Ready. Protecting {len(self.ref_store.references)} official brands.")

    def verify(
        self,
        image_input: Any,
        debug: bool = False,
        debug_dir: Optional[Path] = None,
        asset_mode: str = "auto",
        skip_vlm: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end forensic verification on candidate media.
        asset_mode can be 'auto', 'logo', or 'favicon'.
        """
        start_time = time.time()

        # ============================================================
        # STAGE 1: MEDIA NORMALIZATION
        # ============================================================
        norm = MediaNormalizer.normalize_media(image_input)
        canvas_white = norm["canvas_white"]
        canvas_black = norm["canvas_black"]
        alpha_mask = norm["alpha_mask"]
        normalized_rgb = norm["normalized_rgb"]
        original_img = norm["original"]
        meta = norm["metadata"]

        # ============================================================
        # STAGE 2: FAST SCREENING (pHash, dHash & SigLIP)
        # ============================================================
        query_hashes = PerceptualHasher.extract_hashes(normalized_rgb)
        key_frames = norm.get("key_frames", [])
        key_frame_siglip_embs = []
        if key_frames and len(key_frames) > 1:
            query_siglip_emb = self.siglip.extract_multiframe_embedding(key_frames, strategy="mean")
            key_frame_siglip_embs = [self.siglip.extract_embedding(kf) for kf in key_frames]
        else:
            query_siglip_emb = self.siglip.extract_embedding(normalized_rgb)
        query_siglip_black = self.siglip.extract_embedding(canvas_black)
        query_dinov2_emb = self.dinov2.extract_embedding(normalized_rgb)

        # Rank all 52 protected brands using fast visual screening across all variants
        candidate_scores = []
        for brand, ref_data in self.ref_store.references.items():
            variants = ref_data.get("variants", [ref_data])
            if asset_mode == "favicon":
                fav_only = [v for v in variants if v.get("asset_type") == "favicon"]
                if fav_only:
                    variants = fav_only
            elif asset_mode == "logo":
                logo_only = [v for v in variants if v.get("asset_type") != "favicon"]
                if logo_only:
                    variants = logo_only

            best_sim = -1.0
            best_variant = ref_data
            best_p_dist = 999
            best_d_dist = 999

            for v in variants:
                hash_comp = PerceptualHasher.compare_hashes(query_hashes, v["hashes"])
                p_dist = hash_comp["phash_hamming_distance"]
                d_dist = hash_comp["dhash_hamming_distance"]

                v_sig_norm = v["siglip_embeddings"]["normalized"]
                v_sig_black = v["siglip_embeddings"]["black"]
                sim_norm = SigLIPModel.cosine_similarity(query_siglip_emb, v_sig_norm)
                sim_black = SigLIPModel.cosine_similarity(query_siglip_black, v_sig_black)
                sig_sim = max(sim_norm, sim_black)

                # For multi-frame GIF candidates, check keyframe frame-level alignments
                if key_frame_siglip_embs:
                    for kf_emb in key_frame_siglip_embs:
                        kf_sim = max(
                            SigLIPModel.cosine_similarity(kf_emb, v_sig_norm),
                            SigLIPModel.cosine_similarity(kf_emb, v_sig_black)
                        )
                        if kf_sim > sig_sim:
                            sig_sim = kf_sim

                if sig_sim > best_sim + 1e-4 or (abs(sig_sim - best_sim) <= 1e-4 and p_dist < best_p_dist):
                    best_sim = sig_sim
                    best_variant = v
                    best_p_dist = p_dist
                    best_d_dist = d_dist

            candidate_scores.append({
                "brand_id": brand,
                "siglip_score": best_sim,
                "phash_dist": best_p_dist,
                "dhash_dist": best_d_dist,
                "ref_data": best_variant,
            })

        # Sort candidates by SigLIP score descending
        candidate_scores.sort(key=lambda x: x["siglip_score"], reverse=True)
        top_cand = candidate_scores[0]
        runner_up = candidate_scores[1] if len(candidate_scores) > 1 else top_cand

        top_brand = top_cand["brand_id"]
        margin = float(top_cand["siglip_score"] - runner_up["siglip_score"])

        # ============================================================
        # STAGE 3: FORENSIC ANALYSIS (DINOv2, Color, Edge, OCR)
        # ============================================================
        # Dynamically evaluate forensic consensus across all variants of top_brand
        brand_ref = self.ref_store.references.get(top_brand, top_cand["ref_data"])
        brand_variants = brand_ref.get("variants", [top_cand["ref_data"]])
        if asset_mode == "favicon":
            fav_only = [v for v in brand_variants if v.get("asset_type") == "favicon"]
            if fav_only:
                brand_variants = fav_only
        elif asset_mode == "logo":
            logo_only = [v for v in brand_variants if v.get("asset_type") != "favicon"]
            if logo_only:
                brand_variants = logo_only

        query_color_prof = ColorAnalyzer.extract_color_profile(normalized_rgb, alpha_mask=alpha_mask)

        best_variant = top_cand["ref_data"]
        best_forensic_score = -999.0
        best_color_comp = None
        best_dino_score = None
        best_p_dist = top_cand["phash_dist"]
        best_d_dist = top_cand["dhash_dist"]

        for v in brand_variants:
            v_dino_emb = v["dinov2_embeddings"]["normalized"]
            d_score = DINOv2Model.cosine_similarity(query_dinov2_emb, v_dino_emb)
            c_comp = ColorAnalyzer.compare_color_profiles(query_color_prof, v["color_profile"])
            h_comp = PerceptualHasher.compare_hashes(query_hashes, v["hashes"])
            p_dist = h_comp["phash_hamming_distance"]
            d_dist = h_comp["dhash_hamming_distance"]

            # Multi-modal forensic score to pick the exact matching canvas/variant
            forensic_score = (
                d_score * 2.0
                + c_comp["histogram_similarity"] * 1.5
                - (c_comp["cielab_delta_e"] / 10.0)
                - (min(p_dist, 32) / 16.0)
            )
            if forensic_score > best_forensic_score:
                best_forensic_score = forensic_score
                best_variant = v
                best_color_comp = c_comp
                best_dino_score = d_score
                best_p_dist = p_dist
                best_d_dist = d_dist

        top_ref = best_variant
        dinov2_score = best_dino_score
        color_comp = best_color_comp
        top_cand["phash_dist"] = best_p_dist
        top_cand["dhash_dist"] = best_d_dist

        # Sub-pixel Edge Stroke IoU & Alignment
        ref_norm_img = top_ref["image_normalized"]
        edge_comp = EdgeAnalyzer.align_and_compare(normalized_rgb, ref_norm_img)

        # 4. OCR Text Extraction
        key_frames = norm.get("key_frames", [])
        ocr_res = self.ocr.extract_text_with_frames(normalized_rgb, key_frames=key_frames)
        cand_norm_text = ocr_res["normalized_text"]

        ref_ocr_info = top_ref.get("ocr", {})
        ref_norm_text = ref_ocr_info.get("normalized_text", "")

        # OCR Brand & Reference Match
        ocr_match = False
        import re
        from ocr_engine import levenshtein_similarity
        cand_tokens_clean = [re.sub(r'[^A-Z0-9]+', '', t.upper()) for t in ocr_res.get("texts", [])]
        top_brand_clean = re.sub(r'[^A-Z0-9]+', '', top_brand.upper())
        if top_brand_clean in cand_tokens_clean or any(top_brand_clean in t for t in cand_tokens_clean):
            ocr_match = True
        elif any(levenshtein_similarity(t, top_brand_clean) >= 0.75 for t in cand_tokens_clean if len(t) >= 3):
            ocr_match = True
        elif cand_norm_text and ref_norm_text:
            ocr_sim = levenshtein_similarity(cand_norm_text, ref_norm_text)
            ocr_match = ocr_sim >= 0.75
        elif cand_norm_text:
            ocr_sim = levenshtein_similarity(cand_norm_text, top_brand)
            ocr_match = ocr_sim >= 0.75

        # ============================================================
        # STAGE 4: CONDITIONAL VLM FORENSIC ANALYSIS
        # ============================================================
        vlm_res = None
        # Trigger VLM if top candidate is suspicious or has high visual similarity
        should_run_vlm = (
            top_cand["siglip_score"] >= 0.75
            and (
                edge_comp["edge_stroke_iou"] < 0.65
                or color_comp["cielab_delta_e"] > 6.0
                or (cand_norm_text and not ocr_match)
                or (top_cand["siglip_score"] - dinov2_score > 0.07)
            )
        )

        if should_run_vlm and not skip_vlm:
            try:
                vlm_res = self.vlm.forensic_compare(
                    candidate_image=normalized_rgb,
                    reference_image=ref_norm_img,
                    brand_name=top_brand,
                    ocr_hint=cand_norm_text,
                )
            except Exception as e:
                print(f"[LogoForensicsEngine] VLM note: {e}")

        # ============================================================
        # STAGE 5: EVIDENCE FUSION & DECISION
        # ============================================================
        max_dim = max(meta.get("width", 0), meta.get("height", 0))
        if asset_mode == "favicon":
            is_candidate_favicon = True
        elif asset_mode == "logo":
            is_candidate_favicon = False
        else:
            is_candidate_favicon = max_dim <= 128
        candidate_media_type = "favicon" if is_candidate_favicon else "logo"
        top_asset_type = top_ref.get("asset_type", "logo")

        # Stage 4b: Brand Classifier (Trained on Competitor & Positive Datasets)
        classifier_pred = None
        if hasattr(self, "classifier") and self.classifier and self.classifier.loaded:
            classifier_pred = self.classifier.predict(query_siglip_emb)

        forensic_metrics = {
            "phash_hamming_distance": top_cand["phash_dist"],
            "dhash_hamming_distance": top_cand["dhash_dist"],
            "siglip2_semantic_score": round(float(top_cand["siglip_score"]), 4),
            "dinov2_geometry_score": round(float(dinov2_score), 4),
            "cielab_delta_e": round(float(color_comp["cielab_delta_e"]), 2),
            "color_histogram_similarity": round(float(color_comp["histogram_similarity"]), 4),
            "edge_stroke_iou": round(float(edge_comp["edge_stroke_iou"]), 4),
            "injected_edge_ratio": round(float(edge_comp.get("injected_edge_ratio", 0.0)), 4),
            "ssim_score": round(float(edge_comp["ssim_score"]), 4),
            "ocr_text_match": ocr_match,
            "ref_ocr_text": ref_norm_text,
            "asset_type": top_asset_type,
            "is_favicon": is_candidate_favicon,
            "candidate_media_type": candidate_media_type,
            "classifier_is_unknown": classifier_pred.get("is_unknown", False) if classifier_pred else False,
            "classifier_unknown_prob": round(float(classifier_pred.get("unknown_prob", 0.0)), 4) if classifier_pred else 0.0,
            "classifier_predicted_brand": classifier_pred.get("predicted_brand") if classifier_pred else None,
            "classifier_confidence": round(float(classifier_pred.get("confidence", 0.0)), 4) if classifier_pred else 0.0,
        }

        decision_payload = self.decision_engine.evaluate(
            brand_id=top_brand,
            forensic_metrics=forensic_metrics,
            ocr_evidence=ocr_res,
            vlm_evidence=vlm_res,
            candidate_margin=margin,
        )

        elapsed = round(time.time() - start_time, 3)

        # Candidate shortlist summary
        shortlist = [
            {
                "brand_id": c["brand_id"],
                "siglip": round(float(c["siglip_score"]), 4),
                "phash_dist": c["phash_dist"],
            }
            for c in candidate_scores[:5]
        ]

        report = {
            "brand_id": top_brand if decision_payload["verdict"] != "UNKNOWN" else "UNKNOWN",
            "verdict": decision_payload["verdict"],
            "confidence_score": decision_payload["confidence_score"],
            "asset_type": top_asset_type if decision_payload["verdict"] != "UNKNOWN" else candidate_media_type,
            "media_type": candidate_media_type,
            "matched_reference": top_ref.get("filename") if decision_payload["verdict"] != "UNKNOWN" else None,
            "preprocessing_meta": {
                "file_format": meta["file_format"],
                "is_animated": meta["is_animated"],
                "frames_processed": meta["frames_processed"],
                "alpha_channel_detected": meta["alpha_channel_detected"],
                "width": meta["width"],
                "height": meta["height"],
            },
            "forensic_metrics": {
                "phash_hamming_distance": forensic_metrics["phash_hamming_distance"],
                "dhash_hamming_distance": forensic_metrics["dhash_hamming_distance"],
                "siglip2_semantic_score": forensic_metrics["siglip2_semantic_score"],
                "dinov2_geometry_score": forensic_metrics["dinov2_geometry_score"],
                "cielab_delta_e": forensic_metrics["cielab_delta_e"],
                "edge_stroke_iou": forensic_metrics["edge_stroke_iou"],
                "ocr_text_match": forensic_metrics["ocr_text_match"],
                "classifier_predicted_brand": forensic_metrics.get("classifier_predicted_brand"),
                "classifier_confidence": forensic_metrics.get("classifier_confidence", 0.0),
                "classifier_is_unknown": forensic_metrics.get("classifier_is_unknown", False),
                "classifier_unknown_prob": forensic_metrics.get("classifier_unknown_prob", 0.0),
            },
            "ocr": {
                "detected_text": ocr_res["texts"],
                "normalized_text": ocr_res["normalized_text"],
                "confidence": ocr_res["scores"][0] if ocr_res["scores"] else 0.0,
            },
            "micro_differences_detected": decision_payload["micro_differences_detected"],
            "threat_type": decision_payload["threat_type"],
            "action_reason": decision_payload["action_reason"],
            "candidate_brands": shortlist,
            "processing_time_sec": elapsed,
        }

        # ============================================================
        # DEBUG ARTIFACT GENERATION
        # ============================================================
        if debug:
            out_debug = debug_dir or (_project_root / Path("debug"))
            out_debug.mkdir(parents=True, exist_ok=True)

            original_img.save(out_debug / "input.png")
            canvas_white.save(out_debug / "white_render.png")
            canvas_black.save(out_debug / "black_render.png")
            alpha_mask.save(out_debug / "alpha_mask.png")
            normalized_rgb.save(out_debug / "normalized_rgb.png")

            cv2.imwrite(str(out_debug / "edge_candidate.png"), edge_comp["edge_query"])
            cv2.imwrite(str(out_debug / "edge_reference.png"), edge_comp["edge_ref"])
            cv2.imwrite(str(out_debug / "alignment.png"), edge_comp["aligned_query"])
            cv2.imwrite(str(out_debug / "diff_heatmap.png"), edge_comp["diff_heatmap"])

            # OCR visualization canvas — bounding polygons + HUD ledger
            ocr_vis = render_ocr_canvas(normalized_rgb, ocr_res.get("raw_detections", []))
            cv2.imwrite(str(out_debug / "ocr_visualization.png"), cv2.cvtColor(ocr_vis, cv2.COLOR_RGB2BGR))

            with open(out_debug / "comparison_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)

            print(f"[Debug] Saved all forensic debug artifacts to {out_debug}/")

        return report


def print_forensic_banner(report: Dict[str, Any], filepath: str):
    """Prints clean terminal summary banner."""
    print("=" * 60)
    print("BRAND FORENSICS RESULT")
    print("=" * 60)
    print(f"Input:        {filepath}")
    meta = report.get("preprocessing_meta", {})
    dim_str = f" ({meta.get('width', '?')}x{meta.get('height', '?')}, {meta.get('file_format', '?')})" if meta else ""
    media_type = report.get("media_type", report.get("asset_type", "logo")).upper()
    print(f"Media Type:   {media_type}{dim_str}")
    asset_str = f" ({report.get('asset_type', 'logo').upper()}: {report.get('matched_reference')})" if report.get("matched_reference") else ""
    print(f"Top Brand:    {report['brand_id']}{asset_str}")
    print("-" * 60)
    fm = report["forensic_metrics"]
    model_name_display = getattr(config, "SIGLIP_MODEL_NAME", "SigLIP 2")
    short_model = "SigLIP 2 SO400M" if "so400m" in model_name_display.lower() else "SigLIP 2 Base"
    print(f"{short_model}:")
    print(f"  Top Brand:  {report['brand_id']}")
    print(f"  Similarity: {fm['siglip2_semantic_score']:.4f}")
    print("DINOv2:")
    print(f"  Similarity: {fm['dinov2_geometry_score']:.4f}")
    print("Color:")
    print(f"  Delta E:    {fm['cielab_delta_e']:.2f}")
    print("Edge IoU:")
    print(f"  IoU:        {fm['edge_stroke_iou']:.4f}")
    print("OCR:")
    print(f"  Match:      {'Match' if fm['ocr_text_match'] else 'No Match'}")
    print("pHash:")
    print(f"  Distance:   {fm['phash_hamming_distance']}")
    if "classifier_predicted_brand" in fm and fm["classifier_predicted_brand"] is not None:
        cls_str = f"{fm['classifier_predicted_brand']} (conf={fm.get('classifier_confidence', 0.0):.3f})"
        if fm.get("classifier_is_unknown"):
            cls_str = f"UNKNOWN (unk_prob={fm.get('classifier_unknown_prob', 0.0):.3f})"
        print("Classifier:")
        print(f"  Brand/Conf: {cls_str}")
    print("-" * 60)
    print("Final:")
    print(f"  Verdict:    {report['verdict']}")
    print(f"  Threat:     {report['threat_type']}")
    print(f"  Confidence: {report['confidence_score']:.4f}")
    print(f"  Reason:     {report['action_reason']}")
    if report.get("micro_differences_detected"):
        print(f"  Diffs:      {'; '.join(report['micro_differences_detected'])}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Anti-Spoofing Forensics Engine")
    parser.add_argument("target", type=str, nargs="?", default=None, help="Path to image file or directory")
    parser.add_argument("--favicons", action="store_true", help="Process all official reference favicons in Favicon/")
    parser.add_argument("--debug", action="store_true", help="Save forensic debug visual artifacts to debug/")
    parser.add_argument("--batch", action="store_true", help="Process directory in batch mode and output results/")
    parser.add_argument("--json", action="store_true", help="Output raw JSON to stdout")

    args = parser.parse_args()

    if not args.target and not args.favicons:
        parser.print_help()
        sys.exit(1)

    if args.favicons:
        target_path = getattr(config, "FAVICON_DIR", config.BASE_DIR / "Favicon")
        args.batch = True
    else:
        target_path = Path(args.target)

    if not target_path.exists():
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    engine = LogoForensicsEngine()

    # Batch Processing Mode
    if args.batch or target_path.is_dir():
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
        image_files = sorted([
            p for p in target_path.rglob("*")
            if p.is_file() and p.suffix.lower() in valid_exts
        ])

        if not image_files:
            print(f"No valid images found in {target_path}")
            sys.exit(1)

        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        print(f"[Batch] Processing {len(image_files)} images...")
        verdict_counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}
        threat_counts = {}
        all_results = []
        t0 = time.time()

        for idx, img_p in enumerate(image_files, 1):
            res = engine.verify(img_p, debug=args.debug)
            verdict = res["verdict"]
            threat = res["threat_type"]

            verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1
            threat_counts[threat] = threat_counts.get(threat, 0) + 1

            all_results.append(res)
            # Save individual file JSON
            with open(results_dir / f"{img_p.stem}.json", "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2)

            print(f"[{idx:03d}/{len(image_files)}] {img_p.name:<25} -> {res['brand_id']:<15} [{verdict}] ({threat})")

        total_time = round(time.time() - t0, 3)
        summary = {
            "total_images": len(image_files),
            "verdict_counts": verdict_counts,
            "threat_counts": threat_counts,
            "total_time_seconds": total_time,
            "avg_time_per_image": round(total_time / len(image_files), 3),
        }
        with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING COMPLETED")
        print("=" * 60)
        print(f"Total Images: {summary['total_images']}")
        print(f"Verdicts:     MATCH: {verdict_counts['MATCH']}, REVIEW: {verdict_counts['REVIEW']}, UNKNOWN: {verdict_counts['UNKNOWN']}")
        print(f"Time:         {total_time}s ({summary['avg_time_per_image']}s/image)")
        print(f"Saved:        {results_dir}/summary.json")
        print("=" * 60)
        return

    # Single Image Mode
    res = engine.verify(target_path, debug=args.debug)

    if args.json:
        print(json.dumps(res, indent=2))
    else:
        print_forensic_banner(res, str(target_path))


if __name__ == "__main__":
    main()
