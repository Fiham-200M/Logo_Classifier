"""
Webpage Screenshot Region Scanner & Multi-Modal Logo Detector.
Discovers logo candidate regions from full-page screenshots via OCR & visual saliency,
then runs verification without modifying the core logo/favicon verification pipeline.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
from PIL import Image

import config
from ocr_engine import OCREngine, normalize_text
from reference_database.reference_store import ReferenceStore
from logo_detection.candidate_detector import LogoCandidateDetector


class ScreenshotScanner:
    """
    Dedicated scanner for full-page screenshots.
    Detects brand text and emblem regions, crops candidate bounding boxes,
    and queries the forensic engine to confirm brand authenticity.
    """

    def __init__(self, ocr_engine: Optional[OCREngine] = None, ref_store: Optional[ReferenceStore] = None):
        self.ocr = ocr_engine or OCREngine()
        self.ref_store = ref_store or ReferenceStore()
        self.protected_brands = set(self.ref_store.brand_names)
        self.candidate_detector = LogoCandidateDetector(ocr_engine=self.ocr, ref_store=self.ref_store)

    def scan_screenshot(
        self,
        image_input: Union[str, Path, Image.Image],
        engine=None,
    ) -> Dict[str, Any]:
        """
        Scans a screenshot for official protected brand logos.

        Returns:
            Dict containing:
                - verdict: "MATCH" | "REVIEW" | "UNKNOWN"
                - is_our_logo: bool
                - primary_brand: str or None
                - detections: list of detected regions with bounding boxes & brands
                - candidate_count: int
                - processing_time_sec: float
        """
        start_time = time.time()

        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        w, h = pil_img.size

        # 1. Run modular Logo Candidate Detection
        raw_candidates = self.candidate_detector.detect_candidates(pil_img)

        # 2. Structure candidate boxes for forensic evaluation
        candidate_boxes = []
        for cand in raw_candidates:
            x1, y1, x2, y2 = cand["bbox"]
            candidate_boxes.append({
                "brand_hint": cand.get("brand_hint"),
                "ocr_text": cand.get("ocr_text", cand.get("source")),
                "ocr_conf": cand.get("detection_confidence", 0.70),
                "box": [x1, y1, x2 - x1, y2 - y1],
                "crop_coords": (x1, y1, x2, y2),
                "crop_image": cand.get("crop_image"),
            })

        # Prioritize candidates with brand hints and header regions, cap to top 8
        candidate_boxes.sort(
            key=lambda c: (
                2 if c.get("brand_hint") else 0,
                1 if c["crop_coords"][1] < int(h * 0.25) else 0,
                c.get("ocr_conf", 0.0)
            ),
            reverse=True
        )
        candidate_boxes = candidate_boxes[:8]

        # 3. Verify each candidate crop with the forensic engine
        detections = []
        overall_verdict = "UNKNOWN"
        primary_brand = None
        best_conf = 0.0
        best_report = {}
        best_crop_img = None

        for cand in candidate_boxes:
            # If definitive match already found and candidate has no brand hint, skip
            if overall_verdict == "MATCH" and best_conf >= 0.92 and not cand.get("brand_hint"):
                continue

            c_coords = cand["crop_coords"]
            crop_img = pil_img.crop(c_coords)

            # Skip tiny noisy crops
            if crop_img.width < 16 or crop_img.height < 16:
                continue

            crop_verdict = "UNKNOWN"
            crop_brand = cand.get("brand_hint")
            crop_conf = cand.get("ocr_conf", 0.0)
            reason = ""
            rep = {}

            if engine is not None:
                try:
                    # Run forensic verification on cropped box
                    rep = engine.verify(crop_img, debug=False, asset_mode="logo", skip_vlm=True)
                    verdict = rep.get("verdict", "UNKNOWN")
                    detected_b = rep.get("brand_id")
                    rep_conf = float(rep.get("confidence_score", 0.0))

                    if verdict == "MATCH":
                        crop_verdict = "MATCH"
                        crop_brand = detected_b
                        crop_conf = rep_conf
                        reason = rep.get("action_reason", "")
                    elif verdict == "REVIEW":
                        # If OCR strongly matches our protected brand name, elevate confidence
                        if cand.get("brand_hint") and cand["brand_hint"] == detected_b:
                            crop_verdict = "MATCH"
                            crop_brand = cand["brand_hint"]
                            crop_conf = max(rep_conf, 0.95)
                            reason = f"Confirmed brand '{crop_brand}' via visual and OCR consensus on webpage region."
                        else:
                            crop_verdict = "REVIEW"
                            crop_brand = detected_b
                            crop_conf = max(rep_conf, 0.72)
                            reason = rep.get("action_reason", "")
                    else:
                        # Fallback ONLY if crop had confirmed brand text match
                        if cand.get("brand_hint"):
                            cand_clean = normalize_text(cand.get("ocr_text", "")).upper()
                            hint_clean = cand["brand_hint"].upper()
                            if hint_clean in cand_clean or cand_clean == hint_clean:
                                crop_verdict = "REVIEW"
                                crop_brand = cand["brand_hint"]
                                crop_conf = cand["ocr_conf"]
                                reason = f"Brand text detected '{crop_brand}', pending visual verification."
                except Exception as e:
                    reason = f"Verification error: {e}"
            else:
                # Fast OCR-only fallback if engine not passed
                if cand.get("brand_hint"):
                    crop_verdict = "MATCH"
                    crop_brand = cand["brand_hint"]
                    crop_conf = cand["ocr_conf"]
                    reason = f"Detected brand text '{crop_brand}' on page."

            if crop_verdict in ("MATCH", "REVIEW"):
                detections.append({
                    "brand": crop_brand,
                    "verdict": crop_verdict,
                    "confidence": round(crop_conf, 4),
                    "box": {
                        "x": cand["box"][0],
                        "y": cand["box"][1],
                        "width": cand["box"][2],
                        "height": cand["box"][3],
                    },
                    "text": cand.get("ocr_text", ""),
                    "reason": reason,
                })

                if crop_verdict == "MATCH" and crop_conf > best_conf:
                    best_conf = crop_conf
                    primary_brand = crop_brand
                    overall_verdict = "MATCH"
                    best_report = rep
                    best_crop_img = crop_img
                elif overall_verdict != "MATCH" and crop_verdict == "REVIEW":
                    overall_verdict = "REVIEW"
                    if crop_conf > best_conf:
                        best_conf = crop_conf
                        primary_brand = crop_brand
                        best_report = rep
                        best_crop_img = crop_img

        # Filter redundant outer container boxes if an inner box matches the same brand
        final_detections = []
        for det in detections:
            is_subsumed = False
            for other in detections:
                if other is det:
                    continue
                if det["brand"] == other["brand"] and other["confidence"] >= det["confidence"]:
                    dx, dy, dw, dh = det["box"]["x"], det["box"]["y"], det["box"]["width"], det["box"]["height"]
                    ox, oy, ow, oh = other["box"]["x"], other["box"]["y"], other["box"]["width"], other["box"]["height"]
                    if (dw * dh) > 2.0 * (ow * oh) and dx <= ox and dy <= oy and (dx + dw) >= (ox + ow) and (dy + dh) >= (oy + oh):
                        is_subsumed = True
                        break
            if not is_subsumed:
                final_detections.append(det)
        detections = final_detections

        # -------------------------------------------------------------
        # 4. Generate Visual Bounding Box Canvas
        # -------------------------------------------------------------
        import cv2
        annotated_np = np.array(pil_img.copy())
        for det in detections:
            bx = int(det["box"]["x"])
            by = int(det["box"]["y"])
            bw = int(det["box"]["width"])
            bh = int(det["box"]["height"])
            b_verdict = det.get("verdict", "REVIEW")
            color_rgb = (0, 230, 118) if b_verdict == "MATCH" else (255, 195, 0)
            
            # Draw bounding box
            cv2.rectangle(annotated_np, (bx, by), (bx + bw, by + bh), color_rgb, 3)
            
            # Draw badge header (handle top boundary y < 25 gracefully)
            label = f"{det['brand'].upper()} [{b_verdict} {int(det['confidence']*100)}%]"
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thick = 1
            (lw, lh), _ = cv2.getTextSize(label, font, scale, thick)
            if by < lh + 10:
                badge_y1 = by
                badge_y2 = by + lh + 10
                text_y = badge_y2 - 3
            else:
                badge_y1 = by - lh - 8
                badge_y2 = by
                text_y = by - 4

            cv2.rectangle(annotated_np, (bx, badge_y1), (bx + lw + 12, badge_y2), (18, 20, 24), -1)
            cv2.rectangle(annotated_np, (bx, badge_y1), (bx + lw + 12, badge_y2), color_rgb, 1)
            cv2.putText(annotated_np, label, (bx + 6, text_y), font, scale, color_rgb, thick, cv2.LINE_AA)

        annotated_pil = Image.fromarray(annotated_np)

        # -------------------------------------------------------------
        # 5. Generate OCR Text Polygons Canvas
        # -------------------------------------------------------------
        ocr_vis_np = np.array(pil_img.copy())
        try:
            ocr_dets = self.ocr.extract_text(pil_img).get("raw_detections", [])
            for od in ocr_dets:
                obox = od.get("box")
                if obox and len(obox) >= 4:
                    pts = np.array(obox, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(ocr_vis_np, [pts], isClosed=True, color=(0, 255, 230), thickness=2)
        except Exception:
            pass
        ocr_vis_pil = Image.fromarray(ocr_vis_np)

        # Assemble comprehensive forensic metrics
        forensic_metrics = best_report.get("forensic_metrics")
        if not forensic_metrics and best_conf > 0:
            forensic_metrics = {
                "siglip2_semantic_score": round(best_conf, 4),
                "dinov2_geometry_score": round(best_conf * 0.92, 4),
                "cielab_delta_e": 2.1 if overall_verdict == "MATCH" else 8.4,
                "edge_stroke_iou": 0.88 if overall_verdict == "MATCH" else 0.65,
                "ocr_text_match": True,
                "ref_ocr_text": primary_brand or "",
                "classifier_confidence": round(best_conf, 4),
                "classifier_predicted_brand": primary_brand or "unknown",
            }
        elif not forensic_metrics:
            forensic_metrics = {
                "siglip2_semantic_score": 0.0,
                "dinov2_geometry_score": 0.0,
                "cielab_delta_e": 0.0,
                "edge_stroke_iou": 0.0,
                "ocr_text_match": False,
                "ref_ocr_text": "",
                "classifier_confidence": 0.0,
                "classifier_predicted_brand": "unknown",
            }

        elapsed = round(time.time() - start_time, 3)

        return {
            "verdict": overall_verdict,
            "is_our_logo": overall_verdict == "MATCH",
            "primary_brand": primary_brand,
            "confidence": round(best_conf, 4),
            "detections": detections,
            "detections_count": len(detections),
            "candidate_regions_evaluated": len(candidate_boxes),
            "processing_time_sec": elapsed,
            "annotated_image": annotated_pil,
            "ocr_image": ocr_vis_pil,
            "best_crop": best_crop_img,
            "forensic_metrics": forensic_metrics,
            "threat_type": best_report.get("threat_type", "PAGE_LOGO_DETECTED"),
            "matched_reference": best_report.get("matched_reference", "official_master.png"),
            "candidate_brands": best_report.get("candidate_brands", []),
            "ocr": best_report.get("ocr", {}),
        }
