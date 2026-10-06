"""
Logo Candidate Detection Module.
Discovers and extracts logo candidate bounding boxes and crops from full webpage screenshots.
Modularly decouples region proposal generation from the core Logo Forensics Engine.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

import config
from ocr_engine import OCREngine, normalize_text
from reference_database.reference_store import ReferenceStore


class LogoCandidateDetector:
    """
    Dedicated logo-candidate detection engine for full webpage screenshots.
    Combines OCR token discovery, visual saliency/edge contours, and structural layout heuristics
    (header, navigation, banners) to produce high-recall candidate regions.
    """

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        ref_store: Optional[ReferenceStore] = None
    ):
        self.ocr = ocr_engine or OCREngine()
        self.ref_store = ref_store or ReferenceStore()
        self.protected_brands = set(self.ref_store.brand_names)

    def detect_candidates(
        self,
        image_input: Union[str, Path, Image.Image],
        output_crops_dir: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """
        Extracts candidate logo regions from a full webpage screenshot.

        Returns:
            List of candidate dictionaries:
            [
                {
                    "bbox": [x1, y1, x2, y2],
                    "crop_image": PIL.Image,
                    "crop_path": Optional[str],
                    "source": "header|footer|navigation|content|favicon|ocr_cluster|contour",
                    "detection_confidence": float,
                    "brand_hint": Optional[str],
                    "ocr_text": Optional[str]
                }
            ]
        """
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image_input)}")

        w, h = pil_img.size
        candidates: List[Dict[str, Any]] = []
        seen_boxes: List[Tuple[int, int, int, int]] = []

        # -------------------------------------------------------------
        # 1. OCR Token & Cluster Proposals
        # -------------------------------------------------------------
        raw_ocr = self.ocr.read_text(pil_img)
        for item in raw_ocr:
            text = item.get("text", "")
            conf = float(item.get("confidence", 0.0))
            box = item.get("box")
            if not box or len(box) < 4:
                continue

            clean_text = normalize_text(text)
            clean_with_spaces = text.lower()

            clean_upper = clean_text.upper()
            sub_s = clean_upper.replace("31", "SI").replace("3", "S").replace("1", "I").replace("0", "O")
            sub_e = clean_upper.replace("31", "SI").replace("3", "E").replace("1", "I").replace("0", "O")

            matched_brand = None
            for b in self.protected_brands:
                b_up = b.upper()
                # Direct containment: Brand name is in the OCR text (e.g. "OFFICIAL ASIA200 VIP")
                if (b_up in clean_upper or b in clean_with_spaces or
                    b_up in sub_s or b_up in sub_e):
                    matched_brand = b
                    break
                # OCR text is a major portion of the brand name (at least 4 chars and covering most of brand)
                if len(clean_upper) >= 4 and len(clean_upper) >= (len(b_up) - 1):
                    if clean_upper in b_up or sub_s in b_up or sub_e in b_up:
                        matched_brand = b
                        break
                # Handle fuzzy variations (e.g. 71sia100 -> asia100, a31a200 -> asia200)
                if len(clean_upper) >= 4 and len(b_up) >= 4:
                    from ocr_engine import levenshtein_similarity
                    if (levenshtein_similarity(clean_upper, b_up) >= 0.70 or
                        levenshtein_similarity(sub_s, b_up) >= 0.75):
                        matched_brand = b
                        break

            if matched_brand or (conf > 0.85 and len(clean_text) >= 4):
                xs = [pt[0] for pt in box]
                ys = [pt[1] for pt in box]
                bx1, bx2 = max(0, min(xs)), min(w, max(xs))
                by1, by2 = max(0, min(ys)), min(h, max(ys))

                bw = bx2 - bx1
                bh = by2 - by1
                pad_x = max(18, int(bw * 0.25))
                pad_y = max(14, int(bh * 0.40))

                crop_x1 = max(0, int(bx1 - pad_x))
                crop_y1 = max(0, int(by1 - pad_y))
                crop_x2 = min(w, int(bx2 + pad_x))
                crop_y2 = min(h, int(by2 + pad_y))

                source = "header" if crop_y1 < int(h * 0.25) else ("footer" if crop_y1 > int(h * 0.80) else "content")
                box_tuple = (crop_x1, crop_y1, crop_x2, crop_y2)

                if not self._is_box_redundant(box_tuple, seen_boxes):
                    seen_boxes.append(box_tuple)
                    candidates.append({
                        "bbox": [crop_x1, crop_y1, crop_x2, crop_y2],
                        "source": f"{source}_ocr" if not matched_brand else "brand_ocr_cluster",
                        "detection_confidence": round(conf, 3),
                        "brand_hint": matched_brand,
                        "ocr_text": text,
                    })

        # -------------------------------------------------------------
        # 2. Structural Layout Proposals (Header & Navbar)
        # -------------------------------------------------------------
        # Webpage header typically houses the primary brand emblem
        header_h = min(h, max(80, int(h * 0.22)))
        header_box = (0, 0, w, header_h)
        if not self._is_box_redundant(header_box, seen_boxes, iou_thresh=0.60):
            seen_boxes.append(header_box)
            candidates.append({
                "bbox": [0, 0, w, header_h],
                "source": "header_layout",
                "detection_confidence": 0.80,
                "brand_hint": None,
                "ocr_text": None,
            })

        # Top-left logo anchor region (where 90% of sites position their brand mark)
        top_left_w = min(w, max(180, int(w * 0.45)))
        top_left_h = min(header_h, max(60, int(header_h * 0.90)))
        top_left_box = (0, 0, top_left_w, top_left_h)
        if not self._is_box_redundant(top_left_box, seen_boxes, iou_thresh=0.60):
            seen_boxes.append(top_left_box)
            candidates.append({
                "bbox": [0, 0, top_left_w, top_left_h],
                "source": "navigation_anchor",
                "detection_confidence": 0.85,
                "brand_hint": None,
                "ocr_text": None,
            })

        # -------------------------------------------------------------
        # 3. Visual Saliency & Contour Proposals
        # -------------------------------------------------------------
        np_arr = np.array(pil_img)
        gray = cv2.cvtColor(np_arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            aspect = bw / max(1, bh)
            # Filter for logo-like proportions and significant area
            if (w * h * 0.003) < area < (w * h * 0.35) and 0.4 < aspect < 6.5:
                # Prioritize upper half of webpage
                if y < int(h * 0.55):
                    box_tuple = (x, y, x + bw, y + bh)
                    if not self._is_box_redundant(box_tuple, seen_boxes, iou_thresh=0.50):
                        seen_boxes.append(box_tuple)
                        candidates.append({
                            "bbox": [x, y, x + bw, y + bh],
                            "source": "visual_contour",
                            "detection_confidence": 0.65,
                            "brand_hint": None,
                            "ocr_text": None,
                        })

        # -------------------------------------------------------------
        # 4. Generate PIL Crops and Optional Disk Paths
        # -------------------------------------------------------------
        if output_crops_dir:
            output_crops_dir.mkdir(parents=True, exist_ok=True)

        for idx, cand in enumerate(candidates):
            x1, y1, x2, y2 = cand["bbox"]
            crop_img = pil_img.crop((x1, y1, x2, y2))
            cand["crop_image"] = crop_img
            cand["crop_path"] = None

            if output_crops_dir:
                cpath = output_crops_dir / f"candidate_{idx:02d}_{cand['source']}.png"
                crop_img.save(cpath)
                cand["crop_path"] = str(cpath)

        return candidates

    @staticmethod
    def _is_box_redundant(
        box: Tuple[int, int, int, int],
        existing_boxes: List[Tuple[int, int, int, int]],
        iou_thresh: float = 0.55
    ) -> bool:
        """Determines if a bounding box significantly overlaps an already recorded candidate."""
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area <= 0:
            return True

        for ex1, ey1, ex2, ey2 in existing_boxes:
            ix1 = max(x1, ex1)
            iy1 = max(y1, ey1)
            ix2 = min(x2, ex2)
            iy2 = min(y2, ey2)

            inter_w = max(0, ix2 - ix1)
            inter_h = max(0, iy2 - iy1)
            inter_area = inter_w * inter_h

            if inter_area > 0:
                e_area = max(0, ex2 - ex1) * max(0, ey2 - ey1)
                union_area = area + e_area - inter_area
                iou = inter_area / max(1, union_area)
                if iou >= iou_thresh:
                    return True
        return False
