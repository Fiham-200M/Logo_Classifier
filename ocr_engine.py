"""
OCR Engine and Fuzzy Matching Module.
Supports PaddleOCR (PP-OCRv5) and EasyOCR engines.
Treats OCR as supporting evidence with normalized fuzzy matching and neutral baseline scoring.
"""

import os
import re
import json
import subprocess
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
from PIL import Image

import config
from preprocessing import upscale_for_ocr, load_image_safe, composite_on_background


# ============================================================
# TEXT NORMALIZATION & OCR CONFUSIONS
# ============================================================

def normalize_text(text: Union[str, dict, None]) -> str:
    """Standard alphanumeric lowercase normalization."""
    if isinstance(text, dict):
        text = text.get("text", "")
    elif not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def normalize_ocr_confusions(text: str) -> str:
    """
    Optional mapping for common visual OCR character substitutions.
    O/0/D, I/1/l, S/5, B/8, Z/2.
    """
    norm = normalize_text(text)
    trans = str.maketrans({
        "0": "o",
        "d": "o",
        "1": "i",
        "l": "i",
        "5": "s",
        "8": "b",
        "2": "z",
    })
    return norm.translate(trans)


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in range [0, 1]."""
    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost_ins = curr[j - 1] + 1
            cost_del = prev[j] + 1
            cost_rep = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(cost_ins, cost_del, cost_rep))
        prev = curr

    distance = prev[-1]
    max_len = max(len(a), len(b))
    return max(0.0, 1.0 - (distance / max_len))


def fuzzy_match_score(query_text: str, candidate_brand: str, ref_texts: List[str]) -> float:
    """
    Score similarity between a detected OCR string and a candidate brand / reference texts.
    Prioritizes exact and confusion matches; penalizes single-character conflicts on short brand codes.
    """
    q_norm = normalize_text(query_text)
    c_norm = normalize_text(candidate_brand)

    if not q_norm or not c_norm:
        return 0.0

    # 1. Exact match with candidate brand
    if q_norm == c_norm:
        return 1.0

    # 2. OCR visual confusion mapping (O<->0, I<->1, S<->5, B<->8, Z<->2)
    q_conf = normalize_ocr_confusions(q_norm)
    c_conf = normalize_ocr_confusions(c_norm)
    if q_conf == c_conf:
        return 0.95

    # Check for number conflict / typosquatting (e.g. nusa211 vs nusa2111, surga77 vs surga777, depo89 vs depo88)
    import re
    c_nums = re.findall(r"\d+", c_norm)
    q_nums = re.findall(r"\d+", q_norm)
    has_number_conflict = False
    if c_nums and q_nums:
        c_num = c_nums[-1]
        if c_num not in q_nums:
            has_number_conflict = True
        else:
            for qn in q_nums:
                if c_num in qn and c_num != qn:
                    has_number_conflict = True
                    break

    # 3. Check reference texts stored in database for this brand (secondary supporting evidence)
    for ref in ref_texts:
        r_norm = normalize_text(ref)
        if not r_norm:
            continue
        r_conf = normalize_ocr_confusions(r_norm)
        if q_norm == r_norm or q_conf == r_conf:
            # Reference text matched — but does it actually identify THIS brand?
            # If the brand name contains the query text (e.g. "200m" in "a200m"),
            # the match is weak because the distinguishing prefix is missing.
            if q_norm in c_norm and q_norm != c_norm:
                return 0.35
            # If the reference text also matches the candidate brand name, stronger
            if c_norm in r_norm or r_norm in c_norm:
                return 0.85
            return 0.50
        if c_norm in r_norm and (c_norm in q_norm or q_norm in r_norm):
            return 0.75

    # If there is a number conflict (e.g. nusa211 vs nusa2111), reject high substring score
    if has_number_conflict:
        return 0.20

    # 4. Candidate is contained in query (e.g. "asia100" in "whereasia100begins")
    if c_norm in q_norm:
        return 0.90
    if c_conf in q_conf:
        return 0.85

    # 5. Query is partial substring of candidate (e.g. "200m" inside "a200m")
    if q_norm in c_norm:
        ratio = len(q_norm) / len(c_norm)
        # Partial match cannot give high score because differentiating prefix is missing
        return max(0.20, 0.45 * ratio)

    # 6. Levenshtein edit distance
    lev = levenshtein_similarity(q_norm, c_norm)
    if len(c_norm) <= 7:
        # On short brand codes (e.g. a200m vs k200m, surga11 vs surga22), 1-char difference is a different brand
        lev = max(0.0, lev - 0.45)

    return float(lev)


# ============================================================
# OCR ENGINES (EasyOCR & PaddleOCR)
# ============================================================

class OCREngine:
    def __init__(self, preferred_engine: str = "auto"):
        self.engine_type = None
        self._reader = None
        self._paddle_python = None

        # Check for PaddleOCR persistent worker in .venv-ocr312
        self._worker_proc = None
        paddle_venv = config.BASE_DIR / ".venv-ocr312" / "Scripts" / "python.exe"
        worker_script = config.BASE_DIR / "ocr_worker.py"
        if paddle_venv.exists() and worker_script.exists():
            self._paddle_python = str(paddle_venv)
            try:
                self._worker_proc = subprocess.Popen(
                    [self._paddle_python, str(worker_script)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    bufsize=1,
                )
                while True:
                    line = self._worker_proc.stdout.readline()
                    if not line:
                        break
                    if "OCR_READY" in line:
                        self.engine_type = "paddleocr_worker"
                        break
                if not self.engine_type:
                    self._worker_proc.terminate()
                    self._worker_proc = None
            except Exception:
                self._worker_proc = None

        if not self.engine_type:
            try:
                import easyocr
                self._reader = easyocr.Reader(["en"], gpu=True, verbose=False)
                self.engine_type = "easyocr"
            except Exception:
                try:
                    from paddleocr import PaddleOCR
                    self._reader = PaddleOCR(ocr_version="PP-OCRv5", lang="en", use_doc_orientation_classify=False)
                    self.engine_type = "paddleocr_direct"
                except Exception as e:
                    print(f"[OCREngine] Warning: No OCR engine loaded ({e}).")

        print(f"[OCREngine] Active engine: {self.engine_type}")

    def __del__(self):
        if getattr(self, "_worker_proc", None) is not None:
            try:
                self._worker_proc.stdin.write("QUIT\n")
                self._worker_proc.stdin.flush()
                self._worker_proc.terminate()
            except Exception:
                pass

    def read_text(self, image: Image.Image) -> List[Dict[str, Union[str, float]]]:
        """
        Extract text from PIL Image using the active OCR engine.
        Returns list of dicts: [{'text': str, 'confidence': float}, ...]
        """
        results = []
        if image.mode in ("RGBA", "LA") or ("transparency" in image.info):
            image = composite_on_background(image, (255, 255, 255))
        orig_w, orig_h = image.size
        ocr_img = upscale_for_ocr(image.convert("RGB"), min_height=80)
        scale_x = ocr_img.width / max(1, orig_w)
        scale_y = ocr_img.height / max(1, orig_h)

        if self.engine_type == "paddleocr_worker" and self._worker_proc:
            tmp_path = config.BASE_DIR / f"scratch_ocr_query_{os.getpid()}.png"
            try:
                ocr_img.save(tmp_path)
                self._worker_proc.stdin.write(f"{tmp_path}\n")
                self._worker_proc.stdin.flush()
                while True:
                    resp = self._worker_proc.stdout.readline()
                    if not resp:
                        break
                    if resp.startswith("OCR_RESULT:"):
                        results = json.loads(resp[len("OCR_RESULT:"):].strip())
                        break
                if scale_x != 1.0 or scale_y != 1.0:
                    for item in results:
                        if "box" in item:
                            item["box"] = [[float(pt[0]) / scale_x, float(pt[1]) / scale_y] for pt in item["box"]]
            except Exception:
                pass
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass

        elif self.engine_type == "easyocr" and self._reader:
            try:
                raw = self._reader.readtext(np.array(ocr_img), detail=1, paragraph=False)
                for item in raw:
                    bbox, text, conf = item
                    box_pts = [[float(pt[0]) / scale_x, float(pt[1]) / scale_y] for pt in bbox]
                    results.append({"text": text, "confidence": float(conf), "box": box_pts})
            except Exception:
                pass

        elif self.engine_type == "paddleocr_direct" and self._reader:
            try:
                ocr_img = upscale_for_ocr(image.convert("RGB"), min_height=80)
                res = self._reader.predict(np.array(ocr_img))
                if res:
                    texts = res[0].get("rec_texts", [])
                    scores = res[0].get("rec_scores", [])
                    for t, s in zip(texts, scores):
                        results.append({"text": t, "confidence": float(s)})
            except Exception:
                pass

        # Filter low confidence & too short garbage
        filtered = [
            r for r in results
            if r["confidence"] >= config.OCR_MIN_CONFIDENCE
            and len(normalize_text(r["text"])) >= config.OCR_MIN_CHAR_LENGTH
        ]
        return filtered

    def score_candidates(
        self,
        query_image_or_views: Union[Image.Image, Dict[str, Image.Image]],
        brand_names: List[str],
        ocr_database: Dict[str, dict],
    ) -> Tuple[np.ndarray, List[Dict[str, Union[str, float]]]]:
        """
        Score all candidate brands based on OCR text evidence.

        Returns:
            ocr_scores: Array of shape (len(brand_names),) in range [0, 1]
            detected_texts: List of unique detected text snippets with confidence
        """
        # Collect detections across primary OCR views
        if isinstance(query_image_or_views, dict):
            views_to_test = [
                query_image_or_views.get("original"),
                query_image_or_views.get("white_bg"),
                query_image_or_views.get("contrast_enhanced"),
            ]
        else:
            views_to_test = [query_image_or_views]

        all_detections = []
        seen_texts = set()

        for v in views_to_test:
            if v is not None:
                dets = self.read_text(v)
                for d in dets:
                    n = normalize_text(d["text"])
                    if n and n not in seen_texts:
                        seen_texts.add(n)
                        all_detections.append(d)

        num_brands = len(brand_names)

        # CRITICAL PRINCIPLE: If no OCR text is detected, assign NEUTRAL score to all brands.
        # A logo without readable text is NOT penalized!
        if not all_detections:
            return np.full(num_brands, config.OCR_NEUTRAL_SCORE, dtype=np.float32), []

        scores = np.zeros(num_brands, dtype=np.float32)

        for i, brand in enumerate(brand_names):
            ref_entry = ocr_database.get(brand, {})
            ref_texts = [
                (t.get("text", "") if isinstance(t, dict) else str(t))
                for t in ref_entry.get("texts", [])
            ]

            brand_matches = []
            for det in all_detections:
                text = det["text"]
                conf = det["confidence"]
                match_val = fuzzy_match_score(text, brand, ref_texts)
                brand_matches.append(match_val * conf)

            best_match = max(brand_matches) if brand_matches else 0.0

            # If there is a strong match, score scales from neutral up to 1.0
            # If all detected texts strongly disagree with this brand, score scales down slightly
            if best_match >= 0.50:
                score = config.OCR_NEUTRAL_SCORE + (best_match * (1.0 - config.OCR_NEUTRAL_SCORE))
            elif best_match >= 0.20:
                score = config.OCR_NEUTRAL_SCORE + (best_match - 0.20) * 0.3
            else:
                # Text was detected, but none matches this brand
                score = max(0.15, config.OCR_NEUTRAL_SCORE - 0.30)

            scores[i] = score

        return scores.astype(np.float32), all_detections


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py <image_path>")
        sys.exit(1)

    target = Path(sys.argv[1])
    if not target.exists():
        print(f"Error: File not found: {target}")
        sys.exit(1)

    print(f"[OCREngine] Reading: {target}")
    engine = OCREngine()

    if target.suffix.lower() == ".gif":
        from preprocessing.media_normalizer import MediaNormalizer
        norm = MediaNormalizer.normalize_media(target)
        if norm.get("metadata", {}).get("is_animated"):
            print(f"[OCREngine] Animated GIF detected ({norm['metadata']['frames_processed']} frames). Scanning keyframes...")
            results = []
            seen_texts = set()
            for frame in [norm["normalized_rgb"]] + norm.get("key_frames", []):
                for r in engine.read_text(frame):
                    txt = r.get("text", "").strip()
                    if txt and txt not in seen_texts:
                        seen_texts.add(txt)
                        results.append(r)
        else:
            img = Image.open(target)
            results = engine.read_text(img)
    else:
        img = Image.open(target)
        results = engine.read_text(img)

    print("\n" + "=" * 50)
    print(f"RAW OCR DETECTIONS ({len(results)} found)")
    print("=" * 50)
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Text: '{r.get('text', '')}'  | Confidence: {r.get('confidence', 0.0):.4f}")
    print("=" * 50)

