"""
OCR Model Wrapper Module.
Wraps PaddleOCR (via isolated background worker process) to prevent Windows CUDA/threading lockups.
Provides normalized alphanumeric parsing, confusion resolution, and confidence filtering.
Note: Absence of detected text does NOT penalize strong visual evidence.
"""

from typing import Dict, List, Any, Optional, Tuple
import os
import sys
import re
from pathlib import Path
from PIL import Image

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from ocr_engine import OCREngine, normalize_text, normalize_ocr_confusions


def normalize_clean_string(text: str) -> str:
    """Standardizes string into clean uppercase alphanumeric text."""
    if not text:
        return ""
    # Strip special punctuation but preserve letters and numbers
    cleaned = re.sub(r"[^A-Za-z0-9]+", "", text).upper()
    return cleaned


class OCRModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OCRModel, cls).__new__(cls)
            cls._instance._engine = None
        return cls._instance

    def __init__(self):
        if self._engine is None:
            self._engine = OCREngine()

    def extract_text(self, image: Image.Image) -> Dict[str, Any]:
        """
        Runs OCR on image and extracts structured text evidence.

        Returns:
            Dict containing:
                - texts: List[str] (raw detected strings)
                - scores: List[float] (detection confidence scores)
                - normalized_text: str (concatenated clean uppercase string)
                - has_text: bool
        """
        raw_results = self._engine.read_text(image)

        texts = []
        scores = []
        clean_tokens = []

        for item in raw_results:
            t = item.get("text", "").strip()
            c = float(item.get("confidence", 0.0))
            if t:
                texts.append(t)
                scores.append(round(c, 4))
                norm = normalize_clean_string(t)
                if norm:
                    clean_tokens.append(norm)

        normalized_combined = " ".join(clean_tokens) if clean_tokens else ""

        return {
            "texts": texts,
            "scores": scores,
            "normalized_text": normalized_combined,
            "has_text": len(texts) > 0,
            "raw_detections": raw_results,
        }

    def extract_text_with_frames(
        self,
        image: Image.Image,
        key_frames: Optional[List[Image.Image]] = None,
    ) -> Dict[str, Any]:
        """
        Runs OCR on primary image. If no text is detected and key_frames are provided
        (e.g., for animated GIFs), scans keyframes and aggregates detected text tokens.
        """
        primary = self.extract_text(image)
        if primary.get("has_text") or not key_frames:
            return primary

        all_texts: List[str] = []
        all_scores: List[float] = []
        all_raw: List[Dict[str, Any]] = []

        for frame in key_frames:
            frame_res = self.extract_text(frame)
            if frame_res.get("has_text"):
                for t, s, raw in zip(frame_res["texts"], frame_res["scores"], frame_res["raw_detections"]):
                    if t not in all_texts:
                        all_texts.append(t)
                        all_scores.append(s)
                        all_raw.append(raw)

        if all_texts:
            clean_tokens = [normalize_clean_string(t) for t in all_texts if normalize_clean_string(t)]
            return {
                "texts": all_texts,
                "scores": all_scores,
                "normalized_text": " ".join(clean_tokens) if clean_tokens else "",
                "has_text": True,
                "raw_detections": all_raw,
            }

        return primary


if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python -m models.ocr_model <image_path>")
        print("   or: python models/ocr_model.py <image_path>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: File not found: {img_path}")
        sys.exit(1)

    print(f"[OCR] Processing: {img_path}")
    ocr = OCRModel()

    if img_path.suffix.lower() == ".gif":
        from preprocessing.media_normalizer import MediaNormalizer
        norm = MediaNormalizer.normalize_media(img_path)
        if norm.get("metadata", {}).get("is_animated"):
            print(f"[OCR] Animated GIF detected ({norm['metadata']['frames_processed']} frames). Scanning keyframes...")
            res = ocr.extract_text_with_frames(norm["normalized_rgb"], key_frames=norm.get("key_frames", []))
        else:
            image = Image.open(img_path)
            res = ocr.extract_text(image)
    else:
        image = Image.open(img_path)
        res = ocr.extract_text(image)

    print("\n" + "=" * 50)
    print("OCR EXTRACTION RESULTS")
    print("=" * 50)
    print(f"Has Text:        {res['has_text']}")
    print(f"Detected Tokens: {res['texts']}")
    print(f"Confidences:     {res['scores']}")
    print(f"Normalized Text: '{res['normalized_text']}'")
    print("=" * 50)

