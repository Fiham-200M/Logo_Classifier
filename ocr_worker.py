"""
Persistent PaddleOCR worker subprocess for fast, single-load OCR inference.
"""
import sys
import json
import os
import numpy as np
from PIL import Image

try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(
        ocr_version="PP-OCRv5",
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
except Exception as e:
    ocr = None

# Prefix protocol to ignore library logging
sys.stdout.write("OCR_READY\n")
sys.stdout.flush()

for line in sys.stdin:
    path = line.strip()
    if not path:
        continue
    if path == "QUIT":
        break
    if ocr is None or not os.path.exists(path):
        sys.stdout.write(f"OCR_RESULT:{json.dumps([])}\n")
        sys.stdout.flush()
        continue
    try:
        img = Image.open(path).convert("RGB")
        res = ocr.predict(np.array(img))
        if res and len(res) > 0:
            texts = res[0].get("rec_texts", [])
            scores = res[0].get("rec_scores", [])
            polys = res[0].get("dt_polys", [])
            out = []
            for idx, (t, s) in enumerate(zip(texts, scores)):
                item = {"text": t, "confidence": float(s)}
                if idx < len(polys) and polys[idx] is not None:
                    p = polys[idx]
                    item["box"] = p.tolist() if hasattr(p, "tolist") else list(p)
                out.append(item)
        else:
            out = []
        sys.stdout.write(f"OCR_RESULT:{json.dumps(out)}\n")
        sys.stdout.flush()
    except Exception as e:
        sys.stdout.write(f"OCR_RESULT:{json.dumps([])}\n")
        sys.stdout.flush()
