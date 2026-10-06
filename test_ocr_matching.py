import os
import re
import torch
import easyocr
import numpy as np
from PIL import Image, ImageSequence


# ============================================================
# CONFIG
# ============================================================

OCR_DATABASE = "reference_embeddings/ocr_database.pt"
LOGO_DIR = "logos"

LANGUAGES = ["en"]
USE_GPU = torch.cuda.is_available()

# Minimum OCR confidence for query text
MIN_CONFIDENCE = 0.20


# ============================================================
# HELPERS
# ============================================================

def extract_frames(image_path):
    """
    Yield RGB numpy arrays for static or animated images (GIF).
    """
    with Image.open(image_path) as img:
        is_gif = image_path.lower().endswith(".gif")
        if not is_gif:
            frame = img.convert("RGB")
            yield np.array(frame)
            return

        for frame in ImageSequence.Iterator(img):
            rgb = frame.convert("RGB")
            yield np.array(rgb)


def normalize_text(text):
    """
    Normalize OCR text so small OCR differences do not matter
    too much.

    Example:
        "ZOOMGROUP" -> "zoomgroup"
        "SurgA-Play" -> "surgaplay"
    """
    if isinstance(text, dict):
        text = text.get("text", "")
    elif not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def text_similarity(a, b):
    """
    Simple character-based similarity using SequenceMatcher.
    """
    from difflib import SequenceMatcher

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def load_database():
    print("Loading OCR database...")

    db = torch.load(
        OCR_DATABASE,
        map_location="cpu",
        weights_only=False
    )

    print("OCR database loaded.")
    print(f"Brands: {len(db['names'])}")
    print()

    return db


def create_reader():
    print("Loading EasyOCR...")

    reader = easyocr.Reader(
        LANGUAGES,
        gpu=USE_GPU,
        verbose=True
    )

    print("EasyOCR loaded.")
    print()

    return reader


def extract_ocr(reader, image_path):
    """
    Run OCR on one image (including GIF frames) and return usable detections.
    """
    detections = []
    seen_hashes = set()

    for image_np in extract_frames(image_path):
        frame_hash = hash(image_np.tobytes())
        if frame_hash in seen_hashes:
            continue
        seen_hashes.add(frame_hash)

        try:
            results = reader.readtext(
                image_np,
                detail=1,
                paragraph=False,
                batch_size=1
            )
        except Exception as e:
            continue

        for result in results:
            if len(result) < 3:
                continue
            bbox, text, confidence = result

            if confidence < MIN_CONFIDENCE:
                continue

            text = str(text).strip()
            if not text:
                continue

            detections.append({
                "text": text,
                "confidence": float(confidence)
            })

    return detections


# ============================================================
# DATABASE INDEX
# ============================================================

def build_text_index(db):
    """
    Build:

        OCR text -> brands

    from the reference database.
    """

    index = {}

    for brand in db["names"]:

        entry = db["database"][brand]

        for item in entry.get("texts", []):

            if not item:
                continue

            if isinstance(item, dict):
                ref_text = item.get("text", "")
                normalized = item.get("normalized") or normalize_text(ref_text)
            else:
                ref_text = str(item)
                normalized = normalize_text(ref_text)

            if not normalized:
                continue

            index.setdefault(normalized, []).append({
                "brand": brand,
                "reference_text": ref_text
            })

    return index


# ============================================================
# MATCHING
# ============================================================

def match_ocr(query_detections, db):

    candidates = []

    for detection in query_detections:

        query_text = detection["text"]
        query_confidence = detection["confidence"]

        for brand in db["names"]:

            entry = db["database"][brand]

            for item in entry.get("texts", []):

                ref_text = item.get("text", "") if isinstance(item, dict) else str(item)

                similarity = text_similarity(
                    query_text,
                    ref_text
                )

                if similarity <= 0:
                    continue

                # Combine OCR confidence and text similarity.
                score = similarity * query_confidence

                candidates.append({
                    "brand": brand,
                    "query_text": query_text,
                    "reference_text": ref_text,
                    "ocr_confidence": query_confidence,
                    "text_similarity": similarity,
                    "score": score
                })

    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return candidates


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("OCR BRAND MATCHING TEST")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()

db = load_database()

reader = create_reader()

# Build index just to verify database contents.
text_index = build_text_index(db)

print("=" * 70)
print("DATABASE TEXT INDEX")
print("=" * 70)

print(f"Unique normalized OCR texts: {len(text_index)}")
print()

# ------------------------------------------------------------
# Test every reference logo against the OCR database
# ------------------------------------------------------------

for brand in db["names"]:

    entry = db["database"][brand]

    filename = entry["filename"]
    image_path = os.path.join(LOGO_DIR, filename)

    print("=" * 70)
    print(f"BRAND: {brand}")
    print(f"IMAGE: {filename}")
    print("=" * 70)

    if not os.path.exists(image_path):
        print("[SKIP] Image not found.")
        print()
        continue

    try:
        detections = extract_ocr(
            reader,
            image_path
        )

    except Exception as e:
        print(f"[ERROR] OCR failed: {e}")
        print()
        continue

    if not detections:
        print("OCR: No usable text detected.")
        print()
        continue

    print("OCR detected:")

    for detection in detections:
        print(
            f"  {detection['text']:<30} "
            f"confidence={detection['confidence']:.4f}"
        )

    print()
    print("Top matches:")

    matches = match_ocr(
        detections,
        db
    )

    # Keep only the best match for each brand.
    best_by_brand = {}

    for match in matches:

        brand_name = match["brand"]

        if (
            brand_name not in best_by_brand
            or match["score"] > best_by_brand[brand_name]["score"]
        ):
            best_by_brand[brand_name] = match

    ranked = sorted(
        best_by_brand.values(),
        key=lambda x: x["score"],
        reverse=True
    )

    for rank, match in enumerate(ranked[:5], 1):

        print(
            f"  {rank}. {match['brand']:<15} "
            f"score={match['score']:.4f} "
            f"similarity={match['text_similarity']:.4f} "
            f"OCR={match['ocr_confidence']:.4f} "
            f"query='{match['query_text']}' "
            f"ref='{match['reference_text']}'"
        )

    print()


print("=" * 70)
print("OCR MATCHING TEST COMPLETE")
print("=" * 70)