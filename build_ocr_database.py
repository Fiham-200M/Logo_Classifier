import os
import re
import time

import torch
import easyocr
import numpy as np

from PIL import Image, ImageSequence


# ============================================================
# CONFIG
# ============================================================

LOGO_DIR = "logos"
COLOR_DATABASE_PATH = "reference_embeddings/color_database.pt"
OUTPUT_PATH = "reference_embeddings/ocr_database.pt"

LANGUAGES = ["en"]

# Keep low because logo text can be stylized.
CONFIDENCE_THRESHOLD = 0.25

# ============================================================
# HELPERS
# ============================================================

def normalize_text(text):
    """
    Normalize OCR text for comparison.

    Example:
        "Where Asia $ Fortune"
        -> "whereasiafortune"
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text


def is_image_file(filename):
    return filename.lower().endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif")
    )


def load_color_database():
    if not os.path.exists(COLOR_DATABASE_PATH):
        raise FileNotFoundError(
            f"Color database not found: {COLOR_DATABASE_PATH}"
        )

    db = torch.load(
        COLOR_DATABASE_PATH,
        map_location="cpu",
        weights_only=False,
    )

    return db


def extract_frames(image_path):
    """
    Yield RGB numpy arrays.

    Static images:
        one frame

    GIF:
        every frame individually
    """

    with Image.open(image_path) as img:

        is_gif = image_path.lower().endswith(".gif")

        if not is_gif:
            frame = img.convert("RGB")
            yield np.array(frame)
            return

        # Animated GIF
        for frame in ImageSequence.Iterator(img):
            rgb = frame.convert("RGB")
            yield np.array(rgb)


# ============================================================
# OCR
# ============================================================

def process_logo(reader, brand_name, filename):

    image_path = os.path.join(LOGO_DIR, filename)

    if not os.path.exists(image_path):
        print(f"  [WARNING] Missing file: {image_path}")
        return {
            "filename": filename,
            "num_frames": 0,
            "frames_processed": 0,
            "unique_frames": 0,
            "detections": [],
            "texts": [],
        }

    print()
    print("-" * 70)
    print(f"Brand:    {brand_name}")
    print(f"File:     {filename}")

    detections = []

    total_frames = 0
    processed_frames = 0

    # Avoid OCR'ing identical GIF frames repeatedly.
    seen_frames = set()

    start_time = time.time()

    for frame_index, image_np in enumerate(extract_frames(image_path)):

        total_frames += 1

        # Hash frame content to detect identical frames.
        frame_hash = hash(image_np.tobytes())

        if frame_hash in seen_frames:
            continue

        seen_frames.add(frame_hash)
        processed_frames += 1

        try:
            results = reader.readtext(
                image_np,
                detail=1,
                paragraph=False,
                batch_size=1,
            )

        except Exception as e:
            print(
                f"  [FRAME ERROR] frame={frame_index}: {e}"
            )
            continue

        for result in results:

            if len(result) < 3:
                continue

            bbox, text, confidence = result

            text = str(text).strip()
            confidence = float(confidence)

            if not text:
                continue

            normalized = normalize_text(text)

            if not normalized:
                continue

            detection = {
                "frame": frame_index,
                "text": text,
                "normalized": normalized,
                "confidence": confidence,
                "bbox": bbox,
            }

            detections.append(detection)

    # --------------------------------------------------------
    # Aggregate unique OCR texts
    # --------------------------------------------------------

    aggregated = {}

    for detection in detections:

        normalized = detection["normalized"]

        if normalized not in aggregated:
            aggregated[normalized] = {
                "text": detection["text"],
                "normalized": normalized,
                "max_confidence": detection["confidence"],
                "count": 1,
                "frames": [detection["frame"]],
            }

        else:
            item = aggregated[normalized]

            item["count"] += 1

            if detection["confidence"] > item["max_confidence"]:
                item["max_confidence"] = detection["confidence"]
                item["text"] = detection["text"]

            if detection["frame"] not in item["frames"]:
                item["frames"].append(detection["frame"])

    texts = list(aggregated.values())

    # Sort strongest OCR candidates first.
    texts.sort(
        key=lambda x: (
            x["max_confidence"],
            x["count"],
        ),
        reverse=True,
    )

    elapsed = time.time() - start_time

    print(f"  Total frames:       {total_frames}")
    print(f"  Unique frames:      {processed_frames}")
    print(f"  OCR detections:     {len(detections)}")
    print(f"  Unique OCR texts:   {len(texts)}")
    print(f"  Time:               {elapsed:.2f}s")

    if texts:
        print("  Text candidates:")

        for item in texts[:10]:

            print(
                f"    - {item['text']!r} "
                f"(confidence={item['max_confidence']:.3f}, "
                f"count={item['count']})"
            )

    else:
        print("  No text detected.")

    return {
        "filename": filename,
        "num_frames": total_frames,
        "frames_processed": processed_frames,
        "unique_frames": processed_frames,
        "detections": detections,
        "texts": texts,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("OCR REFERENCE DATABASE BUILDER")
    print("=" * 70)

    print(f"PyTorch:          {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(
            f"VRAM:             "
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    print()

    # --------------------------------------------------------
    # Load existing color database
    # --------------------------------------------------------

    print("Loading color database...")

    color_db = load_color_database()

    names = color_db["names"]
    database = color_db["database"]

    print(f"Brands found: {len(names)}")

    # --------------------------------------------------------
    # Load EasyOCR once
    # --------------------------------------------------------

    print()
    print("Loading EasyOCR...")

    use_gpu = torch.cuda.is_available()

    reader = easyocr.Reader(
        LANGUAGES,
        gpu=use_gpu,
        verbose=True,
    )

    print("EasyOCR loaded.")
    print(f"EasyOCR GPU: {use_gpu}")

    # --------------------------------------------------------
    # Process all reference logos
    # --------------------------------------------------------

    results = {}

    overall_start = time.time()

    for index, brand_name in enumerate(names, start=1):

        entry = database[brand_name]

        filename = entry["filename"]

        print()
        print("=" * 70)
        print(f"[{index}/{len(names)}] Processing {brand_name}")
        print("=" * 70)

        result = process_logo(
            reader,
            brand_name,
            filename,
        )

        results[brand_name] = result

    overall_elapsed = time.time() - overall_start

    # --------------------------------------------------------
    # Build database
    # --------------------------------------------------------

    ocr_database = {
        "version": 1,
        "feature_type": "EasyOCR",
        "languages": LANGUAGES,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "names": names,
        "database": results,
    }

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True,
    )

    torch.save(
        ocr_database,
        OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total_detections = sum(
        len(item["detections"])
        for item in results.values()
    )

    brands_with_text = sum(
        1
        for item in results.values()
        if len(item["texts"]) > 0
    )

    print()
    print("=" * 70)
    print("OCR DATABASE BUILD COMPLETE")
    print("=" * 70)

    print(f"Brands processed:       {len(results)}")
    print(f"Brands with OCR text:   {brands_with_text}")
    print(f"Brands without text:    {len(results) - brands_with_text}")
    print(f"Total detections:       {total_detections}")
    print(f"Total time:             {overall_elapsed:.2f}s")
    print()
    print(f"Saved to:")
    print(f"  {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()