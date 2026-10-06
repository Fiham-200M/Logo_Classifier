import os
import io
import json
import random
from pathlib import Path

import torch
from PIL import Image, ImageEnhance
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
LOGO_DIR = BASE_DIR / "logos"
DB_DIR = BASE_DIR / "reference_embeddings"

DB_FILE = DB_DIR / "brand_database.pt"
BRAND_LIST_FILE = DB_DIR / "brand_list.txt"

REPORT_FILE = DB_DIR / "color_preserving_robustness_report.txt"

MODEL_ID = "google/siglip2-base-patch16-224"

SEED = 42
random.seed(SEED)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
}


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SIGLIP 2 COLOR-PRESERVING ROBUSTNESS TEST")
print("=" * 70)

print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD DATABASE
# ============================================================

print("\nLoading reference database...")

db = torch.load(
    DB_FILE,
    map_location=device,
    weights_only=False,
)

reference_embeddings = db["embeddings"].to(device)

# Normalize again for safety
reference_embeddings = reference_embeddings / (
    reference_embeddings.norm(dim=-1, keepdim=True) + 1e-12
)

brand_list = []

with open(BRAND_LIST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if not line:
            continue

        parts = line.split(maxsplit=1)

        if len(parts) == 2:
            brand_list.append({
                "brand": parts[0],
                "filename": parts[1],
            })


print(f"Reference brands: {len(brand_list)}")
print(f"Embedding shape: {tuple(reference_embeddings.shape)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SigLIP 2...")

processor = AutoProcessor.from_pretrained(MODEL_ID)

model = AutoModel.from_pretrained(MODEL_ID)

model = model.to(device)
model.eval()

print("Model loaded.")


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    """
    Load image while preserving its original colors.

    GIF -> first frame
    Transparency -> composite onto white background
    """

    img = Image.open(path)

    if getattr(img, "is_animated", False):
        img.seek(0)

    img = img.convert("RGBA")

    # Preserve logo colors.
    # Only remove transparency by compositing onto white.
    background = Image.new("RGBA", img.size, (255, 255, 255, 255))

    background.alpha_composite(img)

    return background.convert("RGB")


# ============================================================
# VARIATIONS
# ============================================================

def create_variations(img):
    """
    Create realistic website/rendering variations.

    IMPORTANT:
    - No hue changes
    - No saturation changes
    - No grayscale
    - No color replacement
    """

    variations = []

    # --------------------------------------------------------
    # 1. Original
    # --------------------------------------------------------

    variations.append(("original", img.copy()))

    # --------------------------------------------------------
    # 2. Resize smaller
    # --------------------------------------------------------

    for size in [0.75, 0.50, 0.35]:

        w, h = img.size

        nw = max(16, int(w * size))
        nh = max(16, int(h * size))

        resized = img.resize(
            (nw, nh),
            Image.Resampling.LANCZOS
        )

        variations.append(
            (f"resize_{int(size * 100)}pct", resized)
        )

    # --------------------------------------------------------
    # 3. Add padding
    # --------------------------------------------------------

    for padding in [20, 50, 100]:

        w, h = img.size

        padded = Image.new(
            "RGB",
            (w + padding * 2, h + padding * 2),
            "white",
        )

        padded.paste(img, (padding, padding))

        variations.append(
            (f"padding_{padding}", padded)
        )

    # --------------------------------------------------------
    # 4. Slight crop
    # --------------------------------------------------------

    w, h = img.size

    if w > 20 and h > 20:

        crop_ratio = 0.95

        left = int(w * (1 - crop_ratio) / 2)
        top = int(h * (1 - crop_ratio) / 2)

        right = w - left
        bottom = h - top

        cropped = img.crop(
            (left, top, right, bottom)
        )

        variations.append(
            ("slight_crop", cropped)
        )

    # --------------------------------------------------------
    # 5. JPEG compression
    # --------------------------------------------------------

    for quality in [90, 70, 50]:

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="JPEG",
            quality=quality,
        )

        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")

        variations.append(
            (f"jpeg_q{quality}", compressed)
        )

    # --------------------------------------------------------
    # 6. WebP compression
    # --------------------------------------------------------

    for quality in [90, 70, 50]:

        buffer = io.BytesIO()

        img.save(
            buffer,
            format="WEBP",
            quality=quality,
        )

        buffer.seek(0)

        compressed = Image.open(buffer).convert("RGB")

        variations.append(
            (f"webp_q{quality}", compressed)
        )

    # --------------------------------------------------------
    # 7. Slight brightness change
    # --------------------------------------------------------

    brightness_up = ImageEnhance.Brightness(img).enhance(1.05)

    brightness_down = ImageEnhance.Brightness(img).enhance(0.95)

    variations.append(
        ("brightness_plus5", brightness_up)
    )

    variations.append(
        ("brightness_minus5", brightness_down)
    )

    # --------------------------------------------------------
    # 8. Slight contrast change
    # --------------------------------------------------------

    contrast_up = ImageEnhance.Contrast(img).enhance(1.05)

    contrast_down = ImageEnhance.Contrast(img).enhance(0.95)

    variations.append(
        ("contrast_plus5", contrast_up)
    )

    variations.append(
        ("contrast_minus5", contrast_down)
    )

    return variations


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def get_embedding(image):

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    # Transformers 5.17 in this environment returns an object
    # from get_image_features(), so use the vision model directly.
    vision_output = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    embedding = vision_output.pooler_output

    embedding = embedding / (
        embedding.norm(dim=-1, keepdim=True) + 1e-12
    )

    return embedding


# ============================================================
# TEST
# ============================================================

all_results = []

total = 0
correct = 0

top3_correct = 0

worst_cases = []


print("\nStarting robustness test...\n")

for index, item in enumerate(brand_list, start=1):

    brand = item["brand"]
    filename = item["filename"]

    path = LOGO_DIR / filename

    if not path.exists():

        print(
            f"[WARNING] Missing file: {filename}"
        )

        continue

    print(
        f"[{index:02d}/{len(brand_list)}] "
        f"{brand}"
    )

    try:

        image = load_image(path)

        variations = create_variations(image)

        for variation_name, variation_image in variations:

            embedding = get_embedding(variation_image)

            similarities = (
                embedding @ reference_embeddings.T
            )[0]

            top_values, top_indices = torch.topk(
                similarities,
                k=min(5, len(brand_list))
            )

            top1_index = top_indices[0].item()
            top1_score = top_values[0].item()

            top2_score = top_values[1].item()

            margin = top1_score - top2_score

            top3_indices = [
                x.item()
                for x in top_indices[:3]
            ]

            predicted_brand = brand_list[
                top1_index
            ]["brand"]

            is_correct = (
                predicted_brand == brand
            )

            is_top3 = (
                brand in [
                    brand_list[x]["brand"]
                    for x in top3_indices
                ]
            )

            total += 1

            if is_correct:
                correct += 1

            if is_top3:
                top3_correct += 1

            result = {
                "brand": brand,
                "filename": filename,
                "variation": variation_name,
                "predicted": predicted_brand,
                "correct": is_correct,
                "top1": top1_score,
                "top2": top2_score,
                "margin": margin,
            }

            all_results.append(result)

            if not is_correct:

                worst_cases.append(result)

            # Keep lowest-margin cases too
            worst_cases.append({
                **result,
                "_margin_case": True
            })

    except Exception as e:

        print(
            f"  ERROR: {filename}: {e}"
        )


# ============================================================
# SORT WORST CASES
# ============================================================

wrong_predictions = [
    r for r in all_results
    if not r["correct"]
]

low_margin_cases = sorted(
    all_results,
    key=lambda x: x["margin"]
)[:30]


# ============================================================
# STATISTICS
# ============================================================

accuracy = (
    correct / total * 100
    if total else 0
)

top3_accuracy = (
    top3_correct / total * 100
    if total else 0
)

similarities = [
    r["top1"]
    for r in all_results
]

margins = [
    r["margin"]
    for r in all_results
]

avg_similarity = (
    sum(similarities) / len(similarities)
    if similarities else 0
)

min_similarity = (
    min(similarities)
    if similarities else 0
)

avg_margin = (
    sum(margins) / len(margins)
    if margins else 0
)

min_margin = (
    min(margins)
    if margins else 0
)


# ============================================================
# REPORT
# ============================================================

lines = []

lines.append("=" * 80)
lines.append("SIGLIP 2 COLOR-PRESERVING ROBUSTNESS REPORT")
lines.append("=" * 80)

lines.append("")

lines.append(
    f"Brands tested: {len(brand_list)}"
)

lines.append(
    f"Total variation tests: {total}"
)

lines.append(
    f"Correct Top-1: {correct}/{total}"
)

lines.append(
    f"Top-1 accuracy: {accuracy:.2f}%"
)

lines.append(
    f"Top-3 accuracy: {top3_accuracy:.2f}%"
)

lines.append(
    f"Average Top-1 similarity: {avg_similarity:.4f}"
)

lines.append(
    f"Minimum Top-1 similarity: {min_similarity:.4f}"
)

lines.append(
    f"Average Top-1/Top-2 margin: {avg_margin:.4f}"
)

lines.append(
    f"Minimum Top-1/Top-2 margin: {min_margin:.4f}"
)

lines.append("")

lines.append("-" * 80)
lines.append("WRONG PREDICTIONS")
lines.append("-" * 80)

if not wrong_predictions:

    lines.append(
        "No wrong Top-1 predictions."
    )

else:

    for r in wrong_predictions:

        lines.append(
            f"{r['brand']:15s} | "
            f"{r['variation']:20s} | "
            f"Predicted: {r['predicted']:15s} | "
            f"Top1={r['top1']:.4f} | "
            f"Top2={r['top2']:.4f} | "
            f"Margin={r['margin']:.4f}"
        )


lines.append("")

lines.append("-" * 80)
lines.append("LOWEST-MARGIN CASES")
lines.append("-" * 80)

for r in low_margin_cases:

    lines.append(
        f"{r['brand']:15s} | "
        f"{r['variation']:20s} | "
        f"Pred={r['predicted']:15s} | "
        f"Top1={r['top1']:.4f} | "
        f"Top2={r['top2']:.4f} | "
        f"Margin={r['margin']:.4f} | "
        f"{'CORRECT' if r['correct'] else 'WRONG'}"
    )


lines.append("")

lines.append("-" * 80)
lines.append("PER-BRAND RESULTS")
lines.append("-" * 80)

for brand in [x["brand"] for x in brand_list]:

    brand_results = [
        r for r in all_results
        if r["brand"] == brand
    ]

    if not brand_results:
        continue

    brand_correct = sum(
        1 for r in brand_results
        if r["correct"]
    )

    brand_accuracy = (
        brand_correct /
        len(brand_results) *
        100
    )

    lowest_score = min(
        r["top1"]
        for r in brand_results
    )

    lowest_margin = min(
        r["margin"]
        for r in brand_results
    )

    lines.append(
        f"{brand:15s} | "
        f"{brand_correct:2d}/{len(brand_results):2d} "
        f"({brand_accuracy:6.2f}%) | "
        f"MinSim={lowest_score:.4f} | "
        f"MinMargin={lowest_margin:.4f}"
    )


lines.append("")

lines.append("=" * 80)
lines.append("IMPORTANT")
lines.append("=" * 80)

lines.append(
    "This test preserves the original logo colors."
)

lines.append(
    "It does NOT test color-changing attacks/variants."
)

lines.append(
    "Artificial variations do not guarantee real-world website accuracy."
)

lines.append(
    "A separate color-change rejection test should be performed."
)

with open(
    REPORT_FILE,
    "w",
    encoding="utf-8"
) as f:

    f.write("\n".join(lines))


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("ROBUSTNESS TEST COMPLETE")
print("=" * 70)

print(
    f"Total tests:       {total}"
)

print(
    f"Top-1 correct:     {correct}/{total}"
)

print(
    f"Top-1 accuracy:    {accuracy:.2f}%"
)

print(
    f"Top-3 accuracy:    {top3_accuracy:.2f}%"
)

print(
    f"Average similarity:{avg_similarity:.4f}"
)

print(
    f"Minimum similarity: {min_similarity:.4f}"
)

print(
    f"Average margin:    {avg_margin:.4f}"
)

print(
    f"Minimum margin:    {min_margin:.4f}"
)

print(
    f"\nReport saved to:\n{REPORT_FILE}"
)
