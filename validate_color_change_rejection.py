import os
import math
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from transformers import AutoProcessor, AutoModel

# ============================================================
# CONFIG
# ============================================================

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOGO_DIR = os.path.join(BASE_DIR, "logos")
DB_PATH = os.path.join(BASE_DIR, "reference_embeddings", "brand_database.pt")
BRAND_LIST = os.path.join(BASE_DIR, "reference_embeddings", "brand_list.txt")
REPORT_PATH = os.path.join(
    BASE_DIR,
    "reference_embeddings",
    "color_change_rejection_report.txt"
)

MODEL_ID = "google/siglip2-base-patch16-224"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SIGLIP 2 COLOR-CHANGE REJECTION TEST")
print("=" * 70)
print(f"Device: {device}")

if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# LOAD DATABASE
# ============================================================

print("\nLoading reference database...")

db = torch.load(DB_PATH, map_location=device, weights_only=False)

reference_embeddings = db["embeddings"].to(device)
reference_embeddings = reference_embeddings / (
    reference_embeddings.norm(dim=1, keepdim=True) + 1e-12
)

with open(BRAND_LIST, "r", encoding="utf-8") as f:
    brand_lines = [x.strip() for x in f if x.strip()]

brand_names = []

for line in brand_lines:
    # Expected format:
    # brand_name | filename
    if "|" in line:
        brand = line.split("|")[0].strip()
    else:
        brand = line.strip()

    brand_names.append(brand)

print(f"Reference brands: {len(brand_names)}")
print(f"Embedding shape: {tuple(reference_embeddings.shape)}")

# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SigLIP 2...")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID).to(device)
model.eval()

print("Model loaded successfully.")

# ============================================================
# IMAGE HELPERS
# ============================================================

def load_image(path):
    img = Image.open(path)

    # GIF -> first frame
    if img.mode == "P" or path.lower().endswith(".gif"):
        try:
            img.seek(0)
        except Exception:
            pass

    return img.convert("RGBA")


def get_logo_mask(img):
    """
    Identify likely logo pixels while protecting white/near-white
    background.

    Transparent pixels are always treated as background.

    For opaque images, near-white pixels are treated as background.
    """

    rgba = np.array(img).astype(np.uint8)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    # Transparent background
    mask = alpha > 10

    # Near-white background should remain unchanged.
    near_white = (
        (rgb[:, :, 0] > 245) &
        (rgb[:, :, 1] > 245) &
        (rgb[:, :, 2] > 245)
    )

    mask = mask & (~near_white)

    return mask


def hue_shift(img, degrees):
    """
    Shift hue of logo pixels only.
    Background remains unchanged.
    """

    rgba = np.array(img).astype(np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = get_logo_mask(img)

    hsv = np.array(Image.fromarray(rgb).convert("HSV"))

    shift = int(round((degrees / 360.0) * 255))

    hsv[:, :, 0] = (
        hsv[:, :, 0].astype(np.int16) + shift
    ) % 256

    shifted_rgb = np.array(
        Image.fromarray(hsv.astype(np.uint8)).convert("RGB")
    )

    result = rgb.copy()
    result[mask] = shifted_rgb[mask]

    output = np.dstack([result, alpha])

    return Image.fromarray(output.astype(np.uint8), "RGBA")


def saturation_change(img, factor):
    """
    Change saturation of logo pixels only.
    """

    rgba = np.array(img).astype(np.uint8)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = get_logo_mask(img)

    hsv = np.array(Image.fromarray(rgb).convert("HSV")).astype(np.float32)

    hsv[:, :, 1] *= factor
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)

    changed_rgb = np.array(
        Image.fromarray(hsv.astype(np.uint8)).convert("RGB")
    )

    result = rgb.copy()
    result[mask] = changed_rgb[mask]

    output = np.dstack([result, alpha])

    return Image.fromarray(output.astype(np.uint8), "RGBA")


def target_color_remap(img, target_rgb):
    """
    Recolor all non-background logo pixels toward a target color.

    This is important for black/white logos where hue shifting alone
    may not visibly change the color.
    """

    rgba = np.array(img).astype(np.uint8)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = get_logo_mask(img)

    target = np.array(target_rgb, dtype=np.float32)

    original = rgb.astype(np.float32)

    # Estimate luminance so the logo's shape/detail is preserved.
    luminance = (
        0.299 * original[:, :, 0] +
        0.587 * original[:, :, 1] +
        0.114 * original[:, :, 2]
    )

    luminance_norm = luminance / 255.0

    new_rgb = target.reshape(1, 1, 3) * (
        0.35 + 0.65 * luminance_norm[:, :, None]
    )

    new_rgb = np.clip(new_rgb, 0, 255).astype(np.uint8)

    result = rgb.copy()
    result[mask] = new_rgb[mask]

    output = np.dstack([result, alpha])

    return Image.fromarray(output.astype(np.uint8), "RGBA")


def grayscale_logo(img):
    """
    Remove logo color while keeping background unchanged.
    """

    rgba = np.array(img).astype(np.uint8)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = get_logo_mask(img)

    gray = np.array(ImageOps.grayscale(Image.fromarray(rgb)))
    gray_rgb = np.stack([gray, gray, gray], axis=-1)

    result = rgb.copy()
    result[mask] = gray_rgb[mask]

    output = np.dstack([result, alpha])

    return Image.fromarray(output.astype(np.uint8), "RGBA")


def invert_logo(img):
    """
    Extreme color-change negative.
    """

    rgba = np.array(img).astype(np.uint8)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]

    mask = get_logo_mask(img)

    inverted = 255 - rgb

    result = rgb.copy()
    result[mask] = inverted[mask]

    output = np.dstack([result, alpha])

    return Image.fromarray(output.astype(np.uint8), "RGBA")


# ============================================================
# CREATE COLOR-CHANGE VARIATIONS
# ============================================================

def create_color_variations(img):
    variations = []

    # Moderate hue changes
    variations.append(("hue_minus60", hue_shift(img, -60)))
    variations.append(("hue_minus40", hue_shift(img, -40)))
    variations.append(("hue_minus20", hue_shift(img, -20)))
    variations.append(("hue_plus20", hue_shift(img, 20)))
    variations.append(("hue_plus40", hue_shift(img, 40)))
    variations.append(("hue_plus60", hue_shift(img, 60)))

    # Saturation changes
    variations.append(("saturation_minus30", saturation_change(img, 0.70)))
    variations.append(("saturation_plus30", saturation_change(img, 1.30)))

    # Deliberate target-color changes
    variations.append(("recolor_red", target_color_remap(img, (220, 40, 40))))
    variations.append(("recolor_green", target_color_remap(img, (30, 180, 70))))
    variations.append(("recolor_blue", target_color_remap(img, (40, 90, 220))))
    variations.append(("recolor_purple", target_color_remap(img, (150, 50, 200))))

    # Strong negative cases
    variations.append(("grayscale", grayscale_logo(img)))
    variations.append(("invert", invert_logo(img)))

    return variations


# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def image_embedding(img):
    inputs = processor(
        images=img.convert("RGB"),
        return_tensors="pt"
    )

    inputs = {
        k: v.to(device)
        for k, v in inputs.items()
    }

    vision_output = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    embedding = vision_output.pooler_output

    embedding = embedding / (
        embedding.norm(dim=-1, keepdim=True) + 1e-12
    )

    return embedding[0]


# ============================================================
# TEST
# ============================================================

results = []

logo_files = sorted([
    f for f in os.listdir(LOGO_DIR)
    if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
])

print(f"\nLogos found: {len(logo_files)}")
print("\nGenerating color-change negatives...")

for index, filename in enumerate(logo_files, start=1):

    brand = os.path.splitext(filename)[0]

    path = os.path.join(LOGO_DIR, filename)

    try:
        original = load_image(path)

        variations = create_color_variations(original)

        print(
            f"[{index:02d}/{len(logo_files)}] "
            f"{brand}: {len(variations)} color variations"
        )

        for variation_name, variant in variations:

            embedding = image_embedding(variant)

            similarities = torch.matmul(
                reference_embeddings,
                embedding
            )

            values, indices = torch.topk(
                similarities,
                k=min(5, len(brand_names))
            )

            top1_idx = indices[0].item()
            top1_brand = brand_names[top1_idx]
            top1_score = values[0].item()

            top2_score = values[1].item()

            margin = top1_score - top2_score

            # IMPORTANT:
            # Any prediction of the ORIGINAL brand is a false acceptance.
            accepted_as_original = (
                top1_brand.lower() == brand.lower()
            )

            results.append({
                "original_brand": brand,
                "variation": variation_name,
                "predicted_brand": top1_brand,
                "top1": top1_score,
                "top2": top2_score,
                "margin": margin,
                "accepted_as_original": accepted_as_original
            })

    except Exception as e:
        print(f"ERROR: {filename}: {e}")


# ============================================================
# STATISTICS
# ============================================================

total = len(results)

false_accept_original = sum(
    r["accepted_as_original"]
    for r in results
)

any_known_brand = sum(
    r["predicted_brand"].lower() != ""
    for r in results
)

false_accept_rate = (
    false_accept_original / total * 100
    if total else 0
)

top1_scores = [r["top1"] for r in results]
margins = [r["margin"] for r in results]

avg_top1 = sum(top1_scores) / len(top1_scores)
min_top1 = min(top1_scores)

avg_margin = sum(margins) / len(margins)
min_margin = min(margins)

# ============================================================
# PER-BRAND STATISTICS
# ============================================================

brand_stats = {}

for brand in brand_names:
    brand_results = [
        r for r in results
        if r["original_brand"].lower() == brand.lower()
    ]

    if not brand_results:
        continue

    false_accepts = sum(
        r["accepted_as_original"]
        for r in brand_results
    )

    brand_stats[brand] = {
        "total": len(brand_results),
        "false_accepts": false_accepts,
        "rate": false_accepts / len(brand_results) * 100
    }


# ============================================================
# WORST FALSE ACCEPTS
# ============================================================

false_accept_results = [
    r for r in results
    if r["accepted_as_original"]
]

false_accept_results.sort(
    key=lambda x: x["top1"],
    reverse=True
)

# ============================================================
# REPORT
# ============================================================

with open(REPORT_PATH, "w", encoding="utf-8") as f:

    f.write("=" * 78 + "\n")
    f.write("SIGLIP 2 COLOR-CHANGE REJECTION REPORT\n")
    f.write("=" * 78 + "\n\n")

    f.write("Purpose:\n")
    f.write(
        "Test whether deliberately recolored versions of known logos "
        "are incorrectly accepted as their original brand.\n"
    )
    f.write(
        "Under the project rule, ANY deliberate logo color change is "
        "a negative case.\n\n"
    )

    f.write("-" * 78 + "\n")
    f.write("OVERALL RESULTS\n")
    f.write("-" * 78 + "\n")

    f.write(f"Brands tested:              {len(brand_names)}\n")
    f.write(f"Total color-change tests:   {total}\n")
    f.write(
        f"False accepts as original: {false_accept_original}/{total}\n"
    )
    f.write(
        f"False-accept rate:          {false_accept_rate:.2f}%\n"
    )
    f.write(f"Average Top-1 similarity:   {avg_top1:.4f}\n")
    f.write(f"Minimum Top-1 similarity:   {min_top1:.4f}\n")
    f.write(f"Average Top1/Top2 margin:   {avg_margin:.4f}\n")
    f.write(f"Minimum Top1/Top2 margin:   {min_margin:.4f}\n\n")

    f.write("-" * 78 + "\n")
    f.write("FALSE ACCEPTS\n")
    f.write("-" * 78 + "\n")

    if false_accept_results:

        for r in false_accept_results:
            f.write(
                f"{r['original_brand']:<15} | "
                f"{r['variation']:<22} | "
                f"Predicted: {r['predicted_brand']:<15} | "
                f"Top1={r['top1']:.4f} | "
                f"Top2={r['top2']:.4f} | "
                f"Margin={r['margin']:.4f}\n"
            )

    else:
        f.write("No false accepts of the original brand.\n")

    f.write("\n")
    f.write("-" * 78 + "\n")
    f.write("PER-BRAND FALSE-ACCEPT RATE\n")
    f.write("-" * 78 + "\n")

    for brand in brand_names:

        if brand not in brand_stats:
            continue

        s = brand_stats[brand]

        f.write(
            f"{brand:<15} | "
            f"{s['false_accepts']:>2}/{s['total']:<2} | "
            f"{s['rate']:>6.2f}%\n"
        )

    f.write("\n")
    f.write("-" * 78 + "\n")
    f.write("ALL TEST RESULTS\n")
    f.write("-" * 78 + "\n")

    for r in results:

        f.write(
            f"{r['original_brand']:<15} | "
            f"{r['variation']:<22} | "
            f"Predicted: {r['predicted_brand']:<15} | "
            f"Top1={r['top1']:.4f} | "
            f"Top2={r['top2']:.4f} | "
            f"Margin={r['margin']:.4f} | "
            f"OriginalAccept={'YES' if r['accepted_as_original'] else 'NO'}\n"
        )

print("\n" + "=" * 70)
print("COLOR-CHANGE REJECTION TEST COMPLETE")
print("=" * 70)

print(f"Total tests:              {total}")
print(f"False accepts:            {false_accept_original}")
print(f"False-accept rate:        {false_accept_rate:.2f}%")
print(f"Average Top-1 similarity: {avg_top1:.4f}")
print(f"Minimum Top-1 similarity: {min_top1:.4f}")
print(f"Average margin:           {avg_margin:.4f}")
print(f"Minimum margin:           {min_margin:.4f}")

print("\nReport saved to:")
print(REPORT_PATH)
