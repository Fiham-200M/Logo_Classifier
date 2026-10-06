import os
import torch


DB_DIR = "reference_embeddings"

FILES = [
    "brand_database.pt",
    "color_database.pt",
    "ocr_database.pt",
]


print("=" * 70)
print("COMBINED BRAND DATABASE TEST")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print()


databases = {}

for filename in FILES:

    path = os.path.join(DB_DIR, filename)

    print("-" * 70)
    print(f"Loading: {filename}")
    print("-" * 70)

    if not os.path.exists(path):
        print("[FAIL] File does not exist")
        continue

    db = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    databases[filename] = db

    print("[OK] Loaded")

    print("Keys:")
    for key in db.keys():
        print(f"  - {key}")

    print()


# ============================================================
# SIGLIP
# ============================================================

print("=" * 70)
print("SIGLIP DATABASE")
print("=" * 70)

siglip = databases["brand_database.pt"]

print("Model:", siglip["model_name"])
print("Embedding dimension:", siglip["embedding_dimension"])
print("Number of brands:", siglip["num_brands"])

print("Embedding shape:", tuple(siglip["embeddings"].shape))
print("Embedding dtype:", siglip["embeddings"].dtype)

print("Brand count:", len(siglip["brand_names"]))
print("File count:", len(siglip["file_names"]))

siglip_brands = set(siglip["brand_names"])

print()


# ============================================================
# COLOR
# ============================================================

print("=" * 70)
print("COLOR DATABASE")
print("=" * 70)

color = databases["color_database.pt"]

print("Version:", color["version"])
print("Feature type:", color["feature_type"])
print("Feature size:", color["feature_size"])

color_brands = set(color["names"])

print("Brand count:", len(color["names"]))

print()


# ============================================================
# OCR
# ============================================================

print("=" * 70)
print("OCR DATABASE")
print("=" * 70)

ocr = databases["ocr_database.pt"]

print("Version:", ocr["version"])
print("Feature type:", ocr["feature_type"])
print("Languages:", ocr["languages"])
print("Confidence threshold:", ocr["confidence_threshold"])

ocr_brands = set(ocr["names"])

print("Brand count:", len(ocr["names"]))

print()


# ============================================================
# BRAND CONSISTENCY
# ============================================================

print("=" * 70)
print("BRAND CONSISTENCY CHECK")
print("=" * 70)

print("SigLIP brands:", len(siglip_brands))
print("Color brands :", len(color_brands))
print("OCR brands   :", len(ocr_brands))

common = siglip_brands & color_brands & ocr_brands

print("Common brands:", len(common))

missing_from_color = siglip_brands - color_brands
missing_from_ocr = siglip_brands - ocr_brands

print()

if missing_from_color:
    print("[WARNING] Missing from color database:")
    for brand in sorted(missing_from_color):
        print(" ", brand)
else:
    print("[PASS] All SigLIP brands exist in color database")

if missing_from_ocr:
    print("[WARNING] Missing from OCR database:")
    for brand in sorted(missing_from_ocr):
        print(" ", brand)
else:
    print("[PASS] All SigLIP brands exist in OCR database")


# ============================================================
# ORDER CHECK
# ============================================================

print()
print("=" * 70)
print("BRAND ORDER CHECK")
print("=" * 70)

same_order_color = (
    siglip["brand_names"] == color["names"]
)

same_order_ocr = (
    siglip["brand_names"] == ocr["names"]
)

print("SigLIP <-> Color order:", same_order_color)
print("SigLIP <-> OCR order  :", same_order_ocr)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    len(common) == 52
    and not missing_from_color
    and not missing_from_ocr
):
    print("[PASS] All 52 brands exist in all databases.")
else:
    print("[WARNING] Database brand sets are not identical.")

print()
print("=" * 70)
print("COMBINED DATABASE TEST COMPLETE")
print("=" * 70)