import os
import torch
import easyocr


# ============================================================
# CONFIG
# ============================================================

LOGO_DIR = "logos"

# Languages
LANGUAGES = ["en"]

# Use GPU if available
USE_GPU = torch.cuda.is_available()


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("EASYOCR WINDOWS GPU TEST")
print("=" * 70)

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print(f"EasyOCR GPU: {USE_GPU}")
print()


# ============================================================
# LOAD OCR
# ============================================================

print("Loading EasyOCR...")

reader = easyocr.Reader(
    LANGUAGES,
    gpu=USE_GPU,
    verbose=True
)

print()
print("EasyOCR loaded successfully.")
print()


# ============================================================
# FIND TEST IMAGES
# ============================================================

image_files = []

for filename in sorted(os.listdir(LOGO_DIR)):
    path = os.path.join(LOGO_DIR, filename)

    if os.path.isfile(path):
        ext = os.path.splitext(filename)[1].lower()

        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
            image_files.append(path)


print(f"Found {len(image_files)} image files.")
print()


# ============================================================
# TEST OCR
# ============================================================

for image_path in image_files:

    filename = os.path.basename(image_path)

    print("-" * 70)
    print(f"IMAGE: {filename}")
    print("-" * 70)

    try:

        results = reader.readtext(
            image_path,
            detail=1,
            paragraph=False
        )

        if not results:
            print("No text detected.")
            continue

        for detection in results:

            bbox, text, confidence = detection

            print(f"Text       : {text}")
            print(f"Confidence : {confidence:.4f}")

    except Exception as e:

        print(f"[ERROR] {e}")

    print()


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3

    print("=" * 70)
    print("GPU MEMORY")
    print("=" * 70)

    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved : {reserved:.2f} GB")


print()
print("=" * 70)
print("EASYOCR WINDOWS TEST COMPLETE")
print("=" * 70)