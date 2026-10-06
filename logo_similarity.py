import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_DIR = Path(__file__).resolve().parent / "logos"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Header
# ============================================================

print("=" * 60)
print("SigLIP 2 Logo Similarity Test")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load processor
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Load model
# ============================================================

print("Loading SigLIP 2 model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# ============================================================
# Find images
# ============================================================

our_images = sorted(
    IMAGE_DIR.glob("our_logo*.png")
)

competitor_images = sorted(
    IMAGE_DIR.glob("competitor*.png")
)

print("\nImages found:")

print(f"Our logos: {len(our_images)}")

print(
    f"Competitor logos: {len(competitor_images)}"
)

for image in our_images:
    print("  OUR       :", image.name)

for image in competitor_images:
    print("  COMPETITOR:", image.name)


# ============================================================
# Generate image embedding
# ============================================================

def get_embedding(image_path):

    # ------------------------------------------
    # Open image
    # ------------------------------------------

    image = Image.open(image_path).convert("RGB")


    # ------------------------------------------
    # Process image
    # ------------------------------------------

    inputs = processor(
        images=image,
        return_tensors="pt"
    )


    # ------------------------------------------
    # Move pixel values to GPU
    # ------------------------------------------

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=torch.float16
    )


    # ------------------------------------------
    # Use ONLY the vision encoder
    # ------------------------------------------

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )


    # ------------------------------------------
    # Get pooled vision representation
    # ------------------------------------------

    image_embedding = vision_outputs.pooler_output


    # ------------------------------------------
    # Normalize embedding
    # ------------------------------------------

    image_embedding = F.normalize(
        image_embedding,
        p=2,
        dim=-1
    )


    return image_embedding


# ============================================================
# Generate embeddings
# ============================================================

print("\nGenerating embeddings...")

embeddings = {}

all_images = (
    our_images +
    competitor_images
)

for image_path in all_images:

    print(
        f"Processing: {image_path.name}"
    )

    embeddings[image_path.name] = (
        get_embedding(image_path)
    )


# ============================================================
# OUR LOGO vs COMPETITOR
# ============================================================

print("\n")

print("=" * 60)
print("OUR LOGO vs COMPETITOR LOGOS")
print("=" * 60)

print()

for our_image in our_images:

    our_name = our_image.name

    for competitor_image in competitor_images:

        competitor_name = competitor_image.name

        similarity = torch.sum(
            embeddings[our_name]
            *
            embeddings[competitor_name]
        ).item()

        print(
            f"{our_name:18} vs "
            f"{competitor_name:18} = "
            f"{similarity:.4f}"
        )


# ============================================================
# OUR LOGO vs OUR LOGO
# ============================================================

print("\n")

print("=" * 60)
print("OUR LOGO vs OUR LOGO")
print("=" * 60)

print()

for i in range(len(our_images)):

    for j in range(i + 1, len(our_images)):

        name1 = our_images[i].name
        name2 = our_images[j].name

        similarity = torch.sum(
            embeddings[name1]
            *
            embeddings[name2]
        ).item()

        print(
            f"{name1:18} vs "
            f"{name2:18} = "
            f"{similarity:.4f}"
        )


# ============================================================
# COMPETITOR vs COMPETITOR
# ============================================================

print("\n")

print("=" * 60)
print("COMPETITOR vs COMPETITOR")
print("=" * 60)

print()

for i in range(len(competitor_images)):

    for j in range(i + 1, len(competitor_images)):

        name1 = competitor_images[i].name
        name2 = competitor_images[j].name

        similarity = torch.sum(
            embeddings[name1]
            *
            embeddings[name2]
        ).item()

        print(
            f"{name1:18} vs "
            f"{name2:18} = "
            f"{similarity:.4f}"
        )


# ============================================================
# Test complete
# ============================================================

print("\n")

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)
