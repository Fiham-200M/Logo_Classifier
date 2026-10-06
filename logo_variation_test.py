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
print("SigLIP 2 - Same Logo Variation Test")
print("=" * 60)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# Load model
# ============================================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print("Loading SigLIP 2 model...")

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
)

model = model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# ============================================================
# Images to test
# ============================================================

images = [
    IMAGE_DIR / "our_logo1.png",
    IMAGE_DIR / "our_logo1_small.png",
    IMAGE_DIR / "our_logo1_dark.png",
]


# ============================================================
# Generate embedding
# ============================================================

def get_embedding(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=torch.float16
    )

    with torch.no_grad():

        vision_outputs = model.vision_model(
            pixel_values=pixel_values
        )

    embedding = vision_outputs.pooler_output

    embedding = F.normalize(
        embedding,
        p=2,
        dim=-1
    )

    return embedding


# ============================================================
# Generate embeddings
# ============================================================

embeddings = {}

print("\nGenerating embeddings...\n")

for image_path in images:

    if not image_path.exists():

        print(
            f"WARNING: {image_path.name} not found"
        )

        continue

    print(
        f"Processing: {image_path.name}"
    )

    embeddings[image_path.name] = (
        get_embedding(image_path)
    )


# ============================================================
# Compare every variation
# ============================================================

print("\n")

print("=" * 60)
print("SIMILARITY RESULTS")
print("=" * 60)

print()

image_names = list(embeddings.keys())

for i in range(len(image_names)):

    for j in range(i + 1, len(image_names)):

        name1 = image_names[i]
        name2 = image_names[j]

        similarity = torch.sum(
            embeddings[name1]
            *
            embeddings[name2]
        ).item()

        print(
            f"{name1:25} vs "
            f"{name2:25} = "
            f"{similarity:.4f}"
        )


# ============================================================
# Reference comparison
# ============================================================

reference = "our_logo1.png"

if reference in embeddings:

    print("\n")

    print("=" * 60)
    print("REFERENCE: our_logo1.png")
    print("=" * 60)

    print()

    for name in image_names:

        if name == reference:
            continue

        similarity = torch.sum(
            embeddings[reference]
            *
            embeddings[name]
        ).item()

        print(
            f"our_logo1.png vs "
            f"{name:25} = "
            f"{similarity:.4f}"
        )


# ============================================================
# Complete
# ============================================================

print("\n")

print("=" * 60)
print("VARIATION TEST COMPLETE")
print("=" * 60)
