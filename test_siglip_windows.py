import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/siglip2-base-patch16-224"

IMAGE_1 = "logos/a200m.png"
IMAGE_2 = "logos/a200m.png"
IMAGE_3 = "logos/g200m.png"


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SIGLIP WINDOWS GPU TEST")
print("=" * 70)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")

print("Device:", device)
print("Model:", MODEL_ID)
print()


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading SigLIP model...")
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")
print()


# ============================================================
# LOAD IMAGES
# ============================================================

print("Loading images...")

images = [
    Image.open(IMAGE_1).convert("RGB"),
    Image.open(IMAGE_2).convert("RGB"),
    Image.open(IMAGE_3).convert("RGB"),
]

print("Images loaded.")
print()


# ============================================================
# IMAGE EMBEDDINGS
# ============================================================

print("Creating SigLIP image embeddings...")

inputs = processor(
    images=images,
    return_tensors="pt"
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
    if torch.is_tensor(value)
}

with torch.inference_mode():
    image_features = model.get_image_features(**inputs)

# Some Transformers versions return an object/dict-like structure.
if hasattr(image_features, "pooler_output"):
    image_features = image_features.pooler_output

if isinstance(image_features, tuple):
    image_features = image_features[0]

# Normalize embeddings
image_features = image_features / image_features.norm(
    dim=-1,
    keepdim=True
)

print("Embeddings created.")
print("Embedding shape:", tuple(image_features.shape))
print()


# ============================================================
# COSINE SIMILARITY
# ============================================================

similarity = image_features @ image_features.T

print("=" * 70)
print("SIMILARITY MATRIX")
print("=" * 70)

print("Image 1:", IMAGE_1)
print("Image 2:", IMAGE_2)
print("Image 3:", IMAGE_3)
print()

for i in range(3):
    for j in range(3):
        print(
            f"Similarity [{i+1} -> {j+1}]: "
            f"{similarity[i, j].item():.4f}"
        )

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
    print(f"Reserved:  {reserved:.2f} GB")

print()
print("=" * 70)
print("SIGLIP WINDOWS TEST COMPLETE")
print("=" * 70)