import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from pathlib import Path

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_DIR = Path(__file__).resolve().parent / "logos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SigLIP 2 Square Canvas Normalization Test")
print("=" * 70)

print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to(DEVICE)

model.eval()


def square_normalize(path, size=224):
    image = Image.open(path).convert("RGB")

    # Preserve the complete original aspect ratio.
    image.thumbnail(
        (size, size),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "RGB",
        (size, size),
        (255, 255, 255)
    )

    x = (size - image.width) // 2
    y = (size - image.height) // 2

    canvas.paste(image, (x, y))

    return canvas


def get_embedding(image):
    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    pixel_values = inputs["pixel_values"].to(
        DEVICE,
        dtype=torch.float16
    )

    with torch.no_grad():
        outputs = model.vision_model(
            pixel_values=pixel_values
        )

    vector = outputs.pooler_output

    # Convert to float32 before normalization/dot product
    # to avoid small fp16 numerical errors.
    vector = vector.float()

    return F.normalize(vector, p=2, dim=-1)


files = {
    "normal": IMAGE_DIR / "our_logo1.png",
    "dark": IMAGE_DIR / "our_logo1_dark.png",
    "small": IMAGE_DIR / "our_logo1_small.png",
}

embeddings = {}

print("\nGenerating normalized embeddings...\n")

for name, path in files.items():
    print(f"Processing: {name}")

    image = square_normalize(path)

    embeddings[name] = get_embedding(image)


print("\n")
print("=" * 70)
print("SQUARE NORMALIZATION RESULTS")
print("=" * 70)

pairs = [
    ("normal", "dark"),
    ("normal", "small"),
    ("dark", "small"),
]

for a, b in pairs:

    score = torch.sum(
        embeddings[a] * embeddings[b]
    ).item()

    print(f"{a:10} vs {b:10} = {score:.4f}")


print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
