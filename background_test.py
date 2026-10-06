import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from pathlib import Path

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_DIR = Path(__file__).resolve().parent / "logos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SigLIP 2 Background Normalization Test")
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

def load_normalized(path, background):
    image = Image.open(path).convert("RGBA")

    bg = Image.new(
        "RGBA",
        image.size,
        background
    )

    bg.alpha_composite(image)

    return bg.convert("RGB")


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

    embedding = outputs.pooler_output

    return F.normalize(
        embedding,
        p=2,
        dim=-1
    )


images = {
    "A200M_normal": IMAGE_DIR / "our_logo1.png",
    "A200M_dark": IMAGE_DIR / "our_logo1_dark.png",
}

backgrounds = {
    "WHITE": (255, 255, 255, 255),
    "BLACK": (0, 0, 0, 255),
}

embeddings = {}

print("\nGenerating embeddings...\n")

for image_name, image_path in images.items():

    for background_name, background in backgrounds.items():

        key = f"{image_name}_{background_name}"

        image = load_normalized(
            image_path,
            background
        )

        print(f"Processing: {key}")

        embeddings[key] = get_embedding(image)


print("\n")
print("=" * 70)
print("BACKGROUND NORMALIZATION RESULTS")
print("=" * 70)

comparisons = [
    (
        "A200M_normal_WHITE",
        "A200M_normal_BLACK"
    ),
    (
        "A200M_dark_WHITE",
        "A200M_dark_BLACK"
    ),
    (
        "A200M_normal_WHITE",
        "A200M_dark_WHITE"
    ),
    (
        "A200M_normal_BLACK",
        "A200M_dark_BLACK"
    ),
]

for name1, name2 in comparisons:

    similarity = torch.sum(
        embeddings[name1] *
        embeddings[name2]
    ).item()

    print(
        f"{name1:25} vs "
        f"{name2:25} = "
        f"{similarity:.4f}"
    )


print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
