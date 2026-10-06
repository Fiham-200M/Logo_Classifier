import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from pathlib import Path

MODEL_NAME = "google/siglip2-base-patch16-224"
IMAGE_DIR = Path(__file__).resolve().parent / "logos"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("SigLIP 2 Crop / Padding Normalization Test")
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


def normalize_logo(path, mode):
    image = Image.open(path).convert("RGBA")

    # Composite transparency onto white first.
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    background.alpha_composite(image)
    image = background.convert("RGB")

    if mode == "original":
        return image

    # Find bounding box around non-white pixels.
    pixels = image.load()
    xs = []
    ys = []

    for y in range(image.height):
        for x in range(image.width):
            r, g, b = pixels[x, y]

            # Anything sufficiently different from white
            # is treated as part of the logo.
            if not (r > 245 and g > 245 and b > 245):
                xs.append(x)
                ys.append(y)

    if not xs:
        return image

    left = min(xs)
    top = min(ys)
    right = max(xs) + 1
    bottom = max(ys) + 1

    cropped = image.crop((left, top, right, bottom))

    if mode == "tight":
        return cropped

    if mode == "padded":
        w, h = cropped.size
        longest = max(w, h)

        padding = int(longest * 0.20)
        canvas_size = longest + padding * 2

        canvas = Image.new(
            "RGB",
            (canvas_size, canvas_size),
            (255, 255, 255)
        )

        x = (canvas_size - w) // 2
        y = (canvas_size - h) // 2

        canvas.paste(cropped, (x, y))

        return canvas

    raise ValueError(f"Unknown mode: {mode}")


def embedding(image):
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

    return F.normalize(vector.float(), p=2, dim=-1)


files = {
    "normal": IMAGE_DIR / "our_logo1.png",
    "dark": IMAGE_DIR / "our_logo1_dark.png",
    "small": IMAGE_DIR / "our_logo1_small.png",
}

modes = [
    "original",
    "tight",
    "padded",
]

embeddings = {}

print("\nGenerating embeddings...\n")

for name, path in files.items():

    for mode in modes:

        key = f"{name}_{mode}"

        print(f"Processing: {key}")

        image = normalize_logo(path, mode)
        embeddings[key] = embedding(image)


print("\n")
print("=" * 70)
print("NORMALIZATION RESULTS")
print("=" * 70)

pairs = [
    ("normal_original", "dark_original"),
    ("normal_original", "small_original"),

    ("normal_tight", "dark_tight"),
    ("normal_tight", "small_tight"),

    ("normal_padded", "dark_padded"),
    ("normal_padded", "small_padded"),

    ("normal_original", "small_tight"),
    ("normal_original", "small_padded"),

    ("normal_tight", "small_original"),
    ("normal_padded", "small_original"),
]

for a, b in pairs:

    score = torch.sum(
        embeddings[a] * embeddings[b]
    ).item()

    print(f"{a:22} vs {b:22} = {score:.4f}")


print("\n")
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
