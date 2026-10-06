import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import torch.nn.functional as F
from pathlib import Path

MODEL_NAME = "google/siglip2-base-patch16-224"
BASE_DIR = Path(__file__).resolve().parent / "a200m_eval"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("A200M SigLIP 2 Calibration")
print("=" * 70)
print(f"Device: {DEVICE}")

if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("\nLoading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading SigLIP 2 model...")
model = AutoModel.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16
).to(DEVICE)

model.eval()
print("Model loaded successfully!")

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

    return F.normalize(
        embedding,
        p=2,
        dim=-1
    )

# ------------------------------------------------------------
# Load images
# ------------------------------------------------------------

categories = {
    "SAME BRAND": sorted((BASE_DIR / "same_brand").glob("*.png")),
    "RELATED / LOOKALIKE": sorted((BASE_DIR / "related").glob("*.png")),
    "UNRELATED": sorted((BASE_DIR / "unrelated").glob("*.png")),
}

print("\nImages:")

for category, files in categories.items():
    print(f"\n{category}:")
    for f in files:
        print(f"  {f.name}")

# ------------------------------------------------------------
# Generate embeddings
# ------------------------------------------------------------

all_files = [
    f
    for files in categories.values()
    for f in files
]

embeddings = {}

print("\nGenerating embeddings...\n")

for image_path in all_files:
    print(f"Processing: {image_path.name}")
    embeddings[image_path.name] = get_embedding(image_path)

# ------------------------------------------------------------
# Reference logos
# ------------------------------------------------------------

references = [
    f.name
    for f in categories["SAME BRAND"]
]

# ------------------------------------------------------------
# Compare every image against every official reference
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("IMAGE vs A200M REFERENCE LOGOS")
print("=" * 70)

results = []

for category, files in categories.items():

    for image_path in files:

        # Don't compare a reference against itself
        if category == "SAME BRAND":
            candidate_name = image_path.name
        else:
            candidate_name = image_path.name

        scores = []

        for reference_name in references:

            if candidate_name == reference_name:
                continue

            similarity = torch.sum(
                embeddings[candidate_name] *
                embeddings[reference_name]
            ).item()

            scores.append(
                (reference_name, similarity)
            )

        if not scores:
            continue

        best_reference, best_score = max(
            scores,
            key=lambda x: x[1]
        )

        print(
            f"{category:20} | "
            f"{candidate_name:25} | "
            f"best={best_score:.4f} | "
            f"reference={best_reference}"
        )

        for reference_name, score in scores:
            print(
                f"{'':20} | "
                f"  {reference_name:23} = {score:.4f}"
            )

        results.append(
            {
                "category": category,
                "image": candidate_name,
                "best_score": best_score,
                "best_reference": best_reference,
                "all_scores": scores,
            }
        )

# ------------------------------------------------------------
# Category statistics
# ------------------------------------------------------------

print("\n")
print("=" * 70)
print("CATEGORY SCORE RANGES")
print("=" * 70)

for category in categories:

    category_results = [
        r["best_score"]
        for r in results
        if r["category"] == category
    ]

    if not category_results:
        continue

    print(
        f"{category:20} | "
        f"min={min(category_results):.4f} | "
        f"max={max(category_results):.4f} | "
        f"avg={sum(category_results)/len(category_results):.4f}"
    )

print("\n")
print("=" * 70)
print("CALIBRATION COMPLETE")
print("=" * 70)
print("These are raw similarity measurements only.")
print("No authenticity threshold or final classification is applied.")
