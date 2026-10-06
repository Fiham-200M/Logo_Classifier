from pathlib import Path
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOGO_DIR = PROJECT_DIR / "logos"
DB_DIR = PROJECT_DIR / "reference_embeddings"

MODEL_NAME = "google/siglip2-base-patch16-224"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATABASE_FILE = DB_DIR / "brand_database.pt"
BRAND_LIST_FILE = DB_DIR / "brand_list.txt"


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SIGLIP 2 - BRAND DATABASE VALIDATION")
print("=" * 70)

print(f"Project directory : {PROJECT_DIR}")
print(f"Logo directory    : {LOGO_DIR}")
print(f"Database          : {DATABASE_FILE}")
print(f"Device            : {DEVICE}")

if torch.cuda.is_available():
    print(f"GPU               : {torch.cuda.get_device_name(0)}")

print("=" * 70)


# ============================================================
# LOAD DATABASE
# ============================================================

print("\nLoading reference database...")

database = torch.load(
    DATABASE_FILE,
    map_location="cpu",
    weights_only=False
)

embeddings = database["embeddings"].float()

with open(BRAND_LIST_FILE, "r", encoding="utf-8") as f:
    brand_lines = [line.strip() for line in f if line.strip()]

brands = []

for line in brand_lines:
    parts = line.split(maxsplit=1)

    if len(parts) == 2:
        brand_name, filename = parts
    else:
        brand_name = parts[0]
        filename = parts[0]

    brands.append({
        "name": brand_name,
        "filename": filename
    })


print(f"Reference embeddings : {embeddings.shape}")
print(f"Reference brands     : {len(brands)}")


# ============================================================
# CHECK DATABASE
# ============================================================

if len(brands) != embeddings.shape[0]:
    raise RuntimeError(
        f"Brand count ({len(brands)}) does not match "
        f"embedding count ({embeddings.shape[0]})"
    )

# Normalize again for safety
embeddings = embeddings / (
    embeddings.norm(dim=1, keepdim=True) + 1e-12
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading SigLIP 2 processor...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

print("Loading SigLIP 2 model...")

model = AutoModel.from_pretrained(MODEL_NAME)
model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# FUNCTION
# ============================================================

@torch.no_grad()
def get_embedding(image_path):

    image = Image.open(image_path)

    # Handle GIF / transparency / unusual formats
    if image.format == "GIF":
        image.seek(0)

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    vision_output = model.vision_model(
        pixel_values=inputs["pixel_values"]
    )

    embedding = vision_output.pooler_output

    embedding = embedding / (
        embedding.norm(dim=-1, keepdim=True) + 1e-12
    )

    return embedding[0].cpu()


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING 52 KNOWN LOGOS")
print("=" * 70)

results = []

top1_correct = 0
top3_correct = 0
top5_correct = 0

all_scores = []


for index, brand in enumerate(brands):

    brand_name = brand["name"]
    filename = brand["filename"]

    image_path = LOGO_DIR / filename

    print(
        f"\n[{index + 1:02d}/{len(brands)}] "
        f"{brand_name} ({filename})"
    )

    try:

        query_embedding = get_embedding(image_path)

        # Cosine similarity because embeddings are normalized
        scores = torch.matmul(
            embeddings,
            query_embedding
        )

        sorted_indices = torch.argsort(
            scores,
            descending=True
        )

        top1 = sorted_indices[0].item()
        top3 = sorted_indices[:3].tolist()
        top5 = sorted_indices[:5].tolist()

        top1_brand = brands[top1]["name"]
        top1_score = scores[top1].item()

        expected_index = index

        is_top1 = top1 == expected_index
        is_top3 = expected_index in top3
        is_top5 = expected_index in top5

        if is_top1:
            top1_correct += 1

        if is_top3:
            top3_correct += 1

        if is_top5:
            top5_correct += 1

        all_scores.append(top1_score)

        status = "OK" if is_top1 else "MISS"

        print(f"  Expected : {brand_name}")
        print(
            f"  Top-1    : {top1_brand} "
            f"(similarity={top1_score:.4f})"
        )
        print(f"  Status   : {status}")

        print("  Top-5:")

        for rank, ref_index in enumerate(top5, 1):

            ref_brand = brands[ref_index]["name"]
            ref_score = scores[ref_index].item()

            marker = ""

            if ref_index == expected_index:
                marker = "  <-- EXPECTED"

            print(
                f"    {rank}. "
                f"{ref_brand:<15} "
                f"{ref_score:.4f}{marker}"
            )

        results.append({
            "brand": brand_name,
            "filename": filename,
            "top1": top1_brand,
            "top1_score": top1_score,
            "top1_correct": is_top1
        })

    except Exception as e:

        print(f"  ERROR: {e}")

        results.append({
            "brand": brand_name,
            "filename": filename,
            "top1": "ERROR",
            "top1_score": 0.0,
            "top1_correct": False
        })


# ============================================================
# SUMMARY
# ============================================================

total = len(brands)

top1_accuracy = top1_correct / total * 100
top3_accuracy = top3_correct / total * 100
top5_accuracy = top5_correct / total * 100

print("\n")
print("=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"Total brands : {total}")

print(
    f"Top-1        : "
    f"{top1_correct}/{total} "
    f"({top1_accuracy:.2f}%)"
)

print(
    f"Top-3        : "
    f"{top3_correct}/{total} "
    f"({top3_accuracy:.2f}%)"
)

print(
    f"Top-5        : "
    f"{top5_correct}/{total} "
    f"({top5_accuracy:.2f}%)"
)

if all_scores:

    average_score = sum(all_scores) / len(all_scores)
    minimum_score = min(all_scores)
    maximum_score = max(all_scores)

    print()
    print(f"Average Top-1 similarity : {average_score:.4f}")
    print(f"Minimum Top-1 similarity : {minimum_score:.4f}")
    print(f"Maximum Top-1 similarity : {maximum_score:.4f}")


# ============================================================
# FAILED / MISCLASSIFIED
# ============================================================

misses = [
    result
    for result in results
    if not result["top1_correct"]
]

print("\n" + "=" * 70)
print("TOP-1 MISCLASSIFICATIONS")
print("=" * 70)

if not misses:

    print("None. All 52 logos matched themselves correctly.")

else:

    for result in misses:

        print(
            f"{result['brand']:<15} "
            f"-> {result['top1']:<15} "
            f"score={result['top1_score']:.4f}"
        )


# ============================================================
# SAVE REPORT
# ============================================================

REPORT_FILE = DB_DIR / "validation_report.txt"

with open(REPORT_FILE, "w", encoding="utf-8") as f:

    f.write("SIGLIP 2 BRAND DATABASE VALIDATION\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Total brands : {total}\n")
    f.write(
        f"Top-1       : {top1_correct}/{total} "
        f"({top1_accuracy:.2f}%)\n"
    )
    f.write(
        f"Top-3       : {top3_correct}/{total} "
        f"({top3_accuracy:.2f}%)\n"
    )
    f.write(
        f"Top-5       : {top5_correct}/{total} "
        f"({top5_accuracy:.2f}%)\n"
    )

    if all_scores:
        f.write(
            f"\nAverage Top-1 similarity : "
            f"{sum(all_scores) / len(all_scores):.4f}\n"
        )

        f.write(
            f"Minimum Top-1 similarity : "
            f"{min(all_scores):.4f}\n"
        )

        f.write(
            f"Maximum Top-1 similarity : "
            f"{max(all_scores):.4f}\n"
        )

    f.write("\n\nMISCLASSIFICATIONS\n")
    f.write("=" * 70 + "\n")

    for result in misses:

        f.write(
            f"{result['brand']} -> "
            f"{result['top1']} "
            f"({result['top1_score']:.4f})\n"
        )

print()
print(f"Validation report saved to:")
print(REPORT_FILE)

print("=" * 70)
