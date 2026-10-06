import os
import re
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
import easyocr


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "google/siglip2-base-patch16-224"

BRAND_DB_PATH = "reference_embeddings/brand_database.pt"
COLOR_DB_PATH = "reference_embeddings/color_database.pt"
OCR_DB_PATH = "reference_embeddings/ocr_database.pt"

# Change this to the image you want to test
TEST_IMAGE = "test_images/test.png"

TOP_K = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("COMBINED BRAND INFERENCE TEST")
print("=" * 80)

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")

print(f"Device  : {DEVICE}")
print()


# ============================================================
# LOAD DATABASES
# ============================================================

print("=" * 80)
print("LOADING DATABASES")
print("=" * 80)


brand_db = torch.load(
    BRAND_DB_PATH,
    map_location="cpu",
    weights_only=False
)

color_db = torch.load(
    COLOR_DB_PATH,
    map_location="cpu",
    weights_only=False
)

ocr_db = torch.load(
    OCR_DB_PATH,
    map_location="cpu",
    weights_only=False
)


brand_names = brand_db["brand_names"]

brand_embeddings = brand_db["embeddings"].float()

color_names = color_db["names"]
color_database = color_db["database"]

ocr_names = ocr_db["names"]
ocr_database = ocr_db["database"]


print(f"Brands       : {len(brand_names)}")
print(f"SigLIP shape : {tuple(brand_embeddings.shape)}")
print(f"Color brands : {len(color_names)}")
print(f"OCR brands   : {len(ocr_names)}")
print()


# ============================================================
# VERIFY BRAND ORDER
# ============================================================

print("=" * 80)
print("DATABASE CONSISTENCY")
print("=" * 80)

if brand_names == color_names == ocr_names:
    print("[PASS] All three databases use identical brand order.")
else:
    print("[ERROR] Brand order mismatch!")

    print("SigLIP first:", brand_names[:5])
    print("Color first :", color_names[:5])
    print("OCR first   :", ocr_names[:5])

    raise RuntimeError("Database brand order mismatch.")

print()


# ============================================================
# LOAD SIGLIP
# ============================================================

print("=" * 80)
print("LOADING SIGLIP")
print("=" * 80)

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("[OK] SigLIP loaded.")
print()


# ============================================================
# LOAD OCR
# ============================================================

print("=" * 80)
print("LOADING EASYOCR")
print("=" * 80)

ocr_reader = easyocr.Reader(
    ["en"],
    gpu=torch.cuda.is_available(),
    verbose=True
)

print("[OK] EasyOCR loaded.")
print()


# ============================================================
# IMAGE
# ============================================================

if not os.path.exists(TEST_IMAGE):
    print("=" * 80)
    print("TEST IMAGE NOT FOUND")
    print("=" * 80)
    print(f"Expected: {TEST_IMAGE}")
    print()
    print("Create the folder and put a test image there:")
    print()
    print("  mkdir test_images")
    print()
    print("Then copy an image into:")
    print()
    print(f"  {TEST_IMAGE}")
    print()

    raise SystemExit(1)


print("=" * 80)
print("TEST IMAGE")
print("=" * 80)

print(f"Image: {TEST_IMAGE}")

image = Image.open(TEST_IMAGE).convert("RGB")

print(f"Size : {image.size}")
print()


# ============================================================
# SIGLIP EMBEDDING
# ============================================================

print("=" * 80)
print("SIGLIP ANALYSIS")
print("=" * 80)

inputs = processor(
    images=image,
    return_tensors="pt"
)

inputs = {
    key: value.to(DEVICE)
    for key, value in inputs.items()
}

with torch.no_grad():

    image_outputs = model.get_image_features(
        **inputs
    )

    # Some transformers versions return a tensor,
    # while others may return a structured object.
    if hasattr(image_outputs, "pooler_output"):
        image_embedding = image_outputs.pooler_output
    elif hasattr(image_outputs, "last_hidden_state"):
        image_embedding = image_outputs.last_hidden_state[:, 0]
    else:
        image_embedding = image_outputs

    image_embedding = image_embedding.float()

    image_embedding = torch.nn.functional.normalize(
        image_embedding,
        p=2,
        dim=-1
    )

    reference_embeddings = torch.nn.functional.normalize(
        brand_embeddings.to(DEVICE),
        p=2,
        dim=-1
    )

    siglip_scores = (
        reference_embeddings @ image_embedding.T
    ).squeeze(1)

    siglip_scores = siglip_scores.detach().cpu().numpy()


siglip_order = np.argsort(siglip_scores)[::-1]


print()
print("Top SigLIP matches:")
print()

for rank, index in enumerate(siglip_order[:TOP_K], 1):

    print(
        f"{rank:2d}. "
        f"{brand_names[index]:20s} "
        f"{siglip_scores[index]:.4f}"
    )

print()


# ============================================================
# COLOR FEATURE EXTRACTION
# ============================================================

def extract_color_features(pil_image):

    img = np.array(pil_image)

    # RGB
    rgb = img.astype(np.float32) / 255.0

    # HSV
    hsv_img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    hsv = hsv_img.astype(np.float32)

    features = []

    # --------------------------------------------------------
    # HSV HISTOGRAM
    # --------------------------------------------------------

    h_hist = cv2.calcHist(
        [hsv_img],
        [0],
        None,
        [24],
        [0, 180]
    ).flatten()

    s_hist = cv2.calcHist(
        [hsv_img],
        [1],
        None,
        [16],
        [0, 256]
    ).flatten()

    v_hist = cv2.calcHist(
        [hsv_img],
        [2],
        None,
        [12],
        [0, 256]
    ).flatten()

    h_hist /= (h_hist.sum() + 1e-8)
    s_hist /= (s_hist.sum() + 1e-8)
    v_hist /= (v_hist.sum() + 1e-8)

    hsv_features = np.concatenate([
        h_hist,
        s_hist,
        v_hist
    ])

    # The existing database uses 52 HSV dimensions.
    features.append(hsv_features)

    # --------------------------------------------------------
    # RGB HISTOGRAM
    # --------------------------------------------------------

    rgb_features = []

    for channel in range(3):

        hist = cv2.calcHist(
            [img],
            [channel],
            None,
            [8],
            [0, 256]
        ).flatten()

        hist /= (hist.sum() + 1e-8)

        rgb_features.extend(hist)

    features.append(
        np.array(rgb_features, dtype=np.float32)
    )

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    stats = []

    for channel in range(3):

        channel_data = rgb[:, :, channel]

        stats.extend([
            np.mean(channel_data),
            np.std(channel_data),
            np.min(channel_data),
            np.max(channel_data),
            np.percentile(channel_data, 25),
            np.percentile(channel_data, 50),
        ])

    features.append(
        np.array(stats, dtype=np.float32)
    )

    # --------------------------------------------------------
    # SPATIAL
    # --------------------------------------------------------

    spatial = []

    height, width = rgb.shape[:2]

    for gy in range(4):

        for gx in range(4):

            y1 = int(gy * height / 4)
            y2 = int((gy + 1) * height / 4)

            x1 = int(gx * width / 4)
            x2 = int((gx + 1) * width / 4)

            cell = rgb[y1:y2, x1:x2]

            spatial.extend([
                cell[:, :, 0].mean(),
                cell[:, :, 1].mean(),
                cell[:, :, 2].mean(),
                cell[:, :, 0].std(),
                cell[:, :, 1].std(),
                cell[:, :, 2].std(),
                cell.mean(),
            ])

    features.append(
        np.array(spatial, dtype=np.float32)
    )

    # --------------------------------------------------------
    # DOMINANT COLORS
    # --------------------------------------------------------

    pixels = rgb.reshape(-1, 3).astype(np.float32)

    # Downsample if image is large
    if len(pixels) > 10000:

        indices = np.linspace(
            0,
            len(pixels) - 1,
            10000
        ).astype(int)

        pixels = pixels[indices]

    criteria = (
        cv2.TERM_CRITERIA_EPS +
        cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.2
    )

    k = 8

    try:

        _, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            5,
            cv2.KMEANS_PP_CENTERS
        )

        counts = np.bincount(
            labels.flatten(),
            minlength=k
        )

        order = np.argsort(counts)[::-1]

        dominant = []

        for idx in order:

            weight = counts[idx] / len(labels)

            dominant.extend([
                centers[idx][0],
                centers[idx][1],
                centers[idx][2],
                weight
            ])

        dominant = np.array(
            dominant,
            dtype=np.float32
        )

    except Exception:

        dominant = np.zeros(
            32,
            dtype=np.float32
        )

    features.append(dominant)

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    feature = np.concatenate(features)

    return torch.tensor(
        feature,
        dtype=torch.float32
    )


# ============================================================
# COLOR ANALYSIS
# ============================================================

print("=" * 80)
print("COLOR ANALYSIS")
print("=" * 80)

query_color = extract_color_features(image)

print(f"Query feature size: {query_color.shape[0]}")
print(f"Database feature size: {color_db['feature_size']}")

if query_color.shape[0] != color_db["feature_size"]:
    raise RuntimeError(
        f"Color feature mismatch: "
        f"query={query_color.shape[0]} "
        f"database={color_db['feature_size']}"
    )


# ------------------------------------------------------------
# Normalize feature vectors
# ------------------------------------------------------------

query_color = torch.nn.functional.normalize(
    query_color.unsqueeze(0),
    p=2,
    dim=1
).squeeze(0)


color_scores = []

for name in color_names:

    reference = color_database[name]["feature"].float()

    reference = torch.nn.functional.normalize(
        reference,
        p=2,
        dim=0
    )

    score = torch.dot(
        query_color,
        reference
    ).item()

    color_scores.append(score)


color_scores = np.array(color_scores)

color_order = np.argsort(color_scores)[::-1]


print()
print("Top Color matches:")
print()

for rank, index in enumerate(color_order[:TOP_K], 1):

    print(
        f"{rank:2d}. "
        f"{color_names[index]:20s} "
        f"{color_scores[index]:.4f}"
    )

print()


# ============================================================
# OCR
# ============================================================

print("=" * 80)
print("OCR ANALYSIS")
print("=" * 80)

ocr_results = ocr_reader.readtext(
    np.array(image),
    detail=1,
    paragraph=False
)

query_texts = []

print()

if not ocr_results:

    print("No OCR text detected.")

else:

    print("Detected text:")

    for detection in ocr_results:

        bbox, text, confidence = detection

        print(
            f"  {text!r} "
            f"(confidence={confidence:.4f})"
        )

        # Only use reasonably confident text
        if confidence >= 0.25:

            query_texts.append(
                text
            )

print()


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):

    if isinstance(text, dict):
        text = text.get("text", "")
    elif not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()

    # Keep letters/numbers.
    text = re.sub(
        r"[^a-z0-9]+",
        "",
        text
    )

    return text


def levenshtein_similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    # Dynamic programming
    previous = list(range(len(b) + 1))

    for i, ca in enumerate(a, 1):

        current = [i]

        for j, cb in enumerate(b, 1):

            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (
                0 if ca == cb else 1
            )

            current.append(
                min(
                    insert_cost,
                    delete_cost,
                    replace_cost
                )
            )

        previous = current

    distance = previous[-1]

    max_len = max(
        len(a),
        len(b)
    )

    return 1.0 - (
        distance / max_len
    )


# ============================================================
# OCR SCORE
# ============================================================

ocr_scores = []

for name in ocr_names:

    entry = ocr_database[name]

    reference_texts = entry.get(
        "texts",
        []
    )

    best_score = 0.0

    for query_text in query_texts:

        for ref_item in reference_texts:

            ref_text = ref_item.get("text", "") if isinstance(ref_item, dict) else str(ref_item)

            score = levenshtein_similarity(
                query_text,
                ref_text
            )

            best_score = max(
                best_score,
                score
            )

    ocr_scores.append(best_score)


ocr_scores = np.array(
    ocr_scores,
    dtype=np.float32
)

ocr_order = np.argsort(
    ocr_scores
)[::-1]


print("Top OCR matches:")
print()

if query_texts:

    for rank, index in enumerate(
        ocr_order[:TOP_K],
        1
    ):

        print(
            f"{rank:2d}. "
            f"{ocr_names[index]:20s} "
            f"{ocr_scores[index]:.4f}"
        )

else:

    print("No usable OCR text.")
    print()


# ============================================================
# SCORE NORMALIZATION
# ============================================================

def minmax_normalize(scores):

    minimum = scores.min()
    maximum = scores.max()

    if maximum - minimum < 1e-8:

        return np.ones_like(
            scores,
            dtype=np.float32
        )

    return (
        (scores - minimum)
        /
        (maximum - minimum)
    )


siglip_norm = minmax_normalize(
    siglip_scores
)

color_norm = minmax_normalize(
    color_scores
)

ocr_norm = minmax_normalize(
    ocr_scores
)


# ============================================================
# COMBINED SCORE
# ============================================================

# IMPORTANT:
# These are INITIAL TEST WEIGHTS ONLY.
# We will tune them after evaluating real positive
# and negative examples.

SIGLIP_WEIGHT = 0.60
COLOR_WEIGHT = 0.25
OCR_WEIGHT = 0.15


combined_scores = (
    SIGLIP_WEIGHT * siglip_norm
    +
    COLOR_WEIGHT * color_norm
    +
    OCR_WEIGHT * ocr_norm
)


combined_order = np.argsort(
    combined_scores
)[::-1]


# ============================================================
# FINAL RESULTS
# ============================================================

print("=" * 80)
print("COMBINED RESULTS")
print("=" * 80)

print()
print(
    f"Weights: "
    f"SigLIP={SIGLIP_WEIGHT:.2f}, "
    f"Color={COLOR_WEIGHT:.2f}, "
    f"OCR={OCR_WEIGHT:.2f}"
)

print()

print(
    f"{'Rank':<5}"
    f"{'Brand':<20}"
    f"{'SigLIP':>10}"
    f"{'Color':>10}"
    f"{'OCR':>10}"
    f"{'Combined':>12}"
)

print("-" * 80)

for rank, index in enumerate(
    combined_order[:TOP_K],
    1
):

    print(
        f"{rank:<5}"
        f"{brand_names[index]:<20}"
        f"{siglip_scores[index]:>10.4f}"
        f"{color_scores[index]:>10.4f}"
        f"{ocr_scores[index]:>10.4f}"
        f"{combined_scores[index]:>12.4f}"
    )

print()


# ============================================================
# TOP MATCH DETAILS
# ============================================================

best_index = combined_order[0]

print("=" * 80)
print("TOP COMBINED MATCH")
print("=" * 80)

print(f"Brand    : {brand_names[best_index]}")
print(f"SigLIP   : {siglip_scores[best_index]:.4f}")
print(f"Color    : {color_scores[best_index]:.4f}")
print(f"OCR      : {ocr_scores[best_index]:.4f}")
print(f"Combined : {combined_scores[best_index]:.4f}")

print()


# ============================================================
# GPU MEMORY
# ============================================================

if torch.cuda.is_available():

    print("=" * 80)
    print("GPU MEMORY")
    print("=" * 80)

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print(f"Allocated : {allocated:.2f} GB")
    print(f"Reserved  : {reserved:.2f} GB")

print()

print("=" * 80)
print("COMBINED INFERENCE TEST COMPLETE")
print("=" * 80)