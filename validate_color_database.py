import os
import sys
import math
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps


# ============================================================
# CONFIG
# ============================================================

LOGO_DIR = "logos"
DATABASE_PATH = "reference_embeddings/color_database.pt"

COLOR_THRESHOLD = 0.80

# Feature weights.
#
# IMPORTANT:
# Our primary goal is:
#   1. Same logo + same colors -> high similarity
#   2. Same logo + changed colors -> reject
#
# Color features therefore receive strong weight.
WEIGHTS = {
    "hsv": 0.35,
    "rgb": 0.20,
    "stats": 0.10,
    "spatial": 0.20,
    "dominant": 0.15,
}


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_frames(path, max_frames=32):
    """
    Load an image.

    For animated GIFs:
        - load all animation frames
        - sample uniformly if there are too many frames

    Returns:
        list[PIL.Image]
    """

    try:
        img = Image.open(path)
    except Exception as e:
        raise ValueError(f"Cannot open image: {path} ({e})")

    frames = []

    # Animated image
    if getattr(img, "is_animated", False):
        total = getattr(img, "n_frames", 1)

        # We don't need hundreds of nearly redundant frames.
        if total <= max_frames:
            indices = list(range(total))
        else:
            indices = np.linspace(
                0,
                total - 1,
                max_frames,
                dtype=int
            ).tolist()

        for idx in indices:
            try:
                img.seek(idx)
                frame = img.convert("RGBA")
                frames.append(frame.copy())
            except Exception:
                continue

    else:
        frames.append(img.convert("RGBA"))

    if not frames:
        raise ValueError(f"No usable frames: {path}")

    return frames


# ============================================================
# RGBA -> RGB + MASK
# ============================================================

def prepare_frame(img):
    """
    Convert RGBA image to RGB while preserving transparency.

    Transparent pixels are ignored by the feature extractor.
    """

    rgba = np.asarray(img.convert("RGBA")).astype(np.float32)

    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3] / 255.0

    valid = alpha > 0.05

    if not np.any(valid):
        return None, None

    # Composite partially transparent pixels onto white.
    rgb = rgb * alpha[:, :, None] + 255.0 * (1.0 - alpha[:, :, None])

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    return rgb, valid


# ============================================================
# HSV CONVERSION
# ============================================================

def rgb_to_hsv_np(rgb):
    """
    RGB uint8 -> HSV in range:
        H: 0..360
        S: 0..1
        V: 0..1
    """

    rgb = rgb.astype(np.float32) / 255.0

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    mx = np.maximum.reduce([r, g, b])
    mn = np.minimum.reduce([r, g, b])
    diff = mx - mn

    h = np.zeros_like(mx)

    mask = diff != 0

    rmask = mask & (mx == r)
    gmask = mask & (mx == g)
    bmask = mask & (mx == b)

    h[rmask] = (
        60.0 * ((g[rmask] - b[rmask]) / diff[rmask])
    ) % 360.0

    h[gmask] = (
        60.0 * ((b[gmask] - r[gmask]) / diff[gmask]) + 120.0
    )

    h[bmask] = (
        60.0 * ((r[bmask] - g[bmask]) / diff[bmask]) + 240.0
    )

    s = np.zeros_like(mx)

    nonzero = mx != 0
    s[nonzero] = diff[nonzero] / mx[nonzero]

    v = mx

    return h, s, v


# ============================================================
# HSV HISTOGRAM
# ============================================================

def calculate_hsv_feature(rgb, valid):
    """
    52 dimensions:

        Hue        = 36
        Saturation = 8
        Value      = 8
    """

    pixels = rgb[valid]

    if len(pixels) == 0:
        return None

    h, s, v = rgb_to_hsv_np(pixels)

    # -------------------------
    # Hue
    # -------------------------

    hue_hist, _ = np.histogram(
        h,
        bins=36,
        range=(0.0, 360.0)
    )

    hue_hist = hue_hist.astype(np.float32)

    if hue_hist.sum() > 0:
        hue_hist /= hue_hist.sum()

    # -------------------------
    # Saturation
    # -------------------------

    sat_hist, _ = np.histogram(
        s,
        bins=8,
        range=(0.0, 1.0)
    )

    sat_hist = sat_hist.astype(np.float32)

    if sat_hist.sum() > 0:
        sat_hist /= sat_hist.sum()

    # -------------------------
    # Value
    # -------------------------

    val_hist, _ = np.histogram(
        v,
        bins=8,
        range=(0.0, 1.0)
    )

    val_hist = val_hist.astype(np.float32)

    if val_hist.sum() > 0:
        val_hist /= val_hist.sum()

    return np.concatenate([
        hue_hist,
        sat_hist,
        val_hist,
    ])


# ============================================================
# RGB HISTOGRAM
# ============================================================

def calculate_rgb_feature(rgb, valid):
    """
    24 dimensions:
        R = 8
        G = 8
        B = 8
    """

    pixels = rgb[valid].astype(np.float32) / 255.0

    result = []

    for c in range(3):
        hist, _ = np.histogram(
            pixels[:, c],
            bins=8,
            range=(0.0, 1.0)
        )

        hist = hist.astype(np.float32)

        if hist.sum() > 0:
            hist /= hist.sum()

        result.append(hist)

    return np.concatenate(result)


# ============================================================
# STATISTICS
# ============================================================

def calculate_stats(rgb, valid):
    """
    18 dimensions:

    For each RGB channel:
        mean
        std
        min
        max
        median
        percentile spread
    """

    pixels = rgb[valid].astype(np.float32) / 255.0

    features = []

    for c in range(3):
        x = pixels[:, c]

        mean = np.mean(x)
        std = np.std(x)
        minimum = np.min(x)
        maximum = np.max(x)
        median = np.median(x)

        p25 = np.percentile(x, 25)
        p75 = np.percentile(x, 75)

        # Keep the first six statistics to match 18 dimensions.
        features.extend([
            mean,
            std,
            minimum,
            maximum,
            median,
            p75 - p25,
        ])

    return np.asarray(features, dtype=np.float32)


# ============================================================
# SPATIAL COLOR FEATURE
# ============================================================

def calculate_spatial_feature(rgb, valid):
    """
    Spatial color distribution.

    112 dimensions:

        4 x 7 spatial regions
        x 4 values

    Values per region:
        mean R
        mean G
        mean B
        valid-pixel ratio
    """

    h, w = valid.shape

    result = []

    rows = 4
    cols = 7

    for r in range(rows):
        y0 = int(r * h / rows)
        y1 = int((r + 1) * h / rows)

        for c in range(cols):
            x0 = int(c * w / cols)
            x1 = int((c + 1) * w / cols)

            region_rgb = rgb[y0:y1, x0:x1]
            region_valid = valid[y0:y1, x0:x1]

            count = int(region_valid.sum())
            total = region_valid.size

            if count == 0:
                result.extend([
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ])
            else:
                pixels = region_rgb[region_valid].astype(
                    np.float32
                ) / 255.0

                result.extend([
                    float(np.mean(pixels[:, 0])),
                    float(np.mean(pixels[:, 1])),
                    float(np.mean(pixels[:, 2])),
                    float(count / max(total, 1)),
                ])

    return np.asarray(result, dtype=np.float32)


# ============================================================
# DOMINANT COLORS
# ============================================================

def calculate_dominant_feature(rgb, valid, k=8):
    """
    32 dimensions.

    For each of 8 dominant colors:

        R
        G
        B
        percentage
    """

    pixels = rgb[valid].astype(np.float32) / 255.0

    if len(pixels) == 0:
        return np.zeros(32, dtype=np.float32)

    # Quantize colors.
    quantized = np.floor(pixels * 8.0).astype(np.int32)
    quantized = np.clip(quantized, 0, 7)

    keys = (
        quantized[:, 0] * 64
        + quantized[:, 1] * 8
        + quantized[:, 2]
    )

    unique, counts = np.unique(
        keys,
        return_counts=True
    )

    order = np.argsort(counts)[::-1]

    result = []

    total = len(pixels)

    for i in range(k):

        if i < len(order):
            idx = order[i]
            key = int(unique[idx])
            count = int(counts[idx])

            r = key // 64
            g = (key % 64) // 8
            b = key % 8

            # Center of quantization bucket.
            rr = (r + 0.5) / 8.0
            gg = (g + 0.5) / 8.0
            bb = (b + 0.5) / 8.0

            percentage = count / total

            result.extend([
                rr,
                gg,
                bb,
                percentage,
            ])

        else:
            result.extend([
                0.0,
                0.0,
                0.0,
                0.0,
            ])

    return np.asarray(result, dtype=np.float32)


# ============================================================
# FRAME FEATURE
# ============================================================

def calculate_frame_features(img):
    rgb, valid = prepare_frame(img)

    if rgb is None or valid is None:
        return None

    hsv = calculate_hsv_feature(rgb, valid)

    if hsv is None:
        return None

    rgb_feature = calculate_rgb_feature(rgb, valid)

    stats = calculate_stats(rgb, valid)

    spatial = calculate_spatial_feature(rgb, valid)

    dominant = calculate_dominant_feature(rgb, valid)

    feature = np.concatenate([
        hsv,
        rgb_feature,
        stats,
        spatial,
        dominant,
    ]).astype(np.float32)

    if len(feature) != 238:
        raise ValueError(
            f"Feature size mismatch: {len(feature)} "
            f"(expected 238)"
        )

    return {
        "feature": feature,
        "hsv": hsv,
        "rgb": rgb_feature,
        "stats": stats,
        "spatial": spatial,
        "dominant": dominant,
    }


# ============================================================
# MULTI-FRAME FEATURE
# ============================================================

def calculate_color_features(path):
    """
    For animated GIFs:
        calculate every sampled frame,
        then average the feature groups.
    """

    frames = load_image_frames(path)

    features = []

    for frame in frames:
        result = calculate_frame_features(frame)

        if result is not None:
            features.append(result)

    if not features:
        raise ValueError(
            f"No valid pixels: {path}"
        )

    output = {}

    for key in [
        "feature",
        "hsv",
        "rgb",
        "stats",
        "spatial",
        "dominant",
    ]:
        output[key] = np.mean(
            np.stack([
                x[key] for x in features
            ]),
            axis=0
        ).astype(np.float32)

    return output


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)

    if len(a) != len(b):
        raise ValueError(
            f"Feature length mismatch: "
            f"{len(a)} vs {len(b)}"
        )

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if na < 1e-12 or nb < 1e-12:
        return 0.0

    return float(
        np.dot(a, b) / (na * nb)
    )


# ============================================================
# DATABASE ITEM EXTRACTION
# ============================================================

def get_database_features(item):
    """
    Supports the new database format.

    Example:

        {
            "filename": "...",
            "feature": Tensor[238],
            "hsv": Tensor[52],
            "rgb": Tensor[24],
            "stats": Tensor[18],
            "spatial": Tensor[112],
            "dominant": Tensor[32]
        }
    """

    if not isinstance(item, dict):
        raise ValueError(
            f"Database item is not a dict: {type(item)}"
        )

    required = [
        "feature",
        "hsv",
        "rgb",
        "stats",
        "spatial",
        "dominant",
    ]

    for key in required:
        if key not in item:
            raise ValueError(
                f"Missing database feature: {key}"
            )

    return {
        key: item[key].detach().cpu().numpy()
        if torch.is_tensor(item[key])
        else np.asarray(item[key], dtype=np.float32)
        for key in required
    }


# ============================================================
# WEIGHTED COLOR SIMILARITY
# ============================================================

def color_similarity(query, database):
    """
    Compare each feature group separately.

    This is important because a single 238-D cosine score
    can hide which part caused a match.
    """

    q = query

    d = get_database_features(database)

    scores = {}

    for key in WEIGHTS:
        scores[key] = cosine_similarity(
            q[key],
            d[key]
        )

    weighted = sum(
        scores[key] * WEIGHTS[key]
        for key in WEIGHTS
    )

    return float(weighted), scores


# ============================================================
# BEST MATCH
# ============================================================

def find_best_match(query, database, names):
    results = []

    for name in names:

        score, parts = color_similarity(
            query,
            database[name]
        )

        results.append({
            "name": name,
            "score": score,
            "parts": parts,
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results


# ============================================================
# TEST ONE LOGO
# ============================================================

def test_logo(path, database, names):

    query = calculate_color_features(path)

    matches = find_best_match(
        query,
        database,
        names
    )

    top1 = matches[0]
    top2 = matches[1] if len(matches) > 1 else None

    margin = (
        top1["score"] - top2["score"]
        if top2
        else 0.0
    )

    return {
        "top1": top1,
        "top2": top2,
        "margin": margin,
        "query": query,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("COLOR DATABASE VALIDATION")
    print("=" * 72)

    # --------------------------------------------------------
    # Load database
    # --------------------------------------------------------

    if not os.path.exists(DATABASE_PATH):
        print(
            f"ERROR: Database not found:\n"
            f"{DATABASE_PATH}"
        )
        sys.exit(1)

    db = torch.load(
        DATABASE_PATH,
        map_location="cpu",
        weights_only=False
    )

    names = db["names"]
    database = db["database"]

    print("Loaded color database")
    print(f"Brands: {len(names)}")

    print("\nDatabase structure:")
    print("Version:", db.get("version"))
    print("Feature type:", db.get("feature_type"))
    print("Feature size:", db.get("feature_size"))

    # --------------------------------------------------------
    # Check first entry
    # --------------------------------------------------------

    first_name = names[0]
    first_item = database[first_name]

    print("\nFeature groups:")

    for key in [
        "feature",
        "hsv",
        "rgb",
        "stats",
        "spatial",
        "dominant",
    ]:
        value = first_item[key]

        if torch.is_tensor(value):
            print(
                f"  {key:<10} {tuple(value.shape)}"
            )

    print("\nWeights:")
    for key, value in WEIGHTS.items():
        print(
            f"  {key:<10} {value:.2f}"
        )

    # --------------------------------------------------------
    # Test original logos
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("TEST 1: ORIGINAL LOGOS")
    print("=" * 72)

    total = 0
    correct = 0
    errors = 0

    wrong_predictions = []

    for index, name in enumerate(names, 1):

        item = database[name]
        filename = item["filename"]

        path = os.path.join(
            LOGO_DIR,
            filename
        )

        print(
            f"{name:<15}",
            end=""
        )

        if not os.path.exists(path):
            print("FILE NOT FOUND")
            errors += 1
            continue

        try:
            result = test_logo(
                path,
                database,
                names
            )

            top1 = result["top1"]
            top2 = result["top2"]
            margin = result["margin"]

            predicted = top1["name"]
            score = top1["score"]

            total += 1

            if predicted == name:
                correct += 1
                status = "OK"
            else:
                status = "WRONG"

                wrong_predictions.append({
                    "actual": name,
                    "predicted": predicted,
                    "score": score,
                    "margin": margin,
                })

            print(
                f"-> {predicted:<15} "
                f"score={score:.4f} "
                f"margin={margin:.4f} "
                f"[{status}]"
            )

        except Exception as e:
            errors += 1

            print(
                f"-> ERROR: {e}"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    print(
        f"Total tested:       {total}"
    )

    print(
        f"Correct Top-1:      {correct}/{total}"
    )

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print(
        f"Top-1 accuracy:     {accuracy:.2f}%"
    )

    print(
        f"Errors:             {errors}"
    )

    print(
        f"Color threshold:    {COLOR_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Wrong predictions
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("WRONG PREDICTIONS")
    print("=" * 72)

    if not wrong_predictions:
        print("None")

    else:
        for x in wrong_predictions:
            print(
                f"{x['actual']:<15} -> "
                f"{x['predicted']:<15} "
                f"score={x['score']:.4f} "
                f"margin={x['margin']:.4f}"
            )

    # --------------------------------------------------------
    # Detailed feature analysis
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("FEATURE ANALYSIS")
    print("=" * 72)

    print(
        "Feature weights currently used:"
    )

    for key, weight in WEIGHTS.items():
        print(
            f"  {key:<10}: {weight:.2f}"
        )

    print(
        "\nThis test validates the color database only."
    )

    print(
        "Next step is to combine:"
    )

    print(
        "  SigLIP shape/semantic similarity"
    )

    print(
        "  + color consistency"
    )

    print(
        "  + OCR text similarity"
    )

    print(
        "  + explicit color-change rejection"
    )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)


if __name__ == "__main__":
    main()

