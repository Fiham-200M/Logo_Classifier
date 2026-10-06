import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageSequence

LOGO_DIR = "logos"
OUTPUT = "reference_embeddings/color_database.pt"

os.makedirs("reference_embeddings", exist_ok=True)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image_frames(path):
    """
    Load an image as one or multiple RGB numpy frames.

    Animated GIF:
        Load all useful frames.

    Static PNG/JPG/WebP:
        Return one frame.
    """

    ext = os.path.splitext(path)[1].lower()

    try:
        img = Image.open(path)

        # ----------------------------------------------------
        # Animated GIF
        # ----------------------------------------------------
        if ext == ".gif" and getattr(img, "n_frames", 1) > 1:

            frames = []

            for frame in ImageSequence.Iterator(img):

                try:
                    rgba = frame.convert("RGBA")

                    arr = np.array(rgba)

                    rgb = arr[:, :, :3]
                    alpha = arr[:, :, 3]

                    # Keep frame only if it has visible pixels
                    if np.any(alpha > 5):
                        frames.append((rgb, alpha))

                except Exception:
                    continue

            return frames

        # ----------------------------------------------------
        # Static image
        # ----------------------------------------------------
        rgba = img.convert("RGBA")

        arr = np.array(rgba)

        rgb = arr[:, :, :3]
        alpha = arr[:, :, 3]

        return [(rgb, alpha)]

    except Exception as e:

        print(f"  ERROR loading {path}: {e}")

        return []


# ============================================================
# FOREGROUND EXTRACTION
# ============================================================

def get_foreground_pixels(rgb, alpha):
    """
    Extract meaningful logo pixels.

    Priority:
    1. Alpha channel
    2. Remove almost-white background when necessary
    """

    rgb = rgb.astype(np.uint8)
    alpha = alpha.astype(np.uint8)

    # --------------------------------------------------------
    # Transparent image
    # --------------------------------------------------------

    if np.any(alpha < 250):

        mask = alpha > 10

    else:

        # ----------------------------------------------------
        # Opaque image
        # Remove white / near-white background
        # ----------------------------------------------------

        max_rgb = rgb.max(axis=2)
        min_rgb = rgb.min(axis=2)

        # Near-white pixels
        white = (
            (max_rgb > 245) &
            (min_rgb > 245)
        )

        mask = ~white

        # If almost everything was removed,
        # use the whole image.
        if mask.sum() < 50:

            mask = np.ones(
                rgb.shape[:2],
                dtype=bool
            )

    pixels = rgb[mask]

    return pixels, mask


# ============================================================
# HISTOGRAM
# ============================================================

def normalized_hist(values, bins, value_range):
    hist, _ = np.histogram(
        values,
        bins=bins,
        range=value_range
    )

    hist = hist.astype(np.float32)

    total = hist.sum()

    if total > 0:
        hist /= total

    return hist


# ============================================================
# COLOR FEATURES
# ============================================================

def calculate_features(rgb, alpha):
    """
    Create a robust color fingerprint.

    Features:

        HSV histogram
        RGB histogram
        statistics
        spatial color distribution
        dominant colors
    """

    pixels, mask = get_foreground_pixels(rgb, alpha)

    if len(pixels) < 10:

        return None

    # --------------------------------------------------------
    # Resize image for spatial analysis
    # --------------------------------------------------------

    h, w = rgb.shape[:2]

    small = cv2.resize(
        rgb,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    small_alpha = cv2.resize(
        alpha,
        (64, 64),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # HSV
    # --------------------------------------------------------

    pixels_bgr = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2BGR
    )

    pixels_hsv = cv2.cvtColor(
        pixels.reshape(-1, 1, 3),
        cv2.COLOR_RGB2HSV
    ).reshape(-1, 3)

    hsv = pixels_hsv

    hue = hsv[:, 0]
    sat = hsv[:, 1]
    val = hsv[:, 2]

    # Hue
    hue_hist = normalized_hist(
        hue,
        36,
        (0, 180)
    )

    # Saturation
    sat_hist = normalized_hist(
        sat,
        8,
        (0, 256)
    )

    # Value
    val_hist = normalized_hist(
        val,
        8,
        (0, 256)
    )

    hsv_feature = np.concatenate([
        hue_hist,
        sat_hist,
        val_hist
    ])

    # --------------------------------------------------------
    # RGB histograms
    # --------------------------------------------------------

    rgb_hist = []

    for channel in range(3):

        rgb_hist.append(
            normalized_hist(
                pixels[:, channel],
                8,
                (0, 256)
            )
        )

    rgb_feature = np.concatenate(rgb_hist)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = []

    for channel in range(3):

        values = pixels[:, channel].astype(
            np.float32
        )

        stats.extend([
            values.mean() / 255.0,
            values.std() / 255.0,
            np.percentile(values, 10) / 255.0,
            np.percentile(values, 50) / 255.0,
            np.percentile(values, 90) / 255.0,
        ])

    # HSV statistics
    stats.extend([
        hue.mean() / 180.0,
        sat.mean() / 255.0,
        val.mean() / 255.0,
    ])

    stats = np.array(
        stats,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Spatial color distribution
    #
    # 4 x 4 grid
    # Each cell:
    #
    #   mean RGB
    #   mean HSV
    #   foreground ratio
    # --------------------------------------------------------

    spatial = []

    grid = 4

    for gy in range(grid):

        for gx in range(grid):

            y1 = gy * 64 // grid
            y2 = (gy + 1) * 64 // grid

            x1 = gx * 64 // grid
            x2 = (gx + 1) * 64 // grid

            cell = small[y1:y2, x1:x2]

            cell_alpha = small_alpha[
                y1:y2,
                x1:x2
            ]

            if np.any(cell_alpha > 10):

                cell_pixels = cell[
                    cell_alpha > 10
                ]

            else:

                cell_pixels = cell.reshape(
                    -1,
                    3
                )

            mean_rgb = (
                cell_pixels
                .mean(axis=0)
                / 255.0
            )

            cell_hsv = cv2.cvtColor(
                cell_pixels.reshape(-1, 1, 3),
                cv2.COLOR_RGB2HSV
            ).reshape(-1, 3)

            mean_hsv = np.array([
                cell_hsv[:, 0].mean() / 180.0,
                cell_hsv[:, 1].mean() / 255.0,
                cell_hsv[:, 2].mean() / 255.0,
            ])

            foreground_ratio = np.mean(
                cell_alpha > 10
            )

            spatial.extend(
                mean_rgb.tolist()
                + mean_hsv.tolist()
                + [foreground_ratio]
            )

    spatial = np.array(
        spatial,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Dominant colors
    #
    # K-means with 8 clusters
    # --------------------------------------------------------

    sample_size = min(
        len(pixels),
        3000
    )

    rng = np.random.default_rng(42)

    if len(pixels) > sample_size:

        indices = rng.choice(
            len(pixels),
            sample_size,
            replace=False
        )

        sample = pixels[indices]

    else:

        sample = pixels

    sample_float = np.float32(sample)

    k = min(8, len(sample_float))

    if k >= 2:

        criteria = (
            cv2.TERM_CRITERIA_EPS
            + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.2
        )

        try:

            _, labels, centers = cv2.kmeans(
                sample_float,
                k,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS
            )

            labels = labels.reshape(-1)

            counts = np.bincount(
                labels,
                minlength=k
            ).astype(np.float32)

            weights = counts / counts.sum()

            order = np.argsort(
                weights
            )[::-1]

            dominant = []

            for idx in order:

                color = centers[idx] / 255.0
                weight = weights[idx]

                dominant.extend([
                    color[0],
                    color[1],
                    color[2],
                    weight
                ])

            while len(dominant) < 32:
                dominant.extend([0, 0, 0, 0])

            dominant = dominant[:32]

        except Exception:

            dominant = [0.0] * 32

    else:

        dominant = [0.0] * 32

    dominant = np.array(
        dominant,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Combined feature
    # --------------------------------------------------------

    feature = np.concatenate([
        hsv_feature,
        rgb_feature,
        stats,
        spatial,
        dominant
    ])

    return {
        "feature": feature.astype(np.float32),
        "hsv": hsv_feature.astype(np.float32),
        "rgb": rgb_feature.astype(np.float32),
        "stats": stats.astype(np.float32),
        "spatial": spatial.astype(np.float32),
        "dominant": dominant.astype(np.float32),
    }


# ============================================================
# BUILD DATABASE
# ============================================================

def main():

    files = sorted([
        f
        for f in os.listdir(LOGO_DIR)
        if f.lower().endswith(
            (
                ".png",
                ".jpg",
                ".jpeg",
                ".webp",
                ".gif"
            )
        )
    ])

    print(
        f"Found {len(files)} logo files"
    )

    names = []

    database = {}

    for index, filename in enumerate(files, 1):

        print(
            f"[{index}/{len(files)}] {filename}"
        )

        path = os.path.join(
            LOGO_DIR,
            filename
        )

        frames = load_image_frames(path)

        if not frames:

            print(
                "  WARNING: Could not load image"
            )

            continue

        frame_features = []

        for rgb, alpha in frames:

            result = calculate_features(
                rgb,
                alpha
            )

            if result is not None:

                frame_features.append(result)

        if not frame_features:

            print(
                "  WARNING: No visible pixels"
            )

            continue

        # ----------------------------------------------------
        # Average all animation frames
        # ----------------------------------------------------

        feature = np.mean(
            [
                x["feature"]
                for x in frame_features
            ],
            axis=0
        )

        hsv = np.mean(
            [
                x["hsv"]
                for x in frame_features
            ],
            axis=0
        )

        rgb_feature = np.mean(
            [
                x["rgb"]
                for x in frame_features
            ],
            axis=0
        )

        stats = np.mean(
            [
                x["stats"]
                for x in frame_features
            ],
            axis=0
        )

        spatial = np.mean(
            [
                x["spatial"]
                for x in frame_features
            ],
            axis=0
        )

        dominant = np.mean(
            [
                x["dominant"]
                for x in frame_features
            ],
            axis=0
        )

        name = os.path.splitext(
            filename
        )[0]

        names.append(name)

        database[name] = {
            "filename": filename,
            "num_frames": len(frame_features),

            "feature": torch.tensor(
                feature
            ),

            "hsv": torch.tensor(
                hsv
            ),

            "rgb": torch.tensor(
                rgb_feature
            ),

            "stats": torch.tensor(
                stats
            ),

            "spatial": torch.tensor(
                spatial
            ),

            "dominant": torch.tensor(
                dominant
            ),
        }

        if len(frame_features) > 1:

            print(
                f"  Animated GIF: "
                f"{len(frame_features)} frames loaded"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output = {
        "version": 2,

        "feature_type": (
            "HSV + RGB + statistics "
            "+ spatial + dominant colors"
        ),

        "names": names,

        "database": database,

        "feature_size": (
            len(next(iter(database.values()))
                ["feature"])
            if database
            else 0
        ),
    }

    torch.save(
        output,
        OUTPUT
    )

    print()
    print("=" * 60)
    print("COLOR DATABASE CREATED")
    print("=" * 60)

    print(
        f"Logos:   {len(database)}"
    )

    print(
        f"Output:  "
        f"{os.path.abspath(OUTPUT)}"
    )

    print()

    print(
        f"Feature size: "
        f"{output['feature_size']}"
    )

    print()

    print(
        "Each GIF was processed using "
        "all visible animation frames."
    )


if __name__ == "__main__":
    main()
