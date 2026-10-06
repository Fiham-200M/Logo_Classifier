from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
LOGO_DIR = PROJECT_DIR / "logos"
OUTPUT_DIR = PROJECT_DIR / "reference_embeddings"

MODEL_NAME = "google/siglip2-base-patch16-224"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
}


# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"


print("=" * 70)
print("SIGLIP 2 - BRAND REFERENCE DATABASE")
print("=" * 70)

print(f"Project directory : {PROJECT_DIR}")
print(f"Logo directory    : {LOGO_DIR}")
print(f"Output directory  : {OUTPUT_DIR}")
print(f"Model             : {MODEL_NAME}")
print(f"Device            : {device}")

if torch.cuda.is_available():
    print(f"GPU               : {torch.cuda.get_device_name(0)}")

    vram_gb = (
        torch.cuda.get_device_properties(0).total_memory
        / (1024 ** 3)
    )

    print(f"GPU memory        : {vram_gb:.2f} GB")

print("=" * 70)


# ============================================================
# FIND LOGOS
# ============================================================

if not LOGO_DIR.exists():
    raise FileNotFoundError(
        f"Logo directory does not exist: {LOGO_DIR}"
    )


logo_files = sorted(
    [
        p
        for p in LOGO_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)


print(f"\nFound {len(logo_files)} image files.")

if len(logo_files) == 0:
    raise RuntimeError(
        f"No logo images found in {LOGO_DIR}"
    )


print("\nLogo files:")

for i, path in enumerate(logo_files, 1):
    print(f"  {i:02d}. {path.name}")


# ============================================================
# LOAD PROCESSOR
# ============================================================

print("\n" + "=" * 70)
print("Loading SigLIP 2 processor...")
print("=" * 70)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 70)
print("Loading SigLIP 2 model...")
print("=" * 70)

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    f"Total parameters     : "
    f"{total_parameters:,}"
)

print(
    f"Device                : "
    f"{device}"
)

print(
    f"Model class           : "
    f"{type(model).__name__}"
)

print(
    f"Vision model class    : "
    f"{type(model.vision_model).__name__}"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STORAGE
# ============================================================

embeddings = []
brand_names = []
file_names = []
failed = []


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

print("\n" + "=" * 70)
print("Generating brand embeddings...")
print("=" * 70)


with torch.inference_mode():

    for index, image_path in enumerate(
        logo_files,
        1
    ):

        print(
            f"\n[{index:02d}/{len(logo_files)}] "
            f"{image_path.name}"
        )

        try:

            # ==================================================
            # LOAD IMAGE
            # ==================================================

            image = Image.open(image_path)

            # GIF:
            # use first frame
            if getattr(
                image,
                "is_animated",
                False
            ):
                image.seek(0)

            image = image.convert("RGB")

            print(
                f"  Image size       : "
                f"{image.width} x {image.height}"
            )


            # ==================================================
            # PROCESS IMAGE
            # ==================================================

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
                if torch.is_tensor(value)
            }


            # ==================================================
            # RUN VISION MODEL
            # ==================================================

            vision_output = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )


            # ==================================================
            # GET POOLER OUTPUT
            # ==================================================

            if hasattr(
                vision_output,
                "pooler_output"
            ):

                image_features = (
                    vision_output.pooler_output
                )

            else:

                raise RuntimeError(
                    "Vision model did not return "
                    "pooler_output."
                )


            # ==================================================
            # CHECK EMBEDDING
            # ==================================================

            if not torch.is_tensor(
                image_features
            ):

                raise RuntimeError(
                    "pooler_output is not a tensor."
                )


            print(
                f"  Raw embedding     : "
                f"{tuple(image_features.shape)}"
            )


            # ==================================================
            # NORMALIZE
            # ==================================================

            image_features = (
                image_features
                / (
                    image_features.norm(
                        dim=-1,
                        keepdim=True
                    )
                    + 1e-12
                )
            )


            # ==================================================
            # CPU COPY
            # ==================================================

            embedding = (
                image_features[0]
                .detach()
                .cpu()
                .float()
            )


            # ==================================================
            # SAVE
            # ==================================================

            embeddings.append(
                embedding
            )

            brand_names.append(
                image_path.stem
            )

            file_names.append(
                image_path.name
            )


            print(
                f"  Final embedding   : "
                f"{tuple(embedding.shape)}"
            )

            print(
                "  Status            : OK"
            )


        except Exception as error:

            print(
                f"  ERROR             : {error}"
            )

            failed.append(
                {
                    "file": image_path.name,
                    "error": str(error),
                }
            )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(embeddings) == 0:

    raise RuntimeError(
        "No embeddings were generated."
    )


# ============================================================
# CREATE EMBEDDING MATRIX
# ============================================================

embedding_tensor = torch.stack(
    embeddings
)


# ============================================================
# FINAL NORMALIZATION
#
# This guarantees every stored vector is normalized.
# ============================================================

embedding_tensor = (
    embedding_tensor
    / (
        embedding_tensor.norm(
            dim=1,
            keepdim=True
        )
        + 1e-12
    )
)


# ============================================================
# DATABASE
# ============================================================

database = {
    "model_name": MODEL_NAME,

    "embedding_dimension":
        int(embedding_tensor.shape[1]),

    "num_brands":
        int(embedding_tensor.shape[0]),

    "embeddings":
        embedding_tensor,

    "brand_names":
        brand_names,

    "file_names":
        file_names,
}


# ============================================================
# SAVE DATABASE
# ============================================================

database_path = (
    OUTPUT_DIR /
    "brand_database.pt"
)

torch.save(
    database,
    database_path
)


# ============================================================
# SAVE BRAND LIST
# ============================================================

brand_list_path = (
    OUTPUT_DIR /
    "brand_list.txt"
)

with open(
    brand_list_path,
    "w",
    encoding="utf-8"
) as f:

    for brand_name, file_name in zip(
        brand_names,
        file_names
    ):

        f.write(
            f"{brand_name}\t{file_name}\n"
        )


# ============================================================
# SAVE FAILURE REPORT
# ============================================================

if failed:

    failed_path = (
        OUTPUT_DIR /
        "failed_logos.txt"
    )

    with open(
        failed_path,
        "w",
        encoding="utf-8"
    ) as f:

        for item in failed:

            f.write(
                f"{item['file']}\t"
                f"{item['error']}\n"
            )


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("DATABASE CREATION COMPLETE")
print("=" * 70)

print(
    f"Input logos         : {len(logo_files)}"
)

print(
    f"Successful          : {len(embeddings)}"
)

print(
    f"Failed              : {len(failed)}"
)

print(
    f"Embedding matrix    : "
    f"{tuple(embedding_tensor.shape)}"
)

print(
    f"Embedding dimension : "
    f"{embedding_tensor.shape[1]}"
)

print(
    f"Database file       : "
    f"{database_path}"
)

print(
    f"Brand list          : "
    f"{brand_list_path}"
)


if failed:

    print("\nFailed files:")

    for item in failed:

        print(
            f"  {item['file']}: "
            f"{item['error']}"
        )

else:

    print(
        "\nAll 52 logo images "
        "processed successfully."
    )


print("=" * 70)
