"""
Unified Database Builder.
Builds and maintains multi-reference databases for logos and favicons:
  1. SigLIP multi-reference embeddings (brand_multi_database.pt)
  2. Color features multi-database (color_database.pt)
  3. OCR reference database (ocr_database.pt)
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import torch
import numpy as np
from PIL import Image

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.database.brand_database import BrandDatabase, get_brand_for_favicon_filename
from src.models.siglip_engine import SigLIPEngine
from src.models.color_engine import extract_color_features
from src.preprocessing.image_preprocessor import (
    load_image_safe,
    generate_preprocessing_views,
)


def build_multi_reference_siglip(
    db: BrandDatabase,
    siglip: SigLIPEngine,
    output_path: Path = getattr(config, "BRAND_MULTI_DB_PATH", config.REFERENCE_EMBEDDINGS_DIR / "brand_multi_database.pt"),
) -> Path:
    """
    Extracts SigLIP embeddings for all reference images associated with each brand
    (primary logo, variants in logo_XXX/, and official favicons).
    For each image, extracts embeddings across key deterministic views.
    Saves a dict mapping brand name -> (K, 768) float32 tensor.
    """
    print(f"\n[build_multi_reference_siglip] Extracting multi-reference embeddings for {len(db.brand_names)} brands...")
    multi_embs = {}
    total_refs = 0

    key_views = ["raw_rgb", "square_padded", "white_bg", "black_bg", "contrast_enhanced", "sharpened"]

    for idx, brand in enumerate(db.brand_names, 1):
        paths = db.brand_to_ref_paths.get(brand, [])
        brand_vectors = []

        for p in paths:
            if not p.exists():
                continue
            try:
                img = load_image_safe(p)
                views = generate_preprocessing_views(img)
                for vname in key_views:
                    if vname in views:
                        emb = siglip.extract_embedding(views[vname])
                        brand_vectors.append(emb)
            except Exception as e:
                print(f"  Warning: failed to extract views for {p}: {e}")

        # Fallback to primary reference embedding if empty
        if not brand_vectors and db.reference_embeddings is not None and (idx - 1) < len(db.reference_embeddings):
            brand_vectors.append(db.reference_embeddings[idx - 1])

        if brand_vectors:
            mat = np.stack(brand_vectors, axis=0)  # Shape (K, 768)
            norms = np.linalg.norm(mat, axis=-1, keepdims=True) + 1e-12
            mat = mat / norms
            # Remove near-duplicates to keep representation compact and fast
            unique_vectors = [mat[0]]
            for vec in mat[1:]:
                sims = np.dot(unique_vectors, vec)
                if np.max(sims) < 0.999:
                    unique_vectors.append(vec)

            final_mat = np.stack(unique_vectors, axis=0)
            multi_embs[brand] = torch.from_numpy(final_mat.astype(np.float32))
            total_refs += len(final_mat)
        else:
            print(f"  Warning: No embeddings found for brand '{brand}'")

        if idx % 10 == 0 or idx == len(db.brand_names):
            print(f"  Processed {idx}/{len(db.brand_names)} brands ({total_refs} total reference vectors)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "multi_embeddings": multi_embs,
            "brand_names": db.brand_names,
            "total_refs": total_refs,
        },
        output_path,
    )
    print(f"[build_multi_reference_siglip] Saved {total_refs} vectors across {len(multi_embs)} brands to {output_path}")
    return output_path


def build_multi_reference_color(
    db: BrandDatabase,
    output_path: Path = config.COLOR_DB_PATH,
) -> Path:
    """
    Extracts 238-dim color features for all reference images (logos + favicons) per brand.
    Saves a dict mapping brand name -> List of feature dicts in color_database.pt.
    """
    print(f"\n[build_multi_reference_color] Extracting color features for {len(db.brand_names)} brands...")
    color_db = {}
    total_color_entries = 0

    for idx, brand in enumerate(db.brand_names, 1):
        paths = db.brand_to_ref_paths.get(brand, [])
        brand_color_list = []

        for p in paths:
            if not p.exists():
                continue
            try:
                feat = extract_color_features(p)
                if feat is not None:
                    # Convert numpy arrays to torch tensors for storage compatibility
                    feat_tensors = {
                        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
                        for k, v in feat.items()
                    }
                    brand_color_list.append(feat_tensors)
            except Exception as e:
                print(f"  Warning: color extraction failed for {p}: {e}")

        if brand_color_list:
            color_db[brand] = brand_color_list
            total_color_entries += len(brand_color_list)
        elif brand in db.color_database:
            color_db[brand] = db.color_database[brand]

        if idx % 10 == 0 or idx == len(db.brand_names):
            print(f"  Processed {idx}/{len(db.brand_names)} brands ({total_color_entries} color entries)...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "database": color_db,
            "brand_names": db.brand_names,
            "total_entries": total_color_entries,
        },
        output_path,
    )
    print(f"[build_multi_reference_color] Saved {total_color_entries} color entries across {len(color_db)} brands to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Unified Reference Database Builder")
    parser.add_argument("--multi-siglip", action="store_true", help="Build SigLIP multi-reference database")
    parser.add_argument("--color", action="store_true", help="Build color multi-reference database")
    parser.add_argument("--all", action="store_true", help="Build all multi-reference databases")
    args = parser.parse_args()

    db = BrandDatabase()

    if args.multi_siglip or args.all:
        siglip = SigLIPEngine()
        build_multi_reference_siglip(db, siglip)

    if args.color or args.all:
        build_multi_reference_color(db)


if __name__ == "__main__":
    main()

