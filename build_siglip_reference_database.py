#!/usr/bin/env python3
"""
SigLIP 2 Reference Embedding Database Builder.

Builds the official versioned forensic reference store containing:
  - Master raw logos (PNG / WebP / GIF)
  - Canonical original brand renders
  - Animated multi-frame keyframes
  - Authentic positive variants (padding, background, crop)
  - Official protected brand favicons

Validates every embedding:
  - Exact dimension (1152 for SO400M / 768 for Base)
  - Unit L2 norm (1.0000)
  - No NaN / Inf values

Usage:
    python build_siglip_reference_database.py
    python build_siglip_reference_database.py --force
"""

import sys
import os
import argparse
import time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config
from models.siglip_model import SigLIPModel
from models.dinov2_model import DINOv2Model
from models.ocr_model import OCRModel
from reference_database.reference_store import ReferenceStore


def build_database(force: bool = False):
    print("=" * 60)
    print("SIGLIP 2 REFERENCE DATABASE BUILDER")
    print("=" * 60)

    # 1. Initialize models
    t0 = time.time()
    siglip = SigLIPModel(force_reload=force)
    dinov2 = DINOv2Model()
    ocr = OCRModel()

    model_id = siglip.model_name
    device = siglip.device
    input_size = siglip.input_size
    patch_size = siglip.patch_size
    embedding_dim = siglip.embedding_dim

    # Determine versioned database output path
    slug = config.get_siglip_slug(model_id)
    out_dir = config.REFERENCE_DIR / slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "forensic_reference_store.pt"

    print(f"Target Database:      {out_path}")
    print(f"Active Model ID:      {model_id}")
    print(f"Device / Dtype:       {device} / {siglip.dtype}")
    print(f"Input Resolution:     {input_size}x{input_size}")
    print(f"Patch Size:           {patch_size}")
    print(f"Embedding Dimension:  {embedding_dim}")
    print("-" * 60)

    # 2. Initialize ReferenceStore targeting this versioned path
    store = ReferenceStore(cache_path=out_path)
    brand_names = store.brand_names
    total_brands = len(brand_names)
    print(f"Discovered {total_brands} protected brands.")

    store.references = {}
    successful_brands = 0
    failed_brands = []
    total_variants_embedded = 0

    # 3. Extract signatures for all brands and variants
    for idx, brand in enumerate(brand_names, 1):
        print(f"[{idx:02d}/{total_brands}] Processing brand: {brand}...")
        try:
            sig = store._extract_brand_signature(brand, siglip, dinov2, ocr)
            if not sig:
                print(f"  [ERROR] No reference media found for {brand}!")
                failed_brands.append(brand)
                continue

            variants = sig.get("variants", [sig])
            valid_brand = True

            for v_idx, v in enumerate(variants):
                sig_embs = v.get("siglip_embeddings", {})
                norm_emb = sig_embs.get("normalized")
                white_emb = sig_embs.get("white")
                black_emb = sig_embs.get("black")

                for name, emb in [("normalized", norm_emb), ("white", white_emb), ("black", black_emb)]:
                    if emb is None:
                        print(f"  [ERROR] {brand} variant {v_idx} missing '{name}' embedding!")
                        valid_brand = False
                        break
                    if emb.shape[-1] != embedding_dim:
                        print(f"  [ERROR] {brand} variant {v_idx} '{name}' dim {emb.shape[-1]} != {embedding_dim}!")
                        valid_brand = False
                        break
                    norm_val = np.linalg.norm(emb)
                    if abs(norm_val - 1.0) > 1e-4:
                        print(f"  [WARNING] {brand} variant {v_idx} '{name}' L2 norm {norm_val:.4f} != 1.0! Re-normalizing...")
                        sig_embs[name] = emb / (norm_val + 1e-12)
                    if np.isnan(emb).any() or np.isinf(emb).any():
                        print(f"  [ERROR] {brand} variant {v_idx} '{name}' contains NaN/Inf!")
                        valid_brand = False
                        break

            if valid_brand:
                store.references[brand] = sig
                successful_brands += 1
                total_variants_embedded += len(variants)
                print(f"  -> Success: {len(variants)} variants embedded (SigLIP dim: {embedding_dim})")
            else:
                failed_brands.append(brand)

        except Exception as e:
            print(f"  [EXCEPTION] Failed processing brand {brand}: {e}")
            failed_brands.append(brand)

    # 4. Save metadata and serialized store
    metadata = {
        "model_id": model_id,
        "model_family": "SigLIP 2",
        "image_size": input_size,
        "patch_size": patch_size,
        "embedding_dim": embedding_dim,
        "brand_count": successful_brands,
        "total_variants": total_variants_embedded,
        "creation_timestamp": datetime.now().isoformat(),
        "preprocessing_version": "2.0",
    }

    payload = {
        "metadata": metadata,
        "model_id": metadata["model_id"],
        "embedding_dim": metadata["embedding_dim"],
        "brand_names": brand_names,
        "brand_to_filename": store.brand_to_filename,
        "references": store.references,
        "num_brands": successful_brands,
    }

    torch.save(payload, out_path)
    total_time = time.time() - t0

    # 5. Print summary
    print("\n" + "=" * 60)
    print("SIGLIP 2 REFERENCE DATABASE SUMMARY")
    print("=" * 60)
    print(f"Model ID:              {model_id}")
    print(f"Device:                {device} ({siglip.dtype})")
    print(f"Input Resolution:      {input_size}x{input_size}")
    print(f"Patch Size:            {patch_size}")
    print(f"Embedding Dimension:   {embedding_dim}")
    print()
    print(f"Brands Discovered:     {total_brands}")
    print(f"Successfully Embedded: {successful_brands}")
    print(f"Failed Brands:         {len(failed_brands)}")
    if failed_brands:
        print(f"Failed Brand Names:    {', '.join(failed_brands)}")
    print(f"Total Variants:        {total_variants_embedded}")
    print(f"Database File:         {out_path}")
    print(f"File Size:             {out_path.stat().st_size / (1024**2):.2f} MB")
    print(f"Total Elapsed Time:    {total_time:.1f}s")
    print("=" * 60)

    if successful_brands == total_brands:
        print("\nAll 52 protected brands successfully verified and saved!")
        return 0
    else:
        print(f"\nWarning: Only {successful_brands}/{total_brands} brands embedded successfully.")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build SigLIP 2 reference embedding database")
    parser.add_argument("--force", action="store_true", help="Force rebuild all signatures")
    args = parser.parse_args()
    sys.exit(build_database(force=args.force))
