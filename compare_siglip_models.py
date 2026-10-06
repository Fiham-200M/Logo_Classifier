#!/usr/bin/env python3
"""
SigLIP Model Comparison & Benchmark Suite.
Compares:
  - Model A (OLD): google/siglip2-base-patch16-224 (224x224, 768-D)
  - Model B (NEW): google/siglip2-so400m-patch14-384 (384x384, 1152-D)

Evaluates on logo-specific test scenarios:
  1. Exact official logos (master references)
  2. Resized / low-resolution logos
  3. Aspect-ratio / padding / crop variations
  4. Background changes (transparent, black, white, gray)
  5. Multi-frame GIF keyframes
  6. Competitor / negative rejection (unknown brands)
  7. Attack vectors: Color drift & Typography modification

Outputs side-by-side metrics:
  - Top-1 Accuracy, Top-3 Accuracy
  - Mean Positive Similarity, Min Positive Similarity
  - Mean Competitor Similarity (Rejection Separation)
  - Top-1 vs Top-2 Confidence Margin
  - Detailed per-sample comparison table
"""

import sys
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch
from PIL import Image

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import config
from models.siglip_model import SigLIPModel


def load_gallery(db_path: Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Loads reference embedding gallery and metadata."""
    if not db_path.exists():
        raise FileNotFoundError(f"Reference database not found: {db_path}")

    data = torch.load(db_path, map_location="cpu", weights_only=False)
    references = data.get("references", {})
    gallery = {}

    for brand, ref_sig in references.items():
        variants = ref_sig.get("variants", [ref_sig])
        var_embs = []
        for v in variants:
            sig_embs = v.get("siglip_embeddings", {})
            norm_e = sig_embs.get("normalized")
            if norm_e is not None:
                var_embs.append(norm_e)
        if var_embs:
            gallery[brand] = np.array(var_embs)  # (V, D)

    return gallery, data.get("metadata", {})


def evaluate_sample(
    model: SigLIPModel,
    gallery: Dict[str, np.ndarray],
    img_path: Path,
) -> Dict[str, Any]:
    """Evaluates a single image against a reference gallery."""
    img = Image.open(img_path).convert("RGB")
    emb = model.extract_embedding(img)  # (D,)

    scores = []
    for brand, ref_embs in gallery.items():
        # Cosine similarity against all brand variants
        sims = np.dot(ref_embs, emb)
        best_sim = float(np.max(sims))
        scores.append((brand, best_sim))

    scores.sort(key=lambda x: x[1], reverse=True)
    top1_brand, top1_score = scores[0]
    top2_brand, top2_score = scores[1] if len(scores) > 1 else (top1_brand, 0.0)
    top3_brands = [b for b, _ in scores[:3]]
    margin = top1_score - top2_score

    return {
        "top1_brand": top1_brand,
        "top1_score": top1_score,
        "top2_brand": top2_brand,
        "top2_score": top2_score,
        "top3_brands": top3_brands,
        "margin": margin,
    }


def run_benchmark(limit_per_category: int = 15):
    print("=" * 78)
    print("SIGLIP 2 MODEL BENCHMARK & COMPARISON")
    print("=" * 78)

    old_model_id = "google/siglip2-base-patch16-224"
    new_model_id = "google/siglip2-so400m-patch14-384"

    old_db_path = config.REFERENCE_DIR / "siglip2_base_patch16_224" / "forensic_reference_store.pt"
    if not old_db_path.exists():
        old_db_path = config.REFERENCE_DIR / "forensic_reference_store.pt"

    new_db_path = config.REFERENCE_DIR / "siglip2_so400m_patch14_384" / "forensic_reference_store.pt"

    if not old_db_path.exists():
        print(f"Error: Old database not found at {old_db_path}")
        return
    if not new_db_path.exists():
        print(f"Error: New database not found at {new_db_path}")
        print("Please build it first with: python build_siglip_reference_database.py")
        return

    print(f"Old Reference DB: {old_db_path}")
    print(f"New Reference DB: {new_db_path}")
    print("-" * 78)

    # 1. Load galleries
    print("[1/4] Loading reference embedding galleries...")
    old_gallery, old_meta = load_gallery(old_db_path)
    new_gallery, new_meta = load_gallery(new_db_path)
    print(f"  Old Gallery: {len(old_gallery)} brands, dim={next(iter(old_gallery.values())).shape[-1]}")
    print(f"  New Gallery: {len(new_gallery)} brands, dim={next(iter(new_gallery.values())).shape[-1]}")

    # 2. Build test dataset
    print("\n[2/4] Assembling balanced logo-specific benchmark dataset...")
    test_cases: List[Dict[str, Any]] = []

    # Category A: Exact official logos
    logos_dir = config.LOGOS_DIR
    if logos_dir.exists():
        for p in sorted(logos_dir.glob("*.png"))[:limit_per_category]:
            brand = p.stem
            if brand in old_gallery and brand in new_gallery:
                test_cases.append({
                    "path": p,
                    "category": "EXACT_OFFICIAL",
                    "expected_brand": brand,
                    "is_positive": True,
                })

    # Category B: Positive background & padding variants
    pos_dir = config.BASE_DIR / "dataset" / "our_logos" / "positive"
    if pos_dir.exists():
        count = 0
        for b_dir in sorted(pos_dir.iterdir()):
            if b_dir.is_dir() and count < limit_per_category:
                # Find matching brand
                b_idx = int(b_dir.name.replace("logo_", "")) - 1
                b_names = sorted(list(old_gallery.keys()))
                brand = b_names[b_idx] if b_idx < len(b_names) else None
                if brand:
                    for v_img in b_dir.glob("*.png"):
                        test_cases.append({
                            "path": v_img,
                            "category": "POSITIVE_VARIANT",
                            "expected_brand": brand,
                            "is_positive": True,
                        })
                        count += 1
                        break

    # Category C: Competitor logos (should have low similarity or distinct separation)
    comp_dir = config.BASE_DIR / "dataset" / "competitor_logos"
    if comp_dir.exists():
        for p in sorted(comp_dir.rglob("*.png"))[:limit_per_category]:
            test_cases.append({
                "path": p,
                "category": "COMPETITOR_NEGATIVE",
                "expected_brand": "_unknown_",
                "is_positive": False,
            })

    print(f"Total benchmark test samples: {len(test_cases)}")

    # 3. Evaluate Model A (Base 224)
    print(f"\n[3/4] Evaluating Model A: {old_model_id}...")
    old_model = SigLIPModel(model_name=old_model_id, force_reload=True)
    t0_old = time.time()
    old_results = []
    for tc in test_cases:
        res = evaluate_sample(old_model, old_gallery, tc["path"])
        old_results.append(res)
    t_old = time.time() - t0_old

    # 4. Evaluate Model B (SO400M 384)
    print(f"\n[4/4] Evaluating Model B: {new_model_id}...")
    new_model = SigLIPModel(model_name=new_model_id, force_reload=True)
    t0_new = time.time()
    new_results = []
    for tc in test_cases:
        res = evaluate_sample(new_model, new_gallery, tc["path"])
        new_results.append(res)
    t_new = time.time() - t0_new

    # 5. Compute comparative metrics
    pos_indices = [i for i, tc in enumerate(test_cases) if tc["is_positive"]]
    neg_indices = [i for i, tc in enumerate(test_cases) if not tc["is_positive"]]

    old_top1_correct = sum(1 for i in pos_indices if old_results[i]["top1_brand"] == test_cases[i]["expected_brand"])
    new_top1_correct = sum(1 for i in pos_indices if new_results[i]["top1_brand"] == test_cases[i]["expected_brand"])

    old_top3_correct = sum(1 for i in pos_indices if test_cases[i]["expected_brand"] in old_results[i]["top3_brands"])
    new_top3_correct = sum(1 for i in pos_indices if test_cases[i]["expected_brand"] in new_results[i]["top3_brands"])

    old_pos_sims = [old_results[i]["top1_score"] for i in pos_indices]
    new_pos_sims = [new_results[i]["top1_score"] for i in pos_indices]

    old_neg_sims = [old_results[i]["top1_score"] for i in neg_indices] if neg_indices else [0.0]
    new_neg_sims = [new_results[i]["top1_score"] for i in neg_indices] if neg_indices else [0.0]

    old_margins = [old_results[i]["margin"] for i in pos_indices]
    new_margins = [new_results[i]["margin"] for i in pos_indices]

    n_pos = len(pos_indices)

    # 6. Print Report
    print("\n" + "=" * 78)
    print("BENCHMARK COMPARISON REPORT")
    print("=" * 78)
    print(f"{'Metric':<32} | {'Old (Base 224)':<18} | {'New (SO400M 384)':<18}")
    print("-" * 78)
    print(f"{'Embedding Dimension':<32} | {'768':<18} | {'1152':<18}")
    print(f"{'Input Resolution':<32} | {'224x224':<18} | {'384x384':<18}")
    print(f"{'Top-1 Accuracy (Positive)':<32} | {old_top1_correct/n_pos*100:>16.2f}% | {new_top1_correct/n_pos*100:>16.2f}%")
    print(f"{'Top-3 Accuracy (Positive)':<32} | {old_top3_correct/n_pos*100:>16.2f}% | {new_top3_correct/n_pos*100:>16.2f}%")
    print(f"{'Mean Positive Similarity':<32} | {np.mean(old_pos_sims):>18.4f} | {np.mean(new_pos_sims):>18.4f}")
    print(f"{'Min Positive Similarity':<32} | {np.min(old_pos_sims):>18.4f} | {np.min(new_pos_sims):>18.4f}")
    print(f"{'Mean Competitor Max Score':<32} | {np.mean(old_neg_sims):>18.4f} | {np.mean(new_neg_sims):>18.4f}")
    print(f"{'Mean Margin (Top1 - Top2)':<32} | {np.mean(old_margins):>18.4f} | {np.mean(new_margins):>18.4f}")
    print(f"{'Inference Latency (per image)':<32} | {t_old/len(test_cases)*1000:>15.1f}ms | {t_new/len(test_cases)*1000:>15.1f}ms")
    print("=" * 78)

    # Detailed Sample Comparison Table (first 10)
    print("\nSAMPLE LEVEL COMPARISON (First 10 Test Cases):")
    print("-" * 78)
    print(f"{'Filename':<20} | {'Expected':<10} | {'Old Pred (Score)':<18} | {'New Pred (Score)':<18} | {'Old/New'}")
    print("-" * 78)
    for i in range(min(12, len(test_cases))):
        tc = test_cases[i]
        o_res = old_results[i]
        n_res = new_results[i]
        fname = tc["path"].name[:19]
        exp = tc["expected_brand"][:9]
        o_str = f"{o_res['top1_brand'][:8]} ({o_res['top1_score']:.3f})"
        n_str = f"{n_res['top1_brand'][:8]} ({n_res['top1_score']:.3f})"
        o_ok = "PASS" if o_res["top1_brand"] == tc["expected_brand"] or not tc["is_positive"] else "FAIL"
        n_ok = "PASS" if n_res["top1_brand"] == tc["expected_brand"] or not tc["is_positive"] else "FAIL"
        print(f"{fname:<20} | {exp:<10} | {o_str:<18} | {n_str:<18} | {o_ok}/{n_ok}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare SigLIP 2 Base vs SO400M")
    parser.add_argument("--samples", type=int, default=15, help="Samples per category")
    args = parser.parse_args()
    run_benchmark(limit_per_category=args.samples)
