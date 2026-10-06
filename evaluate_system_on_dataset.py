"""
Evaluate Brand Classifier and Forensic Pipeline on Real Datasets.

Tests:
  1. Authentic Brand Dataset (dataset/our_logos/positive & originals)
     -> Expectation: 100% Brand Recognition, 0% False Rejection.
  2. Competitor Logos (dataset/competitor_logos/competitor)
     -> Expectation: 100% Competitor Rejection (classified as 'unknown' / UNRELATED).
  3. Negative Redesigns (dataset/competitor_logos/negative/redesign)
     -> Expectation: High Rejection rate as 'unknown' / UNRELATED.

Usage:
    python evaluate_system_on_dataset.py
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.models.classifier_engine import ClassifierEngine


def run_benchmark():
    print("=" * 70)
    print("SYSTEM EVALUATION ON REAL DATASETS")
    print("=" * 70)

    # 1. Initialize Classifier Engine
    engine = ClassifierEngine()
    if not engine.loaded:
        print("[ERROR] ClassifierEngine could not load brand_classifier.pt")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Protected Brands: {len(engine.brand_names)} | Total Classes: {engine.num_classes}")

    processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL_NAME)
    model = AutoModel.from_pretrained(config.SIGLIP_MODEL_NAME).to(device).eval()
    vision_model = model.vision_model if hasattr(model, "vision_model") else model

    def get_embedding(img: Image.Image) -> torch.Tensor:
        inputs = processor(images=[img.convert("RGB")], return_tensors="pt", padding=True)
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            outputs = vision_model(pixel_values=pixel_values)
            emb = outputs.pooler_output if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

    # ============================================================
    # TEST 1: COMPETITOR LOGOS (dataset/competitor_logos/competitor)
    # ============================================================
    comp_dir = config.COMPETITOR_LOGOS_DIR / "competitor"
    comp_files = list(comp_dir.glob("*.*")) if comp_dir.exists() else []

    print(f"\n[Test 1] Testing on Real Competitor Logos ({len(comp_files)} files)...")
    comp_rejected = 0
    comp_results = []

    for f in comp_files:
        try:
            img = Image.open(f)
            if getattr(img, "is_animated", False):
                img.seek(0)
            emb = get_embedding(img)
            pred = engine.predict(emb)
            is_rej = pred["is_unknown"] or pred["unknown_prob"] > 0.40
            if is_rej:
                comp_rejected += 1
            comp_results.append({
                "file": f.name,
                "predicted": pred["predicted_brand"],
                "confidence": pred["confidence"],
                "unknown_prob": pred["unknown_prob"],
                "is_unknown": pred["is_unknown"],
            })
            print(f"  {f.name[:35]:<35} -> {pred['predicted_brand']} (unk_prob={pred['unknown_prob']:.3f}, conf={pred['confidence']:.3f}) {'[REJECTED OK]' if is_rej else '[MISCLASSIFIED]'}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")

    comp_rate = (comp_rejected / max(len(comp_files), 1)) * 100
    print(f"\nCompetitor Rejection Rate: {comp_rejected}/{len(comp_files)} ({comp_rate:.1f}%)")

    # ============================================================
    # TEST 2: AUTHENTIC LOGOS WITH BACKGROUND VARIATIONS (dataset/our_logos/positive)
    # ============================================================
    pos_dir = config.OUR_LOGOS_DIR / "positive"
    brand_subdirs = sorted([d for d in pos_dir.iterdir() if d.is_dir()]) if pos_dir.exists() else []

    print(f"\n[Test 2] Testing on Authentic Brand Positive Variations ({len(brand_subdirs)} brands)...")
    auth_correct = 0
    auth_total = 0
    bg_breakdown: Dict[str, List[bool]] = {}

    for i, b_dir in enumerate(brand_subdirs, 1):
        brand_name = engine.brand_names[i - 1] if i - 1 < len(engine.brand_names) else b_dir.name
        files = list(b_dir.glob("*.*"))

        for f in files:
            try:
                img = Image.open(f)
                if getattr(img, "is_animated", False):
                    img.seek(0)
                emb = get_embedding(img)
                pred = engine.predict(emb)
                correct = (pred["predicted_brand"] == brand_name)
                if correct:
                    auth_correct += 1
                auth_total += 1

                # Track by variation type
                v_type = "other"
                if "bg_black" in f.name: v_type = "bg_black"
                elif "bg_white" in f.name: v_type = "bg_white"
                elif "bg_transparent" in f.name: v_type = "bg_transparent"
                elif "bg_dark_gray" in f.name: v_type = "bg_dark_gray"
                elif "bg_light_gray" in f.name: v_type = "bg_light_gray"
                elif "blur" in f.name: v_type = "blur"
                elif "crop" in f.name: v_type = "crop"
                elif "jpeg" in f.name: v_type = "jpeg"

                if v_type not in bg_breakdown:
                    bg_breakdown[v_type] = []
                bg_breakdown[v_type].append(correct)

            except Exception:
                pass

        if i % 10 == 0 or i == len(brand_subdirs):
            print(f"  Processed {i}/{len(brand_subdirs)} brands ({auth_correct}/{auth_total} correct so far)...")

    auth_acc = (auth_correct / max(auth_total, 1)) * 100
    print(f"\nAuthentic Brand Recognition Accuracy: {auth_correct}/{auth_total} ({auth_acc:.2f}%)")
    print("Variation Type Breakdown:")
    for v_type, bools in sorted(bg_breakdown.items()):
        v_acc = sum(bools) / max(len(bools), 1) * 100
        print(f"  - {v_type:<18}: {sum(bools):3d}/{len(bools):3d} ({v_acc:5.1f}%)")

    # ============================================================
    # TEST 3: NEGATIVE REDESIGNS (dataset/competitor_logos/negative/redesign)
    # ============================================================
    redesign_dir = config.COMPETITOR_LOGOS_DIR / "negative" / "redesign"
    redesign_files = list(redesign_dir.rglob("*.*")) if redesign_dir.exists() else []

    if redesign_files:
        print(f"\n[Test 3] Testing on Negative Redesigns ({len(redesign_files)} files)...")
        redesign_rejected = 0
        for f in redesign_files:
            try:
                img = Image.open(f)
                if getattr(img, "is_animated", False):
                    img.seek(0)
                emb = get_embedding(img)
                pred = engine.predict(emb)
                if pred["is_unknown"] or pred["unknown_prob"] > 0.35:
                    redesign_rejected += 1
            except Exception:
                pass
        redesign_rate = (redesign_rejected / max(len(redesign_files), 1)) * 100
        print(f"Redesign Rejection Rate: {redesign_rejected}/{len(redesign_files)} ({redesign_rate:.1f}%)")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Authentic Brand Accuracy (with background variations): {auth_acc:.2f}%")
    print(f"Competitor Logo Rejection Rate                         : {comp_rate:.1f}%")
    if redesign_files:
        print(f"Negative Redesign Rejection Rate                       : {redesign_rate:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
