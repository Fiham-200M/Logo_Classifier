"""
Command-Line Interface for Multi-Stage Logo Recognition System.
Usage:
    python logo_match.py path/to/image.png
    python logo_match.py path/to/image.png --debug
    python logo_match.py path/to/image.png --vlm
    python logo_match.py path/to/image.png --json
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image

import config
from preprocessing import generate_preprocessing_views, load_image_safe
from siglip_engine import SigLIPEngine
from color_features import extract_color_features, compare_color_features
from ocr_engine import OCREngine
from structure_features import extract_structure_feature, compute_structure_similarity
from database import BrandDatabase
from fusion import ScoreFusion
from vlm_fallback import query_vlm_fallback, query_vlm_text_extraction


class LogoRecognizer:
    def __init__(self, use_vlm: bool = False):
        self.use_vlm = use_vlm
        self.db = BrandDatabase()
        self.siglip = SigLIPEngine()
        self.ocr = OCREngine()
        self.fusion = ScoreFusion()

    def predict(self, image_path: Path, force_vlm: bool = False) -> Dict[str, Any]:
        """
        Run end-to-end multi-stage logo recognition on an input image.
        """
        # 1. Load image & generate deterministic multi-views
        img = load_image_safe(image_path)
        views = generate_preprocessing_views(img)
        view_names = list(views.keys())

        num_brands = len(self.db.brand_names)

        # 2. SigLIP Visual Similarity (Multi-view ensemble with multi-ref support)
        if hasattr(self.db, "score_siglip_multi_ref") and getattr(self.db, "brand_multi_embeddings", None):
            siglip_scores, view_breakdown = self.db.score_siglip_multi_ref(
                self.siglip.extract_multi_view_embeddings(views), strategy=config.SIGLIP_ENSEMBLE_STRATEGY
            )
        else:
            siglip_scores, view_breakdown = self.siglip.score_against_references(
                views, self.db.reference_embeddings, strategy=config.SIGLIP_ENSEMBLE_STRATEGY
            )

        # 3. Color Similarity
        query_color = extract_color_features(img)
        color_scores = np.zeros(num_brands, dtype=np.float32)

        if query_color is not None:
            for i, brand in enumerate(self.db.brand_names):
                ref_color = self.db.color_database.get(brand)
                if ref_color is not None:
                    color_scores[i] = compare_color_features(query_color, ref_color)
                else:
                    color_scores[i] = 0.50
        else:
            color_scores.fill(0.50)

        # 4. OCR Text Evidence
        ocr_scores, detected_texts = self.ocr.score_candidates(
            views, self.db.brand_names, self.db.ocr_database
        )

        # 5. Score Fusion
        fused_scores = self.fusion.fuse(
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
        )

        # 6. Check for Ambiguity / Protected Brands / VLM Fallback Trigger
        sorted_idx = np.argsort(fused_scores)[::-1]
        top1_idx, top2_idx = sorted_idx[0], sorted_idx[1]
        top1_brand, top2_brand = self.db.brand_names[top1_idx], self.db.brand_names[top2_idx]
        margin = float(fused_scores[top1_idx] - fused_scores[top2_idx])

        vlm_result = None
        vlm_texts = None
        should_trigger_vlm = force_vlm or self.use_vlm or (
            (margin < config.VLM_AMBIGUITY_MARGIN)
            or (top1_brand in config.PROTECTED_BRANDS and margin < config.PROTECTED_VLM_TRIGGER_MARGIN)
        )

        if should_trigger_vlm and (config.VLM_ENABLED or force_vlm or self.use_vlm):
            # VLM brand disambiguation
            candidate_pool = [self.db.brand_names[i] for i in sorted_idx[:config.VLM_TOP_CANDIDATES]]
            vlm_result = query_vlm_fallback(img, candidate_pool)
            # VLM independent text extraction (separate from OCR)
            vlm_texts = query_vlm_text_extraction(img)

        # 7. Final Decision & Diagnostic Evaluation
        result = self.fusion.evaluate_decision(
            brand_names=self.db.brand_names,
            final_scores=fused_scores,
            siglip_scores=siglip_scores,
            color_scores=color_scores,
            ocr_scores=ocr_scores,
            detected_ocr_texts=detected_texts,
            view_names_used=view_names,
            vlm_result=vlm_result,
            vlm_texts=vlm_texts,
        )
        result["image_path"] = str(image_path)
        return result


def run_batch(recognizer: LogoRecognizer, image_paths: List[Path], debug: bool = False, as_json: bool = False, show_ocr: bool = False):
    """
    Run recognition across a batch of images and print an aggregated report.
    """
    results = []
    correct_count = 0
    decision_counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}

    if not as_json:
        print("=" * 105)
        print(f"BATCH LOGO RECOGNITION ({len(image_paths)} images)")
        print("=" * 105)
        if show_ocr:
            print(f"{'#':<5} {'Filename':<20} {'Prediction':<13} {'Decision':<9} {'Final':<8} {'Status':<6} {'OCR Texts Found'}")
        else:
            print(f"{'#':<5} {'Filename':<22} {'Prediction':<15} {'Decision':<9} {'Final':<8} {'Margin':<8} {'Status'}")
        print("-" * 105)

    for idx, path in enumerate(image_paths, 1):
        expected_brand = path.stem.lower()
        res = recognizer.predict(path)
        pred = res["prediction"]
        dec = res["decision"]
        score = res["final_score"]
        margin = res["margin"]
        ocr_texts = res.get("ocr_text", [])
        vlm_texts = res.get("vlm_text", [])

        decision_counts[dec] = decision_counts.get(dec, 0) + 1
        is_correct = (pred == expected_brand)
        if is_correct:
            correct_count += 1

        res["is_expected_top1"] = is_correct
        results.append(res)

        if not as_json:
            status = "OK" if is_correct else f"DIFF ({expected_brand})"
            if show_ocr:
                ocr_str = ", ".join(f"'{t}'" for t in ocr_texts) if ocr_texts else "[No text]"
                vlm_str = ", ".join(f"'{t}'" for t in vlm_texts) if vlm_texts else "[N/A]"
                print(f"[{idx:02d}] {path.name:<20} {pred:<13} {dec:<9} {score:<8.4f} {status}")
                print(f"     OCR: {ocr_str}")
                print(f"     VLM: {vlm_str}")
            else:
                print(f"[{idx:02d}] {path.name:<22} {pred:<15} {dec:<9} {score:<8.4f} {margin:<8.4f} [{status}]")

    if as_json:
        print(json.dumps({
            "total": len(image_paths),
            "correct_top1": correct_count,
            "accuracy": round(correct_count / max(len(image_paths), 1), 4),
            "decisions": decision_counts,
            "results": results
        }, indent=2))
        return

    print("=" * 105)
    print("BATCH SUMMARY")
    print("=" * 105)
    print(f"Total processed : {len(image_paths)}")
    print(f"Top-1 Accuracy  : {correct_count}/{len(image_paths)} ({correct_count / max(len(image_paths), 1) * 100:.2f}%)")
    print(f"Decisions       : MATCH={decision_counts.get('MATCH', 0)} | REVIEW={decision_counts.get('REVIEW', 0)} | UNKNOWN={decision_counts.get('UNKNOWN', 0)}")
    
    logos_with_ocr = sum(1 for r in results if r.get("ocr_text"))
    print(f"OCR Extraction  : {logos_with_ocr}/{len(image_paths)} logos contain readable text")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Logo Recognition System")
    parser.add_argument("image", type=str, nargs="?", default=None, help="Path to image file or directory")
    parser.add_argument("--all", action="store_true", help="Run recognition on all 52 reference logos in logos/")
    parser.add_argument("--dir", type=str, default=None, help="Directory containing images to process in batch")
    parser.add_argument("--ocr", action="store_true", help="Display all detected OCR text strings and confidence scores")
    parser.add_argument("--debug", action="store_true", help="Print detailed diagnostic breakdown")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM fallback for ambiguous cases")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    # Determine batch mode vs single image mode
    target_path = None
    is_batch = False

    if args.all:
        target_path = config.LOGOS_DIR
        is_batch = True
    elif args.dir:
        target_path = Path(args.dir)
        is_batch = True
    elif args.image:
        target_path = Path(args.image)
        if target_path.is_dir():
            is_batch = True
    else:
        parser.print_help()
        sys.exit(1)

    if not target_path.exists():
        print(f"Error: Target path not found: {target_path}", file=sys.stderr)
        sys.exit(1)

    recognizer = LogoRecognizer(use_vlm=args.vlm)

    if is_batch:
        # Collect image files
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        if target_path == config.LOGOS_DIR:
            # Load according to brand_list.txt order for consistent 52 logos
            image_paths = []
            for b, fname in recognizer.db.brand_to_filename.items():
                p = config.LOGOS_DIR / fname
                if p.exists():
                    image_paths.append(p)
        else:
            image_paths = sorted([
                p for p in target_path.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ])

        if not image_paths:
            print(f"No valid image files found in {target_path}", file=sys.stderr)
            sys.exit(1)

        run_batch(recognizer, image_paths, debug=args.debug, as_json=args.json, show_ocr=args.ocr)
        return

    # Single image mode
    res = recognizer.predict(target_path, force_vlm=args.vlm)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    # Clean human-readable output
    print("=" * 65)
    print(f"Prediction : {res['prediction']}")
    print(f"Decision   : {res['decision']} ({res['reason']})")
    print(f"Final Score: {res['final_score']:.4f} (Margin: {res['margin']:.4f})")
    print(f"Signals    : SigLIP={res['signals']['siglip']:.4f} | Color={res['signals']['color']:.4f} | OCR={res['signals']['ocr']:.4f}")
    
    # OCR text results
    ocr_details = res.get("ocr_details", [])
    if ocr_details:
        ocr_formatted = [f"'{d['text']}' (conf: {d['confidence']:.2f})" for d in ocr_details]
        print(f"OCR Texts  : {', '.join(ocr_formatted)}")
    elif res.get("ocr_text"):
        print(f"OCR Texts  : {res['ocr_text']}")
    else:
        print("OCR Texts  : [No readable text detected]")

    # VLM text results (independent visual text reading)
    vlm_texts = res.get("vlm_text", [])
    if vlm_texts:
        vlm_formatted = ", ".join(f"'{t}'" for t in vlm_texts)
        print(f"VLM Texts  : {vlm_formatted}")
    elif res["vlm_used"]:
        print("VLM Texts  : [VLM found no text]")

    if res["vlm_used"]:
        print(f"VLM Result : {res['vlm_result']}")
    print("=" * 65)

    if args.debug:
        print("\nTop 5 Candidates:")
        print(f"{'Rank':<5}{'Brand':<18}{'Final':<10}{'SigLIP':<10}{'Color':<10}{'OCR':<10}")
        print("-" * 65)
        for c in res["top_candidates"]:
            print(f"{c['rank']:<5}{c['brand']:<18}{c['final_score']:<10.4f}{c['siglip']:<10.4f}{c['color']:<10.4f}{c['ocr']:<10.4f}")
        print()


if __name__ == "__main__":
    main()
