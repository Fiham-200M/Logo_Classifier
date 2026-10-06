"""
Benchmark & Ablation Study Suite.
Runs systematic evaluations across different modality combinations:
  1. SigLIP only
  2. SigLIP + Color
  3. SigLIP + OCR
  4. SigLIP + Color + OCR (Standard Fusion)
  5. Full Pipeline (+ Conditional VLM)

Outputs comparison reports highlighting accuracy, false negative rates, and margins.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import argparse
import sys

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from src.database.brand_database import BrandDatabase
from src.models.siglip_engine import SigLIPEngine
from src.models.ocr_engine import OCREngine
from src.detection.pipeline import MultiStagePipeline
from src.detection.fusion_engine import ScoreFusion
from src.detection.decision_engine import EvidenceDecisionEngine
from src.evaluation.metrics import compute_evaluation_metrics


def run_ablation_study(
    image_paths: List[Path],
    eval_vlm: bool = False,
) -> Dict[str, Any]:
    """
    Executes all ablation configurations on the given set of images.
    """
    db = BrandDatabase()
    siglip = SigLIPEngine()
    ocr = OCREngine()

    configurations = [
        {"name": "1. SigLIP Only", "siglip_w": 1.0, "color_w": 0.0, "ocr_w": 0.0, "vlm": False, "skip_ocr": True},
        {"name": "2. SigLIP + Color", "siglip_w": 0.70, "color_w": 0.30, "ocr_w": 0.0, "vlm": False, "skip_ocr": True},
        {"name": "3. SigLIP + OCR", "siglip_w": 0.75, "color_w": 0.0, "ocr_w": 0.25, "vlm": False, "skip_ocr": False},
        {"name": "4. SigLIP + Color + OCR", "siglip_w": 0.55, "color_w": 0.25, "ocr_w": 0.20, "vlm": False, "skip_ocr": False},
    ]

    if eval_vlm:
        configurations.append(
            {"name": "5. Full Pipeline (+VLM)", "siglip_w": 0.55, "color_w": 0.25, "ocr_w": 0.20, "vlm": True, "skip_ocr": False}
        )

    summary_table = []

    print("\n" + "=" * 95)
    print(f"STARTING ABLATION BENCHMARK ({len(image_paths)} test images)")
    print("=" * 95)

    for cfg in configurations:
        print(f"\nRunning Configuration: {cfg['name']}...")
        pipeline = MultiStagePipeline(db=db, siglip=siglip, ocr=ocr, use_vlm=cfg["vlm"])
        pipeline.fusion = ScoreFusion(
            siglip_weight=cfg["siglip_w"],
            color_weight=cfg["color_w"],
            ocr_weight=cfg["ocr_w"],
        )

        results = []
        for path in image_paths:
            expected = path.stem.lower()
            res = pipeline.process(
                path,
                force_vlm=cfg["vlm"],
                skip_ocr=cfg["skip_ocr"],
                disable_fast_path=True,  # Disable fast path during ablation for fair comparison
            )
            res["expected_brand"] = expected
            results.append(res)

        metrics = compute_evaluation_metrics(results)
        summary_table.append({
            "Configuration": cfg["name"],
            "Accuracy": f"{metrics['top1_accuracy'] * 100:.1f}%",
            "FN Rate": f"{metrics['false_negative_rate'] * 100:.1f}%",
            "MATCH": metrics["decisions"].get("MATCH", 0),
            "REVIEW": metrics["decisions"].get("REVIEW", 0),
            "UNKNOWN": metrics["decisions"].get("UNKNOWN", 0),
            "Mean Score": f"{metrics['mean_score']:.4f}",
            "Mean Margin": f"{metrics['mean_margin']:.4f}",
        })

    print("\n" + "=" * 95)
    print("ABLATION BENCHMARK RESULTS")
    print("=" * 95)
    header = f"{'Configuration':<26} {'Accuracy':<10} {'FN Rate':<10} {'MATCH':<7} {'REVIEW':<8} {'UNKNOWN':<9} {'Mean Score':<12} {'Mean Margin'}"
    print(header)
    print("-" * 95)
    for row in summary_table:
        print(
            f"{row['Configuration']:<26} {row['Accuracy']:<10} {row['FN Rate']:<10} "
            f"{row['MATCH']:<7} {row['REVIEW']:<8} {row['UNKNOWN']:<9} "
            f"{row['Mean Score']:<12} {row['Mean Margin']}"
        )
    print("=" * 95)

    return {"summary": summary_table}


def main():
    parser = argparse.ArgumentParser(description="Brand Detection Pipeline Benchmark")
    parser.add_argument("--dir", type=str, default=None, help="Directory of evaluation images")
    parser.add_argument("--vlm", action="store_true", help="Include VLM in ablation")
    args = parser.parse_args()

    target_dir = Path(args.dir) if args.dir else config.LOGOS_DIR
    valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
    paths = sorted([p for p in target_dir.iterdir() if p.is_file() and p.suffix.lower() in valid_exts])

    if not paths:
        print(f"No test images found in {target_dir}")
        sys.exit(1)

    run_ablation_study(paths, eval_vlm=args.vlm)


if __name__ == "__main__":
    main()
