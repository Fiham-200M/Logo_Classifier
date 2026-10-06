"""
Anti-Phishing / Fake-Brand Detection System — Main CLI.
Usage:
    python main.py path/to/logo.png
    python main.py path/to/logo.png --vlm
    python main.py path/to/logo.png --debug
    python main.py --all
    python main.py --dir path/to/folder
    python main.py --benchmark
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.detection.pipeline import MultiStagePipeline
from src.database.brand_database import BrandDatabase, get_brand_for_favicon_filename


if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


def format_evidence_log(res: Dict[str, Any], image_name: str = "") -> str:
    """Format Phase 8 detailed per-image diagnostic log."""
    top_cands = res.get("top_candidates", [])
    top1 = top_cands[0] if top_cands else {}
    top2 = top_cands[1] if len(top_cands) > 1 else {}

    c1_name = top1.get("brand", "None")
    c2_name = top2.get("brand", "None")

    lines = []
    lines.append(f"\nIMAGE    : {image_name or res.get('image_path', 'unknown')}")
    lines.append("-" * 65)
    lines.append(f"SigLIP   : {c1_name}={top1.get('siglip', 0.0):.4f}  {c2_name}={top2.get('siglip', 0.0):.4f}")
    lines.append(f"Color    : {c1_name}={top1.get('color', 0.0):.4f}  {c2_name}={top2.get('color', 0.0):.4f}")

    ocr_texts = res.get("ocr_text", [])
    ocr_det_str = ", ".join(f'"{t}"' for t in ocr_texts) if ocr_texts else "[No text detected]"
    lines.append(f"OCR      : detected={ocr_det_str} -> {c1_name}={top1.get('ocr', 0.0):.4f}")

    cls_res = res.get("classifier_result")
    if cls_res and cls_res.get("predicted_brand"):
        pred_b = cls_res.get("predicted_brand", "none")
        conf_b = cls_res.get("confidence", 0.0)
        unk_p = cls_res.get("unknown_prob", 0.0)
        lines.append(f"Classif  : pred={pred_b} (conf={conf_b:.4f}) | unknown_prob={unk_p:.4f} -> {c1_name}={top1.get('classifier', 0.0):.4f}")
    elif res.get("classifier_used"):
        lines.append(f"Classif  : [No prediction] -> {c1_name}={top1.get('classifier', 0.0):.4f}")

    vlm_res = res.get("vlm_result")
    vlm_texts = res.get("vlm_text") or []
    stages = res.get("stages_executed", [])

    if "STAGE_3_VLM" in stages or res.get("vlm_used"):
        if vlm_res:
            det_vlm = vlm_res.get("detected_brand") or vlm_res.get("candidate")
            vlm_conf = vlm_res.get("brand_confidence") or vlm_res.get("confidence", 0.0)
            is_comp = vlm_res.get("is_competitor", False)
            vlm_reason = vlm_res.get("reason") or vlm_res.get("reasoning_summary", "")

            if is_comp or det_vlm is None:
                lines.append(f"VLM      : [Competitor/Clone] {vlm_reason or 'No match in protected candidates'}")
            else:
                lines.append(f"VLM      : detected={det_vlm}  confidence={float(vlm_conf):.2f} ({vlm_reason})")
        elif vlm_texts:
            lines.append(f"VLM      : text_read={vlm_texts}")
        else:
            lines.append("VLM      : [Executed - Non-matching or inconclusive]")
    else:
        lines.append("VLM      : [Skipped / Not Triggered]")

    stages_str = " -> ".join(res.get("stages_executed", []))
    lines.append(f"STAGES   : {stages_str}")
    lines.append(f"FINAL    : {res['prediction']}  score={res['final_score']:.4f}  margin={res['margin']:.4f}  decision={res['decision']}")
    lines.append(f"REASON   : {res['reason']}")
    lines.append("-" * 65)
    return "\n".join(lines)


def run_batch(pipeline: MultiStagePipeline, image_paths: List[Path], as_json: bool = False, debug: bool = False):
    results = []
    correct_count = 0
    decision_counts = {"MATCH": 0, "REVIEW": 0, "UNKNOWN": 0}

    if not as_json:
        print("=" * 105)
        print(f"MULTI-STAGE BATCH LOGO / FAVICON RECOGNITION ({len(image_paths)} images)")
        print("=" * 105)
        print(f"{'#':<5} {'Filename':<35} {'Prediction':<15} {'Decision':<9} {'Final':<8} {'Margin':<8} {'FastPath':<9} {'Status'}")
        print("-" * 105)

    for idx, path in enumerate(image_paths, 1):
        expected_brand = get_brand_for_favicon_filename(path.name, pipeline.db.brand_names) or path.stem.lower()
        res = pipeline.process(path)
        pred = res["prediction"]
        dec = res["decision"]
        score = res["final_score"]
        margin = res["margin"]
        fast_path = "YES" if res.get("fast_path_taken") else "NO"

        decision_counts[dec] = decision_counts.get(dec, 0) + 1
        is_correct = (pred == expected_brand)
        if is_correct:
            correct_count += 1

        res["is_expected_top1"] = is_correct
        results.append(res)

        if not as_json:
            status = "OK" if is_correct else f"DIFF ({expected_brand})"
            print(f"[{idx:02d}] {path.name:<35} {pred:<15} {dec:<9} {score:<8.4f} {margin:<8.4f} {fast_path:<9} [{status}]")
            if debug:
                print(format_evidence_log(res, path.name))

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
    fast_count = sum(1 for r in results if r.get("fast_path_taken"))
    print(f"Fast-Path Taken : {fast_count}/{len(image_paths)} ({fast_count / max(len(image_paths), 1) * 100:.1f}%)")
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="Multi-Stage Anti-Phishing Brand & Favicon Detection")
    parser.add_argument("image", type=str, nargs="?", default=None, help="Path to logo or favicon image or folder")
    parser.add_argument("--all", action="store_true", help="Process all reference logos in logos/")
    parser.add_argument("--favicons", action="store_true", help="Process all reference favicons in Favicon/")
    parser.add_argument("--dir", type=str, default=None, help="Process images in directory")
    parser.add_argument("--vlm", action="store_true", help="Enable VLM verification")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR stage")
    parser.add_argument("--no-classifier", action="store_true", help="Skip Brand Classifier stage")
    parser.add_argument("--no-fast-path", action="store_true", help="Disable fast-path visual bypass")
    parser.add_argument("--debug", action="store_true", help="Show full evidence log")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")

    args = parser.parse_args()

    pipeline = MultiStagePipeline(
        use_vlm=args.vlm,
        use_classifier=not args.no_classifier,
    )

    target_path = None
    is_batch = False

    if args.all:
        target_path = config.LOGOS_DIR
        is_batch = True
    elif args.favicons:
        target_path = getattr(config, "FAVICON_DIR", config.BASE_DIR / "Favicon")
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
        print(f"Error: Target path does not exist: {target_path}", file=sys.stderr)
        sys.exit(1)

    if is_batch:
        valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}
        if target_path == config.LOGOS_DIR:
            image_paths = []
            for b, fname in pipeline.db.brand_to_filename.items():
                p = config.LOGOS_DIR / fname
                if p.exists():
                    image_paths.append(p)
        else:
            image_paths = sorted([
                p for p in target_path.iterdir()
                if p.is_file() and p.suffix.lower() in valid_exts
            ])

        run_batch(pipeline, image_paths, as_json=args.json, debug=args.debug)
        return

    # Single image mode
    res = pipeline.process(
        target_path,
        force_vlm=args.vlm,
        skip_ocr=args.no_ocr,
        disable_fast_path=args.no_fast_path,
    )

    if args.json:
        print(json.dumps(res, indent=2))
        return

    # Print human-readable report
    print(format_evidence_log(res, target_path.name))

    if args.debug:
        print("\nTop 5 Candidates:")
        has_cls = any("classifier" in c for c in res.get("top_candidates", []))
        if has_cls:
            print(f"{'Rank':<5}{'Brand':<18}{'Final':<10}{'SigLIP':<10}{'Color':<10}{'OCR':<10}{'Classif':<10}")
            print("-" * 75)
            for c in res.get("top_candidates", []):
                cls_score = c.get("classifier", 0.0)
                print(f"{c['rank']:<5}{c['brand']:<18}{c['final_score']:<10.4f}{c['siglip']:<10.4f}{c['color']:<10.4f}{c['ocr']:<10.4f}{cls_score:<10.4f}")
        else:
            print(f"{'Rank':<5}{'Brand':<18}{'Final':<10}{'SigLIP':<10}{'Color':<10}{'OCR':<10}")
            print("-" * 65)
            for c in res.get("top_candidates", []):
                print(f"{c['rank']:<5}{c['brand']:<18}{c['final_score']:<10.4f}{c['siglip']:<10.4f}{c['color']:<10.4f}{c['ocr']:<10.4f}")
        print()


if __name__ == "__main__":
    main()
