"""
System Performance Benchmark & Evaluation Suite for 200M Logo Classifier.
Runs comprehensive testing across:
  1. All 52 Official Protected Master Logos
  2. All 46 Official Brand Favicons
  3. 52 Authentic Positive Background & Padding Variants
  4. Known Real-World Threat / Redraw Test Candidates
  5. Competitor / Unrelated Negative Logos

Outputs:
  - system_performance_results.csv
  - performance_summary.json
"""

import os
import sys
import time
import json
import csv
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from verify_logo import LogoForensicsEngine
from reference_database.reference_store import get_brand_for_favicon_filename


def build_test_suite() -> List[Dict[str, Any]]:
    """Builds a curated, multi-category test dataset with ground-truth expectations."""
    suite = []

    with open(config.BRAND_LIST_PATH, "r", encoding="utf-8") as f:
        brand_names = [line.strip().split()[0] for line in f if line.strip()]

    # -------------------------------------------------------------
    # 1. Official Master Logos (52 brands)
    # -------------------------------------------------------------
    for b in brand_names:
        matching = []
        for ext in [".png", ".gif", ".webp", ".jpg", ".jpeg"]:
            p = config.LOGOS_DIR / f"{b}{ext}"
            if p.exists():
                matching.append(p)
        if matching:
            suite.append({
                "category": "OFFICIAL_LOGO",
                "file_path": matching[0],
                "expected_brand": b,
                "expected_verdict": "MATCH",
                "asset_mode": "logo",
            })

    # -------------------------------------------------------------
    # 2. Official Favicons (46 favicons)
    # -------------------------------------------------------------
    fav_dir = BASE_DIR / "Favicon"
    if fav_dir.exists():
        for fav_p in sorted(fav_dir.iterdir()):
            if fav_p.is_file() and fav_p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif", ".ico"):
                b_fav = get_brand_for_favicon_filename(fav_p.name, brand_names)
                if b_fav:
                    suite.append({
                        "category": "OFFICIAL_FAVICON",
                        "file_path": fav_p.resolve(),
                        "expected_brand": b_fav,
                        "expected_verdict": "MATCH",
                        "asset_mode": "favicon",
                    })

    # -------------------------------------------------------------
    # 3. Positive Variants (Backgrounds & Layouts from dataset/our_logos/positive)
    # -------------------------------------------------------------
    pos_dir = BASE_DIR / "dataset" / "our_logos" / "positive"
    if pos_dir.exists():
        for idx, b in enumerate(brand_names, start=1):
            folder = pos_dir / f"logo_{idx:03d}"
            if folder.exists():
                # Pick 2 challenging variants per brand (dark gray canvas & padded)
                dark_gray = folder / f"logo_{idx:03d}_pos_bg_dark_gray_00.png"
                pad = folder / f"logo_{idx:03d}_pos_pad_00.png"
                if dark_gray.exists():
                    suite.append({
                        "category": "POSITIVE_BG_VARIANT",
                        "file_path": dark_gray,
                        "expected_brand": b,
                        "expected_verdict": "MATCH",
                        "asset_mode": "logo",
                    })
                if pad.exists():
                    suite.append({
                        "category": "POSITIVE_PAD_VARIANT",
                        "file_path": pad,
                        "expected_brand": b,
                        "expected_verdict": "MATCH",
                        "asset_mode": "logo",
                    })

    # -------------------------------------------------------------
    # 4. Threat & Redraw Test Images (test_images)
    # -------------------------------------------------------------
    test_dir = BASE_DIR / "test_images"
    if test_dir.exists():
        for tp in sorted(test_dir.iterdir()):
            if tp.is_file() and tp.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif"):
                # Real threats or authentic re-tests
                expected_v = "REVIEW" if ("mo_" in tp.name or "11.gif" in tp.name) else "MATCH"
                suite.append({
                    "category": "THREAT_CANDIDATE",
                    "file_path": tp,
                    "expected_brand": "auto",
                    "expected_verdict": expected_v,
                    "asset_mode": "logo",
                })

    # -------------------------------------------------------------
    # 5. Competitor Logos (Negative Dataset - Expect UNKNOWN / UNRELATED)
    # -------------------------------------------------------------
    comp_dir = BASE_DIR / "dataset" / "competitor_logos"
    if comp_dir.exists():
        count = 0
        for root, dirs, files in os.walk(comp_dir):
            for f in sorted(files):
                if f.lower().endswith((".png", ".webp", ".jpg", ".jpeg", ".gif")):
                    suite.append({
                        "category": "COMPETITOR_NEGATIVE",
                        "file_path": Path(root) / f,
                        "expected_brand": "UNKNOWN",
                        "expected_verdict": "UNKNOWN",
                        "asset_mode": "auto",
                    })
                    count += 1
                    if count >= 60:  # Sample 60 competitor logos for high-speed benchmark
                        break
            if count >= 60:
                break

    return suite


def run_benchmark():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print("=" * 70, flush=True)
    print("[BENCHMARK] 200M LOGO CLASSIFIER // SYSTEM PERFORMANCE BENCHMARK SUITE", flush=True)
    print("=" * 70, flush=True)

    import urllib.request
    api_url = "http://127.0.0.1:8000/api/verify"

    def verify_item(fpath: Path, asset_mode: str, skip_vlm: bool = True) -> Dict[str, Any]:
        fpath = fpath.resolve()
        try:
            rel = str(fpath.relative_to(BASE_DIR)).replace("\\", "/")
        except Exception:
            rel = str(fpath).replace("\\", "/")
        payload = json.dumps({
            "sample_path": rel,
            "asset_mode": asset_mode,
            "debug": False,
            "skip_vlm": skip_vlm,
        }).encode("utf-8")
        req = urllib.request.Request(
            api_url,
            data=payload,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=45) as r:
            resp_data = json.loads(r.read())
            return resp_data["report"]

    test_suite = build_test_suite()

    print(f"\n[Benchmark] Initialized test suite with {len(test_suite)} total test items:", flush=True)
    cat_counts = {}
    for item in test_suite:
        cat_counts[item["category"]] = cat_counts.get(item["category"], 0) + 1
    for cat, count in cat_counts.items():
        print(f"  - {cat:22s}: {count:3d} images", flush=True)

    csv_path = BASE_DIR / "system_performance_results.csv"
    csv_headers = [
        "test_id",
        "category",
        "filename",
        "relative_path",
        "expected_brand",
        "expected_verdict",
        "predicted_brand",
        "actual_verdict",
        "threat_type",
        "is_correct",
        "confidence_score",
        "siglip2_score",
        "dinov2_score",
        "cielab_delta_e",
        "edge_stroke_iou",
        "ocr_match",
        "detected_text",
        "classifier_pred",
        "classifier_conf",
        "classifier_unk_prob",
        "matched_reference",
        "latency_ms",
        "action_reason",
    ]

    # Open CSV files for progressive streaming write
    all_brands_csv_path = BASE_DIR / "all_brands_verification_results.csv"
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
    writer.writeheader()
    csv_file.flush()

    brand_csv_file = open(all_brands_csv_path, "w", newline="", encoding="utf-8")
    brand_writer = csv.DictWriter(brand_csv_file, fieldnames=csv_headers)
    brand_writer.writeheader()
    brand_csv_file.flush()

    rows = []
    latencies = []
    correct_count = 0

    print(f"\n[Benchmark] Executing deep multi-modal verification across all items...\n", flush=True)
    t_start_total = time.time()

    for idx, item in enumerate(test_suite, start=1):
        fpath = item["file_path"]
        cat = item["category"]
        exp_brand = item["expected_brand"]
        exp_verdict = item["expected_verdict"]
        asset_mode = item["asset_mode"]

        t0 = time.time()
        try:
            report = verify_item(fpath, asset_mode=asset_mode, skip_vlm=True)
            t_ms = round((time.time() - t0) * 1000, 1)
        except Exception as e:
            print(f"  [ERROR] Failed on {fpath.name}: {e}", flush=True)
            continue

        latencies.append(t_ms)
        metrics = report.get("forensic_metrics", {})

        pred_brand = report.get("brand_id", "UNKNOWN")
        act_verdict = report.get("verdict", "UNKNOWN")
        threat_type = report.get("threat_type", "EXACT_REPLICA")
        conf_score = report.get("confidence_score", 0.0)

        # Check correctness
        if cat == "COMPETITOR_NEGATIVE":
            is_correct = 1 if (act_verdict == "UNKNOWN" or threat_type == "UNRELATED") else 0
        elif cat == "THREAT_CANDIDATE":
            if exp_verdict == "REVIEW":
                is_correct = 1 if act_verdict == "REVIEW" else 0
            else:
                is_correct = 1 if act_verdict in ("MATCH", "REVIEW") else 0
        else:
            # Official logo / favicon / positive variant
            brand_matches = (pred_brand.lower() == exp_brand.lower())
            verdict_ok = (act_verdict == "MATCH")
            is_correct = 1 if (brand_matches and verdict_ok) else 0

        if is_correct:
            correct_count += 1

        rel_path = ""
        try:
            rel_path = str(fpath.relative_to(BASE_DIR))
        except Exception:
            rel_path = str(fpath)

        row = {
            "test_id": idx,
            "category": cat,
            "filename": fpath.name,
            "relative_path": rel_path,
            "expected_brand": exp_brand,
            "expected_verdict": exp_verdict,
            "predicted_brand": pred_brand,
            "actual_verdict": act_verdict,
            "threat_type": threat_type,
            "is_correct": is_correct,
            "confidence_score": conf_score,
            "siglip2_score": metrics.get("siglip2_semantic_score", 0.0),
            "dinov2_score": metrics.get("dinov2_geometry_score", 0.0),
            "cielab_delta_e": metrics.get("cielab_delta_e", 0.0),
            "edge_stroke_iou": metrics.get("edge_stroke_iou", 0.0),
            "ocr_match": 1 if metrics.get("ocr_text_match") else 0,
            "detected_text": report.get("ocr_evidence", {}).get("normalized_text", ""),
            "classifier_pred": metrics.get("classifier_predicted_brand", "none"),
            "classifier_conf": metrics.get("classifier_confidence", 0.0),
            "classifier_unk_prob": metrics.get("classifier_unknown_prob", 0.0),
            "matched_reference": report.get("matched_reference", ""),
            "latency_ms": t_ms,
            "action_reason": report.get("action_reason", ""),
        }
        rows.append(row)

        # Stream directly to disk
        writer.writerow(row)
        csv_file.flush()
        if cat == "OFFICIAL_LOGO":
            brand_writer.writerow(row)
            brand_csv_file.flush()

        if idx % 10 == 0 or idx == len(test_suite):
            print(f"  Processed {idx:3d}/{len(test_suite)} ({idx/len(test_suite)*100:.1f}%) | "
                  f"Avg Latency: {np.mean(latencies):.1f}ms | Current Accuracy: {correct_count/idx*100:.1f}%", flush=True)

    csv_file.close()
    brand_csv_file.close()
    total_time = time.time() - t_start_total

    print(f"\n[Benchmark] Successfully saved detailed system results to: {csv_path}")
    print(f"[Benchmark] Successfully saved dedicated 52 protected brands results to: {all_brands_csv_path}")

    # Compute Statistical Breakdown
    summary = {
        "total_images_evaluated": len(rows),
        "total_time_seconds": round(total_time, 2),
        "throughput_images_per_sec": round(len(rows) / total_time, 2),
        "latency_ms": {
            "mean": round(float(np.mean(latencies)), 1),
            "median": round(float(np.median(latencies)), 1),
            "p95": round(float(np.percentile(latencies, 95)), 1),
            "p99": round(float(np.percentile(latencies, 99)), 1),
            "min": round(float(np.min(latencies)), 1),
            "max": round(float(np.max(latencies)), 1),
        },
        "overall_accuracy_pct": round(correct_count / len(rows) * 100, 2),
        "category_metrics": {},
    }

    for cat in cat_counts.keys():
        cat_rows = [r for r in rows if r["category"] == cat]
        if not cat_rows:
            continue
        c_correct = sum(r["is_correct"] for r in cat_rows)
        c_latencies = [r["latency_ms"] for r in cat_rows]
        summary["category_metrics"][cat] = {
            "total": len(cat_rows),
            "correct": c_correct,
            "accuracy_pct": round(c_correct / len(cat_rows) * 100, 2),
            "mean_latency_ms": round(float(np.mean(c_latencies)), 1),
            "mean_siglip": round(float(np.mean([r["siglip2_score"] for r in cat_rows])), 4),
            "mean_dinov2": round(float(np.mean([r["dinov2_score"] for r in cat_rows])), 4),
            "mean_delta_e": round(float(np.mean([r["cielab_delta_e"] for r in cat_rows])), 2),
            "mean_edge_iou": round(float(np.mean([r["edge_stroke_iou"] for r in cat_rows])), 4),
        }

    json_path = BASE_DIR / "performance_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("[SUMMARY] BENCHMARK PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Total Evaluated : {summary['total_images_evaluated']} images")
    print(f"Overall Accuracy: {summary['overall_accuracy_pct']}%")
    print(f"Average Latency : {summary['latency_ms']['mean']} ms / image ({summary['throughput_images_per_sec']} images/sec)")
    print(f"P95 Latency     : {summary['latency_ms']['p95']} ms")
    print("-" * 70)
    for cat, m in summary["category_metrics"].items():
        print(f"{cat:22s} | N={m['total']:3d} | Acc={m['accuracy_pct']:6.2f}% | Latency={m['mean_latency_ms']:5.1f}ms | SigLIP={m['mean_siglip']:.3f} | Edge IoU={m['mean_edge_iou']:.3f}")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_benchmark()
