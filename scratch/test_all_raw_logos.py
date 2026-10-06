import sys
from pathlib import Path
import json

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify_logo import LogoForensicsEngine

def main():
    engine = LogoForensicsEngine()
    raw_dir = project_root / "dataset" / "our_logos" / "raw_logos"
    
    files = sorted(list(raw_dir.glob("*.*")))
    print(f"Total raw logo files found: {len(files)}", flush=True)
    
    results = []
    issues = []
    
    for idx, f in enumerate(files, 1):
        if f.suffix.lower() not in [".png", ".webp", ".jpg", ".jpeg", ".gif"]:
            continue
        try:
            res = engine.verify(f, skip_vlm=True)
            v = res["verdict"]
            t = res["threat_type"]
            matched_b = res["brand_id"]
            c_score = res["confidence_score"]
            diffs = res["micro_differences_detected"]
            reason = res["action_reason"]
            metrics = res["forensic_metrics"]
            
            # expected brand from stem
            expected_b = f.stem.lower()
            
            item = {
                "file": f.name,
                "expected_brand": expected_b,
                "matched_brand": matched_b,
                "verdict": v,
                "threat_type": t,
                "siglip": metrics.get("siglip2_semantic_score"),
                "dinov2": metrics.get("dinov2_geometry_score"),
                "delta_e": metrics.get("cielab_delta_e"),
                "edge_iou": metrics.get("edge_stroke_iou"),
                "ocr_match": metrics.get("ocr_text_match"),
                "phash": metrics.get("phash_hamming_distance"),
                "diffs": diffs,
                "reason": reason
            }
            results.append(item)
            
            if v != "MATCH" or matched_b != expected_b:
                issues.append(item)
                print(f"[{idx}/{len(files)}] [ISSUE] {f.name}: Verdict={v} ({t}), Matched={matched_b} (Expected={expected_b}), Reason={reason}, Diffs={diffs}", flush=True)
            else:
                print(f"[{idx}/{len(files)}] [MATCH] {f.name}: Matched={matched_b}, SigLIP={metrics.get('siglip2_semantic_score')}, DINO={metrics.get('dinov2_geometry_score')}, DeltaE={metrics.get('cielab_delta_e')}", flush=True)
        except Exception as e:
            print(f"[{idx}/{len(files)}] [ERROR] {f.name}: {e}", flush=True)
            issues.append({"file": f.name, "error": str(e)})

    print("\n" + "="*50, flush=True)
    print(f"SUMMARY: {len(files) - len(issues)}/{len(files)} MATCHED. {len(issues)} ISSUES.", flush=True)
    print("="*50, flush=True)
    
    with open(project_root / "scratch" / "raw_logos_eval.json", "w", encoding="utf-8") as out_f:
        json.dump({"issues": issues, "all": results}, out_f, indent=2)

if __name__ == "__main__":
    main()
