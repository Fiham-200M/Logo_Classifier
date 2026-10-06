import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify_logo import LogoForensicsEngine

engine = LogoForensicsEngine()
test_files = [
    "dataset/our_logos/raw_logos/jagoledak.webp",
    "dataset/our_logos/raw_logos/paduka500.png",
    "dataset/our_logos/raw_logos/sgcwin88.webp",
    "dataset/our_logos/raw_logos/super89.png",
]

all_passed = True
for tf in test_files:
    res = engine.verify(tf, skip_vlm=True)
    v = res["verdict"]
    t = res["threat_type"]
    b = res["brand_id"]
    ref = res["matched_reference"]
    diffs = res["micro_differences_detected"]
    m = res["forensic_metrics"]
    
    status = "PASS" if v == "MATCH" else "FAIL"
    if v != "MATCH":
        all_passed = False
    print(f"[{status}] {tf}")
    print(f"       Verdict: {v} | Threat: {t} | Brand: {b} | Matched Ref: {ref}")
    print(f"       SigLIP: {m['siglip2_semantic_score']} | DINO: {m['dinov2_geometry_score']} | DeltaE: {m['cielab_delta_e']} | EdgeIoU: {m['edge_stroke_iou']} | pHash: {m['phash_hamming_distance']}")
    print(f"       Diffs: {diffs}")

if all_passed:
    print("\nALL 4 TEST CASES PASSED AS MATCH!")
else:
    print("\nSOME CASES STILL FAILED!")
