import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from verify_logo import LogoForensicsEngine
from PIL import Image

engine = LogoForensicsEngine()
top111_p = project_root / "dataset" / "our_logos" / "raw_logos" / "top111.gif"

res = engine.verify(top111_p, skip_vlm=True)
print("TOP CANDIDATES FOR top111.gif:")
for c in res["candidate_brands"]:
    print(f"  Brand: {c['brand_id']} | SigLIP: {c['siglip']} | pHash: {c['phash_dist']}")

ref_top111 = engine.ref_store.references.get("top111")
print("\ntop111 in ReferenceStore:")
print("  Exists:", ref_top111 is not None)
if ref_top111:
    print("  Variants count:", len(ref_top111.get("variants", [])))
    for v in ref_top111.get("variants", []):
        print("   -", v.get("filename"), v.get("path"))
