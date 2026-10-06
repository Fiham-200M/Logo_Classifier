"""
Update forensic_reference_store.pt to include all raw logos from dataset/our_logos/raw_logos.
"""
from pathlib import Path
import torch
import config
from reference_database.reference_store import ReferenceStore
from models.siglip_model import SigLIPModel
from models.dinov2_model import DINOv2Model
from models.ocr_model import OCRModel

def main():
    print("[Updater] Loading reference store...")
    store = ReferenceStore()
    
    raw_dir = config.BASE_DIR / "dataset" / "our_logos" / "raw_logos"
    if not raw_dir.exists():
        print(f"[Updater] raw_logos dir does not exist: {raw_dir}")
        return

    siglip = SigLIPModel()
    dinov2 = DINOv2Model()
    ocr = OCRModel()

    added_count = 0
    for brand in store.brand_names:
        ref = store.references.get(brand)
        if not ref:
            continue

        existing_paths = {str(Path(v.get("path", "")).resolve()).lower() for v in ref.get("variants", []) if v.get("path")}

        for ext in [".png", ".webp", ".jpg", ".jpeg", ".gif"]:
            raw_file = raw_dir / f"{brand}{ext}"
            if raw_file.exists():
                norm_p = str(raw_file.resolve()).lower()
                if norm_p not in existing_paths:
                    print(f" -> Adding raw logo for {brand}: {raw_file.name}")
                    # Reuse cached OCR if available to speed up
                    primary_ocr = ref.get("variants", [{}])[0].get("ocr") if ref.get("variants") else None
                    var_sig = store._extract_file_signature(
                        raw_file, brand, siglip, dinov2, ocr, asset_type="logo", cached_ocr=primary_ocr
                    )
                    if var_sig:
                        ref["variants"].append(var_sig)
                        existing_paths.add(norm_p)
                        added_count += 1

    if added_count > 0:
        print(f"[Updater] Successfully added {added_count} raw logo variants! Saving to {store.cache_path}...")
        store._save_cache()
    else:
        print("[Updater] All raw logos already present in cache.")

if __name__ == "__main__":
    main()
