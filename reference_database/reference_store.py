"""
Forensic Reference Database Store.
Stores precomputed forensic signatures for all 52 official protected brands:
  - Canvases (white, black, alpha mask, normalized RGB)
  - Perceptual Hashes (pHash, dHash)
  - SigLIP 2 Embeddings (white and black renders)
  - DINOv2 Structural Embeddings
  - Color Profiles (HSV 3D histograms, dominant LAB colors, mean RGB/LAB)
  - OCR Text & Slogans
  - Canny Edge Maps
  - Metadata
Features are cached to disk and loaded into memory on startup.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch
import numpy as np
from PIL import Image

import config
from preprocessing.media_normalizer import MediaNormalizer
from forensics.perceptual_hash import PerceptualHasher
from forensics.color_analysis import ColorAnalyzer
from forensics.edge_analysis import EdgeAnalyzer
from models.siglip_model import SigLIPModel
from models.dinov2_model import DINOv2Model
from models.ocr_model import OCRModel


def get_brand_for_favicon_filename(filename: str, brand_names: List[str]) -> Optional[str]:
    """Map a favicon filename (e.g. '001_a200m-seru.fun_favicon.png') to protected brand name."""
    fname = filename.lower()
    if "003_image" in fname or "kalanganasia" in fname or fname == "asia200.png":
        return "asia200"
    if "008_image" in fname or fname == "c200m.png":
        return "c200m"
    if "fufuremix" in fname:
        return "fufuslot"
    if "tertop111" in fname or "top111" in fname:
        return "top111"
    if "tri88" in fname:
        return "tri88"
    for b in sorted(brand_names, key=len, reverse=True):
        if b in fname:
            return b
    return None


class ReferenceStore:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ReferenceStore, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        cache_path: Optional[Path] = None,
        logos_dir: Path = config.LOGOS_DIR,
        favicons_dir: Path = getattr(config, "FAVICON_DIR", config.BASE_DIR / "Favicon"),
        brand_list_path: Path = config.BRAND_LIST_PATH,
    ):
        if self._initialized:
            return

        self.logos_dir = logos_dir
        self.favicons_dir = favicons_dir
        self.brand_list_path = brand_list_path
        self.cache_path = cache_path or getattr(config, "FORENSIC_REF_STORE_PATH", config.REFERENCE_EMBEDDINGS_DIR / "forensic_reference_store.pt")

        self.brand_names: List[str] = []
        self.brand_to_filename: Dict[str, str] = {}
        self.references: Dict[str, Dict[str, Any]] = {}

        self._load_brand_list()
        self.load_or_build()
        self._initialized = True

    def _load_brand_list(self):
        """Loads canonical 52 brands and file mappings."""
        if self.brand_list_path.exists():
            with open(self.brand_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        brand, fname = parts[0], parts[1]
                        self.brand_names.append(brand)
                        self.brand_to_filename[brand] = fname

    def _save_cache(self, siglip_model: Optional[SigLIPModel] = None):
        """Saves current reference dictionary and mappings to disk with comprehensive metadata."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        from datetime import datetime
        active_model_id = getattr(siglip_model, "model_name", config.SIGLIP_MODEL_NAME)
        metadata = {
            "model_id": active_model_id,
            "model_family": "SigLIP 2",
            "image_size": getattr(siglip_model, "input_size", 384),
            "patch_size": getattr(siglip_model, "patch_size", 14),
            "embedding_dim": getattr(siglip_model, "embedding_dim", 1152),
            "brand_count": len(self.references),
            "creation_timestamp": datetime.now().isoformat(),
            "preprocessing_version": "2.0",
        }
        payload = {
            "metadata": metadata,
            "model_id": metadata["model_id"],
            "embedding_dim": metadata["embedding_dim"],
            "brand_names": self.brand_names,
            "brand_to_filename": self.brand_to_filename,
            "references": self.references,
            "num_brands": len(self.references),
        }
        torch.save(payload, self.cache_path)
        print(f"[ReferenceStore] Saved {len(self.references)} forensic brand references (model={metadata['model_id']}, dim={metadata['embedding_dim']}) to {self.cache_path}")

    def _extract_file_signature(
        self,
        ref_path: Path,
        brand: str,
        siglip: SigLIPModel,
        dinov2: DINOv2Model,
        ocr: OCRModel,
        asset_type: str = "logo",
        cached_ocr: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extracts complete forensic signatures for a single file asset."""
        if not ref_path or not ref_path.exists():
            return None

        # 1. Media Normalization
        norm = MediaNormalizer.normalize_media(ref_path)
        c_white = norm["canvas_white"]
        c_black = norm["canvas_black"]
        a_mask = norm["alpha_mask"]
        rgb_norm = norm["normalized_rgb"]

        # 2. Perceptual Hashes
        hashes = PerceptualHasher.extract_hashes(rgb_norm)

        # 3. SigLIP 2 Embeddings
        emb_white = siglip.extract_embedding(c_white)
        emb_black = siglip.extract_embedding(c_black)
        emb_norm = siglip.extract_embedding(rgb_norm)

        # 4. DINOv2 Structural Embeddings
        dino_white = dinov2.extract_embedding(c_white)
        dino_black = dinov2.extract_embedding(c_black)
        dino_norm = dinov2.extract_embedding(rgb_norm)

        # 5. Color Profile
        color_prof = ColorAnalyzer.extract_color_profile(rgb_norm, alpha_mask=a_mask)

        # 6. OCR Text
        if cached_ocr is not None:
            ocr_info = cached_ocr
        else:
            key_frames = norm.get("key_frames", [])
            ocr_info = ocr.extract_text_with_frames(rgb_norm, key_frames=key_frames)

        # 7. Edge Map
        edge_map = EdgeAnalyzer.extract_edges(rgb_norm)

        return {
            "brand_id": brand,
            "asset_type": asset_type,
            "filename": ref_path.name,
            "path": str(ref_path),
            "metadata": norm["metadata"],
            "hashes": hashes,
            "siglip_embeddings": {
                "white": emb_white,
                "black": emb_black,
                "normalized": emb_norm,
            },
            "dinov2_embeddings": {
                "white": dino_white,
                "black": dino_black,
                "normalized": dino_norm,
            },
            "color_profile": color_prof,
            "ocr": ocr_info,
            "edge_map": edge_map,
            "image_white": c_white,
            "image_black": c_black,
            "image_normalized": rgb_norm,
        }

    def _extract_brand_signature(
        self,
        brand: str,
        siglip: SigLIPModel,
        dinov2: DINOv2Model,
        ocr: OCRModel,
    ) -> Optional[Dict[str, Any]]:
        """Extracts complete forensic signatures for a brand, including all file variants (Logos, Favicons, and authentic positive variants)."""
        matching_logo_files: List[Path] = []
        fname = self.brand_to_filename.get(brand)
        if fname and (self.logos_dir / fname).exists():
            matching_logo_files.append(self.logos_dir / fname)

        for ext in [".gif", ".png", ".webp", ".jpg", ".jpeg"]:
            alt = self.logos_dir / f"{brand}{ext}"
            if alt.exists() and alt not in matching_logo_files:
                matching_logo_files.append(alt)

        matching_fav_files: List[Path] = []
        if self.favicons_dir and self.favicons_dir.exists():
            for fav_p in self.favicons_dir.iterdir():
                if fav_p.is_file() and fav_p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif", ".ico"):
                    b_fav = get_brand_for_favicon_filename(fav_p.name, self.brand_names)
                    if b_fav == brand and fav_p not in matching_fav_files:
                        matching_fav_files.append(fav_p)

        # Check for positive dataset variants
        positive_variant_files: List[Path] = []
        try:
            brand_idx = self.brand_names.index(brand) + 1
            pos_brand_dir = config.BASE_DIR / "dataset" / "our_logos" / "positive" / f"logo_{brand_idx:03d}"
            orig_brand_dir = config.BASE_DIR / "dataset" / "our_logos" / "originals" / f"logo_{brand_idx:03d}"
            raw_dir = config.BASE_DIR / "dataset" / "our_logos" / "raw_logos"
            anim_dir = config.BASE_DIR / "dataset" / "our_logos" / "animated" / f"logo_{brand_idx:03d}"

            # Raw logos source
            if raw_dir.exists():
                for ext in [".png", ".webp", ".jpg", ".jpeg", ".gif"]:
                    raw_file = raw_dir / f"{brand}{ext}"
                    if raw_file.exists() and raw_file not in matching_logo_files and raw_file not in positive_variant_files:
                        positive_variant_files.append(raw_file)

            # Canonical originals (all extensions)
            if orig_brand_dir.exists():
                for orig_file in orig_brand_dir.glob("original.*"):
                    if orig_file.is_file() and orig_file not in matching_logo_files and orig_file not in positive_variant_files:
                        positive_variant_files.append(orig_file)

            # Animated frames (sample up to 4 keyframes for animated logos)
            if anim_dir.exists():
                anim_frames = sorted([f for f in anim_dir.iterdir() if f.is_file() and f.suffix.lower() == ".png"])
                if anim_frames:
                    step = max(1, len(anim_frames) // 4)
                    for frame_f in anim_frames[::step][:4]:
                        if frame_f not in matching_logo_files and frame_f not in positive_variant_files:
                            positive_variant_files.append(frame_f)

            if pos_brand_dir.exists():
                key_variant_patterns = [
                    "pos_bg_white",
                    "pos_bg_black",
                    "pos_bg_dark_gray",
                    "pos_bg_light_gray",
                    "pos_bg_transparent",
                    "pos_pad_00",
                    "pos_pad_01",
                    "pos_crop_00",
                ]
                for p_file in sorted(pos_brand_dir.iterdir()):
                    if p_file.is_file() and any(k in p_file.name for k in key_variant_patterns):
                        if p_file not in matching_logo_files and p_file not in positive_variant_files:
                            positive_variant_files.append(p_file)
        except Exception as e:
            print(f"[ReferenceStore] Error discovering positive variants for {brand}: {e}")

        if not matching_logo_files and not matching_fav_files and not positive_variant_files:
            print(f"[ReferenceStore] Warning: Reference file not found for {brand}: {fname}")
            return None

        variants: List[Dict[str, Any]] = []
        primary_ocr: Optional[Dict[str, Any]] = None

        # 1. Primary and logo files
        for fpath in matching_logo_files:
            var_sig = self._extract_file_signature(fpath, brand, siglip, dinov2, ocr, asset_type="logo", cached_ocr=primary_ocr)
            if var_sig:
                if primary_ocr is None:
                    primary_ocr = var_sig.get("ocr")
                variants.append(var_sig)

        # 2. Favicons
        for fpath in matching_fav_files:
            fav_sig = self._extract_file_signature(fpath, brand, siglip, dinov2, ocr, asset_type="favicon")
            if fav_sig:
                variants.append(fav_sig)

        # 3. Positive variants (reuse primary_ocr to avoid redundant OCR computation)
        for fpath in positive_variant_files:
            pos_sig = self._extract_file_signature(fpath, brand, siglip, dinov2, ocr, asset_type="logo", cached_ocr=primary_ocr)
            if pos_sig:
                variants.append(pos_sig)

        if not variants:
            return None

        primary = dict(variants[0])
        primary["variants"] = variants
        return primary

    def load_or_build(self):
        """Loads cached forensic database from disk, or builds it if absent."""
        if self.cache_path.exists():
            try:
                print(f"[ReferenceStore] Loading forensic database from {self.cache_path}...")
                cached = torch.load(self.cache_path, map_location="cpu", weights_only=False)
                active_model_id = config.SIGLIP_MODEL_NAME
                cached_metadata = cached.get("metadata", {})
                cached_model_id = cached_metadata.get("model_id") or cached.get("model_id")

                # Model version validation (Section 14)
                if cached_model_id and cached_model_id != active_model_id:
                    error_msg = (
                        f"\n" + "=" * 60 + "\n"
                        f"ERROR: Model version mismatch in reference database!\n"
                        f"Reference embeddings were generated using:\n"
                        f"  {cached_model_id}\n\n"
                        f"Active model is:\n"
                        f"  {active_model_id}\n\n"
                        f"Please rebuild the reference embedding database using:\n"
                        f"  python build_siglip_reference_database.py\n"
                        f"=" * 60 + "\n"
                    )
                    raise ValueError(error_msg)

                self.references = cached.get("references", {})
                self.brand_names = cached.get("brand_names", self.brand_names)
                self.brand_to_filename = cached.get("brand_to_filename", self.brand_to_filename)

                # Validate embedding dimensions
                if self.references:
                    first_brand = next(iter(self.references.values()))
                    first_var = first_brand.get("variants", [first_brand])[0]
                    first_emb = first_var.get("siglip_embeddings", {}).get("normalized")
                    if first_emb is not None:
                        actual_dim = first_emb.shape[-1]
                        expected_dim = 1152 if "so400m" in active_model_id.lower() else 768
                        if actual_dim != expected_dim:
                            raise ValueError(
                                f"\n" + "=" * 60 + "\n"
                                f"ERROR: Embedding dimension mismatch!\n"
                                f"Database has {actual_dim}-dim embeddings, but active model '{active_model_id}' "
                                f"expects {expected_dim}-dim embeddings.\n"
                                f"Please rebuild the reference embedding database.\n"
                                f"=" * 60 + "\n"
                            )

                # Self-healing: verify all brands have valid signatures & all variants (logos + favicons + positive variants)
                corrupted = []
                for brand in self.brand_names:
                    ref = self.references.get(brand)
                    matching_logos = [ext for ext in [".gif", ".png", ".webp", ".jpg", ".jpeg"] if (self.logos_dir / f"{brand}{ext}").exists()]
                    matching_favs = []
                    if self.favicons_dir and self.favicons_dir.exists():
                        matching_favs = [p.name for p in self.favicons_dir.iterdir() if p.is_file() and get_brand_for_favicon_filename(p.name, self.brand_names) == brand]
                    total_expected = len(matching_logos) + len(matching_favs) + 5
                    cached_vars = ref.get("variants", []) if ref else []
                    cached_fav_names = {v.get("filename") for v in cached_vars if v.get("asset_type") == "favicon"}
                    missing_favs = [f for f in matching_favs if f not in cached_fav_names]
                    if not ref or ref.get("hashes", {}).get("phash") == "0000000000000000" or ref.get("color_profile") is None or len(cached_vars) < total_expected or missing_favs:
                        corrupted.append(brand)

                if corrupted:
                    print(f"[ReferenceStore] Detected missing/favicon/positive references for: {len(corrupted)} brands. Updating signatures...")
                    siglip = SigLIPModel()
                    dinov2 = DINOv2Model()
                    ocr = OCRModel()
                    for brand in corrupted:
                        sig = self._extract_brand_signature(brand, siglip, dinov2, ocr)
                        if sig:
                            self.references[brand] = sig
                    self._save_cache(siglip_model=siglip)

                print(f"[ReferenceStore] Successfully loaded {len(self.references)} protected brands for {active_model_id}.")
                return
            except ValueError:
                raise
            except Exception as e:
                print(f"[ReferenceStore] Cache read error ({e}), rebuilding store...")

        self.build_store()

    def build_store(self):
        """Builds and caches all forensic signatures for all 52 official reference logos."""
        print(f"[ReferenceStore] Building forensic reference store for 52 protected brands ({config.SIGLIP_MODEL_NAME})...")
        siglip = SigLIPModel()
        dinov2 = DINOv2Model()
        ocr = OCRModel()

        self.references = {}

        for idx, brand in enumerate(self.brand_names, 1):
            print(f"[{idx:02d}/{len(self.brand_names)}] Extracting signatures for: {brand}...")
            sig = self._extract_brand_signature(brand, siglip, dinov2, ocr)
            if sig:
                self.references[brand] = sig

        self._save_cache(siglip_model=siglip)

    def get_reference(self, brand: str) -> Optional[Dict[str, Any]]:
        """Retrieves cached forensic reference for a specific brand."""
        return self.references.get(brand)
