"""
Unified Database Management Module.
Integrates SigLIP embeddings, color features, and OCR text databases into a cohesive,
multi-reference aware store.
Supports multiple reference embeddings per brand for robust matching against
different logo variants (e.g. dark, light, compact, horizontal).
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import torch
import numpy as np

import sys
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


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


class BrandDatabase:
    def __init__(
        self,
        brand_db_path: Path = config.BRAND_DB_PATH,
        color_db_path: Path = config.COLOR_DB_PATH,
        ocr_db_path: Path = config.OCR_DB_PATH,
        brand_list_path: Path = config.BRAND_LIST_PATH,
        multi_ref_db_path: Optional[Path] = None,
    ):
        self.brand_db_path = brand_db_path
        self.color_db_path = color_db_path
        self.ocr_db_path = ocr_db_path
        self.brand_list_path = brand_list_path
        self.multi_ref_db_path = multi_ref_db_path or getattr(config, "BRAND_MULTI_DB_PATH", config.REFERENCE_EMBEDDINGS_DIR / "brand_multi_database.pt")

        self.brand_names: List[str] = []
        self.brand_to_idx: Dict[str, int] = {}
        self.brand_to_filename: Dict[str, str] = {}

        # Brand -> list of reference image paths (logos + variants + favicons)
        self.brand_to_ref_paths: Dict[str, List[Path]] = {}
        self.brand_to_favicon_paths: Dict[str, List[Path]] = {}

        # Tensor of primary reference embeddings (N, 768)
        self.reference_embeddings: Optional[np.ndarray] = None

        # Dict[brand, np.ndarray of shape (K, 768)] for multi-reference matching
        self.brand_multi_embeddings: Dict[str, np.ndarray] = {}

        # Dict[brand, Union[dict, List[dict]]] of color feature arrays
        self.color_database: Dict[str, Any] = {}

        # Dict[brand, Dict of OCR texts and detections]
        self.ocr_database: Dict[str, dict] = {}

        self.load()

    def load(self):
        """Load all reference databases from disk."""
        # 1. Load brand list mapping
        if self.brand_list_path.exists():
            with open(self.brand_list_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        brand, fname = parts[0], parts[1]
                        self.brand_names.append(brand)
                        self.brand_to_filename[brand] = fname

        for idx, brand in enumerate(self.brand_names):
            self.brand_to_idx[brand] = idx

        # 2. Load SigLIP primary embeddings
        if self.brand_db_path.exists():
            siglip_data = torch.load(self.brand_db_path, map_location="cpu", weights_only=False)
            emb_tensor = siglip_data["embeddings"]
            if isinstance(emb_tensor, torch.Tensor):
                self.reference_embeddings = emb_tensor.numpy().astype(np.float32)
            else:
                self.reference_embeddings = np.array(emb_tensor, dtype=np.float32)

            # Ensure normalized
            norms = np.linalg.norm(self.reference_embeddings, axis=-1, keepdims=True) + 1e-12
            self.reference_embeddings = self.reference_embeddings / norms

            # Synchronize brand names if not loaded from text
            if not self.brand_names and "brand_names" in siglip_data:
                self.brand_names = list(siglip_data["brand_names"])
                self.brand_to_idx = {b: i for i, b in enumerate(self.brand_names)}

        # 3. Load Color Database
        if self.color_db_path.exists():
            color_data = torch.load(self.color_db_path, map_location="cpu", weights_only=False)
            raw_db = color_data.get("database", {})
            for brand, entry in raw_db.items():
                if isinstance(entry, list):
                    parsed_list = []
                    for item in entry:
                        parsed_list.append({
                            k: (v.numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32))
                            for k, v in item.items()
                            if k in ("feature", "hsv", "rgb", "stats", "spatial", "dominant")
                        })
                    self.color_database[brand] = parsed_list
                elif isinstance(entry, dict):
                    self.color_database[brand] = {
                        k: (v.numpy() if isinstance(v, torch.Tensor) else np.array(v, dtype=np.float32))
                        for k, v in entry.items()
                        if k in ("feature", "hsv", "rgb", "stats", "spatial", "dominant")
                    }

        # 4. Load OCR Database
        if self.ocr_db_path.exists():
            ocr_data = torch.load(self.ocr_db_path, map_location="cpu", weights_only=False)
            self.ocr_database = ocr_data.get("database", {})

        # 5. Populate reference paths (logos, variants, and favicons)
        favicon_dir = getattr(config, "FAVICON_DIR", None)
        favicon_files = []
        if favicon_dir and favicon_dir.exists():
            favicon_files = [p for p in favicon_dir.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif", ".bmp")]

        for i, brand in enumerate(self.brand_names, 1):
            paths = []
            fav_paths = []

            # Main logo
            fname = self.brand_to_filename.get(brand, f"{brand}.png")
            direct_path = config.LOGOS_DIR / fname
            if direct_path.exists():
                paths.append(direct_path)

            # Variants in logo_XXX
            subfolder = config.LOGOS_DIR / f"logo_{i:03d}"
            if subfolder.exists():
                for subfile in subfolder.iterdir():
                    if subfile.is_file() and subfile.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif"):
                        if subfile not in paths:
                            paths.append(subfile)

            # Variants in dataset/our_logos (raw_logos, originals, positive, animated)
            our_logos_dir = getattr(config, "OUR_LOGOS_DIR", config.BASE_DIR / "dataset" / "our_logos")
            raw_dir = our_logos_dir / "raw_logos"
            if raw_dir.exists():
                for ext in [".png", ".webp", ".jpg", ".jpeg", ".gif"]:
                    raw_f = raw_dir / f"{brand}{ext}"
                    if raw_f.exists() and raw_f not in paths:
                        paths.append(raw_f)

            for sub_cat in ["originals", "positive", "animated"]:
                ds_sub = our_logos_dir / sub_cat / f"logo_{i:03d}"
                if ds_sub.exists():
                    for subfile in ds_sub.iterdir():
                        if subfile.is_file() and subfile.suffix.lower() in (".png", ".webp", ".jpg", ".jpeg", ".gif"):
                            if subfile not in paths:
                                paths.append(subfile)

            # Match favicons for this brand
            for fav_p in favicon_files:
                matched_b = get_brand_for_favicon_filename(fav_p.name, self.brand_names)
                if matched_b == brand:
                    fav_paths.append(fav_p)
                    if fav_p not in paths:
                        paths.append(fav_p)

            self.brand_to_ref_paths[brand] = paths
            self.brand_to_favicon_paths[brand] = fav_paths

        # 6. Load pre-computed multi-reference database if present
        if self.multi_ref_db_path and self.multi_ref_db_path.exists():
            try:
                multi_data = torch.load(self.multi_ref_db_path, map_location="cpu", weights_only=False)
                raw_multi = multi_data.get("multi_embeddings", {})
                for brand, emb in raw_multi.items():
                    if isinstance(emb, torch.Tensor):
                        arr = emb.numpy().astype(np.float32)
                    else:
                        arr = np.array(emb, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[np.newaxis, :]
                    norms = np.linalg.norm(arr, axis=-1, keepdims=True) + 1e-12
                    self.brand_multi_embeddings[brand] = arr / norms
            except Exception as e:
                print(f"[BrandDatabase] Warning loading multi-ref DB: {e}")

        # Fallback multi-reference embeddings from primary if not loaded
        if not self.brand_multi_embeddings and self.reference_embeddings is not None:
            for idx, brand in enumerate(self.brand_names):
                if idx < len(self.reference_embeddings):
                    self.brand_multi_embeddings[brand] = self.reference_embeddings[idx : idx + 1]

        total_favs = sum(len(fps) for fps in self.brand_to_favicon_paths.values())
        print(
            f"[BrandDatabase] Loaded {len(self.brand_names)} brands "
            f"(SigLIP embeddings: {self.reference_embeddings.shape if self.reference_embeddings is not None else None}, "
            f"Multi-ref brands: {len(self.brand_multi_embeddings)}, "
            f"Favicons mapped: {total_favs}, "
            f"Color entries: {len(self.color_database)}, OCR entries: {len(self.ocr_database)})"
        )

    def get_brand_name(self, index: int) -> str:
        return self.brand_names[index]

    def get_index(self, brand_name: str) -> Optional[int]:
        return self.brand_to_idx.get(brand_name)

    def score_siglip_multi_ref(
        self,
        view_embeddings: Dict[str, np.ndarray],
        strategy: str = "max",
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Score query view embeddings against all brand reference embeddings (including variants).
        For each brand, computes the maximum similarity across all its reference embeddings.
        Returns:
            final_scores: (N,) array of float similarities in [0, 1]
            view_scores: Dict[view_name, (N,) array]
        """
        num_brands = len(self.brand_names)
        if num_brands == 0 or not view_embeddings:
            return np.zeros(num_brands, dtype=np.float32), {}

        view_scores: Dict[str, np.ndarray] = {}

        for view_name, q_emb in view_embeddings.items():
            scores_for_view = np.zeros(num_brands, dtype=np.float32)
            for idx, brand in enumerate(self.brand_names):
                ref_embs = self.brand_multi_embeddings.get(brand)
                if ref_embs is not None and len(ref_embs) > 0:
                    sims = np.dot(ref_embs, q_emb)
                    scores_for_view[idx] = float(np.max(sims))
                elif self.reference_embeddings is not None and idx < len(self.reference_embeddings):
                    sim = float(np.dot(self.reference_embeddings[idx], q_emb))
                    scores_for_view[idx] = sim
            view_scores[view_name] = np.clip(scores_for_view, 0.0, 1.0)

        # Ensembling across views
        view_matrix = np.stack(list(view_scores.values()), axis=0)  # Shape (V, N)
        view_names = list(view_scores.keys())

        if strategy == "max":
            final_scores = np.max(view_matrix, axis=0)
        elif strategy == "weighted_top_k":
            top_k = min(config.SIGLIP_TOP_K_VIEWS, view_matrix.shape[0])
            sorted_view_indices = np.argsort(view_matrix, axis=0)[::-1, :]
            final_scores = np.zeros(num_brands, dtype=np.float32)
            for b_idx in range(num_brands):
                brand_top_indices = sorted_view_indices[:top_k, b_idx]
                weights = [config.VIEW_WEIGHTS.get(view_names[vi], 1.0) for vi in brand_top_indices]
                total_w = sum(weights) + 1e-12
                sims = view_matrix[brand_top_indices, b_idx]
                final_scores[b_idx] = np.sum(sims * np.array(weights)) / total_w
        elif strategy == "mean_top_k":
            top_k = min(config.SIGLIP_TOP_K_VIEWS, view_matrix.shape[0])
            sorted_views = np.sort(view_matrix, axis=0)[::-1, :]
            final_scores = np.mean(sorted_views[:top_k, :], axis=0)
        else:
            final_scores = np.max(view_matrix, axis=0)

        return np.clip(final_scores, 0.0, 1.0).astype(np.float32), view_scores

    def score_color_multi_ref(self, query_color: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Score query color features against brand color database.
        Supports single or multiple reference color profiles per brand (e.g. logo vs favicon).
        Returns:
            scores: (N,) float32 array in [0, 1]
        """
        from src.models.color_engine import compare_color_features
        num_brands = len(self.brand_names)
        scores = np.full(num_brands, 0.50, dtype=np.float32)

        if query_color is None or not self.color_database:
            return scores

        for idx, brand in enumerate(self.brand_names):
            ref_entry = self.color_database.get(brand)
            if ref_entry is None:
                continue

            if isinstance(ref_entry, list):
                sims = [compare_color_features(query_color, r) for r in ref_entry if r is not None]
                scores[idx] = max(sims) if sims else 0.50
            elif isinstance(ref_entry, dict):
                scores[idx] = compare_color_features(query_color, ref_entry)

        return np.clip(scores, 0.0, 1.0).astype(np.float32)

