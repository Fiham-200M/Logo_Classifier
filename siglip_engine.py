"""
SigLIP 2 Feature Extraction and Multi-View Ensembling Engine.
Uses google/siglip2-base-patch16-224 to produce normalized visual embeddings
and calculate robust multi-view visual similarity scores.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel

import config
from preprocessing import generate_preprocessing_views, load_image_safe


class SigLIPEngine:
    def __init__(self, model_name: str = config.SIGLIP_MODEL_NAME, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name

        print(f"[SigLIPEngine] Loading processor & model: {model_name} on {self.device}...")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extract normalized 768-dim SigLIP embedding for a single PIL image.
        Returns float32 numpy array of shape (768,).
        """
        rgb_img = image.convert("RGB")
        inputs = self.processor(images=rgb_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # MUST match build_brand_database.py: use vision_model().pooler_output
        # NOT get_image_features() which applies a projection head and produces
        # embeddings in a different space than the reference database.
        vision_output = self.model.vision_model(pixel_values=inputs["pixel_values"])
        if hasattr(vision_output, "pooler_output") and vision_output.pooler_output is not None:
            features = vision_output.pooler_output
        else:
            features = vision_output.last_hidden_state[:, 0]

        features = features.float()
        features = features / (features.norm(dim=-1, keepdim=True) + 1e-12)
        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_multi_view_embeddings(self, views: Dict[str, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Extract normalized SigLIP embeddings for a dictionary of preprocessed views.
        """
        embeddings = {}
        for view_name in config.PREPROCESSING_VIEWS:
            if view_name in views:
                embeddings[view_name] = self.extract_embedding(views[view_name])
        return embeddings

    def score_against_references(
        self,
        query_image_or_views: Union[Image.Image, Dict[str, Image.Image]],
        reference_embeddings: np.ndarray,
        strategy: str = config.SIGLIP_ENSEMBLE_STRATEGY,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Compare query image views against reference embeddings.

        Args:
            query_image_or_views: Single PIL Image or Dict of preprocessed view images
            reference_embeddings: Array of shape (N, 768) containing normalized reference vectors
            strategy: 'max', 'weighted_top_k', or 'mean_top_k'

        Returns:
            final_scores: Array of shape (N,) visual similarities in range [0, 1]
            view_scores: Dict[view_name, Array of shape (N,)] per-view similarity scores
        """
        if isinstance(query_image_or_views, dict):
            views = query_image_or_views
        else:
            views = generate_preprocessing_views(query_image_or_views)

        view_embeddings = self.extract_multi_view_embeddings(views)
        ref_norm = reference_embeddings / (np.linalg.norm(reference_embeddings, axis=-1, keepdims=True) + 1e-12)

        view_scores = {}
        for view_name, emb in view_embeddings.items():
            sims = np.dot(ref_norm, emb)  # cosine similarity in [-1, 1]
            # Clip negative similarities to 0 for score normalization
            view_scores[view_name] = np.clip(sims, 0.0, 1.0)

        num_refs = ref_norm.shape[0]
        final_scores = np.zeros(num_refs, dtype=np.float32)

        if not view_scores:
            return final_scores, view_scores

        view_matrix = np.stack(list(view_scores.values()), axis=0)  # Shape: (V, N)
        view_names = list(view_scores.keys())

        if strategy == "max":
            final_scores = np.max(view_matrix, axis=0)

        elif strategy == "mean_top_k":
            k = min(config.SIGLIP_TOP_K_VIEWS, view_matrix.shape[0])
            sorted_views = np.sort(view_matrix, axis=0)[::-1, :]
            final_scores = np.mean(sorted_views[:k, :], axis=0)

        elif strategy == "weighted_top_k":
            # Weight top views based on view fidelity weights in config
            k = min(config.SIGLIP_TOP_K_VIEWS, view_matrix.shape[0])
            weights = np.array([config.VIEW_WEIGHTS.get(v, 0.8) for v in view_names], dtype=np.float32)
            # Weighted matrix
            weighted_matrix = view_matrix * weights[:, None]
            sorted_weighted = np.sort(weighted_matrix, axis=0)[::-1, :]
            final_scores = np.mean(sorted_weighted[:k, :], axis=0)
            # Normalize to preserve range
            weight_norm = np.mean(np.sort(weights)[::-1][:k])
            if weight_norm > 0:
                final_scores = final_scores / weight_norm

        else:
            # Default to original view if present, else mean
            if "original" in view_scores:
                final_scores = view_scores["original"]
            else:
                final_scores = np.mean(view_matrix, axis=0)

        return np.clip(final_scores, 0.0, 1.0).astype(np.float32), view_scores
