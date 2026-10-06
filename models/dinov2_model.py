"""
DINOv2 Visual Structure & Geometry Model Module.
Model: facebook/dinov2-base
Extracts 768-dimensional self-supervised vision transformer embeddings
to capture fine spatial geometry, stroke layout, and structural proportions.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


class DINOv2Model:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DINOv2Model, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: str = "facebook/dinov2-base",
        device: Optional[str] = None
    ):
        if self._initialized:
            return

        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[DINOv2Model] Loading {self.model_name} on {self.device}...")

        self.processor = AutoImageProcessor.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self._initialized = True

    @torch.no_grad()
    def extract_embedding(self, image: Image.Image) -> np.ndarray:
        """
        Extracts L2-normalized 768-d structural embedding from PIL Image using the CLS token.
        """
        rgb_img = image.convert("RGB")
        inputs = self.processor(images=rgb_img, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # Use CLS token representation from last hidden state
        cls_token = outputs.last_hidden_state[:, 0, :]  # Shape: (1, 768)

        # L2 Normalize
        embedding = cls_token / cls_token.norm(p=2, dim=-1, keepdim=True)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 1D normalized DINOv2 embeddings.
        """
        denom = (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        if denom < 1e-12:
            return 0.0
        sim = float(np.dot(vec1, vec2) / denom)
        return max(0.0, min(1.0, sim))
