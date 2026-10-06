"""
SigLIP 2 Vision Foundation Model Module.
Supports google/siglip2-so400m-patch14-384 (and backward-compatible with base).
Extracts invariant semantic embeddings for fast visual screening and forensic verification.
Evaluates dual-canvas (white/black renders) and multi-frame animated GIFs.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel

import config


class SigLIPModel:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SigLIPModel, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        force_reload: bool = False,
    ):
        if self._initialized and not force_reload:
            return

        self.model_name = model_name or config.SIGLIP_MODEL_NAME
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32

        # Safe loading with fallback
        try:
            self._load_model()
        except torch.cuda.OutOfMemoryError as oom_err:
            if self.device == "cuda":
                print(f"[SigLIPModel] CUDA Out-of-Memory encountered on {self.device}. Falling back to CPU...")
                torch.cuda.empty_cache()
                self.device = "cpu"
                self.dtype = torch.float32
                self._load_model()
            else:
                raise oom_err

        self._initialized = True

    def _load_model(self):
        """Loads processor and model weights, discovering architectural parameters."""
        print(f"[SigLIPModel] Loading {self.model_name} on {self.device} ({self.dtype})...")

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # Use dtype parameter (transformers modern API)
        try:
            self.model = AutoModel.from_pretrained(self.model_name, dtype=self.dtype).to(self.device)
        except TypeError:
            self.model = AutoModel.from_pretrained(self.model_name, torch_dtype=self.dtype).to(self.device)

        self.model.eval()

        # Discover configuration at runtime rather than hardcoding
        cfg = self.model.config
        v_cfg = getattr(cfg, "vision_config", None)
        self.input_size = getattr(v_cfg, "image_size", 384)
        self.patch_size = getattr(v_cfg, "patch_size", 14)
        self.hidden_size = getattr(v_cfg, "hidden_size", 1152)

        # Validate embedding dimension dynamically with dummy pass
        with torch.inference_mode():
            dummy = Image.new("RGB", (self.input_size, self.input_size), (128, 128, 128))
            dummy_inputs = self.processor(images=dummy, return_tensors="pt").to(self.device)
            if self.dtype == torch.float16:
                for k, v in dummy_inputs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                        dummy_inputs[k] = v.half()
            test_out = self.model.vision_model(**dummy_inputs)
            self.embedding_dim = test_out.pooler_output.shape[-1]

        total_params = sum(p.numel() for p in self.model.parameters())
        vision_params = sum(p.numel() for p in self.model.vision_model.parameters()) if hasattr(self.model, "vision_model") else total_params

        # Print structured startup banner
        print("=" * 60)
        print("SIGLIP MODEL:")
        print(f"  Model ID:            {self.model_name}")
        print(f"  Device:              {self.device}")
        print(f"  DType:               {self.dtype}")
        print(f"  Input Size:          {self.input_size}x{self.input_size}")
        print(f"  Patch Size:          {self.patch_size}")
        print(f"  Embedding Dimension: {self.embedding_dim}")
        print(f"  Total Parameters:    {total_params:,} ({total_params/1e6:.2f}M / {total_params/1e9:.4f}B)")
        print(f"  Vision Parameters:   {vision_params:,} ({vision_params/1e6:.2f}M)")
        if self.device == "cuda":
            vram_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"  VRAM Allocated:      {vram_mb:.1f} MB")
        print("=" * 60)

    @torch.inference_mode()
    def extract_embedding(self, image: Union[Image.Image, np.ndarray, Path, str]) -> np.ndarray:
        """
        Extracts L2-normalized semantic embedding from an image.
        Uses model's native processor for 384x384 resize and normalization.
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            img = Image.fromarray(image).convert("RGB")
        elif isinstance(image, Image.Image):
            img = image.convert("RGB")
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        inputs = self.processor(images=img, return_tensors="pt").to(self.device)
        if self.dtype == torch.float16:
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    inputs[k] = v.half()

        vision_outputs = self.model.vision_model(**inputs)
        pooler_output = vision_outputs.pooler_output  # (1, embedding_dim)

        assert pooler_output.shape[-1] == self.embedding_dim, (
            f"Embedding dimension mismatch: expected {self.embedding_dim}, got {pooler_output.shape[-1]}"
        )

        # L2 Normalize
        embedding = pooler_output / (pooler_output.norm(p=2, dim=-1, keepdim=True) + 1e-12)
        return embedding.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_dual_canvas_embeddings(
        self, canvas_white: Image.Image, canvas_black: Image.Image
    ) -> Dict[str, np.ndarray]:
        """
        Extracts embeddings for both white-background and black-background renders
        for transparency invariance.
        """
        return {
            "white": self.extract_embedding(canvas_white),
            "black": self.extract_embedding(canvas_black),
        }

    def extract_multiframe_embedding(
        self,
        frames: List[Image.Image],
        strategy: str = "mean",
    ) -> np.ndarray:
        """
        Deterministic aggregation for animated GIF / multi-frame candidates.
        Strategy 'mean': Average normalized frame embeddings and re-normalize to unit length.
        """
        if not frames:
            raise ValueError("frames list cannot be empty for multi-frame embedding extraction")

        if len(frames) == 1:
            return self.extract_embedding(frames[0])

        embs = [self.extract_embedding(f) for f in frames]
        embs_arr = np.array(embs, dtype=np.float32)  # (N, D)

        if strategy == "mean":
            mean_vec = np.mean(embs_arr, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-12:
                mean_vec = mean_vec / norm
            return mean_vec.astype(np.float32)
        else:
            # Default fallback to first frame
            return embs[0]

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates cosine similarity between two 1D normalized embeddings.
        Asserts matching dimensions.
        """
        if vec1.shape[-1] != vec2.shape[-1]:
            raise ValueError(
                f"Embedding dimension mismatch in cosine_similarity: "
                f"vec1 ({vec1.shape[-1]}) vs vec2 ({vec2.shape[-1]}). "
                f"Ensure candidate and reference galleries use the same SigLIP model."
            )

        sim = float(np.dot(vec1, vec2))
        return max(0.0, min(1.0, sim))


def get_siglip_embedding(image: Union[Image.Image, np.ndarray, Path, str]) -> np.ndarray:
    """Convenience helper function returning normalized SigLIP embedding."""
    model = SigLIPModel()
    return model.extract_embedding(image)
