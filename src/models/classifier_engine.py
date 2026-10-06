"""
Brand Classifier Engine.

Loads the trained SigLIP2 + Linear Head classifier and provides
inference methods for brand classification with "unknown" rejection.

Integrates as an additional signal in the multi-stage detection pipeline.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config


class BrandClassifier(nn.Module):
    """Small MLP classifier on top of frozen SigLIP2 embeddings."""
    def __init__(self, embedding_dim: int = 768, num_classes: int = 53):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(x)


class ClassifierEngine:
    """
    Classifier inference engine.
    
    Loads the trained brand classifier and provides:
    - predict(): Returns predicted class and confidence
    - score_candidates(): Returns per-brand classification scores
                         compatible with the pipeline's score arrays
    """

    def __init__(self, classifier_path: Optional[Path] = None, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        versioned_path = getattr(config, "SIGLIP_VERSION_DIR", config.REFERENCE_DIR) / "brand_classifier.pt"
        self.classifier_path = classifier_path or (versioned_path if versioned_path.exists() else config.REFERENCE_DIR / "brand_classifier.pt")
        self.model = None
        self.class_names = []
        self.brand_names = []
        self.num_classes = 0
        self.unknown_idx = -1
        self.loaded = False

        self._load_model()

    def _load_model(self):
        """Load the trained classifier from disk."""
        if not self.classifier_path.exists():
            print(f"[ClassifierEngine] No classifier found at {self.classifier_path}")
            print(f"[ClassifierEngine] Run 'python train_classifier.py' to train one.")
            return

        try:
            checkpoint = torch.load(self.classifier_path, map_location="cpu", weights_only=False)
            self.class_names = checkpoint["class_names"]
            self.brand_names = checkpoint["brand_names"]
            self.num_classes = checkpoint["num_classes"]
            embedding_dim = checkpoint["embedding_dim"]

            self.model = BrandClassifier(
                embedding_dim=embedding_dim,
                num_classes=self.num_classes
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            # Find the "unknown" class index
            if "unknown" in self.class_names:
                self.unknown_idx = self.class_names.index("unknown")

            self.loaded = True
            val_acc = checkpoint.get("best_val_accuracy", 0.0)
            print(f"[ClassifierEngine] Loaded classifier: {self.num_classes} classes, "
                  f"val_acc={val_acc:.4f}")

        except Exception as e:
            print(f"[ClassifierEngine] Failed to load classifier: {e}")

    def predict(self, embedding: torch.Tensor) -> Dict[str, Any]:
        """
        Predict brand from a SigLIP2 embedding.
        
        Args:
            embedding: SigLIP2 embedding tensor of shape (768,) or (1, 768)
            
        Returns:
            Dict with:
                - predicted_brand: str (or "unknown")
                - confidence: float (softmax probability)
                - is_unknown: bool
                - top_k: List of (brand, confidence) tuples
        """
        if not self.loaded:
            return {
                "predicted_brand": None,
                "confidence": 0.0,
                "is_unknown": True,
                "top_k": [],
            }

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        expected_dim = self.model.classifier[0].in_features
        if embedding.shape[-1] != expected_dim:
            return {
                "predicted_brand": None,
                "confidence": 0.0,
                "is_unknown": False,
                "top_k": [],
                "dimension_mismatch": True,
            }

        with torch.no_grad():
            logits = self.model(embedding)
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]
        top_idx = int(np.argmax(probs_np))
        top_conf = float(probs_np[top_idx])
        top_brand = self.class_names[top_idx]

        # Get top-5
        top_k_indices = np.argsort(probs_np)[::-1][:5]
        top_k = [
            (self.class_names[idx], float(probs_np[idx]))
            for idx in top_k_indices
        ]

        is_unknown = (top_idx == self.unknown_idx)

        return {
            "predicted_brand": top_brand if not is_unknown else "unknown",
            "confidence": top_conf,
            "is_unknown": is_unknown,
            "unknown_prob": float(probs_np[self.unknown_idx]) if self.unknown_idx >= 0 else 0.0,
            "top_k": top_k,
        }

    def score_candidates(
        self,
        embedding: Any,
        pipeline_brand_names: List[str],
    ) -> np.ndarray:
        """
        Return per-brand classification scores aligned with the pipeline's brand list.
        
        Maps classifier probabilities to the pipeline's brand_names ordering.
        Brands not in the classifier get a neutral score of 0.5.
        
        Args:
            embedding: SigLIP2 embedding tensor or ndarray of shape (768,) or (1, 768)
            pipeline_brand_names: The brand name list from the pipeline's BrandDatabase
            
        Returns:
            np.ndarray of shape (num_pipeline_brands,) with classifier scores
        """
        num_brands = len(pipeline_brand_names)
        scores = np.full(num_brands, 0.5, dtype=np.float32)  # Neutral default

        if not self.loaded:
            return scores

        if isinstance(embedding, np.ndarray):
            embedding = torch.from_numpy(embedding).float()
        elif not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)

        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)

        embedding = embedding.to(self.device)

        with torch.no_grad():
            logits = self.model(embedding)
            probs = torch.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()[0]

        # Map classifier class probabilities to pipeline brand indices
        classifier_brand_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for i, brand in enumerate(pipeline_brand_names):
            if brand in classifier_brand_to_idx:
                cls_idx = classifier_brand_to_idx[brand]
                scores[i] = float(probs_np[cls_idx])

        return scores
