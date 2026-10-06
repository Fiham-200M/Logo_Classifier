import sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModel, AutoProcessor

_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config
from src.models.classifier_engine import ClassifierEngine

print("Initializing ClassifierEngine...")
engine = ClassifierEngine()
print(f"Loaded: {engine.loaded}, Classes: {engine.num_classes}")

device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(config.SIGLIP_MODEL_NAME)
model = AutoModel.from_pretrained(config.SIGLIP_MODEL_NAME).to(device).eval()
vision_model = model.vision_model if hasattr(model, "vision_model") else model

def test_image(img_path):
    p = Path(img_path)
    if not p.exists():
        print(f"{p.name}: not found")
        return
    img = Image.open(p).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    with torch.no_grad():
        out = vision_model(pixel_values=pixel_values)
        emb = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state.mean(dim=1)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    pred = engine.predict(emb)
    print(f"\n=== {p.name} ===")
    print(f"  Prediction: {pred['predicted_brand']} (conf={pred['confidence']:.4f})")
    print(f"  Is Unknown: {pred['is_unknown']} (unknown_prob={pred['unknown_prob']:.4f})")
    print(f"  Top 3 Candidates:")
    for b, c in pred["top_k"][:3]:
        print(f"    - {b:<15}: {c:.4f}")

test_files = [
    "logos/jagoledak.png",
    "logos/surga88.png",
    "logos/surga55.png",
    r"c:\Users\AI Fiham\Documents\mo_3.png",
    r"c:\Users\AI Fiham\Documents\mo_4.png",
    r"c:\Users\AI Fiham\Documents\mo_5.png",
    r"c:\Users\AI Fiham\Documents\mo_6.png",
    r"c:\Users\AI Fiham\Documents\mo_7.png",
]

for f in test_files:
    test_image(f)
