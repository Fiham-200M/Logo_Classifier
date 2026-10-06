import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.ocr_model import OCRModel
from PIL import Image

ocr = OCRModel()
img = Image.open("dataset/our_logos/raw_logos/raja100.png")
res = ocr.extract_text(img)
print("Detected texts:", res["texts"])
print("Raw detections:")
for d in res["raw_detections"]:
    print(" ", d)
