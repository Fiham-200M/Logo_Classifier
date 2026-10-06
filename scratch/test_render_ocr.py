import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from PIL import Image
import numpy as np
import cv2

def test():
    # Let's inspect raja100.png
    img_path = project_root / "dataset" / "our_logos" / "raw_logos" / "raja100.png"
    img = Image.open(img_path).convert("RGBA")
    w, h = img.size
    print(f"raja100.png size: {w}x{h}")

test()
