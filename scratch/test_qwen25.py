import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import urllib.request
from PIL import Image
from vlm_fallback import image_to_base64

im1 = Image.open("logos/mo_2.jpg")
im2 = Image.open("logos/asia100.png")

prompt = """Compare Image 1 (query) and Image 2 (official asia100 reference).
Look closely at the swords and hilts:
1. In Image 1, how many sword hilts/handles are visible?
2. In Image 2, how many sword hilts/handles are visible?
3. Does Image 1 have an added second sword hilt in the top-right corner that is NOT present in Image 2?

Respond in JSON format:
{
  "image1_hilts": 2,
  "image2_hilts": 1,
  "has_added_sword": true,
  "is_exact_match": false,
  "differences": "Image 1 has an extra sword hilt in the top-right corner"
}"""

payload = {
    "model": "qwen2.5vl:7b",
    "prompt": prompt,
    "images": [image_to_base64(im1), image_to_base64(im2)],
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.1, "num_predict": 512}
}

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=45) as resp:
    d = json.loads(resp.read().decode("utf-8"))
    print("Qwen2.5-VL Response:")
    print(d.get("response"))
