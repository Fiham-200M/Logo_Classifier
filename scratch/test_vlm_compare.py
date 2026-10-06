import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import urllib.request
from PIL import Image
import config
from vlm_fallback import image_to_base64

im_query = Image.open("logos/mo_7.png")
im_ref = Image.open("logos/nusa211.png")

b64_query = image_to_base64(im_query)
b64_ref = image_to_base64(im_ref)

prompt = """You are an expert brand logo verification system.
You are comparing two images:
- First image: The query/test logo.
- Second image: The official reference logo for brand 'nusa211'.

Inspect both images with extreme precision:
1. Identify the graphic emblem / icon / symbol on the left side of the First image.
2. Identify the graphic emblem / icon / symbol on the left side of the Second image.
3. Compare the text, font, and style.
4. If the emblem/icon depicts a different animal/object (e.g. eagle vs dragon), this is an ICON SWAP / SPOOF and NOT an exact match.

Respond in JSON format:
{
  "is_exact_match": false,
  "first_image_icon": "description of icon in first image",
  "second_image_icon": "description of icon in second image",
  "icons_match": false,
  "confidence": 0.95,
  "reason": "concise explanation"
}"""

payload = {
    "model": config.VLM_MODEL,
    "prompt": prompt,
    "images": [b64_query, b64_ref],
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.1, "num_predict": 512}
}

req = urllib.request.Request(
    config.VLM_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=45) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    raw = data.get("response", "")
    if "</think>" in raw:
        raw = raw.split("</think>")[-1].strip()
    print("VLM COMPARISON RESULT:")
    print(raw)
