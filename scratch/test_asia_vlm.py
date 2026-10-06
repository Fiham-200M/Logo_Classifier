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

im1 = Image.open("logos/mo_2.jpg")
im2 = Image.open("logos/asia100.png")

b64_1 = image_to_base64(im1)
b64_2 = image_to_base64(im2)

prompt = """You are a strict anti-phishing logo verification system.
Compare Image 1 (query image) with Image 2 (official brand reference logo for 'asia100').

Inspect every detail of the two images:
1. Count and locate all weapons, swords, hilts, or emblems in Image 1 and Image 2.
2. Are there any added, removed, or modified visual elements (such as an extra sword hilt or pommel)?
3. If Image 1 has added elements, altered graphics, or is a modified spoof, it is NOT an exact match.

Respond in JSON format:
{
  "is_exact_match": false,
  "differences_found": "describe all graphic differences, added/missing elements",
  "is_modified_spoof": true,
  "candidate": null,
  "confidence": 0.95,
  "reason": "concise explanation"
}"""

payload = {
    "model": config.VLM_MODEL,
    "prompt": prompt,
    "images": [b64_1, b64_2],
    "stream": False,
    "format": "json",
    "options": {"temperature": 0.1, "num_predict": 1024}
}

req = urllib.request.Request(
    config.VLM_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST"
)

with urllib.request.urlopen(req, timeout=45) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    txt = data.get("response", "")
    if "</think>" in txt:
        txt = txt.split("</think>")[-1].strip()
    print("VLM Result:")
    print(txt)
