import sys
sys.path.insert(0, ".")
import json
import urllib.request
from PIL import Image
from vlm_fallback import image_to_base64

img = Image.open("logos/mo_1.jpg")
b64 = image_to_base64(img, max_dim=512, min_dim=140)

candidate_brands = ["nusa211", "surgaplay", "surga200m", "asia200"]
candidates_str = ", ".join(f'"{c}"' for c in candidate_brands)

prompt = (
    "You are a strict anti-phishing logo verification system.\n"
    f"Candidate brands: [{candidates_str}].\n"
    "Look at this logo image carefully:\n"
    "- If the logo matches one of the candidate brands exactly, return that brand name.\n"
    "- If the logo is a competitor, copycat, or has different text/numbers (e.g. NUSA2111 vs nusa211), set candidate to null and explain.\n\n"
    "Respond in valid JSON:\n"
    '{"candidate": "NAME or null", "visible_text": "...", "is_competitor": true, "reason": "..."}'
)

payload = {
    "model": "qwen3.5:9b",
    "prompt": prompt,
    "images": [b64],
    "stream": False,
    "format": "json",
    "options": {
        "temperature": 0.1,
        "num_predict": 512,
    },
}

req = urllib.request.Request(
    "http://localhost:11434/api/generate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)

with urllib.request.urlopen(req, timeout=45) as resp:
    data = json.loads(resp.read().decode("utf-8"))
    raw = data.get("response", "")
    print("OUTPUT:\n", raw)
