# Logo & Favicon Forensic Checker — Integration Guide

## What This Is

A REST API service that automatically detects whether a logo or favicon
belongs to one of our 52 protected brands. It replaces the manual YES/NO
decision your team currently makes while crawling web pages.

---

## Files to Share With the Crawler Team

```
crawler_integration/
├── logo_checker.py       <-- The integration module  (copy this into their project)
├── crawler_example.py    <-- Shows how to use it     (reference / copy snippets)
└── README_INTEGRATION.md <-- This file
```

**Only `logo_checker.py` needs to go into their crawler project.**
It uses Python standard library only — no pip install needed.

---

## Step 1: Check the Server is Running

The forensic server must be running on the AI team's machine (leave it on):

```
Server Address : http://192.168.10.113:8000
```

Test it in a browser: `http://192.168.10.113:8000` should show the UI.

---

## Step 2: Copy logo_checker.py

Copy `logo_checker.py` into the same folder as their crawler script.

---

## Step 3: Update API_URL (if needed)

Open `logo_checker.py` and check line 32:

```python
API_URL = "http://192.168.10.113:8000/api/verify"
```

Change the IP address if the server moves.

---

## Step 4: Use in Their Crawler

### Simplest usage — replace your YES/NO input with one line:

```python
from logo_checker import check_image

result = check_image(image_bytes, filename="site.com_favicon.png")

if result["is_our_logo"]:
    print("YES")  # our brand
else:
    print("NO")   # not our brand
```

### What the result dict contains:

| Field | Type | Example | Meaning |
|---|---|---|---|
| `is_our_logo` | bool | `True` | **Main flag** — True = our brand |
| `verdict` | str | `"MATCH"` | `MATCH` / `REVIEW` / `UNKNOWN` |
| `brand` | str | `"raja100"` | Which brand was detected |
| `confidence` | float | `0.97` | 0.0–1.0 confidence score |
| `asset_type` | str | `"favicon"` | `logo` or `favicon` |
| `reason` | str | `"High-confidence..."` | Human-readable explanation |
| `processing_ms` | int | `720` | Time taken (milliseconds) |
| `error` | str/None | `None` | Error message if failed |

### Three verdict types:

| Verdict | Meaning | Action |
|---|---|---|
| `MATCH` | Confirmed — this IS our logo | Mark as YES → flag the page |
| `REVIEW` | Suspicious — possible clone | Log for human review |
| `UNKNOWN` | Not our logo | Mark as NO → skip |

---

## Common Integration Patterns

### Pattern A: You have image bytes (from requests/urllib download)

```python
from logo_checker import check_image

# In your crawler loop:
response = requests.get(image_url)
image_bytes = response.content

result = check_image(image_bytes, filename="site.com_logo.png", asset_mode="logo")
is_ours = result["is_our_logo"]
```

### Pattern B: You have the image URL — let the checker download it

```python
from logo_checker import check_image

result = check_image("https://site.com/favicon.ico", asset_mode="favicon")
is_ours = result["is_our_logo"]
```

### Pattern C: Check both logo + favicon together

```python
from logo_checker import check_page_assets

result = check_page_assets(
    page_url    = "https://site.com",
    logo_url    = "https://site.com/logo.png",      # or None if not found
    favicon_url = "https://site.com/favicon.ico",   # or None if not found
)

print(result["verdict"])    # "MATCH", "REVIEW", or "UNKNOWN"
print(result["is_our_page"])  # True / False
```

### Pattern D: Send full webpage screenshot directly (No cropping needed!)

```python
from logo_checker import check_screenshot

# Crawlers can take a screenshot of the entire page and check it directly:
result = check_screenshot(
    image_source="page_screenshot.png",  # or screenshot bytes
    filename="site.com_screenshot.png",
)

if result["is_our_logo"]:
    print(f"YES! Found brand: {result['brand']} ({result['confidence']:.1%})")
    print(f"Detected {result['detections_count']} logo regions on page:")
    for region in result["detections"]:
        print(f"  - {region['brand']} at box {region['box']}")
else:
    print("NO - No protected brand logos found on this webpage.")
```

---

## Performance

| Metric | Value |
|---|---|
| Avg response time | ~700 ms per image |
| Favicon detection accuracy | 100% (12/12 tested) |
| Classifier validation accuracy | 90.6% |
| Brands protected | 52 brands + favicons |
| Concurrent requests | Supported |

---

## Troubleshooting

### "API call failed: connection refused"
- The server is not running. Start it:
  ```
  python web_server.py 8000
  ```

### "API call failed: timed out"
- Server is overloaded or network is slow. Increase timeout in logo_checker.py:
  ```python
  TIMEOUT = 60
  ```

### Verdict is REVIEW instead of MATCH
- The image is a partial match (e.g. low-resolution, heavily compressed).
- Can manually confirm. REVIEW means "very likely ours but not 100% certain."

### Connection test
Run this to verify everything works:
```
python logo_checker.py
```
Expected output:
```
  Server     : ONLINE  (http://192.168.10.113:8000)
  Brands     : 52 protected
  Device     : CUDA
  Status     : ok
  verdict     : MATCH
  is_our_logo : True
```

---

## API Reference (for advanced integration)

### Endpoint: `POST /api/verify`

**Request body (JSON):**
```json
{
    "image_base64": "data:image/png;base64,<BASE64_DATA>",
    "filename":     "site.com_favicon.png",
    "asset_mode":   "auto",
    "debug":        false,
    "skip_vlm":     true
}
```

**Response (JSON):**
```json
{
    "success": true,
    "filename": "site.com_favicon.png",
    "report": {
        "brand_id":         "raja100",
        "verdict":          "MATCH",
        "confidence_score": 0.97,
        "asset_type":       "favicon",
        "action_reason":    "High-confidence multi-modal consensus...",
        "processing_time_sec": 0.71
    }
}
```

### Health check: `GET /api/health`
```json
{
    "status":       "ok",
    "brands_count": 52,
    "device":       "cuda"
}
```
