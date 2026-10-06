# Multi-Modal Brand Forensics & Anti-Spoofing Engine

An enterprise-grade, multi-modal computer vision and forensic verification engine designed to detect, verify, and protect official brand assets (logos, favicons, and full webpage screenshots) across **52 protected brands** and **62 protected favicons**.

It replaces manual visual inspection with automated high-confidence decisions (`MATCH`, `REVIEW`, `UNKNOWN`), protecting against phishing, typosquatting, cloned layouts, recolored logos, and stylized font impersonation.

---

## Key Capabilities

- **Multi-Modal Feature Fusion:** Combines vision-language semantic embeddings (Google SigLIP 2 SO400M), spatial-geometric structure (Meta DINOv2), perceptual colorimetry (CIE LAB $\Delta E$), contour edge IoU, and perceptual hashing (pHash / dHash).
- **Webpage Screenshot Scanner:** Automatically scans full-page webpage screenshots, detects header/navbar logo regions, evaluates candidate bounding boxes, and overlays visual bounding boxes with confidence badges directly on the image.
- **Fuzzy OCR & Calligraphy Normalization:** Integrates PaddleOCR (PP-OCRv5) and EasyOCR with phonetic and visual substitution algorithms (`A31a200` $\rightarrow$ `ASIA200`, `31` $\rightarrow$ `SI`, `0` $\rightarrow$ `O`) to defeat adversarial stylized typography.
- **Neural Rejection Layer:** 53-class neural classifier defense to reliably reject unknown competitor brands and out-of-distribution imagery with high confidence.
- **Lightweight Crawler Integration:** Zero-dependency client library (`crawler_integration/logo_checker.py`) that crawler teammates can drop into their scraping pipelines without installing heavy ML dependencies.
- **Live Glassmorphism Web Studio:** Interactive web UI on port 8000 for drag-and-drop inspection, side-by-side comparison, diff heatmaps, edge stroke IoU, and OCR bounding polygon canvases.

---

## System Architecture

```text
Input: Logo / Favicon / Full Screenshot
                    │
                    ▼
     [ Media Preprocessing & Normalization ]
     (Alpha transparency compositing, aspect ratio preservation)
                    │
    ┌───────────────┼───────────────┬───────────────┐
    ▼               ▼               ▼               ▼
[SigLIP 2 SO400M] [DINOv2 Base]  [CIE LAB ΔE]   [PaddleOCR / EasyOCR]
Semantic Vision   Geometry &     Color Palette  Text & Calligraphy
Alignment         Contour IoU    Drift Check    Normalization
    │               │               │               │
    └───────────────┼───────────────┴───────────────┘
                    ▼
     [ Multi-Modal Consensus Decision Engine ]
     (Confidence fusion, neural classifier defense layer)
                    │
                    ▼
 Verdict: MATCH (≥90%) | REVIEW (60-89%) | UNKNOWN (<60%)
```

---

## Hardware Requirements

- **GPU (Recommended):** NVIDIA GPU with $\ge$ 6 GB VRAM (RTX 3060, 4060, A10, T4 or higher) with CUDA 12.x or 11.x support.
- **CPU (Fallback):** Supported on x86_64 CPU (PyTorch float32 inference; ~1.5–3s per crop).
- **RAM:** 8 GB minimum (16 GB recommended).
- **Storage:** ~3 GB for model weights and reference embeddings.

---

## Setup & Installation

### 1. Clone & Enter Directory
```bash
git clone <repo_url>
cd siglip_test
```

### 2. Create Virtual Environment
Using Python 3.11 or 3.12:
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate
```

### 3. Install PyTorch with CUDA
Visit [pytorch.org](https://pytorch.org/) to match your CUDA version, or install for CUDA 12.4/12.6:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional PaddleOCR acceleration)*: If utilizing the high-speed isolated PaddleOCR worker:
```bash
pip install paddlepaddle-gpu paddleocr
```

---

## Quick Start & Usage

### 1. Launch the Forensic Web Server & Studio
```bash
python web_server.py 8000
```
Open your browser at:
```
http://localhost:8000
```
- **Logo Verification Tab:** Test isolated logo images (`.png`, `.webp`, `.jpg`, `.gif`).
- **Favicon Tab:** Test low-resolution website favicons (`16x16`, `32x32`, `.ico`, `.webp`).
- **Webpage Screenshot Mode:** Upload or paste full desktop/mobile screenshots. The scanner detects the logo region, highlights it with green/amber bounding boxes, and prints the forensic breakdown.

---

### 2. Verify via Command Line (CLI)

Verify a single logo:
```bash
python verify_logo.py logos/asia200.png
```

Verify with debug artifacts output (heatmaps, edge overlays, OCR visualizations saved in `debug/`):
```bash
python verify_logo.py --debug logos/asia200.png
```

Scan a full webpage screenshot:
```bash
python -c "
from PIL import Image
from verify_logo import LogoForensicsEngine
from screenshot_scanner import ScreenshotScanner

engine = LogoForensicsEngine()
scanner = ScreenshotScanner(ocr_engine=engine.ocr._engine, ref_store=engine.ref_store)
res = scanner.scan_screenshot('path/to/screenshot.png', engine=engine)
print('Brand:', res['primary_brand'], 'Verdict:', res['verdict'], 'Conf:', res['confidence'])
"
```

---

## REST API Reference

The server exposes lightweight JSON endpoints on port `8000`:

### `POST /api/verify` (Single Logo / Favicon)
**Request Body:**
```json
{
  "image_base64": "<base64_encoded_image_string>",
  "asset_mode": "logo",
  "filename": "candidate_logo.png"
}
```
**Response:**
```json
{
  "success": true,
  "report": {
    "verdict": "MATCH",
    "brand_id": "asia200",
    "confidence_score": 0.95,
    "forensic_metrics": {
      "siglip2_semantic_score": 0.9327,
      "dinov2_geometry_score": 0.7135,
      "cielab_delta_e": 3.28,
      "edge_stroke_iou": 0.3209,
      "ocr_text_match": true
    }
  }
}
```

---

### `POST /api/verify_screenshot` (Full Webpage Screenshot)
Scans the entire page, extracts candidate logo regions via OCR clustering and visual saliency, annotates bounding boxes, and returns complete visual evidence.

**Request Body:**
```json
{
  "image_base64": "<base64_encoded_screenshot>",
  "filename": "screenshot.png"
}
```
**Response:**
- `report.verdict`: `"MATCH" | "REVIEW" | "UNKNOWN"`
- `report.brand_id`: Identified official brand.
- `report.confidence_score`: Consensus score (e.g., `0.95`).
- `report.detections`: List of detected regions with coordinates `[x, y, width, height]`.
- `visualizations.alignment`: Base64 image of the full screenshot with bounding box annotations.
- `visualizations.reference_image`: Base64 image of the official reference master variant.

---

### `GET /api/health`
Returns system status, active CUDA device, and number of protected brands loaded.
```json
{
  "status": "online",
  "brands_count": 52,
  "device": "cuda"
}
```

---

## Sharing With Crawler Pipelines

For team members developing web crawlers (e.g. `end-link-lc`), they do **not** need to install PyTorch or vision models.

1. Share `crawler_integration/logo_checker.py` and `crawler_integration/README_INTEGRATION.md` with them.
2. In their scraper, they can verify images via standard HTTP:
```python
from logo_checker import check_image

result = check_image(image_bytes, filename="target.png")
if result["is_our_logo"]:
    print(f"Verified Brand: {result['brand']} ({result['confidence']*100:.1f}%)")
```
It includes built-in SHA-256 caching and automatic multi-host failover.

---

## Directory Structure

```text
siglip_test/
├── config.py                          # Global hyperparameters & model configurations
├── verify_logo.py                     # Core forensics engine CLI and verification pipeline
├── screenshot_scanner.py              # Full-page screenshot candidate detector & box annotator
├── web_server.py                      # REST API server & web dashboard host
├── requirements.txt                   # Dependency specifications
├── README.md                          # System documentation
├── reference_database/                # Reference database loader and variant stores
│   └── reference_store.py
├── reference_embeddings/              # Pre-computed forensic embeddings (52 brands)
│   └── siglip2_so400m_patch14_384/
├── models/                            # Vision & language model wrappers (SigLIP 2, DINOv2)
├── forensics/                         # Colorimeter (CIE LAB), edge IoU, perceptual hashing
├── logo_detection/                    # Candidate region proposals & calligraphy normalization
├── ocr_engine.py                      # Multi-engine OCR wrapper (PaddleOCR / EasyOCR)
├── ocr_worker.py                      # Isolated low-latency OCR worker
├── frontend/                          # High-performance glassmorphism UI
├── crawler_integration/               # Zero-dependency crawler integration module
│   ├── logo_checker.py
│   ├── crawler_example.py
│   └── README_INTEGRATION.md
└── logos/                             # Official master logo variants for protected brands
```
