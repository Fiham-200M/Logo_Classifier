"""
Forensic Logo Verification Web Server.
Lightweight standard-library HTTP server that exposes the LogoForensicsEngine
via REST API endpoints and serves the modern glassmorphism frontend dashboard.
"""

import os
import sys
import json
import base64
import time
import mimetypes
from io import BytesIO
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from PIL import Image

# Ensure project root is in python path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
import torch
from verify_logo import LogoForensicsEngine
from website_intelligence.website_analyzer import WebsiteAnalyzer

# Global singleton engine loaded once at server startup
ENGINE: LogoForensicsEngine = None
WEBSITE_ANALYZER: WebsiteAnalyzer = None


def get_engine():
    global ENGINE
    if ENGINE is None:
        print("[WebServer] Initializing LogoForensicsEngine models onto GPU...")
        ENGINE = LogoForensicsEngine()
        print("[WebServer] Engine ready.")
    return ENGINE


def get_website_analyzer():
    global WEBSITE_ANALYZER
    if WEBSITE_ANALYZER is None:
        WEBSITE_ANALYZER = WebsiteAnalyzer()
    return WEBSITE_ANALYZER


def image_to_base64(img_input) -> str:
    """Converts a PIL Image, numpy array or Path to a data URL string."""
    try:
        if isinstance(img_input, (str, Path)):
            p = Path(img_input)
            if not p.exists():
                return ""
            mime, _ = mimetypes.guess_type(str(p))
            mime = mime or "image/png"
            with open(p, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return f"data:{mime};base64,{encoded}"
        elif isinstance(img_input, Image.Image):
            buf = BytesIO()
            img_input.save(buf, format="PNG")
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{encoded}"
    except Exception as e:
        print(f"[WebServer] Image encode error: {e}")
    return ""


class ForensicRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Keep terminal log concise
        sys.stderr.write(f"[{self.log_date_time_string()}] {self.command} {self.path} -> {args[1]}\n")

    def _send_json(self, status: int, payload: dict):
        try:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (ConnectionResetError, ConnectionAbortedError, BrokenPipeError):
            pass

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self._send_json(200, {
                "status": "online",
                "brands_count": len(get_engine().ref_store.references),
                "device": "cuda" if torch.cuda.is_available() else "cpu",
            })
            return

        if path == "/api/brands":
            brands = get_engine().ref_store.brand_names
            refs = get_engine().ref_store.references
            brand_list = []
            for b in brands:
                r = refs.get(b, {})
                fname = r.get("filename", "")
                brand_list.append({
                    "brand_id": b,
                    "filename": fname,
                    "variants_count": len(r.get("variants", [])),
                })
            self._send_json(200, {"brands": brand_list, "total": len(brand_list)})
            return

        if path == "/api/samples":
            samples = self._get_sample_images()
            self._send_json(200, {"samples": samples})
            return

        if path.startswith("/api/sample_image"):
            params = parse_qs(parsed.query)
            target = params.get("path", [None])[0]
            if not target:
                self._send_json(400, {"error": "Missing path parameter"})
                return
            full_p = (BASE_DIR / target).resolve()
            if not str(full_p).startswith(str(BASE_DIR)) or not full_p.exists():
                self._send_json(404, {"error": "Sample file not found"})
                return
            mime, _ = mimetypes.guess_type(str(full_p))
            mime = mime or "application/octet-stream"
            with open(full_p, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)
            return

        # Serve static files from frontend/ directory
        frontend_dir = BASE_DIR / "frontend"
        if path == "/" or path == "":
            file_path = frontend_dir / "index.html"
        else:
            rel = path.lstrip("/")
            file_path = (frontend_dir / rel).resolve()

        if str(file_path).startswith(str(frontend_dir)) and file_path.exists() and file_path.is_file():
            mime, _ = mimetypes.guess_type(str(file_path))
            mime = mime or "text/plain"
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", f"{mime}; charset=utf-8" if "text" in mime or "javascript" in mime else mime)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return

        self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/verify":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw_data = self.rfile.read(length)
                payload = json.loads(raw_data.decode("utf-8"))

                img_b64 = payload.get("image_base64", "")
                sample_path = payload.get("sample_path", "")
                filename = payload.get("filename", "upload.png")

                temp_file = None
                if sample_path:
                    cand_path = (BASE_DIR / sample_path).resolve()
                    if not cand_path.exists():
                        self._send_json(404, {"error": f"Sample image not found: {sample_path}"})
                        return
                    target_input = cand_path
                    filename = cand_path.name
                elif img_b64:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(img_b64)
                    temp_dir = BASE_DIR / "debug" / "uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = temp_dir / f"upload_{int(time.time()*1000)}_{filename}"
                    with open(temp_file, "wb") as f:
                        f.write(img_bytes)
                    target_input = temp_file
                else:
                    self._send_json(400, {"error": "No image_base64 or sample_path provided"})
                    return

                asset_mode = payload.get("asset_mode", "auto")
                debug_mode = payload.get("debug", True)
                skip_vlm = payload.get("skip_vlm", False)

                # Execute end-to-end verification with optional debug visualizations
                engine = get_engine()
                debug_dir = BASE_DIR / "debug" / "current_run" if debug_mode else None
                if debug_mode and debug_dir:
                    debug_dir.mkdir(parents=True, exist_ok=True)

                report = engine.verify(
                    target_input,
                    debug=debug_mode,
                    debug_dir=debug_dir,
                    asset_mode=asset_mode,
                    skip_vlm=skip_vlm,
                )

                # Collect reference variant image for side-by-side comparison
                ref_b64 = ""
                matched_ref_name = report.get("matched_reference")
                top_brand = report.get("brand_id")
                if debug_mode and top_brand and top_brand != "UNKNOWN":
                    ref_data = engine.ref_store.references.get(top_brand, {})
                    variants = ref_data.get("variants", [])
                    for v in variants:
                        if v.get("filename") == matched_ref_name:
                            ref_img = v.get("image_normalized") or v.get("image_black") or v.get("image_white")
                            if ref_img:
                                ref_b64 = image_to_base64(ref_img)
                            break
                    if not ref_b64 and ref_data.get("image_normalized"):
                        ref_b64 = image_to_base64(ref_data["image_normalized"])

                # Encode debug artifacts as base64 images if debug requested
                visualizations = {}
                if debug_mode and debug_dir:
                    visualizations = {
                        "diff_heatmap": image_to_base64(debug_dir / "diff_heatmap.png"),
                        "edge_candidate": image_to_base64(debug_dir / "edge_candidate.png"),
                        "edge_reference": image_to_base64(debug_dir / "edge_reference.png"),
                        "alignment": image_to_base64(debug_dir / "alignment.png"),
                        "ocr_vis": image_to_base64(debug_dir / "ocr_visualization.png"),
                        "reference_image": ref_b64,
                    }

                response_data = {
                    "success": True,
                    "filename": filename,
                    "asset_mode": asset_mode,
                    "report": report,
                    "visualizations": visualizations,
                }

                self._send_json(200, response_data)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
            return

        if path == "/api/verify_screenshot":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw_data = self.rfile.read(length)
                payload = json.loads(raw_data.decode("utf-8"))

                img_b64 = payload.get("image_base64", "")
                sample_path = payload.get("sample_path", "")
                filename = payload.get("filename", "screenshot.png")

                if sample_path:
                    cand_path = (BASE_DIR / sample_path).resolve()
                    if not cand_path.exists():
                        self._send_json(404, {"error": f"Sample image not found: {sample_path}"})
                        return
                    target_input = cand_path
                    filename = cand_path.name
                elif img_b64:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(img_b64)
                    temp_dir = BASE_DIR / "debug" / "uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = temp_dir / f"screenshot_{int(time.time()*1000)}_{filename}"
                    with open(temp_file, "wb") as f:
                        f.write(img_bytes)
                    target_input = temp_file
                else:
                    self._send_json(400, {"error": "No image_base64 or sample_path provided"})
                    return

                engine = get_engine()
                from screenshot_scanner import ScreenshotScanner
                scanner = ScreenshotScanner(ocr_engine=engine.ocr._engine, ref_store=engine.ref_store)
                scan_res = scanner.scan_screenshot(target_input, engine=engine)

                # Enrich with Website Intelligence context without altering existing schema
                url_context = payload.get("url", "")
                html_context = payload.get("html", "")
                skip_vlm = payload.get("skip_vlm", True)

                profile_dict = {}
                try:
                    analyzer = get_website_analyzer()
                    web_profile = analyzer.analyze_website(
                        url=url_context,
                        html_content=html_context,
                        screenshot_input=target_input,
                        logo_verification_result=scan_res,
                        skip_vlm=skip_vlm,
                    )
                    profile_dict = web_profile.to_dict()
                except Exception as ex_web:
                    print(f"[WebServer] Website intelligence warning: {ex_web}")

                # Collect official reference variant image for side-by-side comparison
                top_brand = scan_res.get("primary_brand")
                ref_b64 = ""
                if top_brand and top_brand != "UNKNOWN":
                    ref_data = engine.ref_store.references.get(top_brand, {})
                    variants = ref_data.get("variants", [])
                    for v in variants:
                        if v.get("filename") == scan_res.get("matched_reference"):
                            ref_img = v.get("image_normalized") or v.get("image_black") or v.get("image_white")
                            if ref_img:
                                ref_b64 = image_to_base64(ref_img)
                            break
                    if not ref_b64 and ref_data.get("image_normalized"):
                        ref_b64 = image_to_base64(ref_data["image_normalized"])

                # Encode visual bounding boxes, OCR polygons, and best crop
                visualizations = {
                    "diff_heatmap": image_to_base64(scan_res.get("annotated_image")),
                    "alignment": image_to_base64(scan_res.get("annotated_image")),
                    "edge_candidate": image_to_base64(scan_res.get("best_crop")),
                    "edge_reference": ref_b64,
                    "ocr_vis": image_to_base64(scan_res.get("ocr_image")),
                    "reference_image": ref_b64,
                }

                response_data = {
                    "success": True,
                    "filename": filename,
                    "asset_mode": "screenshot",
                    "report": {
                        "verdict": scan_res["verdict"],
                        "brand_id": scan_res["primary_brand"],
                        "confidence_score": scan_res["confidence"],
                        "asset_type": "screenshot",
                        "media_type": "full_page_screenshot",
                        "action_reason": f"Webpage screenshot scan: {scan_res['detections_count']} brand regions detected.",
                        "detections": scan_res["detections"],
                        "detections_count": scan_res["detections_count"],
                        "candidate_regions_evaluated": scan_res["candidate_regions_evaluated"],
                        "processing_time_sec": scan_res["processing_time_sec"],
                        "forensic_metrics": scan_res.get("forensic_metrics", {}),
                        "ocr": scan_res.get("ocr", {}),
                        "ocr_evidence": scan_res.get("ocr", {}),
                        "threat_type": scan_res.get("threat_type", "PAGE_LOGO_DETECTED"),
                        "matched_reference": scan_res.get("matched_reference", "official_master.png"),
                        "candidate_brands": scan_res.get("candidate_brands", []),
                        "website_intelligence": profile_dict,
                    },
                    "website_intelligence": profile_dict,
                    "visualizations": visualizations,
                }
                self._send_json(200, response_data)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
            return

        if path == "/api/analyze_website":
            try:
                length = int(self.headers.get("Content-Length", 0))
                raw_data = self.rfile.read(length)
                payload = json.loads(raw_data.decode("utf-8"))

                url = payload.get("url", "")
                html_content = payload.get("html", "")
                img_b64 = payload.get("image_base64", "")
                sample_path = payload.get("sample_path", "")
                skip_vlm = payload.get("skip_vlm", False)

                target_input = None
                if sample_path:
                    cand_path = (BASE_DIR / sample_path).resolve()
                    if cand_path.exists():
                        target_input = cand_path
                elif img_b64:
                    if "," in img_b64:
                        img_b64 = img_b64.split(",", 1)[1]
                    img_bytes = base64.b64decode(img_b64)
                    temp_dir = BASE_DIR / "debug" / "uploads"
                    temp_dir.mkdir(parents=True, exist_ok=True)
                    temp_file = temp_dir / f"website_{int(time.time()*1000)}.png"
                    with open(temp_file, "wb") as f:
                        f.write(img_bytes)
                    target_input = temp_file

                # If screenshot is provided, scan it for logos
                logo_scan_res = None
                if target_input:
                    try:
                        engine = get_engine()
                        from screenshot_scanner import ScreenshotScanner
                        scanner = ScreenshotScanner(ocr_engine=engine.ocr._engine, ref_store=engine.ref_store)
                        logo_scan_res = scanner.scan_screenshot(target_input, engine=engine)
                    except Exception as e_scan:
                        print(f"[WebServer] Logo scan error during website analysis: {e_scan}")

                analyzer = get_website_analyzer()
                profile = analyzer.analyze_website(
                    url=url,
                    html_content=html_content,
                    screenshot_input=target_input,
                    logo_verification_result=logo_scan_res,
                    skip_vlm=skip_vlm,
                )

                self._send_json(200, {
                    "success": True,
                    "url": url,
                    "website_profile": profile.to_dict(),
                    "logo_verification": logo_scan_res,
                })
            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json(500, {"error": str(e)})
            return

        self.send_error(404, "Endpoint not found")

    def _get_sample_images(self):
        """Returns preset sample images separated by logo and favicon for 1-click UI testing."""
        logo_samples = [
            {
                "label": "A200M Dark Gray (Authentic)",
                "category": "authentic",
                "path": "dataset/our_logos/positive/logo_001/logo_001_pos_bg_dark_gray_00.png",
                "expected": "MATCH",
                "mode": "logo",
            },
            {
                "label": "A200M Padded (Authentic)",
                "category": "authentic",
                "path": "dataset/our_logos/positive/logo_001/logo_001_pos_pad_00.png",
                "expected": "MATCH",
                "mode": "logo",
            },
            {
                "label": "Dewi11 Official Master",
                "category": "authentic",
                "path": "logos/dewi11.gif",
                "expected": "MATCH",
                "mode": "logo",
            },
            {
                "label": "Asia100 Official",
                "category": "authentic",
                "path": "logos/asia100.png",
                "expected": "MATCH",
                "mode": "logo",
            },
            {
                "label": "Mo_5 Vector Redraw (Threat)",
                "category": "threat",
                "path": "test_images/mo_5.png",
                "expected": "REVIEW",
                "mode": "logo",
            },
            {
                "label": "Dewi11 Shifted / Altered",
                "category": "threat",
                "path": "test_images/11.gif",
                "expected": "REVIEW",
                "mode": "logo",
            },
            {
                "label": "Hokiemas88 (Competitor)",
                "category": "competitor",
                "path": "dataset/competitor_logos/competitor/logo_hokiemas88.webp",
                "expected": "UNKNOWN",
                "mode": "logo",
            },
            {
                "label": "Apizeus777 (Competitor)",
                "category": "competitor",
                "path": "dataset/competitor_logos/competitor/655de25bde976_Logo Apizeus777 GIF.webp",
                "expected": "UNKNOWN",
                "mode": "logo",
            },
        ]

        favicon_samples = [
            {
                "label": "A200M Official Favicon",
                "category": "authentic",
                "path": "Favicon/001_a200m-seru.fun_favicon.png",
                "expected": "MATCH",
                "mode": "favicon",
            },
            {
                "label": "Asia100 Official Favicon",
                "category": "authentic",
                "path": "Favicon/002_asia100hits.com_favicon.png",
                "expected": "MATCH",
                "mode": "favicon",
            },
            {
                "label": "Dewi11 Official Favicon",
                "category": "authentic",
                "path": "Favicon/011_blue.dewi11moon.com_favicon.png",
                "expected": "MATCH",
                "mode": "favicon",
            },
            {
                "label": "SGCWin88 Official Favicon",
                "category": "authentic",
                "path": "Favicon/028_sgcwin88sweet.site_favicon.png",
                "expected": "MATCH",
                "mode": "favicon",
            },
            {
                "label": "Madura88 Official Favicon",
                "category": "authentic",
                "path": "Favicon/019_madura88-cair.skin_favicon.png",
                "expected": "MATCH",
                "mode": "favicon",
            },
            {
                "label": "Test Candidate Favicon",
                "category": "threat",
                "path": "test_images/favicon.webp",
                "expected": "REVIEW",
                "mode": "favicon",
            },
        ]

        return {
            "logos": [s for s in logo_samples if (BASE_DIR / s["path"]).exists()],
            "favicons": [s for s in favicon_samples if (BASE_DIR / s["path"]).exists()],
        }


def get_network_ip():
    """Detects the primary active LAN IP address."""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.5)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        import socket
        for ip in socket.gethostbyname_ex(socket.gethostname())[2]:
            if not ip.startswith("127."):
                return ip
    except Exception:
        pass
    return "127.0.0.1"


def start_server(host="0.0.0.0", port=8000):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    server_address = (host, port)
    httpd = ThreadingHTTPServer(server_address, ForensicRequestHandler)
    net_ip = get_network_ip()
    print(f"\n=======================================================")
    print(f"[200M LOGO CLASSIFIER] SERVER RUNNING AT: http://{host}:{port}")
    print(f"Local access:   http://127.0.0.1:{port}")
    print(f"Network access: http://{net_ip}:{port}")
    print(f"=======================================================\n")
    # Pre-warm models so first user click is instant
    get_engine()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[WebServer] Stopping server...")
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    start_server(port=port)
