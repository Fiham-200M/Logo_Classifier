"""
==========================================================================
 Logo & Favicon Forensic Checker - Crawler Integration Module
==========================================================================
 Drop this file into your crawler project.
 Usage: see bottom of file or README_INTEGRATION.md

 API Server  : http://192.168.10.113:8000
 Endpoint    : POST /api/verify
 Timeout     : 30 seconds per image
==========================================================================
"""

import base64
import json
import time
from pathlib import Path
from typing import Optional, Union
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError


# ---------------------------------------------------------------------------
# CONFIGURATION  --  only change API_URL if the server IP changes
# ---------------------------------------------------------------------------

API_URL     = "http://192.168.10.113:8000/api/verify"
TIMEOUT     = 30   # seconds
MAX_RETRIES = 2    # retries on network error


# ---------------------------------------------------------------------------
# VERDICT CONSTANTS
# ---------------------------------------------------------------------------

VERDICT_MATCH   = "MATCH"    # Confirmed: This IS our logo / favicon
VERDICT_REVIEW  = "REVIEW"   # Suspicious: possible clone, needs human check
VERDICT_UNKNOWN = "UNKNOWN"  # Not our logo / favicon


# ---------------------------------------------------------------------------
# CORE FUNCTION  --  call this from your crawler loop
# ---------------------------------------------------------------------------

def check_image(image_source, filename="image.png", asset_mode="auto"):
    """
    Check whether an image is one of our official logos or favicons.

    Parameters
    ----------
    image_source : str | bytes | Path
        - str starting with http/https  -> downloads the image from that URL
        - bytes / bytearray             -> uses the raw image bytes directly
        - Path / local file path str    -> reads the file from disk

    filename : str
        Filename for logging (tip: include the domain, e.g. "site.com_logo.png")

    asset_mode : str
        "auto"    -> system decides (recommended for crawlers)
        "logo"    -> force logo mode   (large brand artwork)
        "favicon" -> force favicon mode (small 16-64px tab icons)

    Returns
    -------
    dict with keys:
        is_our_logo   bool   - True if verdict is MATCH
        verdict       str    - "MATCH" | "REVIEW" | "UNKNOWN"
        brand         str    - detected brand name, or None
        confidence    float  - 0.0 to 1.0
        asset_type    str    - "logo" | "favicon" | "unknown"
        reason        str    - human-readable explanation
        processing_ms int    - elapsed time in milliseconds
        error         str    - error message if call failed, else None

    Quick example
    -------------
    result = check_image("https://site.com/favicon.ico", asset_mode="favicon")
    if result["is_our_logo"]:
        print("YES - Our logo!")
    elif result["verdict"] == "REVIEW":
        print("REVIEW - Possible clone, check manually.")
    else:
        print("NO - Not our logo.")
    """

    raw_bytes, error = _load_image(image_source, filename)
    if error:
        return _error_result(error)

    b64_str = "data:image/png;base64," + base64.b64encode(raw_bytes).decode()

    payload = json.dumps({
        "image_base64": b64_str,
        "filename":     filename,
        "asset_mode":   asset_mode,
        "debug":        False,   # no debug images -> faster for bulk use
        "skip_vlm":     True,    # skip slow VLM -> faster for bulk use
    }).encode()

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0  = time.time()
            req = Request(API_URL, data=payload,
                          headers={"Content-Type": "application/json"},
                          method="POST")
            with urlopen(req, timeout=TIMEOUT) as resp:
                data = json.loads(resp.read())
            ms  = int((time.time() - t0) * 1000)

            report  = data.get("report", {})
            verdict = report.get("verdict", "UNKNOWN")
            return {
                "is_our_logo":   verdict == VERDICT_MATCH,
                "verdict":       verdict,
                "brand":         report.get("brand_id"),
                "confidence":    report.get("confidence_score", 0.0),
                "asset_type":    report.get("asset_type", "unknown"),
                "reason":        report.get("action_reason", ""),
                "processing_ms": ms,
                "error":         None,
            }

        except (URLError, HTTPError, OSError) as exc:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return _error_result(f"API call failed: {exc}")

        except (json.JSONDecodeError, KeyError) as exc:
            return _error_result(f"Invalid API response: {exc}")

    return _error_result("Unknown error")


# ---------------------------------------------------------------------------
# BATCH HELPER  --  check logo + favicon of a web page in one call
# ---------------------------------------------------------------------------

def check_page_assets(page_url, logo_url=None, favicon_url=None):
    """
    Check both the logo and favicon of a web page.

    Parameters
    ----------
    page_url    : str  - the page URL (used for logging)
    logo_url    : str  - URL of the main logo image  (optional)
    favicon_url : str  - URL of the favicon           (optional)

    Returns
    -------
    dict with keys:
        page_url    str  - the page URL
        logo        dict - check_image() result for logo, or None
        favicon     dict - check_image() result for favicon, or None
        verdict     str  - "MATCH" | "REVIEW" | "UNKNOWN"
        is_our_page bool - True if logo or favicon is a MATCH

    Example
    -------
    result = check_page_assets(
        page_url    = "https://example-site.com",
        logo_url    = "https://example-site.com/assets/logo.png",
        favicon_url = "https://example-site.com/favicon.ico",
    )
    print(result["verdict"])
    """

    domain         = page_url.split("//")[-1].split("/")[0]
    logo_result    = check_image(logo_url,    f"{domain}_logo.png",    "logo")    if logo_url    else None
    favicon_result = check_image(favicon_url, f"{domain}_favicon.png", "favicon") if favicon_url else None

    results = [r for r in [logo_result, favicon_result] if r]
    if any(r["verdict"] == VERDICT_MATCH   for r in results): overall = "MATCH"
    elif any(r["verdict"] == VERDICT_REVIEW for r in results): overall = "REVIEW"
    else:                                                       overall = "UNKNOWN"

    return {
        "page_url":    page_url,
        "logo":        logo_result,
        "favicon":     favicon_result,
        "verdict":     overall,
        "is_our_page": overall == "MATCH",
    }


# ---------------------------------------------------------------------------
# FULL SCREENSHOT SCANNER HELPER  --  check full-page screenshots
# ---------------------------------------------------------------------------

def check_screenshot(image_source, filename="screenshot.png"):
    """
    Scan an entire webpage screenshot.
    Automatically finds, crops, and verifies our brand logos across the page.

    Parameters
    ----------
    image_source : str | bytes | Path
        - str starting with http/https -> downloads screenshot from URL
        - bytes / bytearray            -> raw screenshot image bytes
        - Path / file path str         -> local screenshot file path

    filename : str
        Filename for logging (e.g. "page_screenshot.png")

    Returns
    -------
    dict with keys:
        is_our_logo      bool  - True if our brand logo was detected on the page
        verdict          str   - "MATCH" | "REVIEW" | "UNKNOWN"
        brand            str   - detected brand name, or None
        confidence       float - 0.0 to 1.0 confidence score
        detections       list  - list of detected regions with bounding boxes [x, y, w, h]
        detections_count int   - number of brand regions located
        processing_ms    int   - elapsed time in milliseconds
        error            str   - error message if call failed, else None
    """
    raw_bytes, error = _load_image(image_source, filename)
    if error:
        return _error_result(error)

    b64_str = "data:image/png;base64," + base64.b64encode(raw_bytes).decode()
    payload = json.dumps({
        "image_base64": b64_str,
        "filename":     filename,
    }).encode()

    screenshot_api_url = API_URL.replace("/api/verify", "/api/verify_screenshot")

    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            req = Request(screenshot_api_url, data=payload,
                          headers={"Content-Type": "application/json"},
                          method="POST")
            with urlopen(req, timeout=TIMEOUT * 2) as resp:
                data = json.loads(resp.read())
            ms = int((time.time() - t0) * 1000)

            report = data.get("report", {})
            verdict = report.get("verdict", "UNKNOWN")
            return {
                "is_our_logo":      verdict == VERDICT_MATCH,
                "verdict":          verdict,
                "brand":            report.get("brand_id"),
                "confidence":       report.get("confidence_score", 0.0),
                "asset_type":       "screenshot",
                "detections":       report.get("detections", []),
                "detections_count": report.get("detections_count", 0),
                "reason":           report.get("action_reason", ""),
                "processing_ms":    ms,
                "error":            None,
            }

        except (URLError, HTTPError, OSError) as exc:
            if attempt < MAX_RETRIES:
                time.sleep(1)
                continue
            return _error_result(f"Screenshot API call failed: {exc}")

        except (json.JSONDecodeError, KeyError) as exc:
            return _error_result(f"Invalid API response: {exc}")

    return _error_result("Unknown error")


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _load_image(source, filename):
    """Load image bytes from URL, bytes, or Path. Returns (bytes, error_str)."""
    try:
        if isinstance(source, (bytes, bytearray)):
            return bytes(source), None

        if isinstance(source, str) and source.startswith(("http://", "https://")):
            req = Request(source, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(req, timeout=10) as r:
                return r.read(), None

        path = Path(source)
        return path.read_bytes(), None

    except Exception as exc:
        return None, f"Failed to load image '{filename}': {exc}"


def _error_result(message):
    return {
        "is_our_logo":   False,
        "verdict":       "UNKNOWN",
        "brand":         None,
        "confidence":    0.0,
        "asset_type":    "unknown",
        "reason":        message,
        "processing_ms": 0,
        "error":         message,
    }


# ---------------------------------------------------------------------------
# STANDALONE CONNECTION TEST  --  run:  python logo_checker.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    HEALTH_URL = API_URL.replace("/api/verify", "/api/health")

    print("=" * 60)
    print("  Logo Forensic API - Connection Test")
    print("=" * 60)

    # 1. Health check
    try:
        with urlopen(Request(HEALTH_URL), timeout=5) as r:
            health = json.loads(r.read())
        print(f"  Server     : ONLINE  ({API_URL})")
        print(f"  Brands     : {health.get('brands_count')} protected")
        print(f"  Device     : {str(health.get('device','')).upper()}")
        print(f"  Status     : {health.get('status')}")
    except Exception as exc:
        print(f"  CANNOT CONNECT to {API_URL}")
        print(f"  Error: {exc}")
        print(f"\n  Make sure the server is running:")
        print(f"    python web_server.py 8000")
        raise SystemExit(1)

    print()

    # 2. Sample image test
    sample = Path(__file__).parent.parent / "Favicon" / "023_raja100-top.com_favicon.png"
    if sample.exists():
        print(f"  Testing favicon: {sample.name}")
        r = check_image(sample, sample.name, "favicon")
        print(f"  verdict     : {r['verdict']}")
        print(f"  brand       : {r['brand']}")
        print(f"  confidence  : {r['confidence']:.1%}")
        print(f"  is_our_logo : {r['is_our_logo']}")
        print(f"  time_ms     : {r['processing_ms']} ms")
    else:
        print("  (sample favicon not found - skipping image test)")

    print()
    print("  All OK. logo_checker.py is ready to use in your crawler.")
    print("=" * 60)
