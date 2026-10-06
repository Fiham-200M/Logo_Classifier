"""
==========================================================================
 EXAMPLE: How to integrate logo_checker.py into your crawler script
==========================================================================
 This shows a typical crawler loop pattern.
 Copy the relevant parts into your existing crawler code.
==========================================================================
"""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

# ---- Import the checker (copy logo_checker.py next to your crawler script)
from logo_checker import check_image, check_page_assets, check_screenshot, VERDICT_MATCH, VERDICT_REVIEW


# ---------------------------------------------------------------------------
#  PATTERN 1: You already have the image bytes in memory
#  (e.g. you used requests.get() to download it)
# ---------------------------------------------------------------------------

def pattern_1_bytes_in_memory():
    """
    Use this if your crawler already downloads images via requests/httpx.
    """
    import urllib.request

    url = "https://example-gambling-site.com/favicon.ico"

    # Download the image (you probably already do this in your crawler)
    with urllib.request.urlopen(url, timeout=10) as r:
        image_bytes = r.read()

    # Check it
    result = check_image(
        image_bytes,            # pass the raw bytes
        filename="example.com_favicon.ico",
        asset_mode="favicon",   # or "logo" or "auto"
    )

    # Use the result
    if result["is_our_logo"]:
        print(f"YES  -> brand={result['brand']}, confidence={result['confidence']:.1%}")
    elif result["verdict"] == VERDICT_REVIEW:
        print(f"REVIEW -> {result['reason']}")
    else:
        print(f"NO   -> Not our logo")

    return result


# ---------------------------------------------------------------------------
#  PATTERN 2: You have the image URL and want to let the checker download it
# ---------------------------------------------------------------------------

def pattern_2_url_direct():
    """
    Use this if you want the simplest possible integration.
    Just pass the URL string directly.
    """
    result = check_image(
        "https://example-site.com/static/logo.png",   # URL string
        asset_mode="logo",
    )
    return result["is_our_logo"]


# ---------------------------------------------------------------------------
#  PATTERN 3: Check both logo + favicon together (most thorough)
# ---------------------------------------------------------------------------

def pattern_3_full_page_check(page_url, logo_url, favicon_url):
    """
    Use this for full page scanning.
    Returns True if either the logo or favicon matches ours.
    """
    result = check_page_assets(
        page_url    = page_url,
        logo_url    = logo_url,
        favicon_url = favicon_url,
    )

    print(f"Page    : {result['page_url']}")
    print(f"Verdict : {result['verdict']}")
    print(f"Our page: {result['is_our_page']}")

    if result["logo"]:
        logo = result["logo"]
        print(f"  Logo   -> {logo['verdict']} | brand={logo['brand']} | {logo['confidence']:.1%}")

    if result["favicon"]:
        fav = result["favicon"]
        print(f"  Favicon-> {fav['verdict']} | brand={fav['brand']} | {fav['confidence']:.1%}")

    return result["is_our_page"]


# ---------------------------------------------------------------------------
#  PATTERN 4: Send full webpage screenshot directly (No CSS/DOM hunting!)
# ---------------------------------------------------------------------------

def pattern_4_screenshot_check(screenshot_path_or_bytes):
    """
    Use this if your crawler takes full webpage screenshots (e.g. Playwright / Selenium).
    The API automatically finds, crops, and verifies our brand logos across the page.
    """
    result = check_screenshot(
        image_source=screenshot_path_or_bytes,
        filename="page_screenshot.png",
    )

    print(f"Our brand found: {result['is_our_logo']}")
    if result["is_our_logo"]:
        print(f"  Brand: {result['brand']} ({result['confidence']:.1%})")
        print(f"  Detected {result['detections_count']} logo regions:")
        for det in result["detections"]:
            print(f"    - {det['brand']} ({det['verdict']}) at {det['box']}")

    return result["is_our_logo"]


# ---------------------------------------------------------------------------
#  COMPLETE CRAWLER LOOP EXAMPLE
#  Replace the URL list and your file-reading logic with your real crawler.
# ---------------------------------------------------------------------------

def crawler_loop_example():
    """
    Drop-in example showing how to integrate into a typical crawler loop.
    """

    # Your list of pages to check (replace with your actual URL source)
    pages_to_check = [
        {
            "page_url":    "https://site-a.com",
            "logo_url":    "https://site-a.com/assets/logo.png",
            "favicon_url": "https://site-a.com/favicon.ico",
        },
        {
            "page_url":    "https://site-b.net",
            "logo_url":    "https://site-b.net/img/brand.png",
            "favicon_url": "https://site-b.net/favicon.png",
        },
        # ... add more pages
    ]

    results = []
    output_file = Path("scan_results.csv")

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "page_url", "verdict", "logo_verdict", "logo_brand",
            "favicon_verdict", "favicon_brand", "scanned_at"
        ])
        writer.writeheader()

        for page in pages_to_check:
            print(f"Checking: {page['page_url']} ...", end=" ", flush=True)

            result = check_page_assets(
                page_url    = page.get("page_url"),
                logo_url    = page.get("logo_url"),
                favicon_url = page.get("favicon_url"),
            )

            logo_r = result.get("logo") or {}
            fav_r  = result.get("favicon") or {}

            row = {
                "page_url":       result["page_url"],
                "verdict":        result["verdict"],
                "logo_verdict":   logo_r.get("verdict", "-"),
                "logo_brand":     logo_r.get("brand", "-"),
                "favicon_verdict":fav_r.get("verdict", "-"),
                "favicon_brand":  fav_r.get("brand", "-"),
                "scanned_at":     datetime.now().isoformat(timespec="seconds"),
            }
            writer.writerow(row)
            results.append(row)

            verdict_icon = {"MATCH": "[YES]", "REVIEW": "[REVIEW]", "UNKNOWN": "[NO]"}.get(result["verdict"], "?")
            print(verdict_icon)

            time.sleep(0.1)   # small delay to avoid hammering the server

    print(f"\nDone! Results saved to: {output_file.absolute()}")
    print(f"Total pages checked : {len(results)}")
    print(f"MATCH (our pages)   : {sum(1 for r in results if r['verdict'] == 'MATCH')}")
    print(f"REVIEW (suspicious) : {sum(1 for r in results if r['verdict'] == 'REVIEW')}")
    print(f"UNKNOWN (others)    : {sum(1 for r in results if r['verdict'] == 'UNKNOWN')}")
    return results


# ---------------------------------------------------------------------------
#  SIMPLE YES/NO WRAPPER  (drop-in replacement for your current manual check)
# ---------------------------------------------------------------------------

def is_our_logo(image_url_or_bytes, asset_mode="auto"):
    """
    Simplest possible wrapper - returns True/False like your current manual YES/NO.

    Before  (manual):
        answer = input("Is this our logo? [y/n]: ")
        result = answer.lower() == "y"

    After  (automated):
        result = is_our_logo("https://site.com/logo.png")
    """
    result = check_image(image_url_or_bytes, asset_mode=asset_mode)
    return result["is_our_logo"]


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running pattern_3_full_page_check example...")
    pattern_3_full_page_check(
        page_url    = "https://example-site.com",
        logo_url    = None,   # replace with real URL
        favicon_url = None,   # replace with real URL
    )
