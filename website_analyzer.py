#!/usr/bin/env python3
"""
Website Intelligence CLI Entry Point.
Analyzes website URLs, HTML structures, and webpage screenshots to detect
brand identities, content categories (e.g. Online Gambling, Sports Betting, Casino),
text evidence, and visual VLM context.

Usage:
    python website_analyzer.py https://dewi11.com
    python website_analyzer.py --url https://dewi11.com --screenshot screenshot.png
    python website_analyzer.py --html page.html --screenshot screenshot.png --json
"""

import sys
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

# Ensure project root is in sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from website_intelligence.website_analyzer import WebsiteAnalyzer


def fetch_url_html(url: str) -> str:
    """Safely fetch HTML content from a URL."""
    try:
        req = Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urlopen(req, timeout=15) as resp:
            content = resp.read()
            # Attempt decode utf-8 with fallback to latin-1
            try:
                return content.decode("utf-8")
            except UnicodeDecodeError:
                return content.decode("latin-1", errors="replace")
    except Exception as e:
        print(f"[WebsiteAnalyzer] Warning: Failed to fetch {url}: {e}", file=sys.stderr)
        return ""


def print_terminal_report(profile_dict: dict, url: str):
    """Prints a structured, high-readability terminal banner for website analysis."""
    print("=" * 65)
    print("           WEBSITE INTELLIGENCE & FORENSIC PROFILE")
    print("=" * 65)
    print(f"URL:            {url or profile_dict.get('url', 'N/A')}")
    print(f"Domain:         {profile_dict.get('domain', 'N/A')}")
    print(f"Verdict:        {profile_dict.get('verdict', 'UNKNOWN')}")

    brand = profile_dict.get("brand_detected")
    brand_conf = profile_dict.get("brand_confidence", 0.0)
    brand_str = f"{brand} ({brand_conf:.1%})" if brand else "None Detected"
    print(f"Detected Brand: {brand_str}")

    print("-" * 65)
    content = profile_dict.get("content", {})
    primary_cat = content.get("primary", "Unknown")
    cat_conf = content.get("confidence", 0.0)
    secondaries = content.get("secondary", [])
    sec_str = f" [{', '.join(secondaries)}]" if secondaries else ""
    print(f"Classification: {primary_cat} ({cat_conf:.1%}){sec_str}")

    keywords = content.get("detected_keywords", [])
    if keywords:
        print(f"Key Indicators: {', '.join(keywords[:10])}")

    print("-" * 65)
    text_ev = profile_dict.get("text_evidence", {})
    if text_ev.get("title"):
        print(f"Title:          {text_ev.get('title')}")
    if text_ev.get("headings"):
        print(f"Headings:       {' | '.join(text_ev.get('headings')[:3])}")
    if text_ev.get("action_buttons"):
        print(f"Action CTAs:    {', '.join(text_ev.get('action_buttons')[:5])}")

    vis_ctx = profile_dict.get("visual_context", {})
    if vis_ctx.get("vlm_available"):
        print("-" * 65)
        print(f"Visual Type:    {vis_ctx.get('website_type_visual', 'N/A')}")
        print(f"Promo Saliency: {vis_ctx.get('brand_visual_saliency', 'N/A')}")
        if vis_ctx.get("raw_vlm_summary"):
            print(f"VLM Context:    {vis_ctx.get('raw_vlm_summary')}")

    logos = profile_dict.get("logos", [])
    if logos:
        print("-" * 65)
        print(f"Logo Candidates Verified: {len(logos)}")
        for i, l in enumerate(logos, 1):
            b_id = l.get("brand", l.get("brand_id", "Unknown"))
            v_stat = l.get("verdict", "UNKNOWN")
            conf = l.get("confidence", 0.0)
            bbox = l.get("bbox", [])
            print(f"  #{i}: Brand={b_id:<12} Verdict={v_stat:<8} Conf={conf:.1%} Box={bbox}")

    print("=" * 65)
    print(f"Analysis Time:  {profile_dict.get('processing_time_sec', 0.0)}s")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Multi-Modal Website Intelligence Analyzer")
    parser.add_argument("target_url", nargs="?", default="", help="Target website URL")
    parser.add_argument("--url", type=str, default="", help="Website URL")
    parser.add_argument("--html", type=str, default="", help="Path to local HTML file")
    parser.add_argument("--screenshot", type=str, default="", help="Path to webpage screenshot")
    parser.add_argument("--skip-vlm", action="store_true", help="Skip VLM contextual analysis")
    parser.add_argument("--json", action="store_true", help="Output raw JSON format")

    args = parser.parse_args()

    url = args.target_url or args.url
    if not url and not args.html and not args.screenshot:
        parser.print_help()
        sys.exit(1)

    html_content = ""
    if args.html:
        html_p = Path(args.html)
        if html_p.exists():
            html_content = html_p.read_text(encoding="utf-8", errors="replace")
    elif url.startswith(("http://", "https://")):
        print(f"[WebsiteAnalyzer] Fetching HTML from {url}...")
        html_content = fetch_url_html(url)

    # Optional logo verification if screenshot is provided
    logo_scan_res = None
    if args.screenshot:
        ss_path = Path(args.screenshot)
        if ss_path.exists():
            try:
                print(f"[WebsiteAnalyzer] Scanning screenshot for brand logos: {ss_path}...")
                from verify_logo import LogoForensicsEngine
                from screenshot_scanner import ScreenshotScanner
                engine = LogoForensicsEngine()
                scanner = ScreenshotScanner(ocr_engine=engine.ocr._engine, ref_store=engine.ref_store)
                logo_scan_res = scanner.scan_screenshot(ss_path, engine=engine)
            except Exception as e:
                print(f"[WebsiteAnalyzer] Warning: Logo scan error: {e}", file=sys.stderr)

    analyzer = WebsiteAnalyzer()
    profile = analyzer.analyze_website(
        url=url,
        html_content=html_content,
        screenshot_input=args.screenshot if args.screenshot else None,
        logo_verification_result=logo_scan_res,
        skip_vlm=args.skip_vlm,
    )

    profile_dict = profile.to_dict()

    if args.json:
        print(json.dumps(profile_dict, indent=2, ensure_ascii=False))
    else:
        print_terminal_report(profile_dict, url)


if __name__ == "__main__":
    main()
