"""
Website Analyzer Module.
Coordinates website content extraction, NLP categorization, screenshot context,
and fuses evidence with logo verification into a unified Website Identity Profile.
"""

import time
from typing import Dict, Any, Optional, Union
from pathlib import Path
from PIL import Image

from website_intelligence.schemas import (
    WebsiteIdentityProfile,
    ExtractedContent,
    ContentCategory,
    VisualContextEvidence,
)
from website_intelligence.content_extractor import WebsiteContentExtractor
from website_intelligence.content_classifier import WebsiteContentClassifier
from website_intelligence.screenshot_context import ScreenshotContextObserver


class WebsiteAnalyzer:
    """
    Unified analyzer for website intelligence.
    Orchestrates HTML parsing, text classification, and VLM context.
    """

    def __init__(self):
        self.extractor = WebsiteContentExtractor()
        self.classifier = WebsiteContentClassifier()
        self.vlm_observer = ScreenshotContextObserver()

    def analyze_website(
        self,
        url: str = "",
        html_content: Optional[str] = None,
        screenshot_input: Optional[Union[str, Path, Image.Image]] = None,
        logo_verification_result: Optional[Dict[str, Any]] = None,
        skip_vlm: bool = False,
    ) -> WebsiteIdentityProfile:
        """
        Synthesizes multi-source website intelligence.
        """
        start_time = time.time()

        # 1. Content Extraction from HTML
        extracted = self.extractor.extract_from_html(html_content or "", url=url)

        # 2. Screenshot Loading & Visual Observation
        pil_ss: Optional[Image.Image] = None
        if screenshot_input is not None:
            if isinstance(screenshot_input, (str, Path)):
                pil_ss = Image.open(screenshot_input).convert("RGB")
            elif isinstance(screenshot_input, Image.Image):
                pil_ss = screenshot_input.convert("RGB")

        # 3. Logo Verification Evidence Integration
        logo_res = logo_verification_result or {}
        detected_brand = logo_res.get("brand_id") or logo_res.get("primary_brand")
        brand_conf = float(logo_res.get("confidence_score") or logo_res.get("confidence") or 0.0)
        overall_verdict = logo_res.get("verdict", "UNKNOWN")
        logos_list = logo_res.get("detections", [])

        # 4. Text & OCR Extraction
        ocr_tokens = []
        if logo_res.get("ocr"):
            ocr_tokens.append(logo_res["ocr"].get("normalized_text", ""))
        for det in logos_list:
            if det.get("text"):
                ocr_tokens.append(det["text"])

        # 5. NLP Content Classification
        content_cat = self.classifier.classify_content(
            extracted_content=extracted,
            ocr_text=" ".join(ocr_tokens),
        )

        # 6. Optional VLM Visual Context
        visual_context = VisualContextEvidence()
        if pil_ss is not None and not skip_vlm:
            visual_context = self.vlm_observer.observe_context(
                pil_ss,
                brand_hint=detected_brand
            )

        elapsed = round(time.time() - start_time, 3)

        return WebsiteIdentityProfile(
            domain=extracted.domain,
            url=url,
            brand_detected=detected_brand,
            brand_confidence=brand_conf,
            verdict=overall_verdict,
            logos=logos_list,
            content=content_cat,
            text_evidence={
                "title": extracted.title,
                "meta_description": extracted.meta_description,
                "headings": extracted.headings,
                "navigation_labels": extracted.navigation_labels,
                "action_buttons": extracted.action_buttons,
                "visible_text_summary": extracted.visible_text_summary,
                "ocr_tokens": ocr_tokens,
            },
            visual_context=visual_context,
            processing_time_sec=elapsed,
        )
