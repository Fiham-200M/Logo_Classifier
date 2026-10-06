"""
Vision-Language Model (VLM) Engine.
Provides three VLM capabilities:
  1. query_vlm_fallback()        — Brand disambiguation (existing)
  2. query_vlm_text_extraction() — Independent text reading (existing)
  3. query_vlm_analysis()        — Full structured analysis (NEW Phase 3)

All functions re-export from the root module, with new analysis added.
"""

import io
import json
import re
import base64
import urllib.request
import urllib.error
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

# Ensure project root is on path
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import config

# Re-export existing working functions
from vlm_fallback import (
    image_to_base64,
    query_vlm_fallback,
    query_vlm_text_extraction,
)


def query_vlm_analysis(
    image: Image.Image,
    candidate_brands: List[str],
    vlm_url: str = config.VLM_URL,
    model_name: str = config.VLM_MODEL,
    timeout: int = config.VLM_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Union[str, float, bool]]]:
    """
    Full structured VLM analysis — asks the VLM to assess:
      - Which brand/logo appears to be present
      - What visible text identifies the brand
      - Does the logo visually resemble a candidate brand
      - Is the logo likely genuine, modified, or unrelated
      - What other visual evidence supports the identification

    Returns structured JSON with a scored confidence, or None on failure.

    Example return:
    {
        "detected_brand": "a200m",
        "brand_confidence": 0.92,
        "logo_present": True,
        "logo_confidence": 0.95,
        "text_evidence": "A200M visible in upper portion",
        "visual_evidence": "Gold gradient text on dark background matches reference",
        "reasoning_summary": "Strong text match + visual style consistency"
    }
    """
    if not candidate_brands:
        return None

    b64_img = image_to_base64(image)
    candidates_str = ", ".join(f'"{c}"' for c in candidate_brands)

    prompt = (
        "You are a brand logo verification system analyzing a screenshot or logo image.\n"
        f"Candidate brands to check against: [{candidates_str}]\n\n"
        "Analyze the image and respond ONLY with valid JSON in this exact format:\n"
        "{\n"
        '  "detected_brand": "BRAND_NAME or null if none detected",\n'
        '  "brand_confidence": 0.0,\n'
        '  "logo_present": true,\n'
        '  "logo_confidence": 0.0,\n'
        '  "text_evidence": "describe any text you can read",\n'
        '  "visual_evidence": "describe visual features: colors, shapes, style",\n'
        '  "reasoning_summary": "brief explanation of your identification"\n'
        "}\n\n"
        "Rules:\n"
        "- detected_brand MUST be from the candidate list or null\n"
        "- brand_confidence and logo_confidence are floats between 0.0 and 1.0\n"
        "- Be honest about uncertainty\n"
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [b64_img],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 768,
        },
    }

    try:
        req = urllib.request.Request(
            vlm_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            response_text = data.get("response", "").strip()

            # Handle thinking models (e.g. qwen3.5 with <think>...</think>)
            if "</think>" in response_text:
                response_text = response_text.split("</think>")[-1].strip()

            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response_text)

            # Validate and normalize the result
            detected = parsed.get("detected_brand")
            if detected and isinstance(detected, str):
                detected = detected.strip()
                # Enforce candidate list membership (case-insensitive)
                matched = None
                for cand in candidate_brands:
                    if cand.lower() == detected.lower():
                        matched = cand
                        break
                if not matched and detected.lower() != "null":
                    detected = None
                else:
                    detected = matched
            else:
                detected = None

            brand_conf = float(parsed.get("brand_confidence", 0.0))
            logo_present = bool(parsed.get("logo_present", False))
            logo_conf = float(parsed.get("logo_confidence", 0.0))

            return {
                "detected_brand": detected,
                "brand_confidence": min(1.0, max(0.0, brand_conf)),
                "logo_present": logo_present,
                "logo_confidence": min(1.0, max(0.0, logo_conf)),
                "text_evidence": str(parsed.get("text_evidence", "")),
                "visual_evidence": str(parsed.get("visual_evidence", "")),
                "reasoning_summary": str(parsed.get("reasoning_summary", "")),
            }

    except Exception as e:
        print(f"[VLM Analysis] Note: {e}")
        return None


def vlm_result_to_score(vlm_analysis: Optional[Dict]) -> float:
    """
    Convert a structured VLM analysis result into a single 0.0-1.0 score
    for use in fusion. Returns neutral 0.50 if VLM was not used or failed.
    """
    if vlm_analysis is None:
        return 0.50  # Neutral — no VLM data

    brand_conf = vlm_analysis.get("brand_confidence", 0.0)
    logo_conf = vlm_analysis.get("logo_confidence", 0.0)
    logo_present = vlm_analysis.get("logo_present", False)

    if not logo_present:
        return 0.20  # VLM says no logo present

    # Weighted combination of brand and logo confidence
    return float(min(1.0, 0.6 * brand_conf + 0.4 * logo_conf))


__all__ = [
    "image_to_base64",
    "query_vlm_fallback",
    "query_vlm_text_extraction",
    "query_vlm_analysis",
    "vlm_result_to_score",
]
