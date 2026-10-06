"""
Vision-Language Model (VLM) Fallback Module.
Called selectively for ambiguous cases where SigLIP and OCR disagree or the margin
between top candidates is below the confidence threshold.
Queries local Ollama (e.g. qwen3.5:9b) with strict schema constraints.
"""

import io
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image

import config


def image_to_base64(image: Image.Image, max_dim: int = 512, min_dim: int = 140) -> str:
    """Ensure legible size for VLM (upscaling if small banner, downscaling if huge) and encode as JPEG base64."""
    img = image.convert("RGB")
    w, h = img.size

    # Upscale small banner/crop images so VLM can clearly read text
    if min(w, h) < min_dim and max(w, h) < max_dim:
        scale = min(3.0, min_dim / max(1, min(w, h)))
        if scale > 1.0:
            img = img.resize((int(round(w * scale)), int(round(h * scale))), Image.Resampling.LANCZOS)
            w, h = img.size

    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def query_vlm_text_extraction(
    image: Image.Image,
    vlm_url: str = config.VLM_URL,
    model_name: str = config.VLM_MODEL,
    timeout: int = config.VLM_TIMEOUT_SECONDS,
) -> Optional[List[str]]:
    """
    Ask the VLM to independently read ALL visible text in the logo image.
    This is separate from OCR — it uses the VLM's visual understanding.

    Returns:
        List of text strings the VLM sees, or None if VLM is unavailable.
    """
    b64_img = image_to_base64(image)

    prompt = (
        "You are a text reading system. Look at this logo image carefully.\n"
        "List ALL visible text, letters, numbers, and words you can see in the image.\n"
        "Include partial text, stylized text, abbreviations, and any characters.\n"
        "Respond ONLY with valid JSON in this exact format:\n"
        '{"texts": ["TEXT1", "TEXT2"], "description": "brief description of what you see"}\n'
        'If no text is visible, return: {"texts": [], "description": "no text visible"}\n'
    )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": [b64_img],
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 512,
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

            import re
            json_match = re.search(r"\{.*?\}", response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group(0))
            else:
                parsed = json.loads(response_text)

            texts = parsed.get("texts", [])
            desc = parsed.get("description", "")

            # Ensure all entries are strings
            clean_texts = [str(t).strip() for t in texts if str(t).strip()]

            if desc:
                print(f"[VLM Text] Description: {desc}")

            return clean_texts

    except Exception as e:
        print(f"[VLM Text Extraction] Note: {e}")
        return None


def query_vlm_fallback(
    image: Image.Image,
    candidate_brands: List[str],
    reference_image: Optional[Image.Image] = None,
    reference_brand: Optional[str] = None,
    vlm_url: str = config.VLM_URL,
    model_name: str = config.VLM_MODEL,
    timeout: int = config.VLM_TIMEOUT_SECONDS,
) -> Optional[Dict[str, Union[str, float]]]:
    """
    Query local VLM to disambiguate between specific candidate brands.
    When reference_image and reference_brand are provided, performs a strict
    two-image visual comparison to detect icon-swaps and logo modifications.

    Returns:
        Dict: {"candidate": str, "confidence": float, "reason": str, "is_competitor": bool}
        or None if VLM is unavailable or call fails.
    """
    if not candidate_brands:
        return None

    b64_img = image_to_base64(image)
    candidates_str = ", ".join(f'"{c}"' for c in candidate_brands)

    if reference_image is not None and reference_brand:
        b64_ref = image_to_base64(reference_image)
        images_payload = [b64_img, b64_ref]
        prompt = (
            "You are an expert anti-phishing logo verification system comparing two images:\n"
            f"- Image 1: The query/test image.\n"
            f"- Image 2: The official reference logo for brand '{reference_brand}'.\n"
            f"Candidate brands: [{candidates_str}].\n\n"
            "Compare Image 1 and Image 2 with microscopic precision:\n"
            "1. Inspect the main icon, mascot, emblem, or symbol in both images (e.g. eagle vs dragon, lion vs rooster).\n"
            "2. Count and locate all secondary graphic elements: swords, weapons, hilts, shields, stars, crowns, or banners.\n"
            "   Check if Image 1 has ADDED or REMOVED any graphic elements (e.g. an extra sword hilt in the corner, extra flare, different weapon count).\n"
            "3. Inspect the text, numbers, and spelling in Image 1.\n"
            "4. If Image 1 has an icon swap, added/removed graphic elements, different weapon count, or altered text, it is a MODIFIED SPOOF/COMPETITOR.\n"
            "   In that case, set is_exact_match to false, candidate to null, and is_competitor to true.\n"
            f"5. If and only if Image 1 matches Image 2's iconography, weapons, text, and structure completely, set candidate to '{reference_brand}'.\n\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "candidate": "BRAND_NAME or null",\n'
            '  "is_exact_match": true,\n'
            '  "graphic_differences": "describe any differences or added/removed elements, or none",\n'
            '  "is_competitor": false,\n'
            '  "confidence": 0.95,\n'
            '  "reason": "concise explanation"\n'
            "}\n"
        )
    else:
        images_payload = [b64_img]
        prompt = (
            "You are a strict anti-phishing logo verification system.\n"
            f"Candidate brands: [{candidates_str}].\n"
            "Look at this logo image carefully:\n"
            "- If the logo matches one of the candidate brands exactly, set candidate to that brand name.\n"
            "- If the logo is a competitor, copycat, or has different text/numbers (e.g. 'NUSA2111' vs 'nusa211'), set candidate to null and explain.\n\n"
            "Respond in valid JSON format:\n"
            '{"candidate": "BRAND_NAME or null", "visible_text": "exact text seen", "is_competitor": false, "confidence": 0.95, "reason": "concise explanation"}\n'
        )

    payload = {
        "model": model_name,
        "prompt": prompt,
        "images": images_payload,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 1024,
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

            if "{" in response_text and "}" in response_text:
                json_str = response_text[response_text.find("{"):response_text.rfind("}") + 1]
                parsed = json.loads(json_str)
            else:
                return None

            candidate = parsed.get("candidate")
            is_comp = bool(parsed.get("is_competitor", False))
            is_exact = parsed.get("is_exact_match", True)
            icons_match = parsed.get("icons_match", True)
            diffs = parsed.get("graphic_differences") or parsed.get("differences") or ""
            reason = parsed.get("reason", "")
            vis_text = parsed.get("visible_text", "")

            # If not exact match, or icons do not match, or differences found
            has_diff = bool(diffs and "none" not in diffs.lower() and "identical" not in diffs.lower())
            if is_exact is False or icons_match is False or has_diff:
                explanation = reason or diffs or f"Graphic modification detected against {reference_brand}"
                return {
                    "candidate": None,
                    "confidence": float(parsed.get("confidence", 0.95)),
                    "is_competitor": True,
                    "visible_text": vis_text,
                    "reason": f"Logo Modification/Spoof: {explanation}",
                }


            if candidate and isinstance(candidate, str) and candidate.lower() != "null":
                # Find matching candidate from candidates list
                matched = None
                for cand in candidate_brands:
                    if cand.lower() == candidate.lower().strip():
                        matched = cand
                        break
                if matched:
                    return {
                        "candidate": matched,
                        "confidence": float(parsed.get("confidence", 0.85)),
                        "is_competitor": False,
                        "visible_text": vis_text,
                        "reason": reason or "Visual feature alignment by VLM",
                    }

            # If VLM identified competitor or null candidate
            return {
                "candidate": None,
                "confidence": float(parsed.get("confidence", 0.90)),
                "is_competitor": True,
                "visible_text": vis_text,
                "reason": reason or "Competitor or copycat logo detected by VLM",
            }

    except Exception as e:
        print(f"[VLM Fallback] Note: {e}")
        return None

