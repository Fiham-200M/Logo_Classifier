"""
Vision-Language Model (VLM) Forensic Module.
Model: qwen2.5vl:7b (via Ollama)
Performs side-by-side microscopic comparison between Candidate Image and Official Reference Logo.
Inspects Typography, Icons/Graphics, Layout, Color Palette, and Alphanumerics.
"""

from typing import Dict, Any, Optional
import io
import json
import base64
import urllib.request
from PIL import Image

import config


def image_to_base64_jpeg(img: Image.Image, quality: int = 95) -> str:
    """Encodes PIL image to Base64 JPEG string."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLMModel:
    def __init__(
        self,
        model_name: str = config.VLM_MODEL,
        vlm_url: str = config.VLM_URL,
        timeout: int = config.VLM_TIMEOUT_SECONDS,
    ):
        self.model_name = model_name
        self.vlm_url = vlm_url
        self.timeout = timeout
        self._service_available = None

    def forensic_compare(
        self,
        candidate_image: Image.Image,
        reference_image: Image.Image,
        brand_name: str,
        ocr_hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Runs microscopic side-by-side forensic analysis.
        Image A = Official Reference Logo
        Image B = Candidate Query Image
        """
        if self._service_available is False:
            return {
                "visually_related": True,
                "micro_modification_detected": False,
                "typography_difference": False,
                "icon_difference": False,
                "color_difference": False,
                "layout_difference": False,
                "text_difference": False,
                "explanation": "VLM service offline (cached fallback)",
                "raw_response": None,
            }
        b64_ref = image_to_base64_jpeg(reference_image)
        b64_cand = image_to_base64_jpeg(candidate_image)

        prompt = (
            "You are a forensic logo verification and anti-spoofing analyst comparing two images:\n"
            f"- Image 1: The OFFICIAL REFERENCE LOGO for brand '{brand_name}'.\n"
            "- Image 2: The CANDIDATE/QUERY image under verification.\n"
            f"{f'Candidate OCR hint: {ocr_hint}' if ocr_hint else ''}\n\n"
            "Examine both images microscopically across 5 forensic axes:\n"
            "1. Typography: Are letter/number shapes, stroke curves, font geometry, and serifs identical or redrawn/altered?\n"
            "2. Graphic elements: Are icons, mascots, shields, swords, crowns, stars, wings identical or modified/injected?\n"
            "3. Layout: Are spacing, relative proportions, and alignment preserved?\n"
            "4. Color: Are the primary brand colors, metallic gradients, and accents consistent?\n"
            "5. Text & Numbers: Does the spelling, brand text, or number match exactly?\n\n"
            "Output your findings in STRICT JSON format:\n"
            "{\n"
            '  "visually_related": true,\n'
            '  "micro_modification_detected": false,\n'
            '  "typography_difference": false,\n'
            '  "icon_difference": false,\n'
            '  "color_difference": false,\n'
            '  "layout_difference": false,\n'
            '  "text_difference": false,\n'
            '  "explanation": "concise description of any findings or confirmation of exact match"\n'
            "}\n"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "images": [b64_ref, b64_cand],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 512},
        }

        try:
            req = urllib.request.Request(
                self.vlm_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                resp_text = data.get("response", "").strip()

                if "</think>" in resp_text:
                    resp_text = resp_text.split("</think>")[-1].strip()

                if "{" in resp_text and "}" in resp_text:
                    json_str = resp_text[resp_text.find("{"):resp_text.rfind("}") + 1]
                    parsed = json.loads(json_str)
                    return {
                        "visually_related": bool(parsed.get("visually_related", True)),
                        "micro_modification_detected": bool(parsed.get("micro_modification_detected", False)),
                        "typography_difference": bool(parsed.get("typography_difference", False)),
                        "icon_difference": bool(parsed.get("icon_difference", False)),
                        "color_difference": bool(parsed.get("color_difference", False)),
                        "layout_difference": bool(parsed.get("layout_difference", False)),
                        "text_difference": bool(parsed.get("text_difference", False)),
                        "explanation": str(parsed.get("explanation", "Forensic visual comparison completed.")),
                        "raw_response": parsed,
                    }
        except Exception as e:
            self._service_available = False
            return {
                "visually_related": True,
                "micro_modification_detected": False,
                "typography_difference": False,
                "icon_difference": False,
                "color_difference": False,
                "layout_difference": False,
                "text_difference": False,
                "explanation": f"VLM unavailable or skipped ({e})",
                "raw_response": None,
            }

        return {
            "visually_related": True,
            "micro_modification_detected": False,
            "typography_difference": False,
            "icon_difference": False,
            "color_difference": False,
            "layout_difference": False,
            "text_difference": False,
            "explanation": "VLM parsing fallback",
            "raw_response": None,
        }
