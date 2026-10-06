"""
Vision-Language Model (VLM) Context Observer Module.
Extracts qualitative high-level visual understanding from full webpage screenshots.
Gracefully degrades if VLM server (Ollama / qwen2.5vl:7b) is offline.
"""

from typing import Dict, Any, Optional
import io
import json
import base64
import urllib.request
from PIL import Image

import config
from website_intelligence.schemas import VisualContextEvidence


def image_to_base64_jpeg(img: Image.Image, max_dim: int = 1024, quality: int = 85) -> str:
    """Downsamples and encodes image to Base64 JPEG for VLM consumption."""
    resized = img.copy().convert("RGB")
    resized.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class ScreenshotContextObserver:
    """
    VLM observer that inspects website screenshots to identify layout,
    promotional banners, and qualitative brand relationships.
    """

    def __init__(
        self,
        vlm_url: str = getattr(config, "VLM_URL", "http://localhost:11434/api/generate"),
        model_name: str = getattr(config, "VLM_MODEL", "qwen2.5vl:7b"),
        timeout: int = 30,
    ):
        self.vlm_url = vlm_url
        self.model_name = model_name
        self.timeout = timeout

    def observe_context(
        self,
        screenshot: Image.Image,
        brand_hint: Optional[str] = None
    ) -> VisualContextEvidence:
        """
        Queries VLM with screenshot to extract visual website context.
        Returns graceful empty evidence if VLM is unavailable.
        """
        try:
            b64_img = image_to_base64_jpeg(screenshot)
            prompt = (
                "You are a cybersecurity web forensics analyst inspecting this website screenshot.\n"
                "Answer concisely:\n"
                "1. What industry or category is this website (e.g. Online Casino, Sportsbook, E-commerce, Tech, Other)?\n"
                "2. Does it display prominent promotional bonuses, cashbacks, slot machines, or login/register forms?\n"
                f"3. What appears to be the primary brand or logo name displayed{' (checking for: ' + brand_hint + ')' if brand_hint else ''}?\n"
                "Provide a brief 2-sentence summary."
            )

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [b64_img],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 120,
                },
            }

            req = urllib.request.Request(
                self.vlm_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                response_text = data.get("response", "").strip()

            resp_lower = response_text.lower()
            is_promotional = any(k in resp_lower for k in ("bonus", "cashback", "promo", "slot", "login", "register"))
            site_type = "Online Gambling / Casino" if any(k in resp_lower for k in ("casino", "gambling", "slot", "betting")) else "Standard Webpage"

            return VisualContextEvidence(
                vlm_available=True,
                website_type_visual=site_type,
                promotional_content_detected=is_promotional,
                brand_visual_saliency=response_text[:120],
                raw_vlm_summary=response_text,
            )

        except Exception as e:
            # Graceful degradation when VLM is offline or times out
            return VisualContextEvidence(
                vlm_available=False,
                website_type_visual="Visual VLM Offline",
                promotional_content_detected=False,
                brand_visual_saliency="",
                raw_vlm_summary="",
            )
