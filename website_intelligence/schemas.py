"""
Website Intelligence Schemas.
Standardized data contracts for website content, classification categories,
visual context, and unified forensic profiles.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict


@dataclass
class ContentCategory:
    primary: str = "Unknown"
    secondary: List[str] = field(default_factory=list)
    confidence: float = 0.0
    detected_keywords: List[str] = field(default_factory=list)


@dataclass
class ExtractedContent:
    url: str = ""
    domain: str = ""
    title: str = ""
    meta_description: str = ""
    headings: List[str] = field(default_factory=list)
    navigation_labels: List[str] = field(default_factory=list)
    action_buttons: List[str] = field(default_factory=list)
    visible_text_summary: str = ""
    full_text_sample: str = ""
    links: List[str] = field(default_factory=list)
    html_available: bool = False


@dataclass
class VisualContextEvidence:
    vlm_available: bool = False
    website_type_visual: str = ""
    promotional_content_detected: bool = False
    brand_visual_saliency: str = ""
    raw_vlm_summary: str = ""


@dataclass
class WebsiteIdentityProfile:
    domain: str = ""
    url: str = ""
    brand_detected: Optional[str] = None
    brand_confidence: float = 0.0
    verdict: str = "UNKNOWN"
    logos: List[Dict[str, Any]] = field(default_factory=list)
    content: ContentCategory = field(default_factory=ContentCategory)
    text_evidence: Dict[str, Any] = field(default_factory=dict)
    visual_context: VisualContextEvidence = field(default_factory=VisualContextEvidence)
    processing_time_sec: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
