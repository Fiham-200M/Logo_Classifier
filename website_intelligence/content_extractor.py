"""
Website Content Extractor Module.
Extracts title, meta descriptions, headings, navigation elements, action buttons,
and structured text summaries from HTML content or raw page dumps.
"""

from typing import Dict, List, Any, Optional
import re
from urllib.parse import urlparse

from website_intelligence.schemas import ExtractedContent

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False


class WebsiteContentExtractor:
    """
    Extracts structured content elements from HTML and URL contexts.
    """

    @staticmethod
    def extract_from_html(html_content: str, url: str = "") -> ExtractedContent:
        domain = ""
        if url:
            domain = urlparse(url).netloc

        if not html_content:
            return ExtractedContent(url=url, domain=domain, html_available=False)

        title = ""
        meta_desc = ""
        headings: List[str] = []
        nav_labels: List[str] = []
        action_buttons: List[str] = []
        visible_tokens: List[str] = []
        links: List[str] = []

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html_content, "html.parser")

            # Title
            if soup.title and soup.title.string:
                title = soup.title.string.strip()

            # Meta Description
            desc_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)}) or \
                       soup.find("meta", attrs={"property": re.compile(r"og:description", re.I)})
            if desc_tag and desc_tag.get("content"):
                meta_desc = desc_tag["content"].strip()

            # Headings (h1 - h3)
            for h in soup.find_all(["h1", "h2", "h3"]):
                txt = h.get_text(strip=True)
                if txt and len(txt) > 2:
                    headings.append(txt)

            # Navigation labels
            for nav in soup.find_all(["nav", "header"]):
                for a in nav.find_all("a"):
                    txt = a.get_text(strip=True)
                    if txt and len(txt) > 1 and len(txt) < 40:
                        nav_labels.append(txt)

            # Action Buttons (Login, Register, Daftar, Masuk, Deposit)
            for btn in soup.find_all(["button", "a", "input"]):
                btn_txt = ""
                if btn.name == "input" and btn.get("type") in ("button", "submit"):
                    btn_txt = btn.get("value", "")
                else:
                    btn_txt = btn.get_text(strip=True)
                if btn_txt and 2 <= len(btn_txt) <= 30:
                    action_buttons.append(btn_txt)

            # Links
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href and not href.startswith("javascript:") and not href.startswith("#"):
                    links.append(href)

            # Remove scripts and style for visible text
            for s in soup(["script", "style", "noscript", "svg"]):
                s.decompose()
            raw_text = soup.get_text(separator=" ", strip=True)
        else:
            # Fast regex fallback if bs4 is unavailable
            m_title = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
            if m_title:
                title = m_title.group(1).strip()

            m_desc = re.search(r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']', html_content, re.I)
            if m_desc:
                meta_desc = m_desc.group(1).strip()

            for h in re.findall(r"<h[1-3][^>]*>(.*?)</h[1-3]>", html_content, re.I | re.DOTALL):
                clean_h = re.sub(r"<[^>]+>", "", h).strip()
                if clean_h:
                    headings.append(clean_h)

            raw_text = re.sub(r"<[^>]+>", " ", html_content)
            raw_text = re.sub(r"\s+", " ", raw_text).strip()

        # Deduplicate while preserving order
        headings = list(dict.fromkeys(headings))[:15]
        nav_labels = list(dict.fromkeys(nav_labels))[:25]
        action_buttons = list(dict.fromkeys(action_buttons))[:20]
        links = list(dict.fromkeys(links))[:50]

        summary_words = raw_text.split()[:120]
        visible_summary = " ".join(summary_words)

        return ExtractedContent(
            url=url,
            domain=domain,
            title=title,
            meta_description=meta_desc,
            headings=headings,
            navigation_labels=nav_labels,
            action_buttons=action_buttons,
            visible_text_summary=visible_summary,
            full_text_sample=raw_text[:2000],
            links=links,
            html_available=True,
        )
