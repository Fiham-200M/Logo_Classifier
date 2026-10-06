"""
Website NLP Content Classifier.
Categorizes websites across industry taxonomy (Gambling, Casino, Sports Betting,
Financial, E-commerce, Entertainment, Technology, Social Media, Crypto, Adult, etc.)
using multi-modal textual evidence (HTML text, headings, OCR tokens, buttons).
"""

from typing import Dict, List, Any, Optional, Tuple
import re

import config
from website_intelligence.schemas import ContentCategory, ExtractedContent


# Industry Taxonomy & Lexicons
CATEGORY_TAXONOMY: Dict[str, List[str]] = {
    "Online Gambling": [
        "gambling", "betting", "taruhan", "judi", "rtp", "gacor", "slot", "slots", "pragmatic",
        "pgsoft", "habanero", "spadegaming", "maxwin", "jackpot", "deposit", "withdraw", "wd",
        "rollingan", "cashback", "bonus new member", "freebet", "turnover", "to x"
    ],
    "Casino": [
        "casino", "live casino", "baccarat", "roulette", "sicbo", "blackjack", "dragon tiger",
        "dealer", "table game", "spin", "croupier"
    ],
    "Sports Betting": [
        "sportsbook", "sports betting", "judi bola", "taruhan bola", "sbobet", "handicap", "parlay",
        "over under", "mix parlay", "odds", "fifa", "premier league", "champions league"
    ],
    "Poker": [
        "poker", "texas holdem", "dominoqq", "capsa", "cemeruang", "idnpoker", "flush", "full house"
    ],
    "Lottery": [
        "togel", "lottery", "toto", "singapore pools", "hongkong pools", "sydney pools", "4d", "3d", "2d", "colok"
    ],
    "Financial": [
        "banking", "finance", "investment", "loan", "forex", "trading", "stocks", "mutual fund", "credit card"
    ],
    "E-commerce": [
        "cart", "checkout", "add to cart", "buy now", "shopping", "store", "product", "shipping", "order"
    ],
    "Crypto": [
        "bitcoin", "crypto", "ethereum", "wallet", "usdt", "blockchain", "token", "nft", "binance", "metamask"
    ],
    "Technology": [
        "software", "cloud", "api", "developer", "saas", "hardware", "cybersecurity", "ai", "machine learning"
    ],
    "Entertainment": [
        "streaming", "movie", "music", "video", "cinema", "game", "gaming", "esports", "series"
    ],
    "Adult": [
        "porn", "xxx", "sex", "adult", "18+", "erotic", "webcam"
    ]
}


class WebsiteContentClassifier:
    """
    Classifies website domain, content, and industry purpose from fused HTML and OCR text.
    """

    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or {}

    def classify_content(
        self,
        extracted_content: Optional[ExtractedContent] = None,
        ocr_text: str = "",
        additional_tokens: Optional[List[str]] = None,
    ) -> ContentCategory:
        """
        Synthesizes text from HTML (title, headings, action buttons) and OCR tokens
        to categorize the website.
        """
        combined_text_parts = []
        if extracted_content:
            combined_text_parts.append(extracted_content.title)
            combined_text_parts.append(extracted_content.meta_description)
            combined_text_parts.extend(extracted_content.headings)
            combined_text_parts.extend(extracted_content.action_buttons)
            combined_text_parts.append(extracted_content.visible_text_summary)

        if ocr_text:
            combined_text_parts.append(ocr_text)

        if additional_tokens:
            combined_text_parts.extend(additional_tokens)

        full_corpus = " ".join(combined_text_parts).lower()

        scores: Dict[str, float] = {}
        detected_keywords_map: Dict[str, List[str]] = {}

        for category, keywords in CATEGORY_TAXONOMY.items():
            cat_score = 0.0
            hits: List[str] = []
            for kw in keywords:
                # Word boundary match
                pattern = r"\b" + re.escape(kw) + r"\b"
                matches = len(re.findall(pattern, full_corpus))
                if matches > 0:
                    hits.append(kw)
                    # Higher weight for specific distinctive terms
                    weight = 2.0 if len(kw.split()) > 1 or kw in ("gacor", "sbobet", "rtp", "baccarat", "jackpot") else 1.0
                    cat_score += matches * weight

            if hits:
                scores[category] = cat_score
                detected_keywords_map[category] = hits

        if not scores:
            return ContentCategory(
                primary="Other",
                secondary=[],
                confidence=0.20,
                detected_keywords=[],
            )

        # Sort categories by score
        sorted_cats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_cat, top_score = sorted_cats[0]

        # Calculate normalized confidence
        confidence = min(0.99, max(0.50, round(top_score / (top_score + 4.0), 3)))

        # Subcategories
        secondary: List[str] = []
        # If Online Gambling is top, related sub-niches like Casino or Sports Betting are secondary
        for cat, sc in sorted_cats[1:4]:
            if sc >= 2.0:
                secondary.append(cat)

        all_detected_kw = []
        for kw_list in detected_keywords_map.values():
            all_detected_kw.extend(kw_list)
        all_detected_kw = list(dict.fromkeys(all_detected_kw))[:15]

        return ContentCategory(
            primary=top_cat,
            secondary=secondary,
            confidence=confidence,
            detected_keywords=all_detected_kw,
        )
