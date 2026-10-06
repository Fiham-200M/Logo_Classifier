"""
Comprehensive 20-Scenario Automated Forensic Test Suite.
Validates Multi-Modal Anti-Spoofing Forensics & Website Intelligence Engine
against all 20 required verification scenarios in Section 24.
"""

import sys
import os
import unittest
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Ensure project root is in sys.path
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from verify_logo import LogoForensicsEngine
from preprocessing.media_normalizer import MediaNormalizer
from preprocessing.gif_processor import GIFProcessor
from screenshot_scanner import ScreenshotScanner
from website_intelligence.website_analyzer import WebsiteAnalyzer


class TestFullForensicSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 65)
        print(" [TestSuite] Initializing 20-Scenario Forensics Engine on GPU...")
        print("=" * 65)
        cls.engine = LogoForensicsEngine()
        cls.scanner = ScreenshotScanner(ocr_engine=cls.engine.ocr._engine, ref_store=cls.engine.ref_store)
        cls.fixtures_dir = _project_root / "tests" / "fixtures"
        cls.fixtures_dir.mkdir(parents=True, exist_ok=True)
        cls._create_all_fixtures()

    @classmethod
    def _create_all_fixtures(cls):
        """Generates all 20 test fixtures deterministically."""
        base_path = _project_root / "logos" / "a200m.png"
        cls.base_img = Image.open(base_path).convert("RGBA")
        w, h = cls.base_img.size

        # 1. Exact official logo
        cls.path_01_exact = cls.fixtures_dir / "01_exact_a200m.png"
        cls.base_img.save(cls.path_01_exact)

        # 2. Resized official logo (1.5x)
        cls.path_02_resized = cls.fixtures_dir / "02_resized_a200m.png"
        cls.base_img.resize((int(w * 1.5), int(h * 1.5)), Image.Resampling.LANCZOS).save(cls.path_02_resized)

        # 3. Compressed official logo (JPEG q=75)
        cls.path_03_compressed = cls.fixtures_dir / "03_compressed_a200m.jpg"
        bg_w = MediaNormalizer.normalize_media(cls.base_img)["canvas_white"]
        bg_w.convert("RGB").save(cls.path_03_compressed, "JPEG", quality=75)

        # 4. Recolored logo (Channel swap: red <-> blue, gold -> cyan)
        rgb_arr = np.array(cls.base_img.convert("RGB")).copy()
        rgb_arr[:, :, [0, 2]] = rgb_arr[:, :, [2, 0]]
        cls.path_04_recolored = cls.fixtures_dir / "04_recolored_a200m.png"
        Image.fromarray(rgb_arr).save(cls.path_04_recolored)

        # 5. Modified typography (Font redraw clone)
        cls.path_05_typography = cls.fixtures_dir / "05_typography_modified.png"
        typo_img = cls.base_img.copy()
        draw_t = ImageDraw.Draw(typo_img)
        draw_t.rectangle([(20, 15), (140, 55)], fill=(20, 20, 20, 255))
        draw_t.text((25, 20), "A200M", fill=(220, 180, 50, 255))
        typo_img.save(cls.path_05_typography)

        # 6. Added icon/graphic (Injected shield badge)
        cls.path_06_added_icon = cls.fixtures_dir / "06_added_icon_a200m.png"
        icon_img = cls.base_img.copy()
        draw_i = ImageDraw.Draw(icon_img)
        draw_i.rectangle([(w - 60, 5), (w - 5, 65)], fill=(220, 20, 20, 255), outline=(255, 255, 255, 255), width=3)
        draw_i.polygon([(w - 50, 50), (w - 32, 15), (w - 15, 50)], fill=(255, 215, 0, 255))
        icon_img.save(cls.path_06_added_icon)

        # 7. Modified number (A200M -> A500M)
        cls.path_07_changed_number = cls.fixtures_dir / "07_changed_number_a500m.png"
        num_img = cls.base_img.copy()
        draw_n = ImageDraw.Draw(num_img)
        draw_n.rectangle([(20, 15), (140, 55)], fill=(20, 20, 20, 255))
        draw_n.text((25, 20), "A500M", fill=(240, 190, 60, 255))
        num_img.save(cls.path_07_changed_number)

        # 8. Unrelated competitor logo (ACME CORP triangle)
        cls.path_08_unrelated = cls.fixtures_dir / "08_unrelated_competitor.png"
        unrel = Image.new("RGB", (300, 300), (240, 240, 245))
        draw_u = ImageDraw.Draw(unrel)
        draw_u.polygon([(150, 40), (250, 240), (50, 240)], fill=(30, 80, 200), outline=(10, 30, 100), width=4)
        draw_u.ellipse([(110, 100), (190, 180)], fill=(220, 50, 50))
        draw_u.text((95, 255), "ACME CORP", fill=(40, 40, 40))
        unrel.save(cls.path_08_unrelated)

        # 9. Visually similar competitor logo (A900M borrowing brand styling)
        cls.path_09_similar_competitor = cls.fixtures_dir / "09_similar_competitor.png"
        sim_img = cls.base_img.copy()
        draw_s = ImageDraw.Draw(sim_img)
        draw_s.rectangle([(15, 10), (160, 60)], fill=(18, 18, 18, 255))
        draw_s.text((25, 20), "A900M", fill=(230, 185, 55, 255))
        sim_img.save(cls.path_09_similar_competitor)

        # 10. Animated transparent GIF
        cls.path_10_animated_gif = cls.fixtures_dir / "10_animated_logo.gif"
        frames = []
        for i in range(4):
            f = cls.base_img.copy()
            d = ImageDraw.Draw(f)
            d.ellipse([(w - 30 + i * 2, 15), (w - 15 + i * 2, 30)], fill=(255, 200, 0, 255))
            frames.append(f.convert("RGB"))
        frames[0].save(cls.path_10_animated_gif, save_all=True, append_images=frames[1:], duration=120, loop=0)

        # 11. Transparent PNG (Actual alpha transparency)
        cls.path_11_transparent_png = cls.fixtures_dir / "11_transparent_brand.png"
        trans_rgba = np.array(cls.base_img.convert("RGBA")).copy()
        # Mark near-black background pixels as transparent
        bg_mask = (trans_rgba[:, :, 0] < 35) & (trans_rgba[:, :, 1] < 35) & (trans_rgba[:, :, 2] < 35)
        trans_rgba[bg_mask, 3] = 0
        Image.fromarray(trans_rgba).save(cls.path_11_transparent_png)

        # 12. Black-background logo
        cls.path_12_black_bg = cls.fixtures_dir / "12_black_bg_a200m.png"
        bg_black = Image.new("RGBA", cls.base_img.size, (0, 0, 0, 255))
        bg_black.paste(cls.base_img, (0, 0), cls.base_img)
        bg_black.convert("RGB").save(cls.path_12_black_bg)

        # 13. White-background logo
        cls.path_13_white_bg = cls.fixtures_dir / "13_white_bg_a200m.png"
        bg_white = Image.new("RGBA", cls.base_img.size, (255, 255, 255, 255))
        bg_white.paste(cls.base_img, (0, 0), cls.base_img)
        bg_white.convert("RGB").save(cls.path_13_white_bg)

        # 14. Same logo with minor compression artifacts (WebP q=88)
        cls.path_14_minor_compression = cls.fixtures_dir / "14_minor_compression_a200m.webp"
        cls.base_img.convert("RGB").save(cls.path_14_minor_compression, "WEBP", quality=88)

        # 15. Same logo with different canvas dimensions (Padded canvas)
        cls.path_15_padded_canvas = cls.fixtures_dir / "15_padded_canvas_a200m.png"
        padded = Image.new("RGBA", (w + 120, h + 80), (20, 20, 20, 255))
        padded.paste(cls.base_img, (60, 40), cls.base_img)
        padded.save(cls.path_15_padded_canvas)

        # 16. Favicon version of protected brand
        cls.path_16_favicon_official = _project_root / "Favicon" / "001_a200m-seru.fun_favicon.png"

        # 17. Modified favicon (Reversed hue)
        cls.path_17_modified_favicon = cls.fixtures_dir / "17_modified_favicon.png"
        if cls.path_16_favicon_official.exists():
            fav_arr = np.array(Image.open(cls.path_16_favicon_official).convert("RGB")).copy()
            fav_arr[:, :, [0, 2]] = fav_arr[:, :, [2, 0]]
            Image.fromarray(fav_arr).save(cls.path_17_modified_favicon)
        else:
            fav_m = Image.new("RGB", (32, 32), (255, 0, 100))
            fav_m.save(cls.path_17_modified_favicon)

        # 18. Full webpage screenshot with single protected logo
        cls.path_18_screenshot_single = cls.fixtures_dir / "18_screenshot_single.png"
        ss1 = Image.new("RGB", (1280, 800), (25, 27, 34))
        d_ss1 = ImageDraw.Draw(ss1)
        # Header bar
        d_ss1.rectangle([(0, 0), (1280, 90)], fill=(15, 17, 22))
        # Place logo in header
        logo_crop = cls.base_img.resize((180, 60), Image.Resampling.LANCZOS)
        ss1.paste(logo_crop.convert("RGB"), (40, 15))
        # Nav links
        d_ss1.text((280, 35), "HOME   SLOTS   LIVE CASINO   PROMOTIONS   VIP", fill=(200, 200, 200))
        d_ss1.rectangle([(1120, 25), (1240, 65)], fill=(212, 175, 55))
        d_ss1.text((1145, 38), "DEPOSIT", fill=(0, 0, 0))
        ss1.save(cls.path_18_screenshot_single)

        # 19. Full webpage screenshot with multiple logos (Header + Footer)
        cls.path_19_screenshot_multi = cls.fixtures_dir / "19_screenshot_multi.png"
        ss2 = ss1.copy()
        d_ss2 = ImageDraw.Draw(ss2)
        # Footer bar
        d_ss2.rectangle([(0, 700), (1280, 800)], fill=(12, 14, 18))
        d_ss2.text((40, 720), "OFFICIAL PARTNERS:", fill=(150, 150, 150))
        # Paste second logo in footer
        asia_path = _project_root / "logos" / "asia100.png"
        if asia_path.exists():
            asia_img = Image.open(asia_path).convert("RGB").resize((160, 50), Image.Resampling.LANCZOS)
            ss2.paste(asia_img, (220, 715))
        ss2.save(cls.path_19_screenshot_multi)

        # 20. Website screenshot with no protected logo (Tech news blog)
        cls.path_20_screenshot_no_logo = cls.fixtures_dir / "20_screenshot_no_logo.png"
        ss3 = Image.new("RGB", (1280, 800), (255, 255, 255))
        d_ss3 = ImageDraw.Draw(ss3)
        d_ss3.rectangle([(0, 0), (1280, 80)], fill=(33, 150, 243))
        d_ss3.text((40, 25), "TECH WORLD INSIGHTS - DAILY DEVELOPER NEWS", fill=(255, 255, 255))
        d_ss3.rectangle([(50, 120), (800, 450)], fill=(245, 245, 245), outline=(220, 220, 220))
        d_ss3.text((80, 160), "Artificial Intelligence breakthrough in agentic architectures.", fill=(30, 30, 30))
        ss3.save(cls.path_20_screenshot_no_logo)

    # ============================================================
    # 20 DETAILED FORENSIC VERIFICATION SCENARIOS
    # ============================================================

    def test_01_exact_official_logo(self):
        """Scenario 1: Exact official logo -> MATCH"""
        res = self.engine.verify(self.path_01_exact, skip_vlm=True)
        print(f"\n[Scenario 01] Exact Official Logo: Verdict={res['verdict']}, Brand={res['brand_id']}, Threat={res['threat_type']}")
        self.assertEqual(res["verdict"], "MATCH")
        self.assertEqual(res["threat_type"], "EXACT_REPLICA")
        self.assertGreaterEqual(res["forensic_metrics"]["siglip2_semantic_score"], 0.98)

    def test_02_resized_official_logo(self):
        """Scenario 2: Resized official logo -> MATCH"""
        res = self.engine.verify(self.path_02_resized, skip_vlm=True)
        print(f"[Scenario 02] Resized Logo: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertEqual(res["verdict"], "MATCH")

    def test_03_compressed_official_logo(self):
        """Scenario 3: Compressed official logo -> MATCH"""
        res = self.engine.verify(self.path_03_compressed, skip_vlm=True)
        print(f"[Scenario 03] Compressed Logo: Verdict={res['verdict']}, Conf={res['confidence_score']}")
        self.assertEqual(res["verdict"], "MATCH")

    def test_04_recolored_logo(self):
        """Scenario 4: Recolored logo -> REVIEW (COLOR_SPOOF)"""
        res = self.engine.verify(self.path_04_recolored, skip_vlm=True)
        print(f"[Scenario 04] Recolored Logo: Verdict={res['verdict']}, Threat={res['threat_type']}, DeltaE={res['forensic_metrics']['cielab_delta_e']}")
        self.assertEqual(res["verdict"], "REVIEW")
        self.assertIn(res["threat_type"], ["COLOR_SPOOF", "VECTOR_REDRAW"])
        self.assertGreater(res["forensic_metrics"]["cielab_delta_e"], 5.0)

    def test_05_modified_typography(self):
        """Scenario 5: Modified typography -> REVIEW (VECTOR_REDRAW)"""
        res = self.engine.verify(self.path_05_typography, skip_vlm=True)
        print(f"[Scenario 05] Modified Typography: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertEqual(res["verdict"], "REVIEW")
        self.assertIn(res["threat_type"], ["VECTOR_REDRAW", "TEXT_MODIFICATION", "LAYOUT_MODIFICATION", "ELEMENT_INJECTION", "COLOR_SPOOF"])

    def test_06_added_icon(self):
        """Scenario 6: Added icon/graphic -> REVIEW (ELEMENT_INJECTION)"""
        res = self.engine.verify(self.path_06_added_icon, skip_vlm=True)
        print(f"[Scenario 06] Added Icon: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertEqual(res["verdict"], "REVIEW")
        self.assertIn(res["threat_type"], ["ELEMENT_INJECTION", "COLOR_SPOOF", "VECTOR_REDRAW"])

    def test_07_modified_number(self):
        """Scenario 7: Modified number/text -> REVIEW (NUMBER_COLLISION)"""
        res = self.engine.verify(self.path_07_changed_number, skip_vlm=True)
        print(f"[Scenario 07] Modified Number: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertEqual(res["verdict"], "REVIEW")
        self.assertIn(res["threat_type"], ["NUMBER_COLLISION", "TEXT_MODIFICATION"])

    def test_08_unrelated_competitor_logo(self):
        """Scenario 8: Unrelated competitor logo -> UNKNOWN / UNRELATED"""
        res = self.engine.verify(self.path_08_unrelated, skip_vlm=True)
        print(f"[Scenario 08] Unrelated Competitor: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertEqual(res["verdict"], "UNKNOWN")
        self.assertIn(res["threat_type"], ["UNRELATED", "UNKNOWN"])

    def test_09_visually_similar_competitor(self):
        """Scenario 9: Visually similar competitor logo -> REVIEW / UNKNOWN"""
        res = self.engine.verify(self.path_09_similar_competitor, skip_vlm=True)
        print(f"[Scenario 09] Similar Competitor: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertIn(res["verdict"], ["REVIEW", "UNKNOWN"])

    def test_10_animated_transparent_gif(self):
        """Scenario 10: Animated transparent GIF -> multi-frame composite verification"""
        norm = MediaNormalizer.normalize_media(self.path_10_animated_gif)
        self.assertTrue(norm["metadata"]["is_animated"])
        self.assertGreaterEqual(norm["metadata"]["frames_processed"], 4)
        res = self.engine.verify(self.path_10_animated_gif, skip_vlm=True)
        print(f"[Scenario 10] Animated GIF: Verdict={res['verdict']}, Frames={norm['metadata']['frames_processed']}")
        self.assertIn(res["verdict"], ["MATCH", "REVIEW"])

    def test_11_transparent_png(self):
        """Scenario 11: Transparent PNG -> Alpha mask extraction & dual canvases"""
        norm = MediaNormalizer.normalize_media(self.path_11_transparent_png)
        self.assertTrue(norm["metadata"]["alpha_channel_detected"])
        self.assertIsNotNone(norm["alpha_mask"])
        self.assertIsNotNone(norm["canvas_white"])
        self.assertIsNotNone(norm["canvas_black"])

    def test_12_black_background_logo(self):
        """Scenario 12: Black-background logo -> MATCH"""
        res = self.engine.verify(self.path_12_black_bg, skip_vlm=True)
        print(f"[Scenario 12] Black Background: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertEqual(res["verdict"], "MATCH")

    def test_13_white_background_logo(self):
        """Scenario 13: White-background logo -> MATCH"""
        res = self.engine.verify(self.path_13_white_bg, skip_vlm=True)
        print(f"[Scenario 13] White Background: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertEqual(res["verdict"], "MATCH")

    def test_14_minor_compression_artifacts(self):
        """Scenario 14: Minor compression artifacts (WebP) -> MATCH"""
        res = self.engine.verify(self.path_14_minor_compression, skip_vlm=True)
        print(f"[Scenario 14] Minor Compression: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertEqual(res["verdict"], "MATCH")

    def test_15_padded_canvas_dimensions(self):
        """Scenario 15: Different canvas dimensions (padded) -> MATCH or REVIEW (brand correctly mapped)"""
        res = self.engine.verify(self.path_15_padded_canvas, skip_vlm=True)
        print(f"[Scenario 15] Padded Canvas: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertIn(res["verdict"], ["MATCH", "REVIEW"])
        self.assertEqual(res["brand_id"], "a200m")

    def test_16_favicon_protected_brand(self):
        """Scenario 16: Favicon version of protected brand -> MATCH"""
        if not self.path_16_favicon_official.exists():
            self.skipTest("Official favicon file not found")
        res = self.engine.verify(self.path_16_favicon_official, skip_vlm=True)
        print(f"[Scenario 16] Official Favicon: Verdict={res['verdict']}, Brand={res['brand_id']}")
        self.assertEqual(res["verdict"], "MATCH")
        self.assertEqual(res["brand_id"], "a200m")

    def test_17_modified_favicon(self):
        """Scenario 17: Modified favicon -> REVIEW / UNKNOWN"""
        res = self.engine.verify(self.path_17_modified_favicon, skip_vlm=True)
        print(f"[Scenario 17] Modified Favicon: Verdict={res['verdict']}, Threat={res['threat_type']}")
        self.assertIn(res["verdict"], ["REVIEW", "UNKNOWN"])

    def test_18_webpage_screenshot_single_logo(self):
        """Scenario 18: Full webpage screenshot containing protected logo -> Candidate detected & brand mapped"""
        scan_res = self.scanner.scan_screenshot(self.path_18_screenshot_single, engine=self.engine)
        print(f"[Scenario 18] Webpage Single Logo: Verdict={scan_res['verdict']}, Detections={scan_res['detections_count']}, Brand={scan_res['primary_brand']}")
        self.assertIn(scan_res["verdict"], ["MATCH", "REVIEW"])
        self.assertGreaterEqual(scan_res["detections_count"], 1)
        self.assertEqual(scan_res["primary_brand"], "a200m")

    def test_19_webpage_screenshot_multiple_logos(self):
        """Scenario 19: Full webpage screenshot containing multiple logos -> Multiple candidate detections"""
        scan_res = self.scanner.scan_screenshot(self.path_19_screenshot_multi, engine=self.engine)
        print(f"[Scenario 19] Webpage Multi Logo: Verdict={scan_res['verdict']}, Detections={scan_res['detections_count']}")
        self.assertIn(scan_res["verdict"], ["MATCH", "REVIEW"])
        self.assertGreaterEqual(scan_res["detections_count"], 2)

    def test_20_website_screenshot_no_logo(self):
        """Scenario 20: Website screenshot with no protected logo -> 0 protected detections & UNKNOWN"""
        scan_res = self.scanner.scan_screenshot(self.path_20_screenshot_no_logo, engine=self.engine)
        print(f"[Scenario 20] Webpage No Logo: Verdict={scan_res['verdict']}, Detections={scan_res['detections_count']}")
        self.assertEqual(scan_res["verdict"], "UNKNOWN")
        self.assertEqual(scan_res["detections_count"], 0)


if __name__ == "__main__":
    unittest.main()
