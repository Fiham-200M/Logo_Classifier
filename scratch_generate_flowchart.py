import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_system_flow_chart(output_path: str = "system_architecture_flow.png"):
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica']
    fig, ax = plt.subplots(figsize=(24, 14), dpi=300)
    fig.patch.set_facecolor("#0B1120") # Sleek deep midnight theme
    ax.set_facecolor("#0B1120")
    ax.axis("off")

    # Title & Subtitle Header
    ax.text(
        0.5, 0.965,
        "MULTI-MODAL LOGO & FAVICON FORENSICS ENGINE",
        fontsize=24, fontweight="bold", color="#F8FAFC",
        ha="center", va="center"
    )
    ax.text(
        0.5, 0.935,
        "End-to-End Anti-Spoofing Architecture Protecting 52 Official Brands & 46 Authentic Favicons",
        fontsize=13, color="#94A3B8", ha="center", va="center"
    )

    def draw_box(x, y, w, h, bg_color, border_color, title, subtitle="", items=None, title_color="#FFFFFF", title_size=11):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            facecolor=bg_color, edgecolor=border_color, linewidth=2, zorder=2
        )
        ax.add_patch(box)
        
        # Header text
        ax.text(x + w / 2, y + h - 0.035, title, fontsize=title_size, fontweight="bold",
                color=title_color, ha="center", va="center", zorder=3)
        
        if subtitle:
            ax.text(x + w / 2, y + h - 0.068, subtitle, fontsize=8.5, fontstyle="italic",
                    color="#94A3B8", ha="center", va="center", zorder=3)

        if items:
            start_y = y + h - (0.10 if subtitle else 0.075)
            line_spacing = (h - (0.115 if subtitle else 0.09)) / max(len(items), 1)
            for i, it in enumerate(items):
                ax.text(x + 0.016, start_y - i * line_spacing, it, fontsize=9,
                        color="#E2E8F0", ha="left", va="center", zorder=3)

    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", style="->", rad=0.0, lw=2.5, text=""):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=color,
            linewidth=lw,
            mutation_scale=18,
            zorder=4
        )
        ax.add_patch(arrow)
        if text:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + 0.018
            ax.text(mx, my, text, fontsize=8.5, fontweight="bold", color=color, ha="center", va="bottom", zorder=5)

    # ==========================================
    # STAGE 1: INGESTION & PREPROCESSING
    # ==========================================
    col1_bg = patches.FancyBboxPatch((0.02, 0.05), 0.165, 0.84, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#131C31", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(col1_bg)
    ax.text(0.102, 0.865, "STAGE 1\nINGESTION & PREPROCESSING", fontsize=11, fontweight="bold", color="#38BDF8", ha="center", va="center")

    draw_box(
        0.035, 0.64, 0.135, 0.18,
        bg_color="#0F172A", border_color="#38BDF8",
        title="Candidate Media Input",
        items=[
            "- Formats: PNG, JPG, WebP",
            "- Formats: Static & Animated GIF",
            "- Favicons: 32x32 to 64x64",
            "- Logos & Banners: up to 1024x1024"
        ]
    )

    draw_box(
        0.035, 0.38, 0.135, 0.21,
        bg_color="#0F172A", border_color="#0EA5E9",
        title="Media Normalizer",
        subtitle="preprocessing/media_normalizer.py",
        items=[
            "- Dual Canvas Rendering",
            "  (Black & White composites)",
            "- Alpha Mask Isolation",
            "- Multi-frame Keyframe Extraction",
            "- Dimension Auto-Detection:",
            "  (FAVICON <= 128px vs LOGO)"
        ]
    )

    draw_box(
        0.035, 0.09, 0.135, 0.23,
        bg_color="#0F172A", border_color="#6366F1",
        title="Forensic Reference Store",
        subtitle="forensic_reference_store.pt",
        items=[
            "- 52 Protected Brands (logos/)",
            "- 46 Authentic Favicons (Favicon/)",
            "- Precomputed Signatures:",
            "  - SigLIP 2 Normalized Embeddings",
            "  - DINOv2 Part Geometric Features",
            "  - 64-bit pHash & dHash Hashes",
            "  - CIE LAB & HSV Color Profiles",
            "  - Morphological Edge Stroke Masks"
        ]
    )

    draw_arrow(0.102, 0.64, 0.102, 0.59, text="Raw Media")
    draw_arrow(0.102, 0.38, 0.102, 0.32, text="Normalized")

    # ==========================================
    # STAGE 2: FAST RETRIEVAL & SCREENING
    # ==========================================
    col2_bg = patches.FancyBboxPatch((0.205, 0.05), 0.17, 0.84, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#131C31", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(col2_bg)
    ax.text(0.29, 0.865, "STAGE 2\nCANDIDATE RETRIEVAL", fontsize=11, fontweight="bold", color="#818CF8", ha="center", va="center")

    draw_box(
        0.22, 0.58, 0.14, 0.24,
        bg_color="#0F172A", border_color="#818CF8",
        title="SigLIP 2 Screening",
        subtitle="google/siglip2-base-patch16-224",
        items=[
            "- 768-dim Vision Transformer",
            "- Multi-view semantic encoding",
            "- Fast Cosine Index Ranking against",
            "  all 52 brand galleries & favicons",
            "- Top Brand Candidate Discovery",
            "- Runner-up margin calculation"
        ]
    )

    draw_box(
        0.22, 0.26, 0.14, 0.24,
        bg_color="#0F172A", border_color="#A855F7",
        title="Perceptual Hash Screen",
        subtitle="forensics/perceptual_hash.py",
        items=[
            "- 64-bit Discrete Cosine pHash",
            "- 64-bit Gradient dHash",
            "- Hamming distance measurement:",
            "  - Dist <= 6: Exact / Near Replica",
            "  - Dist > 25: Divergent geometry",
            "- Instant screen of unrelated media"
        ]
    )

    draw_box(
        0.22, 0.09, 0.14, 0.12,
        bg_color="#1E1B4B", border_color="#818CF8",
        title="Candidate Shortlist",
        items=[
            "Top Brand Identification",
            "Asset Type: LOGO vs FAVICON"
        ],
        title_color="#C7D2FE"
    )

    draw_arrow(0.17, 0.485, 0.22, 0.70, text="Views")
    draw_arrow(0.17, 0.205, 0.22, 0.38, text="Cached Signatures")
    draw_arrow(0.29, 0.58, 0.29, 0.50)
    draw_arrow(0.29, 0.26, 0.29, 0.21)

    # ==========================================
    # STAGE 3: MULTI-MODAL FORENSIC MODALITIES
    # ==========================================
    col3_bg = patches.FancyBboxPatch((0.395, 0.05), 0.23, 0.84, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#131C31", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(col3_bg)
    ax.text(0.51, 0.865, "STAGE 3\nDEEP FORENSIC MODALITIES", fontsize=11, fontweight="bold", color="#34D399", ha="center", va="center")

    draw_box(
        0.41, 0.65, 0.20, 0.17,
        bg_color="#0F172A", border_color="#10B981",
        title="DINOv2 Structural Layout",
        subtitle="facebook/dinov2-base",
        items=[
            "- Self-supervised geometric ViT",
            "- Captures spatial part alignment & curves",
            "- Exposes redrawn fonts & clone typography",
            "- Flags spatial drift: (SigLIP - DINOv2 > 0.07)"
        ]
    )

    draw_box(
        0.41, 0.45, 0.20, 0.17,
        bg_color="#0F172A", border_color="#F59E0B",
        title="CIE LAB & HSV Color Forensics",
        subtitle="forensics/color_analysis.py",
        items=[
            "- CIE LAB Delta E color perceptual distance",
            "- 3D HSV histogram intersection score",
            "- Delta E <= 6.0: Authentic brand palette",
            "- Delta E > 12.0: Recolor / Spoofing alert"
        ]
    )

    draw_box(
        0.41, 0.25, 0.20, 0.17,
        bg_color="#0F172A", border_color="#EC4899",
        title="Sub-Pixel Edge Stroke IoU",
        subtitle="forensics/edge_analysis.py",
        items=[
            "- Morphological gradient stroke extraction",
            "- Sub-pixel translation & scale alignment",
            "- Stroke Overlap IoU (Valid >= 0.68)",
            "- Injected stroke ratio (>12% foreign edges)"
        ]
    )

    draw_box(
        0.41, 0.08, 0.20, 0.14,
        bg_color="#0F172A", border_color="#06B6D4",
        title="OCR & Alphanumeric Collision",
        subtitle="PP-OCRv5 via models/ocr_model.py",
        items=[
            "- Alphanumeric token extraction & norm",
            "- Levenshtein fuzzy string similarity",
            "- Squatter collision check (e.g. A200M vs A500M)"
        ]
    )

    draw_arrow(0.36, 0.15, 0.41, 0.735, text="Top Match")
    draw_arrow(0.36, 0.15, 0.41, 0.535)
    draw_arrow(0.36, 0.15, 0.41, 0.335)
    draw_arrow(0.36, 0.15, 0.41, 0.15)

    # ==========================================
    # STAGE 4: CONDITIONAL VLM AUDITOR
    # ==========================================
    col4_bg = patches.FancyBboxPatch((0.645, 0.45), 0.14, 0.44, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#131C31", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(col4_bg)
    ax.text(0.715, 0.865, "STAGE 4\nCONDITIONAL VLM", fontsize=11, fontweight="bold", color="#F43F5E", ha="center", va="center")

    draw_box(
        0.655, 0.49, 0.12, 0.33,
        bg_color="#0F172A", border_color="#F43F5E",
        title="VLM Forensic Specialist",
        subtitle="Qwen2-VL / Gemini Vision",
        items=[
            "Activation Triggers:",
            "- SigLIP >= 0.75 AND:",
            "  * Edge IoU < 0.65 OR",
            "  * Delta E > 6.0 OR",
            "  * OCR text mismatch OR",
            "  * DINOv2 divergence",
            "",
            "Visual Audit Focus:",
            "- Added secondary icons",
            "- Modified mascots",
            "- Altered character glyphs"
        ]
    )

    draw_arrow(0.61, 0.535, 0.655, 0.65, text="If Anomaly", color="#F43F5E")

    # ==========================================
    # STAGE 5: DECISION ENGINE & TAXONOMY
    # ==========================================
    col5_bg = patches.FancyBboxPatch((0.805, 0.05), 0.175, 0.84, boxstyle="round,pad=0.01,rounding_size=0.015",
                                    facecolor="#131C31", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(col5_bg)
    ax.text(0.892, 0.865, "STAGE 5\nDECISION ENGINE", fontsize=11, fontweight="bold", color="#FBBF24", ha="center", va="center")

    draw_box(
        0.82, 0.63, 0.145, 0.19,
        bg_color="#0F172A", border_color="#FBBF24",
        title="Evidence Fusion Engine",
        subtitle="fusion/decision_engine.py",
        items=[
            "- Multi-Modal Weighted Consensus",
            "  (SigLIP + DINOv2 + Edge + Color + OCR)",
            "- Media-Aware Logic (Logo vs Favicon)",
            "- Threat Taxonomy Classification",
            "- Automated Forensic Reason & Diffs"
        ]
    )

    draw_arrow(0.61, 0.735, 0.82, 0.735, color="#10B981")
    draw_arrow(0.61, 0.335, 0.82, 0.68, color="#EC4899")
    draw_arrow(0.61, 0.15, 0.82, 0.65, color="#06B6D4")
    draw_arrow(0.775, 0.65, 0.82, 0.70, color="#F43F5E", text="Audit Log")

    # Threat Categories Box
    draw_box(
        0.82, 0.43, 0.145, 0.17,
        bg_color="#18181B", border_color="#A1A1AA",
        title="Forensic Threat Types",
        items=[
            "- EXACT_REPLICA (Official Asset)",
            "- COLOR_SPOOF (Palette Shift)",
            "- VECTOR_REDRAW (Geometry Mismatch)",
            "- ELEMENT_INJECTION (New Graphics)",
            "- NUMBER_COLLISION (Squatter Token)",
            "- UNRELATED (Competitor Logo / Favicon)"
        ],
        title_color="#E4E4E7",
        title_size=10
    )

    draw_arrow(0.892, 0.63, 0.892, 0.60)

    # 3 Final Output Verdict Cards
    # MATCH
    match_card = patches.FancyBboxPatch((0.82, 0.29), 0.145, 0.10, boxstyle="round,pad=0.01,rounding_size=0.015",
                                       facecolor="#064E3B", edgecolor="#10B981", linewidth=2.5, zorder=3)
    ax.add_patch(match_card)
    ax.text(0.892, 0.355, "MATCH", fontsize=13, fontweight="bold", color="#A7F3D0", ha="center", va="center", zorder=4)
    ax.text(0.892, 0.32, "Authentic Official Logo or Favicon", fontsize=8.5, color="#ECFDF5", ha="center", va="center", zorder=4)

    # REVIEW
    review_card = patches.FancyBboxPatch((0.82, 0.17), 0.145, 0.10, boxstyle="round,pad=0.01,rounding_size=0.015",
                                        facecolor="#78350F", edgecolor="#F59E0B", linewidth=2.5, zorder=3)
    ax.add_patch(review_card)
    ax.text(0.892, 0.235, "REVIEW", fontsize=13, fontweight="bold", color="#FDE68A", ha="center", va="center", zorder=4)
    ax.text(0.892, 0.20, "Altered / Modified Brand Identity", fontsize=8.5, color="#FFFBEB", ha="center", va="center", zorder=4)

    # UNKNOWN
    unknown_card = patches.FancyBboxPatch((0.82, 0.05), 0.145, 0.10, boxstyle="round,pad=0.01,rounding_size=0.015",
                                         facecolor="#881337", edgecolor="#F43F5E", linewidth=2.5, zorder=3)
    ax.add_patch(unknown_card)
    ax.text(0.892, 0.115, "UNKNOWN", fontsize=13, fontweight="bold", color="#FECDD3", ha="center", va="center", zorder=4)
    ax.text(0.892, 0.08, "Competitor Logo / Unrelated Emblem", fontsize=8.5, color="#FFF1F2", ha="center", va="center", zorder=4)

    draw_arrow(0.892, 0.43, 0.892, 0.39, color="#E2E8F0")

    # Output file
    plt.tight_layout()
    output_p = Path(output_path).resolve()
    plt.savefig(output_p, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Flow chart successfully saved to: {output_p}")

if __name__ == "__main__":
    create_system_flow_chart()
