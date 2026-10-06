import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def generate_combined_architecture(output_path: str = "system_architecture_combined.png"):
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'Helvetica']
    
    # 5400 x 7200 at 300 DPI -> 18 x 24 inches
    fig, ax = plt.subplots(figsize=(18, 24), dpi=300)
    fig.patch.set_facecolor("#0B1120") # Deep midnight slate
    ax.set_facecolor("#0B1120")
    ax.axis("off")

    # Header Title
    ax.text(
        0.5, 0.980,
        "MULTI-MODAL LOGO & FAVICON FORENSICS ENGINE",
        fontsize=22, fontweight="bold", color="#F8FAFC",
        ha="center", va="center"
    )
    ax.text(
        0.5, 0.963,
        "Complete System Architecture: Fast Retrieval, Multi-Modal Sensory Modalities & Forensic Veto Engine",
        fontsize=11.5, color="#94A3B8", ha="center", va="center"
    )
    ax.text(
        0.5, 0.949,
        "Protecting 52 Official Brand Logos (logos/) & 46 Authentic Favicons (Favicon/)",
        fontsize=10.5, color="#38BDF8", ha="center", va="center", fontweight="bold"
    )

    def draw_card(x, y, w, h, bg_color, border_color, title, subtitle="", items=None, 
                  title_color="#FFFFFF", title_size=10, subtitle_color="#94A3B8", text_color="#E2E8F0", item_size=8.2, lw=1.8):
        box = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=2
        )
        ax.add_patch(box)
        
        has_items = items is not None and len(items) > 0
        if not subtitle and not has_items:
            ax.text(x + w / 2, y + h / 2, title, fontsize=title_size, fontweight="bold",
                    color=title_color, ha="center", va="center", zorder=3)
            return

        title_y = y + h - (0.022 if h > 0.08 else 0.016)
        ax.text(x + w / 2, title_y, title, fontsize=title_size, fontweight="bold",
                color=title_color, ha="center", va="center", zorder=3)
        
        cur_y = title_y
        if subtitle:
            cur_y -= 0.015
            ax.text(x + w / 2, cur_y, subtitle, fontsize=item_size - 0.5, fontstyle="italic",
                    color=subtitle_color, ha="center", va="center", zorder=3)

        if has_items:
            start_y = cur_y - 0.016
            line_spacing = (start_y - y - 0.008) / max(len(items), 1)
            for i, it in enumerate(items):
                ax.text(x + 0.012, start_y - i * line_spacing, it, fontsize=item_size,
                        color=text_color, ha="left", va="center", zorder=3)

    def draw_arrow(x1, y1, x2, y2, color="#38BDF8", style="->", rad=0.0, lw=2.0, text="", text_color=None):
        arrow = patches.FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=color,
            linewidth=lw,
            mutation_scale=15,
            zorder=4
        )
        ax.add_patch(arrow)
        if text:
            mx = (x1 + x2) / 2
            my = (y1 + y2) / 2 + 0.007
            tc = text_color if text_color else color
            ax.text(mx, my, text, fontsize=7.8, fontweight="bold", color=tc, ha="center", va="bottom", zorder=5)

    # =========================================================================
    # SECTION 1: INGESTION & DUAL PREPROCESSING (y: 0.775 to 0.935)
    # =========================================================================
    sec1_bg = patches.FancyBboxPatch((0.03, 0.778), 0.94, 0.158, boxstyle="round,pad=0.01,rounding_size=0.012",
                                     facecolor="#10192C", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(sec1_bg)
    ax.text(0.05, 0.924, "1. INGESTION & PREPROCESSING PIPELINE", fontsize=10, fontweight="bold", color="#38BDF8", ha="left")

    # Input Media Card
    draw_card(
        0.33, 0.880, 0.34, 0.046,
        bg_color="#1E293B", border_color="#38BDF8",
        title="Input Candidate Media",
        subtitle="Logo Image / Favicon / WebP / Static & Animated GIF (32x32 to 1024x1024)",
        title_color="#38BDF8", title_size=10.5
    )

    # Format & Animation Detector
    draw_card(
        0.35, 0.820, 0.30, 0.040,
        bg_color="#0F172A", border_color="#0EA5E9",
        title="Format & Dimension Detector",
        subtitle="MediaNormalizer: Static vs Multi-Frame | FAVICON (<=128px) vs LOGO",
        title_color="#F8FAFC", title_size=9.2
    )
    draw_arrow(0.50, 0.880, 0.50, 0.860, color="#38BDF8")

    # Static Branch
    draw_card(
        0.05, 0.788, 0.38, 0.038,
        bg_color="#0F172A", border_color="#0284C7",
        title="Canvas Normalization & Dual Backgrounds",
        subtitle="Alpha Channel Compositing on Pure White & Pure Black Canvases",
        title_color="#BAE6FD", title_size=8.8
    )
    draw_arrow(0.40, 0.820, 0.24, 0.826, color="#38BDF8", rad=0.1, text="Static Image / Favicon")

    # Animated Branch
    draw_card(
        0.57, 0.788, 0.38, 0.038,
        bg_color="#0F172A", border_color="#0284C7",
        title="Keyframe Decomposition & Temporal Alpha Compositing",
        subtitle="Multi-Frame Iterator & Keyframe Accumulation for Animated GIFs",
        title_color="#BAE6FD", title_size=8.8
    )
    draw_arrow(0.60, 0.820, 0.76, 0.826, color="#38BDF8", rad=-0.1, text="Animated WebP / GIF")

    # =========================================================================
    # SECTION 2: REFERENCE RETRIEVAL & SCREENING (y: 0.635 to 0.765)
    # =========================================================================
    sec2_bg = patches.FancyBboxPatch((0.03, 0.635), 0.94, 0.128, boxstyle="round,pad=0.01,rounding_size=0.012",
                                     facecolor="#10192C", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(sec2_bg)
    ax.text(0.05, 0.750, "2. FAST CANDIDATE RETRIEVAL & PERCEPTUAL SCREENING", fontsize=10, fontweight="bold", color="#818CF8", ha="left")

    # Reference Store
    draw_card(
        0.05, 0.645, 0.26, 0.095,
        bg_color="#1E1B4B", border_color="#6366F1",
        title="Forensic Reference Store",
        subtitle="reference_embeddings/forensic_reference_store.pt",
        items=[
            "- 52 Protected Brands (logos/)",
            "- 46 Authentic Brand Favicons (Favicon/)",
            "- Cached SigLIP2, DINOv2 & pHash Signatures",
            "- Pre-extracted LAB & Edge Reference Masks"
        ],
        title_color="#C7D2FE", title_size=9.2
    )

    # SigLIP 2 Retrieval
    draw_card(
        0.34, 0.645, 0.28, 0.095,
        bg_color="#0F172A", border_color="#818CF8",
        title="Google SigLIP 2 Screening",
        subtitle="google/siglip2-base-patch16-224 (768-D)",
        items=[
            "- Fast Cosine Matrix Index Retrieval",
            "- Multi-view semantic scoring across all brands",
            "- Identifies Top Candidate & Runner-up Margin",
            "- Flags high semantic match (score >= 0.75)"
        ],
        title_color="#818CF8", title_size=9.2
    )

    # Perceptual Hashing
    draw_card(
        0.65, 0.645, 0.30, 0.095,
        bg_color="#0F172A", border_color="#A855F7",
        title="Perceptual Hashing Screen",
        subtitle="64-bit DCT pHash & Gradient dHash",
        items=[
            "- Fast 64-bit Hamming distance calculation",
            "- Dist <= 6: High perceptual match / exact",
            "- Dist > 25: Geometric / structural divergence",
            "- Discards unrelated candidate media"
        ],
        title_color="#C084FC", title_size=9.2
    )

    draw_arrow(0.24, 0.788, 0.40, 0.740, color="#818CF8")
    draw_arrow(0.76, 0.788, 0.60, 0.740, color="#818CF8")
    draw_arrow(0.31, 0.692, 0.34, 0.692, color="#6366F1", text="Indexed Gallery")
    draw_arrow(0.62, 0.692, 0.65, 0.692, color="#818CF8")

    # =========================================================================
    # SECTION 3: SENSORY CHANNELS & CONDITIONAL VLM (y: 0.435 to 0.620)
    # =========================================================================
    sec3_bg = patches.FancyBboxPatch((0.03, 0.430), 0.94, 0.192, boxstyle="round,pad=0.01,rounding_size=0.012",
                                     facecolor="#10192C", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(sec3_bg)
    ax.text(0.05, 0.610, "3. DEEP FORENSIC SENSORY MODALITIES & CONDITIONAL VLM SPECIALIST", fontsize=10, fontweight="bold", color="#34D399", ha="left")

    # Channel 1: SigLIP 2
    draw_card(
        0.05, 0.470, 0.17, 0.130,
        bg_color="#0F172A", border_color="#818CF8",
        title="SigLIP 2 ViT",
        subtitle="Semantic Similarity",
        items=[
            "- 768-D Semantic Space",
            "- Macro Visual Semantics",
            "- Brand Context Ranking",
            "- Match: >= 0.88",
            "- Suspicious: >= 0.75"
        ],
        title_color="#818CF8"
    )

    # Channel 2: DINOv2
    draw_card(
        0.23, 0.470, 0.17, 0.130,
        bg_color="#0F172A", border_color="#10B981",
        title="Meta DINOv2 ViT",
        subtitle="facebook/dinov2-base",
        items=[
            "- Part Geometry Layout",
            "- Typography Contours",
            "- Detects Vector Redraws",
            "- Spatial Drift Detection",
            "- Match: >= 0.80"
        ],
        title_color="#34D399"
    )

    # Channel 3: Color Forensics
    draw_card(
        0.41, 0.470, 0.17, 0.130,
        bg_color="#0F172A", border_color="#F59E0B",
        title="Color Forensics",
        subtitle="CIE LAB & 3D HSV",
        items=[
            "- Delta E LAB Distance",
            "- 3D HSV Histogram Overlap",
            "- Authentic: Delta E <= 6.0",
            "- Palette Drift: Delta E > 12",
            "- Color Spoofing Guard"
        ],
        title_color="#FBBF24"
    )

    # Channel 4: Edge Stroke Analysis
    draw_card(
        0.59, 0.470, 0.17, 0.130,
        bg_color="#0F172A", border_color="#EC4899",
        title="Edge Stroke Analysis",
        subtitle="Sub-pixel Morph IoU",
        items=[
            "- Sub-pixel Alignment",
            "- Contour Overlap IoU",
            "- Authentic IoU >= 0.68",
            "- Redraw IoU < 0.55",
            "- Injected Strokes: > 12%"
        ],
        title_color="#F472B6"
    )

    # Channel 5: PaddleOCR
    draw_card(
        0.77, 0.470, 0.18, 0.130,
        bg_color="#0F172A", border_color="#06B6D4",
        title="PaddleOCR PP-OCRv5",
        subtitle="Text & Number Guard",
        items=[
            "- Isolated Worker Process",
            "- Token Normalization",
            "- Levenshtein Fuzzy Sim",
            "- Number Collision Alert",
            "- A200M vs A500M"
        ],
        title_color="#22D3EE"
    )

    # Conditional VLM Badge
    draw_card(
        0.28, 0.438, 0.44, 0.026,
        bg_color="#311425", border_color="#F43F5E",
        title="Conditional VLM Specialist (Qwen2-VL / Gemini Vision)",
        subtitle="Trigger: SigLIP >= 0.75 AND (Delta E > 6.0 | Edge IoU < 0.65 | OCR Mismatch | DINOv2 Drift)",
        title_color="#FDA4AF", title_size=8.0, subtitle_color="#FECDD3", lw=1.5
    )

    # Arrows from Section 2 to Section 3
    draw_arrow(0.48, 0.645, 0.135, 0.600, color="#818CF8")
    draw_arrow(0.48, 0.645, 0.315, 0.600, color="#10B981")
    draw_arrow(0.48, 0.645, 0.495, 0.600, color="#F59E0B")
    draw_arrow(0.48, 0.645, 0.675, 0.600, color="#EC4899")
    draw_arrow(0.48, 0.645, 0.860, 0.600, color="#06B6D4")

    # =========================================================================
    # SECTION 4: MULTI-MODAL FUSION & FORENSIC VETO ENGINE (y: 0.175 to 0.415)
    # =========================================================================
    sec4_bg = patches.FancyBboxPatch((0.03, 0.170), 0.94, 0.245, boxstyle="round,pad=0.01,rounding_size=0.012",
                                     facecolor="#10192C", edgecolor="#1E293B", linewidth=1.5, zorder=1)
    ax.add_patch(sec4_bg)
    ax.text(0.05, 0.402, "4. MULTI-MODAL FUSION & FORENSIC VETO ENGINE", fontsize=10, fontweight="bold", color="#FBBF24", ha="left")

    # Fusion Box
    draw_card(
        0.28, 0.352, 0.44, 0.042,
        bg_color="#18181B", border_color="#C084FC",
        title="Multi-Modal Fusion & Evidence Aggregator",
        subtitle="Weighted Consensus (SigLIP + DINOv2 + Edge IoU + Color + OCR) & Media-Aware Favicon Rules",
        title_color="#E9D5FF", title_size=10.0, subtitle_color="#DDD6FE", lw=2.0
    )

    # Connect channels to Fusion Engine
    draw_arrow(0.135, 0.470, 0.33, 0.394, color="#818CF8")
    draw_arrow(0.315, 0.470, 0.42, 0.394, color="#10B981")
    draw_arrow(0.495, 0.438, 0.50, 0.394, color="#F59E0B")
    draw_arrow(0.675, 0.470, 0.58, 0.394, color="#EC4899")
    draw_arrow(0.860, 0.470, 0.67, 0.394, color="#06B6D4")

    # Forensic Veto Diamond
    diamond = patches.Polygon(
        [[0.50, 0.334], [0.56, 0.292], [0.50, 0.250], [0.44, 0.292]],
        facecolor="#0F172A", edgecolor="#38BDF8", linewidth=2.5, zorder=3
    )
    ax.add_patch(diamond)
    ax.text(0.50, 0.298, "Forensic", fontsize=10.5, fontweight="bold", color="#F8FAFC", ha="center", va="center", zorder=4)
    ax.text(0.50, 0.284, "Veto Checks", fontsize=9.5, fontweight="bold", color="#38BDF8", ha="center", va="center", zorder=4)

    draw_arrow(0.50, 0.352, 0.50, 0.334, color="#C084FC")

    # 3 Veto Gate Cards (Symmetrically aligned at y = 0.180, h = 0.065)
    # Left Veto (REVIEW)
    draw_card(
        0.05, 0.180, 0.27, 0.065,
        bg_color="#1C1917", border_color="#F59E0B",
        title="Altered Brand Identity Gate",
        items=[
            "- Token / Number Collision (e.g. A200M vs A500M)",
            "- Color Drift: Delta E > 12.0 (Structural Match)",
            "- Typography Redraw: Edge IoU < 0.55",
            "- Secondary Elements: > 12% Injected Strokes"
        ],
        title_color="#FDE68A", title_size=8.2, item_size=7.2
    )
    draw_arrow(0.44, 0.292, 0.185, 0.245, color="#F59E0B", text="Altered Brand", text_color="#FDE68A")

    # Center Veto (UNKNOWN)
    draw_card(
        0.365, 0.180, 0.27, 0.065,
        bg_color="#1C1917", border_color="#F43F5E",
        title="Unrelated & Competitor Gate",
        items=[
            "- Low Geometry: DINOv2 < 0.65",
            "- Divergent Edges: Stroke IoU < 0.35",
            "- High Hash Distance: pHash Dist > 25",
            "- Zero Brand or Official Favicon Consensus"
        ],
        title_color="#FECDD3", title_size=8.2, item_size=7.2
    )
    draw_arrow(0.50, 0.250, 0.50, 0.245, color="#F43F5E", text="No Match", text_color="#FECDD3")

    # Right Veto (MATCH)
    draw_card(
        0.68, 0.180, 0.27, 0.065,
        bg_color="#064E3B", border_color="#10B981",
        title="Authentic Official Consensus Gate",
        items=[
            "- High ViT Consensus: SigLIP2 >= 0.88 & DINOv2 >= 0.80",
            "- Stroke Overlap IoU >= 0.68",
            "- Color Fidelity: CIE LAB Delta E <= 6.0",
            "- pHash Dist <= 6 | Clean OCR / No Collision"
        ],
        title_color="#A7F3D0", title_size=8.2, item_size=7.2
    )
    draw_arrow(0.56, 0.292, 0.815, 0.245, color="#10B981", text="Authentic Asset", text_color="#A7F3D0")

    # =========================================================================
    # SECTION 5: FINAL OUTPUT VERDICTS (y: 0.020 to 0.150)
    # Symmetrically aligned cards at y = 0.020, h = 0.125
    # =========================================================================
    # REVIEW CARD
    draw_card(
        0.05, 0.020, 0.27, 0.125,
        bg_color="#2A170A", border_color="#F59E0B",
        title="REVIEW Verdict",
        subtitle="Altered / Spoofed Brand Identity",
        items=[
            "THREAT TAXONOMY:",
            "- COLOR_SPOOF (Palette Shift)",
            "- NUMBER_COLLISION (Brand Squatter)",
            "- VECTOR_REDRAW (Stroke Redraw)",
            "- ELEMENT_INJECTION (New Badges / Icons)",
            "",
            "ACTION: Route to Forensics Dashboard",
            "Human Specialist Verification Required"
        ],
        title_color="#FDE68A", title_size=10.5, subtitle_color="#FEF3C7", item_size=7.8, lw=2.2
    )
    draw_arrow(0.185, 0.180, 0.185, 0.145, color="#F59E0B")

    # UNKNOWN CARD
    draw_card(
        0.365, 0.020, 0.27, 0.125,
        bg_color="#2D0612", border_color="#F43F5E",
        title="UNKNOWN Verdict",
        subtitle="Competitor Logo / Favicon / Unrelated",
        items=[
            "THREAT TAXONOMY:",
            "- UNRELATED (Competitor Logo / Emblem)",
            "- UNRELATED (Competitor Favicon)",
            "",
            "DETECTION REASON:",
            "- No structural or visual consensus with",
            "  our 52 protected brands or favicons.",
            "",
            "ACTION: Safely Ignore / Unprotected External"
        ],
        title_color="#FECDD3", title_size=10.5, subtitle_color="#FFE4E6", item_size=7.8, lw=2.2
    )
    draw_arrow(0.50, 0.180, 0.50, 0.145, color="#F43F5E")

    # MATCH CARD
    draw_card(
        0.68, 0.020, 0.27, 0.125,
        bg_color="#032D23", border_color="#10B981",
        title="MATCH Verdict",
        subtitle="Authentic Official Asset Confirmed",
        items=[
            "THREAT TAXONOMY:",
            "- EXACT_REPLICA (Official Protected Asset)",
            "",
            "VERIFIED ASSETS:",
            "- Authentic Official Logo (logos/)",
            "- Official Brand Favicon (Favicon/)",
            "",
            "ACTION: Auto-Approve / Genuine Brand Verified"
        ],
        title_color="#A7F3D0", title_size=10.5, subtitle_color="#D1FAE5", item_size=7.8, lw=2.2
    )
    draw_arrow(0.815, 0.180, 0.815, 0.145, color="#10B981")

    # Save to disk
    plt.tight_layout()
    out_p = Path(output_path).resolve()
    plt.savefig(out_p, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Refined Master Architecture successfully generated at: {out_p}")

if __name__ == "__main__":
    generate_combined_architecture("system_architecture_combined.png")
