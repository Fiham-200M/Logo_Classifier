import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

def render_ocr_canvas(normalized_rgb: Image.Image, raw_detections: list, brand_name: str = ""):
    """
    Renders a high-tech forensic OCR canvas:
    1. Outlines each detected word with precision bounding polygons on the logo.
    2. Appends a sleek, professional HUD ledger with each token on its own row,
       complete with confidence score, token index, and pristine typography.
    """
    orig_img = normalized_rgb.convert("RGB")
    img_w, img_h = orig_img.size
    
    # Scale image up if it is very small so bounding boxes and logo details look sharp
    render_scale = 1.0
    if img_w < 600:
        render_scale = 600.0 / img_w
    target_w = int(img_w * render_scale)
    target_h = int(img_h * render_scale)
    
    scaled_img = orig_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    img_arr = np.array(scaled_img)
    
    # Draw bounding polygons on the image
    palette = [
        (0, 255, 136),   # Emerald / Green
        (0, 229, 255),   # Cyan
        (255, 179, 0),   # Amber
        (168, 85, 247),  # Purple
        (236, 72, 153),  # Pink
    ]
    
    overlay = img_arr.copy()
    tokens_info = []
    
    for idx, det in enumerate(raw_detections):
        txt = det.get("text", "").strip()
        conf = float(det.get("confidence", 0.0))
        box = det.get("box")
        color = palette[idx % len(palette)]
        
        tokens_info.append({
            "idx": idx + 1,
            "text": txt,
            "conf": conf,
            "color": color
        })
        
        if box:
            pts = np.array(box, dtype=np.float32)
            pts[:, 0] *= render_scale
            pts[:, 1] *= render_scale
            pts = pts.astype(np.int32)
            
            # Semi-transparent highlight inside box
            cv2.fillPoly(overlay, [pts], color)
            # Crisp outline
            cv2.polylines(img_arr, [pts], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)
            
            # Corner markers for forensic look
            for pt in pts:
                cv2.circle(img_arr, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(img_arr, (int(pt[0]), int(pt[1])), 4, color, 1, cv2.LINE_AA)
            
            # Badge anchor point (top-left of box)
            x_min = int(np.min(pts[:, 0]))
            y_min = int(np.min(pts[:, 1]))
            badge_text = f"#{idx+1}"
            
            # Draw small tag circle or pill
            tag_y = max(18, y_min - 4)
            tag_x = max(4, x_min)
            cv2.rectangle(img_arr, (tag_x, tag_y - 14), (tag_x + 24, tag_y + 2), (15, 23, 42), -1)
            cv2.rectangle(img_arr, (tag_x, tag_y - 14), (tag_x + 24, tag_y + 2), color, 1)
            cv2.putText(img_arr, badge_text, (tag_x + 3, tag_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Blend polygon highlight overlay
    cv2.addWeighted(overlay, 0.20, img_arr, 0.80, 0, img_arr)
    
    # Calculate HUD Ledger dimensions
    row_height = 36
    header_height = 42
    footer_height = 10
    total_tokens = len(tokens_info)
    ledger_height = header_height + max(1, total_tokens) * row_height + footer_height
    
    canvas_w = max(640, target_w)
    canvas_h = target_h + ledger_height + 20
    
    # Dark modern canvas background (#0b0f19)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:] = (11, 15, 25)
    
    # Center the logo in the upper portion
    x_offset = (canvas_w - target_w) // 2
    y_offset = 12
    canvas[y_offset:y_offset + target_h, x_offset:x_offset + target_w] = img_arr
    
    # Border around logo preview
    cv2.rectangle(canvas, (x_offset - 1, y_offset - 1), (x_offset + target_w, y_offset + target_h), (30, 41, 59), 1)
    
    # Draw HUD Ledger in the bottom portion
    ledger_y = y_offset + target_h + 10
    
    # Ledger panel background (#0f172a with subtle border #1e293b)
    cv2.rectangle(canvas, (16, ledger_y), (canvas_w - 16, ledger_y + ledger_height - 6), (15, 23, 42), -1)
    cv2.rectangle(canvas, (16, ledger_y), (canvas_w - 16, ledger_y + ledger_height - 6), (30, 41, 59), 1)
    
    # Header: "PADDLEOCR DETECTED TOKENS (N found)"
    header_str = f"PADDLEOCR DETECTED TOKENS ({total_tokens} detected)" if total_tokens else "PADDLEOCR: NO TEXT DETECTED"
    cv2.putText(canvas, header_str, (32, ledger_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA)
    
    # Engine badge on the right
    engine_badge = "PP-OCRv5 Active"
    (eb_w, _), _ = cv2.getTextSize(engine_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    eb_x = canvas_w - 32 - eb_w - 16
    cv2.rectangle(canvas, (eb_x, ledger_y + 12), (canvas_w - 32, ledger_y + 32), (30, 41, 59), -1)
    cv2.putText(canvas, engine_badge, (eb_x + 8, ledger_y + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 211, 153), 1, cv2.LINE_AA)
    
    # Horizontal divider
    cv2.line(canvas, (24, ledger_y + 36), (canvas_w - 24, ledger_y + 36), (30, 41, 59), 1)
    
    # Render each token on its OWN separate row!
    curr_y = ledger_y + header_height + 8
    
    if not tokens_info:
        cv2.putText(canvas, "No alphanumeric glyphs or typographic tokens detected.", (36, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 116, 139), 1, cv2.LINE_AA)
    else:
        for t_info in tokens_info:
            idx_num = t_info["idx"]
            text_val = t_info["text"]
            conf_val = t_info["conf"]
            c_color = t_info["color"]
            
            # Row background highlight on hover or alternating
            cv2.rectangle(canvas, (24, curr_y - 4), (canvas_w - 24, curr_y + row_height - 10), (19, 29, 50), -1)
            
            # 1. Badge pill: [#1]
            pill_text = f"#{idx_num}"
            cv2.rectangle(canvas, (32, curr_y), (60, curr_y + 20), (30, 41, 59), -1)
            cv2.rectangle(canvas, (32, curr_y), (60, curr_y + 20), c_color, 1)
            cv2.putText(canvas, pill_text, (37, curr_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, c_color, 1, cv2.LINE_AA)
            
            # 2. Confidence badge: [98%]
            conf_text = f"{conf_val:.0%}"
            cv2.rectangle(canvas, (68, curr_y), (120, curr_y + 20), (16, 42, 35), -1)
            cv2.putText(canvas, conf_text, (75, curr_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (52, 211, 153), 1, cv2.LINE_AA)
            
            # 3. Pristine Token String: clearly separated, bold and readable
            display_text = f'"{text_val}"'
            # Truncate if extremely long to avoid overflowing
            if len(display_text) > 48:
                display_text = display_text[:45] + '..."'
            cv2.putText(canvas, display_text, (132, curr_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (241, 245, 249), 2, cv2.LINE_AA)
            
            curr_y += row_height

    return canvas

# Test on raja100.png and paduka500.png
from models.ocr_model import OCRModel
ocr = OCRModel()

for fname in ["raja100.png", "paduka500.png"]:
    p = project_root / "dataset" / "our_logos" / "raw_logos" / fname
    img = Image.open(p).convert("RGBA")
    res = ocr.extract_text(img)
    # mock boxes if worker didn't provide box yet
    dets = res["raw_detections"]
    cv_out = render_ocr_canvas(img, dets, fname)
    out_p = project_root / "scratch" / f"vis_{fname}"
    cv2.imwrite(str(out_p), cv2.cvtColor(cv_out, cv2.COLOR_RGB2BGR))
    print(f"Generated test canvas for {fname} at {out_p}")
