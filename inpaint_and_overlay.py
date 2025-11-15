# inpaint_and_overlay.py
"""
inpaint_and_overlay.py

Functions:
 - inpaint_image_with_boxes(image_pil, annotations, prefer_lama=True, lama_device='cpu')
 - overlay_translated_text(image_pil, annotations, font_path=None)
 - process_and_save(input_path, annotations, output_path, translated_texts=None, font_path=None, prefer_lama=True)

Annotations format (accepted):
[
  { "text": "柔らかい",
    "bounding_box": [{"x":514,"y":37}, {"x":608,"y":37}, {"x":608,"y":57}, {"x":514,"y":57}] },
  or
  { "text": "...", "bounding_box": [[514,37], [608,37], [608,57], [514,57]] }
]
You may include "translated_text" or "new_text" in each annotation; that will be used when overlaying.
"""
from typing import List, Dict, Tuple, Optional
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
import math
import textwrap
import os

MIN_FONT_SIZE = 12

# ---------------------------
# Utilities
# ---------------------------


def normalize_polygon(poly):
    """
    Accepts polygon in either list-of-dicts [{"x":..,"y":..}, ...]
    or list-of-lists [[x,y], ...] and returns list-of-tuples [(x,y),...]
    """
    if not poly:
        return []
    out = []
    # dict style
    if isinstance(poly[0], dict):
        for p in poly:
            x = float(p.get("x", p.get("X", 0)))
            y = float(p.get("y", p.get("Y", 0)))
            out.append((x, y))
    else:
        for p in poly:
            out.append((float(p[0]), float(p[1])))
    return out


def poly_to_rect_coords(polygon: List[Dict[str, int]]) -> Tuple[int, int, int, int]:
    """
    Returns integer rectangle coordinates x_min, y_min, x_max, y_max
    polygon may be a list-of-dicts or list-of-lists
    """
    pts = normalize_polygon(polygon)
    if not pts:
        return 0, 0, 0, 0
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    return x_min, y_min, x_max, y_max


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    # rgb -> bgr for cv2
    return arr[:, :, ::-1].copy()


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(rgb)


def ensure_font_exists(path: str) -> bool:
    return path and os.path.exists(path)


# ---------------------------
# OpenCV fallback inpaint (improved)
# ---------------------------


def inpaint_with_opencv(pil_img: Image.Image, mask_pil: Image.Image, method: str = 'telea', dilate_iters: int = 1,
                        blur_ksize: int = 7, inpaint_radius: int = 3) -> Image.Image:
    """
    Uses OpenCV inpaint as fallback. We dilate and blur the mask to make results smoother.
    method: 'telea' or 'ns'
    """
    bgr = pil_to_cv2(pil_img)
    mask = np.array(mask_pil.convert("L"), dtype=np.uint8)

    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if dilate_iters > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_bin = cv2.dilate(mask_bin, kernel, iterations=dilate_iters)

    if blur_ksize and blur_ksize % 2 == 1:
        mask_blur = cv2.GaussianBlur(mask_bin, (blur_ksize, blur_ksize), 0)
        _, mask_bin = cv2.threshold(mask_blur, 10, 255, cv2.THRESH_BINARY)

    inpaint_flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    out = cv2.inpaint(
        bgr, mask_bin, inpaintRadius=inpaint_radius, flags=inpaint_flag)
    return cv2_to_pil(out)


# ---------------------------
# Create mask from annotations (boxes)
# ---------------------------


def create_mask_for_boxes(img_size: Tuple[int, int], annotations: List[Dict], expand: int = 6) -> Image.Image:
    """
    Build a single mask image (L) where box/polygon regions are white (255).
    expand: pixels to expand bounding box (helps remove nearby strokes)
    """
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        poly_raw = ann.get("bounding_box") or ann.get(
            "bbox") or ann.get("poly") or ann.get("polygon")
        if not poly_raw:
            continue
        poly = normalize_polygon(poly_raw)
        x1, y1, x2, y2 = poly_to_rect_coords(poly)
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(img_size[0], x2 + expand)
        y2 = min(img_size[1], y2 + expand)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask


# ---------------------------
# Robust angle detection
# ---------------------------


def get_text_angle(poly_pts: List[Tuple[float, float]]) -> float:
    """
    Compute robust angle (degrees) for the polygon using cv2.minAreaRect.
    Returns angle in degrees (positive means rotation angle from horizontal).
    """
    if not poly_pts or len(poly_pts) < 2:
        return 0.0
    try:
        pts = np.array(poly_pts, dtype=np.float32)
        rect = cv2.minAreaRect(pts)  # ((cx,cy),(w,h),angle)
        angle = rect[2]
        if angle < -45:
            angle = angle + 90
        return float(angle)
    except Exception:
        (x1, y1), (x2, y2) = poly_pts[0], poly_pts[1]
        return math.degrees(math.atan2(y2 - y1, x2 - x1))


# ---------------------------
# Choose font by script hint (simple)
# ---------------------------


def choose_font_for_text(sample_text: str, fallback: Optional[str] = None) -> Optional[str]:
    """
    Very simple heuristic to choose a font path for a given sample_text.
    You should provide paths to fonts available on your system.
    Returns path to a font file or None to use PIL default.
    """
    FONT_MAP = {
        "cjk": ["NotoSansCJK-Regular.ttc", "NotoSansCJKsc-Regular.otf"],
        "devanagari": ["NotoSansDevanagari-Regular.ttf"],
        "arabic": ["NotoNaskhArabic-Regular.ttf", "NotoSansArabic.ttf"],
        "thai": ["NotoSansThai-Regular.ttf"],
        "default": [fallback or "arial.ttf"]
    }

    candidates = FONT_MAP.get("default")
    for ch in sample_text:
        code = ord(ch)
        if 0x4E00 <= code <= 0x9FFF:
            candidates = FONT_MAP.get("cjk")
            break
        if 0x0900 <= code <= 0x097F:
            candidates = FONT_MAP.get("devanagari")
            break
        if 0x0600 <= code <= 0x06FF:
            candidates = FONT_MAP.get("arabic")
            break
        if 0x0E00 <= code <= 0x0E7F:
            candidates = FONT_MAP.get("thai")
            break

    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


# ---------------------------
# Helpers for adaptive color & sampling
# ---------------------------


def sample_background_color(img: Image.Image, bbox: Tuple[int, int, int, int], pad: int = 6) -> Tuple[int, int, int]:
    """Sample average RGB color around bbox (pad defines how many px around)"""
    x1, y1, x2, y2 = bbox
    w, h = img.size
    sx = max(0, x1 - pad)
    sy = max(0, y1 - pad)
    ex = min(w, x2 + pad)
    ey = min(h, y2 + pad)
    crop = img.crop((sx, sy, ex, ey)).convert("RGB")
    arr = np.array(crop)
    if arr.size == 0:
        return (255, 255, 255)
    mean = tuple(map(int, arr.reshape(-1, 3).mean(axis=0)))
    return mean


def choose_text_color(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return black or white text color for best contrast on rgb"""
    r, g, b = rgb
    # luminance formula
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if lum > 160 else (255, 255, 255)


# ---------------------------
# Overlay translated text nicely (optimized)
# ---------------------------

def overlay_translated_text(img_pil: Image.Image, annotations: List[Dict], font_path: Optional[str] = None):
    """
    Overlay translated/new_text contained in annotations into the bounding boxes.
    Adaptive font size, expand bounding area if needed, vertical stacking for tall boxes,
    adaptive text color/background blending.
    Returns a PIL RGB image.
    """
    # safety minimum font size for readability (project requirement)
    MIN_FONT_SIZE = 12

    out = img_pil.convert("RGBA")
    w_img, h_img = out.size

    for ann in annotations:
        raw_poly = ann.get("bounding_box") or ann.get(
            "bbox") or ann.get("poly") or ann.get("polygon")
        if not raw_poly:
            continue
        poly = normalize_polygon(raw_poly)
        if not poly:
            continue

        x1, y1, x2, y2 = poly_to_rect_coords(poly)
        box_w = max(2, x2 - x1)
        box_h = max(2, y2 - y1)

        new_text = (ann.get("new_text") or ann.get(
            "translated_text") or ann.get("text") or "").strip()
        if not new_text:
            continue

        # choose font candidate
        chosen_font = font_path
        if font_path is None:
            maybe = choose_font_for_text(new_text)
            if maybe:
                chosen_font = maybe

        # detect vertical (tall narrow)
        aspect = float(box_h) / max(1.0, box_w)
        want_vertical = aspect >= 2.5  # stricter threshold for true vertical labels

        # Heuristics for font sizing:
        start_fs = max(MIN_FONT_SIZE, int(box_h * 0.36))
        start_fs = min(start_fs, 72)
        max_try_fs = min(72, int(min(w_img, h_img) * 0.2))

        if want_vertical:
            # base on box width for vertical labels
            start_fs = max(MIN_FONT_SIZE, int(box_w * 0.7))

        final_font = None
        final_wrapped = new_text
        text_w = text_h = 0
        used_fs = start_fs

        def load_font(fs):
            try:
                return ImageFont.truetype(chosen_font, fs) if chosen_font else ImageFont.load_default()
            except Exception:
                return ImageFont.load_default()

        # Candidates: try larger sizes first to prefer big readable labels
        if len(new_text.split()) <= 3 and not want_vertical:
            # short labels - allow larger tries
            high_start = min(start_fs + 10, max_try_fs)
        else:
            high_start = min(start_fs + 6, max_try_fs)
        fs_candidates = list(range(high_start, MIN_FONT_SIZE - 1, -1))

        # ensure wrapped defined
        wrapped = new_text

        for fs in fs_candidates:
            font = load_font(fs)
            # wrapping strategy
            if want_vertical:
                words = [w for w in new_text.split() if w.strip()]
                if len(words) <= 3:
                    # per-char vertical stack (remove spaces)
                    wrapped = "\n".join(list(new_text.replace(" ", "")))
                else:
                    est_chars = max(1, int(box_w / max(1, (fs * 0.6))))
                    wrapped = "\n".join(textwrap.wrap(
                        new_text, width=max(est_chars, 1),
                        break_long_words=False, break_on_hyphens=False))
            else:
                est_chars = max(1, int(box_w / max(1, (fs * 0.6))))
                wrapped = "\n".join(textwrap.wrap(
                    new_text, width=max(est_chars, 1),
                    break_long_words=False, break_on_hyphens=False))

            # measure on a generous tmp to avoid clipping issues
            tmp_img = Image.new(
                "RGBA", (max(64, box_w + 160), max(48, box_h + 160)), (255, 255, 255, 0))
            td = ImageDraw.Draw(tmp_img)
            try:
                bbox = td.multiline_textbbox(
                    (0, 0), wrapped, font=font, spacing=2)
            except Exception:
                bbox = (0, 0, 0, 0)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            # accept if reasonably fits into expanded box
            if tw <= box_w * 1.05 and th <= box_h * 1.05:
                final_font = font
                final_wrapped = wrapped
                text_w, text_h = tw, th
                used_fs = fs
                break

        # shrink fallback if nothing matched
        if final_font is None:
            for fs in range(start_fs - 1, MIN_FONT_SIZE - 1, -1):
                font = load_font(fs)
                est_chars = max(1, int(box_w / max(1, (fs * 0.6))))
                wrapped = "\n".join(textwrap.wrap(
                    new_text, width=max(est_chars, 1),
                    break_long_words=False, break_on_hyphens=False))
                tmp_img = Image.new(
                    "RGBA", (max(64, box_w + 160), max(48, box_h + 160)), (255, 255, 255, 0))
                td = ImageDraw.Draw(tmp_img)
                try:
                    bbox = td.multiline_textbbox(
                        (0, 0), wrapped, font=font, spacing=2)
                except Exception:
                    bbox = (0, 0, 0, 0)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
                if tw <= box_w * 1.05 and th <= box_h * 1.05:
                    final_font = font
                    final_wrapped = wrapped
                    text_w, text_h = tw, th
                    used_fs = fs
                    break

        # final fallback: smallest readable and accept expansion
        if final_font is None:
            final_font = load_font(max(MIN_FONT_SIZE, min(start_fs, 20)))
            final_wrapped = "\n".join(textwrap.wrap(
                new_text, width=max(1, int(box_w / 8)),
                break_long_words=False, break_on_hyphens=False))
            tmp_img = Image.new(
                "RGBA", (box_w + 160, box_h + 160), (255, 255, 255, 0))
            td = ImageDraw.Draw(tmp_img)
            try:
                bbox = td.multiline_textbbox(
                    (0, 0), final_wrapped, font=final_font, spacing=2)
            except Exception:
                bbox = (0, 0, 0, 0)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            used_fs = int(final_font.size if hasattr(
                final_font, "size") else start_fs)

        # Try to expand placement rect if text bigger than original box (bounded)
        expand_x = max(0, int((text_w - box_w) / 2))
        expand_y = max(0, int((text_h - box_h) / 2))
        pad_x = min(expand_x, int(w_img * 0.25))
        pad_y = min(expand_y, int(h_img * 0.12))

        place_x1 = max(0, x1 - pad_x)
        place_y1 = max(0, y1 - pad_y)
        place_x2 = min(w_img, x2 + pad_x)
        place_y2 = min(h_img, y2 + pad_y)
        place_w = max(1, place_x2 - place_x1)
        place_h = max(1, place_y2 - place_y1)

        # Sample background to pick text color & background
        avg_rgb = sample_background_color(
            img_pil, (place_x1, place_y1, place_x2, place_y2), pad=6)
        text_color = choose_text_color(avg_rgb)

        # Build temporary canvas sized to text but clamp to place size * 1.2
        tmp_w = int(min(max(48, text_w + 24), place_w * 1.2))
        tmp_h = int(min(max(32, text_h + 16), place_h * 1.2))
        tmp_w = max(tmp_w, 32)
        tmp_h = max(tmp_h, 24)

        tmp = Image.new("RGBA", (tmp_w, tmp_h), (255, 255, 255, 0))
        td = ImageDraw.Draw(tmp)

        # compute spacing, padding scaled to font
        spacing_px = max(
            1, int(getattr(final_font, "size", MIN_FONT_SIZE) * 0.12))
        padding_px = max(
            4, int(getattr(final_font, "size", MIN_FONT_SIZE) * 0.28))

        # center text on tmp (or left-align for checkbox-like small boxes)
        left_align = (x1 < w_img * 0.35 and box_w < w_img * 0.25)
        if left_align:
            text_x = padding_px
        else:
            text_x = max(padding_px, int((tmp_w - text_w) / 2))
        text_y = max(padding_px, int((tmp_h - text_h) / 2))

        # compute exact inner bbox and draw a semi-opaque background using avg_rgb
        try:
            inner_bb = td.multiline_textbbox(
                (text_x, text_y), final_wrapped, font=final_font, spacing=spacing_px)
            bx0, by0, bx1, by1 = inner_bb
        except Exception:
            bx0, by0, bx1, by1 = text_x - 2, text_y - \
                2, text_x + text_w + 2, text_y + text_h + 2

        bx0 = max(0, bx0 - 4)
        by0 = max(0, by0 - 3)
        bx1 = min(tmp_w, bx1 + 4)
        by1 = min(tmp_h, by1 + 3)

        # choose alpha so text remains legible
        alpha_bg = 200 if text_color == (255, 255, 255) else 220
        bg_rgba = (avg_rgb[0], avg_rgb[1], avg_rgb[2], alpha_bg)
        td.rectangle([bx0, by0, bx1, by1], fill=bg_rgba)

        # stroke width proportional to font
        stroke_fill = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)
        stroke_width = max(0, int(max(1, used_fs * 0.06)))

        # draw text (vertical/horizontal)
        try:
            td.multiline_text((text_x, text_y), final_wrapped, font=final_font, fill=text_color,
                              spacing=spacing_px, stroke_width=stroke_width, stroke_fill=stroke_fill)
        except TypeError:
            td.multiline_text((text_x, text_y), final_wrapped, font=final_font, fill=text_color,
                              spacing=spacing_px)

        # Decide rotation angle but keep English near-horizontal
        raw_angle = get_text_angle(poly)
        if all(ord(ch) < 128 for ch in new_text):
            angle = raw_angle if abs(raw_angle) <= 10 else 0.0
        else:
            angle = raw_angle if abs(raw_angle) <= 15 else 0.0

        # Rotate and place centered in place rectangle, clamped to image
        rotated = tmp.rotate(-angle, expand=True)
        rx = place_x1 + (place_w - rotated.width) / 2
        ry = place_y1 + (place_h - rotated.height) / 2
        rx = max(0, min(w_img - rotated.width, rx))
        ry = max(0, min(h_img - rotated.height, ry))

        out.alpha_composite(rotated, dest=(int(round(rx)), int(round(ry))))

    return out.convert("RGB")

# ---------------------------
# High level pipeline (attempt LaMa then fallback)
# ---------------------------


def inpaint_image_with_boxes(pil_img: Image.Image, annotations: List[Dict], expand_mask: int = 8) -> Image.Image:
    """
    Create mask from boxes, run LaMa inpaint if available and prefer_lama True,
    otherwise fallback to OpenCV inpaint.
    Returns: inpainted_pil
    """
    mask = create_mask_for_boxes(pil_img.size, annotations, expand=expand_mask)

    out = inpaint_with_opencv(pil_img, mask, method='telea')
    return out
