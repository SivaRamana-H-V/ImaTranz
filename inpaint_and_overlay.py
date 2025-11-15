"""
inpaint_and_overlay.py

Functions:
 - inpaint_image_with_boxes(image_pil, annotations, use_lama=True)
 - overlay_translated_text(image_pil, annotations, font_path)
 - process_and_save(input_path, annotations, output_path, translated_texts, font_path)

annotations format (list of dicts):
[
  { "text": "柔らかい",
    "bounding_box": [{"x":514,"y":37}, {"x":608,"y":37}, {"x":608,"y":57}, {"x":514,"y":57}] },
  ...
]

translated_texts must match annotations order OR you can include "new_text" inside each annotation.
"""
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Try import lama-cleaner style API. If not installed, we'll fallback.
try:
    from lama_cleaner.model_manager import ModelManager
    from lama_cleaner.schema import Config
    LAMA_AVAILABLE = True
except Exception:
    LAMA_AVAILABLE = False

# ---------------------------
# Utilities
# ---------------------------


def poly_to_rect_coords(polygon: List[Dict[str, int]]) -> Tuple[int, int, int, int]:
    xs = [p["x"] for p in polygon]
    ys = [p["y"] for p in polygon]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    return x_min, y_min, x_max, y_max


def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    # rgb -> bgr for cv2
    return arr[:, :, ::-1].copy()


def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(rgb)

# ---------------------------
# LaMa inpainting (CPU) - lightweight usage
# ---------------------------


def inpaint_with_lama(pil_img: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """
    Requires lama-cleaner installed and a ModelManager API.
    This uses device="cpu". If not available, raise ImportError.
    """
    if not LAMA_AVAILABLE:
        raise ImportError("lama-cleaner not available")

    # initialize model manager (cached by lama-cleaner)
    model = ModelManager(name="lama", device="cpu")  # device cpu
    cfg = Config()
    # Convert PIL to expected input: PIL Image and mask
    result = model(pil_img.convert("RGB"), mask_pil.convert("L"), cfg)
    return result

# ---------------------------
# OpenCV fallback (fast, CPU)
# ---------------------------


def inpaint_with_opencv(pil_img: Image.Image, mask_pil: Image.Image, method='telea') -> Image.Image:
    """
    Uses OpenCV inpaint as fallback. Not as perfect as deep inpaint but decent for many cases.
    method: 'telea' or 'ns'
    """
    bgr = pil_to_cv2(pil_img)
    mask = np.array(mask_pil.convert("L"))
    # OpenCV expects mask binary 0/255 (uint8)
    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    inpaint_flag = cv2.INPAINT_TELEA if method == 'telea' else cv2.INPAINT_NS
    # inpaint requires 8-bit 3-ch image
    out = cv2.inpaint(bgr, mask_bin, inpaintRadius=3, flags=inpaint_flag)
    return cv2_to_pil(out)

# ---------------------------
# Create mask from annotations (boxes)
# ---------------------------


def create_mask_for_boxes(img_size: Tuple[int, int], annotations: List[Dict], expand: int = 6) -> Image.Image:
    """
    Build a single mask image (L) where box regions are white (255).
    expand: pixels to expand bounding box (helps remove nearby strokes)
    """
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        poly = ann["bounding_box"]
        x1, y1, x2, y2 = poly_to_rect_coords(poly)
        # expand conservatively
        x1 = max(0, x1 - expand)
        y1 = max(0, y1 - expand)
        x2 = min(img_size[0], x2 + expand)
        y2 = min(img_size[1], y2 + expand)
        draw.rectangle([x1, y1, x2, y2], fill=255)
    return mask

# ---------------------------
# Overlay translated text nicely
# ---------------------------


def overlay_translated_text(img_pil: Image.Image, annotations: List[Dict], font_path: str = "arial.ttf"):
    """
    Draws translated/new text inside each bounding box.
    Each annotation should either have "new_text" or we expect separate translated_texts list externally.
    Handles vertical boxes by rotating rendered text.
    """
    out = img_pil.convert("RGBA")
    draw = ImageDraw.Draw(out)

    for ann in annotations:
        poly = ann["bounding_box"]
        x1, y1, x2, y2 = poly_to_rect_coords(poly)
        box_w = x2 - x1
        box_h = y2 - y1
        new_text = ann.get("new_text", ann.get(
            "translated_text", ann.get("text", "")))
        # if text is empty skip
        if not new_text:
            continue

        # Choose font size to fit (binary search)
        # Start from large and reduce until fits
        max_font = max(8, int(box_h * 0.9))
        good_font = None
        good_wrapped = new_text
        for fs in range(max_font, 7, -1):
            try:
                font = ImageFont.truetype(font_path, fs)
            except Exception:
                font = ImageFont.load_default()
            # estimate size with multiline_textbbox
            wrapped = new_text
            # naive wrap for long text
            # guess chars per line ~ box_w / (fs * 0.6)
            est_chars = max(1, int(box_w / max(1, fs * 0.6)))
            if len(new_text) > est_chars:
                import textwrap
                wrapped = "\n".join(textwrap.wrap(new_text, width=est_chars))
            bbox = draw.multiline_textbbox(
                (0, 0), wrapped, font=font, spacing=2)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            if tw <= box_w - 4 and th <= box_h - 4:
                good_font = font
                good_wrapped = wrapped
                text_w, text_h = tw, th
                break

        if good_font is None:
            try:
                good_font = ImageFont.truetype(font_path, 10)
            except Exception:
                good_font = ImageFont.load_default()
            # fallback wrap
            import textwrap
            good_wrapped = "\n".join(textwrap.wrap(
                new_text, width=max(1, int(box_w / 6))))
            bbox = draw.multiline_textbbox(
                (0, 0), good_wrapped, font=good_font, spacing=2)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]

        # determine orientation: if box is tall vs wide -> treat as vertical
        is_vertical = box_h > box_w * 1.4

        if not is_vertical:
            x_text = x1 + (box_w - text_w) / 2
            y_text = y1 + (box_h - text_h) / 2
            # Text color: try dark gray for natural Amazon look
            draw.multiline_text((x_text, y_text), good_wrapped, font=good_font, fill=(
                40, 40, 40), spacing=2, align="center")
        else:
            # render rotated text onto a temporary image, then paste rotated into box center
            tmp = Image.new("RGBA", (box_h, box_w),
                            (255, 255, 255, 0))  # swapped dims
            td = ImageDraw.Draw(tmp)
            # choose font scaled down so the text fits in swapped area
            # draw text centered horizontally in tmp
            bbox_tmp = td.multiline_textbbox(
                (0, 0), good_wrapped, font=good_font, spacing=2)
            tw_tmp = bbox_tmp[2] - bbox_tmp[0]
            th_tmp = bbox_tmp[3] - bbox_tmp[1]
            tx = (tmp.width - tw_tmp) / 2
            ty = (tmp.height - th_tmp) / 2
            td.multiline_text((tx, ty), good_wrapped, font=good_font, fill=(
                40, 40, 40), spacing=2, align="center")
            # rotate back to vertical orientation and paste
            rotated = tmp.rotate(90, expand=True)
            # compute paste position (center into rectangle)
            rx = x1 + (box_w - rotated.width) / 2
            ry = y1 + (box_h - rotated.height) / 2
            out.alpha_composite(rotated, dest=(int(rx), int(ry)))

    return out.convert("RGB")

# ---------------------------
# High level pipeline (attempt LaMa then fallback)
# ---------------------------


def inpaint_image_with_boxes(pil_img: Image.Image, annotations: List[Dict], expand_mask: int = 8, prefer_lama: bool = True):
    """
    Create mask from boxes, run LaMa inpaint if available and prefer_lama True,
    otherwise fallback to OpenCV inpainting.
    Returns: inpainted_pil
    """
    mask = create_mask_for_boxes(pil_img.size, annotations, expand=expand_mask)

    # Try LaMa if requested and available
    if prefer_lama and LAMA_AVAILABLE:
        try:
            out = inpaint_with_lama(pil_img, mask)
            return out
        except Exception as e:
            print("LaMa inpainting failed, falling back to OpenCV inpaint:", e)

    # Fallback
    out = inpaint_with_opencv(pil_img, mask, method='telea')
    return out

# ---------------------------
# Convenience: full process and save
# ---------------------------


def process_and_save(input_path: str,
                     annotations: List[Dict],
                     output_path: str,
                     translated_texts: List[str] = None,
                     font_path: str = "arial.ttf",
                     prefer_lama: bool = True):
    """
    input_path -> loads image
    annotations -> list of boxes (same order as translated_texts), you can also include "new_text" per annotation
    translated_texts -> list of strings (optional) to inject (same order)
    output_path -> where to save final image
    """
    img = Image.open(input_path).convert("RGB")

    # attach new_text into annotations if provided
    if translated_texts:
        for i, t in enumerate(translated_texts):
            if i < len(annotations):
                annotations[i]["new_text"] = t

    # 1) inpaint
    inpainted = inpaint_image_with_boxes(
        img, annotations, expand_mask=8, prefer_lama=prefer_lama)

    # 2) overlay text
    final = overlay_translated_text(
        inpainted, annotations, font_path=font_path)

    final.save(output_path, quality=95)
    return output_path


# ---------------------------
# Minimal example usage when run as script
# ---------------------------
if __name__ == "__main__":
    import json
    import sys
    # Example usage:
    # python inpaint_and_overlay.py input.jpg annotations.json output.jpg
    if len(sys.argv) < 4:
        print("Usage: python inpaint_and_overlay.py input.jpg annotations.json output.jpg")
        sys.exit(1)
    input_p = sys.argv[1]
    ann_p = sys.argv[2]
    out_p = sys.argv[3]
    with open(ann_p, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    # Optionally you can prepare translated list:
    translated = [ann.get("translated_text") or ann.get(
        "new_text") or "" for ann in annotations]
    process_and_save(input_p, annotations, out_p, translated_texts=translated)
    print("Saved ->", out_p)
