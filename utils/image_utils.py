import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import textwrap
from typing import List, Dict, Tuple
from .geometry_utils import poly_to_rect_coords, normalize_polygon, get_text_angle

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    return arr[:, :, ::-1].copy()

def cv2_to_pil(img_bgr: np.ndarray) -> Image.Image:
    rgb = img_bgr[:, :, ::-1]
    return Image.fromarray(rgb)

def create_mask_for_boxes(img_size: Tuple[int, int], annotations: List[Dict], expand: int = 6) -> Image.Image:
    """Create mask for text removal"""
    mask = Image.new("L", img_size, 0)
    draw = ImageDraw.Draw(mask)
    for ann in annotations:
        poly_raw = ann.get("bounding_box") or ann.get("bbox") or ann.get("poly") or ann.get("polygon")
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

def inpaint_with_opencv(pil_img: Image.Image, mask_pil: Image.Image, method: str = 'telea', 
                       dilate_iters: int = 1, blur_ksize: int = 7, inpaint_radius: int = 3) -> Image.Image:
    """OpenCV-based inpainting"""
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
    out = cv2.inpaint(bgr, mask_bin, inpaintRadius=inpaint_radius, flags=inpaint_flag)
    return cv2_to_pil(out)

def sample_background_color(img: Image.Image, bbox: Tuple[int, int, int, int], pad: int = 6) -> Tuple[int, int, int]:
    """Sample average RGB color around bounding box"""
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
    """Choose text color for best contrast"""
    r, g, b = rgb
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return (0, 0, 0) if lum > 160 else (255, 255, 255)

def upscale_image(pil_img, scale_factor=2):
    """Upscale image using LANCZOS filter"""
    new_size = (int(pil_img.width * scale_factor), int(pil_img.height * scale_factor))
    return pil_img.resize(new_size, resample=Image.LANCZOS)