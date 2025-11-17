import os
import textwrap
import math
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple
import streamlit as st
import requests
import json
import re
import tempfile
import zipfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image
import concurrent.futures
from google.cloud import vision
from google.cloud import translate_v2 as translate
import logging
import json
from google.oauth2 import service_account
import html

logging.basicConfig(level=logging.INFO)

# ----------------- CONFIG & AUTHENTICATION CHECK -----------------
st.set_page_config(layout="wide", page_title="Amazon Image Translator (GCP)")
st.title("Amazon Product Image Translator — GCP Vision + Translate")

MAX_IMAGES = 3
BOX_EXPANSION_INPAINT = 12  # Increased for cleaner text removal
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

TEXT_COLOR = 'white'
FILL_COLOR = (0, 0, 0, 255)
INITIAL_FONT_SIZE = 50
FONT_PATH = "assets/fonts/POPPINS-MEDIUM.TTF"
DISTANCE_THRESHOLD = 400
Y_DIFF_THRESHOLD = 2
MIN_FONT_SIZE = 12

try:
    # 1️⃣ Streamlit Cloud method (secrets.toml)
    gcp_credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    vision_client = vision.ImageAnnotatorClient(credentials=gcp_credentials)
    translate_client = translate.Client(credentials=gcp_credentials)

    logging.info(
        "✅ GCP clients initialized successfully using Streamlit secrets.")

except Exception as e:
    logging.warning(f"Streamlit secrets failed: {e}")

    try:
        # 2️⃣ Fallback: Environment variable (Render.com, local machine)
        creds_dict = json.loads(os.environ["GOOGLE_CREDENTIALS"])

        gcp_credentials = service_account.Credentials.from_service_account_info(
            creds_dict
        )

        vision_client = vision.ImageAnnotatorClient(
            credentials=gcp_credentials)
        translate_client = translate.Client(credentials=gcp_credentials)

        logging.info(
            "✅ GCP clients initialized successfully using environment variable.")

    except Exception as e2:
        logging.error(f"❌ Failed to initialize GCP clients: {e2}")
        st.error(
            f"🚨 Failed to initialize GCP Clients.\n"
            "Check Streamlit secrets or Render environment variables."
        )
        st.stop()

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


def draw_rotated_text_block(base_img, text, font, angle, box_rect):
    """
    Draw a FULL WORD rotated by `angle` inside bounding box.
    text = word to draw
    font = PIL font
    angle = 0, 90, 180, 270
    box_rect = (x1, y1, x2, y2)
    """

    x1, y1, x2, y2 = box_rect
    box_w = x2 - x1
    box_h = y2 - y1

    # --- 1) Render text on transparent canvas ---
    temp = Image.new("RGBA", (2000, 500), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((0, 0), text, font=font, fill="black")

    # --- 2) Crop tight ---
    bbox = temp.getbbox()
    text_img = temp.crop(bbox)

    # --- 3) Rotate whole word ---
    rotated = text_img.rotate(angle, expand=True)

    # --- 4) Center it inside bounding box ---
    rx = x1 + (box_w - rotated.width) / 2
    ry = y1 + (box_h - rotated.height) / 2

    # --- 5) Paste ---
    base_img.alpha_composite(rotated, dest=(int(rx), int(ry)))

    return base_img


# ---------------------------
# Overlay translated text nicely (optimized)
# ---------------------------
def overlay_translated_text(img_pil: Image.Image, annotations: List[Dict], font_path=None):
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
        box_w = max(3, x2 - x1)
        box_h = max(3, y2 - y1)

        new_text = (ann.get("new_text") or ann.get(
            "translated_text") or ann.get("text") or "").strip()
        if not new_text:
            continue

        aspect = float(box_h) / max(1.0, box_w)
        want_vertical = aspect >= 2.5

        # -------------------------------
        # FONT SELECTION HELPER
        # -------------------------------
        def pick_font_for_size(w_limit, h_limit, text, vertical=False):
            for fs in range(72, MIN_FONT_SIZE - 1, -1):
                f = ImageFont.truetype(font_path, fs)
                tmp = Image.new("RGBA", (2000, 800), (0, 0, 0, 0))
                d = ImageDraw.Draw(tmp)

                if vertical:
                    bbox = d.textbbox((0, 0), text, font=f)
                else:
                    wrapped = "\n".join(textwrap.wrap(
                        text, width=max(1, int(w_limit / max(1, (fs * 0.6)))),
                        break_on_hyphens=False, break_long_words=False
                    ))
                    bbox = d.multiline_textbbox((0, 0), wrapped, font=f)

                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

                if tw <= w_limit * 1.05 and th <= h_limit * 1.05:
                    return f, (tw, th), wrapped if not vertical else text

            # fallback
            f = ImageFont.truetype(font_path, MIN_FONT_SIZE)
            return f, (w_limit, h_limit), text

        # ============================================================
        # CASE 1 — STRICT VERTICAL MODE
        # ============================================================
        if want_vertical:
            font, (tw, th), _ = pick_font_for_size(
                box_w, box_h, new_text, vertical=True)

            # TTB or BTT
            angle = -90
            if get_text_angle(poly) < -30:
                angle = 270

            out = draw_rotated_text_block(
                out, new_text, font, angle, (x1, y1, x2, y2)
            )

            continue   # ********** CRITICAL — DO NOT PROCESS HORIZONTAL **********

        # ============================================================
        # CASE 2 — NORMAL HORIZONTAL MODE
        # ============================================================

        font, (tw, th), wrapped = pick_font_for_size(
            box_w, box_h, new_text, vertical=False)

        # expand area if needed
        needed_w = tw + int(font.size * 1.8)
        needed_h = th + int(font.size * 1.6)

        expand_x = max(0, int((needed_w - box_w) / 2))
        expand_y = max(0, int((needed_h - box_h) / 2))

        # clamp expansion inside image
        expand_x = min(expand_x, int(w_img * 0.30))
        expand_y = min(expand_y, int(h_img * 0.20))

        px1 = max(0, x1 - expand_x)
        py1 = max(0, y1 - expand_y)
        px2 = min(w_img, x2 + expand_x)
        py2 = min(h_img, y2 + expand_y)

        place_w = px2 - px1
        place_h = py2 - py1

        avg_rgb = sample_background_color(img_pil, (px1, py1, px2, py2))
        text_color = choose_text_color(avg_rgb)
        stroke = (255, 255, 255) if text_color == (0, 0, 0) else (0, 0, 0)

        # padded canvas
        tmp = Image.new("RGBA", (place_w, place_h), (255, 255, 255, 0))
        d = ImageDraw.Draw(tmp)

        padding = max(4, int(font.size * 0.3))
        spacing = max(1, int(font.size * 0.12))

        # measure
        bb = d.multiline_textbbox(
            (padding, padding), wrapped, font=font, spacing=spacing)
        bx0, by0, bx1, by1 = bb
        d.rectangle([bx0-3, by0-3, bx1+3, by1+3],
                    fill=(avg_rgb[0], avg_rgb[1], avg_rgb[2], 210))

        d.multiline_text(
            (padding, padding), wrapped,
            font=font, fill=text_color,
            spacing=spacing,
            stroke_width=max(1, int(font.size * 0.06)),
            stroke_fill=stroke
        )

        angle = get_text_angle(poly)
        if abs(angle) > 15:
            angle = 0

        rotated = tmp.rotate(-angle, expand=True)

        rx = px1 + (place_w - rotated.width) / 2
        ry = py1 + (place_h - rotated.height) / 2

        out.alpha_composite(rotated, dest=(int(rx), int(ry)))

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


# ---------------------------
# Amazon image extraction (Robust Logic)
# ---------------------------


def normalize_amazon_image_url(url: str) -> str:
    """Return simplified version (try to keep hi-res)"""
    if not url:
        return url
    url = re.sub(r"\?.*$", "", url)
    url = re.sub(r"\._[A-Za-z0-9,._-]+_\.", ".", url)
    m = re.search(r"(https?://[^/]+/images/I/[^.]+)\.(?:jpg|png|jpeg)$", url)
    if m:
        base = m.group(1)
        return base + ".jpg"
    return url


def extract_amazon_image_urls_from_html(html, base_url, max_images=None):
    soup = BeautifulSoup(html, "html.parser")
    found = []

    def add(u):
        if not u:
            return
        if u.startswith("//"):
            u = "https:" + u
        elif u.startswith("/"):
            u = urljoin(base_url, u)
        u = normalize_amazon_image_url(u)
        if "m.media-amazon.com/images/I/" not in u and "images-fe.ssl-images-amazon.com/images/I/" not in u:
            return
        if u not in found:
            found.append(u)

    for m in re.finditer(r'"colorImages"\s*:\s*\{.*?"initial"\s*:\s*(\[[^\]]+\])', html, re.DOTALL):
        try:
            arr = json.loads(m.group(1))
            for e in arr:
                u = e.get("hiRes") or e.get("large") or e.get(
                    "mainUrl") or e.get("thumb")
                add(u)
        except Exception:
            pass

    for script in soup.find_all("script"):
        if not script.string:
            continue
        for match in re.findall(r'"large"\s*:\s*"([^"]+)"', script.string):
            add(match)
        for match in re.findall(r'"hiRes"\s*:\s*"([^"]+)"', script.string):
            add(match)

    # ... (other extraction methods omitted for brevity) ...

    if max_images:
        return found[:max_images]
    return found


def fetch_image_urls(product_url, max_image=None, timeout=15):
    r = requests.get(product_url, headers=HEADERS,
                     timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return extract_amazon_image_urls_from_html(r.text, r.url, max_images=max_image)


def download_image(url, timeout=12):
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


# ---------------------------
# GCP Vision OCR: IMAGE LEVEL (CRITICAL FIX FOR COHERENCE)
# ---------------------------
def vision_ocr_paragraphs(pil_img):
    """
    Uses Google Vision API but returns PARAGRAPH-LEVEL boxes only.
    Much cleaner for translation overlays.
    """
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())

    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(response.error.message)

    results = []

    annotation = response.full_text_annotation

    for page in annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:

                # collect paragraph text
                words = []
                for word in para.words:
                    symbols = "".join([s.text for s in word.symbols])
                    words.append(symbols)
                text = " ".join(words).strip()

                if not text:
                    continue

                # paragraph bounding box
                bbox = para.bounding_box.vertices
                vertices = [{"x": v.x, "y": v.y} for v in bbox]

                results.append({
                    "text": text,
                    "bounding_box": vertices
                })

    return results


# ---------------------------
# Translate via GCP (v2 Client)
# ---------------------------
def gcloud_translate_texts(text_list, target="en"):
    if not text_list:
        return []
    try:
        resp = translate_client.translate(text_list, target_language=target)
        out = []
        if isinstance(resp, dict):
            out = [html.unescape(resp.get("translatedText", ""))]
        else:
            for it in resp:
                # Some responses may be dicts or strings
                if isinstance(it, dict):
                    out.append(html.unescape(it.get("translatedText", "")))
                else:
                    out.append(html.unescape(str(it)))
        # Ensure length
        if len(out) != len(text_list):
            out = (out + text_list)[:len(text_list)]
        return out
    except Exception as e:
        logging.error(f"GCP Translate error: {e}")
        return text_list


def upscale_image(pil_img, scale_factor=2):
    """Upscale image using LANCZOS filter for better quality."""
    new_size = (int(pil_img.width * scale_factor),
                int(pil_img.height * scale_factor))
    return pil_img.resize(new_size, resample=Image.LANCZOS)

# ---------------------------
# Translate (FINAL LAYOUT)
# ---------------------------


def translate_text(ocr_blocks):
    """
    Translate a list of blocks (each block is dict with 'text','bounding_box').
    Returns the same list with 'new_text' added.
    """
    if not ocr_blocks:
        return []

    texts = [b["text"] for b in ocr_blocks]
    translations = gcloud_translate_texts(texts, target="en")
    for b, tr in zip(ocr_blocks, translations):
        b["new_text"] = tr
    return ocr_blocks

# ---------------------------
# Full pipeline (per-image)
# ---------------------------


def process_image_gcp(pil_img):
    """
    Full per-image pipeline:
      - Vision OCR -> block extraction
      - Merge close blocks into readable lines
      - Translate merged blocks
      - Inpaint + overlay -> cleaned, translated final
    Returns: (cleaned_pil, final_translated_pil, meta)
    """
    try:
        # 1) OCR (block-level)
        blocks = vision_ocr_paragraphs(pil_img)
    except Exception as e:
        return pil_img, pil_img, {"error": str(e)}

    if not blocks:
        return pil_img, pil_img, {"detected": 0}

    # 2) Translate
    translated_blocks = translate_text(blocks)
    # 3) Inpaint & overlay
    cleaned, final_img, meta = process_inpaint_pipeline(
        pil_img.copy(), translated_blocks)
    meta["detected"] = len(translated_blocks)
    return cleaned, final_img, meta


def process_inpaint_pipeline(img_pil, annotations):
    """
    This will be the function called by Streamlit.
    Must return (cleaned, final_image, meta)
    """
    # 1) Remove original text
    cleaned = inpaint_image_with_boxes(
        img_pil, annotations, expand_mask=12)

    # 2) Overlay translated text
    final = overlay_translated_text(
        cleaned, annotations, font_path=FONT_PATH)

    # Metadata for UI
    meta = {
        "detected": len(annotations),
        "orig_blocks": [a["text"] for a in annotations],
        "trans_blocks": [a.get("new_text", a["text"]) for a in annotations]
    }

    return cleaned, final, meta


# ---------------------------
# Streamlit UI wiring
# ---------------------------
if "results" not in st.session_state:
    st.session_state.results = []

st.markdown("Paste an **Amazon product URL**. The app will fetch up to N images and translate to English using GCP (Vision API + Translation API).")
col1, col2 = st.columns([3, 1])
with col2:
    max_images = st.number_input("Max images to process", 1)
    timeout_per_image = st.slider("Per-image timeout (sec)", 5, 40, 15)
url = st.text_input("Amazon product URL",
                    placeholder="https://amzn.asia/d/...")
run = st.button("Fetch & Translate")

if run and url:
    with st.spinner("Fetching product images..."):
        try:
            image_urls = fetch_image_urls(url, max_image=max_images)
        except Exception as e:
            st.error(f"Failed to fetch product page: {e}")
            image_urls = []

    if not image_urls:
        st.warning("No product images found.")
    else:
        st.success(f"Found {len(image_urls)} image(s). Starting processing...")

        results = []

        # Concurrent download + OCR/Translate
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(image_urls))) as ex:
            download_futs = {
                ex.submit(download_image, u): u for u in image_urls}
            downloaded = {}
            for fut in concurrent.futures.as_completed(download_futs, timeout=timeout_per_image * len(image_urls)):
                u = download_futs[fut]
                try:
                    downloaded[u] = fut.result()
                except Exception as e:
                    st.warning(f"Failed download {u}: {e}")

            proc_futs = {ex.submit(process_image_gcp, upscale_image(
                img, scale_factor=2)): u for u, img in downloaded.items()}
            for fut in concurrent.futures.as_completed(proc_futs, timeout=timeout_per_image * len(proc_futs)):
                u = proc_futs[fut]
                try:
                    cleaned, final_img, meta = fut.result()
                    results.append(
                        (u, downloaded[u], cleaned, final_img, meta))
                except Exception as e:
                    st.warning(f"Processing failed for {u}: {e}")

        # ✅ Persist results to session
        st.session_state.results = results

# ✅ Display results (even after rerun)
if st.session_state.results:
    results = st.session_state.results
    for idx, (url_img, orig, cleaned, final_img, meta) in enumerate(results):
        st.markdown(f"### Image {idx+1}")
        c1, c2 = st.columns(2)
        with c1:
            st.image(upscale_image(orig), caption="Original")
        with c2:
            st.image(final_img, caption="Translated (English)")

        if meta.get("detected") is not None and meta.get("detected") > 0:
            c11, c12 = st.columns([1, 2])
            with c11:
                st.markdown("**Original Text:**")
                st.write(' '.join(meta.get("orig_blocks", [])))
            with c12:
                st.write(f"Detected blocks: {meta.get('detected')}")
                st.markdown("**Translated Text:**")
                st.write(', '.join(meta.get("trans_blocks", [])))
        st.markdown("---")

    # ✅ ZIP Download — now works after rerun
    if st.checkbox("Create ZIP of translated images"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        with zipfile.ZipFile(tmp.name, "w") as zf:
            for i, (_, _, _, final_img, _) in enumerate(results):
                b = BytesIO()
                final_img.save(b, format="PNG")
                zf.writestr(f"translated_{i+1}.png", b.getvalue())

        with open(tmp.name, "rb") as fh:
            st.download_button(
                "Download ZIP",
                data=fh.read(),
                file_name="translated_images.zip",
                mime="application/zip"
            )
