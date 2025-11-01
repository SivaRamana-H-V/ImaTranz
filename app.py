import streamlit as st
import requests
import json
import re
import textwrap
import tempfile
import zipfile
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
import cv2
from sklearn.cluster import KMeans
import concurrent.futures
from google.cloud import vision
from google.cloud import translate_v2 as translate
import logging
import os

logging.basicConfig(level=logging.INFO)

# ----------------- CONFIG & AUTHENTICATION CHECK -----------------
st.set_page_config(layout="wide", page_title="Amazon Image Translator (GCP)")
st.title("Amazon Product Image Translator — GCP Vision + Translate")

FONT_PATH = "arial.ttf"  # CHANGE THIS to a font file available on your deploy system
MAX_IMAGES = 3
BOX_EXPANSION_INPAINT = 12  # Increased for cleaner text removal
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# --- GCP Client Initialization ---
# Check for credentials environment variable first
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    st.error("🚨 GCP Authentication Failed: The 'GOOGLE_APPLICATION_CREDENTIALS' environment variable is not set.")
    st.info(
        "Please set the environment variable pointing to your Service Account JSON key.")
    # Use st.stop() to halt execution if authentication is guaranteed to fail
    # st.stop()

try:
    vision_client = vision.ImageAnnotatorClient()
    translate_client = translate.Client()
except Exception as e:
    st.error(
        f"🚨 Failed to initialize GCP Clients. Check your permissions and setup. Error: {e}")
    # st.stop()
# --- End GCP Client Initialization ---

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
        return base + "._SL1500_.jpg"
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
# GCP Vision OCR: DOCUMENT LEVEL (CRITICAL FIX FOR COHERENCE)
# ---------------------------
def vision_ocr_blocks_texts(pil_img, min_conf=0.0):
    """
    Uses DOCUMENT_TEXT_DETECTION to get coherent paragraphs/blocks.
    Returns list of (bbox, text, confidence)
    bbox = [(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
    """
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())

    response = vision_client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")

    results = []

    # full_text_annotation.pages -> blocks -> paragraphs -> words -> symbols
    fta = response.full_text_annotation
    if not fta or not getattr(fta, "pages", None):
        return []

    for page in fta.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                # Reconstruct paragraph text from words->symbols
                words = paragraph.words
                if not words:
                    continue

                paragraph_words = []
                for w in words:
                    # each word has symbols list; join them to form the word string
                    try:
                        word_text = "".join([s.text for s in w.symbols])
                    except Exception:
                        # fallback: try getattr
                        word_text = getattr(w, "text", "")
                    if word_text:
                        paragraph_words.append(word_text)

                if not paragraph_words:
                    continue

                # join words with spaces to form paragraph text
                paragraph_text = " ".join(paragraph_words).strip()
                if not paragraph_text:
                    continue

                # bounding box from paragraph
                verts = paragraph.bounding_box.vertices
                pts = []
                for v in verts:
                    x = int(getattr(v, "x", 0) or 0)
                    y = int(getattr(v, "y", 0) or 0)
                    pts.append((x, y))
                while len(pts) < 4:
                    pts.append(pts[-1])

                # paragraph confidence fallback: try paragraph.confidence, then block
                conf = getattr(paragraph, "confidence", None)
                if conf is None:
                    conf = getattr(block, "confidence", 1.0)

                if conf >= min_conf:
                    results.append((pts, paragraph_text, float(conf)))

    return results


# ---------------------------
# Translate via GCP (v2 Client)
# ---------------------------
def gcloud_translate_texts(text_list, target="en"):
    """
    Robust translation wrapper using translate_client.
    Returns a list of translations (same length as text_list).
    Falls back to original texts on error.
    """
    if not text_list:
        return []
    try:
        # translate_client.translate accepts list and returns list of dicts
        resp = translate_client.translate(text_list, target_language=target)
        if isinstance(resp, dict):
            # single response
            return [resp.get("translatedText", "")]
        elif isinstance(resp, list):
            out = []
            for item in resp:
                out.append(item.get("translatedText", "")
                           if isinstance(item, dict) else str(item))
            # ensure same length
            if len(out) != len(text_list):
                # fallback: map items best-effort
                logging.warning(
                    "Translate API returned unexpected length, falling back partial results.")
                # pad or trim
                out = (out + text_list)[:len(text_list)]
            return out
        else:
            logging.warning(
                "Unexpected translate response type, returning originals.")
            return text_list
    except Exception as e:
        logging.error(f"GCP Translate error: {e}")
        # Fallback to original text to ensure pipeline continues
        return text_list


# ---------------------------
# Color extraction (Adaptive Color)
# ---------------------------
def extract_dominant_color(pil_img, n_colors=3):
    try:
        img = pil_img.copy().convert("RGB").resize((120, 120))
        arr = np.array(img).reshape(-1, 3).astype(float)
        kmeans = KMeans(n_clusters=n_colors, n_init='auto')
        kmeans.fit(arr)
        colors = kmeans.cluster_centers_.astype(int)
        colors = sorted(colors, key=lambda c: np.mean(c))
        pick = colors[len(colors) // 2]
        return tuple(map(int, pick))
    except Exception:
        return (40, 160, 80)


# ---------------------------
# Dynamic Font Fitter
# ---------------------------

def fit_text_into_box(draw, text, font_path, box_w, box_h, max_font=None, min_font=8):
    """Dynamically fits text by shrinking/wrapping to box dimensions using modern Pillow methods."""
    if max_font is None:
        max_font = int(box_h * 0.9)
    font_size = max(min_font, max_font)

    # Use a dummy string and a standard font size (e.g., 10) to get a base width for wrapping
    try:
        base_font_for_wrap = ImageFont.truetype(font_path, 10)
    except Exception:
        base_font_for_wrap = ImageFont.load_default()

    while font_size >= min_font:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # 1. Estimate average character width for text wrapping (using base font is safe here)
        # Using textbbox to find the width of 'M' at a fixed size (10) for the wrapping calculation

        # ❌ ERROR SOURCE REMOVED: Replaced font.getlength/font.getsize with modern call

        # Temporarily use the small base font to estimate wrap width
        dummy_bbox = draw.textbbox((0, 0), "M", font=base_font_for_wrap)
        avg_char_w = dummy_bbox[2] - dummy_bbox[0]

        max_chars = max(1, int(box_w / max(1, avg_char_w)))
        wrapped = "\n".join(textwrap.wrap(text, width=max_chars))

        # 2. Multiline text size calculation using draw.multiline_textbbox (Correct Method)
        bbox_text = draw.multiline_textbbox(
            (0, 0), wrapped, font=font, spacing=2)
        tw = bbox_text[2] - bbox_text[0]
        th = bbox_text[3] - bbox_text[1]

        if tw <= box_w - 4 and th <= box_h - 4:
            return font, wrapped, tw, th
        font_size -= 1

    # Final fallback: use min font and truncate (Simplified logic to avoid old function calls)
    try:
        font = ImageFont.truetype(font_path, min_font)
    except Exception:
        font = ImageFont.load_default()

    # Fallback wrapping using a conservative character estimate
    max_chars_fallback = int(box_w / 6)
    truncated = text[: max(5, int(len(text) * 0.7))] + "…"
    wrapped = "\n".join(textwrap.wrap(
        truncated, width=max(1, max_chars_fallback)))

    # Final measurement using the correct method
    bbox_text = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=2)
    tw = bbox_text[2] - bbox_text[0]
    th = bbox_text[3] - bbox_text[1]
    return font, wrapped, tw, th

# ---------------------------
# Translate and overlay neatly (FINAL LAYOUT)
# ---------------------------


def translate_and_overlay_image_gcp(pil_img, ocr_results, font_path=FONT_PATH):
    """
    Enhanced version:
    - Adaptive contrast-based color scheme
    - Dynamic transparency (based on region brightness)
    - Smart text centering and padding
    - Crisp anti-aliased rendering for readable overlays
    """
    pil = pil_img.convert("RGBA")
    pil = pil.resize(pil.size, resample=Image.LANCZOS)
    overlay = Image.new("RGBA", pil.size, (255, 255, 255, 0))
    draw_overlay = ImageDraw.Draw(overlay, "RGBA")

    # Extract theme/accent color
    accent_color = extract_dominant_color(pil)
    bright = np.mean(accent_color) > 140

    # Colors
    text_color = (255, 255, 255, 245)
    shadow_color = (0, 0, 0, 200)
    box_base = (0, 0, 0, 160) if bright else (0, 0, 0, 210)
    accent_fill = tuple(list(accent_color) + [255])

    # Collect texts
    texts = [t.strip() for (_, t, _) in ocr_results if t.strip()]
    if not texts:
        return pil_img

    translations = gcloud_translate_texts(texts, target="en")
    if not isinstance(translations, list) or len(translations) != len(texts):
        translations = [(translations[i] if i < len(translations) else texts[i])
                        for i in range(len(texts))]

    for (bbox, orig_text, conf), translated in zip(ocr_results, translations):
        translated = translated.strip()
        if not translated:
            continue

        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < 8 or box_h < 8:
            continue

        # --- Adjust background transparency by local brightness ---
        region = pil.crop((x1, y1, x2, y2)).convert("L")
        brightness = np.mean(np.array(region))
        # adaptive_alpha = 230 - int((brightness / 255) * 100)
        box_fill_color = (0, 0, 0, 240)

        # --- Fit translated text ---
        font, wrapped, tw, th = fit_text_into_box(
            draw_overlay, translated, font_path, box_w * 0.95, box_h * 0.9)

        # Add small padding for text inside box
        pad_x, pad_y = int(box_w * 0.05), int(box_h * 0.05)
        tx = x1 + (box_w - tw) / 2
        ty = y1 + (box_h - th) / 2

        # --- Smooth rectangle + accent stripe ---
        draw_overlay.rounded_rectangle(
            [x1, y1, x2, y2], radius=6, fill=box_fill_color)
        draw_overlay.rectangle([x1, y1, x1 + 4, y2], fill=accent_fill)

        # --- Text shadow & highlight ---
        draw_overlay.multiline_text(
            (tx + 1.5, ty + 1.5), wrapped, font=font, fill=shadow_color, spacing=2)
        draw_overlay.multiline_text(
            (tx, ty), wrapped, font=font, fill=text_color, spacing=2)

    # Blend overlays smoothly
    final = Image.alpha_composite(pil, overlay)
    # Slight smoothing for anti-alias)
    final = final.filter(ImageFilter.SHARPEN)
    final = final.convert("RGB")
    return final, translations

# ---------------------------
# Full pipeline (per-image)
# ---------------------------


def process_image_gcp(pil_img):
    """
    OCR → Translate → Draw boxes
    (no text removal)
    """
    try:
        ocr_blocks = vision_ocr_blocks_texts(pil_img)
    except Exception as e:
        return pil_img, pil_img, {"error": str(e)}

    if not ocr_blocks:
        return pil_img, pil_img, {"detected": 0}

    # Draw translated boxes directly
    translated_image, trans_texts = translate_and_overlay_image_gcp(
        pil_img.copy(), ocr_blocks)

    meta = {
        "detected": len(ocr_blocks),
        "orig_blocks": [t for (_, t, _) in ocr_blocks],
        "trans_blocks": trans_texts
    }
    return pil_img, translated_image, meta

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

        st.write(image_urls)
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

            proc_futs = {ex.submit(process_image_gcp, img)
                                   : u for u, img in downloaded.items()}
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
            st.image(orig, caption="Original", use_container_width=True)
        with c2:
            st.image(final_img, caption="Translated (English)",
                     use_container_width=True)

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
