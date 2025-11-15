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
from sklearn.cluster import KMeans
import concurrent.futures
from google.cloud import vision
from google.cloud import translate_v2 as translate
import logging
import json
from google.oauth2 import service_account
from inpaint_and_overlay import inpaint_image_with_boxes, overlay_translated_text

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

TEXT_COLOR = 'white'
FILL_COLOR = (0, 0, 0, 255)
INITIAL_FONT_SIZE = 50
FONT_PATH = "arial.ttf"
DISTANCE_THRESHOLD = 400
Y_DIFF_THRESHOLD = 2

try:
    # Load credentials from Streamlit secrets (stored securely in the cloud)
    gcp_credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    vision_client = vision.ImageAnnotatorClient(credentials=gcp_credentials)
    translate_client = translate.Client(credentials=gcp_credentials)

    logging.info(
        "✅ GCP clients initialized successfully using Streamlit secrets.")

except Exception as e:
    st.error(
        f"🚨 Failed to initialize GCP Clients. Check your secrets configuration. Error: {e}")
    st.stop()

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
def vision_ocr_blocks_texts(pil_img, min_conf=0.0):
    """
    Perform OCR using Google Cloud Vision API.
    Returns list of dictionaries with 'text' and 'bounding_box'.
    """

    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    image = vision.Image(content=buf.getvalue())

    response = vision_client.text_detection(image=image)
    if response.error.message:
        raise Exception(f"Vision API error: {response.error.message}")

    annotations = response.text_annotations
    if not annotations:
        logging.warning("No text annotations detected.")
        return []

    texts_data = []
    # The first item is the full text; others are word-level boxes
    for text_obj in annotations[1:]:
        vertices = text_obj.bounding_poly.vertices
        bbox = []
        for v in vertices:
            bbox.append({
                "x": getattr(v, "x", 0),
                "y": getattr(v, "y", 0)
            })
        texts_data.append({
            "text": text_obj.description,
            "bounding_box": bbox
        })

    # Optionally save output to a local JSON file
    try:
        with open("detected_texts.json", "w", encoding="utf-8") as f:
            json.dump(texts_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.warning(f"Failed to write detected_texts.json: {e}")

    logging.info(f"✅ Detected {len(texts_data)} text boxes.")
    return texts_data

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


def upscale_image(pil_img, scale_factor=2):
    """Upscale image using LANCZOS filter for better quality."""
    new_size = (int(pil_img.width * scale_factor),
                int(pil_img.height * scale_factor))
    return pil_img.resize(new_size, resample=Image.LANCZOS)

# ---------------------------
# Translate (FINAL LAYOUT)
# ---------------------------


def translate_text(ocr_results):
    """
    Enhanced version:
    - Adaptive contrast-based color scheme
    - Dynamic transparency
    - Smart centering and readable overlays
    """
    translated_annotations = ocr_results.copy()  # ✅ initialize as a list

    for item in ocr_results:
        original_text = item['text']

        if original_text.count('\n') > 1 and len(original_text) > 20:
            continue

        # Translate text
        translated_text = gcloud_translate_texts(
            [original_text], target="en")[0]
        item['new_text'] = translated_text

    return translated_annotations  # ✅ return list, not single string


def get_box_coords(item):
    """Helper to extract (x_min, y_min, x_max, y_max) from a data item."""
    vertices = item["bounding_box"]
    x_min = min(v["x"] for v in vertices)
    y_min = min(v["y"] for v in vertices)
    x_max = max(v["x"] for v in vertices)
    y_max = max(v["y"] for v in vertices)
    return x_min, y_min, x_max, y_max


def merge_close_boxes(data, distance_threshold, y_diff_threshold):
    """
    Merges adjacent bounding boxes and their text if they are close enough
    horizontally and on the same vertical line.
    """
    if not data:
        return []

    # Sort data primarily by y_min (top line), then by x_min (left to right)
    # This is crucial for processing text in reading order
    sorted_data = sorted(data, key=lambda item: (
        get_box_coords(item)[1], get_box_coords(item)[0]))

    merged_data = []

    # Skip the massive, first bounding box that contains all the text
    # You can adjust this condition based on your exact data structure
    current_data_index = 0
    if len(sorted_data[0]["text"].split('\n')) > 1:
        merged_data.append(sorted_data[0])
        current_data_index = 1

    i = current_data_index
    while i < len(sorted_data):
        current_item = sorted_data[i]
        c_xmin, c_ymin, c_xmax, c_ymax = get_box_coords(current_item)

        # Start a new merged entry with the current item
        merged_text = current_item["text"]
        new_xmin, new_ymin, new_xmax, new_ymax = c_xmin, c_ymin, c_xmax, c_ymax
        j = i + 1

        while j < len(sorted_data):
            next_item = sorted_data[j]
            n_xmin, n_ymin, n_xmax, n_ymax = get_box_coords(next_item)

            # Check for horizontal proximity (c_xmax to n_xmin)
            horizontal_gap = n_xmin - c_xmax

            # Check for vertical alignment (assuming they are roughly on the same line)
            # Use overlap or small difference in y_min/y_max
            y_diff = abs(c_ymin - n_ymin)

            if horizontal_gap <= distance_threshold and y_diff <= y_diff_threshold:
                # Merge the text
                merged_text += next_item["text"]

                # Expand the bounding box
                new_xmin = min(new_xmin, n_xmin)
                new_ymin = min(new_ymin, n_ymin)
                new_xmax = max(new_xmax, n_xmax)
                new_ymax = max(new_ymax, n_ymax)

                # Move to the next item to check for further merging
                j += 1
            else:
                # Items are too far apart (or on a new line), stop merging
                break

        # Finalize the merged entry (create a simplified, combined bounding box)
        merged_data.append({
            "text": merged_text,
            "bounding_box": [
                {"x": new_xmin, "y": new_ymin},
                {"x": new_xmax, "y": new_ymin},
                {"x": new_xmax, "y": new_ymax},
                {"x": new_xmin, "y": new_ymax}
            ]
        })

        # Move the main loop index past all merged items
        i = j

    return merged_data


# ---------------------------
# Full pipeline (per-image)
# ---------------------------

def process_image_gcp(pil_img):
    """
    OCR → Translate → Draw boxes
    (no text removal)
    """
    try:
        text_data = vision_ocr_blocks_texts(pil_img)
    except Exception as e:
        return pil_img, pil_img, {"error": str(e)}

    if not text_data:
        return pil_img, pil_img, {"detected": 0}

    bounding_data = []

    with open('detected_texts.json', 'r', encoding='utf-8') as f:
        bounding_data = json.load(f)

    merged_data_final = merge_close_boxes(
        bounding_data, DISTANCE_THRESHOLD, Y_DIFF_THRESHOLD)

    translated_annotations = translate_text(merged_data_final)
    # overlay neatly
    cleaned, translated_image, meta = process_inpaint_pipeline(
        pil_img.copy(), translated_annotations
    )

    return cleaned, translated_image, meta


def process_inpaint_pipeline(img_pil, annotations):
    """
    This will be the function called by Streamlit.
    Must return (cleaned, final_image, meta)
    """
    # 1) Remove original text
    cleaned = inpaint_image_with_boxes(
        img_pil, annotations, expand_mask=8, prefer_lama=False)

    # 2) Overlay translated text
    final = overlay_translated_text(
        cleaned, annotations, font_path="arial.ttf")

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
