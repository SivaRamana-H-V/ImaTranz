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
from inpaint_and_overlay import inpaint_image_with_boxes, overlay_translated_text
import html

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


def get_box_coords(item):
    """Helper to extract (x_min, y_min, x_max, y_max) from a data item."""
    vertices = item["bounding_box"]
    x_min = min(v["x"] for v in vertices)
    y_min = min(v["y"] for v in vertices)
    x_max = max(v["x"] for v in vertices)
    y_max = max(v["y"] for v in vertices)
    return x_min, y_min, x_max, y_max


def merge_vision_boxes(text_items, min_height=15, y_merge_thresh=25):
    """
    Merge small Vision word-boxes into readable horizontal line blocks.
    This is CRITICAL for good translation placement.

    Steps:
        1. Remove boxes that are too thin (Vision garbage)
        2. Sort items by Y position
        3. Group into horizontal lines
        4. Merge words into a single bounding box per line
    """
    cleaned = []

    # 1) Remove tiny broken boxes
    for t in text_items:
        poly = t["bounding_box"]
        xs = [p["x"] for p in poly]
        ys = [p["y"] for p in poly]
        h = max(ys) - min(ys)
        if h >= min_height:
            cleaned.append(t)

    if not cleaned:
        return []

    # 2) Sort by Y first
    cleaned.sort(key=lambda a: min(p["y"] for p in a["bounding_box"]))

    merged = []
    current_line = [cleaned[0]]

    def in_same_line(a, b):
        ay1 = min(p["y"] for p in a["bounding_box"])
        by1 = min(p["y"] for p in b["bounding_box"])
        return abs(ay1 - by1) < y_merge_thresh

    for i in range(1, len(cleaned)):
        if in_same_line(current_line[-1], cleaned[i]):
            current_line.append(cleaned[i])
        else:
            merged.append(current_line)
            current_line = [cleaned[i]]

    merged.append(current_line)

    # 3) Build merged blocks
    final = []
    for line in merged:
        line = sorted(line, key=lambda a: min(
            p["x"] for p in a["bounding_box"]))
        text_combined = " ".join(item["text"] for item in line)

        xs = []
        ys = []
        for item in line:
            for p in item["bounding_box"]:
                xs.append(p["x"])
                ys.append(p["y"])

        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)

        final.append({
            "text": text_combined.strip(),
            "bounding_box": [
                {"x": x1, "y": y1},
                {"x": x2, "y": y1},
                {"x": x2, "y": y2},
                {"x": x1, "y": y2},
            ]
        })

    return final

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

    # 2) Merge nearby boxes into lines/blocks (scale-aware)
    # merged = merge_vision_boxes(blocks)
    # 3) Translate
    translated_blocks = translate_text(blocks)
    # 4) Inpaint & overlay
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
