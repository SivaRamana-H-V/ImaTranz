import streamlit as st
import tempfile
import zipfile
from io import BytesIO
import os

from config.settings import setup_page_config, MAX_IMAGES
from services.image_processor import ImageProcessor
from services.amazon_scraper import AmazonScraper
from utils.image_utils import upscale_image


def main():
    setup_page_config()
    st.title("Amazon Product Image Translator — GCP Vision + Translate")

    # Initialize session state
    if "results" not in st.session_state:
        st.session_state.results = []
    if "processor" not in st.session_state:
        st.session_state.processor = ImageProcessor()

        # Initialize GCP services with detailed feedback
        with st.spinner("🔐 Initializing GCP services..."):
            success = st.session_state.processor.initialize_services()

        if not success:
            st.error("""
            🔐 **GCP Services Not Available**
            
            **To fix this:**
            
            1. **Add GCP credentials to Render:**
               - Go to your Render dashboard → ImaTranz service → Environment
               - Add: `GCP_SERVICE_ACCOUNT_JSON` = [your-json-here]
            
            2. **Get GCP credentials:**
               - Visit [Google Cloud Console](https://console.cloud.google.com)
               - Create service account with **Vision API** and **Translate API**
               - Download JSON key file
               - Copy entire JSON content into Render
            
            3. **Required permissions:**
               - Cloud Translation API User
               - Vision AI User
            
            The app will auto-redeploy once you add credentials!
            """)
            return

    # UI Components
    st.markdown(
        "Paste an **Amazon product URL** to translate images to English.")

    col1, col2 = st.columns([3, 1])
    with col2:
        max_images = st.number_input(
            "Max images to process", 1, 100, MAX_IMAGES)
        timeout_per_image = st.slider("Per-image timeout (sec)", 5, 40, 15)

    url = st.text_input("Amazon product URL",
                        placeholder="https://amzn.asia/d/...")
    run = st.button("Fetch & Translate")

    # Processing logic
    if run and url:
        scraper = AmazonScraper()
        with st.spinner("Fetching product images..."):
            try:
                image_urls = scraper.fetch_image_urls(
                    url, max_images=max_images)
            except Exception as e:
                st.error(f"Failed to fetch product page: {e}")
                image_urls = []

        if not image_urls:
            st.warning("No product images found.")
        else:
            st.success(
                f"Found {len(image_urls)} image(s). Starting processing...")

            results = st.session_state.processor.process_images_concurrently(
                image_urls, max_images, timeout_per_image
            )

            st.session_state.results = results

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        for idx, (url_img, orig, cleaned, final_img, meta) in enumerate(results):
            st.markdown(f"### Image {idx+1}")
            c1, c2 = st.columns(2)
            with c1:
                st.image(upscale_image(orig), caption="Original")
            with c2:
                st.image(final_img, caption="Translated (English)")

            if meta.get("detected") and meta.get("detected") > 0:
                c11, c12 = st.columns([1, 2])
                with c11:
                    st.markdown("**Original Text:**")
                    st.write(' '.join(meta.get("orig_blocks", [])))
                with c12:
                    st.write(f"Detected blocks: {meta.get('detected')}")
                    st.markdown("**Translated Text:**")
                    st.write(', '.join(meta.get("trans_blocks", [])))
            st.markdown("---")

        # ZIP download
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


if __name__ == "__main__":
    main()
