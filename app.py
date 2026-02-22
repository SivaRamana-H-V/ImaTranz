import streamlit as st
import tempfile
import zipfile
from io import BytesIO
import os
from PIL import Image

from config.settings import setup_page_config, MAX_IMAGES
from services.image_processor import ImageProcessor
from services.amazon_scraper import AmazonScraper
from utils.image_utils import upscale_image


def image_to_base64(pil_img: Image.Image) -> str:
    """Convert PIL image to base64 data URL"""
    import io
    import base64
    buf = io.BytesIO()
    # Save as PNG to preserve quality, or JPEG for smaller size
    pil_img.save(buf, format="PNG")
    img_str = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def main():
    setup_page_config()
    st.title("Amazon Product Image Translator")

    # Initialize processor with proper session handling
    if "processor" not in st.session_state:
        st.session_state.processor = ImageProcessor()
        st.session_state.gcp_initialized = False

    # Initialize GCP services if not already done
    if not st.session_state.gcp_initialized:
        with st.spinner("🔐 Initializing GCP services..."):
            success = st.session_state.processor.initialize_services()

        if success:
            st.session_state.gcp_initialized = True
            # st.success("✅ GCP initialized successfully!") # Removed as per request
        else:
            # Show detailed error
            error_msg = st.session_state.processor.initialization_error or "Unknown initialization error"
            st.error(f"❌ GCP initialization failed:\n{error_msg}")

    # Check if GCP is actually working
    gcp_working = False
    if st.session_state.gcp_initialized:
        if (hasattr(st.session_state.processor, 'gcp_services') and
            st.session_state.processor.gcp_services and
                st.session_state.processor.gcp_services.vision_client):
            gcp_working = True
        else:
            # Reset if clients are missing
            st.session_state.gcp_initialized = False
            st.rerun()

    # Only show main app if GCP is working
    if not gcp_working:
        st.info("⏳ Waiting for GCP services to initialize...")
        return

    # Initialize results
    if "results" not in st.session_state:
        st.session_state.results = []

    # UI Components - only show if GCP is working
    st.markdown(
        "Paste an **Amazon product URL** to translate images to English.")

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

            try:
                results = st.session_state.processor.process_images_concurrently(
                    image_urls, max_images, 15
                )
                st.session_state.results = results
                st.success("✅ Processing completed successfully!")
            except Exception as e:
                st.error(f"Processing failed: {e}")

    # Display results
    if st.session_state.results:
        results = st.session_state.results
        for idx, (url_img, orig_path, cleaned_path, final_img_path, meta) in enumerate(results):
            
            # Load images from disk (Critical for Cloud Run persistence)
            try:
                orig = Image.open(orig_path)
                final_img = Image.open(final_img_path)
            except Exception as e:
                st.error(f"Error loading image files: {e}")
                continue

            st.markdown(f"### Image {idx+1}")
            c1, c2 = st.columns(2)
            with c1:
                st.image(image_to_base64(upscale_image(orig)), caption="Original")
            with c2:
                st.image(image_to_base64(final_img), caption="Translated (English)")

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
                for i, (_, _, _, final_img_path, _) in enumerate(results):
                    # Write directly from disk to ZIP
                    try:
                        zf.write(final_img_path, arcname=f"translated_{i+1}.png")
                    except Exception as e:
                        st.error(f"Error zipping file {i+1}: {e}")

            with open(tmp.name, "rb") as fh:
                st.download_button(
                    "Download ZIP",
                    data=fh.read(),
                    file_name="translated_images.zip",
                    mime="application/zip"
                )


if __name__ == "__main__":
    main()
