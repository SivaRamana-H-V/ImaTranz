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

    # Debug info
    st.sidebar.markdown("### 🔧 Debug Info")
    st.sidebar.write(
        f"GCP Env Var: {'GCP_SERVICE_ACCOUNT_JSON' in os.environ}")

    # Initialize processor with proper session handling
    if "processor" not in st.session_state:
        st.session_state.processor = ImageProcessor()

        # Initialize GCP services
        with st.spinner("🔐 Initializing GCP services..."):
            success = st.session_state.processor.initialize_services()

        if success:
            st.session_state.gcp_initialized = True
            st.sidebar.success("✅ GCP initialized successfully!")
        else:
            st.session_state.gcp_initialized = False
            st.sidebar.error("❌ GCP initialization failed")

    # Check if GCP is actually working
    if (hasattr(st.session_state.processor, 'gcp_services') and
        st.session_state.processor.gcp_services and
            st.session_state.processor.gcp_services.vision_client):

        st.success("✅ GCP Services Ready! You can now translate Amazon images.")
        st.session_state.gcp_initialized = True
    else:
        st.session_state.gcp_initialized = False

    # Only show main app if GCP is working
    if not st.session_state.gcp_initialized:
        st.error("""
        🔐 **GCP Services Not Available**
        
        The logs show GCP is working, but there might be a session issue.
        
        **Please try:**
        1. **Hard refresh** the page (Ctrl+F5)
        2. **Clear browser cache**
        3. **Try incognito/private window**
        
        If still not working, the GCP credentials might need Vision/Translate API access.
        """)
        return

    # Initialize results
    if "results" not in st.session_state:
        st.session_state.results = []

    # Test GCP connection
    if st.sidebar.button("Test GCP Connection"):
        try:
            test_text = ["Hello World"]
            translated = st.session_state.processor.gcp_services.translate_texts(
                test_text, target="es")
            st.sidebar.success(
                f"✅ GCP Test: '{test_text[0]}' → '{translated[0]}'")
        except Exception as e:
            st.sidebar.error(f"❌ GCP Test Failed: {e}")

    # UI Components - only show if GCP is working
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

            try:
                results = st.session_state.processor.process_images_concurrently(
                    image_urls, max_images, timeout_per_image
                )
                st.session_state.results = results
                st.success("✅ Processing completed successfully!")
            except Exception as e:
                st.error(f"Processing failed: {e}")

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
