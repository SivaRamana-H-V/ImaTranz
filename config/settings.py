import os
import streamlit as st

# Constants
MAX_IMAGES = 3
BOX_EXPANSION_INPAINT = 12
INITIAL_FONT_SIZE = 50
FONT_PATH = "assets/fonts/POPPINS-MEDIUM.TTF"
DISTANCE_THRESHOLD = 400
Y_DIFF_THRESHOLD = 2
MIN_FONT_SIZE = 12

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

class AppConfig:
    PAGE_TITLE = "Amazon Image Translator (GCP)"
    PAGE_ICON = "🚀"
    LAYOUT = "wide"

def setup_page_config():
    st.set_page_config(
        page_title=AppConfig.PAGE_TITLE,
        layout=AppConfig.LAYOUT
    )