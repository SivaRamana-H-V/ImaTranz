import re
import requests
import json
from io import BytesIO
from typing import List, Optional
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from PIL import Image
from config.settings import HEADERS

class AmazonScraper:
    def __init__(self):
        self.headers = HEADERS

    def normalize_amazon_image_url(self, url: str) -> str:
        """Normalize Amazon image URL"""
        if not url:
            return url
        url = re.sub(r"\?.*$", "", url)
        url = re.sub(r"\._[A-Za-z0-9,._-]+_\.", ".", url)
        m = re.search(r"(https?://[^/]+/images/I/[^.]+)\.(?:jpg|png|jpeg)$", url)
        if m:
            base = m.group(1)
            return base + ".jpg"
        return url

    def extract_amazon_image_urls_from_html(self, html, base_url, max_images=None):
        """Extract image URLs from Amazon product page"""
        soup = BeautifulSoup(html, "html.parser")
        found = []

        def add(u):
            if not u:
                return
            if u.startswith("//"):
                u = "https:" + u
            elif u.startswith("/"):
                u = urljoin(base_url, u)
            u = self.normalize_amazon_image_url(u)
            if "m.media-amazon.com/images/I/" not in u and "images-fe.ssl-images-amazon.com/images/I/" not in u:
                return
            if u not in found:
                found.append(u)

        # Extract from colorImages JSON
        for m in re.finditer(r'"colorImages"\s*:\s*\{.*?"initial"\s*:\s*(\[[^\]]+\])', html, re.DOTALL):
            try:
                arr = json.loads(m.group(1))
                for e in arr:
                    u = e.get("hiRes") or e.get("large") or e.get("mainUrl") or e.get("thumb")
                    add(u)
            except Exception:
                pass

        # Extract from script tags
        for script in soup.find_all("script"):
            if not script.string:
                continue
            for match in re.findall(r'"large"\s*:\s*"([^"]+)"', script.string):
                add(match)
            for match in re.findall(r'"hiRes"\s*:\s*"([^"]+)"', script.string):
                add(match)

        if max_images:
            return found[:max_images]
        return found

    def fetch_image_urls(self, product_url: str, max_images: Optional[int] = None, timeout: int = 15) -> List[str]:
        """Fetch image URLs from Amazon product page"""
        r = requests.get(product_url, headers=self.headers, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return self.extract_amazon_image_urls_from_html(r.text, r.url, max_images=max_images)

    def download_image(self, url: str, timeout: int = 12) -> Image.Image:
        """Download image from URL"""
        r = requests.get(url, headers=self.headers, timeout=timeout)
        r.raise_for_status()
        return Image.open(BytesIO(r.content)).convert("RGB")