import concurrent.futures
import logging
from typing import List, Dict, Tuple
from PIL import Image
from services.amazon_scraper import AmazonScraper
from services.gcp_services import GCPServices
from utils.image_utils import create_mask_for_boxes, inpaint_with_opencv, upscale_image
from utils.text_utils import TextOverlay
from config.settings import FONT_PATH, BOX_EXPANSION_INPAINT

logging.basicConfig(level=logging.INFO)

class ImageProcessor:
    def __init__(self):
        self.gcp_services = GCPServices()
        self.text_overlay = TextOverlay(FONT_PATH)

    def initialize_services(self):
        """Initialize GCP services"""
        self.gcp_services.initialize()

    def process_image_gcp(self, pil_img: Image.Image) -> Tuple[Image.Image, Image.Image, Dict]:
        """Full image processing pipeline"""
        try:
            # OCR detection
            blocks = self.gcp_services.vision_ocr_paragraphs(pil_img)
        except Exception as e:
            return pil_img, pil_img, {"error": str(e)}

        if not blocks:
            return pil_img, pil_img, {"detected": 0}

        # Translation
        translated_blocks = self.translate_text(blocks)
        # Inpainting and overlay
        cleaned, final_img, meta = self.process_inpaint_pipeline(pil_img.copy(), translated_blocks)
        meta["detected"] = len(translated_blocks)
        return cleaned, final_img, meta

    def translate_text(self, ocr_blocks: List[Dict]) -> List[Dict]:
        """Translate OCR blocks"""
        if not ocr_blocks:
            return []

        texts = [b["text"] for b in ocr_blocks]
        translations = self.gcp_services.translate_texts(texts, target="en")
        for b, tr in zip(ocr_blocks, translations):
            b["new_text"] = tr
        return ocr_blocks

    def process_inpaint_pipeline(self, img_pil: Image.Image, annotations: List[Dict]) -> Tuple[Image.Image, Image.Image, Dict]:
        """Complete inpainting and text overlay pipeline"""
        # Remove original text
        mask = create_mask_for_boxes(img_pil.size, annotations, expand=BOX_EXPANSION_INPAINT)
        cleaned = inpaint_with_opencv(img_pil, mask, method='telea')

        # Overlay translated text
        final = self.text_overlay.overlay_translated_text(cleaned, annotations)

        # Metadata
        meta = {
            "detected": len(annotations),
            "orig_blocks": [a["text"] for a in annotations],
            "trans_blocks": [a.get("new_text", a["text"]) for a in annotations]
        }

        return cleaned, final, meta

    def process_images_concurrently(self, image_urls: List[str], max_images: int, timeout_per_image: int) -> List:
        """Process multiple images concurrently"""
        scraper = AmazonScraper()
        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(image_urls))) as ex:
            # Download images
            download_futs = {ex.submit(scraper.download_image, u): u for u in image_urls}
            downloaded = {}
            for fut in concurrent.futures.as_completed(download_futs, timeout=timeout_per_image * len(image_urls)):
                u = download_futs[fut]
                try:
                    downloaded[u] = fut.result()
                except Exception as e:
                    logging.warning(f"Failed download {u}: {e}")

            # Process images
            proc_futs = {ex.submit(self.process_image_gcp, upscale_image(img, scale_factor=2)): u 
                        for u, img in downloaded.items()}
            for fut in concurrent.futures.as_completed(proc_futs, timeout=timeout_per_image * len(proc_futs)):
                u = proc_futs[fut]
                try:
                    cleaned, final_img, meta = fut.result()
                    results.append((u, downloaded[u], cleaned, final_img, meta))
                except Exception as e:
                    logging.warning(f"Processing failed for {u}: {e}")

        return results